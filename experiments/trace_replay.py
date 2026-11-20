"""Replay prepared FreeInference rows; only the model runs, never anonymized tools."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .gpu_run import ENGINE, write_json
from .hybrid_gpu import INGRESS

_METRICS = frozenset({
    "num_running_reqs", "num_queue_reqs", "full_token_usage", "mamba_usage",
    "kv_available_tokens", "kv_evictable_tokens", "hicache_host_used_tokens",
    "hicache_host_total_tokens", "hicache_host_mamba_used_states", "hicache_host_mamba_total_states",
    "hicache_backup_bytes_total", "load_back_bytes_total", "evicted_tokens_total",
    "load_back_duration_seconds_sum", "hicache_backup_duration_seconds_sum",
})


async def record_resources(http, record):
    try:
        response = await http.get(ENGINE + "/metrics", timeout=2)
        response.raise_for_status()
        metrics = {}
        for line in response.text.splitlines():
            if not line.startswith("sglang:"):
                continue
            key = line.split("{", 1)[0].split(" ", 1)[0].removeprefix("sglang:")
            if key in _METRICS:
                value = float(line.rsplit(maxsplit=1)[1])
                if math.isfinite(value):
                    metrics[key] = max(metrics.get(key, value), value)
        record.add("resource-sample", metrics=metrics)
    except httpx.HTTPError as exc:
        record.add("resource-error", error=str(exc))


async def sample_resources(http, record):
    while True:
        await asyncio.sleep(1)
        await record_resources(http, record)



def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


class Recorder:
    def __init__(self, log):
        self.log = log

    def add(self, kind, **value):
        self.log.write(json.dumps({"kind": kind, "mono_ns": time.monotonic_ns(), **value},
                                  ensure_ascii=False, separators=(",", ":")) + "\n")


@dataclass
class Branch:
    session: str
    branch: str
    isolation: str
    salt: str
    seq: int = 0
    version: str | None = None
    ids: list[int] = field(default_factory=list)
    message_ids: list[str] = field(default_factory=list)


class Replay:
    def __init__(self, http, recorder, run, controlled):
        self.http, self.record, self.run, self.controlled = http, recorder, run, controlled
        self.branches = {}
        self.committed = set()
        self.ready: asyncio.Queue = asyncio.Queue()
        self.transfers = set()

    async def event(self, b: Branch, kind, payload):
        if not self.controlled:
            return {}
        evt = {"run_id": self.run, "session_id": b.session, "branch_id": b.branch,
               "expected_version": b.version, "seq": b.seq + 1, "event_id": uuid.uuid4().hex,
               "kind": kind, "payload": payload}
        response = await self.http.post(INGRESS + "/agentic/v1/events", json=evt)
        body = response.json()
        if not response.is_success:
            raise RuntimeError((kind, body))
        assert body["event_id"] == evt["event_id"]
        b.seq, b.version = evt["seq"], body["version_id"]
        self.record.add("control", request=evt, response=body)
        return body

    async def prompt(self, b: Branch, request):
        ids = request["input_ids"]
        if ids == b.ids:
            return
        append = ids[:len(b.ids)] == b.ids
        kind = "append" if append else "remove" if b.ids[:len(ids)] == ids else "replace"
        segment = digest(request["identity"])
        b.message_ids = [*b.message_ids, segment] if append else [segment]
        await self.event(b, kind, {"isolation_domain": b.isolation, "message_ids": b.message_ids})
        if not append:
            b.salt = digest([b.salt, segment, kind])
        b.ids = ids

    async def declare(self, b, kind, rid, turn, call, *, wait_type=None, oracle_wait=None):
        aid = uuid.uuid4().hex
        args = {"action_kind": kind, "action_id": aid, "rid": rid, "turn_id": turn,
                "call_id": call, "ordinal": 0}
        if wait_type is not None:
            args["wait_type"] = wait_type
        if oracle_wait is not None:
            args["oracle_wait_ms"] = oracle_wait
        start = time.monotonic_ns()
        reply = await self.event(b, "ready", args)
        if self.controlled:
            permit = reply.get("permit")
            if not permit or permit["action_id"] != aid or permit["version_id"] != b.version or permit["expires_mono_ns"] <= time.monotonic_ns():
                raise RuntimeError(("ready denied", args, reply))
            self.record.add("ready-ack", rid=rid, action_kind=kind,
                            duration_ms=(time.monotonic_ns()-start)/1e6)
        return aid

    async def batches(self):
        try:
            while True:
                group = [await self.ready.get()]
                await asyncio.sleep(0)
                while len(group) < 8:
                    try:
                        group.append(self.ready.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                job = asyncio.create_task(self.batch(group))
                self.transfers.add(job)
                job.add_done_callback(self.transfers.discard)
        finally:
            active = tuple(self.transfers)
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)

    async def batch(self, group):
        requests = [entry[1] for entry in group]
        rids = [item["rid"] for item in requests]
        begun = time.monotonic_ns()
        self.record.add("batch-start", rids=rids, batch_size=len(group))
        try:
            if self.controlled:
                reply = await self.http.post(INGRESS + "/agentic/v1/batch", json={"items": requests})
                reply.raise_for_status()
                results = {r["rid"]: r for r in reply.json()["items"]}
            else:
                payload = {k: [item[k] for item in requests] for k in ("rid", "input_ids", "sampling_params")}
                payload.update(cache_salt=[entry[0].salt for entry in group], stream=True)
                results = {}
                async with self.http.stream("POST", ENGINE + "/generate", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        body = line[5:].strip()
                        if body == "[DONE]":
                            break
                        item = json.loads(body)
                        meta = item.get("meta_info", {})
                        if reason := meta.get("finish_reason"):
                            results[meta["id"]] = {"status": "error" if reason["type"] == "abort" else "completed",
                                                   "result": item}
            self.record.add("batch-finish", rids=rids, duration_ms=(time.monotonic_ns()-begun)/1e6)
            for rid, entry in zip(rids, group, strict=True):
                if not entry[2].done():
                    entry[2].set_result(results.get(rid, {"status": "error", "error": "missing output"}))
        except (ValueError, KeyError, httpx.HTTPError) as exc:
            for _, _, future in group:
                if not future.done():
                    future.set_exception(RuntimeError(str(exc)))

    async def requests(self, row, requests, *, oracle):
        for req in requests:
            key = digest(req["identity"])
            if req["predecessor"] and digest(req["predecessor"]) not in self.committed:
                raise RuntimeError(f"request {key} would overtake its confirmed predecessor")
            branch_key = digest(req["identity"][:3])
            if branch_key not in self.branches:
                domain = digest([row["week"], row["user_id"]])
                session = digest([self.run, row["week"], row["session_id"]])
                self.branches[branch_key] = Branch(session, branch_key, domain, digest([self.run, domain]))
            b = self.branches[branch_key]
            await self.prompt(b, req)
            turn = digest([row["week"], row["session_id"], row["turn_index"]])
            rid = uuid.uuid4().hex
            self.record.add("model-ready", identity=req["identity"], rid=rid)
            aid = await self.declare(b, "batch", rid, turn, key)
            future = asyncio.get_running_loop().create_future()
            await self.ready.put((b, {"session_id": b.session, "branch_id": b.branch, "version_id": b.version,
                "rid": rid, "input_ids": req["input_ids"], "sampling_params": {"max_new_tokens": req["max_new_tokens"],
                    "temperature": 0, "ignore_eos": True}}, future))
            result = await future
            if result["status"] == "completed":
                meta = result["result"]["meta_info"]
                if meta.get("prompt_tokens") != len(req["input_ids"]):
                    raise RuntimeError(f"projected prompt length changed for {rid}")
            self.record.add("model", identity=req["identity"], rid=rid, status=result["status"], response=result)
            if result["status"] != "completed" or len(result["result"]["output_ids"]) != req["max_new_tokens"]:
                raise RuntimeError(("generation did not complete", rid, result))
            await self.event(b, "commit", {"action_id": aid})
            self.committed.add(key)
            if req["tools"]:
                tool_id = await self.declare(b, "tool", uuid.uuid4().hex, turn, key, wait_type="trace_barrier",
                                             oracle_wait=req["barrier_ms"] if oracle else None)
                await self.event(b, "start", {"action_id": tool_id})
                start = time.monotonic_ns()
                for call in req["tools"]:
                    await self.requests(row, call["subagent"], oracle=oracle)
                    if self.controlled and call["subagent"]:
                        child_key = digest(call["subagent"][0]["identity"][:3])
                        await self.event(self.branches[child_key], "end", {})
                await asyncio.sleep(max(0, req["barrier_ms"]/1e3 - (time.monotonic_ns()-start)/1e9))
                self.record.add("tool-barrier", identity=req["identity"], min_ms=req["barrier_ms"],
                                duration_ms=(time.monotonic_ns()-start)/1e6)
                await self.event(b, "finish", {"action_id": tool_id, "status": "completed"})
                await self.event(b, "commit", {"action_id": tool_id})

    async def turns(self, rows, *, oracle):
        sessions = defaultdict(list)
        for row in sorted(rows, key=lambda r: (r["week"], r["session_id"], r["turn_order"])):
            sessions[row["week"], row["session_id"]].append(row)
        ordered = sorted(sessions.values(), key=lambda rs: (rs[0]["start_ms"], rs[0]["selection_hash"]))
        outputs = []
        dispatcher = asyncio.create_task(self.batches())
        try:
            for offset in range(0, len(ordered), 8):
                group = ordered[offset:offset+8]
                timeline = sorted((row for seq in group for row in seq), key=lambda r: (r["start_ms"], r["selection_hash"]))
                offsets = {}
                previous, elapsed = timeline[0]["start_ms"], 0.0
                for row in timeline:
                    elapsed += min(1000, max(0, row["start_ms"] - previous)) / 1000
                    offsets[row["selection_hash"]] = elapsed
                    previous = row["start_ms"]
                clock = asyncio.get_running_loop()
                epoch = clock.time()

                async def one_session(sequence):
                    prev = None
                    invalid_chain = False
                    for row in sequence:
                        expected = row["previous_selected_turn"]
                        assert expected == prev, (expected, prev)
                        prev = [row["week"], row["session_id"], row["turn_index"]]
                        arrival = epoch + offsets[row["selection_hash"]]
                        await asyncio.sleep(max(0.0, arrival - clock.time()))
                        outcome = {"identity": prev, "stratum": row["stratum"], "arrival": arrival}
                        try:
                            if invalid_chain:
                                raise RuntimeError("previous selected turn failed")
                            async with asyncio.timeout_at(arrival + 900):
                                root_key = digest(row["requests"][0]["identity"][:3])
                                if self.controlled and root_key in self.branches:
                                    await self.event(self.branches[root_key], "resume", {})
                                await self.requests(row, row["requests"], oracle=oracle)
                                if self.controlled:
                                    await self.event(self.branches[root_key], "commit", {
                                        "turn_id": digest(outcome["identity"]), "turn_end": True})
                            outcome.update(status="completed", wall_ms=(clock.time()-arrival)*1000)
                            if self.controlled:
                                # Selection ending does not prove that the actual
                                # conversation expired. Park it; only known child
                                # activations receive an expiry event in replay.
                                await self.event(self.branches[root_key], "suspend", {})
                        except (RuntimeError, httpx.HTTPError, TimeoutError, KeyError, ValueError) as exc:
                            invalid_chain = True
                            outcome.update(status="timeout" if isinstance(exc, TimeoutError) else "error", error=str(exc), wall_ms=None)
                        self.record.add("turn-end", **outcome)
                        outputs.append(outcome)

                await asyncio.gather(*(one_session(seq) for seq in group))
        finally:
            dispatcher.cancel()
            await asyncio.gather(dispatcher, return_exceptions=True)
        return outputs


def summary(rows):
    latencies = sorted(r["wall_ms"] if r["wall_ms"] is not None else math.inf for r in rows)
    return {"statuses": dict(Counter(r["status"] for r in rows)), **{
        f"p{int(p*100)}_ms": (value if math.isfinite(value) else None)
        for p in (0.5, 0.95, 0.99) if latencies and (value := latencies[math.ceil(p*len(latencies))-1]) >= 0}}


def _decode_budget(requests):
    return sum(r["max_new_tokens"] + sum(_decode_budget(t["subagent"]) for t in r["tools"]) for r in requests)


def _subset(rows, turns_per_stratum, max_turn_decode=None):
    if turns_per_stratum is None:
        return rows
    by_identity = {(r["week"], r["session_id"], r["turn_index"]): r for r in rows}
    kept, strata = set(), Counter()
    for row in sorted(rows, key=lambda r: r["selection_hash"]):
        stratum = row["stratum"]
        if max_turn_decode is not None and _decode_budget(row["requests"]) > max_turn_decode:
            continue
        if strata[stratum] >= turns_per_stratum:
            continue
        strata[stratum] += 1
        kept.add(row["selection_hash"])
        while row["previous_selected_turn"] is not None:
            row = by_identity[tuple(row["previous_selected_turn"])]
            kept.add(row["selection_hash"])
    if max_turn_decode is not None and set(strata) != {r["stratum"] for r in rows}:
        raise ValueError("turn decode budget drops an entire source stratum")
    return [r for r in rows if r["selection_hash"] in kept]

def _drift_barriers(rows, mode):
    def barriers(requests):
        for req in requests:
            if req["tools"]:
                yield req
                for call in req["tools"]:
                    yield from barriers(call["subagent"])

    if mode not in {"recorded", "early-quarter", "long-double"}:
        raise ValueError(f"unknown wait-drift mode: {mode}")
    altered = []
    if mode != "recorded":
        ordered = sorted((req for row in rows for req in barriers(row["requests"])),
                         key=lambda req: digest(req["identity"]))
        for req in ordered[3::4]:
            old = req["barrier_ms"]
            new = old / 4 if mode == "early-quarter" else min(2000, 2 * old)
            req["barrier_ms"] = new
            altered.append({"identity": req["identity"], "recorded_ms": old, "effective_ms": new})
    return altered



async def run(source: Path, out: Path, name: str, token: str | None, oracle: bool,
              *, turns_per_stratum: int | None = None, max_turn_decode: int | None = None,
              wait_drift: str = "recorded"):
    manifest = json.loads(source.with_suffix(".manifest.json").read_text())
    with source.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == manifest["selected_rows_sha256"]
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(rows) == manifest["selected_count"] and rows
    rows = _subset(rows, turns_per_stratum, max_turn_decode)
    altered_waits = _drift_barriers(rows, wait_drift)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with (out / "logs" / f"{name}-timeline.jsonl").open("x", buffering=1) as log:
        async with httpx.AsyncClient(headers=headers, timeout=900, trust_env=False) as http:
            replay = Replay(http, Recorder(log), "trace-"+uuid.uuid4().hex, bool(token))
            await record_resources(http, replay.record)
            collector = asyncio.create_task(sample_resources(http, replay.record))
            try:
                started = time.monotonic_ns()
                result = await replay.turns(rows, oracle=oracle)
                elapsed_ms = (time.monotonic_ns() - started) / 1e6
            finally:
                collector.cancel()
                await asyncio.gather(collector, return_exceptions=True)
                await record_resources(http, replay.record)
    report = {"name": name, "mode": "controlled" if token else "native", "sample_sha256": manifest["selected_rows_sha256"],
              "tested_turn_hashes": [row["selection_hash"] for row in rows],
              "turns_per_stratum_limit": turns_per_stratum, "max_turn_decode": max_turn_decode,
              "wait_drift": wait_drift, "altered_barriers": altered_waits,
              "replay_elapsed_ms": elapsed_ms,
              "tested_strata": dict(Counter(row["stratum"] for row in rows)),
              "summary": summary(result), "turns": result}
    write_json(out / f"{name}.json", report)
    print(report["summary"], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--turns-per-stratum", type=int)
    args = parser.parse_args()
    if args.oracle and not args.controlled:
        parser.error("oracle requires controlled replay")
    run_token = os.environ["AGENTIC_LOCAL_TOKEN"] if args.controlled else None
    if args.turns_per_stratum is not None and args.turns_per_stratum < 1:
        parser.error("--turns-per-stratum must be positive")
    asyncio.run(run(args.sample, args.run_dir, args.name, run_token, args.oracle,
                    turns_per_stratum=args.turns_per_stratum))


if __name__ == "__main__":
    main()
