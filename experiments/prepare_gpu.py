"""Fixed-surface GPU experiment: does admitted prepare reduce foreground latency?"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
import uuid
from pathlib import Path

import httpx

from .gpu_run import ENGINE, engine, write_json
from .hybrid_gpu import INGRESS, ingress

MODEL = "qwen35-9b-nvfp4"
WAIT = "fixed_fixture_wait"


def target_text() -> str:
    return ("Reference facts: apples are red, pears are green, and the target checksum is 42.\n" * 320
            + "Using those facts, answer with just the checksum as an integer.")


def chat(text: str) -> dict:
    return {"model": MODEL, "messages": [{"role": "user", "content": text}],
            "chat_template_kwargs": {"enable_thinking": False}, "max_completion_tokens": 8,
            "temperature": 0, "stream": False}


async def reset_cache(client):
    response = await client.post(ENGINE + "/flush_cache", params={"timeout": 5})
    response.raise_for_status()


async def calibrate(root: Path, client: httpx.AsyncClient, name: str) -> Path:
    res = await client.post(ENGINE + "/v1/tokenize", json=chat(target_text()))
    res.raise_for_status()
    ids = res.json()["tokens"]
    if len(ids) < 2:
        raise RuntimeError("fixture tokenizer did not return a reusable prefix")
    bucket = next(n for n in (1, 1024, 4096, 8192, 16384, 32768) if n >= len(ids))
    samples, waits, evictions = [], [], []
    with (root / "logs" / f"{name}-calibration.jsonl").open("x", buffering=1) as f:
        for i in range(25):
            await reset_cache(client)
            salt = uuid.uuid4().hex
            started = time.monotonic_ns()
            res = await client.post(ENGINE + "/generate", json={"input_ids": ids[:-1], "cache_salt": salt,
                "sampling_params": {"max_new_tokens": 0, "temperature": 0}})
            elapsed = (time.monotonic_ns() - started) / 1e6
            f.write(json.dumps({"phase": "warmup" if i < 5 else "prefill", "salt": salt,
                "wall_ms": elapsed, "http": res.status_code, "response": res.json()})+"\n")
            res.raise_for_status()
            meta = res.json()["meta_info"]
            assert meta["prompt_tokens"] == len(ids) - 1 and meta["cached_tokens"] == 0
            if i >= 5:
                samples.append(elapsed)
                start = time.monotonic_ns()
                await asyncio.sleep(0.8)
                ms = (time.monotonic_ns()-start)/1e6
                waits.append({"duration_ms": ms, "completed": True})
                probe = await client.post(ENGINE + "/generate", json={"input_ids": ids, "cache_salt": salt,
                    "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
                probe.raise_for_status()
                cached = probe.json()["meta_info"]["cached_tokens"]
                evictions.append({"evicted": cached != len(ids) - 1})
                f.write(json.dumps({"phase": "fixture-wait", "duration_ms": ms, "cached_tokens": cached,
                                     "expected_prefix": len(ids) - 1})+"\n")
    # This calibration is for a low-pressure isolated preparation-only fixture.
    # There is no migration action in this fixture; exclude it rather than
    # representing zero-valued transfer estimates as measured bandwidth.
    calibration = {"buckets": [{"token_count": bucket, "batch_size": 1, "prefill_ms": statistics.median(samples),
                  "d2h_ms": 0, "h2d_ms": 0, "recurrent_ms": 0, "samples": len(samples)}],
                   "waits": {WAIT: waits}, "evictions": {"low": evictions},
                   "scope": "fixed low-pressure prepare fixture only; no host moves", "target_tokens": len(ids)}
    path = root / f"{name}-calibration.json"
    write_json(path, calibration)
    return path


class Branch:
    def __init__(self, http, run):
        self.http, self.run = http, run
        self.id = uuid.uuid4().hex
        self.version = None
        self.seq = 0

    async def event(self, kind, payload):
        eid = uuid.uuid4().hex
        evt = {"run_id": self.run, "session_id": self.id, "branch_id": self.id, "seq": self.seq + 1,
               "event_id": eid, "expected_version": self.version, "kind": kind, "payload": payload}
        res = await self.http.post("/agentic/v1/events", json=evt)
        res.raise_for_status()
        ack = res.json()
        assert ack["event_id"] == eid
        self.seq, self.version = evt["seq"], ack["version_id"]
        return ack

    async def lookup(self, event_id):
        res = await self.http.get(f"/agentic/v1/events/{self.run}/{event_id}")
        res.raise_for_status()
        return res.json()


def action(kind: str, **kw) -> dict:
    id = uuid.uuid4().hex
    return {"action_kind": kind, "action_id": id, "turn_id": id, "call_id": id, "rid": id, "ordinal": 0, **kw}


async def case(http, run, *, prepare: bool, wait_ms: int, log):
    b = Branch(http, run)
    source, target = [uuid.uuid4().hex], [uuid.uuid4().hex]
    await b.event("append", {"isolation_domain": b.id, "message_ids": source})
    body = chat(target_text())
    ack, planned = None, None
    start_wait = time.monotonic_ns()
    if prepare:
        planned = action("prepare", chat=body, boundary_message_ids=source, target_message_ids=target, wait_type=WAIT)
        ack = await b.event("ready", planned)
    await asyncio.sleep(max(0.0, wait_ms/1000 - (time.monotonic_ns()-start_wait)/1e9))
    status = (await b.lookup(ack["event_id"]))["status"] if ack else None
    ready = status == "completed"
    boundary = time.monotonic_ns()
    await b.event("replace", {"message_ids": target, **({"candidate_action_id": planned["action_id"]} if ready else {})})
    model = action("model")
    permit = (await b.event("ready", model))["permit"]
    assert permit["version_id"] == b.version and permit["expires_mono_ns"] > time.monotonic_ns(), permit
    req_begin = time.monotonic_ns()
    res = await http.post("/v1/chat/completions", json=body, headers={"X-Agentic-Session": b.id,
        "X-Agentic-Branch": b.id, "X-Agentic-Version": b.version, "X-Agentic-Request": model["rid"]})
    wall_ns = time.monotonic_ns()
    response = res.json()
    row = {"run_id": run, "rid": model["rid"], "candidate_action_id": planned["action_id"] if planned else None,
           "boundary_ns": boundary, "prepared": ready, "prepare_ack": ack, "prepare_status": status,
           "fixture_wait_ms": (boundary-start_wait)/1e6, "wait_ms": wait_ms,
           "http": res.status_code, "result": response,
           "foreground_ms": (wall_ns-boundary)/1e6, "chat_ms": (wall_ns-req_begin)/1e6}
    log.write(json.dumps(row, ensure_ascii=False)+"\n")
    res.raise_for_status()
    await b.event("commit", {"action_id": model["action_id"]})
    await b.event("end", {})
    return row


async def compare(root: Path, name: str, token: str, pairs: int, wait_ms: int) -> dict:
    config = {"Authorization": f"Bearer {token}"}
    rows = []
    async with httpx.AsyncClient(base_url=INGRESS, headers=config, trust_env=False, timeout=180) as http:
        with (root / "logs" / f"{name}-pairs.jsonl").open("x", buffering=1) as log:
            # Warm JIT/cache software in unrelated isolation domains. Do not
            # count its timing or import its prefills into the measured pairs.
            await case(http, "warmup-"+uuid.uuid4().hex, prepare=False, wait_ms=0, log=log)
            for i in range(pairs):
                await reset_cache(http)
                runs = {}
                for arm in (False, True) if i % 2 == 0 else (True, False):
                    runs[arm] = await case(http, "fixture-"+uuid.uuid4().hex, prepare=arm, wait_ms=wait_ms, log=log)
                base, early = runs[False], runs[True]
                a, b = (row["result"]["choices"][0]["message"]["content"] for row in (base, early))
                row = {"pair": i, "cold_rid": base["rid"], "early_rid": early["rid"], "same_response": a == b,
                       "expected_response": a == b == "42", "prepare_status": early["prepare_status"],
                       "prepared": early["prepared"], "cold_ms": base["foreground_ms"], "early_ms": early["foreground_ms"],
                       "cold_total_ms": base["fixture_wait_ms"] + base["foreground_ms"],
                       "early_total_ms": early["fixture_wait_ms"] + early["foreground_ms"]}
                rows.append(row)
                print(name, row, flush=True)
    def latencies(key):
        values = sorted(r[key] for r in rows)
        return statistics.median(values), values[math.ceil(.95 * len(values)) - 1]

    receipts = {}
    for line in (root / "logs" / f"{name}-ingress.log").read_text().splitlines():
        if "agentic_layer:engine receipt " in line:
            receipt = json.loads(line.split("agentic_layer:engine receipt ", 1)[1])
            receipts[receipt["rid"]] = receipt
    for row in rows:
        for arm in ("cold", "early"):
            row[f"{arm}_hit"] = receipts[row[f"{arm}_rid"]]["matched_tokens"]
        if (not row["expected_response"] or row["cold_hit"] != 0 or
                row["prepared"] != (row["early_hit"] > 0)):
            raise RuntimeError(f"prepare correctness failed: {row}")
    return {"kind": "fixed-surface GPU preparation fixture (not DSH E3 or trace)", "wait_ms": wait_ms,
            "pairs": rows, "prepared_pairs": sum(r["prepared"] for r in rows),
            "same_responses": all(r["same_response"] for r in rows),
            "latencies_ms_p50_p95": {key: latencies(key) for key in
                                     ("cold_ms", "early_ms", "cold_total_ms", "early_total_ms")}}


async def native(root, name, repeats, wait_ms):
    rows = []
    async with httpx.AsyncClient(timeout=180, trust_env=False) as http:
        with (root / "logs" / f"{name}-pairs.jsonl").open("x", buffering=1) as log:
            for i in range(repeats + 1):
                await reset_cache(http)
                arrived = time.monotonic_ns()
                if i:
                    await asyncio.sleep(wait_ms / 1000)
                boundary = time.monotonic_ns()
                response = await http.post(ENGINE + "/v1/chat/completions", json={
                    **chat(target_text()), "cache_salt": uuid.uuid4().hex})
                finished = time.monotonic_ns()
                response.raise_for_status()
                data = response.json()
                assert data["choices"][0]["message"]["content"] == "42", data
                row = {"warmup": i == 0, "foreground_ms": (finished-boundary)/1e6,
                       "total_ms": (finished-arrived)/1e6, "response": data}
                log.write(json.dumps(row)+"\n")
                if i:
                    rows.append(row)
    latencies = {}
    for key in ("foreground_ms", "total_ms"):
        values = sorted(row[key] for row in rows)
        latencies[key] = [statistics.median(values), values[math.ceil(.95*repeats)-1]]
    return {"kind": "fixed-surface GPU native baseline", "policy": "native", "wait_ms": wait_ms,
            "runs": rows, "latencies_ms_p50_p95": latencies}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--wait-ms", type=int, default=800)
    p.add_argument("--policy", choices=("native", "strong", "joint", "joint-no-predict"), default="joint")
    p.add_argument("--calibration", type=Path, help="reuse the same frozen calibration across policy arms")
    args = p.parse_args()
    assert args.pairs > 0 and 0 <= args.wait_ms <= 2000
    root = args.run_dir.resolve()
    if args.policy == "native":
        with engine(root, "native-radix", args.name):
            result = asyncio.run(native(root, args.name, args.pairs, args.wait_ms))
        write_json(root / f"{args.name}.json", result)
        print(result["latencies_ms_p50_p95"], flush=True)
        return
    with engine(root, "extended", args.name):
        async def evidence():
            async with httpx.AsyncClient(timeout=180, trust_env=False) as http:
                return await calibrate(root, http, args.name)
        path = args.calibration or asyncio.run(evidence())
        with ingress(root, args.name, args.policy, path) as token:
            result = asyncio.run(compare(root, args.name, token, args.pairs, args.wait_ms))
    write_json(root / f"{args.name}.json", {**result, "policy": args.policy,
        "calibration_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


if __name__ == "__main__":
    main()
