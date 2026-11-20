"""Probe physical transfer and single-request cancellation on the pinned GPU engine."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

import httpx

from .gpu_run import ENGINE, engine, token_fixture, write_json
from .hybrid_gpu import Driver, ident


def salt() -> str:
    return hashlib.sha256(ident().encode()).hexdigest()


async def cancel_transfer(d: Driver) -> dict:
    length = 16384
    ids = token_fixture(length + 1, 42)
    version, namespace = ident(), salt()
    _, prepared, prefill_ms = await d.generate(ids[:-1], namespace, version, prepare=True)
    assert prepared["status"] == "ready" and prepared["handle"], prepared
    handle = prepared["handle"]
    released, = await d.rpc(d.command("release", version=version, handle=handle))
    assert released["status"] == "ok", released

    offload = d.command("offload", version=version, handle=handle)
    cancel = d.command("cancel", rid=offload["rid"], version=version, handle=handle,
                       payload={"action_id": offload["action_id"]})
    reclaim = d.command("reclaim", version=version, handle=handle)
    future = d.control.watch(offload["action_id"])
    try:
        # One scheduler drain processes all three operations in order, before
        # the next native transfer-ack drain can retire the in-flight record.
        issued, canceled, refused = await d.rpc(offload, cancel, reclaim)
        assert issued["status"] == "accepted", issued
        assert canceled["status"] == "accepted", canceled
        assert refused["status"] == "unavailable" and "in-flight" in refused["error"], refused
        complete = await d.wait(future)
    finally:
        d.control.unwatch(offload["action_id"])
    assert complete["status"] == "canceled", complete
    assert complete["transfer_bytes"] > 0 and complete["details"]["transfer_metrics_complete"], complete
    assert complete["details"]["token_ids"] == ids[:-1], complete
    assert complete["details"]["residency"] == "hbm" and not complete["details"]["demoted"], complete
    assert complete["matched_tokens"] == length, complete

    cached, observed, _ = await d.generate(ids, namespace, ident(), n=8, logprob=-1)
    fresh, baseline, _ = await d.generate(ids, salt(), ident(), n=8, logprob=-1)
    assert observed["matched_tokens"] >= length, observed
    assert baseline["matched_tokens"] == 0, baseline
    assert cached["output_ids"] == fresh["output_ids"] and len(cached["output_ids"]) == 8, (cached, fresh)
    cached_probs = cached["meta_info"]["output_token_logprobs"]
    fresh_probs = fresh["meta_info"]["output_token_logprobs"]
    assert cached_probs == fresh_probs and len(cached_probs) == 8, (cached_probs, fresh_probs)
    result = {"length": length, "prefill_ms": prefill_ms, "offload_action_id": offload["action_id"],
              "cancel_action_id": cancel["action_id"], "reclaim_action_id": reclaim["action_id"],
              "transfer_bytes": complete["transfer_bytes"], "transfer_us": complete["transfer_us"],
              "transfer_metrics_complete": complete["details"]["transfer_metrics_complete"],
              "residency": complete["details"]["residency"], "demoted": complete["details"]["demoted"],
              "matched_tokens": observed["matched_tokens"], "baseline_matched_tokens": baseline["matched_tokens"],
              "output_ids": cached["output_ids"], "output_token_logprobs": cached_probs}
    d.record("canceled-transfer-verified", result)
    return result


async def cancel_batch_member(d: Driver) -> dict:
    ids = token_fixture(1025, 43)
    # Use the same eight-token correctness horizon as the cold-reference
    # gate; long unconstrained greedy streams can diverge with batch shape.
    count = 8
    baseline, baseline_receipt, _ = await d.generate(ids, salt(), ident(), n=count)
    assert baseline_receipt["matched_tokens"] == 0, baseline_receipt
    assert len(baseline["output_ids"]) == count, baseline

    rids = [ident(), ident()]
    versions = [ident(), ident()]
    namespaces = [salt(), salt()]
    binds = [d.command("bind", rid=rid, version=version,
                       payload={"expected_ids": ids, "cache_salt": namespace, "kind": "batch",
                                "session_id": version, "branch_id": version,
                                "max_new_tokens": count, "hold": False})
             for rid, version, namespace in zip(rids, versions, namespaces, strict=True)]
    futures = [d.control.watch(bind["action_id"]) for bind in binds]
    try:
        replies = await d.rpc(*binds)
        assert all(reply["status"] == "ok" for reply in replies), replies
        params = {"max_new_tokens": count, "temperature": 0, "ignore_eos": True}
        body = {"input_ids": [ids, ids], "rid": rids, "cache_salt": namespaces,
                "sampling_params": [params, params], "stream": True}
        d.record("native-batch-request", body)
        active = set()
        finished = {}
        chunks = {rid: [] for rid in rids}
        cancel_ack = None
        done_marker = False
        async with d.http.stream("POST", "/generate", json=body) as response:
            d.record("native-batch-status", {"status": response.status_code})
            response.raise_for_status()
            async for line in response.aiter_lines():
                d.record("native-batch-line", line)
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done_marker = True
                    break
                item = json.loads(data)
                meta = item["meta_info"]
                rid = meta["id"]
                assert rid in chunks, item
                chunks[rid].append(item)
                reason = meta.get("finish_reason")
                if reason:
                    finished[rid] = item
                    active.discard(rid)
                elif meta.get("completion_tokens", 0) > 0:
                    active.add(rid)
                if cancel_ack is None and len(active) == 2:
                    # Both rids have emitted unfinished real GPU output. The
                    # scheduler's cancel targets only the selected binding.
                    cmd = d.command("cancel", rid=rids[0], version=versions[0])
                    cancel_ack, = await d.rpc(cmd)
                    assert cancel_ack["status"] == "accepted", cancel_ack
        assert done_marker and cancel_ack is not None, (done_marker, cancel_ack, finished)
        assert set(finished) == set(rids), finished
        assert finished[rids[0]]["meta_info"]["finish_reason"]["type"] == "abort", finished[rids[0]]
        assert finished[rids[1]]["meta_info"]["finish_reason"]["type"] == "length", finished[rids[1]]
        victim, survivor = [await d.wait(future) for future in futures]
        assert victim["status"] == "canceled" and victim["rid"] == rids[0], victim
        assert survivor["status"] in ("ready", "partial") and survivor["rid"] == rids[1], survivor
        assert survivor["seq"] > cancel_ack["seq"], (cancel_ack, survivor)
        assert survivor["details"]["token_ids"] == ids and survivor["matched_tokens"] == 0, survivor

        # The pinned launcher uses cumulative streaming; also accept native
        # incremental streaming if explicitly enabled in an attached engine.
        survivor_chunks = chunks[rids[1]]
        incremental = any(len(x["output_ids"]) < x["meta_info"].get("completion_tokens", 0)
                          for x in survivor_chunks)
        output_ids = ([token for item in survivor_chunks for token in item["output_ids"]]
                      if incremental else survivor_chunks[-1]["output_ids"])
        assert len(output_ids) == count and output_ids == baseline["output_ids"], (output_ids, baseline)
        result = {"path": "native /generate batch stream", "same_scheduler_batch_proven": False,
                  "victim_rid": rids[0], "survivor_rid": rids[1],
                  "cancel_ack_seq": cancel_ack["seq"], "victim_completion_seq": victim["seq"],
                  "survivor_completion_seq": survivor["seq"], "victim_status": victim["status"],
                  "survivor_status": survivor["status"], "survivor_matched_tokens": survivor["matched_tokens"],
                  "baseline_matched_tokens": baseline_receipt["matched_tokens"],
                  "survivor_output_ids": output_ids, "stream_incremental": incremental}
        d.record("single-victim-verified", result)
        return result
    finally:
        for bind in binds:
            d.control.unwatch(bind["action_id"])


async def verify(root: Path, name: str, expected_triton: bool) -> dict:
    from agentic_layer.engine_client import EngineClient

    control = EngineClient(root / "engine.sock")
    await control.handshake()
    assert all(control.capabilities[x] for x in ("bind", "prepare", "offload", "reclaim")), control.capabilities
    with (root / "logs" / f"{name}-wire.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        async with httpx.AsyncClient(base_url=ENGINE, trust_env=False, timeout=180) as http:
            d = Driver(control, http, log, name)
            d.record("handshake", control.snapshot)
            transfer = await cancel_transfer(d)
            batch = await cancel_batch_member(d)
            await control.poll()
            d.record("final-snapshot", control.snapshot)
            counters = control.snapshot["details"]["transfer_counters"]
            kind = "triton" if expected_triton else "native"
            assert all(counters.get(f"d2h_{pool}_{kind}_bytes", 0) > 0 for pool in ("full", "mamba")), counters
    return {"arm": name, "triton_copy": expected_triton, "transfer": transfer,
            "single_victim": batch, "transfer_counters": counters}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--triton", action="store_true")
    parser.add_argument("--attach", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.attach:
        result = asyncio.run(verify(root, args.name, args.triton))
    else:
        os.environ["AGENTIC_TRITON_COPY"] = str(int(args.triton))
        with engine(root, "extended-hicache", args.name):
            result = asyncio.run(verify(root, args.name, args.triton))
    write_json(root / f"{args.name}.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
