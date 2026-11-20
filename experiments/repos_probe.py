#!/usr/bin/env python3
"""Probe cache engineering (drop + repos) on the live engine.

Sequence:
1. bind + generate prompt A, so its full hybrid chain is resident;
2. ``repos`` drops a middle token span and rotates the surviving keys;
3. bind + generate the compacted prompt with a fresh namespace and confirm the
   engine reuses the rotated chain instead of re-prefilling it;
4. repeat the compacted prompt cold and compare greedy outputs.

Prints one JSON document. ``--sha`` keeps the synthetic prompt reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import uuid

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agentic_layer.protocol import decode_line, encode_line  # noqa: E402


def engine_call(path: str, epoch: str | None, ops: list[dict]) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(120)
        sock.connect(path)
        sock.sendall(encode_line({"epoch": epoch, "ops": ops}))
        with sock.makefile("rb") as stream:
            reply = decode_line(stream.readline())
    if reply.get("error"):
        raise SystemExit(f"engine error: {reply['error']}")
    return reply


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def make_ids(n: int, seed: int) -> list[int]:
    out = []
    x = seed
    for _ in range(n):
        x = (x * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append(1000 + (x >> 33) % 28000)
    return out


def bind(sock_path: str, epoch: str, rid: str, version: str, ids: list[int], salt: str,
         max_new: int, *, kind: str = "chat", hold: bool = False) -> dict:
    return engine_call(sock_path, epoch, [{
        "op": "bind", "action_id": f"bind-{rid}-{uuid.uuid4().hex[:10]}", "run_id": "repos-probe", "rid": rid,
        "version_id": version, "handle": None,
        "payload": {"expected_ids": ids, "cache_salt": salt, "kind": kind, "hold": hold,
                    "max_new_tokens": max_new, "session_id": "probe", "branch_id": rid}}])["replies"][0]


def generate(ids: list[int], rid: str, salt: str, max_new: int) -> dict:
    body = {"input_ids": ids, "rid": rid, "cache_salt": salt,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True}}
    started = time.monotonic()
    response = httpx.post("http://127.0.0.1:30080/generate", json=body, timeout=300)
    response.raise_for_status()
    payload = response.json()
    payload["_wall_s"] = time.monotonic() - started
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default="/root/autodl-tmp/agentic-layer-runs/5090-nvfp4-01/engine.sock")
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--drop-start", type=int, default=512)
    parser.add_argument("--drop-end", type=int, default=1024)
    parser.add_argument("--probe", type=int, default=32)
    parser.add_argument("--max-new", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    epoch = engine_call(args.socket, None, [])["epoch"]
    run = uuid.uuid4().hex[:6]
    source = make_ids(args.length, args.seed)
    base = engine_call(args.socket, epoch, [{"op": "snapshot", "action_id": f"base-{uuid.uuid4().hex[:8]}",
        "run_id": "repos-probe", "rid": "base", "version_id": "v1", "handle": None, "payload": {}}])
    base_seq = base["replies"][0]["seq"]
    salt1 = digest("source")
    bound = bind(args.socket, epoch, f"u1-{run}", "v1", source, salt1, 0, kind="prepare", hold=True)
    assert bound["status"] == "ok", bound
    warm = generate(source, f"u1-{run}", salt1, 0)
    time.sleep(0.5)
    snap = engine_call(args.socket, epoch, [{"op": "snapshot", "action_id": f"s1-{uuid.uuid4().hex[:10]}", "run_id": "repos-probe",
                                             "rid": "s1", "version_id": "v1", "handle": None,
                                             "payload": {"after_seq": base_seq}}])["replies"][0]
    receipts = {r["rid"]: r for r in snap["details"]["completions"]}
    receipt = receipts.get(f"u1-{run}")
    assert receipt and receipt["handle"], (sorted(receipts), snap["seq"])
    keep = [i for i in range(len(source)) if not args.drop_start <= i < args.drop_end]
    compact = [source[i] for i in keep]
    salt2 = digest("compacted")
    repos = engine_call(args.socket, epoch, [{
        "op": "repos", "action_id": f"repos-{uuid.uuid4().hex[:10]}", "run_id": "repos-probe", "rid": "repos-1",
        "version_id": "v2", "handle": receipt["handle"],
        "payload": {"expected_ids": compact, "cache_salt": salt2,
                    "dropped": [[args.drop_start, args.drop_end]]}}])["replies"][0]
    assert repos["status"] == "completed", repos
    tail = make_ids(args.probe, args.seed + 1)
    reused_ids = compact + tail
    bound = bind(args.socket, epoch, f"u2-{run}", "v2", reused_ids, salt2, args.max_new)
    assert bound["status"] == "ok", bound
    reused = generate(reused_ids, f"u2-{run}", salt2, args.max_new)
    salt3 = digest("cold")
    bound = bind(args.socket, epoch, f"u3-{run}", "v3", reused_ids, salt3, args.max_new)
    assert bound["status"] == "ok", bound
    cold = generate(reused_ids, f"u3-{run}", salt3, args.max_new)
    time.sleep(0.5)
    snap2 = engine_call(args.socket, epoch, [{"op": "snapshot", "action_id": f"s2-{uuid.uuid4().hex[:10]}", "run_id": "repos-probe",
                                              "rid": "s2", "version_id": "v3", "handle": None,
                                              "payload": {"after_seq": snap["seq"]}}])["replies"][0]
    receipts = {r["rid"]: r for r in snap2["details"]["completions"]}
    report = {
        "source_tokens": len(source),
        "dropped": args.drop_end - args.drop_start,
        "kept": len(compact),
        "repos": {k: repos.get(k) for k in ("status", "matched_tokens", "error")} | (repos.get("details") or {}),
        "warm": {"wall_s": warm["_wall_s"], "ttft": warm["meta_info"].get("e2e_latency")},
        "reused": {"wall_s": reused["_wall_s"], "matched": receipts.get(f"u2-{run}", {}).get("matched_tokens"),
                   "output": reused.get("output_ids")},
        "cold": {"wall_s": cold["_wall_s"], "matched": receipts.get(f"u3-{run}", {}).get("matched_tokens"),
                 "output": cold.get("output_ids")},
        "same_output": reused.get("output_ids") == cold.get("output_ids"),
    }
    t_reused = receipts.get(f"u2-{run}", {}).get("matched_tokens") or 0
    report["prefill_tokens_reused"] = t_reused
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if repos["status"] == "completed" and t_reused >= len(compact) else 1


if __name__ == "__main__":
    raise SystemExit(main())
