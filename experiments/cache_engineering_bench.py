#!/usr/bin/env python3
"""Measure the cost of a drop with cache engineering against a trim.

For each context size the bench

1. prefills a source prompt (``prepare``) so the whole chain is device-resident;
2. *trim*: prefills the compacted prompt cold, i.e. what text-level compaction
   pays today;
3. *cache-eng*: calls ``repos`` (drop + key rotation) and prefills only the
   delta, so the retained suffix is never recomputed.

It prints per-size medians, the prefill tokens each arm paid, and the ratio.
The synthetic ids are deterministic; only engine cost is measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import statistics
import sys
import time
import uuid

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agentic_layer.protocol import decode_line, encode_line  # noqa: E402


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def ids(n: int, seed: int) -> list[int]:
    out, x = [], seed
    for _ in range(n):
        x = (x * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append(1000 + (x >> 33) % 28000)
    return out


class Engine:
    def __init__(self, path: str):
        self.path = path
        self.epoch = self.call(None, [])["epoch"]

    def call(self, epoch, ops):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(180)
            sock.connect(self.path)
            sock.sendall(encode_line({"epoch": epoch, "ops": ops}))
            with sock.makefile("rb") as stream:
                reply = decode_line(stream.readline())
        if reply.get("error"):
            raise RuntimeError(reply["error"])
        return reply

    def op(self, op, rid, version, handle, payload):
        return self.call(self.epoch, [{
            "op": op, "action_id": f"{op}-{uuid.uuid4().hex[:12]}", "run_id": "ce-bench",
            "rid": rid, "version_id": version, "handle": handle,
            "payload": payload}])["replies"][0]

    def snap(self, after):
        return self.call(self.epoch, [{
            "op": "snapshot", "action_id": f"s-{uuid.uuid4().hex[:8]}", "run_id": "ce-bench",
            "rid": "s", "version_id": "v", "handle": None, "payload": {"after_seq": after}}])["replies"][0]


def prepare(client: httpx.Client, engine: Engine, prompt, salt, base):
    rid = f"p-{uuid.uuid4().hex[:10]}"
    bound = engine.op("bind", rid, "v", None, {
        "expected_ids": prompt, "cache_salt": salt, "kind": "prepare", "hold": False,
        "max_new_tokens": 0, "session_id": "bench", "branch_id": rid})
    if bound.get("status") != "ok":
        raise RuntimeError(f"bind: {bound}")
    started = time.monotonic()
    response = client.post("http://127.0.0.1:30080/generate", json={
        "input_ids": prompt, "rid": rid, "cache_salt": salt,
        "sampling_params": {"max_new_tokens": 0, "temperature": 0, "ignore_eos": True}})
    response.raise_for_status()
    wall = time.monotonic() - started
    snap = engine.snap(base)
    receipt = next((r for r in snap["details"]["completions"] if r["rid"] == rid), {})
    return wall, receipt.get("handle"), receipt.get("matched_tokens") or 0, snap["seq"]


def chat(client: httpx.Client, engine: Engine, prompt, salt, base, max_new: int = 1):
    rid = f"c-{uuid.uuid4().hex[:10]}"
    bound = engine.op("bind", rid, "v", None, {
        "expected_ids": prompt, "cache_salt": salt, "kind": "chat",
        "max_new_tokens": max_new, "session_id": "bench", "branch_id": rid})
    if bound.get("status") != "ok":
        raise RuntimeError(f"bind: {bound}")
    started = time.monotonic()
    response = client.post("http://127.0.0.1:30080/generate", json={
        "input_ids": prompt, "rid": rid, "cache_salt": salt,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True}})
    response.raise_for_status()
    wall = time.monotonic() - started
    snap = engine.snap(base)
    receipt = next((r for r in snap["details"]["completions"] if r["rid"] == rid), {})
    return wall, receipt.get("matched_tokens") or 0, snap["seq"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default="/root/autodl-tmp/agentic-layer-runs/5090-nvfp4-01/engine.sock")
    parser.add_argument("--sizes", default="1024,4096,8192")
    parser.add_argument("--drop-fraction", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    engine = Engine(args.socket)
    rows = []
    with httpx.Client(timeout=300) as client:
        base = engine.snap(0)["seq"]
        for size in [int(x) for x in args.sizes.split(",")]:
            source = ids(size, size)
            drop_from = size // 4
            drop_to = drop_from + int(size * args.drop_fraction)
            compact = source[:drop_from] + source[drop_to:]
            tail = ids(32, size + 7)
            for repeat in range(args.repeats):
                salt = digest(f"src:{size}:{repeat}:{uuid.uuid4().hex}")
                _, handle, _, base = prepare(client, engine, source, salt, base)
                cold_salt = digest(f"trim:{size}:{repeat}:{uuid.uuid4().hex}")
                trim_wall, trim_matched, base = chat(client, engine, compact + tail, cold_salt, base)
                ce_salt = digest(f"ce:{size}:{repeat}:{uuid.uuid4().hex}")
                started_repos = time.monotonic()
                repos = engine.op("repos", f"r-{uuid.uuid4().hex[:8]}", "v", handle, {
                    "expected_ids": compact, "cache_salt": ce_salt,
                    "dropped": [[drop_from, drop_to]]})
                repos_ms = (time.monotonic() - started_repos) * 1000
                if repos.get("status") != "completed":
                    raise RuntimeError(f"repos: {repos}")
                ce_wall, ce_matched, base = chat(client, engine, compact + tail, ce_salt, base)
                rows.append({
                    "size": size, "repeat": repeat,
                    "trim_ms": trim_wall * 1000, "trim_prefill": len(compact) + len(tail) - trim_matched,
                    "ce_ms": ce_wall * 1000, "ce_prefill": len(compact) + len(tail) - ce_matched,
                    "repos_ms": repos_ms,
                    "rotated_bytes": (repos.get("details") or {}).get("rotated_bytes"),
                })
    summary = {}
    for size in sorted({r["size"] for r in rows}):
        group = [r for r in rows if r["size"] == size]
        summary[str(size)] = {
            "trim_ms_median": statistics.median(r["trim_ms"] for r in group),
            "ce_ms_median": statistics.median(r["ce_ms"] for r in group),
            "trim_prefill": sum(r["trim_prefill"] for r in group),
            "ce_prefill": sum(r["ce_prefill"] for r in group),
            "speedup": (statistics.median(r["trim_ms"] for r in group) /
                        statistics.median(r["ce_ms"] for r in group)),
        }
    report = {"drop_fraction": args.drop_fraction, "repeats": args.repeats,
              "summary": summary, "rows": rows}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    with open("/root/autodl-tmp/agentic-layer-runs/cache-eng-bench.json", "w") as handle:
        json.dump(report, handle, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
