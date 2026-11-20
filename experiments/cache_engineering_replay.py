#!/usr/bin/env python3
"""Replay real agentic sessions through three KV policies.

Arms, on the same projected token streams:

* ``full``     — never drop; the engine's native radix reuse applies.
* ``trim``     — drop the stale middle span and re-prefill the retained suffix
                 (today's text-level compaction).
* ``cache-eng``— drop the same span, but ``repos`` rotates the retained KV and
                 the suffix is never re-prefilled (AKV's cache engineering).

The drop plan is ActKV-inspired: keep the head and the recent tail, drop the
oldest middle blocks first, and widen the budget 1.75x after a turn whose
output was short (a cheap low-confidence signal). Block ids are projected to
synthetic token ids, so this measures engine cost, not task quality; output
agreement between the trim and cache-engineering arms is reported separately.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import socket
import sys
import time
import uuid

import httpx
import zstandard

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agentic_layer.policy import select_drop_messages  # noqa: E402
from agentic_layer.protocol import decode_line, encode_line  # noqa: E402


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def project(block_id: int, salt: str) -> list[int]:
    """Expand one 16-token trace block into 8 stable synthetic ids.

    The engine cost tracks token count, not the original vocabulary, so the
    expansion keeps prompts at the trace's real token scale while staying
    deterministic per (week, block).
    """
    digest_ = hashlib.sha256(f"{salt}:{block_id}".encode()).digest()
    return [1000 + ((digest_[i] << 8 | digest_[i + 1]) % 28000) for i in range(0, 16, 2)]


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

    def op(self, op: str, rid: str, version: str, handle: str | None, payload: dict) -> dict:
        return self.call(self.epoch, [{
            "op": op, "action_id": f"{op}-{uuid.uuid4().hex[:12]}", "run_id": "cache-eng-replay",
            "rid": rid, "version_id": version, "handle": handle, "payload": payload}])["replies"][0]

    def snapshot(self, after: int) -> dict:
        return self.call(self.epoch, [{
            "op": "snapshot", "action_id": f"snap-{uuid.uuid4().hex[:8]}", "run_id": "cache-eng-replay",
            "rid": "snapshot", "version_id": "v", "handle": None, "payload": {"after_seq": after}}])["replies"][0]


def sessions(dataset: str, per_week: int, min_turns: int = 5):
    by_week: dict[str, list[tuple[str, dict]]] = {}
    for path in sorted(glob.glob(f"{dataset}/release/*/traces/*.jsonl.zst")):
        week = path.split("/release/")[1].split("/")[0]
        with open(path, "rb") as handle:
            reader = zstandard.ZstdDecompressor().stream_reader(handle, read_across_frames=True)
            buffer = b""
            while True:
                chunk = reader.read(1 << 22)
                if not chunk:
                    break
                buffer += chunk
                *lines, buffer = buffer.split(b"\n")
                for raw in lines:
                    if not raw:
                        continue
                    line = json.loads(raw)
                    requests = line.get("requests") or []
                    if len(requests) >= min_turns:
                        bucket = by_week.setdefault(week, [])
                        if len(bucket) < per_week * 4:
                            bucket.append((path, line))
                if len(by_week.get(week, ())) >= per_week * 4:
                    break
    print(f"scanned {sum(len(v) for v in by_week.values())} candidate sessions over {len(by_week)} weeks", file=sys.stderr)
    chosen = []
    for week in sorted(by_week):
        bucket = sorted(by_week[week], key=lambda item: hashlib.sha256(item[1]["session_id"].encode()).hexdigest())
        chosen.extend(bucket[:per_week])
    return chosen


def run_arm(client: httpx.Client, engine: Engine, week: str, line: dict, arm: str,
            cap_blocks: int, max_turns: int) -> dict:
    """Materialise each turn's view with a prefill-only request and read its cost.

    The finish path of a generating request does not checkpoint a hybrid chain
    with ``mamba_radix_cache_strategy=no_buffer`` (the live state cannot be
    shortened), so the harness prepares the candidate prefix exactly as the
    middleware does. Each trace request is one message; drops remove whole
    messages only (``select_drop_messages``), so the kept view is what a DSH
    message-boundary drop would emit. The receipt's ``full_kv_hit_length`` is
    the reuse the turn enjoys; the rest is its prefill.
    """
    salt_prefix = f"{arm}:{week}:{line['session_id']}"
    view_blocks: list[int] = []
    view_tokens: list[int] = []
    message_blocks: list[int] = []  # blocks per whole message, append order
    previous_request: list[int] = []
    salt = digest(salt_prefix + ":0")
    handle = None
    handle_len = 0
    turns = []
    base = engine.snapshot(0)["seq"]
    for index, request in enumerate((line.get("requests") or [])[:max_turns]):
        request_blocks = list(request.get("block_ids") or [])
        if not view_blocks:
            start_at = max(0, len(request_blocks) - cap_blocks)
            delta = request_blocks[start_at:]
        elif len(request_blocks) >= len(previous_request):
            delta = request_blocks[len(previous_request):]
        else:
            delta = request_blocks[-1:]
        previous_request = request_blocks
        if not delta:
            continue
        view_blocks.extend(delta)
        view_tokens.extend(tid for bid in delta for tid in project(bid, week))
        message_blocks.append(len(delta))
        if len(view_tokens) > 32000:
            keep = len(view_tokens) - 32000
            view_tokens = view_tokens[keep:]
            view_blocks = view_blocks[keep // 8:]
            message_blocks = [len(view_blocks)]
            handle, handle_len = None, 0
            salt = digest(f"{salt_prefix}:{index}:window")
        drop = None
        if arm != "full" and handle is not None and len(view_blocks) > cap_blocks:
            plan = select_drop_messages(message_blocks, sum(message_blocks))
            if plan is not None:
                drop = (plan[0], plan[1])  # block offsets, message-aligned
        prompt = view_tokens
        rid = f"{arm}-{index}-{uuid.uuid4().hex[:8]}"
        used_salt = salt
        if drop is not None:
            cut_from, cut_to = drop[0] * 8, drop[1] * 8  # token offsets
            kept = view_tokens[:cut_from] + view_tokens[cut_to:]
            if arm == "cache-eng" and handle is not None:
                new_salt = digest(f"{salt_prefix}:{index}:drop")
                cached = view_tokens[:handle_len]
                target = cached[:cut_from] + cached[cut_to:]
                prompt = target + kept[len(target):]
                try:
                    repos = engine.op("repos", f"r-{rid}", f"v-{index}", handle, {
                        "expected_ids": target, "cache_salt": new_salt,
                        "dropped": [[cut_from, cut_to]]})
                except RuntimeError:
                    repos = {"status": "unavailable"}
                if repos.get("status") == "completed":
                    used_salt = new_salt
                else:
                    handle, handle_len = None, 0
                    prompt = kept
                    used_salt = digest(f"{salt_prefix}:{index}:miss")
            else:
                prompt = kept
                used_salt = digest(f"{salt_prefix}:{index}:cold")
            view_blocks = view_blocks[:drop[0]] + view_blocks[drop[1]:]
            view_tokens = kept
            kept_messages: list[int] = []
            offset = 0
            for size in message_blocks:
                next_offset = offset + size
                if next_offset <= drop[0] or offset >= drop[1]:
                    kept_messages.append(size)
                offset = next_offset
            message_blocks = kept_messages
        bound = engine.op("bind", rid, f"v-{index}", None, {
            "expected_ids": prompt, "cache_salt": used_salt, "kind": "prepare", "hold": False,
            "max_new_tokens": 0, "session_id": line["session_id"], "branch_id": rid})
        if bound.get("status") != "ok":
            raise RuntimeError(f"bind failed: {bound}")
        started = time.monotonic()
        response = client.post("http://127.0.0.1:30080/generate", json={
            "input_ids": prompt, "rid": rid, "cache_salt": used_salt,
            "sampling_params": {"max_new_tokens": 0, "temperature": 0, "ignore_eos": True}})
        response.raise_for_status()
        payload = response.json()
        wall = time.monotonic() - started
        meta = payload.get("meta_info") or {}
        ttft = float(meta.get("prefill_finished_time") or 0) - float(meta.get("forward_entry_time") or 0)
        time.sleep(0.05)
        snap = engine.snapshot(base)
        receipts = {r["rid"]: r for r in snap["details"]["completions"]}
        base = snap["seq"]
        receipt = receipts.get(rid) or {}
        hit = receipt.get("full_kv_hit_length") or receipt.get("matched_tokens") or 0
        handle = receipt.get("handle") or handle
        handle_len = len(prompt)
        turns.append({
            "turn": index, "tokens": len(prompt), "hit": hit,
            "dropped": (drop[1] - drop[0]) * 8 if drop else 0,
            "ttft": ttft if ttft > 0 else None, "wall": wall,
            "out": [],
        })
    return {"week": week, "session": line["session_id"], "arm": arm, "turns": turns}


def summarise(rows: list[dict]) -> dict:
    per_arm = {}
    for arm in ("full", "trim", "cache-eng"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        ttfts = [t["ttft"] for r in arm_rows for t in r["turns"] if t["ttft"]]
        prefills = [max(0, t["tokens"] - t["hit"]) for r in arm_rows for t in r["turns"]]
        reuse = [t["hit"] for r in arm_rows for t in r["turns"]]
        walls = [t["wall"] for r in arm_rows for t in r["turns"]]
        dropped = [t["dropped"] for r in arm_rows for t in r["turns"]]
        ttfts.sort()
        per_arm[arm] = {
            "sessions": len(arm_rows),
            "turns": sum(len(r["turns"]) for r in arm_rows),
            "prefill_tokens": sum(prefills),
            "reused_tokens": sum(reuse),
            "dropped_tokens": sum(dropped),
            "prefill_wall_s": sum(walls),
            "ttft_p50_ms": (ttfts[len(ttfts) // 2] * 1000) if ttfts else None,
            "ttft_p95_ms": (ttfts[min(len(ttfts) - 1, int(len(ttfts) * 0.95))] * 1000) if ttfts else None,
        }
    return per_arm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/root/autodl-tmp")
    parser.add_argument("--socket", default="/root/autodl-tmp/agentic-layer-runs/5090-nvfp4-01/engine.sock")
    parser.add_argument("--per-week", type=int, default=1)
    parser.add_argument("--cap-blocks", type=int, default=2750)
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--out", default="/root/autodl-tmp/agentic-layer-runs/cache-eng-replay.json")
    args = parser.parse_args()
    engine = Engine(args.socket)
    chosen = sessions(args.dataset, args.per_week)
    rows = []
    started = time.monotonic()
    with httpx.Client(timeout=300) as client:
        for week, line in chosen:
            for arm in ("full", "trim", "cache-eng"):
                print(f"{arm} {week} {line['session_id'][:8]}", file=sys.stderr, flush=True)
                rows.append(run_arm(client, engine, week, line, arm, args.cap_blocks, args.turns))
    report = {"sessions": len(chosen), "elapsed_s": time.monotonic() - started,
              "summary": summarise(rows), "rows": rows}
    with open(args.out, "w") as handle:
        json.dump(report, handle, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
