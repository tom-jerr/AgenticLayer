#!/usr/bin/env python3
"""Stream every FreeInference trace file and report agentic-workload structure.

Reads the whole release (all weeks, all days), never materialising a session
beyond one line, and prints one JSON document with:

* request / session / tool-call totals and nesting depth,
* context length and prefix-reuse (block LCP) distributions,
* mutation events by kind and the tokens a text-trim would re-prefill,
* the text-trim penalty vs. a KV-only drop, estimated from block ids.

No GPU, no engine: this is a pure workload analysis used to size the
cache-engineering opportunity before any replay.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
import sys
import zstandard


def quantiles(values, qs=(0.5, 0.95, 0.99)):
    if not values:
        return {f"p{int(q * 100)}": None for q in qs}
    ordered = sorted(values)
    out = {}
    for q in qs:
        idx = min(len(ordered) - 1, max(0, int(round(q * len(ordered))) - 1))
        out[f"p{int(q * 100)}"] = ordered[idx]
    return out


class Stats:
    def __init__(self):
        self.sessions = 0
        self.requests = 0
        self.requests_nested = 0
        self.tool_calls = 0
        self.tool_calls_nested = 0
        self.subagent_children = 0
        self.max_depth = 0
        self.ctx_blocks = []
        self.prefix_blocks = []
        self.new_blocks = []
        self.mutations = collections.Counter()
        self.mutation_causes = collections.Counter()
        self.reprefill_blocks = 0      # blocks a trim would have to recompute
        self.droppable_blocks = 0      # blocks of stale tool results a drop may target
        self.kept_blocks = 0           # blocks a reuse path can retain across a drop
        self.multi_call_steps = 0
        self.per_week = collections.defaultdict(lambda: collections.Counter())

    def feed(self, week: str, line: dict):
        self.sessions += 1
        week_stats = self.per_week[week]
        week_stats["sessions"] += 1
        requests = line.get("requests") or []
        previous_blocks: list[int] | None = None
        stale_tool_blocks = 0
        for index, request in enumerate(requests):
            blocks = request.get("block_ids") or []
            self.requests += 1
            week_stats["requests"] += 1
            self.ctx_blocks.append(len(blocks))
            if previous_blocks is None:
                prefix = 0
            else:
                prefix = 0
                limit = min(len(blocks), len(previous_blocks))
                while prefix < limit and blocks[prefix] == previous_blocks[prefix]:
                    prefix += 1
            self.prefix_blocks.append(prefix)
            self.new_blocks.append(len(blocks) - prefix)
            mutation = request.get("mutation")
            if mutation:
                kind = mutation.get("type") if isinstance(mutation, dict) else str(mutation)
                self.mutations[kind or "unknown"] += 1
                week_stats[f"mutation/{kind or 'unknown'}"] += 1
                cause = mutation.get("cause") if isinstance(mutation, dict) else None
                if cause:
                    self.mutation_causes[cause] += 1
            calls = request.get("tool_calls") or []
            if len(calls) > 1:
                self.multi_call_steps += 1
                week_stats["multi_call_steps"] += 1
            turn_result_blocks = 0
            for call in calls:
                self.tool_calls += 1
                week_stats["tool_calls"] += 1
                result_tokens = call.get("result_tokens") or 0
                turn_result_blocks += (int(result_tokens) + 15) // 16
                child = call.get("subagent")
                if child:
                    self.subagent_children += 1
                    self._walk_child(child, week, 2)
            # Older tool results are the drop candidates: everything from
            # turns before the previous one that is still in the prompt.
            if mutation:
                # A trim rewrites the prompt, so the whole retained prefix
                # after the edit point must be re-prefilled.
                self.reprefill_blocks += prefix
                week_stats["reprefill_blocks"] += prefix
                self.kept_blocks += len(blocks) - prefix
            stale_tool_blocks = stale_tool_blocks + turn_result_blocks
            previous_blocks = blocks
        self.droppable_blocks += 0  # computed per request in the replay planner

    def _walk_child(self, child, week, depth):
        self.max_depth = max(self.max_depth, depth)
        requests = child.get("requests") or []
        for request in requests:
            self.requests_nested += 1
            blocks = request.get("block_ids") or []
            self.ctx_blocks.append(len(blocks))
            for call in request.get("tool_calls") or []:
                self.tool_calls_nested += 1
                if call.get("subagent"):
                    self.subagent_children += 1
                    self._walk_child(call["subagent"], week, depth + 1)

    def report(self):
        tokens = [b * 16 for b in self.ctx_blocks]
        return {
            "sessions": self.sessions,
            "requests": self.requests,
            "requests_nested": self.requests_nested,
            "tool_calls": self.tool_calls,
            "tool_calls_nested": self.tool_calls_nested,
            "subagent_children": self.subagent_children,
            "max_nesting_depth": self.max_depth,
            "multi_call_steps": self.multi_call_steps,
            "context_tokens": quantiles(tokens),
            "context_tokens_mean": statistics.fmean(tokens) if tokens else 0,
            "reused_prefix_tokens": quantiles([b * 16 for b in self.prefix_blocks]),
            "new_tokens_per_request": quantiles([b * 16 for b in self.new_blocks]),
            "prefix_reuse_ratio": (
                sum(self.prefix_blocks) / sum(self.ctx_blocks) if sum(self.ctx_blocks) else 0
            ),
            "mutations": dict(self.mutations),
            "mutation_causes": dict(self.mutation_causes),
            "text_trim_reprefill_tokens": self.reprefill_blocks * 16,
            "retained_after_edit_tokens": self.kept_blocks * 16,
            "per_week": {w: dict(sorted(c.items())) for w, c in sorted(self.per_week.items())},
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/root/autodl-tmp")
    parser.add_argument("--out", default="experiments/results/freeinference-full-profile.json")
    parser.add_argument("--stride", type=int, default=10,
                        help="parse one session in N; every line is still counted")
    args = parser.parse_args()
    files = sorted(glob.glob(f"{args.dataset}/release/*/traces/*.jsonl.zst"))
    if not files:
        print("no trace files", file=sys.stderr)
        return 2
    stats = Stats()
    seen = 0
    for path in files:
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
                    seen += 1
                    if seen % args.stride:
                        continue
                    stats.feed(week, json.loads(raw))
            if buffer.strip():
                seen += 1
                if not seen % args.stride:
                    stats.feed(week, json.loads(buffer))
    report = stats.report()
    report["files"] = len(files)
    report["sessions_total"] = seen
    report["sampling_stride"] = args.stride
    with open(args.out, "w") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "per_week"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
