"""E2: fixed harness-confirmed worker calls, real host work and GPU return waves.

This driver owns the confirmed call list and all-of commit, not the DSH plugin.
The existing Policy models KV bytes, not host execution or branch slots. The only
ablation here is a FIFO limit on ready workers from tool start through GPU return.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier

import httpx

from .gpu_run import ENGINE, MODEL, Probe, _verify_fixed_graph, configuration, engine, token_fixture, write_json

WORKER_COUNTS = (1, 2, 4, 8)
PROFILES = ("cpu", "io")
ARMS = ("immediate", "budget")
IO_COUNTERS = ("read_bytes", "write_bytes")
MIB = 1024 * 1024


def now() -> int:
    return time.monotonic_ns()


def ms(end: int, start: int) -> float:
    return (end - start) / 1e6


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def timings(values) -> dict | None:
    ordered = sorted(values)
    if not ordered:
        return None
    return {"min": ordered[0], "median": statistics.median(ordered),
            "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "p99": ordered[math.ceil(0.99 * len(ordered)) - 1], "max": ordered[-1]}


def peak_concurrency(rows, start: str, finish: str) -> int:
    intervals = [(r[start], r[finish]) for r in rows if start in r and finish in r and r[start] < r[finish]]
    edges = [(t, delta) for a, b in intervals for t, delta in ((a, 1), (b, -1))]
    peak = active = 0
    for _, delta in sorted(edges):
        active += delta
        peak = max(peak, active)
    return peak


@dataclass(frozen=True)
class Call:
    id: str
    profile: str
    seed: bytes
    work_count: int
    expected_sha256: str
    block: bytes = b""

    def declaration(self) -> dict:
        return {"call_id": self.id, "tool": self.profile, "arguments": {"seed": self.seed.hex(),
                "iterations" if self.profile == "cpu" else "bytes": self.work_count},
                "predecessors": [], "access": "private exclusive scratch file" if self.block else "pure CPU",
                "expected_sha256": self.expected_sha256}


def cpu_digest(seed: bytes, iterations: int) -> str:
    result = hashlib.pbkdf2_hmac("sha256", seed, b"E2 immutable fixture", iterations)
    return hashlib.sha256(result).hexdigest()


def confirm_calls(profile: str, count: int, args) -> tuple[Call, ...]:
    calls = []
    for i in range(count):
        ident = f"worker-{i:02d}"
        seed = f"agentic-e2/v1/{profile}/{ident}".encode()
        if profile == "cpu":
            calls.append(Call(ident, profile, seed, args.cpu_iterations, cpu_digest(seed, args.cpu_iterations)))
        else:
            block = hashlib.shake_256(seed).digest(MIB)
            expected = hashlib.sha256()
            for _ in range(args.io_mib):
                expected.update(block)
            calls.append(Call(ident, profile, seed, args.io_mib * MIB, expected.hexdigest(), block))
    return tuple(calls)


def io_counters() -> dict:
    fields = dict(line.split(": ") for line in Path("/proc/thread-self/io").read_text().splitlines())
    return {key: int(fields[key]) for key in IO_COUNTERS}


def run_tool(call: Call, scratch: Path) -> dict:
    before = io_counters()
    thread_start, start = time.thread_time_ns(), now()
    sync_ms = read_ms = None
    if call.profile == "cpu":
        result = cpu_digest(call.seed, call.work_count)
        worked = call.work_count
    else:
        # Each confirmed call can only touch its own exclusively-created output.
        # Synchronous writeback is real filesystem work; readback may be cached.
        path = scratch / call.id
        with path.open("xb", buffering=0) as handle:
            for _ in range(call.work_count // MIB):
                if handle.write(call.block) != MIB:
                    raise OSError("short fixture write")
            sync_start = now()
            os.fdatasync(handle.fileno())
            sync_ms = ms(now(), sync_start)
        actual = hashlib.sha256()
        worked, read_start = 0, now()
        with path.open("rb", buffering=0) as handle:
            buf = memoryview(bytearray(MIB))
            while n := handle.readinto(buf):
                actual.update(buf[:n])
                worked += n
        read_ms = ms(now(), read_start)
        result = actual.hexdigest()
        path.unlink()
    thread_finish, finish = time.thread_time_ns(), now()
    after = io_counters()
    return {"call_id": call.id, "sha256": result, "work_count": worked, "start_ns": start, "finish_ns": finish,
            "tool_wall_ms": ms(finish, start), "thread_cpu_ms": ms(thread_finish, thread_start),
            "fdatasync_ms": sync_ms, "readback_ms": read_ms,
            **{f"kernel_{key}": after[key] - before[key] for key in IO_COUNTERS}}


class Wave:
    def __init__(self, http, raw, args, label):
        self.http, self.raw, self.args, self.label = http, raw, args, label
        self.salt = uuid.uuid4().hex
        self.transport_failed = False

    def event(self, event: str, **fields) -> None:
        self.raw.write(json.dumps({"attempt": self.label, "event": event, "mono_ns": now(), **fields},
                                  separators=(",", ":")) + "\n")

    async def generate(self, tag: str, ids: list[int], tokens: int) -> dict:
        # Force a fixed ordinary vocabulary symbol so dynamic batch composition
        # cannot change the work or output. All tokens still run through SGLang.
        token = 1000 + hashlib.sha256(tag.encode()).digest()[0]
        rid = uuid.uuid4().hex
        request = {"rid": rid, "input_ids": ids, "cache_salt": f"{self.salt}-{tag}",
                   "sampling_params": {"max_new_tokens": tokens, "ignore_eos": True, "temperature": 0,
                                       "logit_bias": {str(token): 10000.0}}}
        start = now()
        self.event("gpu-submit", tag=tag, rid=rid, submitted_ns=start)
        try:
            response = await self.http.post("/generate", json=request)
        except httpx.HTTPError as error:
            self.transport_failed = True
            self.event("gpu-transport-error", tag=tag, rid=rid, request=request, error=repr(error))
            raise
        finish = now()
        try:
            data = response.json()
        except ValueError:
            data = {"body": response.text}
        self.event("gpu-return", tag=tag, rid=rid, status=response.status_code, request=request,
                   response=data, finished_ns=finish)
        response.raise_for_status()
        meta = data["meta_info"]
        if (meta["id"] != rid or meta["prompt_tokens"] != len(ids) or meta["completion_tokens"] != tokens
                or meta["cached_tokens"] != 0 or meta["finish_reason"]["type"] != "length"
                or data["output_ids"] != [token] * tokens):
            raise ValueError(f"fixed GPU request, work or output mismatch: {tag}/{rid}: {meta}")
        queue_s = meta["queue_time"]
        if not math.isfinite(queue_s) or queue_s < 0 or not meta.get("forward_entry_time"):
            raise ValueError(f"missing real scheduler queue timing: {tag}/{rid}")
        return {"rid": rid, "gpu_start_ns": start, "gpu_finish_ns": finish, "gpu_wall_ms": ms(finish, start),
                "gpu_queue_ms": queue_s * 1000, "gpu_e2e_ms": meta["e2e_latency"] * 1000,
                "prompt_tokens": len(ids), "completion_tokens": tokens, "output_ids": data["output_ids"],
                "input_sha256": digest(ids), "weight_version": meta["weight_version"],
                "retractions": meta["num_retractions"]}

    async def worker(self, call, ordinal, scratch, pool, ready, phase_limit=None) -> dict:
        dispatched = now()
        observation = {"call_id": call.id, "dispatch_ns": dispatched, "admission_wait_ms": ms(dispatched, ready)}
        try:
            async with (phase_limit if phase_limit and self.args.budget_stage == 'tool' else nullcontext()):
                tool = await asyncio.get_running_loop().run_in_executor(pool, run_tool, call, scratch)
            observation.update(tool, executor_queue_ms=ms(tool["start_ns"], dispatched))
            self.event("tool-result", **observation)
            if (tool["call_id"], tool["sha256"], tool["work_count"]) != (call.id, call.expected_sha256, call.work_count):
                raise ValueError(f"fixture result or work count mismatch: {call.id}")
            ids = prompt(self.args.worker_prompt_tokens, 800 + ordinal, tool["sha256"])
            async with (phase_limit if phase_limit and self.args.budget_stage == 'gpu' else nullcontext()):
                gpu = await self.generate(call.id, ids, self.args.worker_tokens)
            observation.update(gpu, tool_to_gpu_submit_ms=ms(gpu["gpu_start_ns"], tool["finish_ns"]))
        except Exception as error:
            observation["error"] = repr(error)
        self.event("worker-terminal", **observation)
        return observation

    async def foreground(self, index: int, ready: int) -> dict:
        # Only exogenous arrivals are timed; the CPU and IO tools never sleep.
        await asyncio.sleep(max(0, (ready - now()) / 1e9))
        tag = f"foreground-{index:02d}"
        ids = token_fixture(self.args.root_prompt_tokens, 100 + index)
        return await self.root(tag, ready, ids)

    async def root(self, tag: str, ready: int, ids: list[int]) -> dict:
        row = {"root_id": tag, "ready_ns": ready}
        try:
            row.update(await self.generate(tag, ids, self.args.root_tokens))
            row["active_ms"] = ms(row["gpu_finish_ns"], ready)
        except Exception as error:
            row.update(error=repr(error), failure_elapsed_ms=ms(now(), ready))
        self.event("root-terminal", **row)
        return row


def prompt(length: int, seed: int, content_sha256: str) -> list[int]:
    return token_fixture(length - 32, seed) + [1000 + byte for byte in bytes.fromhex(content_sha256)]


async def dispatch_confirmed(wave, calls, limit, scratch, pool, ready) -> list[dict]:
    """Each slot accepts the next confirmed call only after its worker returns."""
    if wave.args.budget_stage != 'worker':
        phase = asyncio.Semaphore(limit)
        wave.event("phase-budget", stage=wave.args.budget_stage, concurrency_limit=limit,
                   ready_ns=ready, call_ids=[call.id for call in calls])
        return list(await asyncio.gather(*(wave.worker(call, i, scratch, pool, ready, phase)
                                           for i, call in enumerate(calls))))
    pending = iter(enumerate(calls))
    results = [None] * len(calls)

    async def slot(index):
        for ordinal, call in pending:
            wave.event("permit", call_id=call.id, slot=index, concurrency_limit=limit, ready_ns=ready)
            results[ordinal] = await wave.worker(call, ordinal, scratch, pool, ready)

    for call in calls[limit:]:
        wave.event("defer", call_id=call.id, reason="worker_budget", concurrency_limit=limit, ready_ns=ready)
    await asyncio.gather(*(slot(index) for index in range(limit)))
    return results


async def attempt(wave, calls, limit, scratch, pool) -> dict:
    args = wave.args
    ready = now()
    wave.event("confirmed-all-of", ready_ns=ready, jobs=len(calls), limit=limit,
               calls=[call.declaration() for call in calls])
    foreground = [asyncio.create_task(wave.foreground(i, ready + i * args.root_spacing_ms * 1_000_000))
                  for i in range(args.foreground_roots)]
    workers = await dispatch_confirmed(wave, calls, limit, scratch, pool, ready)
    # The all-of commits once, in declaration order, only after all terminal
    # statuses. Any worker failure suppresses the success model call, not siblings.
    joined = None
    if any("error" in worker for worker in workers):
        all_of = {"root_id": "all-of", "ready_ns": ready, "error": "required worker failed",
                  "failure_elapsed_ms": ms(now(), ready)}
        wave.event("root-terminal", **all_of)
    else:
        joined = [{key: row[key] for key in ("call_id", "sha256", "work_count", "output_ids")} for row in workers]
        wave.event("all-of-commit", observations=joined, join_elapsed_ms=ms(now(), ready))
        all_of = await wave.root("all-of", ready, prompt(args.root_prompt_tokens, 42, digest(joined)))
    fg = await asyncio.gather(*foreground)
    roots = [all_of, *fg]
    root_stats = timings(r["active_ms"] for r in roots if "active_ms" in r)
    fg_stats = timings(r["active_ms"] for r in fg if "active_ms" in r)
    gpu_finishes = [r["gpu_finish_ns"] for r in workers if "gpu_finish_ns" in r]
    gpu_starts = [r["gpu_start_ns"] for r in workers if "gpu_start_ns" in r]
    metrics = ("admission_wait_ms", "executor_queue_ms", "tool_wall_ms", "thread_cpu_ms", "fdatasync_ms",
               "readback_ms", "tool_to_gpu_submit_ms", "gpu_wall_ms", "gpu_queue_ms", "gpu_e2e_ms")
    ok = all("error" not in r for r in [*workers, *roots])
    return {"pass": ok, "job_limit": limit, "deferred_jobs": len(calls) - limit, "planned_roots": len(roots),
            "failed_workers": sum("error" in r for r in workers), "failed_roots": sum("error" in r for r in roots),
            "roots": roots, "workers": workers, "join_sha256": digest(joined) if joined is not None else None,
            "root_active_p95_ms": root_stats["p95"] if ok else None,
            "foreground_active_p95_ms": fg_stats["p95"] if all("error" not in r for r in fg) else None,
            "root_active_success_only_ms": root_stats, "foreground_active_success_only_ms": fg_stats,
            "all_of_active_ms": all_of.get("active_ms"),
            "worker_timings_ms": {key: timings(r[key] for r in workers if r.get(key) is not None) for key in metrics},
            "peak_concurrent_tool_intervals": peak_concurrency(workers, "start_ns", "finish_ns"),
            "peak_outstanding_worker_gpu_http": peak_concurrency(workers, "gpu_start_ns", "gpu_finish_ns"),
            "kernel_io_bytes": {key: sum(r.get(f"kernel_{key}", 0) for r in workers) for key in IO_COUNTERS},
            "root_gpu_queue_ms": timings(r["gpu_queue_ms"] for r in roots if "gpu_queue_ms" in r),
            "worker_gpu_submit_span_ms": ms(max(gpu_starts), min(gpu_starts)) if gpu_starts else None,
            "worker_gpu_finish_span_ms": ms(max(gpu_finishes), min(gpu_finishes)) if gpu_finishes else None}


def work_signature(row):
    gpu_keys = ("input_sha256", "prompt_tokens", "completion_tokens", "output_ids", "weight_version")
    worker_keys = ("call_id", "sha256", "work_count", *gpu_keys)
    return (row["join_sha256"], [[r[k] for k in worker_keys] for r in row["workers"]],
            [[r[k] for k in gpu_keys] for r in row["roots"]])


def median_ci(values: list[float]) -> list[float] | None:
    """Exact order-statistic interval for a paired median, at least 95% coverage."""
    n = len(values)
    ordered = sorted(values)
    ci = None
    tail = 0
    for i in range(n // 2):
        tail += math.comb(n, i)
        if 40 * tail > 2 ** n:
            break
        ci = [ordered[i], ordered[-i - 1]]
    return ci


def summarize(pairs: list[dict]) -> list[dict]:
    summary = []
    for profile, jobs in dict.fromkeys((p["profile"], p["jobs"]) for p in pairs):
        cases = [p for p in pairs if (p["profile"], p["jobs"]) == (profile, jobs)]
        effects = [p["p95_improvement_percent"] for p in cases if p.get("pass")]
        ci = median_ci(effects)
        median = statistics.median(effects) if effects else None
        summary.append({"profile": profile, "jobs": jobs, "planned_pairs": len(cases), "passed_pairs": len(effects),
                        "timings_ms_across_passed_pairs": {
                            key: {arm: timings(p["arms"][arm][key] for p in cases if p.get("pass")) for arm in ARMS}
                            for key in ("root_active_p95_ms", "foreground_active_p95_ms", "all_of_active_ms")},
                        "paired_p95_improvement_percent": effects, "median_improvement_percent": median,
                        "paired_median_ci95_percent": ci,
                        "credible_over_5_percent": bool(len(effects) == len(cases) and cases[0]["budget_limit"] < jobs
                                                        and ci and median > 5 and ci[0] > 0)})
    return summary


async def measure(report, args, log, scratch, calls, cores) -> None:
    connections = args.foreground_roots + max(args.workers) + 1
    limits = httpx.Limits(max_connections=connections, max_keepalive_connections=connections)
    with ThreadPoolExecutor(max_workers=max(args.workers), initializer=os.sched_setaffinity,
                            initargs=(0, cores)) as pool:
        async with httpx.AsyncClient(base_url=ENGINE, trust_env=False, timeout=180, limits=limits) as http:
            # Warm both host thread creation and the real GPU path without sharing
            # request salts or results with any timed arm.
            width = max(args.workers)
            barrier = Barrier(width, timeout=10)
            await asyncio.gather(*(asyncio.get_running_loop().run_in_executor(pool, barrier.wait) for _ in range(width)))
            warm = Wave(http, log, args, "warmup")
            await warm.generate("warmup", token_fixture(args.worker_prompt_tokens, 3), args.worker_tokens)
            for pair in report["pairs"]:
                for arm in pair["arm_order"]:
                    limit = pair["jobs"] if arm == "immediate" else pair["budget_limit"]
                    label = f"{pair['profile']}-{pair['jobs']}-{pair['repeat']}-{arm}"
                    directory = scratch / label
                    directory.mkdir()
                    wave = Wave(http, log, args, label)
                    rows = calls[pair["profile"]][:pair["jobs"]]
                    pair["arms"][arm] = await attempt(wave, rows, limit, directory, pool)
                    if wave.transport_failed:
                        # An unacknowledged request could still occupy engine slots.
                        raise RuntimeError(f"transport failure in {label}; not measuring another arm on this engine")
                a, b = (pair["arms"][arm] for arm in ARMS)
                pair["pass"] = a["pass"] and b["pass"] and work_signature(a) == work_signature(b)
                if pair["pass"]:
                    old, new = a["root_active_p95_ms"], b["root_active_p95_ms"]
                    pair["p95_improvement_percent"] = 100 * (old - new) / old
                wave.event("paired-sample", sample=pair)
                print(label.rsplit("-", 1)[0], {arm: pair["arms"][arm]["root_active_p95_ms"] for arm in ARMS},
                      "pass" if pair["pass"] else "FAILED", flush=True)


@contextmanager
def attached():
    with httpx.Client(base_url=ENGINE, trust_env=False, timeout=180) as client:
        yield client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--name", default="admission-e2")
    parser.add_argument("--engine-arm", choices=("native-radix", "extended"), default="native-radix")
    parser.add_argument("--attach-log", type=Path, help="attach to the owned engine with this actual capture log")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--workers", nargs="+", type=int, choices=WORKER_COUNTS, default=list(WORKER_COUNTS))
    parser.add_argument("--budget", type=int, choices=WORKER_COUNTS, default=2)
    parser.add_argument("--budget-stage", choices=("worker", "tool", "gpu"), default="worker",
                        help="hold one slot through both phases, tool only, or GPU return only")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--tool-cores", type=int, default=2)
    parser.add_argument("--cpu-iterations", type=int, default=250_000)
    parser.add_argument("--io-mib", type=int, default=32)
    parser.add_argument("--foreground-roots", type=int, default=20)
    parser.add_argument("--root-spacing-ms", type=int, default=15)
    parser.add_argument("--worker-prompt-tokens", type=int, default=1024)
    parser.add_argument("--worker-tokens", type=int, default=64)
    parser.add_argument("--root-prompt-tokens", type=int, default=256)
    parser.add_argument("--root-tokens", type=int, default=4)
    args = parser.parse_args()
    allowed = sorted(os.sched_getaffinity(0))
    if (len(set(args.workers)) != len(args.workers) or len(set(args.profiles)) != len(args.profiles)
            or not 1 <= args.tool_cores <= len(allowed)
            or min(args.repeats, args.cpu_iterations, args.io_mib, args.foreground_roots, args.root_spacing_ms,
                   args.worker_tokens, args.root_tokens) <= 0
            or not 32 <= args.worker_prompt_tokens <= 32768 - args.worker_tokens
            or not 32 <= args.root_prompt_tokens <= 32768 - args.root_tokens):
        parser.error("require unique conditions, positive work/arrival counts, and valid CPU and token budgets")
    if (os.getenv("AGENTIC_FIXED_GRAPH"), os.getenv("AGENTIC_DECODE_GRAPH")) != ("1", "full"):
        parser.error("set AGENTIC_FIXED_GRAPH=1 and AGENTIC_DECODE_GRAPH=full")
    root = args.run_dir.resolve()
    graph_log = args.attach_log.resolve() if args.attach_log else root / "logs" / f"{args.name}.log"
    raw_path = root / "logs" / f"{args.name}-waves.jsonl"
    output = root / f"{args.name}.json"
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    cores = set(allowed[:args.tool_cores])
    report = {"scope": "fixed benchmark-harness confirmation and dispatch; not production DSH admission",
              "budget_stage": args.budget_stage,
              "fixture": "independent foreground model turns plus one deterministic all-of branch turn; "
                         "constrained GPU decoding fixes outputs, not a model-quality benchmark",
              "io_accounting": "fdatasync is measured separately; readback may be page-cache served; "
                               "Linux per-thread I/O counters are reported as observed",
              "primary_metric": "nearest-rank P95 of all planned active roots per attempt, including the all-of root",
              "failure_rule": "no primary P95 or paired effect when any required worker/root fails or work differs",
              "comparison_rule": "median paired P95 reduction >5% and exact 95% paired median interval excludes zero",
              "invocation": [sys.executable, "-m", "experiments.admission_gpu", *sys.argv[1:]],
              "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "tool_cpu_affinity": sorted(cores), "engine_log": str(graph_log), "raw": str(raw_path), "pairs": []}
    for repeat in range(1, args.repeats + 1):
        for profile in args.profiles:
            for jobs in args.workers:
                report["pairs"].append({"repeat": repeat, "profile": profile, "jobs": jobs,
                                        "budget_limit": min(args.budget, jobs),
                                        "arm_order": ARMS if repeat % 2 else ARMS[::-1], "arms": {}})
    with raw_path.open("x", encoding="utf-8", buffering=1) as log:
        try:
            calls = {profile: confirm_calls(profile, max(args.workers), args) for profile in args.profiles}
            context = attached() if args.attach_log else engine(root, args.engine_arm, args.name)
            with context as client:
                probe = Probe(client, root / "logs" / f"{args.name}-config.jsonl")
                try:
                    report["engine_config"] = configuration(probe, args.engine_arm)
                    info = probe.get("/server_info").json()
                    model = probe.get("/model_info").json()
                    if model["model_path"] != str(MODEL) or info["max_running_requests"] != 8 or not info["enable_metrics"]:
                        raise RuntimeError("wrong pinned model, engine concurrency, or missing scheduler metrics")
                    report["model"] = model
                    report["graph"] = (_verify_fixed_graph(client, graph_log) if args.attach_log else
                                       json.loads((root / "logs" / f"{args.name}-graph.json").read_text()))
                finally:
                    probe.close()
                with tempfile.TemporaryDirectory(prefix="admission-scratch-", dir=root) as directory:
                    asyncio.run(measure(report, args, log, Path(directory), calls, cores))
        except BaseException as error:
            report["error"] = repr(error)
            raise
        finally:
            report["summary"] = summarize(report["pairs"])
            report["pass"] = "error" not in report and all(p.get("pass", False) for p in report["pairs"])
            write_json(output, report)
            print(output, flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
