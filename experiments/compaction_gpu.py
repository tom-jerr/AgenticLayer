"""E3: paid A||B -> A'||B transition through authenticated ingress and native SGLang.

Run with AGENTIC_FIXED_GRAPH=1 AGENTIC_DECODE_GRAPH=full python -m experiments.compaction_gpu
--run-dir RUN --name e3 --pairs 10. This is a controller fixture, not DSH's
compaction transaction. The fixed tool is an opaque barrier: the retained B is
already committed, and no predicted tool output or token IDs enter the proposal.
The task checks one extractive fact and its use, not open-ended summary quality.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx

from .gpu_run import ENGINE, engine, write_json
from .hybrid_gpu import INGRESS, ingress
from .prepare_gpu import MODEL, Branch, action, reset_cache


@dataclass(frozen=True)
class Scenario:
    nominal_ms: int
    actual_ms: float
    changed: bool = False


WAITS = (25, 250, 2000)
SCENARIOS = {**{f"stable-{ms}": Scenario(ms, ms) for ms in WAITS},
             **{f"early-{ms}": Scenario(ms, ms / 4) for ms in WAITS},
             "changed-prefix-2000": Scenario(2000, 2000, changed=True)}
CITIES = ("oslo", "rome", "perth", "milan")
ARMS = ("foreground", "early")
TERMINAL = {"completed", "canceled", "error", "committed"}
ELAPSED_STEP_MS = 25
CALIBRATION_SAMPLES = 20
METRICS = ("foreground_ms", "total_ms", "summary_ms", "summary_exposed_ms", "generation_ms", "wait_ms")


def now() -> int:
    return time.monotonic_ns()


def ms(start: int, end: int) -> float:
    return (end - start) / 1e6


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def emit(log, row: dict) -> None:
    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def message(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def history(city: str) -> dict:
    notes = "".join(f"Archive entry {i:03d}: neutral routine inspection closed with no further action.\n"
                    for i in range(380))
    return message("system", "The following is the archived case log. Use only its active case facts.\n"
                   + notes + f"Active case field record: assigned city label = {city}.")


def retained(changed: bool) -> dict:
    case = "UPPERCASE" if changed else "lowercase"
    notes = "".join(f"Reference entry {i:03d}: routine material review noted stock and no changes.\n"
                    for i in range(260))
    return message("user", f"Output formatting for this request: {case}.\n" + notes
                   + f"What assigned city label does the case summary give? Output only that word, in {case}; "
                     "no punctuation, label, or explanation.")


def summary_prompt(source: dict) -> dict:
    return chat(source, message("user", "Summarize the active case by extracting its assigned city label. "
                                     "Reply with that one lowercase word only, without punctuation or explanation."))


def compacted(text: str) -> dict:
    return message("system", f"Compacted case log. The assigned city label is: {text}")


def chat(*messages: dict) -> dict:
    return {"model": MODEL, "messages": list(messages), "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}, "max_completion_tokens": 12}


def content(response: dict) -> str:
    choice, = response["choices"]
    text = choice["message"]["content"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("native chat did not return nonempty text")
    return text


def wait_type(nominal: int, elapsed: int) -> str:
    return f"e3_{nominal}ms_elapsed_{elapsed}ms"


async def tool_wait(duration_ms: float) -> int:
    await asyncio.sleep(duration_ms / 1000)
    return now()


async def calibrate(root: Path, name: str, scenarios: list[str]) -> Path:
    """Measure native target prefill and the nominal tool clock on held-out calls."""
    path = root / f"{name}-calibration.json"
    nominal_waits = sorted({SCENARIOS[key].nominal_ms for key in scenarios})
    async with httpx.AsyncClient(base_url=ENGINE, trust_env=False, timeout=180) as http:
        with (root / "logs" / f"{name}-calibration.jsonl").open("x", encoding="utf-8", buffering=1) as log:
            async def wait_samples(nominal):
                samples = []
                for i in range(CALIBRATION_SAMPLES):
                    started = now()
                    sample = {"duration_ms": ms(started, await tool_wait(nominal)), "completed": True}
                    emit(log, {"phase": "nominal_tool", "nominal_ms": nominal, "sample": i, **sample})
                    samples.append(sample)
                return nominal, samples

            async def prefill():
                await reset_cache(http)
                started = now()
                response = await http.post("/v1/chat/completions", json=summary_prompt(history("berlin")))
                result = response.json()
                emit(log, {"phase": "heldout_summary", "http": response.status_code,
                           "wall_ms": ms(started, now()), "response": result})
                response.raise_for_status()
                text = content(result)
                if text.strip() != "berlin":
                    raise RuntimeError("held-out summary failed the fixture's extractive check")
                response = await http.post("/v1/tokenize", json=chat(compacted(text), retained(False)))
                response.raise_for_status()
                ids = response.json()["tokens"]
                if len(ids) < 2:
                    raise RuntimeError("native tokenization produced no reusable prefix")
                samples, evictions = [], []
                for i in range(5 + CALIBRATION_SAMPLES):
                    await reset_cache(http)
                    salt = uuid.uuid4().hex
                    started = now()
                    probe = await http.post("/generate", json={"input_ids": ids[:-1], "cache_salt": salt,
                        "sampling_params": {"max_new_tokens": 0, "temperature": 0}})
                    elapsed = ms(started, now())
                    raw = probe.json()
                    emit(log, {"phase": "prefill", "warmup": i < 5, "sample": i, "wall_ms": elapsed,
                               "http": probe.status_code, "response": raw})
                    probe.raise_for_status()
                    meta = raw["meta_info"]
                    if meta["prompt_tokens"] != len(ids) - 1 or meta["cached_tokens"] != 0:
                        raise RuntimeError(f"prefill calibration was not an exact cold prefix: {meta}")
                    if i < 5:
                        continue
                    samples.append(elapsed)
                    delay = nominal_waits[(i - 5) % len(nominal_waits)]
                    started = now()
                    actual = ms(started, await tool_wait(delay))
                    probe = await http.post("/generate", json={"input_ids": ids, "cache_salt": salt,
                        "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
                    raw = probe.json()
                    emit(log, {"phase": "reuse_probe", "sample": i, "wait_ms": actual,
                               "expected_cached_tokens": len(ids) - 1, "http": probe.status_code, "response": raw})
                    probe.raise_for_status()
                    evictions.append({"evicted": raw["meta_info"]["cached_tokens"] != len(ids) - 1})
                # Two spare completion budgets cover the summary text and case formatting differences.
                bucket = next(n for n in (1024, 4096, 8192, 16384, 32768) if n >= len(ids) + 24)
                return {"token_count": bucket, "batch_size": 1, "prefill_ms": statistics.median(samples),
                        "d2h_ms": 0, "h2d_ms": 0, "recurrent_ms": 0, "samples": len(samples)}, evictions, len(ids)

            async with asyncio.TaskGroup() as group:
                timer_tasks = [group.create_task(wait_samples(nominal)) for nominal in nominal_waits]
                prefill_task = group.create_task(prefill())
            bucket, evictions, target_tokens = prefill_task.result()
            waits = {}
            for task in timer_tasks:
                nominal, samples = task.result()
                # The policy clock starts at proposal receipt, not tool start.
                # Round elapsed age up and condition the measured nominal waits on survival to that age.
                for age in range(0, nominal + 1, ELAPSED_STEP_MS):
                    waits[wait_type(nominal, age)] = [
                        {"duration_ms": s["duration_ms"] - age, "completed": True}
                        for s in samples if s["duration_ms"] > age]
    write_json(path, {"scope": "E3 low-pressure same-shape batch-one native prefill; no host moves",
                      "buckets": [bucket], "waits": waits, "evictions": {"low": evictions},
                      "heldout_target_tokens": target_tokens})
    return path


async def run_arm(http, scenario: Scenario, arm: str, city: str, row: dict, *, tool_runner=None) -> None:
    b = Branch(http, "e3-" + uuid.uuid4().hex)
    events = row["events"] = []
    row.update(run_id=b.run, branch_id=b.id, prepare_outcome="not_requested" if arm == "foreground" else None)
    seq_lock = asyncio.Lock()
    tasks = []
    source, tail = history(city), retained(False)
    boundary_ids = [digest(source)]
    original_ids = boundary_ids + [digest(tail)]
    turn_id = uuid.uuid4().hex

    async def event(kind, payload):
        async with seq_lock:
            expected = b.version
            begin = now()
            ack = await b.event(kind, payload)
        events.append({"kind": kind, "sent_ns": begin, "ack_ns": now(), "expected_version": expected,
                       "payload": {k: v for k, v in payload.items() if k != "chat"}, "ack": ack})
        return ack

    async def ready(work):
        ack = await event("ready", work)
        permit = ack["permit"]
        if (ack["status"] != "granted" or not permit or permit["action_id"] != work["action_id"]
                or permit["version_id"] != b.version or permit["expires_mono_ns"] <= now()):
            raise RuntimeError(f"no valid permit: {ack}")
        return permit["version_id"]

    async def generate(kind, body, dependencies=()):
        work = action(kind, turn_id=turn_id, dependencies=list(dependencies))
        trace = row[kind] = {"rid": work["rid"], "body_sha256": digest(body), "started_ns": now()}
        version = await ready(work)
        trace.update(version_id=version, http_started_ns=now())
        response = await http.post("/v1/chat/completions", json=body, headers={"X-Agentic-Session": b.id,
            "X-Agentic-Branch": b.id, "X-Agentic-Version": version, "X-Agentic-Request": work["rid"]})
        trace.update(http_finished_ns=now(), http=response.status_code, response=response.json())
        response.raise_for_status()
        text = content(trace["response"])
        ack = await event("commit", {"action_id": work["action_id"], "turn_end": kind == "model",
                                     "turn_id": turn_id})
        trace.update(committed_ns=now(), content=text)
        if ack["status"] != "committed":
            raise RuntimeError(f"{kind} was not committed: {ack}")
        return text, work["action_id"]

    proposal = None

    async def summarize():
        nonlocal proposal
        text, summary_id = await generate("summary", summary_prompt(source))
        result = compacted(text)
        if arm == "early":
            elapsed = math.ceil(ms(row["wait_started_ns"], now()) / ELAPSED_STEP_MS) * ELAPSED_STEP_MS
            if "tool_return_ns" in row or elapsed > scenario.nominal_ms:
                row["prepare_outcome"] = "summary_overran_tool_window"
            else:
                body = chat(result, tail)
                plan = action("prepare", turn_id=turn_id, chat=body, boundary_message_ids=boundary_ids,
                              target_message_ids=[digest(result), digest(tail)],
                              dependencies=[summary_id], wait_type=wait_type(scenario.nominal_ms, elapsed))
                proposal = row["prepare"] = {"rid": plan["rid"], "action_id": plan["action_id"],
                    "body_sha256": digest(body), "proposed_ns": now(), "elapsed_bucket_ms": elapsed}
                proposal["ack"] = await event("ready", plan)
        return result, summary_id

    try:
        row["source_version"] = (await event("append", {"isolation_domain": b.id,
                                                "message_ids": original_ids}))["version_id"]
        row["started_ns"] = now()
        tool = action("tool", turn_id=turn_id, wait_type=f"e3_tool_{scenario.nominal_ms}ms")
        await ready(tool)
        if (await event("start", {"action_id": tool["action_id"]}))["status"] != "running":
            raise RuntimeError("confirmed tool did not start")
        row["wait_started_ns"] = now()

        async def return_tool():
            if tool_runner is None:
                row["tool_return_ns"] = await tool_wait(scenario.actual_ms)
            else:
                row["tool_observations"] = await tool_runner()
                row["tool_return_ns"] = now()
            await event("finish", {"action_id": tool["action_id"], "status": "completed",
                                   "wait_ms": ms(row["wait_started_ns"], row["tool_return_ns"])})
            if (await event("commit", {"action_id": tool["action_id"]}))["status"] != "committed":
                raise RuntimeError("tool barrier did not commit")

        timer = asyncio.create_task(return_tool())
        tasks.append(timer)
        if arm == "early":
            summary = asyncio.create_task(summarize())
            tasks.append(summary)
        await timer
        replacement, summary_id = await summary if arm == "early" else await summarize()

        # One status read at the safe call boundary; never await optional prefill completion.
        candidate_id = None
        if proposal is not None:
            proposal["lookup_started_ns"] = now()
            status = await b.lookup(proposal["ack"]["event_id"])
            proposal.update(at_cut=status, checked_ns=now())
            if status["status"] == "completed":
                candidate_id = proposal["action_id"]
                row["prepare_outcome"] = "ready_at_cut"
            else:
                row["prepare_outcome"] = f"fallback_{status['status']}"
        if scenario.changed:
            # A revision of the retained prefix becomes known only now, after the
            # tool fence. This is a real replace and invalidates the old proposal.
            # It never changes A's facts or requires a second summary.
            tail = retained(True)
            row["drift_ack"] = await event("replace", {"message_ids": boundary_ids + [digest(tail)]})
            if candidate_id is not None:
                row["prepare_outcome"] = "invalidated_ready_prefix"
            candidate_id = None
        final = chat(replacement, tail)
        row["cut_ns"] = now()
        cut = await event("replace", {"message_ids": [digest(replacement), digest(tail)],
                                     **({"candidate_action_id": candidate_id} if candidate_id else {})})
        row.update(cut_version=cut["version_id"], used_candidate=candidate_id is not None)
        if cut["version_id"] in (row["source_version"], row.get("drift_ack", {}).get("version_id")):
            raise RuntimeError("compaction CAS did not advance the committed version")
        await generate("model", final, (tool["action_id"], summary_id))
        if proposal is not None:
            proposal["after_model"] = await b.lookup(proposal["ack"]["event_id"])
        row["foreground_ms"] = ms(row["tool_return_ns"], row["model"]["committed_ns"])
        row["total_ms"] = ms(row["started_ns"], row["model"]["committed_ns"])
        row["summary_ms"] = ms(row["summary"]["started_ns"], row["summary"]["committed_ns"])
        row["summary_exposed_ms"] = ms(max(row["tool_return_ns"], row["summary"]["started_ns"]),
                                      max(row["tool_return_ns"], row["summary"]["committed_ns"]))
        row["generation_ms"] = ms(row["model"]["started_ns"], row["model"]["committed_ns"])
        row["wait_ms"] = ms(row["wait_started_ns"], row["tool_return_ns"])
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if b.version is not None:
            await event("end", {})


async def measure(root: Path, name: str, token: str, rows: list[dict], warmups: int) -> None:
    async with (httpx.AsyncClient(base_url=INGRESS, headers={"Authorization": f"Bearer {token}"},
                                trust_env=False, timeout=180) as client,
                httpx.AsyncClient(base_url=ENGINE, trust_env=False, timeout=180) as native):
        with (root / "logs" / f"{name}-pairs.jsonl").open("x", encoding="utf-8", buffering=1) as log:
            # Warm each code path in separate scopes; do not use warmup samples in the estimates.
            longest = max((SCENARIOS[p["scenario"]].nominal_ms for p in rows))
            warmup = [{"scenario": f"stable-{longest}", "pair": i, "city": CITIES[i % len(CITIES)],
                       "order": ARMS, "arms": {}, "warmup": True} for i in range(warmups)]
            for planned in warmup + rows:
                for arm in planned["order"]:
                    row = {"status": "error"}
                    planned["arms"][arm] = row
                    try:
                        await reset_cache(native)
                        await run_arm(client, SCENARIOS[planned["scenario"]], arm, planned["city"], row)
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    else:
                        row["status"] = "recorded"
                    finally:
                        emit(log, {"phase": "arm", "scenario": planned["scenario"], "pair": planned["pair"],
                                   "city": planned["city"], "warmup": planned["warmup"], "arm": arm, **row})
                print(name, planned["scenario"], planned["pair"],
                      {k: (v["status"], v.get("prepare_outcome"), v.get("total_ms"))
                       for k, v in planned["arms"].items()}, flush=True)


def engine_evidence(path: Path) -> tuple[dict, dict]:
    receipts, policies = defaultdict(list), defaultdict(list)
    with path.open(encoding="utf-8") as log:
        for line in log:
            if "agentic_layer:engine receipt " in line:
                data = json.loads(line.split("agentic_layer:engine receipt ", 1)[1])
                receipts[data["action_id"]].append(data)
            elif "agentic_layer:policy " in line:
                data = json.loads(line.split("agentic_layer:policy ", 1)[1])
                policies[data["action_id"]].append(data)
    return receipts, policies


def verify_pairs(rows: list[dict], receipts: dict, policies: dict) -> None:
    for pair in rows:
        scenario = SCENARIOS[pair["scenario"]]
        city = pair["city"]
        for arm, row in pair["arms"].items():
            for kind in ("summary", "prepare", "model"):
                if kind in row:
                    rid = row[kind]["rid"]
                    row[kind]["native_receipts"] = receipts.get(rid, [])
                    row[kind]["policy_decisions"] = policies.get(rid, [])
            if row["status"] != "recorded":
                continue
            errors = []
            summary, final = row["summary"], row["model"]
            candidates = row.get("prepare", {}).get("native_receipts", [])
            ready = next((r for r in candidates if r["status"] == "ready"), None)
            expect = city.upper() if scenario.changed else city
            if summary["content"].strip() != city or final["content"].strip() != expect:
                errors.append("summary or final answer differs from the fixture fact")
            for work, version in ((summary, row["source_version"]), (final, row["cut_version"])):
                if not work["native_receipts"] or any(
                    ack["version_id"] != version or ack["run_id"] != row["run_id"]
                    for ack in work["native_receipts"]
                ):
                    errors.append(f"missing or wrong-version native ACK for {work['rid']}")
                if work["response"]["choices"][0]["finish_reason"] != "stop":
                    errors.append(f"native completion did not end normally for {work['rid']}")
            final_ack = final["native_receipts"][-1] if final["native_receipts"] else None
            row["final_hit_tokens"] = final_ack["matched_tokens"] if final_ack else None
            row["prepare_checkpoint_tokens"] = ready["details"]["checkpoint_tokens"] if ready else None
            if row["used_candidate"] or row["prepare_outcome"] == "invalidated_ready_prefix":
                if not ready or not ready["handle"]:
                    errors.append("controller claimed completed preparation without a native ready/lease ACK")
            if row["used_candidate"]:
                expected_hit = final["response"]["usage"]["prompt_tokens"] - 1
                if (row["final_hit_tokens"] != expected_hit or row["prepare_checkpoint_tokens"] != expected_hit
                        or row["prepare"]["body_sha256"] != final["body_sha256"]):
                    errors.append("cutover did not reuse the exact native checkpoint of the final prompt")
            elif row["final_hit_tokens"] != 0:
                errors.append("cold or invalidated target reused cached tokens")
            if "prepare" in row and row["prepare"]["after_model"]["status"] not in TERMINAL:
                errors.append("candidate was not terminal after foreground fallback")
            if errors:
                row.update(status="error", error="; ".join(errors))
            else:
                row["status"] = "ok"
        a, b = (pair["arms"].get(arm, {}) for arm in ARMS)
        if not a or not b:
            pair["status"] = "missing"
        elif a["status"] != "ok" or b["status"] != "ok":
            pair["status"] = "error"
        elif any(a[kind][field] != b[kind][field] for kind in ("summary", "model")
                 for field in ("content", "body_sha256")):
            pair.update(status="error", error="arms differ in raw generated summary/final answer or native chat body")
        else:
            pair["status"] = "ok"


def percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    ranks = {f"p{p}": ordered[math.ceil(p * len(ordered) / 100) - 1] for p in (50, 95, 99)}
    return {key: value if math.isfinite(value) else "+inf" for key, value in ranks.items()}


def paired_effect(pairs: list[dict], key: str, all_correct: bool) -> dict | None:
    gains = [100 * (a[key] - b[key]) / a[key] for pair in pairs
             for a, b in [(pair["arms"]["foreground"], pair["arms"]["early"])]]
    if not gains:
        return None
    rng = random.Random(73121)
    samples = sorted(statistics.median(rng.choices(gains, k=len(gains))) for _ in range(10000))
    low, high = samples[249], samples[9749]
    return {"success_pairs": len(gains), "median_paired_reduction_pct": statistics.median(gains),
            "bootstrap_95_pct": [low, high], "credible_over_5_pct": all_correct and len(gains) >= 10 and low > 5}


def aggregate(rows: list[dict], scenarios: list[str]) -> dict:
    result = {}
    for key in scenarios:
        planned = [p for p in rows if p["scenario"] == key]
        good = [p for p in planned if p["status"] == "ok"]
        early = [p["arms"]["early"] for p in good]
        result[key] = {"planned_pairs": len(planned), "verified_pairs": len(good),
            "pair_statuses": dict(Counter(p["status"] for p in planned)),
            "early_outcomes": dict(Counter(p["arms"].get("early", {}).get("prepare_outcome") or "missing"
                                           for p in planned)),
            "verified_reuses": sum(r["used_candidate"] for r in early),
            "verified_ready_invalidations": sum(r["prepare_outcome"] == "invalidated_ready_prefix" for r in early),
            "summary_generations_observed": {arm: sum(
                "response" in p["arms"].get(arm, {}).get("summary", {}) for p in planned) for arm in ARMS},
            "arm_statuses": {arm: dict(Counter(p["arms"].get(arm, {}).get("status", "missing")
                                                for p in planned)) for arm in ARMS},
            "latencies_ms": {arm: {metric: {
                "all_planned": percentiles([
                    p["arms"][arm][metric] if p["arms"].get(arm, {}).get("status") == "ok" else float("inf")
                    for p in planned]),
                "verified_pairs_only": percentiles([p["arms"][arm][metric] for p in good])}
                for metric in METRICS} for arm in ARMS},
            "paired_effects": {metric: paired_effect(good, metric, len(good) == len(planned))
                               for metric in ("foreground_ms", "total_ms")}}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--policy", choices=("lookahead-only", "strong"), default="lookahead-only")
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS))
    args = parser.parse_args()
    if args.pairs < 1 or args.warmups < 0 or len(set(args.scenarios)) != len(args.scenarios):
        parser.error("positive pairs, nonnegative warmups, and distinct scenarios required")
    if (os.getenv("AGENTIC_FIXED_GRAPH"), os.getenv("AGENTIC_DECODE_GRAPH")) != ("1", "full"):
        parser.error("E3 requires AGENTIC_FIXED_GRAPH=1 AGENTIC_DECODE_GRAPH=full")
    root = args.run_dir.resolve()
    output = root / f"{args.name}.json"
    if output.exists():
        parser.error(f"output already exists: {output}")
    rows = [{"scenario": key, "pair": i, "city": CITIES[i % len(CITIES)], "arms": {}, "warmup": False,
             "order": ARMS if (i + index) % 2 == 0 else ARMS[::-1], "status": "missing"}
            for i in range(args.pairs) for index, key in enumerate(args.scenarios)]
    report = {"kind": "controlled GPU E3, not DSH", "engine_arm": "extended", "controller_policy": args.policy,
              "fixture": "Opaque fixed tool fence; one required native extractive summary and final chat per arm. "
                         "A||B starts as a logical committed view, not a primed GPU cache. A is frozen at tool start; "
                         "B is already known. The changed-prefix case revises B after the fence, leaving A unchanged. "
                         "Do not generalize the city-label check to open-ended summarization quality.",
              "timing": "total: before tool admission to final commit; foreground: actual tool return to final commit. "
                        "Running summaries are joined once; unready optional prep is superseded at the cut. "
                        "Ready/cold ACKs and any native late work remain in the raw logs.",
              "scenarios": {k: {"nominal_wait_ms": SCENARIOS[k].nominal_ms, "actual_wait_ms": SCENARIOS[k].actual_ms,
                                "changed_prefix": SCENARIOS[k].changed} for k in args.scenarios},
              "logs": {"engine": f"logs/{args.name}.log", "graph": f"logs/{args.name}-graph.json",
                       "ingress": f"logs/{args.name}-ingress.log", "pairs": f"logs/{args.name}-pairs.jsonl",
                       "calibration": f"logs/{args.name}-calibration.jsonl"},
              "runs": rows}
    try:
        with engine(root, "extended", args.name):
            calibration = asyncio.run(calibrate(root, args.name, args.scenarios))
            report.update(calibration=calibration.name, calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest())
            with ingress(root, args.name, args.policy, calibration) as token:
                asyncio.run(measure(root, args.name, token, rows, args.warmups))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    try:
        path = root / report["logs"]["ingress"]
        evidence = engine_evidence(path) if path.exists() else ({}, {})
        verify_pairs(rows, *evidence)
    except Exception as exc:
        report["evidence_error"] = f"{type(exc).__name__}: {exc}"
    report["summary"] = aggregate(rows, args.scenarios)
    failed = "error" in report or "evidence_error" in report or any(r["status"] != "ok" for r in rows)
    report["status"] = "error" if failed else "ok"
    write_json(output, report)
    print(output, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
