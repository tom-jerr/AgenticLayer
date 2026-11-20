"""Isolate explicit lifecycle reclaim and host placement on the pinned GPU engine.

HBM bytes are owned Full KV and recurrent allocator slots, not the CUDA pool's
preallocated footprint. The two scopes have separate engines. Within a scope,
conditions are isolated by native cache resets and execution order is rotated.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import statistics
import time
from pathlib import Path

import httpx

from .compaction_gpu import paired_effect, percentiles
from .gpu_run import ENGINE, engine, token_fixture, write_json
from .hybrid_gpu import Driver, ident

LENGTHS = (4096, 16384)
GREEDY_TOKENS = 8
CONDITIONS = {"lifecycle": ("reclaim", "retain"),
              "placement": ("resident", "restored", "recompute"),
              "prefetch": ("demand", "prefetch")}
ARMS = {"lifecycle": "extended", "placement": "extended-hicache", "prefetch": "extended-hicache"}


def require(condition: bool, message: str, evidence) -> None:
    if not condition:
        raise RuntimeError(f"{message}: {evidence}")


def allocation(reply: dict) -> dict[str, int]:
    return {"hbm_bytes": reply["unique_hbm_bytes"], "host_bytes": reply["unique_host_bytes"],
            "full_slots": reply["full_kv_slots"], "recurrent_slots": reply["recurrent_slots"]}


def delta(after: dict, before: dict) -> dict:
    return {key: after[key] - value for key, value in before.items()}


async def snapshot(d: Driver, phase: str) -> dict:
    await d.control.poll()
    state = d.control.snapshot
    d.record("snapshot", {"phase": phase, "receipt": state})
    require(not any(state[key] for key in ("fg_queue", "bg_queue", "reserved_bytes")),
            f"scheduler not idle at {phase}", state)
    return state


async def reset_cache(d: Driver) -> dict:
    # Only the placement prepare path takes a persistent lease, and it must
    # receive its release ACK before returning. Failed cases stop this engine
    # instead of flushing a potentially live handle or transfer.
    response = await d.http.post("/flush_cache", params={"timeout": 5})
    d.record("native-cache-reset", {"status": response.status_code, "response": response.text})
    response.raise_for_status()
    require(response.text.startswith("Cache flushed."), "cache reset not confirmed", response.text)
    return await snapshot(d, "after-cache-reset")


def validate_native(result: dict, receipt: dict, prompt_tokens: int, output_tokens: int, hit: int) -> None:
    meta = result["meta_info"]
    expected = (prompt_tokens, output_tokens, output_tokens, hit, hit, hit)
    observed = (meta["prompt_tokens"], meta["completion_tokens"], len(result["output_ids"]),
                meta["cached_tokens"], receipt["matched_tokens"], receipt["details"]["full_kv_hit_length"])
    require(observed == expected, "native token counts or scheduler match differ", (expected, observed))
    require((meta.get("finish_reason") or {}).get("type") == "length" and not receipt["error"],
            "native request did not finish normally", (meta, receipt["error"]))


async def generate(d: Driver, ids: list[int], namespace: str, version: str, *, n: int, hit: int) -> tuple[dict, dict]:
    result, receipt, wall_ms = await d.generate(ids, namespace, version, n=n, logprob=-1 if n > 1 else None)
    validate_native(result, receipt, len(ids), n, hit)
    meta = result["meta_info"]
    timing = {"forward_wall_ms": wall_ms, "forward_e2e_ms": 1000 * meta["e2e_latency"],
              "forward_first_token_ms": 1000 * meta["first_token_latency"]}
    require(all(math.isfinite(value) and value > 0 for value in timing.values()), "invalid forward timings", timing)
    return {"rid": receipt["rid"], "receipt_seq": receipt["seq"], "prompt_tokens": len(ids),
            "cached_tokens": hit, "output_ids": result["output_ids"], **timing}, receipt


def checkpoint(receipt: dict, length: int) -> str:
    details = receipt["details"]
    require(receipt["status"] == "ready" and receipt["handle"] and details["residency"] == "hbm"
            and details["checkpoint_tokens"] == length,
            "finished request lacks a complete HBM checkpoint", (receipt["status"], receipt["handle"], length))
    return receipt["handle"]


async def lifecycle(d: Driver, condition: str, length: int, clean: dict) -> dict:
    # Each branch generates one real token: the finished leaf is exactly the
    # prompt, since that emitted token has not passed through the model yet.
    branches, commands = [], []
    for seed in (1701, 1702):
        version = ident()
        run, receipt = await generate(d, token_fixture(length, seed), ident() + ident(), version, n=1, hit=0)
        handle = checkpoint(receipt, length)
        branches.append({**run, "handle": handle})
        commands.append(d.command("reclaim", version=version, handle=handle))
    baseline = allocation(clean)
    finished = allocation(await snapshot(d, "finished-branches"))
    owned = delta(finished, baseline)
    require(owned["full_slots"] == 2 * length and owned["recurrent_slots"] == 2
            and owned["hbm_bytes"] > 0 and owned["host_bytes"] == 0,
            "finished branches do not own two independent hybrid checkpoints", owned)

    receipts = []
    reclaim_ms = 0.0
    if condition == "reclaim":
        start = time.monotonic_ns()
        replies = await d.rpc(*commands)
        reclaim_ms = (time.monotonic_ns() - start) / 1e6
        previous = finished
        for command, reply in zip(commands, replies, strict=True):
            details = reply["details"]
            freed = delta(previous, allocation(reply))
            require(reply["status"] == "completed" and reply["handle"] == command["handle"]
                    and details["residency"] == "absent" and not reply["error"]
                    and freed["full_slots"] == details["freed_full_slots"] == length
                    and freed["recurrent_slots"] == details["freed_recurrent_slots"] == 1
                    and freed["hbm_bytes"] > 0 and not freed["host_bytes"],
                    "reclaim did not physically free the finished hybrid leaf", (reply, freed))
            receipts.append({"action_id": reply["action_id"], "seq": reply["seq"], **freed})
            previous = allocation(reply)

    before = allocation(await snapshot(d, "before-unrelated-forward"))
    require(before == (baseline if condition == "reclaim" else finished),
            "unexpected ownership at the next forward boundary", (before, baseline, finished))
    forward, _ = await generate(d, token_fixture(length, 1703), ident() + ident(), ident(), n=1, hit=0)
    after = allocation(await snapshot(d, "after-unrelated-forward"))
    return {"branches": branches, "reclaim_receipts": receipts, "baseline": baseline,
            "finished_branches": finished, "before_forward": before, "after_forward": after,
            "metrics": {"reclaim_wall_ms": reclaim_ms, "reclaimed_hbm_bytes": finished["hbm_bytes"] - before["hbm_bytes"],
                        "hbm_before_forward_bytes": before["hbm_bytes"] - baseline["hbm_bytes"],
                        "hbm_after_forward_bytes": after["hbm_bytes"] - baseline["hbm_bytes"],
                        "reclaim_plus_forward_wall_ms": reclaim_ms + forward["forward_wall_ms"],
                        **{key: forward[key] for key in ("forward_wall_ms", "forward_e2e_ms", "forward_first_token_ms")}},
            "unrelated_forward": forward}


async def prepare(d: Driver, prefix: list[int], namespace: str, version: str) -> tuple[str, float]:
    result, receipt, wall_ms = await d.generate(prefix, namespace, version, prepare=True)
    handle = checkpoint(receipt, len(prefix))
    # The two ops execute in order on the scheduler. The first must not free
    # anything; the second balances the prepare lease before movement/reset.
    refused, released = await d.rpc(d.command("reclaim", version=version, handle=handle),
                                   d.command("release", version=version, handle=handle))
    require(refused["status"] == "unavailable" and "leased" in (refused["error"] or "")
            and allocation(refused) == allocation(released) == allocation(receipt)
            and released["status"] == "ok" and released["handle"] == handle,
            "leased checkpoint not protected and released", (refused, released))
    validate_native(result, receipt, len(prefix), 0, 0)
    return handle, wall_ms


def counters(state: dict) -> dict[str, int]:
    return state["details"]["transfer_counters"]


def transfer(direction: str, receipt: dict, wall_ms: float, before: dict, after: dict,
             *, prefix: list[int], handle: str, triton: bool) -> dict:
    details = receipt["details"]
    down = direction == "d2h"
    require(receipt["handle"] == handle and receipt["error"] is None
            and details["token_ids"] == prefix and details["full_kv_hit_length"] == len(prefix)
            and details["residency"] == ("host" if down else "hbm") and details["demoted"] is down
            and receipt["matched_tokens"] == (0 if down else len(prefix))
            and details["transfer_metrics_complete"] and receipt["transfer_bytes"] > 0 and receipt["transfer_us"] > 0,
            "incomplete physical transfer ACK", {k: v for k, v in receipt.items() if k != "details"})
    require(allocation(after) == allocation(receipt), "ownership changed after transfer ACK", allocation(after))
    copied = delta(counters(after), counters(before))
    kind, other = ("triton", "native") if triton else ("native", "triton")
    require(all(value >= 0 and (key.startswith(direction + "_") or value == 0) for key, value in copied.items())
            and all(copied[f"{direction}_{pool}_{kind}_bytes"] > 0
                    and copied[f"{direction}_{pool}_{other}_bytes"] == 0 for pool in ("full", "mamba")),
            "transfer missing isolated Full KV and Mamba counter evidence", copied)
    return {"action_id": receipt["action_id"], "seq": receipt["seq"], "wall_ms": wall_ms,
            "cuda_event_ms": receipt["transfer_us"] / 1000, "ack_bytes": receipt["transfer_bytes"],
            "submitted_bytes_by_pool_and_kernel": {k: v for k, v in copied.items() if k.startswith(direction + "_")}}


async def placement(d: Driver, condition: str, length: int, clean: dict, copy_mode: str, wait_ms: int = 0) -> dict:
    # Fixed fixture from the existing hybrid /generate verifier. Both warm
    # conditions prefill exactly the same prefix; all three generate on the
    # same prefix + one uncached token, in independent namespaces.
    ids = token_fixture(length + 1, 42)
    prefix = ids[:-1]
    namespace = ident() + ident()
    baseline = allocation(clean)
    ready, waiting = clean, clean
    prepare_ms = None
    d2h = h2d = None
    wait_started = cut = restore_acked = None
    if condition != "recompute":
        version = ident()
        handle, prepare_ms = await prepare(d, prefix, namespace, version)
        resident = await snapshot(d, "resident-checkpoint")
        resident_owned = delta(allocation(resident), baseline)
        require(resident_owned["full_slots"] == length and resident_owned["recurrent_slots"] == 1
                and resident_owned["hbm_bytes"] > 0 and resident_owned["host_bytes"] == 0
                and counters(resident) == counters(clean), "prepare did not isolate a resident checkpoint", resident_owned)
        ready = waiting = resident
        if condition in ("restored", "demand", "prefetch"):
            off, off_ms = await d.transfer("offload", version, handle, "completed", protect=True)
            waiting = await snapshot(d, "after-d2h")
            d2h = transfer("d2h", off, off_ms, resident, waiting, prefix=prefix, handle=handle, triton=copy_mode != "0")
            demoted = delta(allocation(waiting), baseline)
            require(demoted["hbm_bytes"] == demoted["full_slots"] == demoted["recurrent_slots"] == 0
                    and demoted["host_bytes"] > 0, "offload did not demote both HBM pools", demoted)
            if condition == "restored":
                on, on_ms = await d.transfer("restore", version, handle, "ready")
            else:
                async def restore() -> tuple[dict, float]:
                    nonlocal restore_acked
                    result = await d.transfer("restore", version, handle, "ready")
                    restore_acked = time.monotonic_ns()
                    return result

                wait_started = time.monotonic_ns()
                cut = wait_started + wait_ms * 1_000_000
                if condition == "prefetch":
                    async with asyncio.TaskGroup() as tasks:
                        copy = tasks.create_task(restore())
                        await asyncio.sleep(max(0, (cut - time.monotonic_ns()) / 1e9))
                    on, on_ms = copy.result()
                else:
                    await asyncio.sleep(max(0, (cut - time.monotonic_ns()) / 1e9))
                    on, on_ms = await restore()
            ready = await snapshot(d, "after-h2d")
            h2d = transfer("h2d", on, on_ms, waiting, ready, prefix=prefix, handle=handle, triton=copy_mode == "1")
            require(delta(allocation(ready), baseline) == {**resident_owned, "host_bytes": demoted["host_bytes"]}
                    and d2h["ack_bytes"] == h2d["ack_bytes"], "restore did not recreate the copied checkpoint",
                    (allocation(ready), resident_owned, demoted, d2h, h2d))

    hit = length if condition != "recompute" else 0
    forward, _ = await generate(d, ids, namespace, ident(), n=GREEDY_TOKENS, hit=hit)
    forward_done = time.monotonic_ns()
    end = await snapshot(d, "after-target-forward")
    require(counters(end) == counters(ready), "unexpected copy during target generation", delta(counters(end), counters(ready)))
    forward_wall = forward["forward_wall_ms"]
    return {"baseline": baseline, "waiting": allocation(waiting), "before_forward": allocation(ready),
            "after_forward": allocation(end), "prepare_wall_ms": prepare_ms, "d2h": d2h, "h2d": h2d,
            "target_forward": forward,
            "metrics": {"waiting_hbm_bytes": waiting["unique_hbm_bytes"] - clean["unique_hbm_bytes"],
                        "waiting_host_bytes": waiting["unique_host_bytes"] - clean["unique_host_bytes"],
                        "hbm_before_forward_bytes": ready["unique_hbm_bytes"] - clean["unique_hbm_bytes"],
                        "d2h_bytes": d2h["ack_bytes"] if d2h else 0, "h2d_bytes": h2d["ack_bytes"] if h2d else 0,
                        "restore_plus_forward_wall_ms": (h2d["wall_ms"] if h2d else 0) + forward_wall,
                        "migrate_plus_forward_wall_ms": sum(x["wall_ms"] for x in (d2h, h2d) if x) + forward_wall,
                        **({"tool_wait_to_forward_ms": (forward_done - wait_started) / 1e6,
                            "cut_to_forward_ms": (forward_done - cut) / 1e6,
                            "restore_acked_by_cut": restore_acked <= cut} if wait_started is not None else {}),
                        **({"d2h_wall_ms": d2h["wall_ms"], "d2h_cuda_event_ms": d2h["cuda_event_ms"],
                            "h2d_wall_ms": h2d["wall_ms"], "h2d_cuda_event_ms": h2d["cuda_event_ms"]} if d2h and h2d else {}),
                        **{key: forward[key] for key in ("forward_wall_ms", "forward_e2e_ms", "forward_first_token_ms")}}}


def compare(scope: str, cases: dict) -> None:
    if scope == "lifecycle":
        a, b = cases["reclaim"], cases["retain"]
        require(a["finished_branches"] == b["finished_branches"]
                and [x["output_ids"] for x in a["branches"]] == [x["output_ids"] for x in b["branches"]]
                and a["unrelated_forward"]["output_ids"] == b["unrelated_forward"]["output_ids"]
                and delta(b["after_forward"], a["after_forward"]) == delta(b["before_forward"], a["before_forward"]),
                "lifecycle pair differs in work or retained ownership did not survive the unrelated forward", cases)
    else:
        outputs = {name: row["target_forward"]["output_ids"] for name, row in cases.items()}
        reference = "resident" if scope == "placement" else "demand"
        require(all(ids == outputs[reference] for ids in outputs.values()), f"{scope} greedy outputs differ", outputs)
        if scope == "prefetch":
            for key in ("d2h_bytes", "h2d_bytes"):
                require(cases["demand"]["metrics"][key] == cases["prefetch"]["metrics"][key] > 0,
                        "prefetch changed transfer work", (key, cases))


async def measure(root: Path, name: str, scope: str, rows: list[dict], copy_mode: str, wait_ms: int) -> None:
    from agentic_layer.engine_client import EngineClient

    control = EngineClient(root / "engine.sock")
    with (root / "logs" / f"{name}-wire.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        async with httpx.AsyncClient(base_url=ENGINE, trust_env=False, timeout=180) as http:
            d = Driver(control, http, log, name)
            await control.handshake()
            d.record("handshake", control.snapshot)
            required = ("bind", "reclaim") if scope == "lifecycle" else ("bind", "reclaim", "prepare", "release", "offload", "restore")
            require(all(control.capabilities.get(key) for key in required), "engine capabilities missing", control.capabilities)
            clean = await reset_cache(d)
            baseline = allocation(clean)
            normal_timeout = control.timeout
            for row in rows:
                # The first host-copy call can synchronously initialize its
                # native kernel. Warmup is not a measured sample; keep later
                # trials on the production control deadline.
                control.timeout = 120.0 if scope != "lifecycle" and row["warmup"] else normal_timeout
                conditions = row["order"]
                row["status"] = "error"
                try:
                    for condition in conditions:
                        row["active_condition"] = condition
                        d.record("condition-start", {"length": row["length"], "repetition": row["repetition"],
                                                     "condition": condition, "warmup": row["warmup"]})
                        if scope == "lifecycle":
                            case = await lifecycle(d, condition, row["length"], clean)
                        else:
                            case = await placement(d, condition, row["length"], clean, copy_mode, wait_ms)
                        row["conditions"][condition] = case
                        clean = await reset_cache(d)
                        require(allocation(clean) == baseline, "cache cleanup did not restore baseline", (baseline, allocation(clean)))
                        case["after_cleanup"] = allocation(clean)
                        d.record("condition-complete", case)
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    d.record("pair-error", row)
                    raise
                row.pop("active_condition")
                try:
                    if scope == "placement":
                        cases = row["conditions"]
                        row["resident_restore_greedy_equal"] = (
                            cases["resident"]["target_forward"]["output_ids"]
                            == cases["restored"]["target_forward"]["output_ids"])
                    compare(scope, row["conditions"])
                except RuntimeError as exc:
                    row.update(status="error", error=str(exc))
                else:
                    row["status"] = "ok"
                d.record("paired-sample", row)


def planned_rows(scope: str, lengths: list[int], repetitions: int, warmups: int) -> list[dict]:
    names = CONDITIONS[scope]
    rows = []
    for repetition in range(-warmups, repetitions):
        for index, length in enumerate(lengths):
            rotation = (repetition + index) % len(names)
            rows.append({"length": length, "repetition": repetition, "warmup": repetition < 0,
                         "order": names[rotation:] + names[:rotation], "conditions": {}, "status": "missing"})
    return rows


def summarize(scope: str, lengths: list[int], rows: list[dict]) -> list[dict]:
    output = []
    for length in lengths:
        bucket = [row for row in rows if row["length"] == length and not row["warmup"]]
        verified = [row for row in bucket if row["status"] == "ok"]
        conditions = {}
        for name in CONDITIONS[scope]:
            samples = [row["conditions"][name]["metrics"] for row in verified]
            conditions[name] = {key: statistics.median(s[key] for s in samples) for key in samples[0]} if samples else None
        warm = [row for row in bucket if row.get("resident_restore_greedy_equal")]
        item = {"length": length, "verified_pairs": len(verified), "planned_pairs": len(bucket),
                "errors": sum(r["status"] == "error" for r in bucket),
                "missing": sum(r["status"] == "missing" for r in bucket), "medians": conditions}
        if scope == "placement":
            item["resident_restore_verified_pairs"] = len(warm)
            item["resident_restore_medians"] = {
                name: {key: statistics.median(row["conditions"][name]["metrics"][key] for row in warm)
                       for key in warm[0]["conditions"][name]["metrics"]}
                for name in ("resident", "restored")} if warm else None
        if scope == "prefetch":
            item["ack_by_cut"] = sum(r["conditions"]["prefetch"]["metrics"]["restore_acked_by_cut"] for r in verified)
            item["tool_wait_to_forward_ms"] = {}
            pairs = [{"arms": {"foreground": r["conditions"]["demand"]["metrics"],
                               "early": r["conditions"]["prefetch"]["metrics"]}} for r in verified]
            for arm in CONDITIONS[scope]:
                item["tool_wait_to_forward_ms"][arm] = percentiles([
                    r["conditions"][arm]["metrics"]["tool_wait_to_forward_ms"] for r in verified])
            item["paired_effect"] = paired_effect(pairs, "tool_wait_to_forward_ms", len(verified) == len(bucket))
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--scope", choices=("both", *CONDITIONS), default="both")
    parser.add_argument("--lengths", nargs="+", type=int, choices=LENGTHS, default=list(LENGTHS))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1, help="untimed pairs per length")
    parser.add_argument("--copy-mode", choices=("0", "1", "d2h"), default="0", help="HiCache kernel mode for placement")
    parser.add_argument("--wait-ms", type=int, default=250, help="fixed tool barrier for the prefetch scope")
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0 or args.wait_ms < 0 or len(set(args.lengths)) != len(args.lengths):
        parser.error("need positive repetitions, nonnegative warmups, and distinct lengths")
    root = args.run_dir.resolve()
    output = root / f"{args.name}.json"
    if output.exists():
        parser.error(f"output already exists: {output}")
    report = {"name": args.name, "requested_repetitions": args.repetitions, "warmups": args.warmups,
              "hbm_measurement": "native Full KV and recurrent allocator owned bytes; not process VRAM",
              "wall_measurement": "Driver bind + native HTTP generate + completion; plus metrics sum separate wall intervals",
              "scopes": {}}
    failed = False
    os.environ["AGENTIC_TRITON_COPY"] = args.copy_mode
    for scope in ("lifecycle", "placement") if args.scope == "both" else (args.scope,):
        name = f"{args.name}-{scope}"
        rows = planned_rows(scope, args.lengths, args.repetitions, args.warmups)
        arm = ARMS[scope]
        result = {"engine_arm": arm, "wire": f"logs/{name}-wire.jsonl", "engine_log": f"logs/{name}.log", "rows": rows}
        if scope != "lifecycle":
            result.update(copy_mode=args.copy_mode, greedy_tokens=GREEDY_TOKENS, length_is="prepared prefix tokens")
        if scope == "prefetch":
            result["wait_ms"] = args.wait_ms
        try:
            with engine(root, arm, name):
                asyncio.run(measure(root, name, scope, rows, args.copy_mode, args.wait_ms))
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["summary"] = summarize(scope, args.lengths, rows)
        result["status"] = "error" if "error" in result or any(r["status"] != "ok" for r in rows) else "ok"
        report["scopes"][scope] = result
        failed |= result["status"] != "ok"
    write_json(output, report)
    print(output, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
