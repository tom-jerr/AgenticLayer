"""Measure isolated native and Triton HiCache copies on physical hybrid checkpoints.

Run with the pinned environment, for example:
    PYTHONPATH=agenticlayer RUN/env/python/bin/python -m experiments.copy_calibrate --run-dir RUN --name copy-crossover

Each arm owns a fresh engine. Transfer timing is the scheduler's CUDA-event
measurement at ACK; wall minus event time is a proxy including queueing and
other control/completion overhead, not a measured queue interval.
The per-pool kernel counters are submitted bytes, associated with an ACK by
isolating each transfer; only the aggregate byte count comes from the ACK.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import os
import statistics
from pathlib import Path

import httpx

from .gpu_run import ENGINE, engine, token_fixture, write_json
from .hybrid_gpu import Driver, ident

LENGTHS = (512, 4096, 16384)
WARMUPS = 5
SAMPLES = 20
HOST_LIMIT = 16 * 1024**3
MODE_BY_ARM = {"native": "0", "triton": "1", "d2h": "d2h"}


def percentile95(values: list[float]) -> float:
    """Nearest-rank p95, including for the 20-sample calibration buckets."""
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def counters(snapshot: dict) -> dict[str, int]:
    return snapshot["details"]["transfer_counters"]


def sample_transfer(direction: str, reply: dict, wall_ms: float,
                    before: dict[str, int], after: dict[str, int], triton: bool) -> dict:
    if not reply["details"].get("transfer_metrics_complete"):
        raise RuntimeError(f"{direction} ACK lacks complete physical metrics: {reply}")
    if reply["transfer_bytes"] <= 0 or reply["transfer_us"] <= 0:
        raise RuntimeError(f"{direction} ACK did not measure a physical transfer: {reply}")
    delta = {key: after[key] - before[key] for key in before if key.startswith(direction + "_")}
    if any(value < 0 for value in delta.values()):
        raise RuntimeError(f"{direction} counters went backwards: {delta}")
    kind = "triton" if triton else "native"
    for pool in ("full", "mamba"):
        if delta[f"{direction}_{pool}_{kind}_bytes"] <= 0:
            raise RuntimeError(f"{direction} missing {pool} {kind} copy: {delta}")
    if not triton and any(delta[f"{direction}_{pool}_triton_bytes"] for pool in ("full", "mamba")):
        raise RuntimeError(f"native arm unexpectedly used Triton: {delta}")
    physical_ms = reply["transfer_us"] / 1000
    return {
        "ack_status": reply["status"],
        "ack_metrics_complete": True,
        "ack_bytes": reply["transfer_bytes"],
        "physical_ms": physical_ms,
        "wall_ms": wall_ms,
        "queue_and_overhead_estimate_ms": max(0.0, wall_ms - physical_ms),
        "wall_minus_physical_ms": wall_ms - physical_ms,
        "submitted_bytes_by_pool_and_kernel": delta,
        "submitted_total_bytes": sum(delta.values()),
    }


async def measure_arm(root: Path, name: str, copy_mode: str) -> dict:
    from agentic_layer.engine_client import EngineClient

    control = EngineClient(root / "engine.sock")
    await control.handshake()
    if not all(control.capabilities[x] for x in ("prepare", "offload", "restore", "reclaim")):
        raise RuntimeError(f"missing scheduler capabilities: {control.capabilities}")
    rows = []
    with (root / "logs" / f"{name}-wire.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        async with httpx.AsyncClient(base_url=ENGINE, trust_env=False, timeout=180) as http:
            driver = Driver(control, http, log, name)
            driver.record("handshake", control.snapshot)
            for repetition in range(WARMUPS + SAMPLES):
                for length in LENGTHS:
                    seed = 42 + repetition
                    ids = token_fixture(length, seed)
                    version = ident()
                    salt = hashlib.sha256(ident().encode()).hexdigest()
                    await control.poll()
                    baseline = control.snapshot
                    if baseline["unique_host_bytes"] > HOST_LIMIT:
                        raise RuntimeError(f"host ownership exceeds 16 GiB before sample: {baseline}")
                    _, prepared, prefill_ms = await driver.generate(ids, salt, version, prepare=True)
                    if prepared["status"] != "ready" or not prepared["handle"]:
                        raise RuntimeError(f"checkpoint not ready: {prepared}")
                    handle = prepared["handle"]
                    released, = await driver.rpc(driver.command("release", version=version, handle=handle))
                    if released["status"] != "ok":
                        raise RuntimeError(f"failed to release prepare lease: {released}")

                    await control.poll()
                    before = counters(control.snapshot).copy()
                    off, off_wall = await driver.transfer("offload", version, handle, "completed")
                    if not off["details"].get("demoted") or off["details"].get("residency") != "host":
                        raise RuntimeError(f"offload did not demote checkpoint: {off}")
                    if off["unique_host_bytes"] > HOST_LIMIT:
                        raise RuntimeError(f"host ownership exceeds 16 GiB after offload: {off}")
                    after_off = counters(control.snapshot).copy()
                    d2h = sample_transfer("d2h", off, off_wall, before, after_off, copy_mode != "0")
                    on, on_wall = await driver.transfer("restore", version, handle, "ready")
                    if on["matched_tokens"] != length or on["details"].get("residency") != "hbm":
                        raise RuntimeError(f"restore did not materialize checkpoint: {on}")
                    after_on = counters(control.snapshot).copy()
                    h2d = sample_transfer("h2d", on, on_wall, after_off, after_on, copy_mode == "1")

                    device, = await driver.rpc(driver.command("reclaim", version=version, handle=handle))
                    if device["status"] != "completed" or device["details"].get("residency") != "host":
                        raise RuntimeError(f"could not reclaim restored device checkpoint: {device}")
                    host, = await driver.rpc(driver.command("reclaim", version=version, handle=handle,
                                                            payload={"residency": "host"}))
                    if host["status"] != "completed" or host["details"].get("residency") != "absent":
                        raise RuntimeError(f"could not reclaim host checkpoint: {host}")
                    await control.poll()
                    cleaned = control.snapshot
                    for key in ("unique_hbm_bytes", "unique_host_bytes"):
                        if cleaned[key] > baseline[key]:
                            raise RuntimeError(f"sample retained {key}: before={baseline[key]} after={cleaned[key]}")
                    row = {"length": length, "seed": seed, "repetition": repetition,
                           "warmup": repetition < WARMUPS, "prefill_ms": prefill_ms,
                           "d2h": d2h, "h2d": h2d,
                           "cleanup": {"device": device["details"], "host": host["details"],
                                       "before_hbm_bytes": baseline["unique_hbm_bytes"],
                                       "after_hbm_bytes": cleaned["unique_hbm_bytes"],
                                       "before_host_bytes": baseline["unique_host_bytes"],
                                       "after_host_bytes": cleaned["unique_host_bytes"]}}
                    driver.record("copy-calibration", row)
                    rows.append(row)
            driver.record("final-snapshot", control.snapshot)
    return {"arm": name, "copy_mode": copy_mode, "samples": rows}


def summarize(arms: dict[str, dict]) -> list[dict]:
    result = []
    for length in LENGTHS:
        for direction in ("d2h", "h2d"):
            metrics = {}
            for arm, data in arms.items():
                rows = [row[direction] for row in data["samples"] if row["length"] == length and not row["warmup"]]
                if len(rows) != SAMPLES:
                    raise RuntimeError(f"incomplete {arm} {length} {direction} bucket")
                metrics[arm] = {}
                for key in ("physical_ms", "wall_ms", "queue_and_overhead_estimate_ms"):
                    values = [row[key] for row in rows]
                    metrics[arm][key] = {"median": statistics.median(values), "p95": percentile95(values)}
            winner = min(metrics, key=lambda arm: (metrics[arm]["physical_ms"]["median"], arm))
            result.append({"length": length, "direction": direction, "measurements": metrics,
                           "lower_physical_median": winner})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--name", required=True, help="unique output and engine log prefix")
    parser.add_argument("--arms", nargs="+", choices=MODE_BY_ARM, default=list(MODE_BY_ARM))
    args = parser.parse_args()
    if len(args.arms) != len(set(args.arms)):
        parser.error("duplicate arm")
    root = args.run_dir.resolve()
    arms = {}
    previous = os.environ.get("AGENTIC_TRITON_COPY")
    try:
        for kind in args.arms:
            mode = MODE_BY_ARM[kind]
            os.environ["AGENTIC_TRITON_COPY"] = mode
            name = f"{args.name}-{kind}"
            with engine(root, "extended-hicache", name):
                arms[kind] = asyncio.run(measure_arm(root, name, mode))
    finally:
        if previous is None:
            os.environ.pop("AGENTIC_TRITON_COPY", None)
        else:
            os.environ["AGENTIC_TRITON_COPY"] = previous
    report = {"warmups_per_shape": WARMUPS, "samples_per_shape": SAMPLES,
              "seed_sequence": [42 + i for i in range(WARMUPS + SAMPLES)],
              "queue_estimate_note": "max(0, end-to-end command/ACK wall time minus CUDA-event transfer time); includes RPC, polling and completion overhead",
              "pool_bytes_note": "per-pool kernel counters are submitted bytes observed across an isolated ACK; ACK bytes are aggregate",
              "directions": {"native": ["native", "native"], "triton": ["triton", "triton"], "d2h": ["triton", "native"]},
              "summary": summarize(arms), "arms": arms}
    write_json(root / f"{args.name}.json", report)
    for row in report["summary"]:
        metrics = " ".join(f"{arm} {m['physical_ms']['median']:.3f}/{m['physical_ms']['p95']:.3f} ms"
                           for arm, m in row["measurements"].items())
        print(f"{row['length']:5d} {row['direction'].upper()} median/p95 {metrics}; "
              f"lower median: {row['lower_physical_median']}", flush=True)
    print(f"Report: {root / f'{args.name}.json'}", flush=True)


if __name__ == "__main__":
    main()
