"""Compare migrated KV against a warm segmentation control and cold numeric gate."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

import httpx

from .gpu_run import ENGINE, LENGTHS, SEEDS, engine, token_fixture, write_json
from .hybrid_gpu import Driver, ident


async def warm_case(gpu, prompt, target, *, migrate):
    version, namespace = ident(), hashlib.sha256(ident().encode()).hexdigest()
    _, prepared, _ = await gpu.generate(prompt[:-1], namespace, version, prepare=True)
    handle = prepared["handle"]
    assert prepared["status"] == "ready" and handle, prepared
    released, = await gpu.rpc(gpu.command("release", version=version, handle=handle))
    assert released["status"] == "ok", released
    if migrate:
        off, _ = await gpu.transfer("offload", version, handle, "completed")
        assert off["details"]["demoted"] and off["transfer_bytes"] > 0, off
        on, _ = await gpu.transfer("restore", version, handle, "ready")
        assert on["matched_tokens"] == len(prompt) - 1 and on["transfer_bytes"] > 0, on
    scored, hit, _ = await gpu.generate(prompt + target, namespace, ident(), n=0, logprob=len(prompt)-1)
    scores = [(row[0], row[1]) for row in scored["meta_info"]["input_token_logprobs"] if row[0] is not None]
    assert [token for _, token in scores] == target and hit["matched_tokens"] == len(prompt)-1, (scores, hit)
    greedy, _, _ = await gpu.generate(prompt, namespace, ident(), n=8, logprob=-1)
    return scores, greedy["output_ids"], hit["matched_tokens"]


async def verify(root, name, reference):
    from agentic_layer.engine_client import EngineClient

    cold = json.loads(reference.read_text())["measurements"]
    ctl = EngineClient(root / "engine.sock")
    await ctl.handshake()
    rows = []
    with (root / "logs" / f"{name}-wire.jsonl").open("x", buffering=1) as log:
        async with httpx.AsyncClient(base_url=ENGINE, trust_env=False, timeout=180) as http:
            gpu = Driver(ctl, http, log, name)
            gpu.record("handshake", ctl.snapshot)
            for n in LENGTHS:
                for seed in SEEDS:
                    baseline = cold[f"{n}/{seed}"]
                    target = baseline["forced_ids"]
                    prompt = token_fixture(n, seed)
                    warm, warm_ids, _ = await warm_case(gpu, prompt, target, migrate=False)
                    moved, moved_ids, hit = await warm_case(gpu, prompt, target, migrate=True)
                    envelope, deviations, warm_deviations, migration_deltas, stable, greedy_allowed = [], [], [], [], [], []
                    warm_greedy_allowed = []
                    for i, ((score, _), (control, _)) in enumerate(zip(moved, warm, strict=True)):
                        observed = [x[i] for x in baseline["forced_logprob"]]
                        allowed = max(1e-4, 2*(max(observed)-min(observed)))
                        deviations.append(max(abs(score-x) for x in observed))
                        warm_deviations.append(max(abs(control-x) for x in observed))
                        migration_deltas.append(abs(score-control))
                        envelope.append(allowed)
                        ids = {r[i] for r in baseline["greedy"]}
                        stable.append(len(ids) == 1)
                        greedy_allowed.append(moved_ids[i] in ids)
                        warm_greedy_allowed.append(warm_ids[i] in ids)
                    row = {"length": n, "seed": seed, "matched_tokens": hit,
                           "greedy_ids": moved_ids, "warm_control_greedy_ids": warm_ids, "position_stable": stable,
                           "teacher_ids": target, "max_cold_delta_per_pos": deviations, "envelope_per_pos": envelope,
                           "max_warm_cold_delta_per_pos": warm_deviations,
                           "warm_cold_gate_pass": all(warm_greedy_allowed) and
                               all(d <= e for d, e in zip(warm_deviations, envelope, strict=True)),
                           "migration_delta_per_pos": migration_deltas,
                           "cold_gate_pass": all(greedy_allowed) and
                               all(d <= e for d, e in zip(deviations, envelope, strict=True)),
                           "warm_migration_gate_pass": moved_ids == warm_ids and max(migration_deltas) <= 1e-4}
                    gpu.record("numeric-gate", row)
                    rows.append(row)
                    print(n, seed, {"cold_gate": row["cold_gate_pass"], "warm_cold_gate": row["warm_cold_gate_pass"],
                                    "warm_migration_gate": row["warm_migration_gate_pass"],
                                    "max_cold_delta": max(deviations), "max_migration_delta": max(migration_deltas)}, flush=True)
    return {"cold_reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(), "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--triton", action="store_true")
    a = p.parse_args()
    reference = a.reference or a.run_dir / "reference-cold-forced.json"
    os.environ["AGENTIC_TRITON_COPY"] = str(int(a.triton))
    with engine(a.run_dir, "extended-hicache", a.name):
        result = asyncio.run(verify(a.run_dir, a.name, reference))
    write_json(a.run_dir / f"{a.name}.json", {"triton_copy": a.triton, **result})
    if not all(r["cold_gate_pass"] and r["warm_migration_gate_pass"] for r in result["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
