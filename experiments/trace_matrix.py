"""Run replay arms on one owned GPU, with the same locked sample in every arm."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import os
import random
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

from .gpu_run import PROJECT, engine, write_json
from .hybrid_gpu import ingress
from .trace_replay import run

DEFAULT_ARMS = ("native-radix", "native-hicache", "strong", "joint")


def percentile(values, quantile):
    return sorted(values)[math.ceil(quantile*len(values))-1]


def paired(reports, count=10000, *, reference="strong", candidate="joint"):
    if not all(reference in arms and candidate in arms for arms in reports):
        return None
    clusters = defaultdict(list)
    for arms in reports:
        before = {tuple(r["identity"]): r for r in arms[reference]["turns"]}
        after = {tuple(r["identity"]): r for r in arms[candidate]["turns"]}
        assert before.keys() == after.keys()
        for key, row in before.items():
            clusters[key[:2]].append((row["wall_ms"], after[key]["wall_ms"]))
    if not clusters:
        return None
    samples = tuple(clusters.values())

    def estimate(groups):
        strong, joint = zip(*(pair for group in groups for pair in group), strict=True)
        a, b = (percentile([math.inf if latency is None else latency for latency in arm], .95)
                for arm in (strong, joint))
        return b-a if math.isfinite(a) and math.isfinite(b) else None

    observed = estimate(samples)
    rng = random.Random(20261118)
    bootstrap = [estimate(rng.choices(samples, k=len(samples))) for _ in range(count)]
    ci = None if observed is None or None in bootstrap else [percentile(bootstrap, .025), percentile(bootstrap, .975)]
    return {"joint_minus_strong_p95_ms": observed, "ci95_ms": ci,
            "clusters": len(clusters), "bootstrap_repeats": count, "seed": 20261118,
            "unavailable_bootstrap_draws": bootstrap.count(None)}

@contextmanager
def _cold_arm_on_shared_engine(client, root, engine_name, name):
    response = client.post("/flush_cache", params={"timeout": 5})
    response.raise_for_status()
    if not response.text.startswith("Cache flushed."):
        raise RuntimeError(f"{name}: cold-arm reset refused: {response.text}")
    raw = root / "logs" / f"{engine_name}.log"
    start = raw.stat().st_size
    try:
        yield
    finally:
        with raw.open("rb") as source, (root / "logs" / f"{name}.log").open("xb") as dest:
            source.seek(start)
            dest.write(source.read())
        # Refer to the single capture instead of recapturing after policy changes.
        (root / "logs" / f"{name}-graph.json").symlink_to(f"{engine_name}-graph.json")



def main():
    from agentic_layer.policy import MECHANISMS, MODES

    native = frozenset(DEFAULT_ARMS[:2])
    leave_one = {f"joint-minus-{mechanism}": mechanism for mechanism in sorted(MECHANISMS)}
    arm_names = (*DEFAULT_ARMS, *sorted(MODES - set(DEFAULT_ARMS)), *leave_one)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--arms", nargs="+", choices=arm_names, default=DEFAULT_ARMS)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--turns-per-stratum", type=int)
    p.add_argument("--max-turn-decode", "--smoke-max-decode", type=int,
                   help="cap eligibility by recorded decode budget without shortening any selected turn")
    p.add_argument("--calibration", type=Path)
    p.add_argument("--smoke-uncalibrated", action="store_true", help="integration only; no joint benefit claim")
    p.add_argument("--reuse-engine", action="store_true", help="reuse one fixed-Graph HiCache engine; flush cache and restart ingress between arms")
    p.add_argument("--wait-drift", choices=("recorded", "early-quarter", "long-double"), default="recorded",
                   help="perturb the same frozen quarter of barriers in all arms")
    a = p.parse_args()
    if len(a.arms) != len(set(a.arms)) or a.repeats < 1 or (a.turns_per_stratum is not None and a.turns_per_stratum < 1):
        p.error("invalid repeats, duplicate arms or invalid per-stratum limit")
    if any(arm not in native for arm in a.arms) and (bool(a.calibration) == a.smoke_uncalibrated):
        p.error("controlled arms need frozen --calibration or explicit --smoke-uncalibrated")
    if a.max_turn_decode is not None and (not a.turns_per_stratum or a.max_turn_decode < 1):
        p.error("--max-turn-decode requires --turns-per-stratum and a positive value")
    if os.getenv("AGENTIC_DISABLE_MECHANISMS"):
        p.error("Use named leave-one arms instead of inheriting AGENTIC_DISABLE_MECHANISMS")
    if a.reuse_engine and (native.intersection(a.arms) or os.getenv("AGENTIC_FIXED_GRAPH") != "1"):
        p.error("--reuse-engine requires only controlled arms and AGENTIC_FIXED_GRAPH=1")
    root = a.run_dir.resolve()
    sources = ("agenticlayer/agentic_layer/engine_cache.py", "agenticlayer/agentic_layer/server.py",
               "agenticlayer/agentic_layer/policy.py", "agenticlayer/agentic_layer/versions.py",
               "experiments/trace_replay.py", "experiments/trace_matrix.py", "experiments/launch_engine.sh")

    def source_hashes():
        return {file: hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() for file in sources}

    expected_sources = source_hashes()
    runs = []
    engine_name = f"{a.name}-shared-engine"
    share = engine(root, "extended-hicache", engine_name) if a.reuse_engine else nullcontext(None)
    with share as client:
        for i in range(a.repeats):
            lesson = {}
            count = len(a.arms)
            start = i % count
            for arm in a.arms[start:] + a.arms[:start]:
                if source_hashes() != expected_sources:
                    raise RuntimeError("replay source changed between arms; use a fresh run name")
                name = f"{a.name}-{i+1}-{arm}"
                disabled = leave_one.get(arm)
                controlled = arm not in native
                mode = "joint" if disabled else arm
                if disabled:
                    os.environ["AGENTIC_DISABLE_MECHANISMS"] = disabled
                try:
                    cold = (_cold_arm_on_shared_engine(client, root, engine_name, name) if client else
                            engine(root, "extended-hicache" if controlled else arm, name))
                    with cold, (ingress(root, name, mode, a.calibration) if controlled else nullcontext(None)) as token:
                        lesson[arm] = asyncio.run(run(a.sample, root, name, token, arm == "oracle",
                                                      turns_per_stratum=a.turns_per_stratum,
                                                      max_turn_decode=a.max_turn_decode, wait_drift=a.wait_drift))
                finally:
                    if disabled:
                        os.environ.pop("AGENTIC_DISABLE_MECHANISMS", None)
            first = next(iter(lesson.values()))
            if source_hashes() != expected_sources:
                raise RuntimeError("replay source changed during an arm; use a fresh run name")
            assert all(r["sample_sha256"] == first["sample_sha256"] and
                       r["tested_turn_hashes"] == first["tested_turn_hashes"] and
                       r["altered_barriers"] == first["altered_barriers"] for r in lesson.values())
            runs.append(lesson)
    effects = {}
    if not a.smoke_uncalibrated:
        for reference, candidate in [*(("baseline", arm) for arm in a.arms if arm not in native | {"baseline", "oracle"}),
                                     *(("joint", arm) for arm in a.arms if arm in leave_one),
                                     ("strong", "joint"), ("strong", "joint-on-time"), ("joint", "oracle")]:
            match = paired(runs, reference=reference, candidate=candidate)
            if match:
                match["candidate_minus_reference_p95_ms"] = match.pop("joint_minus_strong_p95_ms")
                effects[f"{candidate} vs {reference}"] = match
    report = {"sample": str(a.sample.resolve()), "calibration_sha256": hashlib.sha256(a.calibration.read_bytes()).hexdigest() if a.calibration else None,
              "scope": "uncalibrated connection smoke" if a.smoke_uncalibrated else "GPU projected trace replay",
              "shared_engine": engine_name if a.reuse_engine else None,
              "source_sha256": expected_sources,
              "wait_drift": a.wait_drift,
              "run_summaries": [{arm: {"name": r["name"], **r["summary"]} for arm, r in arms.items()} for arms in runs],
              "paired": paired(runs) if not a.smoke_uncalibrated else None, "effects": effects}
    write_json(root / f"{a.name}.json", report)
    print(report, flush=True)
    if any(r["summary"]["statuses"].keys() != {"completed"} for arms in runs for r in arms.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
