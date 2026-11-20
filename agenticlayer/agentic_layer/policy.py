"""Deterministic, side-effect-free action selection for the AgenticLayer scheduler.

The scheduler owns serialization, leases, reservations and execution.  A Selection
is only a recommendation against one snapshot: reserve its resources atomically,
then obtain a fresh snapshot and choose again.  In particular, a policy decision
is not an engine completion or a permission to bypass the engine's safety checks.

Calibration JSON has ``buckets`` (token_count, batch_size, prefill_ms,
d2h_ms, h2d_ms, recurrent_ms, samples), ``waits`` mapping wait types to
[{duration_ms, completed}], and ``evictions`` mapping pressure labels to
[{evicted: bool}]. Buckets are upper bounds, never extrapolated. Wait samples
may be right-censored (completed=false). No file is required at startup.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

GIB = 1024 ** 3
STALE_NS = 100_000_000
AGE_NS = 500_000_000
COOLDOWN_NS = 500_000_000
WINDOW_MS = 2000.0
KINDS = frozenset({"admit", "defer", "retain", "offload", "restore", "recompute", "prepare", "pause", "reclaim"})
_JOINT_RANKING = frozenset({"joint", "joint-on-time", "joint-no-prefetch", "joint-no-predict", "oracle"})
MODES = _JOINT_RANKING | {"baseline", "simple", "strong", "parallel-only", "lifecycle-only",
                          "offload-only", "lookahead-only", "ttl", "task-prefix"}
MECHANISMS = frozenset({"parallel", "lifecycle", "placement", "lookahead", "prefetch", "prediction"})


@dataclass(frozen=True, slots=True)
class Snapshot:
    now_ns: int
    observed_ns: int
    hbm_capacity_bytes: int
    hbm_used_bytes: int
    host_capacity_bytes: int
    host_used_bytes: int
    hbm_reserved_bytes: int = 0
    host_reserved_bytes: int = 0
    fg_queue: int = 0
    bg_queue: int = 0
    optional_background_inflight: bool = False


@dataclass(frozen=True, slots=True)
class Node:
    node_id: str
    service_ms: float
    predecessors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Candidate:
    action_id: str
    version_id: str
    session_id: str
    branch_id: str
    kind: str
    ready_ns: int
    expires_mono_ns: int
    # The caller establishes legality from current version, leases, dependencies,
    # copy fences and engine capabilities. Policy never infers physical safety.
    legal: bool = True
    necessary: bool = False
    expired: bool = False
    foreground: bool = False
    background: bool = False
    token_count: int = 1
    batch_size: int = 1
    node_id: str = ""
    matched_tokens: int = 0
    # Additional peak reservations, including decode debt and copy destinations.
    # Negative resident deltas are never credited before an engine ACK.
    hbm_delta_bytes: int = 0
    host_delta_bytes: int = 0
    hbm_reservation_bytes: int = 0
    host_reservation_bytes: int = 0
    decode_hbm_bytes: int = 0
    transfer_bytes: int = 0
    # Candidate-specific estimates are in milliseconds; None means unknown.
    # A measured caller estimate can rank mandatory alternatives without a
    # calibration file; it does not enable optional work.
    estimated_action_ms: float | None = None
    saved_fg_ms: float | None = None
    added_fg_ms: float = 0.0
    stall_ms: float | None = None
    restall_ms: float | None = None
    reuse_probability: float = 1.0
    wait_type: str = ""
    wait_elapsed_ms: float = 0.0
    oracle_remaining_ms: float | None = None
    last_migration_ns: int | None = None
    suspended_since_ns: int | None = None
    # Estimated coexistence/residency window; bounded by 2 s in scoring.
    residency_ms: float = 2000.0
    pressure: str = ""


@dataclass(frozen=True, slots=True)
class Decision:
    action_id: str
    version_id: str
    kind: str
    accepted: bool
    reason: str
    score: float | None
    costs: Mapping[str, float]
    expires_mono_ns: int


@dataclass(frozen=True, slots=True)
class Selection:
    chosen: Decision | None
    decisions: tuple[Decision, ...]


@dataclass(frozen=True, slots=True)
class Bucket:
    token_count: int
    batch_size: int
    prefill_ms: float
    d2h_ms: float
    h2d_ms: float
    recurrent_ms: float
    samples: int
    # Optional measured setup and throughput allow actual transfer-byte scaling.
    d2h_setup_ms: float = 0.0
    h2d_setup_ms: float = 0.0
    d2h_bytes_per_ms: float = 0.0
    h2d_bytes_per_ms: float = 0.0
    queue_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class WaitSample:
    duration_ms: float
    completed: bool


@dataclass(frozen=True, slots=True)
class Calibration:
    buckets: tuple[Bucket, ...]
    waits: Mapping[str, tuple[WaitSample, ...]] = field(default_factory=dict)
    evictions: Mapping[str, tuple[bool, ...]] = field(default_factory=dict)
    _wait_curves: Mapping[str, tuple[tuple[float, ...], tuple[float, ...]]] = field(
        init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Calibration is frozen for a run. Build Kaplan–Meier once, not inside
        # each scheduler tick for every ready restore or prepare action.
        curves = {}
        for kind, samples in self.waits.items():
            if len(samples) < 10:
                continue
            grouped: dict[float, list[int]] = {}
            for sample in samples:
                counts = grouped.setdefault(sample.duration_ms, [0, 0])
                counts[0 if sample.completed else 1] += 1
            times, neg_survival = [], []
            at_risk, survival = len(samples), 1.0
            for elapsed, (events, censored) in sorted(grouped.items()):
                if events:
                    survival *= 1.0 - events / at_risk
                at_risk -= events + censored
                times.append(elapsed)
                neg_survival.append(-survival)
            curves[kind] = tuple(times), tuple(neg_survival)
        object.__setattr__(self, "_wait_curves", curves)

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("calibration must be an object")
        buckets = tuple(Bucket(**item) for item in raw["buckets"])
        waits = {key: tuple(WaitSample(**sample) for sample in values)
                 for key, values in raw.get("waits", {}).items()}
        evictions = {key: tuple(sample["evicted"] for sample in values)
                     for key, values in raw.get("evictions", {}).items()}
        result = cls(buckets, waits, evictions)
        result.validate()
        return result

    def validate(self) -> None:
        seen: set[tuple[int, int]] = set()
        for b in self.buckets:
            key = (b.token_count, b.batch_size)
            if key in seen or b.token_count not in (1, 1024, 4096, 8192, 16384, 32768) or b.batch_size not in (1, 4, 8) or b.samples < 0:
                raise ValueError("invalid or duplicate calibration bucket")
            seen.add(key)
            if any(not math.isfinite(x) or x < 0 for x in (b.prefill_ms, b.d2h_ms, b.h2d_ms, b.recurrent_ms, b.d2h_setup_ms, b.h2d_setup_ms, b.d2h_bytes_per_ms, b.h2d_bytes_per_ms, b.queue_ms)):
                raise ValueError("invalid calibration cost")
        for samples in self.waits.values():
            if any(not math.isfinite(s.duration_ms) or s.duration_ms < 0 or not isinstance(s.completed, bool) for s in samples):
                raise ValueError("invalid wait sample")
        if any(not isinstance(v, bool) for values in self.evictions.values() for v in values):
            raise ValueError("invalid eviction sample")

    def bucket(self, tokens: int, batch: int) -> Bucket | None:
        # Batch and prefix bounds must both cover the request; select the smallest
        # covering shape deterministically, not the nearest lower measurement.
        choices = (b for b in self.buckets if b.token_count >= tokens and b.batch_size >= batch and b.samples >= 10)
        return min(choices, key=lambda b: (b.token_count, b.batch_size), default=None)

    def remaining_p50(self, wait_type: str, elapsed_ms: float) -> float | None:
        curve = self._wait_curves.get(wait_type)
        if curve is None:
            return None
        times, negative = curve
        start = bisect_right(times, elapsed_ms)
        base = negative[start - 1] if start else -1.0
        if base == 0.0:
            return None
        median = bisect_left(negative, base * 0.5, lo=start)
        return times[median] - elapsed_ms if median < len(times) else None

    def eviction_probability(self, pressure: str) -> float | None:
        samples = self.evictions.get(pressure, ())
        return sum(samples) / len(samples) if len(samples) >= 10 else None


def critical_paths(nodes: Sequence[Node]) -> dict[str, float]:
    """Longest known remaining service path from each node; reject cyclic DAGs."""
    by_id = {node.node_id: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("duplicate DAG node")
    successors: dict[str, list[str]] = {key: [] for key in by_id}
    for node in nodes:
        if not math.isfinite(node.service_ms) or node.service_ms < 0:
            raise ValueError("invalid node service estimate")
        for parent in node.predecessors:
            if parent not in by_id:
                raise ValueError("unknown DAG predecessor")
            successors[parent].append(node.node_id)
    visiting: set[str] = set()
    result: dict[str, float] = {}

    def visit(key: str) -> float:
        if key in result:
            return result[key]
        if key in visiting:
            raise ValueError("cyclic DAG")
        visiting.add(key)
        value = by_id[key].service_ms + max((visit(child) for child in successors[key]), default=0.0)
        visiting.remove(key)
        result[key] = value
        return value

    for key in sorted(by_id):
        visit(key)
    return result


class Policy:
    def __init__(self, mode: str, calibration: Calibration | None = None, *, disabled: frozenset[str] = frozenset()):
        if mode not in MODES:
            raise ValueError(f"unknown policy mode: {mode}")
        if disabled - MECHANISMS:
            raise ValueError(f"unknown disabled mechanisms: {sorted(disabled - MECHANISMS)}")
        if calibration is not None:
            calibration.validate()
        self.mode = mode
        self.calibration = calibration
        self.disabled = disabled
        self._observed: dict[str, tuple[float, float]] = {}
        self._migration_blocked = False

    def observe(self, wait_type: str, transfer_ms: float, queue_ms: float) -> None:
        """Update per-wait-type actual transfer and queue EWMAs after completion."""
        if not wait_type or any(not math.isfinite(x) or x < 0 for x in (transfer_ms, queue_ms)):
            raise ValueError("invalid completion observation")
        previous = self._observed.get(wait_type)
        self._observed[wait_type] = ((transfer_ms, queue_ms) if previous is None else
                                     (0.8 * previous[0] + 0.2 * transfer_ms,
                                      0.8 * previous[1] + 0.2 * queue_ms))

    def _mode_allowed(self, c: Candidate, background: bool) -> bool:
        if c.necessary:
            return True  # Never disable correctness-critical foreground progress.
        if (c.kind == "reclaim" and "lifecycle" in self.disabled or
                c.kind in {"retain", "offload", "restore"} and "placement" in self.disabled or
                c.kind == "restore" and background and "prefetch" in self.disabled or
                c.kind == "prepare" and "lookahead" in self.disabled):
            return False
        if self.mode in {"baseline", "parallel-only", "task-prefix"}:
            return c.kind == "admit"
        if self.mode in {"lifecycle-only", "ttl"}:
            return c.kind in {"admit", "reclaim"}
        if self.mode == "offload-only":
            return c.kind in {"admit", "offload", "restore"} and not (c.kind == "restore" and background)
        if self.mode == "lookahead-only":
            return c.kind in {"admit", "prepare"}
        if self.mode in {"joint-no-prefetch", "joint-no-predict"} and c.kind == "restore" and background:
            return False
        if self.mode == "joint-no-predict" and c.kind == "prepare":
            return False
        return True

    def choose(self, snapshot: Snapshot, candidates: Sequence[Candidate], nodes: Sequence[Node] = ()) -> Selection:
        s = snapshot
        if (s.now_ns < 0 or s.observed_ns < 0 or s.observed_ns > s.now_ns or
            min(s.hbm_capacity_bytes, s.host_capacity_bytes, s.hbm_used_bytes, s.host_used_bytes,
                s.hbm_reserved_bytes, s.host_reserved_bytes, s.fg_queue, s.bg_queue) < 0):
            raise ValueError("invalid resource snapshot")
        if len({c.action_id for c in candidates}) != len(candidates):
            raise ValueError("duplicate candidate action_id")
        paths = critical_paths(nodes)
        occupancy = ((s.hbm_used_bytes + s.hbm_reserved_bytes) / s.hbm_capacity_bytes
                     if s.hbm_capacity_bytes else 1.0)
        if occupancy >= 0.85:
            self._migration_blocked = True
        elif occupancy <= 0.70:
            self._migration_blocked = False
        host_occupancy = ((s.host_used_bytes + s.host_reserved_bytes) / s.host_capacity_bytes
                          if s.host_capacity_bytes else 1.0)
        pressure = "high" if occupancy >= 0.85 else "medium" if occupancy >= 0.60 else "low"
        decisions: list[Decision] = []
        viable: list[tuple[Candidate, Decision]] = []
        for c in candidates:
            if (c.kind not in KINDS or not c.action_id or not c.version_id or not c.session_id or not c.branch_id or
                c.ready_ns < 0 or c.ready_ns > s.now_ns or c.expires_mono_ns < 0 or
                c.token_count < 1 or c.batch_size not in range(1, 9) or c.matched_tokens < 0 or
                (c.necessary and (not c.foreground or c.background or c.kind not in {"admit", "restore", "recompute"})) or
                min(c.hbm_reservation_bytes, c.host_reservation_bytes, c.decode_hbm_bytes, c.transfer_bytes) < 0 or
                not 0 <= c.reuse_probability <= 1 or
                any(not math.isfinite(x) or x < 0 for x in (c.added_fg_ms, c.wait_elapsed_ms, c.residency_ms)) or
                any(x is not None and (not math.isfinite(x) or x < 0) for x in (c.estimated_action_ms, c.saved_fg_ms, c.stall_ms, c.restall_ms, c.oracle_remaining_ms))):
                raise ValueError(f"invalid candidate: {c.action_id}")
            mandatory = c.necessary or (c.kind == "reclaim" and c.expired)
            # Pause controls future launches; it is not itself background work.
            background = c.kind != "pause" and (c.background or (not c.foreground and c.kind in {"prepare", "offload", "restore"}))
            aged = c.foreground and s.now_ns - c.ready_ns >= AGE_NS
            reason = ""
            hbm_need = max(0, c.hbm_delta_bytes) + c.hbm_reservation_bytes + c.decode_hbm_bytes
            host_need = max(0, c.host_delta_bytes) + c.host_reservation_bytes
            if not c.legal:
                reason = "illegal"
            elif c.expires_mono_ns <= s.now_ns:
                reason = "expired_permit"
            elif (hbm_need > 0 and s.hbm_used_bytes + s.hbm_reserved_bytes + hbm_need > s.hbm_capacity_bytes) or (host_need > 0 and s.host_used_bytes + s.host_reserved_bytes + host_need > s.host_capacity_bytes):
                reason = "capacity"
            elif not self._mode_allowed(c, background):
                reason = "disabled_by_mode"
            elif not mandatory and self.calibration is None:
                reason = "uncalibrated"
            elif self.mode == "ttl" and c.kind == "reclaim" and not c.expired and (c.suspended_since_ns is None or s.now_ns - c.suspended_since_ns < 200_000_000):
                reason = "ttl_cooldown"
            elif not mandatory and background and s.now_ns - s.observed_ns > STALE_NS:
                reason = "stale_snapshot"
            elif not mandatory and background and s.fg_queue:
                reason = "engine_foreground_queued"
            elif not mandatory and background and s.optional_background_inflight:
                reason = "background_inflight"
            elif not mandatory and c.kind == "restore" and background and self._migration_blocked:
                reason = "hbm_hysteresis"
            elif not mandatory and c.kind in {"restore", "offload"} and c.last_migration_ns is not None and s.now_ns - c.last_migration_ns < COOLDOWN_NS:
                reason = "migration_cooldown"
            bucket = self.calibration.bucket(c.token_count, c.batch_size) if self.calibration else None
            costs: dict[str, float] = {"hbm_need_bytes": float(hbm_need), "host_need_bytes": float(host_need),
                                      "critical_path_ms": paths.get(c.node_id, 0.0)}
            if c.estimated_action_ms is not None:
                costs["action_cost_ms"] = c.estimated_action_ms
            score: float | None = None
            if not reason and not mandatory and bucket is None:
                reason = "missing_calibration_bucket"
            if bucket is not None:
                transfer = (bucket.d2h_ms if c.kind == "offload" else bucket.h2d_ms if c.kind in {"restore", "retain"} else 0.0)
                setup = bucket.d2h_setup_ms if c.kind == "offload" else bucket.h2d_setup_ms
                bandwidth = bucket.d2h_bytes_per_ms if c.kind == "offload" else bucket.h2d_bytes_per_ms
                if c.kind in {"offload", "restore"} and bandwidth > 0:
                    transfer = setup + c.transfer_bytes / bandwidth
                queue = bucket.queue_ms
                if c.wait_type in self._observed:
                    observed_transfer, queue = self._observed[c.wait_type]
                    if c.kind in {"offload", "restore"}:
                        transfer = observed_transfer
                reload_ms = transfer + queue + (bucket.recurrent_ms if c.kind in {"restore", "retain"} else 0.0)
                action_cost = (c.estimated_action_ms if c.estimated_action_ms is not None else
                               reload_ms if c.kind in {"restore", "offload"} else
                               bucket.prefill_ms if c.kind in {"prepare", "recompute"} else 0.0)
                saved = c.saved_fg_ms if c.saved_fg_ms is not None else min(bucket.prefill_ms, reload_ms) if c.kind == "retain" else bucket.prefill_ms if c.kind in {"prepare", "restore"} else paths.get(c.node_id, 0.0) if c.kind == "admit" else 0.0
                stall = c.stall_ms if c.stall_ms is not None else saved
                restall = c.restall_ms if c.restall_ms is not None else bucket.prefill_ms
                window = min(WINDOW_MS, c.residency_ms) / 1000.0
                lambda_h = max(0.0, occupancy - 0.60) / 0.40 * bucket.prefill_ms
                lambda_d = max(0.0, host_occupancy - 0.60) / 0.40 * bucket.prefill_ms
                hbm_cost = lambda_h * (c.hbm_delta_bytes + c.hbm_reservation_bytes + c.decode_hbm_bytes) / GIB * window
                host_cost = lambda_d * (c.host_delta_bytes + c.host_reservation_bytes) / GIB * window
                added_fg = c.added_fg_ms + (action_cost if c.foreground and c.kind in {"restore", "recompute"} else 0.0)
                score = saved * c.reuse_probability - added_fg - hbm_cost - host_cost
                costs.update(prefill_ms=bucket.prefill_ms, transfer_ms=transfer, queue_ms=queue,
                             reload_ms=reload_ms, action_cost_ms=action_cost, saved_fg_ms=saved,
                             added_fg_ms=added_fg, hbm_cost=hbm_cost, host_cost=host_cost,
                             lambda_h=lambda_h, lambda_d=lambda_d, stall_ms=stall, restall_ms=restall)
                if (not reason and not mandatory and background and "prediction" not in self.disabled
                        and self.mode not in {"simple", "joint-no-predict"}):
                    remaining = c.oracle_remaining_ms if self.mode == "oracle" else self.calibration.remaining_p50(c.wait_type, c.wait_elapsed_ms)
                    p_evict = self.calibration.eviction_probability(c.pressure or pressure)
                    if remaining is None or p_evict is None:
                        reason = "insufficient_prediction_samples"
                    else:
                        eta = 1.0 if action_cost == 0 else min(1.0, remaining / action_cost)
                        costs.update(wait_remaining_ms=remaining, eviction_probability=p_evict, eta=eta)
                        if eta * stall < transfer + queue + p_evict * restall:
                            reason = "insufficient_slack"
                        elif (self.mode == "joint-on-time" and c.kind in {"prepare", "restore"}
                              and remaining < action_cost):
                            # The exact-state CAS and transfer fence only expose
                            # a hit after the entire action. A partial checkpoint
                            # at the expected cut does not earn fractional reuse.
                            reason = "unlikely_to_finish_before_cut"
                if not reason and not mandatory and not aged and self.mode in _JOINT_RANKING and score < 0:
                    reason = "negative_utility"
            decision = Decision(c.action_id, c.version_id, c.kind, False, reason, score, costs, c.expires_mono_ns)
            decisions.append(decision)
            if not reason:
                viable.append((c, decision))
        if not viable:
            return Selection(None, tuple(decisions))

        def rank(item: tuple[Candidate, Decision]) -> tuple:
            c, d = item
            tie = (c.ready_ns, c.session_id, c.branch_id, c.action_id)
            if c.kind == "reclaim" and c.expired:
                return (0, 0, *tie)
            if c.necessary:
                # Aging wins across foreground branches; cost breaks ties between
                # equivalent alternatives for the same ready branch.
                aged = self.mode != "simple" and s.now_ns - c.ready_ns >= AGE_NS
                return (1, 0 if aged else 1, c.ready_ns, c.session_id, c.branch_id,
                        d.costs.get("action_cost_ms", math.inf), c.action_id)
            if self.mode != "simple" and c.foreground and s.now_ns - c.ready_ns >= AGE_NS:
                return (2, 0, *tie)
            if self.mode in _JOINT_RANKING:
                return (3, -(d.score or 0.0), *tie)
            if self.mode == "task-prefix":
                return (3, -d.costs.get("critical_path_ms", 0.0), -c.matched_tokens, *tie)
            category = {"admit": 0, "restore": 1, "recompute": 1, "retain": 1, "offload": 1,
                        "reclaim": 1, "pause": 1, "prepare": 2, "defer": 3}[c.kind]
            # Each independent strong subpolicy selects within its own class.
            subscore = (-d.costs.get("critical_path_ms", 0.0) if c.kind == "admit" else
                        d.costs.get("action_cost_ms", 0.0) if c.kind in {"restore", "recompute"} else
                        -d.costs.get("stall_ms", 0.0) if c.kind == "prepare" else -(d.score or 0.0))
            return (3, category, subscore, *tie)

        candidate, selected = min(viable, key=rank)
        chosen = replace(selected, accepted=True, reason="selected")
        final = tuple(chosen if d.action_id == candidate.action_id else replace(d, reason="not_selected") if not d.reason else d for d in decisions)
        return Selection(chosen, final)
