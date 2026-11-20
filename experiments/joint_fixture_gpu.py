"""E5: put three confirmed candidates in one policy snapshot and execute them on SGLang.

This is an injected all-of resource-competition fixture for the production
Policy, not an ingress- or DSH-driven trace. A bounded background launch lane
matches the coordinator's mutual exclusion. All arms serve the same requests;
a declined preparation uses the normal cold forward, and a declined prefetch
uses explicit foreground H2D with the same physical checkpoint and fences.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import httpx

from agentic_layer.engine_client import EngineClient
from agentic_layer.policy import Bucket, Calibration, Candidate, Node, Policy, Snapshot, WaitSample

from .gpu_run import ENGINE, engine, token_fixture, write_json
from .hybrid_gpu import Driver, ident
from .state_ablation_gpu import allocation, checkpoint, generate, prepare, reset_cache, snapshot, validate_native
from .trace_matrix import percentile

SIZE = 16384
CHILD_SIZE = 4096
OUT = 8
PREP_SEED, RESUME_SEED, CHILD_SEED = 42, 42, 43
WAIT_KIND = 'fixture-tool'
ARMS = {
    'strong': ('strong', ()), 'joint': ('joint', ()), 'simple': ('simple', ()),
    'joint-on-time': ('joint-on-time', ()),
    'joint-on-time-minus-lookahead': ('joint-on-time', ('lookahead',)),
    'joint-on-time-minus-prefetch': ('joint-on-time', ('prefetch',)),
    'joint-on-time-minus-prediction': ('joint-on-time', ('prediction',)),
    'joint-minus-lookahead': ('joint', ('lookahead',)),
    'joint-minus-prefetch': ('joint', ('prefetch',)),
    'joint-minus-prediction': ('joint', ('prediction',)),
}


def now():
    return time.monotonic_ns()


def dt(start, end=None):
    return ((now() if end is None else end)-start)/1e6


def namespace():
    return ident() + ident()


async def calibrate(d, wait_ms, count):
    """Freeze real device curves and low-pressure wait/eviction observations."""
    cold, child, down, up, waits, evictions = [], [], [], [], [], []
    prefix = token_fixture(SIZE + 1, PREP_SEED)
    for k in range(count + 2):
        await reset_cache(d)
        ns, version = namespace(), ident()
        leaf, pwall = await prepare(d, prefix[:-1], ns, version)
        observed_start = now()
        off, _ = await d.transfer('offload', version, leaf, 'completed', protect=True)
        assert off['transfer_us'] > 0 and off['details']['transfer_metrics_complete']
        await asyncio.sleep(max(0, (wait_ms-dt(observed_start))/1e3))
        waiting = dt(observed_start)
        on, _ = await d.transfer('restore', version, leaf, 'ready')
        assert on['transfer_us'] > 0 and on['details']['transfer_metrics_complete']
        warm, _ = await generate(d, prefix, ns, ident(), n=OUT, hit=SIZE)
        _, route = await generate(d, prefix, namespace(), ident(), n=OUT, hit=0)
        short, _ = await generate(d, token_fixture(CHILD_SIZE, CHILD_SEED), namespace(), ident(), n=OUT, hit=0)
        if k >= 2:
            cold.append(pwall)
            child.append(short['forward_wall_ms'])
            down.append(off['transfer_us']/1000)
            up.append(on['transfer_us']/1000)
            waits.append(WaitSample(waiting, True))
            # Observe HBM candidate survival, not presence just after deliberate restore.
            # Place an independent exact target in HBM, and check it after the same wait.
            hns = namespace()
            await prepare(d, prefix[:-1], hns, ident())
            await asyncio.sleep(wait_ms/1e3)
            result, receipt, _ = await d.generate(prefix, hns, ident(), n=1)
            evictions.append(receipt['matched_tokens'] != SIZE or result['meta_info']['cached_tokens'] != SIZE)
            assert route['matched_tokens'] == 0 and warm['cached_tokens'] == SIZE
    return (Calibration((Bucket(CHILD_SIZE, 1, statistics.median(child), 0, 0, 0, count),
                         Bucket(SIZE, 1, statistics.median(cold), statistics.median(down), statistics.median(up), 0, count)),
                        {WAIT_KIND: tuple(waits)}, {'low': tuple(evictions)}),
            {'prepare_wall_ms': cold, 'child_wall_ms': child, 'd2h_ms': down, 'h2d_ms': up,
             'waits': [asdict(x) for x in waits], 'evicted': evictions})


async def case(d, label, calibration, wait_ms, pace_ms, matched_segmentation):
    initial = await reset_cache(d)
    base = allocation(initial)
    res_ns, res_version = namespace(), ident()
    prep_ns, prep_version = namespace(), ident()
    res_ids = token_fixture(SIZE + 1, RESUME_SEED)
    prep_ids = token_fixture(SIZE + 1, PREP_SEED)
    handle, _ = await prepare(d, res_ids[:-1], res_ns, res_version)
    off, _ = await d.transfer('offload', res_version, handle, 'completed', protect=True)
    assert off['details']['demoted'] and off['transfer_bytes'] > 0 and off['details']['transfer_metrics_complete']
    ready = await snapshot(d, 'three-confirmed-candidates')
    resident = allocation(ready)
    assert resident['hbm_bytes'] == base['hbm_bytes'] and resident['host_bytes'] > base['host_bytes']
    layout = ready['details']; size = off['transfer_bytes']
    mode, disabled = ARMS[label]
    policy = Policy(mode, calibration, disabled=frozenset(disabled))
    launch, tool_target = now(), wait_ms * 1_000_000
    deadline = launch + max(1_000_000_000, tool_target * 2)
    elapsed = lambda: dt(launch)
    raw_decisions, started_at, finished_at, work = [], {}, {}, {}
    jobs, background = {}, None

    def make(kind, **params):
        return Candidate(kind, kind, kind, kind, kind, launch, deadline, **params)

    candidates = {'admit': make('admit', foreground=True, node_id='admit', token_count=CHILD_SIZE,
                    decode_hbm_bytes=OUT*layout['full_slot_bytes']),
                  'prepare': make('prepare', background=True, token_count=SIZE,
                    hbm_reservation_bytes=size, wait_type=WAIT_KIND),
                  'restore': make('restore', background=True, token_count=SIZE,
                    hbm_reservation_bytes=size, transfer_bytes=size, wait_type=WAIT_KIND)}
    child_service = calibration.bucket(CHILD_SIZE, 1).prefill_ms
    nodes = [Node('admit', child_service)]

    async def run_work(kind):
        started_at[kind] = elapsed()
        if kind == 'admit':
            response, _ = await generate(d, token_fixture(CHILD_SIZE, CHILD_SEED), namespace(), ident(), n=OUT, hit=0)
            work[kind] = response
        elif kind == 'prepare':
            result, receipt, service = await d.generate(prep_ids[:-1], prep_ns, prep_version, prepare=True)
            leaf = checkpoint(receipt, SIZE)
            validate_native(result, receipt, SIZE, 0, 0)
            released, = await d.rpc(d.command('release', version=prep_version, handle=leaf))
            assert released['status'] == 'ok'
            work[kind] = {'service_ms': service}
        else:
            credit_start = now()
            # Issue-budget test, not a change to PCIe speed or the device kernel.
            # All arms incur the same debit above the calibrated copy service.
            if pace_ms:
                await asyncio.sleep(pace_ms / 1e3)
            issue_wait_ms = dt(credit_start)
            ack, wall = await d.transfer('restore', res_version, handle, 'ready')
            assert ack['matched_tokens'] == SIZE and ack['transfer_bytes'] == size and ack['details']['transfer_metrics_complete']
            work[kind] = {'transfer_wall_ms': wall, 'transfer_us': ack['transfer_us'], 'transfer_bytes': ack['transfer_bytes'],
                          'pacing_before_issue_ms': issue_wait_ms, 'pacing_target_ms': pace_ms}
        finished_at[kind] = elapsed()

    # Feed only one writer with the shared native control socket. At most one
    # background copy/prefill is outstanding, but the necessary child can run
    # in the GPU's own continuous batch concurrently.
    async with asyncio.TaskGroup() as group:
        while candidates and (now()-launch < tool_target or 'admit' in candidates):
            if background is not None and background.done():
                background = None
            # Slot bounds are conservative while a just-submitted background
            # request has not materialized. Native bind enforces true slots.
            await d.control.poll()
            obs = d.control.snapshot
            st = obs['details']
            clock = now()
            view = Snapshot(clock, d.control.snapshot_ns, layout['hbm_capacity_bytes'], obs['unique_hbm_bytes'],
                layout['host_capacity_bytes'], obs['unique_host_bytes'], st['hbm_reserved_bytes'], st['host_reserved_bytes'],
                obs['fg_queue'], obs['bg_queue'], background is not None)
            frame = [replace(c, wait_elapsed_ms=dt(launch, clock)) for c in candidates.values()
                     if c.kind == 'admit' or clock-launch < tool_target]
            selected = policy.choose(view, frame, nodes)
            raw_decisions.append({'elapsed_ms': dt(launch, clock), 'background_busy': background is not None,
                                  'decisions': [asdict(q) for q in selected.decisions]})
            if selected.chosen is None:
                # Work that was disabled by this ablation cannot become eligible.
                for x in selected.decisions:
                    if x.reason == 'disabled_by_mode':
                        candidates.pop(x.kind)
                if 'admit' in candidates and now() >= deadline:
                    raise RuntimeError('confirmed child missed its deadline')
                await asyncio.sleep(0.004)
                continue
            kind = selected.chosen.kind
            candidates.pop(kind)
            jobs[kind] = group.create_task(run_work(kind))
            if kind != 'admit':
                background = jobs[kind]
                # Submit the chosen native operation before opening another round.
                await asyncio.sleep(0)
        await asyncio.sleep(max(0, (launch+tool_target-now())/1e9))
        released_ms = elapsed()
        # Actual continuation owns the old host state. If a background restore
        # is in flight it must drain; otherwise the same H2D runs on demand.
        if 'restore' not in jobs:
            group.create_task(run_work('restore'))
        cut_used = 'prepare' in jobs and jobs['prepare'].done()
    # Keep natural cold fallback as the original baseline. The matched arm
    # moves the same segmented prefix work to the foreground and pays for it
    # after the cut; it isolates scheduling from warm/cold kernel numerics.
    target_ns = prep_ns if cut_used else namespace()
    synchronous_prepare_ms = None
    if matched_segmentation and not cut_used:
        _, synchronous_prepare_ms = await prepare(d, prep_ids[:-1], target_ns, ident())
    prep_result, _ = await generate(d, prep_ids, target_ns, ident(), n=OUT,
                                    hit=SIZE if cut_used or matched_segmentation else 0)
    res_result, _ = await generate(d, res_ids, res_ns, ident(), n=OUT, hit=SIZE)
    wall = elapsed()
    return {'label': label, 'elapsed_ms': wall, 'tool_return_ms': released_ms, 'start_ms': started_at,
            'finish_ms': finished_at, 'first': raw_decisions[0]['decisions'], 'decisions': raw_decisions,
            'resumed': res_result, 'target': prep_result, 'child': work['admit'], 'preparation_used': cut_used,
            'early_restore_selected': 'restore' in jobs, 'synchronous_prepare_ms': synchronous_prepare_ms,
            'work': work, 'before_gate': resident}


async def compare(root, args, report):
    ctl = EngineClient(root/'engine.sock', timeout=120)
    rows = report['runs']
    with (root/'logs'/f'{args.name}-wire.jsonl').open('x', buffering=1) as log:
        async with httpx.AsyncClient(base_url=ENGINE, trust_env=False, timeout=180) as http:
            d = Driver(ctl, http, log, args.name)
            await ctl.handshake()
            calibration, raw = await calibrate(d, args.wait_ms, 10)
            report['calibration'] = raw
            d.record('calibration', raw)
            pace_ms = calibration.bucket(SIZE, 1).h2d_ms * (1 / args.h2d_fraction - 1)
            if pace_ms:
                calibration = replace(calibration, buckets=tuple(
                    replace(bucket, h2d_ms=bucket.h2d_ms + pace_ms) if bucket.token_count == SIZE else bucket
                    for bucket in calibration.buckets))
            ctl.timeout = 2
            for row in rows:
                for label in row['order']:
                    arm = await case(d, label, calibration, row['return_ms'], pace_ms, args.matched_segmentation)
                    row['arms'][label] = arm
                    d.record('completed-arm', {'repeat': row['repeat'], 'return_ms': row['return_ms'], **arm})
                ref = row['arms'][args.arms[0]]
                if not all(arm[k]['output_ids'] == ref[k]['output_ids'] for arm in row['arms'].values()
                           for k in ('resumed', 'target', 'child')):
                    row['status'] = 'different_output'
                    d.record('pair-error', row)
                    continue
                row['status'] = 'ok'


def summarize(rows, arms):
    # All planned cases count. Incomplete or semantically different work never
    # yields a qualifying speedup.
    if any(row['status'] != 'ok' for row in rows):
        return {'verified_pairs': sum(row['status'] == 'ok' for row in rows), 'planned_pairs': len(rows)}
    stats = {}
    for arm in arms:
        samples = [r['arms'][arm]['elapsed_ms'] for r in rows]
        stats[arm] = {'p50_ms': statistics.median(samples), 'p95_ms': percentile(samples, .95),
                      'cache_preparations_used': sum(r['arms'][arm]['preparation_used'] for r in rows),
                      'early_restores': sum(r['arms'][arm]['early_restore_selected'] for r in rows)}
    pairs = {}
    if 'strong' in arms:
        for arm in arms:
            if arm == 'strong':
                continue
            xy = [(r['arms']['strong']['elapsed_ms'], r['arms'][arm]['elapsed_ms']) for r in rows]
            rng = random.Random(20261120)
            distributions = []
            for _ in range(10000):
                a, b = zip(*rng.choices(xy, k=len(xy)), strict=True)
                distributions.append(percentile(b, .95)-percentile(a, .95))
            intervals = [percentile(distributions, q) for q in (.025, .975)]
            gain = 100*(1 - stats[arm]['p95_ms']/stats['strong']['p95_ms'])
            pairs[arm] = {'p95_reduction_pct': gain, 'delta_p95_ci95_ms': intervals,
                          'credible_over_5_pct': gain > 5 and intervals[1] < 0}
    return {'verified_pairs': len(rows), 'planned_pairs': len(rows), 'arms': stats, 'vs_strong': pairs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--pairs', type=int, default=6)
    p.add_argument('--wait-ms', type=int, default=600)
    p.add_argument('--return-ms', type=int, nargs='+', help='actual wait(s); calibration remains at the nominal wait')
    p.add_argument('--h2d-fraction', type=float, default=1,
                   help='pace H2D issuance to a fraction of the calibrated service rate; same native copies in every arm')
    p.add_argument('--matched-segmentation', action='store_true',
                   help='pay the same prepare/prefix-hit split after the cut when speculation misses')
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    a = p.parse_args()
    if not (a.pairs > 0 and 0 < a.wait_ms <= 2000 and len(a.arms) == len(set(a.arms))):
        p.error('invalid pairs, waits, or arms')
    if (os.getenv('AGENTIC_FIXED_GRAPH'), os.getenv('AGENTIC_DECODE_GRAPH')) != ('1', 'full'):
        p.error('requires AGENTIC_FIXED_GRAPH=1 AGENTIC_DECODE_GRAPH=full')
    returns = a.return_ms or [a.wait_ms]
    if (not 0 < a.h2d_fraction <= 1 or any(not 0 < r <= 2000 for r in returns)
            or len(set(returns)) != len(returns)):
        p.error('invalid actual waits')
    rows = []
    for i in range(a.pairs):
        for index, returned in enumerate(returns):
            j = (i+index) % len(a.arms)
            rows.append({'repeat': i+1, 'return_ms': returned, 'order': a.arms[j:]+a.arms[:j],
                         'status': 'missing', 'arms': {}})
    report = {'scope': 'actual GPU, injected candidate set through production Policy and native cache RPC',
              'nominal_wait_ms': a.wait_ms, 'h2d_issuance_fraction': a.h2d_fraction,
              'matched_segmentation': a.matched_segmentation, 'runs': rows}
    try:
        with engine(a.run_dir, 'extended-hicache', a.name):
            asyncio.run(compare(a.run_dir, a, report))
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['summary'] = {str(wait): summarize([r for r in rows if r['return_ms'] == wait], a.arms)
                             for wait in returns}
        write_json(a.run_dir/f'{a.name}.json', report)
        print(json.dumps(report['summary']), flush=True)
    if any(r['status'] != 'ok' for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
