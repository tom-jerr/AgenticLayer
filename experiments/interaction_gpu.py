"""Cross real read parallelism and paid context preparation under one GPU graph."""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import tempfile
import time
from pathlib import Path

import httpx

from .compaction_gpu import ARMS, CITIES, SCENARIOS, calibrate, emit, engine_evidence, run_arm, verify_pairs
from .gpu_run import ENGINE, engine, write_json
from .hybrid_gpu import INGRESS, ingress
from .prepare_gpu import reset_cache
from .trace_matrix import percentile

SCENARIO = 'stable-2000'
READS = (('a.txt', '17\n', 900), ('b.txt', '25\n', 700))
CELLS = tuple((parallel, arm) for parallel in (False, True) for arm in ARMS)


def label(parallel, arm):
    return ('parallel/' if parallel else 'serial/') + arm


async def reads(folder, parallel):
    async def one(name, expected, service_ms):
        started = time.monotonic_ns()
        # Fixed service window for the anonymized external tool; the scoped
        # read itself is real and identical. Not an IO resource-pressure test.
        await asyncio.sleep(service_ms / 1000)
        data = await asyncio.to_thread((folder/name).read_text)
        assert data == expected, name
        return {'name': name, 'data': data, 'service_ms': service_ms,
                'started_ns': started, 'completed_ns': time.monotonic_ns()}
    out = await asyncio.gather(*(one(*row) for row in READS)) if parallel else [await one(*row) for row in READS]
    assert [r['name'] for r in out] == [x[0] for x in READS]
    assert ((max(x['started_ns'] for x in out) < min(x['completed_ns'] for x in out)) == parallel)
    return out


async def measure(root, name, token, rows, directory):
    cfg = {'Authorization': f'Bearer {token}'}
    async with (httpx.AsyncClient(base_url=INGRESS, headers=cfg, timeout=180, trust_env=False) as client,
                httpx.AsyncClient(base_url=ENGINE, timeout=180, trust_env=False) as native):
        with (root/'logs'/f'{name}-work.jsonl').open('x', buffering=1) as log:
            for order, serial, parallel in zip(*rows, strict=True):
                for use_parallel, arm in order:
                    trial = parallel if use_parallel else serial
                    row = trial['arms'][arm] = {'status': 'error'}
                    try:
                        await reset_cache(native)
                        await run_arm(client, SCENARIOS[SCENARIO], arm, trial['city'], row,
                                      tool_runner=lambda: reads(directory, use_parallel))
                    except Exception as exc:
                        row['error'] = f'{type(exc).__name__}: {exc}'
                    else:
                        row['status'] = 'recorded'
                    finally:
                        emit(log, {'pair': trial['pair'], 'cell': label(use_parallel, arm), **row})


def compare(rows):
    _, serial, parallel = rows
    primary, canceled = [], 0
    for p, q in zip(serial, parallel, strict=True):
        styles = p['arms'], q['arms']
        if p['status'] != 'ok' or q['status'] != 'ok':
            continue
        shapes = {(x['summary']['body_sha256'], x['summary']['content'], x['model']['body_sha256'], x['model']['content'],
                   tuple((r['name'], r['data'], r['service_ms']) for r in x['tool_observations']))
                  for style in styles for x in style.values()}
        if len(shapes) != 1:
            p['status'] = q['status'] = 'different_work'
            continue
        primary.append({label(i == 1, k): v['total_ms'] for i, s in enumerate(styles) for k, v in s.items()})
        canceled += sum(x['prepare_outcome'].startswith('fallback') for s in styles for x in s.values())
    report = {'verified_pairs': len(primary), 'planned_pairs': len(serial), 'fallbacks': canceled,
              'qualified': len(primary) == len(serial)}
    if not report['qualified']:
        return report

    names = [label(p, arm) for p, arm in CELLS]

    def effects(values):
        a, b, c, d = (percentile([v[name] for v in values], .95) for name in names)
        return (b-a, d-c, (d-c)-(b-a), d-a)

    a = effects(primary)
    rng = random.Random(20261120)
    simulated = [effects(rng.choices(primary, k=len(primary))) for _ in range(10000)]
    report['p95_ms'] = {c: percentile([row[c] for row in primary], .95) for c in names}
    report['effects_ms'] = {k: {'p95_difference': v, 'ci95': [percentile(x, .025), percentile(x, .975)]}
                            for k, v, x in zip(('lookahead_in_serial', 'lookahead_in_parallel', 'interaction', 'both_vs_neither'),
                                               a, zip(*simulated, strict=True), strict=True)}
    report['candidate_used'] = {label(flag, arm): sum(row['arms'][arm]['used_candidate'] for row in style)
                                for flag, style in zip((False, True), (serial, parallel), strict=True) for arm in ARMS}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--pairs', type=int, default=10)
    a = p.parse_args()
    if a.pairs < 1 or (os.getenv('AGENTIC_FIXED_GRAPH'), os.getenv('AGENTIC_DECODE_GRAPH')) != ('1', 'full'):
        p.error('requires positive pairs and AGENTIC_FIXED_GRAPH=1 AGENTIC_DECODE_GRAPH=full')
    root = a.run_dir
    orders, serial, parallel = [], [], []
    for i in range(a.pairs):
        start = i % len(CELLS)
        orders.append(CELLS[start:]+CELLS[:start])
        for style in (serial, parallel):
            style.append({'scenario': SCENARIO, 'pair': i+1, 'city': CITIES[i % len(CITIES)], 'arms': {}, 'status': 'missing'})
    rows = orders, serial, parallel
    report = {'scope': 'controller compaction with two real fixed-service-window read calls and mandatory all-of; not DSH',
              'cases': {'serial': serial, 'parallel': parallel}, 'window_calibration': SCENARIO}
    try:
        with tempfile.TemporaryDirectory(prefix=f'{a.name}-workspace-', dir=root / 'tmp') as scratch:
            directory = Path(scratch)
            for filename, content, _ in READS:
                (directory/filename).write_text(content)
            with engine(root, 'extended', a.name):
                path = asyncio.run(calibrate(root, a.name, [SCENARIO]))
                with ingress(root, a.name, 'lookahead-only', path) as token:
                    asyncio.run(measure(root, a.name, token, rows, directory))
                receipts, policies = engine_evidence(root/'logs'/f'{a.name}-ingress.log')
                verify_pairs([*serial, *parallel], receipts, policies)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['summary'] = compare(rows)
        write_json(root/f'{a.name}.json', report)
        print(report['summary'], flush=True)
    if not report['summary']['qualified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
