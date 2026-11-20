"""Fixed-Graph E4 fixture: rewrite the parent while its old prefix serves a live child."""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import httpx

from agentic_layer.engine_client import EngineClient

from .gpu_run import ENGINE, engine, token_fixture, write_json
from .hybrid_gpu import Driver, ident
from .state_ablation_gpu import allocation, checkpoint, generate, reset_cache, snapshot


async def sample(d, explicit, size, window_ms, hicache):
    baseline = allocation(await reset_cache(d))
    namespace = ident() + ident()
    old, new, child = (ident() for _ in range(3))
    prefix = token_fixture(size, 42)
    rewritten = token_fixture(256, 57) + prefix[256:]
    original, ready = await generate(d, prefix, namespace, old, n=1, hit=0)
    old_handle = checkpoint(ready, size)
    def command(op, version, handle, *, discard=False):
        return d.command(op, version=version, handle=handle,
                         payload={'discard_unbacked': True} if discard else None)
    lease, = await d.rpc(command('lease', old, old_handle))
    assert lease['status'] == 'ok' and lease['matched_tokens'] == size

    parent, new_receipt = await generate(d, rewritten, namespace, new, n=1, hit=0)
    checkpoint(new_receipt, size)
    # Parent replacement completed; its old state is now owned by the child.
    inactive = time.monotonic_ns()
    held = allocation(await snapshot(d, 'parent-rewritten-child-live'))
    blocked, = await d.rpc(command('reclaim', old, old_handle, discard=hicache))
    assert blocked['status'] == 'unavailable' and 'leased' in blocked['error'] and allocation(blocked) == held
    child_output, child_receipt = await generate(d, prefix + [6543], namespace, child, n=1, hit=size)
    leaf_handle = checkpoint(child_receipt, size + 1)
    leaf_free, release = await d.rpc(command('reclaim', child, leaf_handle, discard=hicache),
                                    command('release', old, old_handle))
    assert leaf_free['status'] == 'completed' and leaf_free['details']['freed_full_slots'] == 1, leaf_free
    assert leaf_free['details']['freed_recurrent_slots'] == 1 and release['status'] == 'ok'
    assert allocation(release) == held, (release, held)
    if hicache:
        conservative, = await d.rpc(command('reclaim', old, old_handle))
        assert conservative['status'] == 'unavailable' and 'unbacked' in conservative['error'], conservative
        assert allocation(conservative) == held
    boundary = time.monotonic_ns()
    comparison_end = boundary + window_ms * 1_000_000
    stored_bytes = (held['hbm_bytes'] - baseline['hbm_bytes']) // 2
    assert stored_bytes > 0 and held['hbm_bytes'] - baseline['hbm_bytes'] == stored_bytes * 2
    done = None
    if explicit:
        reclaim = command('reclaim', old, old_handle, discard=hicache)
        completed, = await d.rpc(reclaim)
        done = time.monotonic_ns()
        assert completed['status'] == 'completed' and completed['details']['freed_full_slots'] == size
        assert completed['details']['freed_recurrent_slots'] == 1 and completed['details']['residency'] == 'absent'
        assert completed['unique_host_bytes'] == held['host_bytes'] == baseline['host_bytes']
        assert held['hbm_bytes'] - completed['unique_hbm_bytes'] == stored_bytes
        duplicate, = await d.rpc(reclaim)
        assert duplicate == completed, 'duplicate operation must replay receipt, not free another leaf'
    await asyncio.sleep(max(0., (comparison_end - time.monotonic_ns()) / 1e9))
    after = allocation(await snapshot(d, 'fixed-observation-window'))
    assert after['hbm_bytes'] == held['hbm_bytes'] - (stored_bytes if explicit else 0)
    next_output, _ = await generate(d, rewritten + [6543], namespace, ident(), n=1, hit=size)
    exposed = (min(done, comparison_end) if done is not None else comparison_end) - boundary
    return {'explicit_reclaim': explicit, 'protected_remaining_bytes': stored_bytes,
            'live_child_ms': (boundary-inactive) / 1e6, 'expired_owned_byte_seconds': stored_bytes * exposed / 1e9,
            'expired_fw_ms': exposed / 1e6, 'before': held, 'after': after,
            'old_view': original, 'new_view': parent, 'child': child_output, 'parent_followup': next_output}


async def compare(root, name, repetitions, sizes, window_ms, hicache):
    control = EngineClient(root/'engine.sock')
    runs = []
    async with httpx.AsyncClient(base_url=ENGINE, trust_env=False, timeout=180) as http:
        with (root/'logs'/f'{name}-wire.jsonl').open('x', buffering=1) as log:
            d = Driver(control, http, log, name)
            await control.handshake()
            for size in sizes:
                for i in range(repetitions):
                    pair = {}
                    for extra in ((False, True) if i % 2 == 0 else (True, False)):
                        pair['reclaim' if extra else 'retain'] = await sample(d, extra, size, window_ms, hicache)
                    a, b = pair['retain'], pair['reclaim']
                    assert all(a[k]['output_ids'] == b[k]['output_ids']
                               for k in ('old_view', 'new_view', 'child', 'parent_followup')), pair
                    row = {'length': size, 'pair': i, **pair}
                    runs.append(row)
                    d.record('paired-sample', row)
            await reset_cache(d)
    scores = []
    for size in sizes:
        rows = [r for r in runs if r['length'] == size]
        retain, reclaim = (statistics.median(r[k]['expired_owned_byte_seconds'] for r in rows)
                           for k in ('retain', 'reclaim'))
        scores.append({'length': size, 'pairs': len(rows), 'retained_byte_seconds': retain,
                       'reclaimed_byte_seconds': reclaim, 'reduction_pct': 100 * (1 - reclaim/retain)})
    return {'observation_window_ms': window_ms, 'driver': 'native lease and ACK; authenticated wire, fixed branch fixture',
            'hicache': hicache,
            'summary': scores, 'pairs': runs}


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--run-dir', type=Path, required=True)
    args.add_argument('--name', required=True)
    args.add_argument('--repetitions', type=int, default=5)
    args.add_argument('--sizes', nargs='+', type=int, default=(4096, 16384))
    args.add_argument('--window-ms', type=int, default=600)
    args.add_argument('--hicache', action='store_true', help='verify explicit discard of an isolated unbacked leaf')
    a = args.parse_args()
    if not (a.repetitions > 0 and a.window_ms > 0 and all(256 < n <= 32764 for n in a.sizes)):
        args.error('invalid repetition count, window, or prefix length')
    with engine(a.run_dir, 'extended-hicache' if a.hicache else 'extended', a.name):
        result = asyncio.run(compare(a.run_dir, a.name, a.repetitions, a.sizes, a.window_ms, a.hicache))
    write_json(a.run_dir / f'{a.name}.json', result)
    print(result['summary'], flush=True)


if __name__ == '__main__':
    main()
