"""Freeze supported GPU cost buckets and held-in FreeInference barrier samples."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def file_digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def local_waits(requests):
    for request in requests:
        tools = request['tools']
        # Actual completion of projected children additionally depends on our GPU.
        # Do not use historical tool clocks as measurements of that service time.
        if tools and all(t['subagent_session_id'] is None for t in tools):
            duration = request['barrier_ms']
            # The replay itself stops this wait at the fixed bound. A recorded
            # janky or missing latency was normalized to 0 by the projector.
            if duration > 0 and all(t['latency_ms'] > 0 for t in tools):
                yield {'duration_ms': duration, 'completed': True}
        for tool in tools:
            yield from local_waits(tool['subagent'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--copy', type=Path, required=True)
    p.add_argument('--sample', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.sample.with_suffix('.manifest.json').read_text())
    sample_hash = file_digest(a.sample)
    if manifest['split'] != 'calibration' or sample_hash != manifest['selected_rows_sha256']:
        raise ValueError('expected a hash-matched calibration split, not heldout')
    waits, count = [], 0
    with a.sample.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['week'] not in manifest['weeks']:
                raise ValueError('request outside frozen calibration weeks')
            waits.extend(local_waits(row['requests']))
            count += 1
    if count != manifest['selected_count'] or len(waits) < 10:
        raise ValueError('insufficient frozen trace wait evidence')
    rows = json.loads(a.copy.read_text())['arms']['native']['samples']
    buckets = []
    for length in (4096, 16384):
        samples = [row for row in rows if row['length'] == length and not row['warmup']]
        if len(samples) < 10:
            raise ValueError(f'under-sampled GPU cost bucket {length}')
        for sample in samples:
            if not all(sample[d]['ack_metrics_complete'] and sample[d]['physical_ms'] > 0 for d in ('d2h', 'h2d')):
                raise ValueError(f'incomplete physical transfer in GPU cost bucket {length}')
        buckets.append({'token_count': length, 'batch_size': 1, 'samples': len(samples),
                        'prefill_ms': statistics.median(r['prefill_ms'] for r in samples),
                        'd2h_ms': statistics.median(r['d2h']['physical_ms'] for r in samples),
                        'h2d_ms': statistics.median(r['h2d']['physical_ms'] for r in samples),
                        'recurrent_ms': 0.0})
    data = {'buckets': buckets, 'waits': {'trace_barrier': waits}, 'evictions': {},
            'provenance': {'copy_sha256': file_digest(a.copy), 'projected_trace_sha256': sample_hash,
                           'weeks': manifest['weeks'], 'turns': count, 'local_wait_samples': len(waits),
                           'recurrent_ms': 'not timed separately: already included in measured hybrid restore',
                           'prefill_ms': 'bind-to-terminal prefill wall, not GPU kernel time',
                           'evictions': 'unmeasured on held-in replay; optional background movement remains gated',
                           'coverage': 'TP1 single request only; no extrapolation beyond measured buckets'}}
    with a.out.open('x') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print({'waits': len(waits), 'buckets': buckets, 'calibration_sha256': file_digest(a.out)})


if __name__ == '__main__':
    main()
