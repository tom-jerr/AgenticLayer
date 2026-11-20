"""Recompute a FreeInference GPU factorial from its arm reports and native timelines."""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .plot_freeinference import load, ndjson
from .trace_matrix import paired, percentile

ARM_ORDER = ('eager-l2', 'eager-l3', 'graph-l2', 'graph-l3')
COLORS = ('#52687c', '#267f79', '#b0642d', '#6d59a5')
CONTRASTS = (('checkpoints with eager decode', 'eager-l2', 'eager-l3'),
             ('checkpoints with graph decode', 'graph-l2', 'graph-l3'),
             ('graph with level 2', 'eager-l2', 'graph-l2'),
             ('graph with level 3', 'eager-l3', 'graph-l3'))


def evidence(root, launch):
    report = load(root / f'{launch}.json')
    work, cache = {}, {}
    latencies, queue, decode, peak = {}, [], [], {}
    for row in ndjson(root / 'logs' / f'{launch}-timeline.jsonl'):
        kind = row['kind']
        if kind == 'resource-sample':
            for key in ('full_token_usage', 'mamba_usage'):
                if key in row['metrics']:
                    value = row['metrics'][key]
                    peak[key] = max(value, peak.get(key, 0))
        if kind != 'model' or row['status'] != 'completed':
            continue
        week, session, path, step = row['identity']
        identity = week, session, tuple(map(tuple, path)), step
        assert identity not in work, identity
        body = row['response']['result']
        meta = body['meta_info']
        work[identity] = (meta['prompt_tokens'], meta['completion_tokens'])
        cache[identity] = meta['cached_tokens']
        latencies[identity] = meta['first_token_latency'] * 1000
        if 'queue_time' in meta:
            queue.append(meta['queue_time'] * 1000)
        if (a := meta.get('prefill_finished_time')) and (b := meta.get('request_finished_ts')) and b >= a:
            decode.append((b-a) * 1000)
    assert report['summary']['statuses'] == {'completed': len(report['turns'])} and work and queue and decode
    metrics = dict(requests=len(work), prompt_tokens=sum(v[0] for v in work.values()),
                   output_tokens=sum(v[1] for v in work.values()), cache_hits=sum(bool(v) for v in cache.values()),
                   cached_tokens=sum(cache.values()), ttft_median_ms=statistics.median(latencies.values()),
                   decode_phase_median_ms=statistics.median(decode), queue_samples=len(queue),
                   queue_min_ms=min(queue), queue_median_ms=statistics.median(queue), queue_max_ms=max(queue),
                   full_usage_sampled_peak=peak.get('full_token_usage'), mamba_usage_sampled_peak=peak.get('mamba_usage'),
                   replay_ms=report['replay_elapsed_ms'],
                   **{f'turn_{key}': value for key, value in report['summary'].items() if key.endswith('_ms')})
    return report, work, cache, latencies, metrics


def output_figures(directory, arms, order):
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6), layout='constrained')
    for ax, key, label, scale in zip(axes, ('turn_p50_ms', 'ttft_median_ms', 'cached_tokens'),
                                     ('Active turn P50 (s)', 'Engine TTFT median (ms)', 'Cached prompt tokens (k)'),
                                     (1000, 1, 1000), strict=True):
        ax.bar(order, [arms[name][4][key] / scale for name in order], color=COLORS[:len(order)])
        ax.set_ylabel(label)
        ax.tick_params(axis='x', labelrotation=25)
    fig.savefig(directory / '01_engine_ablation.svg')
    fig.savefig(directory / '01_engine_ablation.png', dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.6), layout='constrained')
    for name, color in zip(order, COLORS):
        report = arms[name][0]
        origin = min(turn['arrival'] for turn in report['turns'])
        finished = sorted(turn['arrival'] - origin + turn['wall_ms']/1000 for turn in report['turns'])
        ax.step([0, *finished, report['replay_elapsed_ms']/1000], [0, *range(1, len(finished)+1), len(finished)],
                where='post', label=name, color=color)
    ax.set(xlabel='Replay time from first arrival (s)', ylabel='Completed root turns')
    ax.legend(frameon=False)
    fig.savefig(directory / '02_ablation_completion.svg')
    fig.savefig(directory / '02_ablation_completion.png', dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.1), layout='constrained')
    for i, (name, color) in enumerate(zip(order, COLORS)):
        m = arms[name][4]
        low, high = m['queue_min_ms'], m['queue_max_ms']
        ax.hlines(i, low, high, color=color, linewidth=2)
        ax.plot([low, high], [i, i], '|', color=color)
        ax.scatter(m['queue_median_ms'], i, color=color, zorder=3)
    ax.set_xscale('symlog', linthresh=1)
    ax.set(yticks=range(len(order)), yticklabels=order, xlabel='SGLang queue min / median / max (ms; symlog)')
    ax.invert_yaxis()
    fig.savefig(directory / '03_ablation_queue_spread.svg')
    fig.savefig(directory / '03_ablation_queue_spread.png', dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--factorial', type=Path, required=True)
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    metadata = {arm['config']: arm for arm in load(a.factorial)['arms'] if arm['repeat'] == a.repeat}
    assert metadata, 'repeat absent from report'
    order = [name for name in ARM_ORDER if name in metadata]
    arms = {name: evidence(a.factorial.parent, metadata[name]['launch_name']) for name in order}
    reference = arms[order[0]]
    for report, work, *_ in arms.values():
        assert work == reference[1] and report['tested_turn_hashes'] == reference[0]['tested_turn_hashes']
    a.out.mkdir(exist_ok=False, parents=True)
    with (a.out / 'arms.csv').open('x', newline='') as dest:
        writer = csv.DictWriter(dest, fieldnames=['config', *reference[4]])
        writer.writeheader()
        writer.writerows(dict(config=name, **arm[4]) for name, arm in arms.items())
    with (a.out / 'effects.csv').open('x', newline='') as dest:
        writer = csv.writer(dest)
        writer.writerow(('single_toggle', 'off', 'on', 'paired_delta_p95_ms', 'ci95_low_ms', 'ci95_high_ms',
                         'clusters', 'delta_p50_ms', 'median_ttft_delta_ms_on_hit_requests'))
        for effect, off, on in CONTRASTS:
            if off not in arms or on not in arms:
                continue
            before, _, cache_before, ttft_before, m0 = arms[off]
            after, _, cache_after, ttft_after, m1 = arms[on]
            result = paired([{'strong': before, 'joint': after}])
            extra_hits = [k for k, n in cache_after.items() if n > cache_before[k]]
            hit_delta = statistics.median(ttft_after[k] - ttft_before[k] for k in extra_hits) if extra_hits else None
            writer.writerow((effect, off, on, result['joint_minus_strong_p95_ms'], *result['ci95_ms'], result['clusters'],
                             m1['turn_p50_ms'] - m0['turn_p50_ms'], hit_delta))
    output_figures(a.out, arms, order)
    print(a.out)


if __name__ == '__main__':
    main()
