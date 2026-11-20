"""Render thesis metrics from FreeInference GPU replay timelines and source profile."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ARMS = ('native-radix', 'native-hicache', 'strong', 'joint')
COLORS = dict(zip(ARMS, ('#516478', '#21877b', '#246bb5', '#c76b24'), strict=True))
QUANTILES = {'p50': .50, 'p95': .95, 'p99': .99}
RESOURCE_GAUGES = ('full_token_usage', 'mamba_usage', 'hicache_host_used_tokens', 'hicache_host_mamba_used_states',
                   'num_running_reqs', 'num_queue_reqs')
RESOURCE_COUNTERS = ('hicache_backup_bytes_total', 'load_back_bytes_total', 'evicted_tokens_total',
                     'hicache_backup_duration_seconds_sum', 'load_back_duration_seconds_sum')
SCHEDULING = {'engine_queue_ms': 'SGLang scheduler queue', 'batch_ready_ack_ms': 'Model permit RTT',
              'tool_ready_ack_ms': 'Tool-barrier permit RTT', 'policy_decision_ms': 'Policy CPU wall span'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def ndjson(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            yield json.loads(line)


def quantile(numbers, q):
    values = sorted(numbers)
    if not values:
        return None
    value = values[math.ceil(q * len(values)) - 1]
    return value if math.isfinite(value) else None


def spread(numbers):
    return {'n': len(numbers), 'min': min(numbers, default=None),
            'median': statistics.median(numbers) if numbers else None, 'max': max(numbers, default=None),
            **{k: quantile(numbers, q) for k, q in QUANTILES.items()}}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def add(values, key, value, scale=1):
    if finite(value) and value >= 0:
        values[key].append(value * scale)


def difference(meta, end, start):
    a, b = meta.get(end), meta.get(start)
    return a-b if finite(a) and finite(b) and a >= b > 0 else None


def collect(run, matrices):
    evidence = defaultdict(list)
    for matrix in matrices:
        for repeat in load(matrix)['run_summaries']:
            for arm, summary in repeat.items():
                evidence[arm].append((summary, load(run / f"{summary['name']}.json")))
    if not evidence:
        raise ValueError('no GPU replay runs in matrix')
    model_reference = None
    source_reference = None
    result = {}
    for arm in ARMS:
        if arm not in evidence:
            continue
        values, statuses, resource = defaultdict(list), Counter(), defaultdict(list)
        model_work, missing_work, run_names, completions_by_run, turns_count = [], [], [], {}, 0
        for original_summary, report in evidence[arm]:
            name = report['name']
            run_names.append(name)
            turns = report['turns']
            source = (report['sample_sha256'], sorted(report['tested_turn_hashes']),
                      sorted((tuple(row['identity']), row['stratum']) for row in turns))
            if source_reference is None:
                source_reference = source
            elif source_reference != source:
                raise ValueError(f'{name} did not use the paired source and turns')
            counts = Counter(t['status'] for t in turns)
            if counts != original_summary['statuses'] or len(turns) != len(report['tested_turn_hashes']):
                raise ValueError(f'{name} has inconsistent turn counts')
            statuses.update(counts)
            turns_count += len(turns)
            latencies = []
            origin = min(turn['arrival'] for turn in turns)
            completed_at = []
            for turn in turns:
                wall = turn['wall_ms']
                if turn['status'] == 'completed':
                    if not finite(wall) or wall < 0:
                        raise ValueError(f'{name}: completed turn has invalid wall_ms')
                    completed_at.append(turn['arrival'] - origin + wall/1000)
                    values['successful_active_turn_ms'].append(wall)
                    latencies.append(wall)
                else:
                    latencies.append(math.inf)
            completions_by_run[name] = sorted(completed_at)
            values['active_turn_all_ms'].extend(latencies)
            for k, q in QUANTILES.items():
                if quantile(latencies, q) != original_summary[f'{k}_ms']:
                    raise ValueError(f'{name}: {k} is not the all-submit nearest-rank percentile')

            per_run, request_work, ready = defaultdict(list), {}, {}
            resource_samples = resource_errors = 0
            for row in ndjson(run / 'logs' / f'{name}-timeline.jsonl'):
                kind = row['kind']
                if kind == 'model-ready':
                    ready[row['rid']] = row['mono_ns']
                elif kind == 'model' and row['status'] == 'completed':
                    data = row['response']['result']
                    m = data['meta_info']
                    tokens = m['completion_tokens']
                    if len(data['output_ids']) != tokens:
                        raise ValueError(f'{name}: GPU completion length differs')
                    identity = json.dumps(row['identity'], separators=(',', ':'))
                    if identity in request_work:
                        raise ValueError(f'{name}: duplicate model request identity: {identity}')
                    request_work[identity] = (m['prompt_tokens'], tokens)
                    add(values, 'engine_queue_ms', m.get('queue_time'), 1000)
                    add(values, 'engine_ttft_ms', m.get('first_token_latency'), 1000)
                    add(values, 'engine_prefill_span_ms', difference(m, 'prefill_finished_time', 'forward_entry_time'), 1000)
                    add(values, 'post_prefill_to_finish_ms', difference(m, 'request_finished_ts', 'prefill_finished_time'), 1000)
                    start = ready.pop(row['rid'], None)
                    if start is not None:
                        add(values, 'model_ready_to_replay_done_ms', row['mono_ns']-start, 1e-6)
                    add(values, 'cached_prompt_tokens', m.get('cached_tokens'))
                    if finite(m.get('cached_tokens')):
                        add(values, 'uncached_prompt_tokens', m['prompt_tokens']-m['cached_tokens'])
                    detail = m.get('cached_tokens_details') or {}
                    if 'host' in detail:
                        add(values, 'host_cached_tokens_reported', detail['host'])
                elif kind == 'ready-ack':
                    add(values, f"{row['action_kind']}_ready_ack_ms", row['duration_ms'])
                elif kind == 'tool-barrier':
                    add(values, 'replayed_tool_barrier_ms', row['duration_ms'])
                elif kind == 'resource-sample':
                    resource_samples += 1
                    for key, value in row['metrics'].items():
                        add(per_run, key, value)
                elif kind == 'resource-error':
                    resource_errors += 1
            if report['mode'] == 'controlled':
                ingress = run / 'logs' / f'{name}-ingress.log'
                marker, receipt_marker = 'agentic_layer:policy ', 'agentic_layer:engine receipt '
                if ingress.is_file():
                    with ingress.open(encoding='utf-8') as handle:
                        for line in handle:
                            if marker in line:
                                decision = json.loads(line.split(marker, 1)[1])
                                if decision['reason'] == 'selected':
                                    add(values, 'policy_decision_ms', decision.get('decision_ms'))
                            elif receipt_marker in line:
                                receipt = json.loads(line.split(receipt_marker, 1)[1])
                                detail = receipt.get('details', {})
                                if 'full_kv_hit_length' in detail:
                                    add(values, 'engine_full_match_tokens', detail['full_kv_hit_length'])
                                    add(values, 'engine_hybrid_match_tokens', receipt.get('matched_tokens'))
            if counts.get('completed', 0) == len(turns):
                if model_reference is None:
                    model_reference = request_work
                elif model_reference != request_work:
                    raise ValueError(f'{name}: model workload differs despite completed paired turns')
            else:
                missing_work.append(name)
            model_work.extend(request_work.values())
            for key in RESOURCE_GAUGES:
                if per_run[key]:
                    resource[f'observed_peak_{key}'].append(max(per_run[key]))
            for key in RESOURCE_COUNTERS:
                if len(per_run[key]) >= 2:
                    a, b = per_run[key][0], per_run[key][-1]
                    if b < a:
                        raise ValueError(f'{name}: counter {key} reset during replay')
                    resource[f'observed_delta_{key}'].append(b-a)
            resource['metric_samples'].append(resource_samples)
            resource['metric_errors'].append(resource_errors)
            elapsed = report.get('replay_elapsed_ms')
            if finite(elapsed) and elapsed > 0:
                add(values, 'completed_output_tokens_per_s', sum(t[1] for t in request_work.values()) / (elapsed / 1000))
        cached_count = len(values['cached_prompt_tokens'])
        prompt_total = sum(k[0] for k in model_work)
        result[arm] = {'runs': run_names, 'turns': turns_count, 'statuses': dict(statuses),
                       'completion_offsets_s_from_first_arrival': completions_by_run,
                       'model_requests_completed': len(model_work), 'prompt_tokens': prompt_total,
                       'output_tokens': sum(k[1] for k in model_work), 'incomplete_workload_runs': missing_work,
                       'cached_prompt_pct': 100*sum(values['cached_prompt_tokens']) / prompt_total
                           if prompt_total and cached_count == len(model_work) else None,
                       'metrics': {key: spread(vals) for key, vals in values.items() if key != 'active_turn_all_ms'},
                       'active_turn_all_ms': {k: quantile(values['active_turn_all_ms'], q) for k, q in QUANTILES.items()},
                       'sampled_resources': {key: spread(rows) for key, rows in resource.items()}}
    if source_reference is None:
        raise ValueError('no recognized strategy in matrix')
    return {'sample_sha256': source_reference[0], 'selected_turn_hashes': source_reference[1], 'arms': result,
            'paper_sections': ['3.6.2', '4.6.2', '5.6.2', '5.6.4'],
            'units': {'engine_queue_ms': 'scheduler native queue (not GPU compute)',
                      'batch_ready_ack_ms': 'replay ready POST to permit-receipt, including HTTP',
                      'tool_ready_ack_ms': 'replay tool-barrier ready POST to permit-receipt, including HTTP',
                      'policy_decision_ms': 'controller candidate assembly and policy selection; only passes with selected actions',
                      'engine_prefill_span_ms': 'native wall from forward entry to prefill finish; may include gaps',
                      'post_prefill_to_finish_ms': 'native wall including decode and API drain, not isolated kernel time',
                      'resources': 'sampled SGLang gauge peaks and observed counter deltas; not allocator byte peaks'},
            'unmeasured': ['true tool service or all-of exposed wait', 'task quality and compaction semantics',
                           'exact unique HBM/DRAM byte-seconds', 'dedicated GPU kernel-time breakdown']}


def save(fig, out, name):
    for ext in ('png', 'svg'):
        fig.savefig(out / f'{name}.{ext}', dpi=180)
    plt.close(fig)


def draw_bars(data, out):
    arms = data['arms']
    fig, ax = plt.subplots(figsize=(8.5, 4.6), layout='constrained')
    width = .78 / len(arms)
    for i, (arm, report) in enumerate(arms.items()):
        xs = [j + width*(i-(len(arms)-1)/2) for j in range(3)]
        values = [report['active_turn_all_ms'][k] for k in QUANTILES]
        first = True
        for x, y in zip(xs, values, strict=True):
            label = f"{arm} (n={report['turns']})" if first else None
            first = False
            if y is None:
                ax.plot(x, 0, 'x', color=COLORS[arm], clip_on=False, label=label)
                ax.annotate('not reached', (x, 0), xytext=(0, 5), textcoords='offset points',
                            ha='center', rotation=75, fontsize=8)
            else:
                ax.bar(x, y/1000, width*.93, color=COLORS[arm], label=label)
    ax.set(title='FreeInference held-out replay: active-turn latency', ylabel='Wall time (s)',
           ylim=(0, None), xticks=range(3), xticklabels=('P50', 'P95', 'P99'))
    fig.legend(loc='outside upper center', ncol=min(4, len(arms)), fontsize=9, frameon=False)
    save(fig, out, '01_active_turn_latency')


def draw_weeks(profile, out):
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    weeks = profile['weeks']
    for key, label, color in (('root_sessions', 'Root sessions', '#516478'), ('root_turns', 'Root turns', '#21877b'),
                               ('requests', 'Model requests', '#246bb5'), ('tool_calls', 'Tool calls', '#c76b24')):
        series = [profile['week_counts'][w][key] for w in weeks]
        if sum(series) != profile['counts'][key] or min(series) <= 0:
            raise ValueError(f'{key}: invalid weekly totals')
        ax.plot(range(len(weeks)), series, '.-', color=color, label=label)
    for key, color in (('calibration_weeks', '#246bb5'), ('heldout_weeks', '#c76b24')):
        idx = [weeks.index(w) for w in profile[key]]
        ax.axvspan(min(idx)-.35, max(idx)+.35, alpha=.08, color=color)
    ax.set(title='FreeInference dataset: 16-week workload volume (not GPU timing)', ylabel='Records per week (log scale)',
           yscale='log', xticks=range(len(weeks)), xticklabels=[w[5:10] for w in weeks], xlabel='Week start (MM-DD, 2026)')
    ax.tick_params(axis='x', labelrotation=45)
    ax.legend(ncol=2, fontsize=9)
    save(fig, out, '02_weekly_workload')


def draw_scheduling(data, out):
    panels = {metric: label for metric, label in SCHEDULING.items()
              if metric == 'engine_queue_ms' or any(metric in arm['metrics'] for arm in data['arms'].values())}
    fig, axes = plt.subplots(1, len(panels), figsize=(3.15*len(panels)+1.1, 4.2),
                             sharey=True, squeeze=False, layout='constrained')
    axes = axes[0]
    for ax, (key, title) in zip(axes, panels.items(), strict=True):
        for y, (arm, report) in enumerate(data['arms'].items()):
            s = report['metrics'].get(key)
            if not s or not s['n']:
                ax.text(.02, y, 'n/a', va='center', color='#777777', transform=ax.get_yaxis_transform())
                continue
            color = COLORS[arm]
            ax.hlines(y, s['min'], s['max'], color=color, linewidth=2)
            ax.plot([s['min'], s['max']], [y, y], '|', color=color, markersize=11)
            ax.plot(s['median'], y, 'o', color=color, markersize=5)
            ax.annotate(f"n={s['n']}", (.98, y), xycoords=ax.get_yaxis_transform(),
                        xytext=(-1, 6), textcoords='offset points', ha='right', fontsize=8,
                        color=color)
        ax.set(title=title, xlabel='Milliseconds (symlog)')
        ax.set_xscale('symlog', linthresh=.1)
        ax.set_xlim(left=0)
    labels = [arm for arm in data['arms']]
    axes[0].set(yticks=range(len(labels)), yticklabels=labels, ylim=(len(labels)-.5, -.5))
    fig.suptitle('Scheduling latency: minimum, median and maximum')
    fig.legend(handles=[Line2D([0], [0], color='#555555', marker='|', label='Min–max'),
                        Line2D([0], [0], color='#555555', marker='o', linestyle='none', label='Median')],
               loc='outside lower center', ncol=2, frameon=False, fontsize=9)
    save(fig, out, '03_scheduling_min_median_max')


def draw_completions(data, out):
    fig, ax = plt.subplots(figsize=(8.5, 4.7), layout='constrained')
    for arm, report in data['arms'].items():
        repeats = report['completion_offsets_s_from_first_arrival']
        for i, times in enumerate(repeats.values()):
            ax.step([0, *times], range(len(times)+1), where='post', color=COLORS[arm],
                    label=arm if i == 0 else None, alpha=.85 if len(repeats) == 1 else .48)
    ax.set(title='GPU replay: cumulative active-turn completions', ylabel='Completed turns',
           xlabel='Time since first projected arrival (s)', xlim=(0, None), ylim=(0, None))
    ax.legend(ncol=2, fontsize=9)
    save(fig, out, '04_gpu_completion_progress')


def write_metrics(data, dest):
    with dest.open('w', newline='', encoding='utf-8') as handle:
        w = csv.writer(handle)
        w.writerow(('arm', 'metric', 'n', 'min', 'median', 'max', 'p50', 'p95', 'p99'))
        for arm, report in data['arms'].items():
            w.writerow((arm, 'active_turn_all_ms', report['turns'], None, None, None,
                        *(report['active_turn_all_ms'][p] for p in QUANTILES)))
            for metric, row in {**report['metrics'], **report['sampled_resources']}.items():
                w.writerow((arm, metric, *(row[k] for k in ('n', 'min', 'median', 'max', *QUANTILES))))
            for metric in ('model_requests_completed', 'prompt_tokens', 'output_tokens', 'cached_prompt_pct'):
                w.writerow((arm, metric, None, None, report[metric], None, None, None, None))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--matrix', type=Path, action='append', required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    data = collect(a.run_dir, a.matrix)
    profile = load(a.profile)
    a.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'axes.grid': True, 'axes.axisbelow': True, 'grid.alpha': .25,
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    draw_bars(data, a.out)
    draw_weeks(profile, a.out)
    draw_completions(data, a.out)
    draw_scheduling(data, a.out)
    write_metrics(data, a.out / 'summary.csv')
    with (a.out / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')
    print({arm: (report['statuses'], report['active_turn_all_ms']) for arm, report in data['arms'].items()})


if __name__ == '__main__':
    main()
