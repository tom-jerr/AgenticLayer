"""Summarize fixed-graph FreeInference matrices from raw, arm-local receipts."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from .plot_ablation import evidence
from .plot_freeinference import load, ndjson

BUCKETS = list(range(1, 9))


def read_matrix(path):
    source = load(path)
    root = path.parent
    bars, runs, records = {}, {}, []
    capture_reference = None
    reference = None
    output_reference = None
    for i, arms in enumerate(source['run_summaries']):
        for arm, summary in arms.items():
            launch = summary['name']
            log_path = root/'logs'/f'{launch}.log'
            graph = load(root/'logs'/f'{launch}-graph.json')
            config = graph['cuda_graph_config']
            assert config['decode']['backend'] == 'full' and config['prefill']['backend'] == 'disabled'
            assert graph['captured_decode_buckets'] == config['decode']['bs'] == BUCKETS
            assert capture_reference is None or graph == capture_reference, f'{launch}: graph configuration drift'
            capture_reference = graph
            actual = log_path.read_text()
            gen_bs = Counter(int(b) for b in re.findall(r'Decode batch, #running-req: (\d+)[^\n]*cuda graph: True', actual))
            assert gen_bs and not any(b not in BUCKETS for b in gen_bs), f'{launch}: unverified Graph execution'
            assert not re.search(r'Prefill batch[^\n]*cuda graph: True', actual), launch
            report, work, _, _, m = evidence(root, launch)
            barrier_count, outputs = 0, {}
            for event in ndjson(root/'logs'/f'{launch}-timeline.jsonl'):
                if event['kind'] == 'tool-barrier':
                    barrier_count += 1
                elif event['kind'] == 'model' and event['status'] == 'completed':
                    outputs[json.dumps(event['identity'])] = event['response']['result']['output_ids']
            if output_reference is None:
                output_reference = outputs
            assert output_reference.keys() == outputs.keys()
            exact_outputs = sum(seq == output_reference[key] for key, seq in outputs.items())
            contract = (report['sample_sha256'], report['tested_turn_hashes'], work, barrier_count,
                        report.get('altered_barriers', ()))
            assert reference is None or contract == reference, f'{launch}: different workload'
            reference = contract
            if i == 0:
                runs[arm] = report
                bars[arm] = m
            prefix = 'agentic_layer:policy '
            decisions = Counter()
            controller = root/'logs'/f'{launch}-ingress.log'
            if controller.exists():
                with controller.open() as handle:
                    for line in handle:
                        if prefix in line:
                            row = json.loads(line.split(prefix, 1)[1]); decisions[(row['kind'], row['reason'])] += 1
            records.append(dict(matrix=path.stem, arm=arm, repeat=i+1, launch=launch, barrier_count=barrier_count,
                                exact_output_sequences_vs_first=exact_outputs,
                                graph_decode_logged_batch_sizes=json.dumps(gen_bs, sort_keys=True),
                                controller_decisions=json.dumps({f'{kind}/{reason}': count for (kind, reason), count in sorted(decisions.items())}),
                                **m))
    return capture_reference, bars, runs, records


def figures(root, bars, runs):
    n = len(bars)
    height = max(3.2, n*.38+0.6)
    fig, ax = plt.subplots(figsize=(8, height), layout='constrained')
    ax.barh(list(bars), [r['turn_p95_ms']/1000 for r in bars.values()], color='#2e797f')
    ax.set_xlabel('Active-turn P95 (seconds; first repeat)')
    ax.invert_yaxis()
    fig.savefig(root/'01_active_p95.svg')
    fig.savefig(root/'01_active_p95.png', dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
    for arm, result in runs.items():
        origin = min(row['arrival'] for row in result['turns'])
        completions = sorted(row['arrival'] - origin + row['wall_ms']/1000 for row in result['turns'])
        ax.step([0, *completions], range(len(completions)+1), label=arm, where='post')
    ax.set(xlabel='Time from first arrival (seconds; first repeat)', ylabel='Completed root turns')
    ax.legend(ncols=2 if n > 7 else 1, frameon=False, fontsize=8)
    fig.savefig(root/'02_completions.svg')
    fig.savefig(root/'02_completions.png', dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, height), layout='constrained')
    for i, m in enumerate(bars.values()):
        a, b = m['queue_min_ms'], m['queue_max_ms']
        ax.hlines(i, a, b, color='#2e797f')
        ax.plot([a, b], [i, i], '|', color='#2e797f')
        ax.scatter(m['queue_median_ms'], i, color='#2e797f')
    ax.set(yticks=range(n), yticklabels=list(bars), xlabel='Native queue min / median / max (ms; symlog, first repeat)')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(left=0)
    ax.invert_yaxis()
    fig.savefig(root/'03_scheduling_min_median_max.svg')
    fig.savefig(root/'03_scheduling_min_median_max.png', dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    plt.switch_backend('Agg')
    results, rows = {}, []
    for matrix in a.matrix:
        graph, bars, runs, records = read_matrix(matrix)
        results[matrix.stem] = graph, bars, runs
        rows.extend(records)
    graphs = [v[0] for v in results.values()]
    assert all(g['cuda_graph_config'] == graphs[0]['cuda_graph_config'] for g in graphs)
    a.out.mkdir(parents=True, exist_ok=False)
    with (a.out/'arms.csv').open('x', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for name, (_, bars, runs) in results.items():
        directory = a.out/name
        directory.mkdir()
        figures(directory, bars, runs)
    print(a.out)


if __name__ == '__main__':
    main()
