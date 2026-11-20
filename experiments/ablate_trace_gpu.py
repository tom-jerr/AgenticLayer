"""Cross decode graph capture and recurrent checkpoint density on identical trace turns."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

from .gpu_run import Probe, engine, token_fixture, write_json
from .trace_replay import run

CONFIGS = {'eager-l2': ('disabled', 2), 'eager-l3': ('disabled', 3),
           'graph-l2': ('full', 2), 'graph-l3': ('full', 3)}


def model_evidence(path):
    result, values = {}, defaultdict(list)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['kind'] != 'model' or row['status'] != 'completed':
                continue
            body = row['response']['result']
            meta = body['meta_info']
            identity = json.dumps(row['identity'], separators=(',', ':'))
            if identity in result:
                raise RuntimeError(f'duplicate request {identity}')
            result[identity] = {'work': (meta['prompt_tokens'], meta['completion_tokens']), 'output': body['output_ids']}
            values['ttft_ms'].append(meta['first_token_latency'] * 1000)
            values['cached_tokens'].append(meta['cached_tokens'])
            first, last = meta.get('prefill_finished_time'), meta.get('request_finished_ts')
            if first and last and last >= first:
                values['prefill_finish_to_request_finish_ms'].append((last-first)*1000)
    return result, {'requests': len(result), 'cached_prompt_tokens': sum(values['cached_tokens']),
                    'requests_with_hits': sum(n > 0 for n in values['cached_tokens']),
                    'median_ms': {k: statistics.median(v) for k, v in values.items() if k.endswith('_ms')}}


def same_tokens(reference, changed):
    assert reference.keys() == changed.keys(), 'different completed request identities'
    same = 0
    for key, old in reference.items():
        new = changed[key]
        assert old['work'] == new['work'], (key, old['work'], new['work'])
        same += old['output'] == new['output']
    return same


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--sample', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--configs', nargs='+', choices=CONFIGS, default=list(CONFIGS))
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--turns-per-stratum', type=int, default=3)
    p.add_argument('--max-turn-decode', type=int, default=768)
    a = p.parse_args()
    if min(a.repeats, a.turns_per_stratum, a.max_turn_decode) < 1 or len(a.configs) != len(set(a.configs)):
        p.error('positive counts and unique configurations required')
    root = a.run_dir.resolve()
    report = {'sample': str(a.sample.resolve()), 'arms': [], 'diagnosis':
              'single SGLang engine per arm; matched Full and Mamba pool capacities, two orthogonal toggles'}
    sample, numerical_reference, capacity_reference = None, None, None
    for i in range(a.repeats):
        shift = i % len(a.configs)
        for label in a.configs[shift:] + a.configs[:shift]:
            backend, checkpoint = CONFIGS[label]
            name = f'{a.name}-{i+1}-{label}'
            os.environ.update(AGENTIC_DECODE_GRAPH=backend, AGENTIC_CHECKPOINT_LEVEL=str(checkpoint),
                              AGENTIC_FULL_TOKENS='278241', AGENTIC_MAMBA_SLOTS='157')
            with engine(root, 'native-radix', name) as client:
                warm = Probe(client, root/'logs'/f'{name}-numerics.jsonl')
                try:
                    info = warm.get('/server_info').json()
                    assert info['hybrid_cache_checkpoints_level'] == checkpoint, info['hybrid_cache_checkpoints_level']
                    warm.gen('context-32766+2', token_fixture(32766, 47), 2, logprob=-1)
                    # Cold stable shape under every backend before the measured namespace.
                    nums = []
                    for length in (512, 4096, 16384):
                        cold, _ = warm.gen('isolated-numeric', token_fixture(length, 43), 8, logprob=-1)
                        nums.append({'ids': cold['output_ids'], 'logprobs': [v[0] for v in cold['meta_info']['output_token_logprobs']]})
                    warm.get('/flush_cache?timeout=5')
                finally:
                    warm.close()
                current = asyncio.run(run(a.sample, root, name, None, False,
                                          turns_per_stratum=a.turns_per_stratum, max_turn_decode=a.max_turn_decode))
            if current['summary']['statuses'] != {'completed': len(current['turns'])}:
                raise RuntimeError(('incomplete real trace', name, current['summary']))
            hashes = current['tested_turn_hashes']
            model, stats = model_evidence(root/'logs'/f'{name}-timeline.jsonl')
            if not model:
                raise RuntimeError(f'no completed model requests in {name}')
            if sample is None:
                sample = hashes, model
                numerical_reference = nums
            assert sample[0] == hashes, 'different frozen root turns'
            ids_same = all(x['ids'] == y['ids'] for x, y in zip(numerical_reference, nums, strict=True))
            logprob_delta = max(abs(v-w) for x, y in zip(numerical_reference, nums, strict=True)
                                for v, w in zip(x['logprobs'], y['logprobs'], strict=True))
            engine_log = (root/'logs'/f'{name}.log').read_text()
            slots = (int(re.findall(r'KV Cache is allocated[^\n]*#tokens: (\d+)', engine_log)[-1]),
                     int(re.findall(r'Mamba Cache is allocated[^\n]*max_mamba_cache_size: (\d+)', engine_log)[-1]))
            capacity_reference = capacity_reference or slots
            assert slots == capacity_reference == (278241, 157), (slots, capacity_reference)
            graph_batches = len(re.findall(r'Decode batch[^\n]*cuda graph: True', engine_log))
            if backend == 'full' and not graph_batches:
                raise RuntimeError(f'{name}: no observed CUDA graph decode batches')
            item = {'config': label, 'repeat': i+1, 'launch_name': name, 'decode_graph': backend,
                    'checkpoint_level': checkpoint, 'decode_log_samples_with_graph': graph_batches,
                    'pool_capacity': {'full_tokens': slots[0], 'mamba_states': slots[1]},
                    'summary': current['summary'], 'elapsed_ms': current['replay_elapsed_ms'], 'model': stats,
                    'exact_trace_sequences_vs_first': same_tokens(sample[1], model),
                    'isolated_greedy_same_vs_first': ids_same, 'isolated_max_logprob_delta_vs_first': logprob_delta}
            print(item, flush=True)
            report['arms'].append(item)
    write_json(root/f'{a.name}.json', report)


if __name__ == '__main__':
    main()
