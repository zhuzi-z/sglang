"""Join published cuts into stage/chunk host-window samples. No torch imports."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def build_samples(root, expected_pcp):
    groups = defaultdict(lambda: {'spans': {}, 'inputs': None, 'layouts': {},
                                  'clock_ids': set(), 'issues': set()})
    manifests = list(Path(root).glob('**/exports/*/export.json'))
    for manifest in manifests:
        info = json.loads(manifest.read_text())
        clock = info.get('host_clock', {})
        clock_id = (clock.get('source'), clock.get('boot_id'), clock.get('time_namespace'))
        health = info.get('health', {})
        bad_health = any(health.get(key, 0) for key in
                         ('dropped_full', 'dropped_busy', 'rejected', 'observer_errors', 'failure'))
        for table, field in (('schedule_batch', 'spans'), ('chunk_inputs', 'inputs'), ('layouts', 'layouts')):
            path = manifest.parent / (table + '.jsonl')
            if not path.exists():
                continue
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    context = row.get('context')
                    if not context or len(context) != 4 or context[3] != 'real':
                        continue
                    rank = row['rank']
                    key = (*context[:3], row['chunk'], rank['pp'])
                    group = groups[key]
                    if bad_health:
                        group['issues'].add('process_health_not_clean')
                    if field == 'inputs':
                        data = {k: v for k, v in row.items()
                                if k not in ('mono_ns', 'seq', 'context', 'chunk', 'rank')}
                        if group['inputs'] is not None and group['inputs'] != data:
                            group['issues'].add('conflicting_input_snapshots')
                        group['inputs'] = data
                    else:
                        pcp = rank['pcp']
                        if pcp in group[field] and group[field][pcp] != row:
                            group['issues'].add('duplicate_rank_record')
                        group[field][pcp] = row
                        if field == 'spans':
                            group['clock_ids'].add(clock_id)
    samples = []
    for key, group in sorted(groups.items()):
        spans = group['spans']
        issues = group['issues']
        if set(spans) != set(range(expected_pcp)):
            issues.add('incomplete_pcp_spans')
        if any(not row.get('valid') for row in spans.values()):
            issues.add('invalid_host_span')
        if any(row.get('iter_latency') is not None and row['iter_latency'] < 0 for row in spans.values()):
            issues.add('negative_rank_span')
        clock_ok = len(group['clock_ids']) == 1 and all(next(iter(group['clock_ids']), (None,)))
        if not clock_ok:
            issues.add('unverified_shared_host_clock')
        inputs = group['inputs']
        if not inputs or not inputs.get('input_complete'):
            issues.add('missing_or_incomplete_inputs')
        layouts = group['layouts']
        if set(layouts) != set(range(expected_pcp)):
            issues.add('incomplete_pcp_layouts')
        if any(row.get('pcp_size') != expected_pcp for row in layouts.values()):
            issues.add('unexpected_pcp_size')
        counts = [tuple(row.get('local_valid_counts') or ()) for row in layouts.values()]
        if counts and (len(set(counts)) != 1 or len(counts[0]) != expected_pcp):
            issues.add('inconsistent_pcp_counts')
        if counts and inputs and sum(counts[0]) != inputs['total_query_tokens']:
            issues.add('query_sum_mismatch')
        starts = [row['start_mono_ns'] for row in spans.values() if row.get('start_mono_ns') is not None]
        ends = [row['end_mono_ns'] for row in spans.values() if row.get('end_mono_ns') is not None]
        span_ok = set(spans) == set(range(expected_pcp)) and all(row.get('valid') for row in spans.values())
        start = min(starts) if clock_ok and span_ok and len(starts) == expected_pcp else None
        end = max(ends) if clock_ok and span_ok and len(ends) == expected_pcp else None
        requests = inputs['requests'] if inputs else []
        attention_pairs = sum(r['query_length'] * r['past_length'] +
                              r['query_length'] * (r['query_length'] + 1) // 2
                              for r in requests) if requests and all(r.get('past_length') is not None for r in requests) else None
        samples.append({'collection_id': key[0], 'engine_session_id': key[1],
                        'step_id': key[2], 'chunk_id': key[3], 'pp_stage_id': key[4],
                        'time_kind': 'stage_host_main_forward_window',
                        'start_mono_ns': start, 'end_mono_ns': end,
                        'latency_ms': (end-start)/1e6 if start is not None and end is not None else None,
                        'rank_host_latency_ms': {rank: row['iter_latency']*1000 if row.get('iter_latency') is not None else None
                                                 for rank, row in spans.items()},
                        'inputs': inputs,
                        'causal_attention_token_pairs': attention_pairs,
                        'local_valid_counts': counts[0] if counts else None,
                        'valid_for_host_prediction': not issues,
                        'valid_for_device_prediction': False,
                        'device_start_ns': None, 'device_end_ns': None,
                        'issues': sorted(issues)})
    return samples, {'published_process_cuts': len(manifests), 'samples': len(samples),
                     'valid_host_samples': sum(s['valid_for_host_prediction'] for s in samples),
                     'valid_device_samples': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-pcp', type=int, required=True)
    args = parser.parse_args()
    if args.expected_pcp < 1:
        parser.error('--expected-pcp must be positive')
    samples, report = build_samples(args.input, args.expected_pcp)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as stream:
        for sample in samples:
            stream.write(json.dumps(sample) + '\n')
    output.with_suffix(output.suffix + '.report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return 0 if report['valid_host_samples'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
