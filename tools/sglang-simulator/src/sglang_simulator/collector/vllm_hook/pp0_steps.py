"""PP0-only host boundary collection and offline, clock-checked step joins.

No device synchronization, future polling, tensors, hardware profiler or hot-path
file I/O. Full-step tables are built OFFLINE across the PP0 worker processes.
"""
import json
from pathlib import Path
import time


def keep_event(kind, identity):
    if kind.startswith('startup.') or kind in ('native.boundary_installed', 'native.boundary_unavailable'):
        return True
    pp, pcp = identity[3], identity[4]
    if pp == 0:
        return (kind in ('worker.execute_model.enter', 'native.local_drain.ready',
                         'native.handoff_submitted', 'native.completion_unresolved')
                or (kind.startswith('native.') and kind.endswith('.error'))
                or (kind == 'chunk.inputs' and pcp == 0))
    if pp == -1:
        return kind in ('step.inputs', 'request.add', 'request.scheduled', 'pp0.result_ready')
    return False


def observe_result(stepper, event_kind='pp0.result_ready'):
    """Attach AFTER native dispatch; inspect only a future that is already done.

    _sampled fires for set_exception too, so its callback alone cannot prove a
    successful model result. Capture our own successful-future observation.
    """
    from . import trace_v2 as trace
    step = stepper.in_flight[-1]
    slot = getattr(step, 'slot', None)
    context = getattr(slot, '_sim_collection_context', None)
    if context is None:
        return
    def completed(future):
        stamp = time.monotonic_ns()
        try:
            # done callbacks run after set_result/set_exception. Never wait.
            if not future.done():
                return
            failure = future.exception()
            success = failure is None and future.result() is not None
            trace.emit(event_kind, trace.Scope(context), data={
                'observed_mono_ns': stamp, 'success': success,
                'error_type': type(failure).__name__ if failure else None,
                'time_kind': 'host_successful_sample_future_observed'})
        except Exception:
            if trace._RECORDER is not None:
                trace._RECORDER.observer_errors += 1
    step.future.add_done_callback(completed)


def export_tables(window):
    names = ('requests', 'step_inputs', 'chunk_inputs', 'pp0_points')
    streams = {n: (window/(n+'.jsonl')).open('w') for n in names}
    counts = dict.fromkeys(names, 0)
    try:
        for source in sorted(window.glob('events.*.jsonl')):
            with source.open() as f:
                for line in f:
                    row = json.loads(line)
                    kind = row['event']
                    if kind.startswith('request.'):
                        name = 'requests'
                    elif kind == 'step.inputs':
                        name = 'step_inputs'
                    elif kind == 'chunk.inputs':
                        name = 'chunk_inputs'
                    elif kind in ('worker.execute_model.enter', 'native.local_drain.ready',
                                  'native.handoff_submitted', 'pp0.result_ready',
                                  'native.completion_unresolved') or kind.endswith('.error'):
                        name = 'pp0_points'
                    else:
                        continue
                    streams[name].write(json.dumps(row, separators=(',', ':'))+'\n')
                    counts[name] += 1
    finally:
        for stream in streams.values():
            stream.close()
    return counts


def build_steps(sources, expected_pcp=8):
    """sources=[(process.json, events)]. Incomplete/cross-clock rows stay invalid.

    Carries spans across export cuts by joining collection+engine+step context.
    A shared batch duration is NOT a per-request isolated service time.
    """
    groups, seen = {}, set()
    for meta, events in sources:
        clock = (meta.get('clock_domain'), meta.get('host_clock', {}).get('boot_id'),
                 meta.get('host_clock', {}).get('time_namespace'))
        for e in events:
            context = e.get('context')
            if not context or context[3] != 'real':
                continue
            unique = (meta['process_id'], e['seq'])
            if unique in seen:
                continue
            seen.add(unique)
            groups.setdefault(tuple(context), []).append((e, clock))
    results = []
    for context, records in sorted(groups.items()):
        issues, starts, ends, sends, chunks = [], {}, {}, {}, {}
        step_input, result, clocks = None, None, set()
        for e, clock in records:
            kind, data, rank = e['event'], e.get('data') or {}, e['rank']
            pcp, chunk = rank['pcp'], e['chunk']
            if kind in ('step.inputs', 'pp0.result_ready') or (rank['pp'] == 0 and kind in (
                    'worker.execute_model.enter', 'native.local_drain.ready', 'native.handoff_submitted', 'chunk.inputs')):
                clocks.add(clock)
            if kind == 'step.inputs':
                step_input = data
            elif kind == 'pp0.result_ready':
                result = data
            elif rank['pp'] != 0:
                continue
            elif kind == 'worker.execute_model.enter':
                if pcp in starts:
                    issues.append('duplicate_worker_start')
                starts[pcp] = e['mono_ns']
            elif kind in ('native.local_drain.ready', 'native.handoff_submitted'):
                dest = ends if kind == 'native.local_drain.ready' else sends
                if (chunk, pcp) in dest:
                    issues.append('duplicate_chunk_boundary')
                dest[chunk, pcp] = data['observed_mono_ns']
            elif kind == 'chunk.inputs':
                if chunk in chunks:
                    issues.append('duplicate_chunk_input')
                chunks[chunk] = data
            elif kind == 'native.completion_unresolved' or kind.endswith('.error'):
                issues.append(kind)
        expected = set(range(expected_pcp))
        if set(starts) != expected:
            issues.append('missing_worker_start')
        if not chunks or set(chunks) != set(range(max(chunks, default=-1)+1)):
            issues.append('missing_chunk_input')
        expected_pairs = {(c, p) for c in chunks for p in expected}
        if set(ends) != expected_pairs or not expected_pairs:
            issues.append('incomplete_local_completion')
        if set(sends) != expected_pairs or not expected_pairs:
            issues.append('incomplete_handoff_submission')
        if step_input is None or not step_input.get('native_dispatch_ok'):
            issues.append('missing_or_failed_step_input')
        if result is None or not result.get('success'):
            issues.append('missing_or_failed_result')
        if len(clocks) != 1 or not all(all(x) for x in clocks):
            issues.append('unverified_clock_domain')
        requests = [r for c in sorted(chunks) for r in chunks[c].get('requests', ())]
        if not requests or any(any(r.get(k) is None for k in ('req_id', 'query_length', 'past_length', 'input_length')) for r in requests):
            issues.append('incomplete_request_features')
        by_req = {}
        for r in requests:
            by_req[r['req_id']] = by_req.get(r['req_id'], 0) + (r.get('query_length') or 0)
        if step_input and by_req != {r['req_id']: r['query_length'] for r in step_input['requests']}:
            issues.append('chunk_step_token_mismatch')
        if any(not c.get('input_complete') for c in chunks.values()):
            issues.append('chunk_input_incomplete')
        t1 = min(starts.values(), default=None)
        t2 = max(ends.values(), default=None)
        handoff = max(sends.values(), default=None)
        t3 = (result or {}).get('observed_mono_ns')
        if t1 is not None and t2 is not None and t3 is not None and not t1 <= t2 <= t3:
            issues.append('timestamp_order_invalid')
        if any(ends.get((c,p), 0) < starts.get(p, 0) for c,p in ends):
            issues.append('rank_completion_before_start')
        if any(sends.get(pair, 0) < stamp for pair, stamp in ends.items()):
            issues.append('handoff_before_completion')
        results.append({'context': context, 'step_id': context[2],
            't1_worker_start_mono_ns': t1, 't2_local_completion_observed_mono_ns': t2,
            'handoff_submitted_mono_ns': handoff, 't3_result_ready_observed_mono_ns': t3,
            'pp0_window_ms': (t2-t1)/1e6 if t1 is not None and t2 is not None else None,
            'whole_step_window_ms': (t3-t1)/1e6 if t1 is not None and t3 is not None else None,
            'remaining_pipeline_window_ms': (t3-t2)/1e6 if t2 is not None and t3 is not None else None,
            'requests': requests, 'request_count': len(by_req), 'chunk_count': len(chunks),
            'total_query_tokens': sum(by_req.values()),
            'chunks': [{'chunk': c, **chunks[c]} for c in sorted(chunks)],
            'rank_starts': starts, 'rank_chunk_completions': [{'chunk':c,'pcp':p,'mono_ns':v} for (c,p),v in sorted(ends.items())],
            'clock': next(iter(clocks)) if len(clocks)==1 else None,
            'valid': not issues, 'issues': sorted(set(issues)),
            'time_kind': 'host_worker_start_to_native_completion_and_result_observation',
            'valid_pure_device_timing': False, 'per_request_isolated_latency': False})
    return results
