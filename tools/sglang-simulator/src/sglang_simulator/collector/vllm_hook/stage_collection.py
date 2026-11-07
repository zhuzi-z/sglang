"""Four-stage low-volume host boundaries and OFFLINE rank aggregation.

No tensor data reads, new device waits, hardware profiling or hot-path file I/O.
Observation validity is distinct from suitability as a service-time label.
"""
import json
from . import host_breakdown as host

PROTOCOL_KINDS = frozenset({
    'handoff_submit_enter', 'handoff_submitted', 'handoff_barrier_done',
    'local_drain.enter', 'local_drain.ready', 'local_drain.error',
    'handoff_flush.enter', 'handoff_flush.submitted', 'handoff_flush.error',
    'payload_metadata',
})
NATIVE_KINDS = PROTOCOL_KINDS | frozenset({
    'forward_dispatch_enter', 'forward_dispatch_return', 'receive_reference_returned',
    'producer_flush.enter', 'producer_flush.submitted', 'producer_flush.error',
    'handoff_dependency_wait.enter', 'handoff_dependency_wait.ready',
    'handoff_dependency_wait.error', 'output_ready_observed', 'completion_unresolved',
})


def keep_event(kind, identity):
    if kind.startswith('startup.') or kind in ('native.boundary_installed', 'native.boundary_unavailable'):
        return True
    pp, pcp = identity[3], identity[4]
    if pp in range(4):
        if kind in ('chunk.inputs', 'chunk.layout', 'stage.config'):
            return pcp == 0
        if kind in ('worker.execute_model.enter', 'worker.execute_model.exit',
                    'worker.sample_tokens.enter', 'worker.sample_tokens.exit'):
            return True
        if host.ENABLED:
            if kind.startswith(('pp.', 'runner.', 'wrapper.execute_model.')):
                return pcp == 0
            if kind in ('worker.pp_push.enter', 'worker.pp_push.exit', 'worker.pp_settle.enter', 'worker.pp_settle.exit'):
                return pcp == 0
            if kind in ('native.helper_request.enter', 'native.helper_request.return', 'native.launch_call.enter', 'native.launch_call.return', 'native.helper_fence.enter', 'native.helper_fence.return', 'native.output_get.return'):
                return True
        return kind.startswith('native.') and (
            kind[7:] in NATIVE_KINDS or kind.endswith('.error'))
    if pp == -1:
        if host.ENABLED and kind in ('rpc.execute_model.dispatch', 'rpc.sample_tokens.dispatch'):
            return True
        return kind in ('step.inputs', 'request.add', 'request.scheduled',
                        'pipeline.result_ready', 'engine.dispatch.observed',
                        'rpc.pp_push.dispatch', 'rpc.pp_settle.dispatch')
    return False


def export_tables(window):
    names = ('requests', 'step_inputs', 'chunk_inputs', 'layouts', 'stage_configs', 'stage_points',
             'stage_rank_windows', 'protocol_points', 'controller_points')
    streams = {n: (window / (n + '.jsonl')).open('w') for n in names}
    counts = dict.fromkeys(names, 0)
    def write(name, row):
        streams[name].write(json.dumps(row, separators=(',', ':')) + '\n')
        counts[name] += 1
    try:
        for source in sorted(window.glob('events.*.jsonl')):
            with source.open() as f:
                for line in f:
                    row = json.loads(line)
                    kind, data = row['event'], row.get('data') or {}
                    if kind.startswith('request.'):
                        write('requests', row)
                    elif kind in ('step.inputs', 'chunk.inputs', 'chunk.layout'):
                        write({'step.inputs':'step_inputs', 'chunk.inputs':'chunk_inputs',
                               'chunk.layout':'layouts'}[kind], row)
                    elif kind == 'stage.config':
                        write('stage_configs', row)
                    elif kind.startswith(('native.', 'worker.', 'pp.', 'runner.', 'wrapper.')):
                        write('stage_points', row)
                        if data.get('launch_id') is not None:
                            write('protocol_points', row)
                        if kind in ('native.local_drain.ready', 'native.output_ready_observed') and row.get('context'):
                            start, end = data.get('start_mono_ns'), data.get('observed_mono_ns')
                            write('stage_rank_windows', {**row,
                                'start_mono_ns': start, 'end_mono_ns': end,
                                'observed_latency_ms': (end-start)/1e6 if start is not None and end is not None else None,
                                'time_kind': 'host_dispatch_to_native_completion_observed',
                                'valid_observation_window': start is not None and end is not None and end >= start,
                                'target_service_s': None, 'valid_service_label': False,
                                'valid_pure_device_timing': False,
                                'device_elapsed_ns': None, 'device_queue_ns': None})
                    elif kind.startswith(('engine.', 'rpc.')) or kind == 'pipeline.result_ready':
                        write('controller_points', row)
    finally:
        for stream in streams.values(): stream.close()
    return counts


def build_samples(sources, expected_pcp=8):
    """Join all relevant process/cut raw events; never subtract cross-host clocks.

    Returns four-stage chunk rows with complete inputs and per-rank evidence.
    Original native output completion on PP3 is composite sampling/copy scope.
    Missing/duplicate/unresolved evidence remains invalid, with explicit issues.
    """
    groups, steps, workers, seen = {}, {}, {}, set()
    for meta, events in sources:
        clock = (meta.get('clock_domain'), meta.get('host_clock', {}).get('boot_id'),
                 meta.get('host_clock', {}).get('time_namespace'))
        for event in events:
            unique = (meta['process_id'], event['seq'])
            if unique in seen: continue
            seen.add(unique)
            ctx = event.get('context')
            if not ctx or ctx[3] != 'real': continue
            context = tuple(ctx)
            kind, rank = event['event'], event['rank']
            if kind == 'step.inputs': steps[context] = event.get('data') or {}
            if kind == 'worker.execute_model.enter':
                workers.setdefault((context, rank['pp']), {})[rank['pcp']] = event['mono_ns']
            chunk = event.get('chunk', -1)
            if rank['pp'] in range(4) and chunk >= 0:
                groups.setdefault((context, chunk, rank['pp']), []).append((event, clock))
    result = []
    for (context, chunk, pp), records in sorted(groups.items()):
        # Protocol-only/untracked work cannot create a labelled real sample.
        if not any(e['event'] in ('chunk.inputs', 'native.forward_dispatch_enter') for e, _ in records): continue
        starts, returned, completions, drains, barriers, sends = {}, {}, {}, {}, {}, {}
        features, clocks, issues = None, set(), []
        scopes = set()
        payload_bytes = {}
        def put(dest, pcp, stamp):
            if pcp in dest: issues.append('duplicate_boundary')
            dest[pcp] = stamp
        for e, clock in records:
            kind, pcp, data = e['event'], e['rank']['pcp'], e.get('data') or {}
            if kind == 'chunk.inputs':
                if features is not None: issues.append('duplicate_chunk_input')
                features = data
            elif kind == 'native.forward_dispatch_enter':
                put(starts, pcp, data.get('start_mono_ns')); clocks.add(clock)
            elif kind == 'native.forward_dispatch_return':
                put(returned, pcp, data.get('observed_mono_ns'))
            elif kind == 'native.local_drain.enter':
                put(drains, pcp, data.get('observed_mono_ns'))
            elif kind in ('native.local_drain.ready', 'native.output_ready_observed'):
                put(completions, pcp, data.get('observed_mono_ns')); clocks.add(clock)
                scopes.add(data.get('completion_scope'))
            elif kind == 'native.handoff_barrier_done':
                barriers[pcp] = data.get('observed_mono_ns')
            elif kind == 'native.handoff_submitted':
                sends[pcp] = data.get('observed_mono_ns')
                payload_bytes[pcp] = data.get('payload_bytes')
            elif kind == 'native.completion_unresolved' or kind.endswith('.error'):
                issues.append(kind)
        expected = set(range(expected_pcp))
        if set(starts) != expected or any(v is None for v in starts.values()): issues.append('incomplete_forward_start')
        if set(completions) != expected or any(v is None for v in completions.values()): issues.append('incomplete_completion')
        if pp < 3 and (set(sends) != expected or set(drains) != expected or set(barriers) != expected):
            issues.append('incomplete_handoff_boundaries')
        if len(clocks) != 1 or not all(all(v) for v in clocks): issues.append('unverified_stage_clock')
        features = features or {}
        requests = features.get('requests') or []
        if not requests or not features.get('input_complete') or any(any(r.get(k) is None for k in
                ('req_id','query_length','past_length','input_length')) for r in requests):
            issues.append('incomplete_request_features')
        if requests and sum(r['query_length'] or 0 for r in requests) != features.get('global_valid_tokens'):
            issues.append('token_sum_mismatch')
        step = steps.get(context)
        if step is None or not step.get('native_dispatch_ok'): issues.append('missing_or_failed_step_input')
        else:
            expected_requests = {r['req_id']: r for r in step.get('requests', ())}
            if any(r['req_id'] not in expected_requests for r in requests):
                issues.append('chunk_step_request_mismatch')
            elif any(r['query_length'] != expected_requests[r['req_id']]['query_length'] or
                     r['past_length'] != expected_requests[r['req_id']]['past_length'] for r in requests):
                issues.append('chunk_step_feature_mismatch')
        per_rank = []
        def delta(end, begin):
            if end is None or begin is None: return None
            if end < begin: issues.append('timestamp_order_invalid'); return None
            return (end-begin)/1e6
        for pcp in sorted(expected):
            per_rank.append({'pcp_rank': pcp, 'forward_start_mono_ns': starts.get(pcp),
                'completion_observed_mono_ns': completions.get(pcp),
                'host_forward_submit_ms': delta(returned.get(pcp), starts.get(pcp)),
                'residual_native_wait_ms': delta(completions.get(pcp), drains.get(pcp)) if pp < 3 else None,
                'forward_return_to_drain_begin_ms': delta(drains.get(pcp), returned.get(pcp)),
                'world_barrier_observed_ms': delta(barriers.get(pcp), completions.get(pcp)) if pp < 3 else None,
                'handoff_submit_observed_ms': delta(sends.get(pcp), barriers.get(pcp)),
                'payload_bytes': payload_bytes.get(pcp)})
        for pcp in expected: delta(completions.get(pcp), starts.get(pcp))
        s = min((v for v in starts.values() if v is not None), default=None)
        end = max((v for v in completions.values() if v is not None), default=None)
        duration = delta(end, s) if not issues or s is not None else None
        result.append({'context':context, 'step_id':context[2], 'chunk':chunk, 'pp_stage':pp,
            'requests':requests, 'request_count':len(requests), 'features':features,
            'start_mono_ns':s, 'end_mono_ns':end, 'observed_stage_window_ms':duration,
            'completion_scopes':sorted(x for x in scopes if x), 'rank_evidence':per_rank,
            'pcp_ranks_complete':set(starts)==expected and set(completions)==expected,
            'host_clock_verified':len(clocks)==1 and all(all(v) for v in clocks),
            'clock':next(iter(clocks)) if len(clocks)==1 else None,
            'valid_observation_window':not issues, 'issues':sorted(set(issues)),
            'time_kind':'stage_host_dispatch_to_native_completion_observed',
            'label_quality':'host_observation_only', 'target_service_s':None,
            'valid_service_label':False, 'valid_pure_device_timing':False,
            'device_elapsed_ns':None, 'device_queue_ns':None,
            'per_request_isolated_latency':False})
    return result
