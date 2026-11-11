"""TPU input contracts and legacy-compatible continuous export controls.

No torch imports, device events, forced reads, synchronization or hardware
profiler start. Time fields always describe host observations.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time
import os
import uuid
import hashlib
import inspect
import logging
import math

REQUEST_ITERATION_LIMIT = 16384
SCHEDULE_KEY = '_sim_collection_schedule_id'
MEMBERS_KEY = '_sim_collection_step_members'

from sglang_simulator.hook import BaseHook
from . import trace_v2 as trace
from .trace_v2_hooks import HOOKS as TIMELINE_HOOKS, V2ExecutorHook, V2WorkerHook, V2RunnerTimingHook, V2PPWaveHook

EXPORT_KEY = '_sim_collection_export_id'


def observe(callback):
    try:
        return callback()
    except BaseException:
        # Even a broken observer-error counter must not replace a native call.
        try:
            if trace._RECORDER is not None:
                trace._RECORDER.observer_errors += 1
        except BaseException:
            pass
        return None


def cpu_int(value):
    # Scalars here come from documented CPU metadata, never device tensors.
    return int(value) if value is not None else None


def step_snapshot(output):
    past = {}
    for request in getattr(output, 'scheduled_new_reqs', ()):
        past[request.req_id] = cpu_int(getattr(request, 'num_computed_tokens', None))
    cached = getattr(output, 'scheduled_cached_reqs', None)
    for i, rid in enumerate(getattr(cached, 'req_ids', ())):
        values = getattr(cached, 'num_computed_tokens', None)
        past[rid] = cpu_int(values[i]) if values is not None and i < len(values) else None
    requests = tuple({'req_id': rid, 'query_length': cpu_int(q),
                      'past_length': past.get(rid)}
                     for rid, q in output.num_scheduled_tokens.items())
    total = cpu_int(output.total_num_scheduled_tokens)
    return {'requests': requests, 'request_count': len(requests),
            'total_num_scheduled_tokens': total,
            'query_sum_valid': sum(r['query_length'] for r in requests) == total,
            'past_complete': all(r['past_length'] is not None for r in requests),
            'source': 'SchedulerOutput_before_execute', 'contract': 3,
            'schedule_id': getattr(output, SCHEDULE_KEY, None),
            'members': getattr(output, MEMBERS_KEY, ()),
            'membership_complete': getattr(output, '_sim_collection_membership_complete', False)}


def hashed_request_id(rid):
    return hashlib.sha256(rid.encode('utf-8')).hexdigest()


SAMPLE_CONTEXT = ContextVar('sim_collection_sample', default=None)
OUTPUT_CONTEXT = ContextVar('sim_collection_output', default=None)


def worker_metadata_selected():
    rec = trace._RECORDER
    return rec is not None and rec.identity[4] == 0


def batch_hashes(runner):
    batch = getattr(runner, 'input_batch', None)
    ids = getattr(batch, 'req_ids', None)
    count = getattr(batch, 'num_reqs', None)
    if ids is None or count is None:
        return None
    return tuple(hashed_request_id(rid) for rid in ids[:count])


def sample_snapshot(runner):
    batch = getattr(runner, 'input_batch', None)
    ids = getattr(batch, 'req_ids', None)
    count = getattr(batch, 'num_reqs', None)
    state = getattr(runner, 'execute_model_state', None)
    scheduler = getattr(state, 'scheduler_output', None)
    scheduled = getattr(scheduler, 'scheduled_spec_decode_tokens', None)
    draft_counts = None
    if ids is not None and count is not None and scheduled is not None:
        # Only len() of CPU lists; no token values, truth tests or device reads.
        draft_counts = tuple(len(scheduled[rid]) if rid in scheduled else 0
                             for rid in ids[:count])
    config = getattr(runner, 'speculative_config', None)
    verify_metadata = getattr(state, 'spec_decode_metadata_list', None)
    is_spec_step = any(md is not None for md in verify_metadata) if verify_metadata is not None else None
    return {'request_ids_sha256': batch_hashes(runner),
            'scheduled_draft_counts': draft_counts,
            'has_scheduled_drafts': any(n > 0 for n in draft_counts) if draft_counts is not None else None,
            'execute_state_present': state is not None,
            'spec_verify_step': is_spec_step,
            'sample_phase': 'spec_verify' if is_spec_step is True else 'non_spec' if is_spec_step is False else None,
            'spec_configured': config is not None,
            'spec_method': getattr(config, 'method', None),
            'async_scheduling': getattr(getattr(runner, 'scheduler_config', None), 'async_scheduling', None),
            'phase_source': 'sample_entry_cpu_conditions_not_forward_phase'}


def begin_host_span(metadata=None, parent=None, **fields):
    span = {'span_id': uuid.uuid4().hex, 'parent_span_id': parent,
            'request_ids_sha256': None,
            'time_kind': 'host_native_call', 'hostonly': True,
            'device_elapsed_ns': None, 'device_completion_confirmed': False,
            'valid_pure_device_timing': False, 'valid_service_label': False,
            **(metadata or {}), **fields}
    # UUID generation and metadata assembly are observer work, not native time.
    span['start_mono_ns'] = observe(time.monotonic_ns)
    return span


def end_host_span(kind, span, scope, ok, *, end_ns, **fields):
    if span is not None:
        observe(lambda: trace.emit(kind, scope, data={
            **span, 'native_ok': ok, **fields, 'end_mono_ns': end_ns}))


def iteration_state(scheduler):
    state = getattr(scheduler, '_sim_collection_request_iterations', None)
    if state is None:
        state = {}
        scheduler._sim_collection_request_iterations = state
    return state


def admit_request(scheduler, rid, request):
    state = iteration_state(scheduler)
    previous = state.pop(rid, None)
    if len(state) >= REQUEST_ITERATION_LIMIT:
        return None
    lifecycle = getattr(scheduler, '_sim_collection_lifecycle_sequence', 0) + 1
    scheduler._sim_collection_lifecycle_sequence = lifecycle
    duplicate = previous is not None and previous['identity'] == id(request)
    entry = {'identity': id(request) if request is not None else None,
             'iteration': None if duplicate else 0,
             'lifecycle_id': lifecycle, 'duplicate_id': duplicate}
    state[rid] = entry
    return entry


def forget_request(scheduler, rid, request=None, lifecycle_id=None):
    state = iteration_state(scheduler)
    entry = state.get(rid)
    if entry is not None and (request is None or entry['identity'] == id(request)) and (
            lifecycle_id is None or entry['lifecycle_id'] == lifecycle_id):
        state.pop(rid, None)


def cache_split(new, request):
    total = cpu_int(getattr(new, 'num_computed_tokens', None))
    external = cpu_int(getattr(request, 'num_external_computed_tokens', None))
    if external is None:
        external = cpu_int(getattr(new, 'num_external_computed_tokens', None))
    valid = total is not None and external is not None and 0 <= external <= total
    return {'total_kv_hit_len': total, 'ext_kv_hit_len': external,
            'local_kv_hit_len': total - external if valid else None,
            'cache_split_valid': valid,
            'cache_source': 'new_request_preexecute_snapshot'}


def schedule_members(scheduler, output):
    """Ordered CPU-only metadata; never use mutable scheduler state for past KV."""
    new_reqs = {r.req_id: r for r in getattr(output, 'scheduled_new_reqs', ())}
    cached = getattr(output, 'scheduled_cached_reqs', None)
    cached_indices = {rid: i for i, rid in enumerate(getattr(cached, 'req_ids', ()))}
    state = iteration_state(scheduler)
    members = []
    for rid, query in getattr(output, 'num_scheduled_tokens', {}).items():
        new = new_reqs.get(rid)
        request = getattr(scheduler, 'requests', {}).get(rid)
        entry = state.get(rid)
        if entry is not None and request is not None and entry['identity'] != id(request):
            state.pop(rid, None)
            entry = None
        if entry is None and new is not None:
            entry = admit_request(scheduler, rid, request)
        if entry is not None and entry['iteration'] is not None:
            entry['iteration'] += 1
        index = cached_indices.get(rid)
        def cached_int(name):
            values = getattr(cached, name, None)
            return cpu_int(values[index]) if index is not None and values is not None and index < len(values) else None
        past = cpu_int(getattr(new, 'num_computed_tokens', None)) if new is not None else cached_int('num_computed_tokens')
        input_length = cpu_int(getattr(request, 'num_prompt_tokens', None))
        if input_length is None and new is not None:
            input_length = cpu_int(getattr(new, 'num_prompt_tokens', None))
            prompt = getattr(new, 'prompt_token_ids', None)
            if input_length is None and prompt is not None:
                input_length = len(prompt)
        before = cpu_int(getattr(request, 'num_output_tokens', None))
        if before is None:
            before = cpu_int(getattr(new, 'num_output_tokens', None)) if new is not None else cached_int('num_output_tokens')
        resumed = False if new is not None else None
        if index is not None:
            resumed_ids = getattr(cached, 'resumed_req_ids', None)
            flags = getattr(cached, 'resumed_from_preemption', None)
            if resumed_ids is not None:
                resumed = rid in resumed_ids
            elif flags is not None and index < len(flags):
                resumed = bool(flags[index])
        member = {'request_id_sha256': hashed_request_id(rid), 'query_length': cpu_int(query),
                  'past_kv_length': past, 'input_length': input_length,
                  'output_tokens_before': before,
                  'request_iteration': entry['iteration'] if entry is not None else None,
                  'request_lifecycle_id': entry['lifecycle_id'] if entry is not None else None,
                  'duplicate_id': entry['duplicate_id'] if entry is not None else False,
                  'new_request': new is not None, 'resumed': resumed}
        if new is not None:
            member.update(cache_split(new, request))
        member['membership_complete'] = all(member[k] is not None for k in (
            'query_length', 'past_kv_length', 'input_length', 'output_tokens_before',
            'request_iteration', 'resumed'))
        members.append(member)
    return tuple(members)


def finish_reason_name(request):
    reason = getattr(request, 'finish_reason', None)
    if reason is None:
        get_reason = getattr(request, 'get_finished_reason', None)
        reason = get_reason() if get_reason is not None else None
    return getattr(reason, 'name', reason if isinstance(reason, str) else None)


def active_member(scheduler, rid):
    return (getattr(scheduler, '_sim_collection_active_members', None) or {}).get(hashed_request_id(rid), {})


class CollectionExecutorHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm.v1.executor.multiproc_executor'
    HOOK_CLASS_NAME = 'MultiprocExecutor'

    @classmethod
    def hook(cls, target):
        original = target.collective_rpc

        @wraps(original)
        def rpc(self, method, *args, **kwargs):
            local_export = None
            is_start = True
            native_args = args[1] if len(args) > 1 else kwargs.get('args', ())
            snapshot = None
            if method == 'execute_model' and native_args:
                output = native_args[0]
                if getattr(output, 'total_num_scheduled_tokens', 0) > 0:
                    snapshot = observe(lambda: step_snapshot(output))
            if method == 'profile':
                inner = dict((args[2] if len(args) > 2 else kwargs.get('kwargs')) or {})
                export_id = inner.setdefault(EXPORT_KEY, uuid.uuid4().hex)
                is_start = native_args[0] if native_args else True
                rec = trace._RECORDER
                if rec is not None and hasattr(rec, 'export_cut'):
                    local_export = observe(lambda: rec.export_cut(export_id, is_start))
                if local_export is None:
                    local_export = {'complete': False, 'failure': 'controller_export_unavailable'}
                if len(args) > 2:
                    changed = list(args)
                    changed[2] = inner
                    args = tuple(changed)
                else:
                    kwargs = dict(kwargs, kwargs=inner)
            # V2 assigns the step context inside this call, before dispatch.
            try:
                result = original(self, method, *args, **kwargs)
            except BaseException:
                context = getattr(self, '_sim_v2_last', None)
                if snapshot is not None and context is not None:
                    observe(lambda: trace.emit('step.inputs', trace.Scope(context), data={
                        **snapshot, 'native_dispatch_ok': False}))
                raise
            if method in ('execute_model', 'sample_tokens', 'pp_push', 'pp_settle'):
                self._sim_collection_last_method = method
            context = getattr(self, '_sim_v2_last', None)
            if snapshot is not None and context is not None:
                observe(lambda: trace.emit('step.inputs', trace.Scope(context), data={
                    **snapshot, 'native_dispatch_ok': True}))
            if method == 'profile' and not is_start:
                receipts = result if isinstance(result, (list, tuple)) else (result,)
                if local_export is not None:
                    receipts = (*receipts, local_export)
                if any(not isinstance(value, dict) or value.get('complete') is not True
                       for value in receipts):
                    # A control-plane error is visible to the caller; never
                    # report successful stop when one process failed to export.
                    raise RuntimeError('collector export incomplete; inspect process/export metadata')
            return result
        target.collective_rpc = rpc

        # Headless executors lack these in the verified vLLM build. Preserve
        # native implementations when present; do not swallow RPC failures.
        for name, start in (('start_profile', True), ('stop_profile', False)):
            if getattr(target, name, None) is not None:
                continue
            def install(is_start):
                def profile(self, profile_prefix=None, profiler_kwargs=None):
                    return self.collective_rpc('profile', args=(is_start, profile_prefix, profiler_kwargs))
                return profile
            setattr(target, name, install(start))


class CollectionWorkerHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm_torchtpu.worker.tpu_worker'
    HOOK_CLASS_NAME = 'TPUWorker'

    @classmethod
    def hook(cls, target):
        def profile(self, is_start=True, profile_prefix=None, profiler_kwargs=None,
                    **kwargs):
            # Deliberately matches legacy collector semantics: export accumulated
            # records on BOTH triggers. No hardware profiler is invoked.
            rec = trace._RECORDER
            if rec is None or not hasattr(rec, 'export_cut'):
                return {'complete': False, 'failure': 'collection_recorder_unavailable'}
            return rec.export_cut(kwargs.get(EXPORT_KEY), is_start)
        target.profile = profile

        execute = target.execute_model
        @wraps(execute)
        def wrapped_execute(self, *args, **kwargs):
            scope = trace.ACTIVE.get()
            rec = trace._RECORDER
            if (getattr(rec, 'focus_stage', False) and rec.identity[4] == 0
                    and not getattr(self, '_sim_stage_config_recorded', False)):
                def config_snapshot():
                    cfg = getattr(self, 'vllm_config', None)
                    model = getattr(cfg, 'model_config', None)
                    parallel = getattr(cfg, 'parallel_config', None)
                    runner = getattr(self, 'model_runner', None)
                    trace.emit('stage.config', scope, data={
                        'model': str(getattr(model, 'model', 'unknown')),
                        'dtype': str(getattr(model, 'dtype', 'unknown')),
                        'quantization': getattr(model, 'quantization', None),
                        'pp_size': getattr(parallel, 'pipeline_parallel_size', None),
                        'pcp_size': getattr(parallel, 'prefill_context_parallel_size', None),
                        'tp_size': getattr(parallel, 'tensor_parallel_size', None),
                        'layer_partition': os.getenv('VLLM_PP_LAYER_PARTITION'),
                        'async_handoff': os.getenv('TPU_PP_ASYNC_HANDOFF'),
                        'lazy_push': os.getenv('TPU_PP_LAZY_PUSH'),
                        'max_num_tokens': cpu_int(getattr(runner, 'max_num_tokens', None)),
                        'max_num_reqs': cpu_int(getattr(runner, 'max_num_reqs', None)),
                        'token_buckets': tuple(int(v) for v in getattr(runner, 'num_tokens_paddings', ())),
                        'source': 'worker_cpu_config', 'contract': 1})
                    self._sim_stage_config_recorded = True
                observe(config_snapshot)
            if scope is not None:
                self._sim_collection_pending = (scope.context, time.monotonic_ns())
            try:
                return execute(self, *args, **kwargs)
            except BaseException:
                self._sim_collection_pending = None
                raise
        target.execute_model = wrapped_execute

        sample = getattr(target, 'sample_tokens', None)
        if sample is None:
            return
        @wraps(sample)
        def wrapped_sample(self, *args, **kwargs):
            from .native_boundaries import OUTPUT_WAVE
            wave = getattr(getattr(self, 'model_runner', None), '_pp_wave', None)
            token = OUTPUT_WAVE.set(wave)
            try:
                result = sample(self, *args, **kwargs)
            finally:
                OUTPUT_WAVE.reset(token)
            pending = getattr(self, '_sim_collection_pending', None)
            self._sim_collection_pending = None
            if pending is None:
                return result
            context, started = pending
            recorded = False
            def record(kind):
                nonlocal recorded
                if recorded:
                    return
                recorded = True
                if kind == 'ready_observed':
                    from .native_boundaries import output_ready
                    observe(lambda: output_ready(getattr(self.model_runner, '_pp_wave', None), context))
                ended = time.monotonic_ns()
                trace.emit('step.output_' + kind, trace.Scope(context), data={
                    'start_mono_ns': started, 'end_mono_ns': ended,
                    'iter_latency': (ended - started) / 1e9,
                    'time_kind': 'host_output_' + kind,
                    'device_elapsed_ns': None})
            native_get = getattr(result, 'get_output', None)
            if native_get is None:
                observe(lambda: record('call_return'))
                return result
            @wraps(native_get)
            def get_output(*get_args, **get_kwargs):
                value = native_get(*get_args, **get_kwargs)
                observe(lambda: record('ready_observed'))
                return value
            def attach():
                result.get_output = get_output
                return True
            if not observe(attach):
                trace.emit('step.output_observer_unavailable', trace.Scope(context), data={
                    'time_kind': 'unavailable', 'iter_latency': None,
                    'reason': 'output_get_output_not_assignable'})
            return result
        target.sample_tokens = wrapped_sample


class CollectionRunnerHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm_torchtpu.runner.tpu_runner'
    HOOK_CLASS_NAME = 'TPUModelRunner'

    @classmethod
    def hook(cls, target):
        original = target._prepare_inputs
        @wraps(original)
        def prepare(self, output, start_index, *args, **kwargs):
            scope = observe(trace.ACTIVE.get)
            rec = trace._RECORDER
            selected = observe(lambda: scope is not None and rec is not None and rec.identity[4] == 0 and (not rec.focus_pp0 or rec.identity[3] == 0))
            def snapshot():
                batch = self.input_batch
                # Capture before native mutable staging buffers can be reused.
                cap = max(getattr(self, 'num_reqs_max_model_len', batch.num_reqs),
                          getattr(self, 'num_reqs_most_model_len', 0) or 0)
                ids = tuple(batch.req_ids[start_index:start_index + cap])
                requests = []
                for j, rid in enumerate(ids, start_index):
                    state = self.requests.get(rid)
                    params = getattr(state, 'sampling_params', None)
                    emitted = getattr(state, 'output_token_ids', None)
                    requests.append({'req_id': rid,
                        'query_length': cpu_int(output.num_scheduled_tokens[rid]),
                        'past_length': cpu_int(batch.num_computed_tokens_cpu[j]),
                        'input_length': cpu_int(batch.num_prompt_tokens[j]),
                        'num_output_tokens': len(emitted) if emitted is not None else None,
                        'max_output_tokens': cpu_int(getattr(params, 'max_tokens', None)),
                        'logprobs': cpu_int(getattr(params, 'logprobs', None)),
                        'phase_by_lengths': 'prefill' if int(batch.num_computed_tokens_cpu[j]) < int(batch.num_prompt_tokens[j]) else 'decode',
                        'prompt_completed_by_lengths': int(batch.num_computed_tokens_cpu[j]) + int(output.num_scheduled_tokens[rid]) >= int(batch.num_prompt_tokens[j])})
                return requests
            captured = observe(snapshot) if selected else None
            result = original(self, output, start_index, *args, **kwargs)
            if captured is not None:
                def record():
                    end = cpu_int(result[4])
                    requests = tuple(captured[:end-start_index])
                    plan = self._last_sequence_layout_plan
                    trace.emit('chunk.inputs', scope, chunk=scope.chunk + 1, data={
                        'requests': requests, 'request_count': len(requests),
                        'start_request_index': start_index, 'end_request_index': end,
                        'total_query_tokens': sum(r['query_length'] for r in requests),
                        'global_valid_tokens': cpu_int(plan.global_num_tokens),
                        'global_padded_tokens': cpu_int(plan.global_padded_num_tokens),
                        'local_valid_tokens': cpu_int(plan.local_num_tokens),
                        'local_padded_tokens': cpu_int(plan.local_padded_num_tokens),
                        'input_complete': len(requests) == end-start_index and
                            sum(r['query_length'] for r in requests) == int(plan.global_num_tokens),
                        'output_length_source': 'runner_cpu_state_before_chunk',
                        'contract': 3})
                observe(record)
            return result
        target._prepare_inputs = prepare

        sample = getattr(target, 'sample_tokens', None)
        if sample is None:
            return
        @wraps(sample)
        def sample_tokens(self, *args, **kwargs):
            scope = observe(trace.ACTIVE.get)
            selected = observe(worker_metadata_selected)
            metadata = observe(lambda: sample_snapshot(self)) if selected else None
            span = observe(lambda: begin_host_span(metadata)) if selected else None
            context = {'scope': scope, 'metadata': metadata,
                       'span_id': span['span_id'] if span else None}
            token = SAMPLE_CONTEXT.set(context)
            active_token = observe(lambda: trace.ACTIVE.set(scope))
            ok = False
            try:
                value = sample(self, *args, **kwargs)
                ok = True
                return value
            finally:
                ended = observe(time.monotonic_ns)
                SAMPLE_CONTEXT.reset(token)
                if active_token is not None:
                    observe(lambda: trace.ACTIVE.reset(active_token))
                end_host_span('request.sample_tokens', span, scope, ok, end_ns=ended)
        target.sample_tokens = sample_tokens

        verify = getattr(target, '_sample_spec_verify_chunk', None)
        if verify is None:
            return
        @wraps(verify)
        def verify_chunk(self, *args, **kwargs):
            context = SAMPLE_CONTEXT.get() or {}
            scope = context.get('scope') or observe(trace.ACTIVE.get)
            def snapshot():
                md = args[1] if len(args) > 1 else kwargs.get('md')
                start = args[4] if len(args) > 4 else kwargs.get('cur_start_idx')
                end = args[5] if len(args) > 5 else kwargs.get('cur_end_idx')
                ids = (context.get('metadata') or {}).get('request_ids_sha256')
                if ids is None:
                    ids = batch_hashes(self)
                # Metadata is CHUNK-local (native _prepare_inputs builds it
                # from the global request slice), not a batch-global array.
                lengths = getattr(md, 'draft_lengths_cpu', None)
                counts = tuple(cpu_int(n) for n in lengths[:end-start]) if lengths is not None else None
                return {'request_ids_sha256': ids[start:end] if ids is not None else None,
                        'draft_counts': counts,
                        'draft_count': sum(counts) if counts is not None else None,
                        'start_request_index': start, 'end_request_index': end,
                        'forward_included_in_main_forward': True,
                        'independent_pp_forward_duration': False}
            selected = observe(worker_metadata_selected)
            metadata = observe(snapshot) if selected else None
            span = observe(lambda: begin_host_span(metadata, parent=context.get('span_id'),
                time_kind='host_verify_accept_submission')) if selected else None
            ok = False
            try:
                value = verify(self, *args, **kwargs)
                ok = True
                return value
            finally:
                ended = observe(time.monotonic_ns)
                end_host_span('mtp.verify_accept', span, scope, ok, end_ns=ended)
        target._sample_spec_verify_chunk = verify_chunk


class CollectionSpecManagerHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm_torchtpu.runner.speculative_decoding_manager'
    HOOK_CLASS_NAME = 'SpeculativeDecodingManager'

    @classmethod
    def hook(cls, target):
        for name in ('propose_draft_token_ids', 'take_draft_token_ids'):
            native = getattr(target, name)
            def install(method, original):
                @wraps(original)
                def call(self, *args, **kwargs):
                    context = SAMPLE_CONTEXT.get() or {}
                    scope = context.get('scope') or observe(trace.ACTIVE.get)
                    def snapshot():
                        runner = self.runner
                        data = {'request_ids_sha256': batch_hashes(runner),
                                'identity_source': 'runner.input_batch_before_native',
                                'spec_method': getattr(getattr(runner, 'speculative_config', None), 'method', None),
                                'method': method}
                        if method == 'propose_draft_token_ids':
                            sampled = args[0] if args else kwargs.get('sampled_token_ids')
                            data['sampled_counts'] = cpu_row_counts(sampled)
                            data['return_device_requested'] = args[4] if len(args) > 4 else kwargs.get('return_device', False)
                        return data
                    selected = observe(worker_metadata_selected)
                    metadata = observe(snapshot) if selected else None
                    span = observe(lambda: begin_host_span(metadata, parent=context.get('span_id'),
                        time_kind='host_draft_submission' if method == 'propose_draft_token_ids'
                        else 'host_draft_read_call_including_existing_copy_wait')) if selected else None
                    ok, value = False, None
                    try:
                        value = original(self, *args, **kwargs)
                        ok = True
                        return value
                    finally:
                        ended = observe(time.monotonic_ns)
                        counts = None
                        if ok and span is not None and method == 'take_draft_token_ids':
                            counts = observe(lambda: cpu_row_counts(getattr(value, 'draft_token_ids', None)))
                        end_host_span('mtp.' + method, span, scope, ok, end_ns=ended,
                                      result_present=value is not None if ok else None,
                                      draft_counts=counts)
                return call
            setattr(target, name, install(name, native))


def cpu_row_counts(rows):
    # Only the public, already-materialized CPU list's OUTER rows are visited.
    # In particular never inspect shape/device or private copy-state tensors.
    if type(rows) not in (list, tuple):
        return None
    return tuple(len(row) for row in rows)


class CollectionRunnerTimingHook(V2RunnerTimingHook):
    @classmethod
    def hook(cls, target):
        probe_annotation = None
        if os.getenv('SIM_TPU_DEVICE_PROBE') == '1':
            def existing_annotation():
                from .device_probe import annotation
                return annotation
            probe_annotation = observe(existing_annotation)
        for method in ('stage', 'host_stage'):
            native = getattr(target, method, None)
            if native is None:
                continue
            def install(original, native_method):
                @contextmanager
                @wraps(original)
                def stage(self, *args, **kwargs):
                    name = args[0] if args else kwargs.get('stage', kwargs.get('name'))
                    context = SAMPLE_CONTEXT.get() or OUTPUT_CONTEXT.get() or {}
                    scope = (context.get('scope') or observe(trace.ACTIVE.get)) if isinstance(name, str) else None
                    chunk = scope.chunk if scope is not None and name == 'main_forward' else -1
                    selected = observe(worker_metadata_selected)
                    span = observe(lambda: begin_host_span(context.get('metadata'),
                        parent=context.get('span_id'), native_timing_method=native_method,
                        stage=name, independent_pp_forward_duration=False)) if selected and isinstance(name, str) else None
                    ok = False
                    if name == 'main_forward' and scope is not None and native_method == 'stage':
                        def forward():
                            from .native_boundaries import forward_enter
                            forward_enter(scope)
                        observe(forward)
                    if scope is not None:
                        observe(lambda: trace.emit('runner.' + name + '.enter', scope, chunk,
                            data=span if span is not None else None))
                    try:
                        # Preserve native __enter__/__exit__, yield, suppression
                        # and exceptions. No extra device Event or synchronization.
                        with original(self, *args, **kwargs) as value:
                            annotation = observe(lambda: probe_annotation(name, scope)) if probe_annotation is not None and native_method == 'stage' else None
                            def enter_annotation():
                                annotation.__enter__()
                                return True
                            entered = observe(enter_annotation) if annotation is not None else False
                            try:
                                yield value
                            finally:
                                if entered:
                                    observe(lambda: annotation.__exit__(None, None, None))
                        ok = True
                    finally:
                        ended = observe(time.monotonic_ns)
                        if scope is not None:
                            observe(lambda: trace.emit('runner.' + name + '.exit', scope, chunk, data={
                                **(span or {}), 'end_mono_ns': ended, 'native_ok': ok}))
                        end_host_span('mtp.stage', span, scope, ok, end_ns=ended)
                return stage
            setattr(target, method, install(native, method))


class CollectionPcpLayoutHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm_torchtpu.layers.core.pcp_sequence_layout'
    HOOK_CLASS_NAME = 'PcpSequenceLayoutEligibility'

    @classmethod
    def hook(cls, target):
        original = target._evaluate_streaming_query
        @wraps(original)
        def evaluate(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            scope = trace.ACTIVE.get()
            rec = trace._RECORDER
            if scope is not None and not getattr(rec, 'focus_pp0', False) and (
                    not getattr(rec, 'focus_stage', False) or rec.identity[4] == 0):
                def record():
                    def ints(name):
                        values = getattr(result, name, None)
                        return tuple(int(x) for x in values) if values is not None else None
                    data = {'pcp_size': int(result.pcp_size),
                            'interleave_size': int(result.interleave_size),
                            'local_valid_counts': ints('local_token_counts'),
                            'local_padded_tokens': cpu_int(result.local_padded_tokens),
                            'global_padded_tokens': cpu_int(result.global_padded_tokens)}
                    if trace._RECORDER is not None and trace._RECORDER.identity[4] == 0:
                        data['owner_start_offsets'] = ints('token_owner_start_offsets_per_req')
                        data['absolute_query_start_offsets'] = ints('absolute_query_start_offsets_per_req')
                    trace.emit('chunk.layout', scope, chunk=scope.chunk + 1, data=data)
                observe(record)
            return result
        target._evaluate_streaming_query = evaluate


class CollectionEngineCoreIdentityHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm.v1.engine.core'
    HOOK_CLASS_NAME = 'EngineCore'

    @classmethod
    def hook(cls, target):
        original = target.preprocess_add_request
        @wraps(original)
        def preprocess_add_request(self, request, *args, **kwargs):
            started = observe(time.monotonic_ns)
            result = original(self, request, *args, **kwargs)
            ended = observe(time.monotonic_ns)
            def record():
                external = getattr(request, 'external_req_id', None)
                if external is not None:
                    trace.emit('request.identity_alias', data={
                        'request_id_sha256': hashed_request_id(request.request_id),
                        'external_request_id_sha256': hashed_request_id(external),
                        'identity_source': 'EngineCoreRequest.external_req_id',
                        'start_mono_ns': started, 'end_mono_ns': ended,
                        'native_ok': True, 'source': 'EngineCore.preprocess_add_request',
                        'time_kind': 'host_native_call', 'wall_ns': time.time_ns()})
            observe(record)
            return result
        target.preprocess_add_request = preprocess_add_request


class CollectionSchedulerHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm.v1.core.sched.scheduler'
    HOOK_CLASS_NAME = 'Scheduler'

    @classmethod
    def hook(cls, target):
        if 'add_request' not in target.__dict__:
            return
        initialize = target.__init__
        @wraps(initialize)
        def initialize_scheduler(self, *args, **kwargs):
            result = initialize(self, *args, **kwargs)
            def install_boundaries():
                manager = getattr(self, 'kv_cache_manager', None)
                if manager is None:
                    return
                from sglang_simulator.cache_boundaries import (
                    install_cache_manager, install_connector, install_scheduler_transitions,
                )
                def emit(event, **data):
                    trace.emit(event, trace.ACTIVE.get(), data=data)
                install_cache_manager(self.kv_cache_manager, emit, observe)
                install_scheduler_transitions(self, emit, observe)
                if self.connector is not None:
                    install_connector(self.connector, emit, observe)
            observe(install_boundaries)
            return result
        target.__init__ = initialize_scheduler
        add = target.add_request
        @wraps(add)
        def add_request(self, request, *args, **kwargs):
            started = observe(time.monotonic_ns)
            result = add(self, request, *args, **kwargs)
            ended = observe(time.monotonic_ns)
            def record():
                params = getattr(request, 'sampling_params', None)
                trace.emit('request.add', data={'request_id_sha256': hashed_request_id(request.request_id),
                    'input_length': cpu_int(request.num_prompt_tokens),
                    'max_output_tokens': cpu_int(getattr(request, 'max_tokens', None)),
                    'arrival_wall_s': getattr(request, 'arrival_time', None),
                    'logprobs': cpu_int(getattr(params, 'logprobs', None)),
                    'start_mono_ns': started, 'end_mono_ns': ended, 'native_ok': True,
                    'source': 'Scheduler.add_request', 'time_kind': 'host_native_call'})
            observe(record)
            observe(lambda: admit_request(self, request.request_id, request))
            return result
        target.add_request = add_request

        free = target._free_request
        @wraps(free)
        def free_request(self, request, *args, **kwargs):
            started = observe(time.monotonic_ns)
            result = free(self, request, *args, **kwargs)
            ended = observe(time.monotonic_ns)
            def record():
                member = active_member(self, request.request_id)
                entry = iteration_state(self).get(request.request_id) or {}
                if entry.get('identity') != id(request):
                    entry = {}
                trace.emit('request.free', data={
                    'request_id_sha256': hashed_request_id(request.request_id),
                    'schedule_id': getattr(self, '_sim_collection_active_schedule_id', None),
                    'request_lifecycle_id': member.get('request_lifecycle_id', entry.get('lifecycle_id')),
                    'request_iteration': member.get('request_iteration', entry.get('iteration')),
                    'status': getattr(getattr(request, 'status', None), 'name', None),
                    'finish_reason': finish_reason_name(request),
                    'num_output_tokens': cpu_int(getattr(request, 'num_output_tokens', None)),
                    'finished': True, 'logical_finished': True,
                    'kv_release_confirmed': False, 'wall_ns': time.time_ns(),
                    'start_mono_ns': started, 'end_mono_ns': ended, 'native_ok': True,
                    'source': 'Scheduler._free_request',
                    'time_kind': 'host_logical_completion_observed'})
            observe(record)
            observe(lambda: forget_request(self, request.request_id, request=request))
            return result
        target._free_request = free_request

        update = getattr(target, 'update_from_output', None)
        if update is not None:
            @wraps(update)
            def update_from_output(self, *args, **kwargs):
                def snapshot():
                    output = args[0] if args else kwargs.get('scheduler_output')
                    scheduled = observe(lambda: getattr(output, 'num_scheduled_tokens', None))
                    ids = observe(lambda: tuple(scheduled.keys())) if issubclass(type(scheduled), dict) else None
                    total = observe(lambda: getattr(output, 'total_num_scheduled_tokens', None))
                    # Missing CPU metadata does not confirm an empty batch.
                    if type(total) is not int or total < 0:
                        total = None
                    state = 'unknown'
                    if ids is not None and total is not None:
                        if not ids and total == 0:
                            state = 'empty'
                        elif ids and total > 0:
                            state = 'nonempty'
                    return {
                        'schedule_id': observe(lambda: getattr(output, SCHEDULE_KEY, None)),
                        'request_ids_sha256': observe(lambda: tuple(hashed_request_id(rid) for rid in ids)) if ids is not None else None,
                        'request_count': len(ids) if ids is not None else None,
                        'total_num_scheduled_tokens': total,
                        'num_scheduled_tokens_keys_available': ids is not None,
                        'total_num_scheduled_tokens_available': total is not None,
                        'scheduled_token_batch_state': state,
                        'identity_source': 'SchedulerOutput.num_scheduled_tokens_before_update'}
                def record_batch(ok):
                    if not ok or metadata.get('scheduled_token_batch_state') != 'empty':
                        trace.emit('scheduler.output_update', data={
                            **metadata, 'start_mono_ns': started, 'end_mono_ns': ended,
                            'native_ok': ok, 'source': 'Scheduler.update_from_output',
                            'time_kind': 'host_native_call', 'hostonly': True,
                            'device_elapsed_ns': None, 'device_completion_confirmed': False,
                            'valid_pure_device_timing': False, 'valid_service_label': False,
                            'shared_batch_span': True, 'timing_scope': 'batch_shared',
                            'per_request_apportionment': False})
                def activate():
                    output = args[0] if args else kwargs.get('scheduler_output')
                    self._sim_collection_active_schedule_id = getattr(output, SCHEDULE_KEY, None)
                    self._sim_collection_active_members = {
                        m['request_id_sha256']: m for m in getattr(output, MEMBERS_KEY, ())}
                def clear():
                    self._sim_collection_active_schedule_id = None
                    self._sim_collection_active_members = None
                try:
                    observe(activate)
                    metadata = observe(snapshot) or {'scheduled_token_batch_state': 'unknown'}
                    started = observe(time.monotonic_ns)
                    try:
                        result = update(self, *args, **kwargs)
                    except BaseException:
                        ended = observe(time.monotonic_ns)
                        observe(lambda: record_batch(False))
                        raise
                    ended = observe(time.monotonic_ns)
                    observe(lambda: record_batch(True))
                    def record_item(item):
                        member = active_member(self, item.request_id)
                        tokens = getattr(item, 'new_token_ids', None)
                        reason = getattr(item, 'finish_reason', None)
                        def count(*names):
                            for name in names:
                                value = getattr(item, name, None)
                                if value is not None:
                                    return cpu_int(value)
                            return None
                        trace.emit('request.output_observed', data={
                            'request_id_sha256': hashed_request_id(item.request_id),
                            'schedule_id': getattr(self, '_sim_collection_active_schedule_id', None),
                            'request_lifecycle_id': member.get('request_lifecycle_id'),
                            'request_iteration': member.get('request_iteration'),
                            'new_token_count': len(tokens) if tokens is not None else None,
                            'finish_reason': getattr(reason, 'name', None),
                            'finished': reason is not None,
                            'num_cached_tokens': cpu_int(getattr(item, 'num_cached_tokens', None)),
                            'spec_decode_num_draft_tokens': count('spec_decode_num_draft_tokens', 'num_draft_tokens', 'num_spec_tokens'),
                            'spec_decode_num_accepted_tokens': count('spec_decode_num_accepted_tokens', 'num_accepted_tokens'),
                            'wall_ns': time.time_ns(), 'time_kind': 'host_output_accepted_observed',
                            'start_mono_ns': started, 'end_mono_ns': ended, 'native_ok': True,
                            'source': 'Scheduler.update_from_output', 'shared_batch_span': True})
                    def record():
                        for outputs in (result or {}).values():
                            for item in (getattr(outputs, 'outputs', None) or ()):
                                observe(lambda: record_item(item))
                                def cleanup():
                                    if getattr(item, 'finish_reason', None) is not None:
                                        member = active_member(self, item.request_id)
                                        forget_request(self, item.request_id,
                                                       lifecycle_id=member.get('request_lifecycle_id'))
                                observe(cleanup)
                    observe(record)
                    return result
                finally:
                    observe(clear)
            target.update_from_output = update_from_output
        install_schedule(target)


def install_schedule(target):
    if 'schedule' not in target.__dict__:
        return
    original = target.schedule
    @wraps(original)
    def schedule(self, *args, **kwargs):
        if observe(lambda: getattr(self, '_sim_collection_in_schedule', False)):
            return original(self, *args, **kwargs)
        observe(lambda: setattr(self, '_sim_collection_in_schedule', True))
        try:
            def ready():
                for rid in getattr(self, 'finished_recving_kv_req_ids', ()):
                    observe(lambda: trace.emit('request.remote_kv_signal_observed', data={
                        'request_id_sha256': hashed_request_id(rid), 'wall_ns': time.time_ns(),
                        'load_failed_known': rid in getattr(self, 'failed_recving_kv_req_ids', ()),
                        'ready_success_confirmed': False,
                        'time_kind': 'host_remote_kv_signal_observed'}))
            observe(ready)
            started = observe(time.monotonic_ns)
            result = original(self, *args, **kwargs)
            ended = observe(time.monotonic_ns)
            def tag():
                schedule_id = getattr(self, '_sim_collection_schedule_sequence', 0) + 1
                self._sim_collection_schedule_sequence = schedule_id
                observe(lambda: setattr(result, SCHEDULE_KEY, schedule_id))
                return schedule_id
            schedule_id = observe(tag)
            def record_new():
                for new in getattr(result, 'scheduled_new_reqs', ()):
                    request = getattr(self, 'requests', {}).get(new.req_id)
                    trace.emit('request.scheduled', data={
                        'request_id_sha256': hashed_request_id(new.req_id), 'schedule_id': schedule_id,
                        'input_length': cpu_int(getattr(request, 'num_prompt_tokens', None)),
                        'max_output_tokens': cpu_int(getattr(request, 'max_tokens', None)),
                        **cache_split(new, request)})
            observe(record_new)
            def record_members():
                if getattr(result, 'num_scheduled_tokens', None) is None:
                    return  # Not a SchedulerOutput; do not claim empty complete membership.
                members = schedule_members(self, result)
                complete = all(m['membership_complete'] for m in members)
                observe(lambda: setattr(result, MEMBERS_KEY, members))
                observe(lambda: setattr(result, '_sim_collection_membership_complete', complete))
                if not members:
                    return
                trace.emit('request.step_membership', data={
                    'schedule_id': schedule_id, 'members': members,
                    'request_count': len(members), 'membership_complete': complete,
                    'start_mono_ns': started, 'end_mono_ns': ended,
                    'time_kind': 'host_scheduler_call', 'native_schedule_ok': True,
                    'pp_dispatch_start': False,
                    'source': 'SchedulerOutput_before_execute'})
            observe(record_members)
            return result
        finally:
            observe(lambda: setattr(self, '_sim_collection_in_schedule', False))
    target.schedule = schedule


class CollectionTpuSchedulerHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm_torchtpu.core.pp_chunk_scheduler'
    HOOK_CLASS_NAME = 'TpuPipelineChunkScheduler'

    @classmethod
    def hook(cls, target):
        # The subclass overrides schedule and calls the base implementation.
        # A per-instance nesting guard prevents duplicate request events.
        install_schedule(target)


class CollectionLazyStepperHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm_torchtpu.core.pp_lazy_step'
    HOOK_CLASS_NAME = 'LazyPushStepper'

    @classmethod
    def hook(cls, target):
        add_slot = getattr(target, '_add_slot', None)
        if add_slot is not None:
            @wraps(add_slot)
            def wrapped_add(self, slot, *args, **kwargs):
                def tag():
                    executor = self.executor
                    context = getattr(executor, '_sim_v2_last', None)
                    if getattr(executor, '_sim_collection_last_method', None) == 'pp_push':
                        session = getattr(executor, '_sim_v2_session', None)
                        context = (*session, -1, 'pp_push') if session else None
                    slot._sim_collection_context = context
                observe(tag)
                return add_slot(self, slot, *args, **kwargs)
            target._add_slot = wrapped_add
        for name in ('_dispatch', '_sampled', '_process'):
            native = getattr(target, name)
            def install(method, original):
                @wraps(original)
                def call(self, *args, **kwargs):
                    focus = getattr(trace._RECORDER, 'focus_pp0', False)
                    stage_focus = getattr(trace._RECORDER, 'focus_stage', False)
                    if (focus or stage_focus) and method != '_dispatch':
                        return original(self, *args, **kwargs)
                    dispatched_result = None
                    if method == '_dispatch':
                        dispatched_result = original(self, *args, **kwargs)
                        if not dispatched_result:
                            return dispatched_result
                        if focus:
                            from .pp0_steps import observe_result
                            observe(lambda: observe_result(self))
                            return dispatched_result
                        if stage_focus:
                            from .pp0_steps import observe_result
                            observe(lambda: observe_result(self, 'pipeline.result_ready'))
                    def snapshot():
                        slot = (args[0] if method == '_sampled' else
                                getattr(args[0], 'slot', None)) if args else None
                        if method == '_dispatch':
                            pending = getattr(self, 'in_flight', ())
                            slot = getattr(pending[-1], 'slot', None) if pending else None
                        if slot is None:
                            return  # no logs for empty connector/scheduler polls
                        start = getattr(slot, 'started_last', None)
                        same_drain = slot is not None and getattr(slot, 'started_drain', None) == getattr(self, 'drains', None)
                        context = getattr(slot, '_sim_collection_context', None)
                        valid_reply = method == '_sampled' and start is not None and not same_drain
                        event = 'engine.dispatch.observed' if method == '_dispatch' else 'engine.' + method[1:] + '.enter'
                        trace.emit(event, trace.Scope(context) if context else None, data={
                            'slot_dispatched': getattr(slot, 'dispatched', None),
                            'started_last_mono_s': start,
                            'drains': getattr(self, 'drains', None),
                            'in_flight': len(getattr(self, 'in_flight', ())),
                            'reply_interval_valid': valid_reply,
                            'forward_s': time.monotonic()-start if valid_reply else None,
                            'time_kind': 'engine_reply_observation'})
                    observe(snapshot)
                    if method == '_dispatch':
                        return dispatched_result
                    return original(self, *args, **kwargs)
                return call
            setattr(target, name, install(name, native))


class CombinedExecutorHook(CollectionExecutorHook):
    @classmethod
    def hook(cls, target):
        V2ExecutorHook.hook(target)
        CollectionExecutorHook.hook(target)


class CollectionOutputBoundaryHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm_torchtpu.runner.tpu_runner_async_output'
    HOOK_CLASS_NAME = 'AsyncTPUModelRunnerOutput'

    @classmethod
    def hook(cls, target):
        from .native_boundaries import install_output_class
        raw_init, raw_get = target.__init__, target.get_output

        @wraps(raw_init)
        def tracked_init(self, *args, **kwargs):
            context = OUTPUT_CONTEXT.get()
            status = context.get('init_status') if context else None
            if status is not None:
                status['called'] = True
            try:
                value = raw_init(self, *args, **kwargs)
            except BaseException as error:
                if status is not None:
                    status['error'] = error
                raise
            if status is not None:
                status.update(ok=True, value=value)
            return value
        target.__init__ = tracked_init

        @wraps(raw_get)
        def tracked_get(self, *args, **kwargs):
            context = OUTPUT_CONTEXT.get()
            status = context.get('native_status') if context else None
            if status is not None:
                status['called'] = True
            try:
                value = raw_get(self, *args, **kwargs)
            except BaseException as error:
                ended = observe(time.monotonic_ns)
                if status is not None:
                    status.update(error=error, end_ns=ended)
                raise
            ended = observe(time.monotonic_ns)
            if status is not None:
                status.update(ok=True, value=value, end_ns=ended)
            return value
        target.get_output = tracked_get
        observe(lambda: install_output_class(target))
        observe(install_kv_publish_wait_observer)
        init, get = target.__init__, target.get_output

        @wraps(init)
        def initialize(self, *args, **kwargs):
            status = {'called': False, 'ok': False}
            token = OUTPUT_CONTEXT.set({'init_status': status})
            try:
                try:
                    value = init(self, *args, **kwargs)
                except BaseException:
                    if not status['called']:
                        value = tracked_init(self, *args, **kwargs)
                    elif status['ok']:
                        value = status['value']
                    else:
                        raise status['error']
            finally:
                OUTPUT_CONTEXT.reset(token)
            def snapshot():
                context = SAMPLE_CONTEXT.get() or {}
                scope = context.get('scope') or trace.ACTIVE.get()
                output = getattr(self, '_model_runner_output', None)
                ids = getattr(output, 'req_ids', None)
                metadata = {'request_ids_sha256': tuple(hashed_request_id(rid) for rid in ids) if ids is not None else None,
                            'identity_source': 'output_constructor_cpu_req_ids'}
                self._sim_collection_output = {
                    'scope': scope, 'metadata': metadata,
                    'span_id': context.get('span_id')}
            if observe(worker_metadata_selected):
                observe(snapshot)
            return value
        target.__init__ = initialize

        @wraps(get)
        def get_output(self, *args, **kwargs):
            captured = observe(lambda: getattr(self, '_sim_collection_output', None)) or {}
            scope = captured.get('scope') or observe(trace.ACTIVE.get)
            selected = observe(worker_metadata_selected)
            span = observe(lambda: begin_host_span(captured.get('metadata'),
                parent=captured.get('span_id'),
                completion_scope='output_materialize_including_existing_kv_and_copy_wait')) if selected else None
            status = {'called': False, 'ok': False}
            context = {**captured, 'scope': scope, 'native_status': status,
                       'span_id': span['span_id'] if span else None,
                       'copy_wait_observed': False, 'copy_wait_effective': False}
            token = OUTPUT_CONTEXT.set(context)
            active_token = observe(lambda: trace.ACTIVE.set(scope))
            ok, value = False, None
            try:
                try:
                    value = get(self, *args, **kwargs)
                except BaseException:
                    # The pre-existing boundary observer must not replace a
                    # native result/exception, or prevent its single invocation.
                    if not status['called']:
                        value = tracked_get(self, *args, **kwargs)
                    elif status['ok']:
                        value = status['value']
                    else:
                        raise status['error']
                ok = True
                return value
            finally:
                # tracked_get captures at raw native return, before the existing
                # boundary observer's postprocessing (including its failure path).
                ended = status.get('end_ns')
                counts = observe(lambda: cpu_row_counts(getattr(value, 'sampled_token_ids', None))) if ok and span is not None else None
                end_host_span('request.output_materialize', span, scope, ok, end_ns=ended,
                    sampled_counts=counts,
                    copy_wait_observed=context['copy_wait_observed'],
                    effective_copy_wait=context['copy_wait_effective'],
                    device_completion_confirmed=context['copy_wait_observed'] if ok else False)
                if active_token is not None:
                    observe(lambda: trace.ACTIVE.reset(active_token))
                OUTPUT_CONTEXT.reset(token)
        target.get_output = get_output


class KvPublishWaitHandler(logging.Handler):
    template = 'kv_publish_event_wait req_ids=%s unix_ns=%d duration_ms=%.3f'

    def emit(self, record):
        def capture():
            if record.msg != self.template or not worker_metadata_selected():
                return
            ids, wall_ns, duration_ms = record.args
            if type(duration_ms) not in (int, float) or not math.isfinite(duration_ms) or duration_ms < 0:
                return
            context = OUTPUT_CONTEXT.get() or {}
            trace.emit('request.kv_publish_wait', context.get('scope'), data={
                'request_ids_sha256': tuple(hashed_request_id(rid) for rid in ids),
                'parent_span_id': context.get('span_id'),
                'native_log_wall_ns': wall_ns, 'native_wait_duration_ms': duration_ms,
                'time_kind': 'host_existing_native_wait_duration',
                'source': 'AsyncTPUModelRunnerOutput.get_output.kv_publish_event_wait',
                'completion_scope': 'existing_kv_publish_fence_not_token_copy',
                'device_completion_confirmed': True, 'valid_pure_device_timing': False,
                'observed_mono_ns': time.monotonic_ns()})
        observe(capture)


def install_kv_publish_wait_observer():
    logger = logging.getLogger('vllm_torchtpu.runner.tpu_runner_async_output')
    if not any(isinstance(handler, KvPublishWaitHandler) for handler in logger.handlers):
        logger.addHandler(KvPublishWaitHandler())
    trace.emit('startup.kv_publish_wait_observer', data={
        'native_info_log_enabled': logger.isEnabledFor(logging.INFO),
        'source': 'existing_native_structured_log', 'adds_device_wait': False})


class CollectionMtpProposerHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm_torchtpu.spec_decode.eagle3'
    HOOK_CLASS_NAME = 'Eagle3Proposer'

    @classmethod
    def hook(cls, target):
        native = getattr(target, 'prefill_only', None)
        if native is None:
            observe(lambda: trace.emit('startup.mtp_prefill_unavailable', data={'reason': 'method_missing'}))
            return
        parameters = observe(lambda: inspect.signature(native).parameters)
        if parameters is None or not {'scheduler_output', 'device_seed'} <= parameters.keys():
            observe(lambda: trace.emit('startup.mtp_prefill_unavailable', data={'reason': 'signature_unverified'}))
            return
        source_hash = observe(lambda: hashlib.sha256(inspect.getsource(native).encode()).hexdigest())
        observe(lambda: trace.emit('startup.mtp_prefill_installed', data={
            'source_sha256': source_hash, 'runtime_source_verified_against_r58': False,
            'time_kind': 'host_native_call', 'adds_device_wait': False}))

        @wraps(native)
        def prefill_only(self, *args, **kwargs):
            context = SAMPLE_CONTEXT.get() or {}
            scope = context.get('scope') or observe(trace.ACTIVE.get)
            selected = observe(worker_metadata_selected)
            metadata = observe(lambda: {
                'request_ids_sha256': batch_hashes(self.runner),
                'phase': 'mtp_prefill_only', 'produces_draft_proposals': False,
                'source_sha256': source_hash,
                'source': 'Eagle3Proposer.prefill_only'}) if selected else None
            span = observe(lambda: begin_host_span(metadata, parent=context.get('span_id'),
                time_kind='host_mtp_prefill_submission')) if selected else None
            ok = False
            try:
                value = native(self, *args, **kwargs)
                ok = True
                return value
            finally:
                ended = observe(time.monotonic_ns)
                end_host_span('mtp.prefill_only', span, scope, ok, end_ns=ended)
        target.prefill_only = prefill_only


class CollectionAsyncCopyStateHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm_torchtpu.runner.tpu_runner_async_output'
    HOOK_CLASS_NAME = 'AsyncTPUCopyState'

    @classmethod
    def hook(cls, target):
        native = target.wait
        @wraps(native)
        def wait(self, *args, **kwargs):
            context = OUTPUT_CONTEXT.get() or {}
            selected = observe(worker_metadata_selected)
            before = observe(lambda: self.completed) if selected else None
            span = observe(lambda: begin_host_span(context.get('metadata'),
                parent=context.get('span_id'), completed_before=before,
                time_kind='host_copy_wait_including_existing_native_wait')) if selected else None
            ok = False
            try:
                value = native(self, *args, **kwargs)
                ok = True
                return value
            finally:
                ended = observe(time.monotonic_ns)
                completed = observe(lambda: self.completed) if ok and selected else None
                confirmed = ok and completed is True
                if confirmed and context:
                    context['copy_wait_observed'] = True
                    context['copy_wait_effective'] = context.get('copy_wait_effective', False) or before is False
                end_host_span('request.copy_wait', span, context.get('scope'), ok, end_ns=ended,
                    completed_after=completed, effective_wait=confirmed and before is False,
                    device_completion_confirmed=confirmed,
                    completion_scope='existing_native_token_copy_wait_not_kv_readiness')
        target.wait = wait


class CombinedPPWaveHook(V2PPWaveHook):
    @classmethod
    def hook(cls, target):
        V2PPWaveHook.hook(target)
        from .native_boundaries import install_for_class
        install_for_class(target)


class CombinedWorkerHook(CollectionWorkerHook):
    @classmethod
    def hook(cls, target):
        V2WorkerHook.hook(target)
        CollectionWorkerHook.hook(target)
        from .device_probe import install
        install(target)


# The class installer applies ONLY the first matching hook. Compose additions
# for existing targets, rather than registering a second unreachable hook.
HOOKS = [CombinedExecutorHook if hook is V2ExecutorHook else
         CombinedWorkerHook if hook is V2WorkerHook else
         CollectionRunnerTimingHook if hook is V2RunnerTimingHook else
         CombinedPPWaveHook if hook is V2PPWaveHook else hook
         for hook in TIMELINE_HOOKS]
HOOKS += [CollectionRunnerHook, CollectionPcpLayoutHook, CollectionEngineCoreIdentityHook,
          CollectionSchedulerHook, CollectionTpuSchedulerHook, CollectionLazyStepperHook,
          CollectionOutputBoundaryHook, CollectionSpecManagerHook, CollectionAsyncCopyStateHook,
          CollectionMtpProposerHook]
