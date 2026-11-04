"""TPU input contracts and legacy-compatible continuous export controls.

No torch imports, device events, forced reads, synchronization or hardware
profiler start. Time fields always describe host observations.
"""
from contextlib import contextmanager
from functools import wraps
import time
import os
import uuid

from sglang_simulator.hook import BaseHook
from . import trace_v2 as trace
from .trace_v2_hooks import HOOKS as TIMELINE_HOOKS, V2ExecutorHook, V2WorkerHook, V2RunnerTimingHook

EXPORT_KEY = '_sim_collection_export_id'


def observe(callback):
    try:
        return callback()
    except Exception:
        if trace._RECORDER is not None:
            trace._RECORDER.observer_errors += 1
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
        values = getattr(cached, 'num_computed_tokens', ())
        past[rid] = cpu_int(values[i]) if i < len(values) else None
    requests = tuple({'req_id': rid, 'query_length': cpu_int(q),
                      'past_length': past.get(rid)}
                     for rid, q in output.num_scheduled_tokens.items())
    total = cpu_int(output.total_num_scheduled_tokens)
    return {'requests': requests, 'request_count': len(requests),
            'total_num_scheduled_tokens': total,
            'query_sum_valid': sum(r['query_length'] for r in requests) == total,
            'past_complete': all(r['past_length'] is not None for r in requests),
            'source': 'SchedulerOutput_before_execute', 'contract': 3}


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
                    trace.emit('step.inputs', trace.Scope(context), data={
                        **snapshot, 'native_dispatch_ok': False})
                raise
            if method in ('execute_model', 'sample_tokens', 'pp_push', 'pp_settle'):
                self._sim_collection_last_method = method
            context = getattr(self, '_sim_v2_last', None)
            if snapshot is not None and context is not None:
                trace.emit('step.inputs', trace.Scope(context), data={
                    **snapshot, 'native_dispatch_ok': True})
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
            result = sample(self, *args, **kwargs)
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
            scope = trace.ACTIVE.get()
            rec = trace._RECORDER
            selected = scope is not None and rec is not None and rec.identity[4] == 0
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
                        'logprobs': cpu_int(getattr(params, 'logprobs', None))})
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


class CollectionRunnerTimingHook(V2RunnerTimingHook):
    @classmethod
    def hook(cls, target):
        native = target.stage
        probe_annotation = None
        if os.getenv('SIM_TPU_DEVICE_PROBE') == '1':
            from .device_probe import annotation
            probe_annotation = annotation
        @contextmanager
        @wraps(native)
        def stage(self, *args, **kwargs):
            name = args[0] if args else kwargs.get('stage', kwargs.get('name'))
            scope = trace.ACTIVE.get() if isinstance(name, str) else None
            chunk = scope.chunk if scope is not None and name == 'main_forward' else -1
            ok = False
            if scope is not None:
                trace.emit('runner.' + name + '.enter', scope, chunk)
            try:
                with native(self, *args, **kwargs) as value:
                    if probe_annotation is None:
                        yield value
                    else:
                        with probe_annotation(name, scope):
                            yield value
                ok = True
            finally:
                if scope is not None:
                    trace.emit('runner.' + name + '.exit', scope, chunk, data={'native_ok': ok})
        target.stage = stage


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
            if scope is not None:
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


class CollectionSchedulerHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm.v1.core.sched.scheduler'
    HOOK_CLASS_NAME = 'Scheduler'

    @classmethod
    def hook(cls, target):
        if 'add_request' not in target.__dict__:
            return
        add = target.add_request
        @wraps(add)
        def add_request(self, request, *args, **kwargs):
            def record():
                params = getattr(request, 'sampling_params', None)
                trace.emit('request.add', data={'req_id': request.request_id,
                    'input_length': cpu_int(request.num_prompt_tokens),
                    'max_output_tokens': cpu_int(getattr(request, 'max_tokens', None)),
                    'arrival_wall_s': getattr(request, 'arrival_time', None),
                    'logprobs': cpu_int(getattr(params, 'logprobs', None))})
            observe(record)
            return add(self, request, *args, **kwargs)
        target.add_request = add_request

        free = target._free_request
        @wraps(free)
        def free_request(self, request, *args, **kwargs):
            observe(lambda: trace.emit('request.free', data={'req_id': request.request_id,
                'num_output_tokens': cpu_int(getattr(request, 'num_output_tokens', None))}))
            return free(self, request, *args, **kwargs)
        target._free_request = free_request

        update = getattr(target, 'update_from_output', None)
        if update is not None:
            @wraps(update)
            def update_from_output(self, *args, **kwargs):
                result = update(self, *args, **kwargs)
                def record():
                    for outputs in (result or {}).values():
                        for item in (getattr(outputs, 'outputs', None) or ()):
                            tokens = getattr(item, 'new_token_ids', None)
                            if tokens:
                                trace.emit('request.output_observed', data={
                                    'req_id': item.request_id, 'new_token_count': len(tokens)})
                observe(record)
                return result
            target.update_from_output = update_from_output
        install_schedule(target)


def install_schedule(target):
    if 'schedule' not in target.__dict__:
        return
    original = target.schedule
    @wraps(original)
    def schedule(self, *args, **kwargs):
        if getattr(self, '_sim_collection_in_schedule', False):
            return original(self, *args, **kwargs)
        self._sim_collection_in_schedule = True
        try:
            def ready():
                for rid in getattr(self, 'finished_recving_kv_req_ids', ()):
                    trace.emit('request.remote_kv_ready_observed', data={'req_id': rid})
            observe(ready)
            result = original(self, *args, **kwargs)
            def record():
                for new in getattr(result, 'scheduled_new_reqs', ()):
                    request = self.requests.get(new.req_id)
                    total = cpu_int(getattr(new, 'num_computed_tokens', None))
                    external = cpu_int(getattr(request, 'num_external_computed_tokens', None))
                    if external is None:
                        external = cpu_int(getattr(new, 'num_external_computed_tokens', None))
                    trace.emit('request.scheduled', data={
                        'req_id': new.req_id,
                        'input_length': cpu_int(getattr(request, 'num_prompt_tokens', None)),
                        'max_output_tokens': cpu_int(getattr(request, 'max_tokens', None)),
                        'total_kv_hit_len': total,
                        'ext_kv_hit_len': external,
                        'local_kv_hit_len': total-external if total is not None and external is not None else None,
                        'cache_split_valid': total is not None and external is not None,
                        'cache_source': 'new_request_preexecute_snapshot'})
            observe(record)
            return result
        finally:
            self._sim_collection_in_schedule = False
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
                    dispatched_result = None
                    if method == '_dispatch':
                        dispatched_result = original(self, *args, **kwargs)
                        if not dispatched_result:
                            return dispatched_result
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
         CollectionRunnerTimingHook if hook is V2RunnerTimingHook else hook
         for hook in TIMELINE_HOOKS]
HOOKS += [CollectionRunnerHook, CollectionPcpLayoutHook, CollectionSchedulerHook,
          CollectionTpuSchedulerHook, CollectionLazyStepperHook]
