"""Observe existing TPU drains, never introduce a device wait or graph break.

These are host observations, NOT hardware execution/communication timestamps.
The global synchronize adapter immediately delegates outside scoped PP calls.
"""
from contextvars import ContextVar
from functools import wraps
import hashlib
import inspect
import os
import sys
import time
import threading

from . import trace_v2 as trace
from . import host_breakdown as host

DRAIN_SHA256 = '6e57dc548358406c3000430287b04accb63b9973fa4eb324957f46c35991a07a'
_SYNC = ContextVar('sim_native_boundary_sync', default=None)
_WAVE = ContextVar('sim_native_boundary_wave', default=None)
OUTPUT_WAVE = ContextVar('sim_native_boundary_output_wave', default=None)


def payload_metadata(carried):
    widths = {'torch.bfloat16':2, 'torch.float16':2, 'torch.float32':4,
              'torch.float64':8, 'torch.int8':1, 'torch.uint8':1, 'torch.bool':1,
              'torch.int16':2, 'torch.int32':4, 'torch.int64':8,
              'torch.float8_e4m3fn':1, 'torch.float8_e5m2':1}
    tensors, total = [], 0
    for tensor in carried:
        shape, dtype = getattr(tensor, 'shape', None), getattr(tensor, 'dtype', None)
        if shape is None or dtype is None:
            return {'available':False, 'payload_bytes':None, 'tensors':[]}
        shape, dtype = tuple(int(v) for v in shape), str(dtype)
        count = 1
        for v in shape: count *= v
        size = widths.get(dtype)
        tensors.append({'shape':shape, 'dtype':dtype, 'bytes':count*size if size else None})
        if size is None: total = None
        elif total is not None: total += count*size
    return {'available':bool(tensors) and total is not None,
            'payload_bytes':total if tensors else None, 'tensors':tensors}


def _emit(kind, item=None, **data):
    recorder = trace._RECORDER
    stage_focus = getattr(recorder, 'focus_stage', False)
    if item is None and kind not in ('boundary_installed', 'boundary_unavailable'):
        from .stage_collection import PROTOCOL_KINDS
        if not stage_focus or kind not in PROTOCOL_KINDS or data.get('launch_id') is None or not data.get('protocol_active'):
            return None
    if stage_focus:
        from .stage_collection import keep_event
        if not keep_event('native.' + kind, recorder.identity):
            return None
    if getattr(recorder, "focus_pp0", False):
        from .pp0_steps import keep_event
        if not keep_event("native." + kind, recorder.identity):
            return None
    try:
        stamp = time.monotonic_ns()
        scope = trace.Scope(item['context']) if item is not None else None
        trace.emit('native.' + kind, scope, item['chunk'] if item else -1,
                   data={'observed_mono_ns': stamp, 'time_kind': 'host_native_boundary',
                         'device_start_ns': None, 'device_end_ns': None, **data})
        return stamp
    except Exception:
        if trace._RECORDER is not None:
            trace._RECORDER.observer_errors += 1
        return None


def _item(scope):
    if scope is None:
        return None
    return {'context': scope.context, 'chunk': scope.chunk, 'start_mono_ns': None}


def forward_enter(scope):
    wave = _WAVE.get()
    item = getattr(wave, '_sim_native_current', None)
    if item is not None and scope is not None and item['context'] == scope.context and item['chunk'] == scope.chunk:
        item['start_mono_ns'] = time.monotonic_ns()
        _emit('forward_dispatch_enter', item, start_mono_ns=item['start_mono_ns'])


def output_ready(wave, context):
    """Called only AFTER the caller's original get_output returns.

    Last-stage tail includes native sampling/copy work. It is never marked as
    isolated decoder execution, even when only one chunk remains outstanding.
    """
    item = getattr(wave, '_sim_native_unfinished', None)
    if item is None or item['context'] != context or wave.rank != wave.last:
        return
    _emit('output_ready_observed', item, start_mono_ns=item['start_mono_ns'],
          completion_scope='last_stage_output_including_sampling_and_copy',
          isolated_chunk_execution=False)
    wave._sim_native_unfinished = None


def install(module, sync_module, expected_digest=DRAIN_SHA256, target=None):
    """Install once. Unknown native drain source disables this feature only."""
    if getattr(module, '_sim_native_boundaries', False):
        return True
    if expected_digest is not None:
        try:
            digest = hashlib.sha256(inspect.getsource(module._drain_slice).encode()).hexdigest()
        except (OSError, TypeError):
            digest = None
        if digest != expected_digest:
            _emit('boundary_unavailable', reason='native_drain_source_mismatch', source_sha256=digest)
            return False
    # __build_class__ hooks run BEFORE the module binds its PPWave name.
    target = target if target is not None else module.PPWave
    for name in ('start_forward', 'end_forward', '_launch_blocking', '_launch_async'):
        if not callable(getattr(target, name, None)):
            _emit('boundary_unavailable', reason='native_method_missing', method=name)
            return False

    if not getattr(sync_module.synchronize, '_sim_native_observer', False):
        synchronize = sync_module.synchronize

        @wraps(synchronize)
        def sync(*args, **kwargs):
            state = _SYNC.get()
            if state is None:
                return synchronize(*args, **kwargs)
            phase, item, task = state
            wait = kwargs.get('wait', args[1] if len(args) > 1 else None)
            _emit(phase + '.enter', item, wait=wait, **task)
            measured = host.snapshot() if item is not None else None
            try:
                result = synchronize(*args, **kwargs)
            except BaseException:
                _emit(phase + '.error', item, wait=wait, native_ok=False, **host.elapsed(measured), **task)
                raise
            accounting = host.elapsed(measured)
            # This occurs BEFORE _drain_slice executes the world barrier.
            _emit(phase + '.ready' if wait is True else phase + '.submitted', item,
                  wait=wait, native_ok=True,
                  start_mono_ns=item.get('start_mono_ns') if item else None,
                  completion_scope='local_outstanding_work_at_native_drain' if phase == 'local_drain' else 'native_sync_call',
                  isolated_chunk_execution=False, **accounting, **task)
            return result
        sync._sim_native_observer = True
        sync_module.synchronize = sync

    drain = module._drain_slice

    @wraps(drain)
    def drain_slice(*args, **kwargs):
        state = _SYNC.get()
        if state is None:
            return drain(*args, **kwargs)
        _, item, task = state
        token = _SYNC.set(('local_drain', item, task))
        measured = host.snapshot() if item is not None else None
        try:
            value = drain(*args, **kwargs)
            _emit('handoff_barrier_done', item, drain_call_accounting=host.elapsed(measured), **task)
            return value
        finally:
            _SYNC.reset(token)
    module._drain_slice = drain_slice

    launch_now = module._launch_now

    @wraps(launch_now)
    def now(*args, **kwargs):
        state = _SYNC.get()
        if state is None:
            return launch_now(*args, **kwargs)
        _, item, task = state
        token = _SYNC.set(('handoff_flush', item, task))
        _emit('handoff_submit_enter', item, **task)
        try:
            result = launch_now(*args, **kwargs)
            _emit('handoff_submitted', item, completion_scope='submission_only', **task)
            return result
        finally:
            _SYNC.reset(token)
    module._launch_now = now

    flush = module._flush

    @wraps(flush)
    def producer_flush(*args, **kwargs):
        state = _SYNC.get()
        if state is None:
            return flush(*args, **kwargs)
        _, item, task = state
        token = _SYNC.set(('producer_flush', item, task))
        try:
            return flush(*args, **kwargs)
        finally:
            _SYNC.reset(token)
    module._flush = producer_flush

    native_async = target._launch_async

    @wraps(native_async)
    def launch_async(self, carried):
        # Native code allows only one outstanding helper launch. Freeze identity
        # explicitly: ContextVars do not propagate into the helper thread.
        item = getattr(self, '_sim_native_unfinished', None)
        self._sim_native_helper_item = item
        self._sim_native_helper_carried = carried
        token = _SYNC.set(('producer_flush', item, {'async_helper': True}))
        measured = host.snapshot() if item is not None else None
        if host.ENABLED:
            _emit('helper_request.enter', item)
        try:
            return native_async(self, carried)
        finally:
            if host.ENABLED:
                _emit('helper_request.return', item, **host.elapsed(measured), completion_scope='request_call_return_not_enqueue')
            _SYNC.reset(token)
    target._launch_async = launch_async

    native_launch = target._launch_blocking

    @wraps(native_launch)
    def launch(self, carried):
        item = getattr(self, '_sim_native_unfinished', None)
        helper = bool(getattr(self, 'async_handoff', False)) and self.rank == 0
        # Helper identity is captured when the job is queued, never inferred
        # from the helper's ACTIVE context (which is usually empty).
        if helper and getattr(self, '_sim_native_helper_carried', None) is carried:
            item = getattr(self, '_sim_native_helper_item', item)
        self._sim_native_launch_id = getattr(self, '_sim_native_launch_id', 0) + 1
        task = {'launch_id': self._sim_native_launch_id,
                'execution_thread': 'async_handoff_helper' if threading.current_thread() is getattr(self, '_async_thread', None) else 'caller',
                'drain_enabled': bool(self.drain_before_handoff)}
        if getattr(trace._RECORDER, 'focus_stage', False) and getattr(self, '_sim_stage_protocol_active', False):
            try:
                task.update(protocol_active=True, protocol_session=getattr(self, '_sim_protocol_session', None))
                burst = getattr(self, 'bursts', None)
                if getattr(self, '_sim_protocol_burst', object()) != burst:
                    self._sim_protocol_burst, self._sim_protocol_ordinal = burst, 0
                ordinal = self._sim_protocol_ordinal
                self._sim_protocol_ordinal += 1
                task.update(burst_id=burst, launch_ordinal=ordinal,
                            carried_real_context=item is not None)
                payload = payload_metadata(carried)
                task['payload_bytes'] = payload['payload_bytes']
                signature = tuple((tuple(t['shape']), t['dtype']) for t in payload['tensors'])
                if payload['available'] and signature != getattr(self, '_sim_payload_signature', None):
                    self._sim_payload_signature = signature
                    _emit('payload_metadata', item, **task, payload_tensors=payload['tensors'])
            except Exception:
                if trace._RECORDER is not None:
                    trace._RECORDER.observer_errors += 1
        token = _SYNC.set(('handoff', item, task))
        measured = host.snapshot() if item is not None else None
        if host.ENABLED:
            _emit('launch_call.enter', item, **task)
        try:
            result = native_launch(self, carried)
            if self.drain_before_handoff and getattr(self, '_sim_native_unfinished', None) is item:
                self._sim_native_unfinished = None
            return result
        finally:
            if host.ENABLED:
                _emit('launch_call.return', item, **host.elapsed(measured), **task)
            _SYNC.reset(token)
            if helper and getattr(self, '_sim_native_helper_carried', None) is carried:
                self._sim_native_helper_carried = None
                self._sim_native_helper_item = None
    target._launch_blocking = launch

    if host.ENABLED and callable(getattr(target, '_async_wait', None)):
        native_fence = target._async_wait

        @wraps(native_fence)
        def async_fence(self, *args, **kwargs):
            item = _item(trace.ACTIVE.get())
            waited = getattr(self, '_sim_native_helper_item', None)
            measured = host.snapshot() if item is not None else None
            _emit('helper_fence.enter', item,
                  waited_context=waited['context'] if waited else None)
            try:
                return native_fence(self, *args, **kwargs)
            finally:
                _emit('helper_fence.return', item, **host.elapsed(measured),
                      completion_scope='host_helper_condition_wait_not_device_wait')
        target._async_wait = async_fence

    wait = module._wait

    @wraps(wait)
    def dependency_wait(*args, **kwargs):
        item = _item(trace.ACTIVE.get())
        _emit('handoff_dependency_wait.enter', item)
        measured = host.snapshot() if item is not None else None
        try:
            value = wait(*args, **kwargs)
            _emit('handoff_dependency_wait.ready', item,
                  completion_scope='previous_handoff_dependency_not_current_receive', **host.elapsed(measured))
            return value
        except BaseException:
            _emit('handoff_dependency_wait.error', item, native_ok=False)
            raise
    module._wait = dependency_wait

    start = target.start_forward

    @wraps(start)
    def start_forward(self, *args, **kwargs):
        # Existing V2 wrapper increments chunk inside this call. Any drain in
        # the native start still consumes _unfinished from the PREVIOUS chunk.
        if getattr(trace._RECORDER, 'focus_stage', False):
            scope = trace.ACTIVE.get()
            if scope is not None and scope.context[3] == 'real':
                # Initial compiler/warmup handoffs stay silent. A real step's
                # prologue and subsequent zero pushes remain observable.
                self._sim_stage_protocol_active = True
                self._sim_protocol_session = scope.context[:2]
        value = start(self, *args, **kwargs)
        item = _item(trace.ACTIVE.get())
        self._sim_native_current = item
        _WAVE.set(self)
        if item is not None:
            _emit('receive_reference_returned', item,
                  receive_device_ready=False, has_upstream=self.rank != 0)
        return value
    target.start_forward = start_forward

    end = target.end_forward

    @wraps(end)
    def end_forward(self, *args, **kwargs):
        item = getattr(self, '_sim_native_current', None)
        previous = getattr(self, '_sim_native_unfinished', None)
        if previous is not None and previous is not item:
            _emit('completion_unresolved', previous,
                  reason='next_chunk_before_observed_native_completion')
        self._sim_native_unfinished = item
        _emit('forward_dispatch_return', item, has_downstream=self.rank != self.last)
        try:
            return end(self, *args, **kwargs)
        finally:
            self._sim_native_current = None
            _WAVE.set(None)
    target.end_forward = end_forward
    module._sim_native_boundaries = True
    _emit('boundary_installed', source_sha256=expected_digest,
          added_synchronizations=0, hardware_profiler=False)
    return True


def install_for_class(target):
    if os.getenv('SIM_TPU_NATIVE_BOUNDARIES') != '1':
        return
    try:
        from torch_tpu._internal.sync import sync
        install(sys.modules[target.__module__], sync, target=target)
    except Exception:
        _emit('boundary_unavailable', reason='observer_install_failed')


def install_output_class(target):
    if os.getenv('SIM_TPU_NATIVE_BOUNDARIES') != '1':
        return
    init = target.__init__
    get = target.get_output

    @wraps(init)
    def initialize(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._sim_native_wave = OUTPUT_WAVE.get()
        scope = trace.ACTIVE.get()
        self._sim_native_context = scope.context if scope is not None else None

    @wraps(get)
    def get_output(self, *args, **kwargs):
        wave = getattr(self, '_sim_native_wave', None)
        item = getattr(wave, '_sim_native_unfinished', None)
        if item is not None and item['context'] != getattr(self, '_sim_native_context', None):
            item = None
        measured = host.snapshot() if item is not None else None
        value = get(self, *args, **kwargs)
        accounting = host.elapsed(measured)
        if host.ENABLED:
            _emit('output_get.return', item, **accounting, completion_scope='output_get_including_existing_copy_wait')
        # Native get_output has waited for the token copy's ready event.
        # No copy_state means no corresponding completion guarantee.
        if getattr(self, 'copy_state', None) is not None:
            output_ready(getattr(self, '_sim_native_wave', None),
                         getattr(self, '_sim_native_context', None))
        return value
    target.__init__ = initialize
    target.get_output = get_output
