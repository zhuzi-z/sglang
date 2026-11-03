"""Opt-in host timeline for the verified vLLM 0.29/TorchTPU RPC path."""
from contextlib import contextmanager
from functools import wraps
from itertools import islice
import os
import uuid

from sglang_simulator.hook import BaseHook
from . import trace_v2 as trace


def _valid(context):
    return (isinstance(context, tuple) and len(context) == 4
            and all(isinstance(value, str) for value in context[:2])
            and isinstance(context[2], int) and isinstance(context[3], str))


def _init(target, after):
    original = target.__init__

    @wraps(original)
    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        try:
            after(self)
        except Exception as exc:
            trace.emit('startup.observer_disabled', data=type(exc).__name__)
    target.__init__ = wrapped


class V2ExecutorHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm.v1.executor.multiproc_executor'
    HOOK_CLASS_NAME = 'MultiprocExecutor'

    @classmethod
    def hook(cls, target):
        def ready(self):
            if trace.initialize() is None:
                return
            if hasattr(self, 'rpc_broadcast_mq') and self.rpc_broadcast_mq is None:
                trace.emit('startup.controller.passive')
                return
            # One startup-only cohort check. Never send new RPC kwargs to an
            # old worker; a follower cannot initiate this check and stays passive.
            capabilities = self.collective_rpc('_sim_trace_v2_transport_capabilities', timeout=5)
            if (not isinstance(capabilities, (list, tuple))
                    or len(capabilities) != getattr(self, 'world_size', len(capabilities))
                    or not capabilities or any(value != 2 for value in capabilities)):
                trace.emit('startup.cohort_disabled', data='missing_v2_transport')
                return
            collection = os.getenv('SIM_TRACE_COLLECTION_ID', '')
            every = max(int(os.getenv('SIM_TRACE_EVERY', '1')), 1)
            self._sim_v2_step = 0
            self._sim_v2_last = None
            self._sim_v2_every = every
            self._sim_v2_session = (collection if 0 < len(collection) <= 128 else uuid.uuid4().hex, uuid.uuid4().hex)
            trace.emit('startup.controller.ready',
                       trace.Scope((*self._sim_v2_session, 0, 'startup')),
                       data=len(capabilities))
        _init(target, ready)
        original = target.collective_rpc

        @wraps(original)
        def rpc(self, method, *args, **kwargs):
            call_args, call_kwargs = args, kwargs
            if trace._RECORDER is None or trace._RECORDER.failure is not None:
                return original(self, method, *args, **kwargs)
            # Prepare observation outside the native call. Never retry native RPC.
            try:
                if isinstance(method, str) and method in ('execute_model', 'sample_tokens', 'pp_push', 'pp_settle') and hasattr(self, '_sim_v2_session'):
                    native_args = args[1] if len(args) > 1 else kwargs.get('args', ())
                    output = native_args[0] if native_args and method == 'execute_model' else None
                    if method == 'execute_model':
                        self._sim_v2_step += 1
                        phase = 'real' if getattr(output, 'total_num_scheduled_tokens', 0) > 0 else 'state'
                        self._sim_v2_last = ((*self._sim_v2_session, self._sim_v2_step, phase)
                                             if self._sim_v2_step % self._sim_v2_every == 0 else None)
                    context = (self._sim_v2_last if method in ('execute_model', 'sample_tokens')
                               else (*self._sim_v2_session, -1, method))
                    if context is not None:
                        inner = args[2] if len(args) > 2 else kwargs.get('kwargs')
                        if trace.RPC_CONTEXT not in (inner or {}):
                            inner = dict(inner or {})
                            inner[trace.RPC_CONTEXT] = context
                            if len(args) > 2:
                                positional = list(args)
                                positional[2] = inner
                                call_args = tuple(positional)
                            else:
                                call_kwargs = dict(kwargs, kwargs=inner)
                            data = None
                            if method == 'execute_model':
                                requests = getattr(output, 'num_scheduled_tokens', {})
                                selected = tuple(rid for rid in islice(requests, 64)
                                                 if isinstance(rid, str) and len(rid) <= 256)
                                data = (len(requests), len(selected) < len(requests), selected)
                            trace.emit('rpc.' + method + '.dispatch', trace.Scope(context), data=data)
            except Exception:
                call_args, call_kwargs = args, kwargs
                trace._RECORDER.observer_errors += 1
                if method == 'execute_model':
                    self._sim_v2_last = None
            return original(self, method, *call_args, **call_kwargs)
        target.collective_rpc = rpc


class V2WorkerWrapperHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm.v1.worker.worker_base'
    HOOK_CLASS_NAME = 'WorkerWrapperBase'

    @classmethod
    def hook(cls, target):
        def transport_capabilities(self):
            callback = getattr(self.worker, '_sim_trace_v2_capabilities', None)
            return 2 if callback is not None and callback() == 2 else 0
        target._sim_trace_v2_transport_capabilities = transport_capabilities
        # WorkerProc resolves execute_model on this strict-signature wrapper,
        # before reaching TPUWorker. Consume transport metadata at this boundary.
        original = target.execute_model

        @wraps(original)
        def execute(self, *args, **kwargs):
            context = kwargs.get(trace.RPC_CONTEXT)
            if not _valid(context):
                return original(self, *args, **kwargs)
            kwargs = dict(kwargs)
            del kwargs[trace.RPC_CONTEXT]
            if trace._RECORDER is None or trace._RECORDER.failure is not None:
                return original(self, *args, **kwargs)
            scope = trace.Scope(context)
            token = trace.ACTIVE.set(scope)
            trace.emit('wrapper.execute_model.enter', scope)
            try:
                return original(self, *args, **kwargs)
            finally:
                trace.emit('wrapper.execute_model.exit', scope)
                trace.ACTIVE.reset(token)
        target.execute_model = execute


class V2WorkerHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm_torchtpu.worker.tpu_worker'
    HOOK_CLASS_NAME = 'TPUWorker'

    @classmethod
    def hook(cls, target):
        def capabilities(self):
            methods = ('execute_model', 'sample_tokens', 'pp_push', 'pp_settle')
            return 2 if all(getattr(getattr(getattr(self, name, None), '__func__', None),
                                    '_sim_v2_transport', False) for name in methods) else 0
        target._sim_trace_v2_capabilities = capabilities
        def ready(self):
            recorder = trace.initialize()
            if recorder is not None:
                recorder.identity = (int(os.getenv('RANK_ID', '-1')), getattr(self, 'rank', -1),
                                     getattr(self, 'local_rank', -1), -1, -1, -1, -1)
                trace.emit('startup.worker.ready', data=2)
        _init(target, ready)
        init_device = getattr(target, 'init_device', None)
        if init_device is not None:
            @wraps(init_device)
            def device(self, *args, **kwargs):
                result = init_device(self, *args, **kwargs)
                try:
                    from vllm.distributed import parallel_state as ps
                    recorder = trace._RECORDER
                    if recorder is not None:
                        recorder.identity = (int(os.getenv('RANK_ID', '-1')), self.rank, self.local_rank,
                            ps.get_pp_group().rank_in_group, ps.get_pcp_group().rank_in_group,
                            ps.get_tp_group().rank_in_group, ps.get_dp_group().rank_in_group)
                        trace.emit('startup.worker.device_ready', data=2)
                except Exception:
                    pass
                return result
            target.init_device = device
        for name in ('execute_model', 'sample_tokens', 'pp_push', 'pp_settle'):
            original = getattr(target, name, None)
            if original is None:
                continue

            def install(method, native):
                @wraps(native)
                def wrapped(self, *args, **kwargs):
                    context = kwargs.get(trace.RPC_CONTEXT)
                    scope = trace.ACTIVE.get()
                    token = None
                    if _valid(context):
                        kwargs = dict(kwargs)
                        del kwargs[trace.RPC_CONTEXT]
                        scope = trace.Scope(context)
                        token = trace.ACTIVE.set(scope)
                    if scope is None:
                        return native(self, *args, **kwargs)
                    if trace._RECORDER is None or trace._RECORDER.failure is not None:
                        try:
                            return native(self, *args, **kwargs)
                        finally:
                            if token is not None:
                                trace.ACTIVE.reset(token)
                    trace.emit('worker.' + method + '.enter', scope)
                    try:
                        return native(self, *args, **kwargs)
                    finally:
                        trace.emit('worker.' + method + '.exit', scope)
                        if token is not None:
                            trace.ACTIVE.reset(token)
                wrapped._sim_v2_transport = True
                return wrapped
            setattr(target, name, install(name, original))
        # Intentionally leave profile untouched: native TPU profiling survives.


class V2PPWaveHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm_torchtpu.distributed.pp_wave'
    HOOK_CLASS_NAME = 'PPWave'

    @classmethod
    def hook(cls, target):
        for name in ('before_step', 'start_forward', 'end_forward', 'push', 'settle'):
            original = getattr(target, name, None)
            if original is None:
                continue

            def install(method, native):
                @wraps(native)
                def wrapped(self, *args, **kwargs):
                    scope = trace.ACTIVE.get()
                    if scope is None:
                        return native(self, *args, **kwargs)
                    if method == 'start_forward':
                        scope.chunk += 1
                    trace.emit('pp.' + method + '.enter', scope)
                    try:
                        return native(self, *args, **kwargs)
                    finally:
                        trace.emit('pp.' + method + '.exit', scope)
                return wrapped
            setattr(target, name, install(name, original))


class V2RunnerTimingHook(BaseHook):
    HOOK_MODULE_NAME = 'vllm_torchtpu.runner.stage_timing'
    HOOK_CLASS_NAME = 'RunnerStageTiming'

    @classmethod
    def hook(cls, target):
        original = target.stage

        @contextmanager
        @wraps(original)
        def stage(self, *args, **kwargs):
            name = args[0] if args else kwargs.get('stage', kwargs.get('name'))
            scope = trace.ACTIVE.get()
            if not isinstance(name, str):
                scope = None
            # Use the native context unchanged; do not add device events/syncs.
            chunk = scope.chunk if scope is not None and name == 'main_forward' else -1
            if scope is not None:
                trace.emit('runner.' + name + '.enter', scope, chunk)
            try:
                with original(self, *args, **kwargs) as value:
                    yield value
            finally:
                if scope is not None:
                    trace.emit('runner.' + name + '.exit', scope, chunk)
        target.stage = stage


HOOKS = [V2ExecutorHook, V2WorkerWrapperHook, V2WorkerHook, V2PPWaveHook, V2RunnerTimingHook]
