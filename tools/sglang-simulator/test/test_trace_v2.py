"""Host-only invariants: never import torch or touch accelerator hardware."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace as NS
from types import ModuleType

import pytest

from sglang_simulator.collector.vllm_hook import trace_v2 as trace
from sglang_simulator.collector.vllm_hook.trace_v2_hooks import (
    V2ExecutorHook, V2PPWaveHook, V2RunnerTimingHook, V2WorkerHook, V2WorkerWrapperHook,
)


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    rec = trace.Recorder(tmp_path, capacity=256, interval=60)
    monkeypatch.setattr(trace, '_RECORDER', rec)
    yield rec
    rec.close()


def rows(rec):
    rec.close()
    return [json.loads(line) for line in (rec.directory / 'events.jsonl').read_text().splitlines()]


def make_chain(native_error=None):
    class Wave:
        def start_forward(self, rows):
            return rows
        def end_forward(self, tensors, rows):
            return tensors

    class Timing:
        calls = 0
        @contextmanager
        def stage(self, name):
            self.calls += 1
            yield 'native-context-value'

    wave, timing = Wave(), Timing()
    V2PPWaveHook.hook(Wave)
    V2RunnerTimingHook.hook(Timing)

    class Worker:
        rank, local_rank = 3, 0
        def __init__(self):
            self.calls = []
        def execute_model(self, output, tag=None):
            self.calls.append(('execute', output, tag))
            if native_error is not None:
                raise native_error
            for _ in range(2):
                wave.start_forward(1)
                with timing.stage('main_forward') as value:
                    assert value == 'native-context-value'
                wave.end_forward('tensors', 1)
            return output
        def sample_tokens(self, grammar):
            self.calls.append(('sample', grammar))
            return grammar
        def profile(self, is_start=True, profile_prefix=None):
            return (is_start, profile_prefix)
        def pp_push(self):
            return 'native-push'
        def pp_settle(self):
            return 'native-settle'

    native_profile = Worker.profile
    V2WorkerHook.hook(Worker)
    assert Worker.profile is native_profile
    worker = Worker()

    class Wrapper:
        def __init__(self, worker):
            self.worker = worker
        def execute_model(self, output, tag=None):
            return self.worker.execute_model(output, tag=tag)
        def __getattr__(self, name):
            return getattr(self.worker, name)
    V2WorkerWrapperHook.hook(Wrapper)

    class Executor:
        def __init__(self, worker):
            self.worker = Wrapper(worker)
            self.calls = []
        def collective_rpc(self, method, timeout=None, args=(), kwargs=None,
                           non_block=False, unique_reply_rank=None):
            if method == '_sim_trace_v2_transport_capabilities':
                return [getattr(self.worker, method)()]
            self.calls.append((method, timeout, non_block, unique_reply_rank))
            return getattr(self.worker, method)(*args, **(kwargs or {}))

    V2ExecutorHook.hook(Executor)
    return Executor(worker), worker, timing


def test_context_crosses_rpc_without_changing_native_arguments(recorder):
    executor, worker, timing = make_chain()
    output = NS(num_scheduled_tokens={'request-full-id-1': 2}, total_num_scheduled_tokens=2)
    native_kwargs = {'tag': 'kept'}
    assert executor.collective_rpc('execute_model', 17, (output,), native_kwargs, True) is output
    grammar = object()
    assert executor.collective_rpc('sample_tokens', args=(grammar,), unique_reply_rank=3) is grammar
    assert native_kwargs == {'tag': 'kept'}
    assert executor.calls == [('execute_model', 17, True, None), ('sample_tokens', None, False, 3)]
    assert worker.calls == [('execute', output, 'kept'), ('sample', grammar)]
    assert trace.ACTIVE.get() is None and timing.calls == 2
    data = rows(recorder)
    dispatch = next(r for r in data if r['event'] == 'rpc.execute_model.dispatch')
    assert dispatch['data'] == [1, False, ['request-full-id-1']]
    contexts = {tuple(r['context'][:3]) for r in data if r['context']}
    assert len({ctx for ctx in contexts if ctx[2] != 0}) == 1
    forward = [r for r in data if r['event'] == 'runner.main_forward.enter']
    assert [r['chunk'] for r in forward] == [0, 1]


def test_native_exception_is_identical_and_never_retried(recorder):
    error = RuntimeError('native exception')
    executor, worker, _ = make_chain(error)
    output = NS(num_scheduled_tokens={'r': 1}, total_num_scheduled_tokens=1)
    with pytest.raises(RuntimeError) as caught:
        executor.collective_rpc('execute_model', args=(output,))
    assert caught.value is error and len(worker.calls) == 1
    assert trace.ACTIVE.get() is None
    assert next(r for r in rows(recorder) if r['event'] == 'rpc.execute_model.dispatch')['context'][3] == 'real'


def test_native_profile_is_preserved_and_not_instrumented(recorder):
    executor, _, _ = make_chain()
    before = recorder.accepted
    assert executor.collective_rpc('profile', args=(False, 'prefix')) == (False, 'prefix')
    assert recorder.accepted == before


def test_strict_native_wrapper_consumes_only_transport_context(recorder):
    value = object()
    class Wrapper:
        def execute_model(self, scheduler_output):
            self.received = scheduler_output
            return value
    V2WorkerWrapperHook.hook(Wrapper)
    wrapper = Wrapper()
    output = object()
    context = ('collection', 'engine', 1, 'real')
    assert wrapper.execute_model(output, **{trace.RPC_CONTEXT: context}) is value
    assert wrapper.received is output and trace.ACTIVE.get() is None
    assert len(rows(recorder)) == 2


def test_stage_keyword_arguments_and_yield_value_are_unchanged(recorder):
    class Timing:
        @contextmanager
        def stage(self, stage, option=None):
            self.received = (stage, option)
            yield 'native-value'
    V2RunnerTimingHook.hook(Timing)
    timing = Timing()
    token = trace.ACTIVE.set(trace.Scope(('collection', 'engine', 1, 'real')))
    try:
        with timing.stage(stage='main_forward', option='kept') as value:
            assert value == 'native-value'
    finally:
        trace.ACTIVE.reset(token)
    assert timing.received == ('main_forward', 'kept')
    assert len(rows(recorder)) == 2


def test_old_worker_cohort_does_not_receive_reserved_kwargs(recorder):
    class Executor:
        world_size = 1
        def __init__(self):
            self.calls = []
        def collective_rpc(self, method, *args, **kwargs):
            if method == '_sim_trace_v2_transport_capabilities':
                return [0]
            self.calls.append((args, kwargs))
            return 'native-result'
    V2ExecutorHook.hook(Executor)
    executor = Executor()
    assert not hasattr(executor, '_sim_v2_session')
    assert executor.collective_rpc('execute_model', args=('native',)) == 'native-result'
    assert executor.calls == [((), {'args': ('native',)})]


def test_unwrapped_extension_override_disables_context_injection(recorder):
    executor, worker, _ = make_chain()
    class Extension(type(worker)):
        def sample_tokens(self, grammar):
            return grammar
    extension = Extension()
    passive = type(executor)(extension)
    assert not hasattr(passive, '_sim_v2_session')
    assert passive.collective_rpc('sample_tokens', args=('native',)) == 'native'


def test_request_binding_is_bounded_and_reports_truncation(recorder):
    executor, _, _ = make_chain()
    output = NS(num_scheduled_tokens={f'r{i}': 1 for i in range(100)}, total_num_scheduled_tokens=100)
    executor.collective_rpc('execute_model', args=(output,))
    dispatch = next(r for r in rows(recorder) if r['event'] == 'rpc.execute_model.dispatch')
    assert dispatch['data'][0:2] == [100, True]
    assert len(dispatch['data'][2]) == 64


def test_sampling_is_decided_once_per_step_without_scope_leak(recorder, monkeypatch):
    monkeypatch.setenv('SIM_TRACE_EVERY', '2')
    executor, worker, _ = make_chain()
    output = NS(num_scheduled_tokens={'r': 1}, total_num_scheduled_tokens=1)
    for _ in range(3):
        executor.collective_rpc('execute_model', args=(output,))
        executor.collective_rpc('sample_tokens', args=('grammar',))
    assert len(worker.calls) == 6 and trace.ACTIVE.get() is None
    assert {r['context'][2] for r in rows(recorder) if r['context'] and r['context'][3] != 'startup'} == {2}


def test_lock_contention_drops_without_waiting(tmp_path):
    rec = trace.Recorder(tmp_path, interval=60)
    done = threading.Event()
    rec._lock.acquire()
    thread = threading.Thread(target=lambda: (rec.emit('busy'), done.set()))
    thread.start()
    try:
        assert done.wait(.5), 'producer waited for a collector lock'
    finally:
        rec._lock.release()
        thread.join()
        rec.close()
    assert rec.dropped_busy == 1 and rec.accepted == 0


def test_full_buffer_is_bounded(tmp_path):
    rec = trace.Recorder(tmp_path, capacity=2, interval=60)
    assert rec.emit('one') and rec.emit('two')
    assert not rec.emit('three')
    assert len(rec._events) == 2 and rec.dropped_full == 1
    assert len(rows(rec)) == 2


def test_writer_failure_stops_collection_without_raising(tmp_path):
    blocked = tmp_path / 'file'
    blocked.write_text('not a directory')
    rec = trace.Recorder(blocked, interval=.001)
    rec._thread.join(timeout=1)
    assert rec.failure is not None
    assert not rec.emit('discard') and rec.rejected == 1
    rec.close()


def test_disk_quota_is_bounded_and_reported(tmp_path):
    rec = trace.Recorder(tmp_path, interval=60, max_bytes=1)
    rec.emit('event')
    rec.close()
    assert rec.failure == 'byte_limit' and rec.bytes_written == 0


def test_recorder_clock_failure_does_not_break_native_call(recorder, monkeypatch):
    executor, worker, _ = make_chain()
    output = NS(num_scheduled_tokens={'r': 1}, total_num_scheduled_tokens=1)
    def fail():
        raise RuntimeError('observer failure')
    with monkeypatch.context() as patch:
        patch.setattr(trace.time, 'monotonic_ns', fail)
        assert executor.collective_rpc('execute_model', args=(output,)) is output
    assert len(worker.calls) == 1 and recorder.observer_errors > 0


def test_empty_polls_do_not_create_events_or_change_real_sampling(recorder, monkeypatch):
    monkeypatch.setenv('SIM_TRACE_EVERY', '2')
    executor, worker, _ = make_chain()
    empty = NS(num_scheduled_tokens={}, total_num_scheduled_tokens=0)
    real = NS(num_scheduled_tokens={'r': 1}, total_num_scheduled_tokens=1)
    baseline = recorder.accepted
    for _ in range(1000):
        assert executor.collective_rpc('execute_model', args=(empty,)) is empty
    assert recorder.accepted == baseline and len(worker.calls) == 1000
    executor.collective_rpc('execute_model', args=(real,))
    executor.collective_rpc('sample_tokens', args=('grammar',))
    assert recorder.accepted == baseline
    executor.collective_rpc('execute_model', args=(empty,))
    executor.collective_rpc('execute_model', args=(real,))
    executor.collective_rpc('sample_tokens', args=('grammar',))
    dispatch = [r for r in rows(recorder) if r['event'] == 'rpc.execute_model.dispatch']
    assert len(dispatch) == 1 and dispatch[0]['context'][2:] == [1003, 'real']
    assert trace.ACTIVE.get() is None and len(worker.calls) == 1005


def test_state_poll_instrumentation_requires_explicit_opt_in(recorder, monkeypatch):
    monkeypatch.setenv('SIM_TRACE_INCLUDE_STATE', '1')
    executor, _, _ = make_chain()
    empty = NS(num_scheduled_tokens={}, total_num_scheduled_tokens=0)
    assert executor.collective_rpc('execute_model', args=(empty,)) is empty
    dispatch = next(r for r in rows(recorder) if r['event'] == 'rpc.execute_model.dispatch')
    assert dispatch['context'][3] == 'state'


def test_fork_reset_does_not_reuse_parent_writer(recorder):
    trace._after_fork()
    assert trace._RECORDER is None and trace.ACTIVE.get() is None


def test_startup_events_prove_worker_group_identity_without_inference(recorder, monkeypatch):
    parent = ModuleType('vllm')
    distributed = ModuleType('vllm.distributed')
    distributed.parallel_state = NS(**{
        f'get_{name}_group': (lambda rank=rank: NS(rank_in_group=rank))
        for name, rank in [('pp', 2), ('pcp', 1), ('tp', 0), ('dp', 0)]})
    monkeypatch.setitem(sys.modules, 'vllm', parent)
    monkeypatch.setitem(sys.modules, 'vllm.distributed', distributed)
    monkeypatch.setenv('RANK_ID', '2')
    class Worker:
        rank, local_rank = 17, 1
        def __init__(self):
            self.native_init_calls = 0
        def init_device(self):
            self.native_init_calls += 1
            return 'native-device-result'
    V2WorkerHook.hook(Worker)
    worker = Worker()
    assert worker.init_device() == 'native-device-result'
    assert worker.native_init_calls == 1
    data = rows(recorder)
    assert [r['event'] for r in data] == ['startup.worker.ready', 'startup.worker.device_ready']
    assert data[-1]['rank'] == {'node': 2, 'global': 17, 'local': 1, 'pp': 2, 'pcp': 1, 'tp': 0, 'dp': 0}


def test_follower_startup_does_not_attempt_broadcast(recorder):
    class Executor:
        rpc_broadcast_mq = None
        def __init__(self):
            self.calls = 0
        def collective_rpc(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError('follower must not broadcast')
    V2ExecutorHook.hook(Executor)
    executor = Executor()
    assert executor.calls == 0 and not hasattr(executor, '_sim_v2_session')
    assert rows(recorder)[0]['event'] == 'startup.controller.passive'


def test_bootstrap_v2_does_not_load_torch_or_legacy_hook():
    env = dict(os.environ, SIM_COLLECTOR_ENABLE='1', SIM_COLLECTOR_MODE='pp_v2')
    code = ('import sglang_simulator._pth_bootstrap; import sys; '
            'assert "torch" not in sys.modules; '
            'assert "sglang_simulator.collector.vllm_hook.worker_hook" not in sys.modules')
    subprocess.run([sys.executable, '-S', '-c', code], env=env, check=True, timeout=10)
