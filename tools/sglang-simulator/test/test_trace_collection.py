"""Input/cut/timing contracts, without importing a device runtime."""
from contextlib import contextmanager
import json
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import pytest

from sglang_simulator.collector.vllm_hook import trace_v2 as trace
from sglang_simulator.collector.vllm_hook.trace_collection import CollectionRecorder
from sglang_simulator.collector.vllm_hook.trace_collection_hooks import (
    CollectionExecutorHook, CollectionWorkerHook, CollectionRunnerHook,
    CollectionPcpLayoutHook, CollectionLazyStepperHook, CollectionSchedulerHook,
    step_snapshot,
)


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    rec = CollectionRecorder(tmp_path, capacity=512, interval=60, max_bytes=1024)
    rec.identity = (0, 0, 0, 0, 0, 0, 0)
    monkeypatch.setattr(trace, '_RECORDER', rec)
    yield rec
    rec.close()


def exported(receipt, name='events'):
    root = Path(receipt['path'])
    files = sorted(root.glob('events.*.jsonl')) if name == 'events' else [root / (name + '.jsonl')]
    return [json.loads(line) for p in files for line in p.read_text().splitlines()]


def scope(step=1):
    return trace.Scope(('collection', 'engine', step, 'real'))


def test_both_profile_triggers_cut_without_stopping_collection(recorder):
    trace.emit('before')
    first = recorder.export_cut('a1', True)
    trace.emit('between')
    second = recorder.export_cut('a2', False)
    assert second['complete']
    # start is async; stop serializes behind its cut in the writer queue.
    first['path'] = str(recorder.directory / 'exports' / 'a1')
    assert [r['event'] for r in exported(first)] == ['before']
    assert [r['event'] for r in exported(second)] == ['between']
    trace.emit('after_stop')
    third = recorder.export_cut('a3', False)
    assert [r['event'] for r in exported(third)] == ['after_stop']
    assert recorder.failure is None


def test_byte_limit_rotates_instead_of_silently_stopping(recorder):
    for i in range(30):
        trace.emit('event', data=i)
    receipt = recorder.export_cut('rotate', False)
    assert receipt['complete'] and len(exported(receipt)) == 30
    assert len(list(Path(receipt['path']).glob('events.*.jsonl'))) > 1
    assert recorder.failure is None and recorder.dropped_full == 0


def test_span_crossing_export_cut_is_joined_by_original_identity(recorder):
    s = scope()
    trace.emit('runner.main_forward.enter', s, 0)
    first = recorder.export_cut('begin', False)
    trace.emit('runner.main_forward.exit', s, 0)
    second = recorder.export_cut('end', False)
    assert not exported(first, 'schedule_batch')
    row = exported(second, 'schedule_batch')[0]
    assert row['valid'] and row['start_event_seq'] == 1
    assert row['iter_latency'] >= 0 and row['time_kind'] == 'host_main_forward'
    assert row['device_start_ns'] is None


def test_full_step_snapshot_over_64_ids_and_preexecution_past():
    requests = [NS(req_id='request-' + str(i), num_computed_tokens=i) for i in range(100)]
    output = NS(scheduled_new_reqs=requests, scheduled_cached_reqs=NS(req_ids=[]),
                num_scheduled_tokens={r.req_id: i+1 for i, r in enumerate(requests)},
                total_num_scheduled_tokens=5050)
    snapshot = step_snapshot(output)
    requests[0].num_computed_tokens = 999
    output.num_scheduled_tokens.clear()
    assert snapshot['request_count'] == 100 and snapshot['query_sum_valid']
    assert snapshot['requests'][0]['past_length'] == 0
    assert snapshot['requests'][-1]['req_id'] == 'request-99'


def test_missing_past_is_null_not_zero():
    output = NS(scheduled_new_reqs=[], scheduled_cached_reqs=NS(req_ids=['r']),
                num_scheduled_tokens={'r': 1}, total_num_scheduled_tokens=1)
    data = step_snapshot(output)
    assert data['requests'][0]['past_length'] is None and not data['past_complete']


def test_profile_fanout_receives_same_export_id_without_hardware_profiler(recorder):
    class Worker:
        def profile(self, *args, **kwargs):
            raise AssertionError('hardware profiler must not be called')
        def execute_model(self, *args, **kwargs):
            return None
        def sample_tokens(self):
            return None
    CollectionWorkerHook.hook(Worker)
    worker = Worker()
    class Executor:
        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            self.received = (method, args, kwargs)
            return getattr(worker, method)(*args, **(kwargs or {}))
    CollectionExecutorHook.hook(Executor)
    ex = Executor()
    receipt = ex.stop_profile('label', {'unused': 'kept'})
    assert receipt['complete']
    assert ex.received[0:2] == ('profile', (False, 'label', {'unused': 'kept'}))
    assert ex.received[2]['_sim_collection_export_id'] == receipt['export_id']


def test_async_output_completion_is_observed_only_after_native_get_output(recorder):
    calls = []
    class Output:
        def get_output(self):
            calls.append('native_get')
            return 'result'
    class Worker:
        def execute_model(self):
            return None
        def sample_tokens(self):
            return Output()
    CollectionWorkerHook.hook(Worker)
    worker = Worker()
    token = trace.ACTIVE.set(scope(3))
    worker.execute_model()
    result = worker.sample_tokens()
    trace.ACTIVE.reset(token)
    first = recorder.export_cut('beforeget', False)
    assert not exported(first, 'step_timings')
    assert result.get_output() == 'result'
    assert result.get_output() == 'result'
    second = recorder.export_cut('afterget', False)
    data = exported(second, 'step_timings')
    assert len(data) == 1 and calls == ['native_get', 'native_get']
    assert data[0]['context'][2] == 3 and data[0]['time_kind'] == 'host_output_ready_observed'
    assert data[0]['device_elapsed_ns'] is None


def test_native_exception_is_preserved_and_pending_timing_cleared(recorder):
    error = RuntimeError('native')
    class Worker:
        def execute_model(self):
            raise error
    CollectionWorkerHook.hook(Worker)
    worker = Worker()
    token = trace.ACTIVE.set(scope())
    try:
        with pytest.raises(RuntimeError) as caught:
            worker.execute_model()
        assert caught.value is error and worker._sim_collection_pending is None
    finally:
        trace.ACTIVE.reset(token)


def test_actual_runner_window_uses_batch_order_and_snapshots_mutable_metadata(recorder):
    class Runner:
        num_reqs_max_model_len = 2
        num_reqs_most_model_len = None
        def _prepare_inputs(self, output, start_index, *args):
            self.input_batch.num_computed_tokens_cpu[0] = 999
            return (None, None, 2, 2, start_index+2)
    CollectionRunnerHook.hook(Runner)
    runner = Runner()
    runner.input_batch = NS(num_reqs=3, req_ids=['b', 'a', 'c'],
                           num_computed_tokens_cpu=[8, 12, 0], num_prompt_tokens=[16,24,32])
    runner.requests = {rid: NS(output_token_ids=[], sampling_params=NS(max_tokens=16, logprobs=None))
                       for rid in ('a','b','c')}
    runner._last_sequence_layout_plan = NS(global_num_tokens=20, global_padded_num_tokens=32,
                                          local_num_tokens=4, local_padded_num_tokens=4)
    output = NS(num_scheduled_tokens={'a':12, 'b':8, 'c':16})
    s = scope();token = trace.ACTIVE.set(s)
    try:
        result = runner._prepare_inputs(output, 0)
        assert result[4] == 2 and s.chunk == -1  # PPWave owns the increment.
    finally:
        trace.ACTIVE.reset(token)
    row = exported(recorder.export_cut('inputs', False), 'chunk_inputs')[0]
    assert row['chunk'] == 0 and row['total_query_tokens'] == 20
    assert [r['req_id'] for r in row['requests']] == ['b','a']
    assert [r['past_length'] for r in row['requests']] == [8,12]
    assert [r['input_length'] for r in row['requests']] == [16,24]


def test_pcp_layout_arrays_are_copied_without_device_reads(recorder):
    result = NS(pcp_size=8, interleave_size=2, local_token_counts=[4,4,2,2,2,2,2,2],
                local_padded_tokens=4, global_padded_tokens=32,
                token_owner_start_offsets_per_req=[0,12], absolute_query_start_offsets_per_req=[12,8])
    class Eligibility:
        def _evaluate_streaming_query(self):
            return result
    CollectionPcpLayoutHook.hook(Eligibility)
    token = trace.ACTIVE.set(scope())
    try:
        assert Eligibility()._evaluate_streaming_query() is result
    finally:
        trace.ACTIVE.reset(token)
    result.local_token_counts[0] = 999
    row = exported(recorder.export_cut('layout', False), 'layouts')[0]
    assert row['local_valid_counts'][0] == 4 and row['owner_start_offsets'] == [0,12]


def test_same_drain_reply_is_explicitly_invalid(recorder):
    class Stepper:
        drains = 2
        in_flight = [1,2]
        def _dispatch(self):
            return 'dispatch'
        def _sampled(self, slot):
            return slot
        def _process(self, step):
            return step
    CollectionLazyStepperHook.hook(Stepper)
    slot = NS(started_last=1.0, started_drain=2, dispatched=0.5)
    assert Stepper()._sampled(slot) is slot
    row = exported(recorder.export_cut('reply', False), 'engine_steps')[0]
    assert row['data']['forward_s'] is None and not row['data']['reply_interval_valid']
    assert row['data']['in_flight'] == 2


def test_process_observation_is_not_mislabeled_as_a_forward_duration(recorder):
    class Stepper:
        drains = 3
        in_flight = []
        def _dispatch(self): return False
        def _sampled(self, slot): return None
        def _process(self, step): return step
    CollectionLazyStepperHook.hook(Stepper)
    step = NS(slot=NS(started_last=1.0, started_drain=2, dispatched=0.5))
    assert Stepper()._process(step) is step
    row = exported(recorder.export_cut('process', False), 'engine_steps')[0]
    assert row['kind'] == 'engine.process.enter'
    assert row['data']['forward_s'] is None and not row['data']['reply_interval_valid']


def test_scheduler_subclass_records_once_and_uses_unadvanced_cache_snapshot(recorder):
    class Scheduler:
        requests = {'r': NS(num_prompt_tokens=24, max_tokens=8, num_computed_tokens=999)}
        def add_request(self, request):
            return request
        def _free_request(self, request):
            return request
        def schedule(self, throttle=False):
            return NS(scheduled_new_reqs=[NS(req_id='r', num_computed_tokens=12)])
    class TpuScheduler(Scheduler):
        def schedule(self, throttle=False):
            return super().schedule(throttle)
    from sglang_simulator.collector.vllm_hook.trace_collection_hooks import CollectionTpuSchedulerHook
    CollectionSchedulerHook.hook(Scheduler)
    CollectionTpuSchedulerHook.hook(TpuScheduler)
    TpuScheduler().schedule(True)
    data = exported(recorder.export_cut('scheduler', False), 'requests')
    assert len(data) == 1 and data[0]['total_kv_hit_len'] == 12
    assert data[0]['ext_kv_hit_len'] is None and not data[0]['cache_split_valid']


def test_slot_context_is_attached_before_reply_callbacks(recorder):
    class Stepper:
        executor = NS(_sim_v2_last=('collection','engine',7,'real'),
                      _sim_collection_last_method='sample_tokens')
        drains = 3
        in_flight = []
        def _add_slot(self, slot):
            self.slot = slot
        def _sampled(self, slot):
            return None
        def _dispatch(self):
            return True
        def _process(self, step):
            return None
    CollectionLazyStepperHook.hook(Stepper)
    stepper = Stepper()
    slot = NS(started_last=None, started_drain=None, dispatched=1.0)
    stepper._add_slot(slot)
    stepper._sampled(slot)
    row = exported(recorder.export_cut('slot', False), 'engine_steps')[0]
    assert row['context'][2] == 7


def test_bootstrap_composes_first_match_hooks_instead_of_losing_new_behavior(recorder, monkeypatch):
    from sglang_simulator.collector.vllm_hook.trace_collection_hooks import HOOKS
    from sglang_simulator.hook import class_hook_entry as entry
    import builtins
    # Execute the actual first-match class builder with a real module identity.
    monkeypatch.setattr(entry, 'CLASS_HOOKS', HOOKS)
    namespace = {'__name__': 'vllm_torchtpu.worker.tpu_worker'}
    native_builder = builtins.__build_class__
    try:
        builtins.__build_class__ = entry._custom_build_class_
        exec('''class TPUWorker:
    rank = local_rank = 0
    def __init__(self): pass
    def execute_model(self, output): return output
    def sample_tokens(self): return None
    def pp_push(self): return None
    def pp_settle(self): return None
    def profile(self, *args, **kwargs): raise AssertionError('hardware profiler')
''', namespace)
    finally:
        builtins.__build_class__ = native_builder
    worker = namespace['TPUWorker']()
    assert worker._sim_trace_v2_capabilities() == 2
    receipt = worker.profile(False)
    assert receipt['complete']
    assert len(HOOKS) == len({(h.HOOK_MODULE_NAME,h.HOOK_CLASS_NAME) for h in HOOKS})


def test_repeated_export_id_is_idempotent(recorder):
    trace.emit('before')
    first = recorder.export_cut('repeat', False)
    trace.emit('after')
    assert recorder.export_cut('repeat', False) == first
    second = recorder.export_cut('next', False)
    assert [r['event'] for r in exported(first)] == ['before']
    assert [r['event'] for r in exported(second)] == ['after']


def test_idle_dispatches_do_not_generate_events(recorder):
    class Stepper:
        def _dispatch(self):
            return False
        def _sampled(self, slot):
            return None
        def _process(self, step):
            return None
    CollectionLazyStepperHook.hook(Stepper)
    for _ in range(1000):
        assert Stepper()._dispatch() is False
    assert recorder.accepted == 0


def test_export_failure_is_reported_without_publishing_a_complete_directory(recorder, monkeypatch):
    def fail(window):
        raise OSError('simulated disk failure')
    monkeypatch.setattr(recorder, '_tables', fail)
    trace.emit('kept_in_pending')
    receipt = recorder.export_cut('failed', False, timeout=2)
    assert not receipt['complete'] and receipt['failure'] == 'OSError'
    assert not (recorder.directory / 'exports' / 'failed').exists()
    assert list((recorder.directory / 'pending').glob('*/events.*.jsonl'))


def test_rpc_worker_timeline_inputs_and_profile_export_integrate(recorder):
    from test_trace_v2 import make_chain
    executor, worker, timing = make_chain()
    CollectionExecutorHook.hook(type(executor))
    CollectionWorkerHook.hook(type(worker))
    executor.start_profile()
    requests = [NS(req_id='r' + str(i), num_computed_tokens=i) for i in range(100)]
    output = NS(scheduled_new_reqs=requests, scheduled_cached_reqs=NS(req_ids=[]),
                num_scheduled_tokens={r.req_id: 2 for r in requests},
                total_num_scheduled_tokens=200)
    assert executor.collective_rpc('execute_model', args=(output,)) is output
    assert executor.collective_rpc('sample_tokens', args=('grammar',)) == 'grammar'
    receipt = executor.stop_profile()
    assert receipt['complete']
    data = exported(receipt, 'step_inputs')
    assert len(data) == 1 and len(data[0]['requests']) == 100
    assert data[0]['query_sum_valid'] and data[0]['past_complete']
    batches = exported(receipt, 'schedule_batch')
    assert {row['chunk'] for row in batches} == {0,1}
    assert all(row['valid'] and row['device_start_ns'] is None for row in batches)
    timings = exported(receipt, 'step_timings')
    assert len(timings) == 1 and timings[0]['context'] == data[0]['context']


def test_stop_profile_propagates_export_failure_instead_of_false_success(recorder):
    class Executor:
        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            return [{'complete': False, 'failure': 'disk_full'}]
    CollectionExecutorHook.hook(Executor)
    with pytest.raises(RuntimeError, match='export incomplete'):
        Executor().collective_rpc('profile', None, (False, None), None)


def test_stop_rejects_missing_export_receipt_from_an_uninstrumented_worker(recorder):
    class Executor:
        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            return [None]
    CollectionExecutorHook.hook(Executor)
    with pytest.raises(RuntimeError, match='export incomplete'):
        Executor().stop_profile()


def make_stage_exports(tmp_path, missing_rank=None, mismatched_clock=False):
    for pcp in range(8):
        if pcp == missing_rank:
            continue
        directory = tmp_path / f'process{pcp}' / 'exports' / 'cut'
        directory.mkdir(parents=True)
        info = {'host_clock': {'source':'time.monotonic_ns', 'boot_id':'boot',
                              'time_namespace': 'different' if mismatched_clock and pcp == 7 else 'time:[1]'},
                'health': {}}
        (directory/'export.json').write_text(json.dumps(info))
        base = {'context':['collection','engine',1,'real'], 'chunk':0,
                'rank':{'pp':0,'pcp':pcp}, 'mono_ns':200, 'seq':pcp+1}
        span = {**base, 'valid':True, 'start_mono_ns':100+pcp,
                'end_mono_ns':200+pcp, 'iter_latency':1e-7}
        (directory/'schedule_batch.jsonl').write_text(json.dumps(span)+'\n')
        layout = {**base, 'pcp_size':8, 'local_valid_counts':[2]*8}
        (directory/'layouts.jsonl').write_text(json.dumps(layout)+'\n')
        if pcp == 0:
            inputs = {**base,'input_complete':True,'total_query_tokens':16,
                      'requests':[{'req_id':'r','query_length':16,'past_length':8}]}
            (directory/'chunk_inputs.jsonl').write_text(json.dumps(inputs)+'\n')


def test_stage_dataset_joins_all_ranks_and_exact_input_features(tmp_path):
    from sglang_simulator.collector.vllm_hook.trace_dataset import build_samples
    make_stage_exports(tmp_path)
    samples, report = build_samples(tmp_path, 8)
    assert report['valid_host_samples'] == 1 and report['valid_device_samples'] == 0
    sample = samples[0]
    assert sample['start_mono_ns'] == 100 and sample['end_mono_ns'] == 207
    assert sample['causal_attention_token_pairs'] == 16*8+16*17//2
    assert sample['device_start_ns'] is None


def test_stage_dataset_rejects_missing_rank(tmp_path):
    from sglang_simulator.collector.vllm_hook.trace_dataset import build_samples
    make_stage_exports(tmp_path, missing_rank=7)
    samples, report = build_samples(tmp_path, 8)
    assert report['valid_host_samples'] == 0
    assert 'incomplete_pcp_spans' in samples[0]['issues']
    assert samples[0]['latency_ms'] is None


def test_stage_dataset_does_not_subtract_unmapped_clock_domains(tmp_path):
    from sglang_simulator.collector.vllm_hook.trace_dataset import build_samples
    make_stage_exports(tmp_path, mismatched_clock=True)
    samples, report = build_samples(tmp_path, 8)
    assert report['valid_host_samples'] == 0
    assert samples[0]['latency_ms'] is None
    assert 'unverified_shared_host_clock' in samples[0]['issues']


def test_failed_forward_preserves_exception_and_is_not_a_valid_latency_sample(recorder):
    from sglang_simulator.collector.vllm_hook.trace_collection_hooks import CollectionRunnerTimingHook
    class Timing:
        @contextmanager
        def stage(self, name):
            yield 'native'
    CollectionRunnerTimingHook.hook(Timing)
    s = scope();s.chunk = 0
    token = trace.ACTIVE.set(s)
    error = RuntimeError('forward failed')
    try:
        with pytest.raises(RuntimeError) as caught:
            with Timing().stage('main_forward') as value:
                assert value == 'native'
                raise error
        assert caught.value is error
    finally:
        trace.ACTIVE.reset(token)
    row = exported(recorder.export_cut('forwarderror', False), 'schedule_batch')[0]
    assert not row['valid'] and row['native_ok'] is False
