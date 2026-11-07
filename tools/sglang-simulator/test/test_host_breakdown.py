"""CPU accounting must preserve existing native execution and failures."""
import time
import pytest
from sglang_simulator.collector.vllm_hook import host_breakdown as host
from sglang_simulator.collector.vllm_hook import native_boundaries as boundary
from sglang_simulator.collector.vllm_hook import stage_collection as stage
from test_native_boundaries import rig, active, forward


def test_blocked_thread_is_not_reported_as_device_time(monkeypatch):
    monkeypatch.setattr(host, 'ENABLED', True)
    start = host.snapshot()
    time.sleep(.02)
    data = host.elapsed(start)
    assert data['call_wall_ns'] >= 15_000_000
    assert data['calling_thread_offcpu_ns'] > data['call_wall_ns'] * .5
    assert data['cpu_accounting_kind'] == 'host_clocks_not_device_time'
    assert 'device_elapsed_ns' not in data


def test_missing_clock_is_fail_open(monkeypatch):
    monkeypatch.setattr(host, 'ENABLED', True)
    monkeypatch.setattr(host.time, 'thread_time_ns', lambda: 1/0)
    assert host.snapshot() is None
    assert host.elapsed(None) == {}


def test_enabled_accounting_keeps_native_call_order(rig, monkeypatch):
    monkeypatch.setattr(host, 'ENABLED', True)
    forward(rig.wave(0), active())
    assert rig.calls == [('sync', True, True), ('barrier', 'world32'),
                         ('sync', False, True), ('sync', False, False)]
    ready = next(e for e in rig.events if e[0] == 'native.local_drain.ready')[3]
    assert ready['call_wall_ns'] >= 0
    assert ready['calling_thread_cpu_ns'] >= 0
    assert ready['process_cpu_ns'] >= 0
    assert ready['call_end_mono_ns'] >= ready['call_start_mono_ns']
    assert ready['isolated_chunk_execution'] is False


def test_failed_native_call_not_retried(rig, monkeypatch):
    monkeypatch.setattr(host, 'ENABLED', True)
    rig.sync.fail = True
    with pytest.raises(RuntimeError, match='device failure'):
        forward(rig.wave(0), active())
    assert rig.calls == [('sync', True, True)]
    error = next(e for e in rig.events if e[0] == 'native.local_drain.error')[3]
    assert error['call_wall_ns'] >= 0
    assert boundary._SYNC.get() is None


def test_host_points_are_opt_in_and_exported(tmp_path, monkeypatch):
    worker = (-1, 0, 0, 1, 0, 0, 0)
    monkeypatch.setattr(host, 'ENABLED', False)
    assert not stage.keep_event('pp.before_step.enter', worker)
    monkeypatch.setattr(host, 'ENABLED', True)
    assert stage.keep_event('pp.before_step.enter', worker)
    assert not stage.keep_event('pp.before_step.enter', (-1, 1, 1, 1, 1, 0, 0))
    import json
    row = dict(event='pp.before_step.enter', data={}, context=None)
    (tmp_path/'events.000.jsonl').write_text(json.dumps(row)+'\n')
    assert stage.export_tables(tmp_path)['stage_points'] == 1


def test_deferred_output_never_borrows_later_step_identity(rig, monkeypatch):
    monkeypatch.setattr(host, 'ENABLED', True)
    monkeypatch.setenv('SIM_TPU_NATIVE_BOUNDARIES', '1')
    class Output:
        def __init__(self): self.copy_state = object()
        def get_output(self): return 'original'
    boundary.install_output_class(Output)
    wave=rig.wave(3)
    s=active(10);forward(wave,s)
    token=boundary.OUTPUT_WAVE.set(wave)
    try: output=Output()
    finally: boundary.OUTPUT_WAVE.reset(token)
    # Later forward can be visible when an earlier async getter completes.
    forward(wave,active(11));rig.events.clear()
    assert output.get_output()=='original'
    assert not any(e[0] in ('native.output_get.return','native.output_ready_observed') for e in rig.events)
