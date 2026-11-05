"""Native call preservation and deferred/async chunk attribution contracts."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace as NS

import pytest

from sglang_simulator.collector.vllm_hook import native_boundaries as boundary
from sglang_simulator.collector.vllm_hook import trace_v2 as trace
from sglang_simulator.collector.vllm_hook.trace_collection import CollectionRecorder
from sglang_simulator.collector.vllm_hook.trace_v2_hooks import V2PPWaveHook


@pytest.fixture
def rig(tmp_path, monkeypatch):
    calls, events = [], []
    def synchronize(tensors, wait=False):
        calls.append(('sync', wait, tensors is None))
        if getattr(sync, 'fail', False):
            raise RuntimeError('device failure')
    sync = NS(synchronize=synchronize)
    dist = ModuleType('torch.distributed')
    def barrier(group):
        calls.append(('barrier', group))
        events.append(('BARRIER', None, None, {}))
    dist.barrier = barrier
    torch = ModuleType('torch'); torch.distributed = dist
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch.distributed', dist)
    sync_parent = ModuleType('torch_tpu._internal.sync'); sync_parent.sync = sync
    monkeypatch.setitem(sys.modules, 'torch_tpu._internal.sync', sync_parent)
    ps = ModuleType('vllm.distributed.parallel_state')
    ps.get_world_group = lambda: NS(cpu_group='world32')
    monkeypatch.setitem(sys.modules, 'vllm.distributed.parallel_state', ps)
    # Exactly the deployed _drain_slice source; installation's fingerprint guard
    # must accept it. The surrounding stub models only PP control, not kernels.
    drain = (Path(__file__).parent / 'native_drain_fixture.txt').read_text()
    stub = '''
from torch_tpu._internal.sync import sync
def _launch_now(op, carried):
    sync.synchronize(None, wait=False)
    out = op(*carried)
    sync.synchronize(list(out), wait=False)
    return out
def _flush(tensors):
    sync.synchronize(list(tensors), wait=False)
def _wait(tensors):
    sync.synchronize(list(tensors), wait=True)
class PPWave:
    def __init__(self, rank=1, async_handoff=False):
        self.rank, self.last = rank, 3
        self.async_handoff = async_handoff
        self.drain_before_handoff = True
        self.pending = None
        self.op = lambda *x: x
    def _launch_blocking(self, carried):
        if self.drain_before_handoff:
            _drain_slice()
        return _launch_now(self.op, carried)
    def _launch_async(self, carried):
        _flush(carried)
        def helper():
            self.helper_context = ACTIVE.get()
            self._launch_blocking(carried)
        import threading
        self.thread = threading.Thread(target=helper)
        self.thread.start()
    def start_forward(self, rows):
        if self.rank:
            self._launch_blocking(self.pending or ())
            self.pending = None
        return object()
    def end_forward(self, tensors, rows):
        if self.rank == self.last:
            return
        if self.rank == 0:
            if self.async_handoff:
                self._launch_async(tensors)
            else:
                self._launch_blocking(tensors)
        else:
            self.pending = tensors
'''
    path = tmp_path/'native_wave_stub.py'; path.write_text(drain + '\n' + stub)
    spec = importlib.util.spec_from_file_location('native_wave_stub', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec.loader.exec_module(module); module.ACTIVE = trace.ACTIVE
    def emit(kind, scope=None, chunk=None, data=None):
        events.append((kind, scope.context if scope else None, chunk, data or {}))
    monkeypatch.setattr(trace, 'emit', emit)
    V2PPWaveHook.hook(module.PPWave)
    assert boundary.install(module, sync)
    yield NS(module=module, wave=module.PPWave, calls=calls, events=events, sync=sync)
    trace.ACTIVE.set(None); boundary._WAVE.set(None); boundary._SYNC.set(None)


def active(step=1):
    s = trace.Scope(('collection', 'engine', step, 'real'))
    trace.ACTIVE.set(s)
    return s


def forward(wave, s):
    wave.start_forward(17)
    boundary.forward_enter(s)
    wave.end_forward((object(),), 17)


def test_middle_stage_previous_chunk_and_ready_before_world_barrier(rig):
    wave = rig.wave(1); s = active()
    forward(wave, s)
    rig.calls.clear(); rig.events.clear()
    wave.start_forward(17)  # chunk1, but drain belongs to chunk0
    ready = [e for e in rig.events if e[0]=='native.local_drain.ready']
    assert len(ready)==1 and ready[0][2]==0 and s.chunk==1
    assert ready[0][3]['start_mono_ns'] is not None
    names = [e[0] for e in rig.events]
    assert names.index('native.local_drain.ready') < names.index('BARRIER') < names.index('native.handoff_barrier_done')
    assert rig.calls == [('sync', True, True), ('barrier', 'world32'), ('sync', False, True), ('sync', False, False)]
    assert names.index('native.handoff_submitted') < names.index('native.receive_reference_returned')
    assert [e for e in rig.events if e[0]=='native.receive_reference_returned'][0][3]['receive_device_ready'] is False


def test_async_helper_retains_identity_without_active_scope(rig):
    wave = rig.wave(0, True); s = active(7)
    forward(wave, s); wave.thread.join(3)
    assert not wave.thread.is_alive() and wave.helper_context is None
    ready = [e for e in rig.events if e[0]=='native.local_drain.ready']
    assert len(ready)==1 and ready[0][1]==s.context and ready[0][2]==0
    assert wave._sim_native_unfinished is None
    assert rig.calls.count(('sync', True, True))==1
    assert rig.calls.count(('sync', False, False))==2  # producer flush + handoff flush


def test_last_stage_output_is_composite_and_never_fake_send(rig):
    wave = rig.wave(3); s = active()
    forward(wave, s)
    rig.events.clear()
    boundary.output_ready(wave, s.context)
    row = rig.events[0]
    assert row[0]=='native.output_ready_observed' and row[2]==0
    assert row[3]['completion_scope']=='last_stage_output_including_sampling_and_copy'
    assert row[3]['isolated_chunk_execution'] is False
    assert not any(e[0]=='native.handoff_submitted' for e in rig.events)
    boundary.output_ready(wave, s.context)
    assert len(rig.events)==1


def test_multiple_chunks_without_completion_explicitly_unresolved(rig):
    wave=rig.wave(0); wave.drain_before_handoff=False; s=active()
    forward(wave,s); forward(wave,s)
    rows=[e for e in rig.events if e[0]=='native.completion_unresolved']
    assert len(rows)==1 and rows[0][2]==0
    assert not any(e[0]=='native.local_drain.ready' for e in rig.events)


def test_device_exception_preserved_and_context_restored(rig):
    wave=rig.wave(0); s=active(); rig.sync.fail=True
    with pytest.raises(RuntimeError,match='device failure'):
        forward(wave,s)
    assert boundary._SYNC.get() is None and boundary._WAVE.get() is None
    assert not any(e[0]=='native.local_drain.ready' for e in rig.events)
    assert not any(e[0]=='native.handoff_submitted' for e in rig.events)


def test_observer_failures_do_not_change_native_calls(rig, monkeypatch):
    emit = trace.emit
    def fail(kind,*a,**kw):
        if kind.startswith('native.'):
            raise ValueError('observer')
        return emit(kind,*a,**kw)
    monkeypatch.setattr(trace,'emit',fail)
    wave=rig.wave(0); forward(wave,active())
    assert rig.calls == [('sync',True,True),('barrier','world32'),('sync',False,True),('sync',False,False)]


def test_install_once_and_unscoped_sync_unchanged(rig):
    original=rig.sync.synchronize
    assert boundary.install(rig.module,rig.sync)
    assert rig.sync.synchronize is original
    before=len(rig.events)
    rig.sync.synchronize(None,wait=True)
    assert len(rig.events)==before


def test_untracked_warmup_does_not_emit_per_handoff_events(rig):
    wave=rig.wave(0);before=len(rig.events)
    wave.start_forward(17);wave.end_forward((object(),),17)
    assert not any(e[0].startswith('native.') for e in rig.events[before:])
    assert len(rig.calls)==4


def test_unknown_drain_source_fails_closed_for_observer_only(rig):
    module=NS(PPWave=rig.wave,_drain_slice=lambda:None)
    original=rig.sync.synchronize
    assert not boundary.install(module,rig.sync)
    assert rig.sync.synchronize is original
    assert rig.events[-1][0]=='native.boundary_unavailable'


def test_internal_sync_get_output_captures_last_stage_without_extra_wait(rig, monkeypatch):
    monkeypatch.setenv('SIM_TPU_NATIVE_BOUNDARIES','1')
    class Output:
        def __init__(self,copy_state): self.copy_state=copy_state
        def get_output(self):
            if self.copy_state is not None: self.copy_state.wait()
            return 'native-result'
    boundary.install_output_class(Output)
    waits=[];wave=rig.wave(3);s=active();forward(wave,s)
    token=boundary.OUTPUT_WAVE.set(wave)
    try: result=Output(NS(wait=lambda:waits.append(1)))
    finally: boundary.OUTPUT_WAVE.reset(token)
    trace.ACTIVE.set(None)  # output consumed later without the dispatch scope
    assert result.get_output()=='native-result' and waits==[1]
    rows=[e for e in rig.events if e[0]=='native.output_ready_observed']
    assert len(rows)==1 and rows[0][1]==s.context and rows[0][2]==0


def test_output_without_copy_does_not_claim_completion(rig,monkeypatch):
    monkeypatch.setenv('SIM_TPU_NATIVE_BOUNDARIES','1')
    class Output:
        def __init__(self):self.copy_state=None
        def get_output(self):return None
    boundary.install_output_class(Output)
    wave=rig.wave(3);s=active();forward(wave,s)
    token=boundary.OUTPUT_WAVE.set(wave)
    try: result=Output()
    finally:boundary.OUTPUT_WAVE.reset(token)
    before=len(rig.events);result.get_output()
    assert len(rig.events)==before and wave._sim_native_unfinished is not None


def test_export_completion_window_and_null_device_queue_across_cuts(tmp_path,monkeypatch):
    rec=CollectionRecorder(tmp_path,capacity=128,interval=60)
    monkeypatch.setattr(trace,'_RECORDER',rec)
    s=active();s.chunk=2
    try:
        trace.emit('native.forward_dispatch_enter',s,data={'observed_mono_ns':100})
        rec.export_cut('begin',False)
        trace.emit('native.local_drain.ready',s,data={'observed_mono_ns':200,
                   'start_mono_ns':100,'completion_scope':'local_outstanding_work_at_native_drain'})
        receipt=rec.export_cut('done',False)
        root=Path(receipt['path'])
        rows=[json.loads(x) for x in (root/'stage_completion_windows.jsonl').read_text().splitlines()]
        assert len(rows)==1 and rows[0]['chunk']==2
        assert rows[0]['observed_latency_ms']==.0001
        assert rows[0]['valid_observation_window'] is True
        assert rows[0]['valid_pure_device_timing'] is False
        assert rows[0]['device_queue_ns'] is None and rows[0]['device_elapsed_ns'] is None
        assert receipt['complete']
    finally:
        rec.close();trace.ACTIVE.set(None)
