from types import SimpleNamespace
import json
import sys
from contextlib import contextmanager
import pytest
from sglang_simulator.collector.vllm_hook import device_probe as probe
from sglang_simulator.collector.vllm_hook import trace_v2 as trace

@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setattr(probe,'_PROBE',None)
    monkeypatch.setattr(trace,'_RECORDER',None)
    monkeypatch.delenv('SIM_TPU_DEVICE_PROBE',raising=False)

class Worker:
    def execute_model(self,output):return output
    def pp_settle(self):return 'settled'


def test_disabled_installs_nothing():
    class W(Worker):pass
    a,b=W.execute_model,W.pp_settle
    probe.install(W)
    assert W.execute_model is a and W.pp_settle is b


def test_real_step_once_stops_after_native_settle(monkeypatch):
    calls=[]
    class P:
        def __init__(self,w):pass
        def start(self,s):calls.append(('start',s.context))
        def finish(self):calls.append(('stop',))
    monkeypatch.setenv('SIM_TPU_DEVICE_PROBE','1');monkeypatch.setattr(probe,'DeviceProbe',P)
    class W(Worker):
        def pp_settle(self):calls.append(('native_settle',));return 73
    probe.install(W);w=W()
    for phase in ['state','real','real']:
        token=trace.ACTIVE.set(trace.Scope(('c','e',1,phase)))
        try:assert w.execute_model(71)==71
        finally:trace.ACTIVE.reset(token)
    assert len([x for x in calls if x[0]=='start'])==2  # DeviceProbe.start guards repeated real steps.
    assert w.pp_settle()==73
    assert calls[-2:]==[('native_settle',),('stop',)]


def test_settle_exception_preserved_no_early_stop(monkeypatch):
    monkeypatch.setenv('SIM_TPU_DEVICE_PROBE','1');calls=[]
    monkeypatch.setattr(probe,'_PROBE',SimpleNamespace(finish=lambda:calls.append(1)))
    class W(Worker):
        def pp_settle(self):raise ValueError('native')
    probe.install(W)
    with pytest.raises(ValueError,match='native'):W().pp_settle()
    assert not calls


def fake_torch(monkeypatch,tmp_path):
    calls=[]
    class Event:
        def __init__(self,**kwargs):pass
        def record(self):calls.append('event_record')
        def elapsed_time(self,other):raise NotImplementedError('not supported')
    class Prof:
        def __init__(self,**kw):self.kw=kw
        def __enter__(self):calls.append('profile_start');return self
        def __exit__(self,*args):calls.append('profile_stop');self.kw['on_trace_ready'](self)
        def export_chrome_trace(self,path):open(path,'w').write('{"traceEvents":[]}')
    @contextmanager
    def record_function(label):calls.append(label);yield
    t=SimpleNamespace(__version__='fake',tpu=SimpleNamespace(Event=Event),profiler=SimpleNamespace(profile=Prof,record_function=record_function,ProfilerActivity=SimpleNamespace(CPU=0,PrivateUse1=1)))
    monkeypatch.setitem(sys.modules,'torch',t)
    monkeypatch.setitem(sys.modules,'torch_tpu._internal.profiler',SimpleNamespace(TpuProfilerConfig=lambda **kw:kw))
    rec=SimpleNamespace(directory=tmp_path,identity=(0,0,0,0,0,0,0),process_id='p',build_id='b')
    monkeypatch.setattr(trace,'_RECORDER',rec)
    return calls


def test_real_diagnostic_lifecycle_does_not_repeat(monkeypatch,tmp_path):
    calls=fake_torch(monkeypatch,tmp_path);p=probe.DeviceProbe(Worker());scope=trace.Scope(('c','e',1,'real'));scope.chunk=0
    p.start(scope);p.start(scope)
    with p.annotation('main_forward',scope):pass
    p.finish();p.finish();p.start(scope)
    assert calls.count('profile_start')==1 and calls.count('profile_stop')==1
    data=json.loads((tmp_path/'device_probe/probe.json').read_text())
    assert data['state']=='complete' and data['event_api']['elapsed_error_type']=='NotImplementedError'
    assert data['device_timing_verified'] is False
    assert data['artifacts'][0]['path']=='trace.json'


def test_annotation_preserves_model_exception(monkeypatch,tmp_path):
    fake_torch(monkeypatch,tmp_path);p=probe.DeviceProbe(Worker());scope=trace.Scope(('c','e',1,'real'));p.start(scope)
    with pytest.raises(ValueError,match='model'):
        with p.annotation('main_forward',scope):raise ValueError('model')
    p.finish()


def test_start_failure_is_not_retried(monkeypatch,tmp_path):
    calls=fake_torch(monkeypatch,tmp_path)
    monkeypatch.setattr(sys.modules['torch'].profiler,'profile',lambda **k:(_ for _ in ()).throw(RuntimeError('profiler busy')))
    p=probe.DeviceProbe(Worker());scope=trace.Scope(('c','e',1,'real'));p.start(scope);p.start(scope)
    assert p.state=='failed' and len(p.report['errors'])==1
