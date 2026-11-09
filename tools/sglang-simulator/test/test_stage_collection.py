"""All-stage retention, honest labels, protocol identity and original-call safety."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from sglang_simulator.collector.vllm_hook import trace_v2 as trace
from sglang_simulator.collector.vllm_hook import native_boundaries as boundary
from sglang_simulator.collector.vllm_hook.trace_collection import CollectionRecorder
from sglang_simulator.collector.vllm_hook.stage_collection import keep_event,build_samples
from test_native_boundaries import rig, active, forward


def sources():
    result=[]
    context=['collection','engine',7,'real']
    reqs=[dict(req_id='a',query_length=64,past_length=128,input_length=1024,
               num_output_tokens=0,max_output_tokens=1,logprobs=None)]
    def event(seq,kind,pp=-1,pcp=-1,chunk=-1,stamp=10,data=None):
        return dict(seq=seq,event=kind,rank=dict(pp=pp,pcp=pcp),context=context,chunk=chunk,
                    mono_ns=stamp,data=data or {})
    def meta(name):return dict(process_id=name,clock_domain='host',host_clock=dict(boot_id='boot',time_namespace='time'))
    result.append((meta('controller'),[event(1,'step.inputs',data=dict(native_dispatch_ok=True,requests=reqs))]))
    for pp in range(4):
        for pcp in range(8):
            es=[event(1,'worker.execute_model.enter',pp,pcp,stamp=5),
                event(2,'native.forward_dispatch_enter',pp,pcp,0,10,dict(start_mono_ns=10)),
                event(3,'native.forward_dispatch_return',pp,pcp,0,20,dict(observed_mono_ns=20)),
                event(4,'native.local_drain.enter',pp,pcp,0,30,dict(observed_mono_ns=30)),
                event(5,'native.local_drain.ready' if pp<3 else 'native.output_ready_observed',pp,pcp,0,100,
                      dict(observed_mono_ns=100,completion_scope='local_outstanding_work_at_native_drain' if pp<3 else 'last_stage_output_including_sampling_and_copy')),
                event(6,'native.handoff_barrier_done',pp,pcp,0,120,dict(observed_mono_ns=120)),
                event(7,'native.handoff_submitted',pp,pcp,0,130,dict(observed_mono_ns=130,payload_bytes=8192))]
            if pcp==0:es.append(event(8,'chunk.inputs',pp,pcp,0,9,dict(requests=reqs,input_complete=True,
                        global_valid_tokens=64,global_padded_tokens=2048,local_padded_tokens=256)))
            result.append((meta(f'{pp}-{pcp}'),es))
    return result


def test_four_stage_join_and_scopes():
    rows=build_samples(sources())
    assert len(rows)==4 and {r['pp_stage'] for r in rows}==set(range(4))
    assert all(r['valid_observation_window'] and r['pcp_ranks_complete'] for r in rows)
    assert all(r['target_service_s'] is None and not r['valid_service_label'] and not r['valid_pure_device_timing'] for r in rows)
    assert rows[0]['rank_evidence'][0]['world_barrier_observed_ms']==20/1e6
    assert rows[0]['rank_evidence'][0]['residual_native_wait_ms']==70/1e6
    assert rows[3]['completion_scopes']==['last_stage_output_including_sampling_and_copy']
    assert rows[3]['rank_evidence'][0]['world_barrier_observed_ms'] is None


@pytest.mark.parametrize('damage,issue',[
    ('missing_rank','incomplete_completion'),('clock','unverified_stage_clock'),
    ('feature','incomplete_request_features'),('order','timestamp_order_invalid'),
    ('unresolved','native.completion_unresolved'),('step','missing_or_failed_step_input'),
    ('duplicate','duplicate_boundary')])
def test_bad_evidence_never_valid(damage,issue):
    ss=sources();meta,es=ss[1]
    if damage=='missing_rank':es[:]=[e for e in es if e['event']!='native.local_drain.ready']
    if damage=='clock':meta['host_clock']['boot_id']='other'
    if damage=='feature':es[-1]['data']['requests'][0]['past_length']=None
    if damage=='order':es[4]['data']['observed_mono_ns']=1
    if damage=='unresolved':es.append({**es[3],'seq':19,'event':'native.completion_unresolved'})
    if damage=='step':ss.pop(0)
    if damage=='duplicate':es.append({**es[4],'seq':20})
    row=next(r for r in build_samples(ss) if r['pp_stage']==0)
    assert not row['valid_observation_window'] and issue in row['issues']


def test_cross_cut_dedup_and_cross_stage_clocks_separate():
    ss=sources()
    for meta,events in ss[1:]:
        pp=events[0]['rank']['pp'];meta['clock_domain']=f'host{pp}';meta['host_clock']['boot_id']=f'boot{pp}'
    assert all(r['valid_observation_window'] for r in build_samples(ss+ss))


def test_stage_focus_exports_and_profile_cut_continue(tmp_path,monkeypatch):
    monkeypatch.setenv('SIM_TRACE_FOCUS','pp_stage')
    rec=CollectionRecorder(tmp_path,interval=60)
    try:
        for pp in range(4):
            rec.identity=(pp,pp*8,0,pp,0,0,0)
            assert rec.emit('chunk.inputs',('c','e',1,'real'),0,{'requests':[]})
            assert rec.emit('stage.config',data={'pp_size':4})
            assert rec.emit('native.local_drain.ready',('c','e',1,'real'),0,
                {'observed_mono_ns':200,'start_mono_ns':100,'launch_id':2})
        cut=rec.export_cut('stagecut',False);assert cut['complete']
        root=Path(cut['path'])
        assert len((root/'chunk_inputs.jsonl').read_text().splitlines())==4
        assert len((root/'stage_configs.jsonl').read_text().splitlines())==4
        windows=[json.loads(s) for s in (root/'stage_rank_windows.jsonl').read_text().splitlines()]
        assert len(windows)==4 and all(not r['valid_service_label'] for r in windows)
        assert not (root/'schedule_batch.jsonl').exists()
        rec.emit('native.handoff_submitted',data={'launch_id':3})
        rec.export_cut('startcut',True)
        last=rec.export_cut('stopcut',False);assert last['complete']
        assert rec.failure is None and rec.observer_errors==0
    finally:rec.close()


def test_retention_all_workers_features_only_pcp0():
    for pp in range(4):
        for pcp in range(8):
            ident=(pp,pp*8+pcp,pcp,pp,pcp,0,0)
            assert keep_event('native.local_drain.ready',ident)
            assert keep_event('worker.execute_model.enter',ident)
            assert keep_event('chunk.inputs',ident)==(pcp==0)
            assert not keep_event('runner.some_extra.enter',ident)
    assert keep_event('pipeline.result_ready',(-1,)*7)


def test_metadata_reads_no_tensor_data():
    class Tensor:
        shape=(2048,4096);dtype='torch.bfloat16'
        def item(self):raise AssertionError('device read')
        def cpu(self):raise AssertionError('device read')
        def numpy(self):raise AssertionError('device read')
    result=boundary.payload_metadata((Tensor(),))
    assert result['available'] and result['payload_bytes']==2048*4096*2
    Tensor.dtype='unknown'
    assert boundary.payload_metadata((Tensor(),))['payload_bytes'] is None


def test_untracked_protocol_and_original_native_calls(rig,monkeypatch):
    recorder=NS(focus_stage=True,focus_pp0=False,identity=(0,0,0,0,0,0,0),observer_errors=0)
    monkeypatch.setattr(trace,'_RECORDER',recorder)
    wave=rig.wave(0)
    wave._sim_stage_protocol_active=True
    wave.start_forward(17);wave.end_forward((object(),),17)
    assert rig.calls==[('sync',True,True),('barrier','world32'),('sync',False,True),('sync',False,False)]
    rows=[e for e in rig.events if e[0]=='native.handoff_submitted']
    assert len(rows)==1 and rows[0][1] is None and rows[0][3]['launch_id']==1
    assert recorder.observer_errors==0


def test_helper_frozen_identity_preserved_in_stage_mode(rig,monkeypatch):
    recorder=NS(focus_stage=True,focus_pp0=False,identity=(0,0,0,0,0,0,0),observer_errors=0)
    monkeypatch.setattr(trace,'_RECORDER',recorder)
    wave=rig.wave(0);wave.async_handoff=True;s=active()
    forward(wave,s);wave.thread.join(2)
    rows=[e for e in rig.events if e[0]=='native.local_drain.ready']
    assert len(rows)==1 and rows[0][1]==s.context
    assert rows[0][3]['launch_ordinal']==0
    assert recorder.observer_errors==0


def test_stage_config_once_and_native_execute_unchanged(tmp_path, monkeypatch):
    from sglang_simulator.collector.vllm_hook.trace_collection_hooks import CollectionWorkerHook
    monkeypatch.setenv('SIM_TRACE_FOCUS','pp_stage')
    rec=CollectionRecorder(tmp_path,interval=60)
    rec.identity=(2,16,0,2,0,0,0)
    monkeypatch.setattr(trace,'_RECORDER',rec)
    calls=[]
    class Worker:
        vllm_config=NS(model_config=NS(model='test-model',dtype='bfloat16',quantization=None),
            parallel_config=NS(pipeline_parallel_size=4,prefill_context_parallel_size=8,tensor_parallel_size=1))
        model_runner=NS(max_num_tokens=2048,max_num_reqs=256,num_tokens_paddings=[256,512,2048])
        def execute_model(self,value):calls.append(value);return value
    CollectionWorkerHook.hook(Worker)
    worker=Worker()
    try:
        assert worker.execute_model(1)==1 and worker.execute_model(2)==2
        cut=rec.export_cut('config',False)
        rows=[json.loads(s) for s in (Path(cut['path'])/'stage_configs.jsonl').read_text().splitlines()]
        assert calls==[1,2] and len(rows)==1
        assert rows[0]['data']['pcp_size']==8 and rows[0]['data']['token_buckets']==[256,512,2048]
        assert rec.observer_errors==0
    finally:rec.close()


def test_stage_focus_initial_warmup_silent(rig,monkeypatch):
    recorder=NS(focus_stage=True,focus_pp0=False,identity=(0,0,0,0,0,0,0),observer_errors=0)
    monkeypatch.setattr(trace,'_RECORDER',recorder)
    wave=rig.wave(0);before=len(rig.events)
    wave.start_forward(17);wave.end_forward((object(),),17)
    assert not any(e[0].startswith('native.') for e in rig.events[before:])
    assert len(rig.calls)==4 and recorder.observer_errors==0


def test_training_detail_keeps_only_consumed_events(monkeypatch):
    from sglang_simulator.collector.vllm_hook import stage_collection as sc
    monkeypatch.setattr(sc, 'TRAINING', True)
    for pp in range(4):
        for pcp in range(8):
            ident = (pp, pp*8+pcp, pcp, pp, pcp, 0, 0)
            for kind in ('native.local_drain.ready', 'native.output_get.return',
                         'native.output_ready_observed', 'native.completion_unresolved',
                         'native.local_drain.error', 'startup.ready', 'native.boundary_installed'):
                assert sc.keep_event(kind, ident)
            for kind in ('chunk.inputs', 'chunk.layout', 'stage.config'):
                assert sc.keep_event(kind, ident) == (pcp == 0)
            for kind in ('native.handoff_flush.submitted', 'native.handoff_flush.enter',
                         'native.helper_fence.return', 'native.handoff_barrier_done',
                         'native.launch_call.return', 'native.local_drain.enter',
                         'native.handoff_submitted', 'native.receive_reference_returned',
                         'native.forward_dispatch_enter', 'native.payload_metadata',
                         'worker.execute_model.enter', 'runner.main_forward.enter', 'pp.push'):
                assert not sc.keep_event(kind, ident)
    ctrl = (-1,)*7
    for kind in sc.TRAINING_CONTROLLER_KINDS:
        assert sc.keep_event(kind, ctrl)
    for kind in ('engine.dispatch.observed', 'rpc.sample_tokens.dispatch'):
        assert not sc.keep_event(kind, ctrl)


def test_training_detail_native_labels_survive(rig,monkeypatch):
    from sglang_simulator.collector.vllm_hook import stage_collection as sc
    from sglang_simulator.collector.vllm_hook import host_breakdown as host
    monkeypatch.setattr(sc,"TRAINING",True)
    monkeypatch.setattr(host,"ENABLED",True)
    recorder=NS(focus_stage=True,focus_pp0=False,identity=(0,0,0,0,0,0,0),observer_errors=0)
    monkeypatch.setattr(trace,"_RECORDER",recorder)
    wave=rig.wave(0);wave.async_handoff=True;s=active()
    calls_before=len(rig.calls)
    forward(wave,s);wave.thread.join(2)
    kinds={e[0] for e in rig.events if e[0].startswith("native.")}-{"native.boundary_installed"}
    assert kinds=={"native.local_drain.ready"}
    row=[e for e in rig.events if e[0]=="native.local_drain.ready"][0]
    assert row[1]==s.context and row[3]["call_start_mono_ns"]<=row[3]["call_end_mono_ns"]
    assert row[3]["start_mono_ns"] is not None and "burst_id" in row[3]
    assert len(rig.calls)>calls_before and recorder.observer_errors==0


def test_tick_detail_adds_pcp0_tick_events_only(monkeypatch):
    from sglang_simulator.collector.vllm_hook import stage_collection as sc
    from sglang_simulator.collector.vllm_hook import host_breakdown as host
    monkeypatch.setattr(sc, 'TRAINING', True)
    monkeypatch.setattr(sc, 'TICK', True)
    monkeypatch.setattr(host, 'ENABLED', True)
    extra = (*sc.TICK_WORKER_KINDS, *sc.TICK_HOST_KINDS, 'runner.main_forward.enter', 'pp.push',
             'wrapper.execute_model.enter')
    for pp in range(4):
        for pcp in range(8):
            ident = (pp, pp*8+pcp, pcp, pp, pcp, 0, 0)
            for kind in ('native.local_drain.ready', 'native.output_get.return', 'native.local_drain.error'):
                assert sc.keep_event(kind, ident)
            for kind in extra:
                assert sc.keep_event(kind, ident) == (pcp == 0)
            for kind in ('native.handoff_flush.submitted', 'native.handoff_flush.enter', 'native.local_drain.enter',
                         'native.payload_metadata', 'native.helper_fence.return', 'native.helper_request.enter'):
                assert not sc.keep_event(kind, ident)
    ctrl = (-1,)*7
    for kind in (*sc.TRAINING_CONTROLLER_KINDS, *sc.TICK_CONTROLLER_KINDS):
        assert sc.keep_event(kind, ctrl)
    assert not sc.keep_event('native.handoff_barrier_done', ctrl)


def test_tick_detail_without_host_breakdown_keeps_native_tick_only(monkeypatch):
    from sglang_simulator.collector.vllm_hook import stage_collection as sc
    from sglang_simulator.collector.vllm_hook import host_breakdown as host
    monkeypatch.setattr(sc, 'TRAINING', True)
    monkeypatch.setattr(sc, 'TICK', True)
    monkeypatch.setattr(host, 'ENABLED', False)
    ident = (0, 0, 0, 0, 0, 0, 0)
    assert sc.keep_event('native.handoff_barrier_done', ident)
    assert not sc.keep_event('runner.main_forward.enter', ident)
    assert not sc.keep_event('worker.pp_push.enter', ident)
