from concurrent.futures import Future
import json
from types import SimpleNamespace as NS
import pytest
from sglang_simulator.collector.vllm_hook import trace_v2 as trace
from sglang_simulator.collector.vllm_hook.pp0_steps import build_steps, keep_event, observe_result
from sglang_simulator.collector.vllm_hook.trace_collection import CollectionRecorder


def sources(chunks=2):
    context = ['collection','engine',1,'real']
    def row(kind, pp=-1, pcp=-1, chunk=-1, stamp=0, data=None):
        return dict(event=kind, context=context, chunk=chunk, mono_ns=stamp,
                    rank={'pp':pp,'pcp':pcp}, data=data or {})
    def source(pid, es):
        for i,e in enumerate(es): e['seq']=i+1
        return ({'process_id':pid, 'clock_domain':'host0',
                 'host_clock':{'boot_id':'boot0','time_namespace':'time0'}},es)
    reqs = [{'req_id':'r'+str(c),'query_length':17+c,'input_length':100+c,'past_length':83} for c in range(chunks)]
    result = [source('engine', [row('step.inputs',data={'native_dispatch_ok':True,'requests':reqs}),
                              row('pp0.result_ready',stamp=1000,data={'observed_mono_ns':1000,'success':True})])]
    for p in range(8):
        es = [row('worker.execute_model.enter',0,p,stamp=10+p)]
        for c in range(chunks):
            es += [row('native.local_drain.ready',0,p,c,data={'observed_mono_ns':100+100*c+p}),
                   row('native.handoff_submitted',0,p,c,data={'observed_mono_ns':120+100*c+p})]
            if p==0:
                es.append(row('chunk.inputs',0,0,c,data={'requests':[reqs[c]],'input_complete':True}))
        result.append(source('worker'+str(p),es))
    return result


def test_all_pcp_chunks_and_features_join():
    r, = build_steps(sources())
    assert r['valid'] and r['t1_worker_start_mono_ns']==10
    assert r['t2_local_completion_observed_mono_ns']==207
    assert r['handoff_submitted_mono_ns']==227 and r['t3_result_ready_observed_mono_ns']==1000
    assert r['requests'][0]['past_length']==83 and r['chunk_count']==2
    assert not r['valid_pure_device_timing'] and not r['per_request_isolated_latency']


def test_interleaved_steps_join_by_identity_and_deduplicate_cuts():
    first=sources(); second=sources(1)
    for meta,events in second:
        meta['process_id']+='next'
        for e in events: e['context']=['collection','engine',2,'real']
    rows=build_steps(first+second+[first[0]])
    assert len(rows)==2 and all(r['valid'] for r in rows)


@pytest.mark.parametrize('damage,issue',[
    ('rank','incomplete_local_completion'),('clock','unverified_clock_domain'),
    ('failure','missing_or_failed_result'),('feature','incomplete_request_features'),
    ('order','timestamp_order_invalid'),('tokens','chunk_step_token_mismatch')])
def test_incomplete_data_never_labelled_valid(damage,issue):
    ss=sources()
    if damage=='rank':ss[-1][1].pop(1)
    if damage=='clock':ss[-1][0]['host_clock']['boot_id']='other'
    if damage=='failure':ss[0][1][-1]['data']['success']=False
    if damage=='feature':ss[1][1][-1]['data']['requests'][0]['input_length']=None
    if damage=='order':ss[0][1][-1]['data']['observed_mono_ns']=1
    if damage=='tokens':ss[0][1][0]['data']['requests']=[]
    r,=build_steps(ss)
    assert not r['valid'] and issue in r['issues']


def test_only_pp0_minimal_events_retained():
    assert keep_event('worker.execute_model.enter',(-1,0,0,0,0,0,0))
    assert not keep_event('worker.execute_model.enter',(1,8,0,1,0,0,0))
    assert not keep_event('runner.main_forward.exit',(0,0,0,0,0,0,0))
    assert keep_event('chunk.inputs',(0,0,0,0,0,0,0))
    assert not keep_event('chunk.inputs',(0,1,1,0,1,0,0))
    assert keep_event('pp0.result_ready',(-1,)*7)


@pytest.mark.parametrize('failed',[False,True])
def test_result_callback_checks_success_and_does_not_wait(monkeypatch,failed):
    f=Future(); events=[]
    monkeypatch.setattr(trace,'emit',lambda kind,scope,**kw:events.append((kind,scope.context,kw)))
    slot=NS(_sim_collection_context=('c','e',1,'real'))
    observe_result(NS(in_flight=[NS(slot=slot,future=f)]))
    assert events==[]
    if failed:f.set_exception(ValueError('test'))
    else:f.set_result(object())
    assert len(events)==1 and events[0][2]['data']['success'] is not failed


def test_minimal_export_both_triggers_and_cross_cut_join(tmp_path,monkeypatch):
    monkeypatch.setenv('SIM_TRACE_FOCUS','pp0_step')
    rec=CollectionRecorder(tmp_path,interval=60)
    rec.identity=(0,0,0,0,0,0,0)
    try:
        rec.emit('worker.execute_model.enter',('c','e',1,'real'))
        rec.emit('runner.main_forward.exit',('c','e',1,'real'))
        rec.emit('chunk.layout',('c','e',1,'real'))
        rec.export_cut('start1',True)
        rec.emit('native.local_drain.ready',('c','e',1,'real'),0,{'observed_mono_ns':999})
        cut=rec.export_cut('stop1',False)
        assert cut['complete']
        roots=sorted((rec.directory/'exports').iterdir())
        assert all((p/'pp0_points.jsonl').exists() for p in roots)
        rows=[json.loads(line) for p in roots for line in (p/'pp0_points.jsonl').read_text().splitlines()]
        assert {r['event'] for r in rows}=={'worker.execute_model.enter','native.local_drain.ready'}
        assert not list(rec.directory.rglob('schedule_batch.jsonl'))
        assert rec.rejected==rec.dropped_full==rec.dropped_busy==0
    finally:rec.close()
