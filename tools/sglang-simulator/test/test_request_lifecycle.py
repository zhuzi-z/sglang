from enum import IntEnum
from types import SimpleNamespace as NS

import pytest

from sglang_simulator.simulation.vllm.lifecycle import (
    RequestLifecycleObserver, install_identity_alias, request_key,
)


class Reason(IntEnum):
    STOP = 0
    ABORT = 2


class Recorder:
    def __init__(self):
        self.rows = []

    def emit(self, event, **data):
        self.rows.append(dict(event=event, **data))


class Scheduler:
    def __init__(self):
        self.requests = {}
        self.pending = None
        self.result = {}

    def add_request(self, request):
        self.requests[request.request_id] = request
        return request

    def schedule(self):
        return self.pending

    def update_from_output(self, output, model_output):
        if model_output == "error":
            raise ValueError("native failed")
        return self.result

    def _free_request(self, request):
        del self.requests[request.request_id]
        return "native-free"


def make_output(new=(), cached=(), query=None):
    query = query or {}
    return NS(scheduled_new_reqs=list(new),
              scheduled_cached_reqs=NS(req_ids=[r for r, _ in cached], resumed_req_ids=set(),
                                       num_computed_tokens=[p for _, p in cached]),
              num_scheduled_tokens=query, total_num_scheduled_tokens=sum(query.values()))


def request(rid, prompt=512, external=None):
    return NS(request_id=rid, num_prompt_tokens=prompt, num_output_tokens=0,
              max_tokens=3, num_external_computed_tokens=external,
              status=NS(name="FINISHED_ABORTED"))


def test_chunked_order_cache_and_accepted_output():
    recorder = Recorder()
    observer = RequestLifecycleObserver(recorder)
    scheduler = Scheduler()
    observer.install(scheduler)
    a, b = request("a", external=128), request("b")
    assert scheduler.add_request(a) is a
    scheduler.add_request(b)
    scheduler.pending = make_output(
        [NS(req_id="a", num_computed_tokens=128), NS(req_id="b", num_computed_tokens=0)],
        query={"b": 256, "a": 256})
    output = scheduler.schedule()
    members = output._hisim_lifecycle_requests
    assert [r["request_id_sha256"] for r in members] == [request_key("b"), request_key("a")]
    assert not members[0]["cache_split_valid"]
    assert members[1]["cache_split_valid"] and members[1]["local_kv_hit_len"] == 0
    assert not members[0]["prefill_complete_after_forward"]
    scheduler.pending = make_output(cached=[("a", 384), ("b", 256)], query={"a": 128, "b": 256})
    second = scheduler.schedule()
    assert all(r["request_iteration"] == 2 for r in second._hisim_lifecycle_requests)
    assert all(r["total_kv_hit_len"] is None for r in second._hisim_lifecycle_requests)
    scheduler.result = {0: NS(outputs=[NS(request_id="a", new_token_ids=[99],
                                        finish_reason=Reason.STOP, num_cached_tokens=128),
                                     NS(request_id="b", new_token_ids=[], finish_reason=Reason.ABORT)],
                            finished_requests={"b"})}
    assert scheduler.update_from_output(second, "native") is scheduler.result
    rows = [r for r in recorder.rows if r["event"] == "request.output_accepted"]
    assert rows[0]["finished"] and rows[0]["finish_reason"] == "STOP"
    assert rows[1]["finished"] and rows[1]["new_token_count"] == 0
    assert all(r["step_id"] == 2 for r in rows)
    assert not observer.iterations
    assert "new_token_ids" not in str(rows)
    assert scheduler._free_request(b) == "native-free"
    assert recorder.rows[-1]["event"] == "request.logical_finish"
    assert recorder.rows[-1]["kv_release_confirmed"] is False


def test_native_error_and_empty_step():
    recorder = Recorder()
    observer = RequestLifecycleObserver(recorder)
    scheduler = Scheduler()
    observer.install(scheduler)
    scheduler.pending = make_output()
    output = scheduler.schedule()
    assert not recorder.rows
    for _ in range(100):
        assert scheduler.schedule() is output
    assert observer.step_id == 101 and not recorder.rows
    with pytest.raises(ValueError, match="native failed"):
        scheduler.update_from_output(output, "error")
    assert observer.active_step_id is None
    assert not [r for r in recorder.rows if r["event"] == "request.output_accepted"]


def test_observer_failure_preserves_native_result(monkeypatch):
    recorder = Recorder()
    observer = RequestLifecycleObserver(recorder)
    scheduler = Scheduler()
    observer.install(scheduler)
    scheduler.pending = make_output()

    def fail(*args):
        raise RuntimeError("observation failed")

    monkeypatch.setattr(observer, "scheduled", fail)
    assert scheduler.schedule() is scheduler.pending
    assert recorder.rows[-1] == {"event": "observer.error", "error_type": "RuntimeError"}


def test_membership_unknown_and_tracking_bound():
    recorder = Recorder()
    observer = RequestLifecycleObserver(recorder, max_tracked_requests=1)
    scheduler = Scheduler()
    observer.install(scheduler)
    scheduler.add_request(request("a"))
    scheduler.add_request(request("b"))
    scheduler.pending = make_output([NS(req_id="a", num_computed_tokens=0)],
                                    query={"a": 1, "b": 2})
    output = scheduler.schedule()
    assert len(observer.iterations) == 1
    assert output._hisim_lifecycle_requests[1]["past_kv_length"] is None
    assert output._hisim_lifecycle_requests[1]["request_iteration"] is None
    assert not recorder.rows[-1]["membership_complete"]
    observer.install(scheduler)
    before = len(recorder.rows)
    scheduler.schedule()
    assert len(recorder.rows) == before + 1


def test_identity_alias_uses_explicit_native_mapping():
    recorder = Recorder()
    observer = RequestLifecycleObserver(recorder)
    native_result = (object(), 3)

    def preprocess(req):
        if req.request_id == "native-error":
            raise ValueError("native failed")
        return native_result

    core = NS(preprocess_add_request=preprocess)
    install_identity_alias(core, observer)
    for internal in ("first-internal", "second-internal"):
        assert core.preprocess_add_request(NS(request_id=internal, external_req_id="external")) is native_result
    assert len(recorder.rows) == 2
    assert recorder.rows[0]["request_id_sha256"] != recorder.rows[1]["request_id_sha256"]
    assert all(r["external_request_id_sha256"] == request_key("external") for r in recorder.rows)
    assert core.preprocess_add_request(NS(request_id="missing-metadata")) is native_result
    assert recorder.rows[-1]["event"] == "observer.error"
    with pytest.raises(ValueError, match="native failed"):
        core.preprocess_add_request(NS(request_id="native-error"))
    assert recorder.rows[-1]["event"] == "request.engine_preprocess_failed"
    assert recorder.rows[-1]["native_ok"] is False


def test_native_call_boundaries_and_update_exception(monkeypatch):
    import sglang_simulator.simulation.vllm.lifecycle as module

    clock = iter(range(100, 200))
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: next(clock))
    recorder = Recorder()
    observer = RequestLifecycleObserver(recorder)
    scheduler = Scheduler()
    observer.install(scheduler)
    req = request("native-id")
    assert scheduler.add_request(req) is req
    add = recorder.rows[-1]
    assert (add["start_mono_ns"], add["end_mono_ns"]) == (100, 101)
    assert add["native_ok"] is True
    scheduler.pending = make_output([NS(req_id="native-id", num_computed_tokens=0)],
                                    query={"native-id": 1})
    output = scheduler.schedule()
    scheduled = recorder.rows[-1]
    assert (scheduled["schedule_enter_mono_ns"], scheduled["schedule_return_mono_ns"]) == (102, 103)
    with pytest.raises(ValueError, match="native failed"):
        scheduler.update_from_output(output, "error")
    failed = recorder.rows[-1]
    assert failed["event"] == "scheduler.output_update" and failed["native_ok"] is False
    assert failed["step_id"] == 1
    assert failed["request_ids_sha256"] == (request_key("native-id"),)
    assert observer.active_step_id is None
    assert scheduler._free_request(req) == "native-free"
    finished = recorder.rows[-1]
    assert finished["native_ok"] and finished["end_mono_ns"] > finished["start_mono_ns"]


def test_broken_recorder_does_not_replace_native_result_or_exception():
    class BrokenRecorder:
        def emit(self, *args, **kwargs):
            raise RuntimeError("writer unavailable")

    observer = RequestLifecycleObserver(BrokenRecorder())
    scheduler = Scheduler()
    observer.install(scheduler)
    req = request("r")
    assert scheduler.add_request(req) is req
    scheduler.pending = make_output(query={"r": 1})
    assert scheduler.schedule() is scheduler.pending
    assert scheduler.update_from_output(scheduler.pending, None) is scheduler.result
    with pytest.raises(ValueError, match="native failed"):
        scheduler.update_from_output(scheduler.pending, "error")
    assert scheduler._free_request(req) == "native-free"
    core = NS(preprocess_add_request=lambda request: request)
    install_identity_alias(core, observer)
    request_with_identity = NS(request_id="internal", external_req_id="external")
    assert core.preprocess_add_request(request_with_identity) is request_with_identity


def test_schedule_failure_observes_without_changing_exception():
    recorder = Recorder()
    observer = RequestLifecycleObserver(recorder)
    scheduler = Scheduler()
    error = RuntimeError("native schedule error")

    def failed():
        raise error

    scheduler.schedule = failed
    observer.install(scheduler)
    with pytest.raises(RuntimeError) as raised:
        scheduler.schedule()
    assert raised.value is error
    assert recorder.rows[-1]["event"] == "scheduler.schedule_failed"
    assert not recorder.rows[-1]["native_ok"]
    assert observer.step_id == 0
