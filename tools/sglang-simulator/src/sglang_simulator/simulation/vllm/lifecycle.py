import hashlib
import time
from functools import wraps

from sglang_simulator.lifecycle_recorder import LifecycleRecorder


def request_key(request_id):
    return hashlib.sha256(str(request_id).encode()).hexdigest()


def enum_name(value):
    return getattr(value, "name", None) if value is not None else None


class RequestLifecycleObserver:
    def __init__(self, recorder, max_tracked_requests=16384):
        self.recorder = recorder
        self.max_tracked_requests = max_tracked_requests
        self.iterations = {}
        self.step_id = 0
        self.active_step_id = None

    def observe(self, callback):
        try:
            callback()
        except Exception as exc:
            try:
                self.recorder.emit("observer.error", error_type=type(exc).__name__)
            except Exception:
                pass

    def record_call(self, event, started_ns, ended_ns, request_ids, native_ok,
                    source, step_id=None):
        self.recorder.emit(
            event, start_mono_ns=started_ns, end_mono_ns=ended_ns,
            request_ids_sha256=tuple(request_key(rid) for rid in request_ids),
            native_ok=native_ok, source=source, step_id=step_id,
            time_kind="host_native_call", device_completion_confirmed=False,
        )

    def scheduled(self, scheduler, output, started_ns, ended_ns=None):
        self.step_id += 1
        output._hisim_lifecycle_step_id = self.step_id
        new = {r.req_id: r for r in output.scheduled_new_reqs}
        cached = output.scheduled_cached_reqs
        computed = {rid: r.num_computed_tokens for rid, r in new.items()}
        computed.update(zip(cached.req_ids, cached.num_computed_tokens))
        rows = []
        for rid, query in output.num_scheduled_tokens.items():
            key = request_key(rid)
            if key in self.iterations or len(self.iterations) < self.max_tracked_requests:
                iteration = self.iterations.get(key, 0) + 1
                self.iterations[key] = iteration
            else:
                iteration = None
            request = scheduler.requests.get(rid)
            prompt = getattr(request, "num_prompt_tokens", None)
            past = computed.get(rid)
            external = getattr(request, "num_external_computed_tokens", None)
            first = rid in new and iteration == 1
            split_valid = first and past is not None and external is not None and 0 <= external <= past
            resumed_ids = getattr(cached, "resumed_req_ids", None)
            resumed = False if rid in new else (rid in resumed_ids if resumed_ids is not None else None)
            rows.append({
                "request_id_sha256": key,
                "request_iteration": iteration,
                "query_length": int(query),
                "past_kv_length": int(past) if past is not None else None,
                "input_length": prompt,
                "output_tokens_before": getattr(request, "num_output_tokens", None),
                "prefill_complete_after_forward": past + query >= prompt
                if past is not None and prompt is not None else None,
                "total_kv_hit_len": past if first else None,
                "ext_kv_hit_len": external if first else None,
                "local_kv_hit_len": past - external if split_valid else None,
                "cache_split_valid": split_valid,
                "resumed": resumed,
            })
        output._hisim_lifecycle_requests = rows
        if rows:
            self.recorder.emit("scheduler.scheduled", step_id=self.step_id,
                               schedule_enter_mono_ns=started_ns,
                               schedule_return_mono_ns=ended_ns, requests=rows,
                               num_tokens=int(output.total_num_scheduled_tokens),
                               membership_complete=all(all(r[field] is not None for field in (
                                   "query_length", "past_kv_length", "input_length",
                                   "output_tokens_before", "request_iteration", "resumed")) for r in rows))

    def accepted(self, scheduler_output, result):
        step_id = getattr(scheduler_output, "_hisim_lifecycle_step_id", None)
        for batch in (result or {}).values():
            for output in batch.outputs:
                key = request_key(output.request_id)
                reason = getattr(output, "finish_reason", None)
                self.recorder.emit(
                    "request.output_accepted", step_id=step_id,
                    request_id_sha256=key,
                    new_token_count=len(output.new_token_ids),
                    finished=reason is not None, finish_reason=enum_name(reason),
                    num_cached_tokens=getattr(output, "num_cached_tokens", None),
                    spec_decode_num_draft_tokens=getattr(output, "spec_decode_num_draft_tokens", None),
                    spec_decode_num_accepted_tokens=getattr(output, "spec_decode_num_accepted_tokens", None),
                )
                if reason is not None:
                    self.iterations.pop(key, None)
            for rid in getattr(batch, "finished_requests", None) or ():
                self.recorder.emit("request.finished_notification", step_id=step_id,
                                   request_id_sha256=request_key(rid))

    def install(self, scheduler):
        if getattr(scheduler, "_hisim_lifecycle_observer", None) is not None:
            return
        scheduler._hisim_lifecycle_observer = self
        add = scheduler.add_request
        schedule = scheduler.schedule
        update = scheduler.update_from_output
        free = scheduler._free_request

        @wraps(add)
        def add_request(request, *args, **kwargs):
            start, native_ok = time.monotonic_ns(), False
            try:
                result = add(request, *args, **kwargs)
                native_ok = True
                return result
            finally:
                end = time.monotonic_ns()
                self.observe(lambda: self.recorder.emit(
                    "request.scheduler_add", request_id_sha256=request_key(request.request_id),
                    input_length=request.num_prompt_tokens,
                    max_output_tokens=getattr(request, "max_tokens", None),
                    start_mono_ns=start, end_mono_ns=end, native_ok=native_ok,
                    time_kind="host_native_call", source="Scheduler.add_request",
                ))

        @wraps(schedule)
        def scheduled(*args, **kwargs):
            start = time.monotonic_ns()
            try:
                output = schedule(*args, **kwargs)
            except BaseException:
                end = time.monotonic_ns()
                self.observe(lambda: self.record_call(
                    "scheduler.schedule_failed", start, end, (), False, "Scheduler.schedule"))
                raise
            end = time.monotonic_ns()
            self.observe(lambda: self.scheduled(scheduler, output, start, end))
            return output

        @wraps(update)
        def accepted(scheduler_output, model_output, *args, **kwargs):
            self.active_step_id = getattr(scheduler_output, "_hisim_lifecycle_step_id", None)
            start, native_ok = time.monotonic_ns(), False
            try:
                result = update(scheduler_output, model_output, *args, **kwargs)
                native_ok = True
            finally:
                end = time.monotonic_ns()
                step_id = self.active_step_id
                self.active_step_id = None
                if scheduler_output.total_num_scheduled_tokens or not native_ok:
                    self.observe(lambda: self.record_call(
                        "scheduler.output_update", start, end,
                        scheduler_output.num_scheduled_tokens, native_ok,
                        "Scheduler.update_from_output", step_id))
            self.observe(lambda: self.accepted(scheduler_output, result))
            return result

        @wraps(free)
        def finished(request, *args, **kwargs):
            start, native_ok = time.monotonic_ns(), False
            try:
                result = free(request, *args, **kwargs)
                native_ok = True
                return result
            finally:
                end = time.monotonic_ns()

                def record():
                    key = request_key(request.request_id)
                    self.recorder.emit("request.logical_finish", step_id=self.active_step_id,
                                       request_id_sha256=key, status=enum_name(request.status),
                                       observed_output_tokens=request.num_output_tokens,
                                       kv_release_confirmed=False, native_ok=native_ok,
                                       start_mono_ns=start, end_mono_ns=end,
                                       time_kind="host_native_call", source="Scheduler._free_request")
                    if native_ok:
                        self.iterations.pop(key, None)

                self.observe(record)

        scheduler.add_request = add_request
        scheduler.schedule = scheduled
        scheduler.update_from_output = accepted
        scheduler._free_request = finished


def install_identity_alias(core, observer):
    preprocess = core.preprocess_add_request

    @wraps(preprocess)
    def preprocessed(request, *args, **kwargs):
        start = time.monotonic_ns()
        try:
            result = preprocess(request, *args, **kwargs)
        except BaseException:
            end = time.monotonic_ns()
            observer.observe(lambda: observer.record_call(
                "request.engine_preprocess_failed", start, end, (request.request_id,),
                False, "EngineCore.preprocess_add_request"))
            raise
        end = time.monotonic_ns()

        def record():
            external = request.external_req_id
            if external is not None:
                observer.recorder.emit(
                    "request.identity_alias",
                    request_id_sha256=request_key(request.request_id),
                    external_request_id_sha256=request_key(external),
                    identity_source="EngineCoreRequest.external_req_id",
                    start_mono_ns=start, end_mono_ns=end, native_ok=True,
                    time_kind="host_native_call", source="EngineCore.preprocess_add_request",
                )

        observer.observe(record)
        return result

    core.preprocess_add_request = preprocessed


def install_lifecycle(core):
    recorder = LifecycleRecorder.from_env("sim")
    core._hisim_lifecycle = recorder
    if recorder is not None:
        observer = RequestLifecycleObserver(recorder)
        observer.install(core.scheduler)
        install_identity_alias(core, observer)
        from sglang_simulator.cache_boundaries import (
            install_cache_manager, install_connector, install_scheduler_transitions,
        )
        observer.observe(lambda: install_cache_manager(
            core.scheduler.kv_cache_manager, recorder.emit, observer.observe))
        observer.observe(lambda: install_scheduler_transitions(
            core.scheduler, recorder.emit, observer.observe))
        connector = getattr(core.scheduler, "connector", None)
        if connector is not None:
            observer.observe(lambda: install_connector(connector, recorder.emit, observer.observe))
        core._hisim_lifecycle_observer = observer
    return recorder
