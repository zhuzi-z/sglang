"""Opt-in native request boundaries; no payload values or device operations."""

import atexit
import hashlib
import importlib
import inspect
import os
import queue
import threading
import time
from functools import wraps
from pathlib import Path

from sglang_simulator.hook import BaseHook


_COLLECTOR_SIDE = (Path(__file__).parent / "collector").is_dir()
_SIM_RECORDER = None
_SIM_INITIALIZED = False
_SIM_LOCK = threading.Lock()
_QUEUE_LOCK = threading.Lock()
_TRACE = None
_TRACE_INITIALIZED = False


def _enabled():
    if _COLLECTOR_SIDE:
        return (os.getenv("SIM_COLLECTOR_ENABLE", "").lower()
                in ("1", "true", "yes", "on")
                and os.getenv("SIM_COLLECTOR_MODE") == "pp_v3")
    return os.getenv("HISIM_LIFECYCLE_ENABLE") == "1"


def _close_sim():
    try:
        if _SIM_RECORDER is not None:
            _SIM_RECORDER.close()
    except BaseException:
        pass


def _after_fork():
    global _SIM_RECORDER, _SIM_INITIALIZED, _SIM_LOCK, _TRACE, _TRACE_INITIALIZED, _QUEUE_LOCK
    _SIM_RECORDER = None
    _SIM_INITIALIZED = False
    _SIM_LOCK = threading.Lock()
    _QUEUE_LOCK = threading.Lock()
    _TRACE = None
    _TRACE_INITIALIZED = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _emit(event, **metadata):
    global _SIM_RECORDER, _SIM_INITIALIZED, _TRACE, _TRACE_INITIALIZED
    try:
        if not _enabled():
            return
        if _COLLECTOR_SIDE:
            if _TRACE is None:
                _TRACE = importlib.import_module(
                    "sglang_simulator.collector.vllm_hook.trace_v2"
                )
            if _TRACE._RECORDER is None and not _TRACE_INITIALIZED:
                with _SIM_LOCK:
                    if _TRACE._RECORDER is None and not _TRACE_INITIALIZED:
                        _TRACE_INITIALIZED = True
                        _TRACE.initialize()
            if _TRACE._RECORDER is not None:
                _TRACE.emit(event, data=metadata)
        else:
            if not _SIM_INITIALIZED:
                with _SIM_LOCK:
                    if not _SIM_INITIALIZED:
                        _SIM_INITIALIZED = True
                        from sglang_simulator.lifecycle_recorder import LifecycleRecorder
                        _SIM_RECORDER = LifecycleRecorder.from_env("sim_frontend")
                        if _SIM_RECORDER is not None:
                            atexit.register(_close_sim)
            if _SIM_RECORDER is not None:
                _SIM_RECORDER.emit(event, **metadata)
    except BaseException:
        pass


def _now():
    try:
        return time.monotonic_ns()
    except BaseException:
        return None


def _observe(function, *args):
    try:
        return function(*args)
    except BaseException:
        return {"observer_metadata_ok": False}


def _record(event, source, start, end, native_ok, metadata, **extra):
    # This includes the emitter itself: even a broken replacement observer is safe.
    try:
        time_kind = extra.pop("_time_kind", "host_native_call")
        fields = {**metadata, **extra, "source": source,
                  "start_mono_ns": start, "end_mono_ns": end,
                  "native_ok": native_ok, "time_kind": time_kind,
                  "device_completion_confirmed": False}
        _emit(event, **fields)
    except BaseException:
        pass


def _hash_id(value):
    # Accept only native string IDs; never stringify payload-bearing objects.
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if type(value) is str else None


def _cpu_len(value):
    # Bypass subclass overrides, and never inspect tensors or token elements.
    if isinstance(value, list):
        return list.__len__(value)
    if isinstance(value, tuple):
        return tuple.__len__(value)
    return None


def _rows(value):
    if isinstance(value, list):
        return list.__iter__(value)
    if isinstance(value, tuple):
        return tuple.__iter__(value)
    return ()


def _bound(signature, args, kwargs):
    return signature.bind(*args, **kwargs).arguments


def _input_metadata(bound):
    return {"external_request_id_sha256": _hash_id(bound.get("request_id"))}


def _assigned_metadata(bound, result):
    # assign_request_id mutates its explicit argument, not its return value.
    request = bound["request"]
    return {
        "request_id_sha256": _hash_id(request.request_id),
        "external_request_id_sha256": _hash_id(request.external_req_id),
        "alias_kind": "external_to_internal_explicit",
    }


def _batch_metadata(bound):
    outputs = bound.get("engine_core_outputs")
    return {
        "request_ids_sha256": tuple(
            _hash_id(row.request_id) for row in _rows(outputs)
        ) if _cpu_len(outputs) is not None else None,
        "engine_core_output_count": _cpu_len(outputs),
        "timing_scope": "batch_shared",
        "per_request_apportionment": False,
        "includes_native_detokenization": True,
        "delivery_observed": False,
    }


def _empty_output_batch(bound):
    outputs = bound.get("engine_core_outputs")
    return type(outputs) in (list, tuple) and len(outputs) == 0


def _batch_result(bound, result):
    return {
        "returned_output_count": _cpu_len(getattr(result, "request_outputs", None)),
        "returned_output_count_is_total": False,
    }


def _state_metadata(bound):
    state = bound["self"]
    parent = state.parent_req
    return {
        "request_id_sha256": _hash_id(state.request_id),
        "external_request_id_sha256": _hash_id(state.external_req_id),
        "parent_external_request_id_sha256": (
            _hash_id(parent.external_req_id) if parent is not None else None
        ),
        "parent_fanout": parent is not None,
        "input_token_count": _cpu_len(bound.get("new_token_ids")),
        "delivery_observed": False,
    }


def _state_result(bound, result):
    if result is None:
        return {"output_suppressed": True, "output_materialized": False}
    return {"output_suppressed": False, "output_materialized": True,
            **_observe(_materialized_metadata, result)}


def _materialized_metadata(result):
    outputs = getattr(result, "outputs", None)
    finished = getattr(result, "finished", None)
    return {
        "output_external_request_id_sha256": _hash_id(result.request_id),
        "finished": finished if type(finished) is bool else None,
        "output_count": _cpu_len(outputs),
        "output_token_counts": tuple(
            _cpu_len(getattr(row, "token_ids", None)) for row in _rows(outputs)
        ) if _cpu_len(outputs) is not None else None,
    }


def _business_metadata(bound):
    context = bound["request_context"]
    business = context.request_id
    external = context.disaggregated_request_id or context.request_uuid or business
    return {
        "business_request_id_sha256": _hash_id(business),
        "external_request_id_sha256": _hash_id(external),
        "alias_kind": "business_to_engine_external_explicit",
    }


def _yield_metadata(result):
    return {
        "output_count": _cpu_len(result),
        "output_token_counts": tuple(
            _cpu_len(getattr(row, "token_ids", None)) for row in _rows(result)
        ) if _cpu_len(result) is not None else None,
        "delivery_observed": False,
    }


def _queue_observer(target, method, before):
    source = "queue.Queue." + method
    try:
        if type(target) is not queue.Queue:
            _capability(None, source, False, reason="exact_queue_instance_unavailable")
            return
        with _QUEUE_LOCK:
            native = getattr(target, method)
            if getattr(native, "_request_boundary_installed", False):
                return
            if not callable(native):
                raise TypeError
            if inspect.iscoroutinefunction(native) or inspect.isasyncgenfunction(native):
                raise TypeError

            @wraps(native)
            def wrapper(*args, **kwargs):
                metadata = _observe(before, args, kwargs)
                start = _now()
                try:
                    result = native(*args, **kwargs)
                except BaseException as exc:
                    end = _now()
                    _record("request.private_queue_" + method, source, start, end,
                            False, metadata, exception_type=type(exc).__name__)
                    raise
                else:
                    end = _now()
                    _record("request.private_queue_" + method, source, start, end,
                            True, metadata)
                    return result
            wrapper._request_boundary_installed = True
            setattr(target, method, wrapper)
        _capability(native, source, True)
    except BaseException:
        _capability(None, source, False, reason="queue_method_unavailable")


def _queue_header(args, kwargs):
    item = args[0] if args else kwargs.get("item")
    if type(item) not in (list, tuple) or len(item) < 2:
        _capability(None, "queue.Queue.put", False, reason="queue_header_unavailable")
        return {}
    metadata = {"external_request_id_sha256": _hash_id(item[0])}
    # Install before native put makes this same output queue visible to consumers.
    _queue_observer(item[1], "get", lambda args, kwargs: {
        **metadata, "queue_wait_is_request_residency": False})
    return metadata


def _generate_metadata(bound):
    try:
        _queue_observer(getattr(bound.get("self"), "_input_queue", None),
                        "put", _queue_header)
    except BaseException:
        _capability(None, "queue.Queue.put", False, reason="queue_instance_unavailable")
    return _business_metadata(bound)


def _infer_metadata(bound):
    return {"external_request_id_sha256": _hash_id(bound.get("infer_id"))}


def _dispatch_metadata(bound):
    return {**_infer_metadata(bound), "queue_dequeued": True,
            "dequeue_observed_mono_ns": _now()}


def _parse_metadata(bound):
    return {"business_request_id_sha256": _hash_id(bound.get("request_id")),
            "external_request_id_sha256": _hash_id(bound.get("request_uuid"))}


class _NativeGeneratorProxy:
    """PEP 380 delegate: exactly one native operation per resume or close."""

    def __init__(self, native, source, metadata):
        self.native = native
        self.source = source
        self.metadata = metadata
        self.alias_emitted = False

    def __iter__(self):
        return self

    def __next__(self):
        return self._resume("next", self.native.__next__)

    def send(self, value):
        return self._resume("send", self.native.send, value)

    def throw(self, *args):
        return self._resume("throw", self.native.throw, *args)

    def close(self):
        return self._resume("close", self.native.close)

    def _resume(self, operation, function, *args):
        start = _now()
        try:
            result = function(*args)
        except StopIteration:
            end = _now()
            self._event("return", operation, start, end, True)
            raise  # Preserves StopIteration.value without inspecting it.
        except BaseException as exc:
            end = _now()
            self._event("error", operation, start, end, False,
                        exception_type=type(exc).__name__)
            raise
        else:
            end = _now()  # Consumer yield pauses and observers are outside this span.
            if operation == "close":
                self._event("close", operation, start, end, True)
            else:
                self._event("yield", operation, start, end, True,
                            **_observe(_yield_metadata, result))
            return result

    def _event(self, outcome, operation, start, end, native_ok, **extra):
        time_kind = "host_native_call" if operation == "close" else "host_generator_resume"
        if native_ok and not self.alias_emitted:
            self.alias_emitted = True
            _record("request.business_external_alias", self.source, start, end,
                    True, self.metadata, _time_kind=time_kind)
        _record("request.private_generate_" + outcome, self.source, start, end,
                native_ok, self.metadata, operation=operation,
                _time_kind=time_kind, **extra)


def _delegate(native, source, before, signature, args, kwargs):
    bound = _observe(_bound, signature, args, kwargs)
    metadata = _observe(before, bound)
    return (yield from _NativeGeneratorProxy(native, source, metadata))


def _capability(native, source, installed, **metadata):
    def fingerprint():
        result = {"function_source_sha256": None, "source_sha256": None}
        try:
            text = inspect.getsource(native)
            result["function_source_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            path = inspect.getsourcefile(native)
            if path and Path(path).suffix == ".py":
                with open(path, "rb") as stream:
                    result["source_sha256"] = hashlib.sha256(stream.read()).hexdigest()
        except BaseException:
            pass
        return result

    stamp = _now()
    _record("request.boundary_capability", source, stamp, stamp, True,
            {**_observe(fingerprint), **metadata, "installed": installed,
             "version_claim": "runtime_fingerprint_only"})


def _install(target, method, source, event, before, after=None, generator=False):
    """Inspect the signature once at installation; preserve native descriptors."""
    try:
        descriptor = inspect.getattr_static(target, method)
        native = descriptor.__func__ if isinstance(
            descriptor, (staticmethod, classmethod)
        ) else descriptor
        if getattr(native, "_request_boundary_installed", False):
            return
        signature = inspect.signature(native)
        if generator and not inspect.isgeneratorfunction(native):
            _capability(native, source, False, reason="not_native_sync_generator")
            return
        if inspect.iscoroutinefunction(native) or inspect.isasyncgenfunction(native):
            _capability(native, source, False, reason="async_source_not_covered")
            return
        if generator:
            @wraps(native)
            def wrapper(*args, **kwargs):
                # Native generator creation validates arguments immediately but
                # executes no body; only delegate resumes receive engine timing.
                delegate = native(*args, **kwargs)
                return _delegate(delegate, source, before, signature, args, kwargs)
        else:
            @wraps(native)
            def wrapper(*args, **kwargs):
                bound = _observe(_bound, signature, args, kwargs)
                metadata = _observe(before, bound)
                empty_output = (event == "request.output_processed"
                                and _observe(_empty_output_batch, bound) is True)
                start = _now()
                try:
                    result = native(*args, **kwargs)
                except BaseException as exc:
                    end = _now()
                    failure_event = (
                        "request.identity_assignment_failed"
                        if event == "request.identity_alias" else event
                    )
                    _record(failure_event, source, start, end, False, metadata,
                            exception_type=type(exc).__name__)
                    raise
                else:
                    end = _now()
                    if empty_output:
                        return result
                    if after is not None:
                        metadata.update(_observe(after, bound, result))
                    _record(event, source, start, end, True, metadata)
                    return result
        wrapper._request_boundary_installed = True
        replacement = wrapper
        if isinstance(descriptor, staticmethod):
            replacement = staticmethod(wrapper)
        elif isinstance(descriptor, classmethod):
            replacement = classmethod(wrapper)
        setattr(target, method, replacement)
        _capability(native, source, True, synchronous_generator=generator)
    except BaseException:
        # Missing methods/signatures are capabilities, never import failures.
        _capability(None, source, False, reason="native_method_unavailable")


class C_RequestInputProcessorHook(BaseHook):
    HOOK_MODULE_NAME = "vllm.v1.engine.input_processor"
    HOOK_CLASS_NAME = "InputProcessor"

    @classmethod
    def hook(cls, target):
        if _enabled():
            source = cls.HOOK_MODULE_NAME + "." + cls.HOOK_CLASS_NAME
            _install(target, "process_inputs", source + ".process_inputs",
                     "request.input_processed", _input_metadata)
            _install(target, "assign_request_id", source + ".assign_request_id",
                     "request.identity_alias", lambda bound: {}, _assigned_metadata)
        return target


class C_RequestOutputProcessorHook(BaseHook):
    HOOK_MODULE_NAME = "vllm.v1.engine.output_processor"
    HOOK_CLASS_NAME = "OutputProcessor"

    @classmethod
    def hook(cls, target):
        if _enabled():
            _install(target, "process_outputs", cls.HOOK_MODULE_NAME + "." +
                     cls.HOOK_CLASS_NAME + ".process_outputs",
                     "request.output_processed", _batch_metadata, _batch_result)
        return target


class C_RequestStateOutputHook(BaseHook):
    HOOK_MODULE_NAME = "vllm.v1.engine.output_processor"
    HOOK_CLASS_NAME = "RequestState"

    @classmethod
    def hook(cls, target):
        if _enabled():
            _install(target, "make_request_output", cls.HOOK_MODULE_NAME + "." +
                     cls.HOOK_CLASS_NAME + ".make_request_output",
                     "request.output_materialization", _state_metadata, _state_result)
        return target


class C_PrivateRequestEngineHook(BaseHook):
    HOOK_MODULE_NAME = "dashllm.core.backend.engine._vllm_v1"
    HOOK_CLASS_NAME = "vLLMEngine"

    @classmethod
    def hook(cls, target):
        if _enabled():
            source = cls.HOOK_MODULE_NAME + "." + cls.HOOK_CLASS_NAME
            _install(target, "generate", source + ".generate",
                     None, _generate_metadata, generator=True)
            _install(target, "stop_generate", source + ".stop_generate",
                     "request.private_abort_requested", _business_metadata)
            _install(target, "_dispatch_request", source + "._dispatch_request",
                     "request.private_dispatched", _dispatch_metadata)
            _install(target, "_add_request", source + "._add_request",
                     "request.private_added", _infer_metadata)
            _install(target, "_parse_output", source + "._parse_output",
                     "request.private_parsed", _parse_metadata,
                     lambda bound, result: _yield_metadata(result))
        return target


HOOKS = [C_RequestInputProcessorHook, C_RequestOutputProcessorHook,
         C_RequestStateOutputHook, C_PrivateRequestEngineHook]
