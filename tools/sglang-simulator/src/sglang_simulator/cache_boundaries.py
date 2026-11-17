"""Opt-in instance observers; native host spans are inclusive, not device time.

Install only from an enabled caller. No class patches, token/block inspection,
extra connector calls, waits, or device synchronization are performed here.
"""

import hashlib
import inspect
import time
from functools import wraps


_INSTALLATIONS = "_hisim_cache_boundary_installations"
_UNKNOWN = "unknown"
_CPU_CONTAINERS = (tuple, list, set, frozenset)
_COUNT_ARGS = (
    "num_new_tokens", "num_new_computed_tokens", "num_lookahead_tokens",
    "num_external_computed_tokens", "num_encoder_tokens", "num_computed_tokens",
    "num_external_tokens", "delay_cache_blocks", "drop_stale_output",
)


def _observe(observe, callback):
    # Do not depend on the observer returning the callback's value.
    box = []

    def once():
        if not box:
            box.append(None)
            box[0] = callback()
        return box[0]

    try:
        observe(once)
    except BaseException:
        pass
    return box[0] if box else None


def _attr(obj, name):
    try:
        return getattr(obj, name, None)
    except BaseException:
        return None


def _integer(value):
    return value if type(value) is int else None


def _boolean(value):
    return value if type(value) is bool else None


def _request_hash(rid):
    if type(rid) is str:
        return hashlib.sha256(rid.encode("utf-8")).hexdigest()
    if type(rid) is int:
        return hashlib.sha256(str(rid).encode("utf-8")).hexdigest()
    return None


def _status(request):
    status = _attr(request, "status")
    if type(status) is str:
        return status
    name = _attr(status, "name")
    return name if type(name) is str else None


def _snapshot(owner, request, rid=None):
    if rid is None:
        rid = _attr(request, "request_id")
    failed = _attr(owner, "failed_recving_kv_req_ids")
    failure = None
    if type(failed) in (set, frozenset) and type(rid) in (str, int):
        failure = rid in failed
    return {
        "request_id_sha256": _request_hash(rid),
        "status": _status(request),
        "failed_recv": failure,
        "num_preemptions": _integer(_attr(request, "num_preemptions")),
        "num_computed_tokens": _integer(_attr(request, "num_computed_tokens")),
    }


def _ids(value):
    # These are documented CPU request-ID collections, never KV block lists.
    if type(value) not in _CPU_CONTAINERS:
        return ()
    return tuple(rid for rid in value if type(rid) in (str, int))


def _request_for(owner, rid):
    requests = _attr(owner, "requests")
    return requests.get(rid) if type(requests) is dict else None


def _argument(args, kwargs, index, name):
    return args[index] if len(args) > index else kwargs.get(name)


def _signals(owner, output):
    records = []
    for field, direction in (("finished_sending", "send"),
                             ("finished_recving", "recv")):
        for rid in _ids(_attr(output, field)):
            request = _request_for(owner, rid)
            records.append((direction, request, _snapshot(owner, request, rid)))
    return records


def _before(owner, method, signature, args, kwargs):
    if method == "_update_from_kv_xfer_finished":
        output = _argument(args, kwargs, 0, "kv_connector_output")
        return {"signals": _signals(owner, output)}
    if method == "get_finished":
        meta = _attr(owner, "_worker_meta")
        failed = _attr(meta, "failed_loads")
        return {"failed_loads": failed.copy() if type(failed) is set else None}
    request = _argument(args, kwargs, 0, "request")
    data = {"request": request, "snapshot": _snapshot(owner, request),
            "arguments": {}}
    if signature is not None:
        bound = signature.bind_partial(*args, **kwargs).arguments
        for name in _COUNT_ARGS:
            value = bound.get(name)
            if type(value) in (int, bool):
                data["arguments"][name] = value
    return data


def _row(source, before, after, start, end, ok, error_type):
    before = before or {}
    after = after or {}
    return {
        "source": source,
        "start_mono_ns": start,
        "end_mono_ns": end,
        "native_ok": ok,
        "native_error_type": error_type,
        "time_kind": "host_native_call",
        "device_completion_confirmed": False,
        "host_span_inclusive": True,
        "exclusive_contribution_known": False,
        "request_id_sha256": before.get("request_id_sha256"),
        "status_before": before.get("status"),
        "status_after": after.get("status"),
        "failed_recv_before": before.get("failed_recv"),
        "failure_flag_before": before.get("failed_recv"),
        "num_preemptions_before": before.get("num_preemptions"),
        "num_preemptions_after": after.get("num_preemptions"),
        "num_computed_tokens_before": before.get("num_computed_tokens"),
        "num_computed_tokens_after": after.get("num_computed_tokens"),
    }


def _lookup_metadata(result):
    arity = len(result) if type(result) is tuple else None
    known = arity in (2, 3, 4)
    return {
        "result_arity": arity,
        "result_contract_known": known,
        "num_computed_tokens": _integer(result[1]) if known else None,
        "boundary": _integer(result[2]) if known and arity >= 3 else None,
        "hit_diverged": _boolean(result[3]) if known and arity == 4 else None,
    }


def _emit_signals(owner, emit, source, records, start, end, ok, error_type):
    for direction, request, before in records:
        after = _snapshot(owner, request)
        row = _row(source, before, after, start, end, ok, error_type)
        row.update(direction=direction, signal_only=True, remote_load_success=None)
        emit("request.kv_transfer_signal", **row)
    if not records and not ok:
        emit("request.kv_transfer_signal", **_row(
            source, None, None, start, end, ok, error_type),
            direction=None, signal_only=True, remote_load_success=None)


def _after(owner, emit, method, source, before, result, start, end, ok,
           error_type):
    before = before or {}
    if method == "_update_from_kv_xfer_finished":
        _emit_signals(owner, emit, source, before.get("signals", ()),
                      start, end, ok, error_type)
        return
    if method == "get_finished":
        records = []
        if ok and type(result) is tuple and len(result) == 2:
            failures = before.get("failed_loads")
            for index, direction in ((0, "send"), (1, "recv")):
                for rid in _ids(result[index]):
                    snap = _snapshot(owner, None, rid)
                    if type(failures) is set:
                        snap["failed_recv"] = rid in failures
                    records.append((direction, None, snap))
        _emit_signals(owner, emit, source, records, start, end, ok, error_type)
        return
    if method == "_try_promote_blocked_waiting_request" and ok and result is not True:
        return  # Native False busy polls intentionally produce no events.
    snapshot = before.get("snapshot", {})
    after = _snapshot(owner, before.get("request"))
    row = _row(source, snapshot, after, start, end, ok, error_type)
    row.update(before.get("arguments", {}))
    if method in ("get_computed_blocks", "get_computed_blocks_for_connector"):
        row.update(_lookup_metadata(result) if ok else _lookup_metadata(None))
        event = "request.kv_lookup"
    elif method == "allocate_slots":
        row["allocation_failed"] = result is None if ok else None
        row["allocation_succeeded"] = result is not None if ok else None
        event = "request.kv_allocation"
    elif method == "free":
        row.update(free_call_succeeded=ok, physical_kv_reclaimed_confirmed=False)
        event = "request.kv_release_call"
    elif method == "get_num_new_matched_tokens":
        known = ok and type(result) is tuple and len(result) == 2
        row.update(
            result_contract_known=known,
            lookup_pending=result[0] is None if known else None,
            num_new_matched_tokens=_integer(result[0]) if known else None,
            load_async=_boolean(result[1]) if known else None,
            remote_load_success=None,
        )
        event = "request.external_match"
    elif method == "update_state_after_alloc":
        row.update(connector_state_updated=ok, remote_load_success=None)
        event = "request.kv_allocation"
    elif method == "_preempt_request":
        row["preempt_call_succeeded"] = ok
        event = "request.preempt"
    else:
        remote = snapshot.get("status") == "WAITING_FOR_REMOTE_KVS"
        promoted = (ok and result is True and remote
                    and after.get("status") in ("WAITING", "PREEMPTED"))
        failure = snapshot.get("failed_recv")
        row.update(
            native_return_true=result is True if ok else None,
            promoted=promoted,
            remote_load_success=(not failure) if promoted and type(failure) is bool
            else None,
        )
        event = ("request.remote_kv_promoted" if remote
                 else "request.blocked_waiting_promoted")
    emit(event, **row)


def _wrap(owner, original, method, source, signature, emit, observe):
    @wraps(original)
    def wrapped(*args, **kwargs):
        before = _observe(observe, lambda: _before(
            owner, method, signature, args, kwargs))
        start = _observe(observe, time.monotonic_ns)
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            end = _observe(observe, time.monotonic_ns)
            _observe(observe, lambda: _after(
                owner, emit, method, source, before, None, start, end, False,
                type(error).__name__))
            raise
        else:
            # Take the end clock BEFORE any post-native metadata/emit work.
            end = _observe(observe, time.monotonic_ns)
            _observe(observe, lambda: _after(
                owner, emit, method, source, before, result, start, end, True, None))
            return result
    return wrapped


def _provenance(original):
    source = inspect.getsource(original)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _install(instance, role, methods, emit, observe):
    try:
        state = getattr(instance, _INSTALLATIONS, None)
        if type(state) is not dict:
            state = {}
            setattr(instance, _INSTALLATIONS, state)
        if role in state:
            return state[role]
        report = {
            "component": role, "available_methods": [], "missing_methods": [],
            "installed_methods": [], "uninstalled_methods": [],
            "source_sha256": {}, "source_hash_kind": "inspect_method_source",
        }
        state[role] = report
    except BaseException:
        return None
    for method in methods:
        original = _attr(instance, method)
        if not callable(original):
            report["missing_methods"].append(method)
            continue
        report["available_methods"].append(method)
        digest = _observe(observe, lambda: _provenance(original))
        report["source_sha256"][method] = digest or _UNKNOWN
        signature = _observe(observe, lambda: inspect.signature(original))
        source = _observe(observe, lambda: (
            type(instance).__module__ + "." + type(instance).__qualname__ + "." + method
        )) or role + "." + method
        try:
            wrapped = _wrap(instance, original, method, source, signature, emit, observe)
            setattr(instance, method, wrapped)
            report["installed_methods"].append(method)
        except BaseException:
            report["uninstalled_methods"].append(method)
    _observe(observe, lambda: emit(
        "request.cache_capabilities", **_row(
            role + ".instance_install", None, None, None, None, None, None),
        **report))
    return report


def install_cache_manager(manager, emit, observe):
    """Observe native lookups, allocation and free calls on this manager only."""
    return _install(manager, "cache_manager", (
        "get_computed_blocks", "get_computed_blocks_for_connector",
        "allocate_slots", "free",
    ), emit, observe)


def install_connector(connector, emit, observe):
    """Observe only naturally invoked connector methods, never poll proactively."""
    return _install(connector, "connector", (
        "get_num_new_matched_tokens", "update_state_after_alloc", "get_finished",
    ), emit, observe)


def install_scheduler_transitions(scheduler, emit, observe):
    """Distinguish failed-recv recovery/promotion from mere transfer signals."""
    return _install(scheduler, "scheduler_transitions", (
        "_try_promote_blocked_waiting_request", "_preempt_request",
        "_update_from_kv_xfer_finished",
    ), emit, observe)
