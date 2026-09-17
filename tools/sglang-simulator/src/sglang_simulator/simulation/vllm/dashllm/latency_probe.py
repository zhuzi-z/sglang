"""DashLLM-side latency probe — logs the segments *outside* EngineCore.

Motivation
----------
Three layers measure "TTFT" with different boundaries, and only the innermost
one is observable from the simulator today:

- control plane (``ds-api-req.first_latency``): API Server receives the first
  request packet -> API Server receives the first response packet.
- turbo (``batch-request.log.first_latency``): turbo dequeues the request ->
  prefill finished.
- EngineCore (``RequestStats.gen_token_latencies[0]``): ``add_request`` enters
  EngineCore -> first token emitted.

The turbo-to-EngineCore gap is invisible: it covers dashservingd dispatch,
``EngineProcessor._process_input``, admission gating and the hop into the
separate EngineCore process. This probe logs that gap.

Why log-only
------------
``EngineProcessor.Process`` and ``_LLMBackend4vLLM.generate`` both run in the
``dashserving.worker`` process, so every timestamp here is same-process and
needs no registry. EngineCore runs in its own process but already exports
``queue_start`` / ``gen_token_latencies`` via ``request.jsonl``, so the two
sides are joined offline on the request id rather than shared in memory.

Join keys
---------
Two different ids are in play and both are logged, because neither alone spans
all three stages:

- ``uuid`` is dashserving's ``context.request_uuid``, the only id available at
  ``Process`` entry.
- ``rid`` is what dashllm hands to vLLM, so it is the one that matches
  ``RequestStats.rid`` on the EngineCore side.

``stage=generate`` sees both, so it is the bridge that lets ``stage=process``
(uuid-keyed) join to ``stage=engine`` (rid-keyed). Without it the process line
would be an island.

vLLM's rid is turbo's request id plus a ``-<suffix>`` segment, e.g. turbo
``ef94b79f-...-f01611350fae`` -> vLLM ``ef94b79f-...-f01611350fae-644db989``, so
``rid.rsplit("-", 1)[0]`` recovers the turbo id for the join against
``batch-request.log``. Since the exact kwarg carrying the id is not guaranteed
across dashllm versions, the probe scans a candidate list and logs which key it
resolved (``idkey``), so a miss shows up in the output instead of silently
degrading into blank ids.

Emitted lines (grep ``lat-probe``)::

    [lat-probe] stage=process  uuid=<id> daemon_ms=<ms> t_in_ms=<ms>
                d_dispatch_ms=<T1->T2> d_first_yield_ms=<T2->T6>
    [lat-probe] stage=generate rid=<id> idkey=<k> uuid=<id> recv_ms=<ms> bs=<n>
                d_pre_engine_ms=<T2->T3> d_first_out_ms=<T3->T5>

Segment map (T-numbers match the traced request path)::

    T1 daemon recv   -> T2 Process entry    = d_dispatch    (socket dispatch)
    T2 Process entry -> T3 generate() call  = d_pre_engine  (_process_input + admission)
    T3 generate()    -> T5 first engine out = d_first_out   (ZMQ hop + engine TTFT)
    T2 Process entry -> T6 first yield out  = d_first_yield (whole python-side span)

``d_first_out`` minus the EngineCore-side ``d_ttft_ms`` isolates the
cross-process hop, which is the quantity this probe exists to expose.

Enabled by default; see ``probe_log.enabled()`` for the kill switch.
"""

import threading
import time

from sglang_simulator.simulation import probe_log
from sglang_simulator.utils import get_logger

logger = get_logger()

_MISSING = probe_log.MISSING

# Most specific first. `request_id` is what dashllm hands to vLLM, so it is the
# one that matches RequestStats.rid; `request_uuid` is the dashserving-level id.
_ID_KEYS = ("request_id", "request_uuid", "infer_id", "id")


def _now_ms() -> float:
    return time.time() * 1000.0


def _resolve_id(kwargs: dict) -> tuple[str, str]:
    """Return ``(id, key_it_came_from)``; ``("", "none")`` when nothing matched."""
    for key in _ID_KEYS:
        val = kwargs.get(key)
        if isinstance(val, str) and val:
            return val, key
    ctx = kwargs.get("request_context")
    if isinstance(ctx, dict):
        for key in _ID_KEYS:
            val = ctx.get(key)
            if isinstance(val, str) and val:
                return val, "request_context." + key
    return "", "none"


def _resolve_uuid(kwargs: dict) -> str:
    """Dig out dashserving's request_uuid, the bridge to the ``stage=process`` line."""
    val = kwargs.get("request_uuid")
    if isinstance(val, str) and val:
        return val
    ctx = kwargs.get("request_context")
    if isinstance(ctx, dict):
        val = ctx.get("request_uuid")
        if isinstance(val, str) and val:
            return val
    return ""


def _install_process_probe() -> bool:
    """Timestamp dashservingd dispatch and the full python-side span."""
    try:
        import dashllm.core.engine.processor as processor_mod
    except Exception:
        return False

    cls = getattr(processor_mod, "EngineProcessor", None)
    if cls is None or getattr(cls, "_sglang_simulator_latency_probe", False):
        return False

    original_process = cls.Process

    def _probed_process(self, request, context):
        # Read before entering: dashservingd's own receive time (ms), the only
        # timestamp available from upstream of this process.
        try:
            daemon_ms = float(context.get_daemon_request_time() or 0) or _MISSING
        except Exception:
            daemon_ms = _MISSING
        t_in_ms = _now_ms()

        # request_uuid is the only id available this early; the generate line
        # carries both ids so the two stages can be joined on it.
        uuid = getattr(context, "request_uuid", "") or ""
        first = True
        for resp in original_process(self, request, context):
            if first:
                first = False
                probe_log.emit(
                    f"{probe_log.PREFIX} stage=process"
                    f" uuid={uuid}"
                    f" daemon_ms={daemon_ms:.1f} t_in_ms={t_in_ms:.1f}"
                    f" d_dispatch_ms={(t_in_ms - daemon_ms) if daemon_ms > 0 else _MISSING:.3f}"
                    f" d_first_yield_ms={_now_ms() - t_in_ms:.3f}"
                )
            yield resp

    cls.Process = _probed_process
    cls._sglang_simulator_latency_probe = True
    return True


def _install_generate_probe() -> bool:
    """Timestamp ``_process_input`` + admission, and the hop into EngineCore.

    Patches the same seam as ``kv_transfer_hook`` (``_LLMBackend4vLLM.generate``)
    because that is the verified place where the fully-built ``generate_input``
    kwargs arrive, including ``_t_dashllm_recv_ms`` and ``_engine_trace``.
    """
    try:
        import dashllm.core.backend._backend_vllm as backend_vllm
    except Exception:
        return False

    cls = getattr(backend_vllm, "_LLMBackend4vLLM", None)
    if cls is None or getattr(cls, "_sglang_simulator_latency_probe", False):
        return False

    original_generate = cls.generate

    def _probed_generate(self, model, **kwargs):
        t3_ms = _now_ms()
        rid, idkey = _resolve_id(kwargs)
        uuid = _resolve_uuid(kwargs)

        # Set by EngineProcessor.Process; absent if a dashllm version stops
        # forwarding the underscore keys, hence the explicit sentinel.
        recv_ms = kwargs.get("_t_dashllm_recv_ms")
        recv_ms = float(recv_ms) if isinstance(recv_ms, (int, float)) else _MISSING

        trace = kwargs.get("_engine_trace")
        # bs = in-flight request count on this rank at entry; a persistently
        # large value means requests pile up before reaching the engine.
        bs = trace.get("bs", _MISSING) if isinstance(trace, dict) else _MISSING

        first = True
        for out in original_generate(self, model, **kwargs):
            if first:
                first = False
                probe_log.emit(
                    f"{probe_log.PREFIX} stage=generate"
                    f" rid={rid} idkey={idkey} uuid={uuid}"
                    f" recv_ms={recv_ms:.1f} bs={bs}"
                    f" d_pre_engine_ms={(t3_ms - recv_ms) if recv_ms > 0 else _MISSING:.3f}"
                    f" d_first_out_ms={_now_ms() - t3_ms:.3f}"
                )
            yield out

    cls.generate = _probed_generate
    cls._sglang_simulator_latency_probe = True
    return True


def install_dashllm_latency_probe() -> None:
    """Install the probe when enabled; never fatal to the serving path."""
    if not probe_log.enabled():
        return
    try:
        ok_process = _install_process_probe()
        ok_generate = _install_generate_probe()
        ok_engine = _install_engine_internals_probe()
    except Exception as e:
        logger.warning("[lat-probe] install failed, probe disabled: %s", e)
        return
    logger.info(
        "[lat-probe] installed (process=%s, generate=%s, engine_internals=%s)",
        ok_process,
        ok_generate,
        ok_engine,
    )


def _mark(name: str, rid, **extra) -> None:
    """Emit one timestamped marker; the offline join does the subtraction.

    Markers carry absolute wall-clock ms instead of pre-computed deltas because
    the stages they cover are split across two threads (request thread vs
    engine-loop thread), so no single call site can see both ends.
    """
    tail = "".join(f" {k}={v}" for k, v in extra.items())
    probe_log.emit(
        f"{probe_log.PREFIX} stage=mark name={name} uuid={rid or ''}"
        f" t_ms={_now_ms():.1f} tid={threading.get_native_id()}{tail}"
    )


def _install_engine_internals_probe() -> bool:
    """Split the A and C segments inside dashllm's vLLM adapter.

    A (start_generate -> process_request) and C (add_request -> EngineCore) are
    each ~150ms while the CPU sits idle (loadavg ~1 on 4 cores), and neither is
    explained by the code on those paths -- they are dict building and a queue
    handoff. What they do straddle is a thread boundary: `_process_request` and
    `_enqueue_input_queue` run on the request thread, while `_add_request` runs
    on `_run_sync_loop`'s engine-loop thread after `_drain_input_queue`. So this
    records the thread id at every point and the `_input_queue` depth at
    enqueue, which is what distinguishes GIL/scheduling starvation from an
    actual queue backlog.

    Also fills a real gap: dashllm's own `dashllm_vllm_time_to_add_request_start`
    is unreachable in this configuration -- `_enable_multi_thread_process_request`
    makes `_generate` set `process_request_finished` up front, so the branch that
    records it never runs.
    """
    try:
        import dashllm.core.backend.engine._vllm_v1 as vllm_v1
    except Exception:
        return False

    cls = getattr(vllm_v1, "vLLMEngine", None)
    if cls is None or getattr(cls, "_sglang_simulator_latency_probe", False):
        return False

    # generate() is a generator, so entry is timed when the caller pulls the
    # first item -- which is exactly when the request thread starts working.
    orig_generate = cls.generate

    def _probed_generate(self, *, request_context=None, **kw):
        rid = (getattr(request_context, "request_uuid", "")
               or getattr(request_context, "request_id", ""))
        _mark("vllm_generate", rid)
        yield from orig_generate(self, request_context=request_context, **kw)

    cls.generate = _probed_generate

    orig_procreq = cls._process_request

    def _probed_procreq(self, *, request_id=None, **kw):
        _mark("procreq_in", request_id)
        try:
            return orig_procreq(self, request_id=request_id, **kw)
        finally:
            _mark("procreq_out", request_id)

    cls._process_request = _probed_procreq

    orig_enqueue = cls._enqueue_input_queue

    def _probed_enqueue(self, command, infer_id, pending_request=None):
        try:
            qsize = self._input_queue.qsize()
        except Exception:
            qsize = -1
        _mark("enqueue", infer_id, qsize=qsize,
              cmd=getattr(command, "name", command))
        return orig_enqueue(self, command, infer_id, pending_request)

    cls._enqueue_input_queue = _probed_enqueue

    orig_add = cls._add_request

    def _probed_add(self, infer_id, pending_request):
        _mark("addreq_in", infer_id)
        try:
            return orig_add(self, infer_id, pending_request)
        finally:
            _mark("addreq_out", infer_id)

    cls._add_request = _probed_add

    cls._sglang_simulator_latency_probe = True
    return True
