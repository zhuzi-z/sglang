import gzip
import json
import os
import sys
import threading
import time
import torch
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from array import array

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils.json import CustomJsonEncoder

# Single knob for export shaping, comma-separated flags
# (SIM_COLLECTOR_EXPORT):
#   content: "ids" (default) keeps input_ids/output_ids, "no_ids" drops them
#   format:  "gzip" (default) writes .jsonl.gz, "plain" writes plain .jsonl
_EXPORT_FLAGS = frozenset(
    os.getenv("SIM_COLLECTOR_EXPORT", "no_ids,gzip").lower().split(",")
)
EXPORT_TOKEN_IDS = "no_ids" not in _EXPORT_FLAGS
EXPORT_GZIP = "plain" not in _EXPORT_FLAGS


@dataclass(slots=True)
class RequestInfos:
    rid: str = ""
    created_time: Optional[float] = None
    client_created_time: Optional[float] = None  # client created time
    server_created_time: Optional[float] = None
    queue_start: float = 0.0
    queue_end: float = 0.0
    output_length: int = 0
    input_length: int = 0
    total_kv_hit_len: int = 0
    local_kv_hit_len: int = 0
    ext_kv_hit_len: int = 0
    remote_kv_ready_ts: float = 0.0  # scheduler first sees finished_recving
    input_ids: array = field(default_factory=lambda: array("i"))
    output_ids: array = field(default_factory=lambda: array("i"))


@dataclass(slots=True)
class BatchInfos:
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    forward_mode: int = 0
    # List of tuples (req_id, extend_input_len, past_kv_len, output_len)
    requests: list[tuple[str, int, int, int]] = field(default_factory=list)
    iter_latency: float = 0.0
    sample_tokens_latency: float = 0.0
    logprobs_req_count: int = 0
    logprobs_tokens: int = 0
    logprobs_n: int = 0
    # X2: per-segment RPC-2 breakdown in ms
    # ({sample, draft, bookkeep, eplb, other}); None when segment hooks are
    # unavailable on this vllm version.
    st_segments_ms: Optional[dict] = None


BATCH_INFOS: list[BatchInfos] = []

REQUEST_INFOS: dict[str, RequestInfos] = defaultdict(RequestInfos)


# --- CUDA-event based timing -------------------------------------------------
# The collector must NOT call torch.cuda.synchronize(): a forced device sync
# serializes the pipeline, wrecks async-scheduling overlap and skews the very
# latencies we try to measure. Instead we bracket each measured region with a
# pair of timing cuda events recorded on the main stream (mirroring the
# async_event_pool pattern in vllm's GPUModelRunner). The elapsed time is read
# lazily from AsyncModelRunnerOutput.get_output() -- i.e. right after vllm's own
# synchronize has already drained the stream past our end event -- so the hook
# never introduces any synchronization of its own.


# CUDA event timing is GPU-only. CPU/TPU builds (e.g. vllm 0.29.0+cpu on
# torch_tpu) raise "Torch not compiled with CUDA enabled" from
# torch.cuda.Event(...). Probe once at import so that failure can never happen
# inside the engine busy loop (wrapped_execute_model constructs the timer before
# delegating to the original forward).
def _probe_cuda_event_timing() -> bool:
    try:
        torch.cuda.Event(enable_timing=True)
        return True
    except Exception:
        return False


_CUDA_EVENT_TIMING = _probe_cuda_event_timing()


class _EventTimer:
    """A start/end pair for one measured region.

    Uses CUDA events when the build provides them (GPU) and falls back to wall
    clock otherwise (CPU/TPU). The fallback measures the host-observed span
    between the two record calls, i.e. it is NOT device time -- flag this when
    comparing TPU numbers against GPU baselines.
    """

    __slots__ = ("_start", "_end", "_t0", "_t1")

    def __init__(self) -> None:
        if _CUDA_EVENT_TIMING:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
        else:
            self._start = None
            self._end = None
        self._t0 = 0.0
        self._t1 = 0.0

    def record_start(self) -> None:
        if self._start is not None:
            self._start.record()
        else:
            self._t0 = time.time()

    def record_end(self) -> None:
        if self._end is not None:
            self._end.record()
        else:
            self._t1 = time.time()

    def elapsed_s(self) -> Optional[float]:
        """Elapsed seconds for the region, or None if not available yet.

        Event.elapsed_time() returns milliseconds and raises if either event
        has not completed. It is only safe to call after a native synchronize
        has passed self._end; we still guard it so a rare not-ready race can
        never crash the serving engine.
        """
        if self._start is not None:
            try:
                return self._start.elapsed_time(self._end) / 1000.0
            except RuntimeError:
                return None
        if self._t0 <= 0.0 or self._t1 <= 0.0:
            return None
        return self._t1 - self._t0


# Bridges the execute_model (RPC-1) timing to the sample_tokens (RPC-2) call of
# the same step. A single slot is enough because the two RPCs run back-to-back
# within a step; only get_output() is overlapped across steps.
_PENDING_EXEC: Optional[tuple] = None

# --- RPC-2 per-segment timing (X2) -----------------------------------------
# CPU wall-clock contribution of the four hookable boundaries inside
# Worker.sample_tokens: _sample / propose_draft_token_ids /
# _bookkeeping_sync / eplb_step+fluid_step. No extra cuda.synchronize is
# inserted, so engine behavior is unchanged; an async GPU tail of one segment
# surfaces as CPU wait inside the next segment that synchronizes on it
# (e.g. draft's GPU time lands in bookkeep's D2H wait), and the residual
# (sample_tokens_latency - sum(segments)) is reported as "other".
SEG_KEYS = ("sample", "draft", "bookkeep", "eplb")
_SEG_STATE = {"active": False, "cur": None}


def _finalize_on_get_output(output, finalize):
    """Run ``finalize`` right after vllm has synchronized this step's output.

    In async scheduling ``output`` is an ``AsyncModelRunnerOutput`` whose
    ``get_output()`` blocks on the D2H copy event -- our timing events are
    guaranteed complete by then, so we read them there. In the synchronous
    path there is no ``get_output`` and vllm has already synchronized before
    returning, so we finalize immediately. Either way the hook never calls
    synchronize itself.
    """
    original_get_output = getattr(output, "get_output", None)
    if original_get_output is None:
        finalize()
        return output

    def wrapped_get_output(*args, **kwargs):
        result = original_get_output(*args, **kwargs)
        finalize()
        return result

    output.get_output = wrapped_get_output
    return output


def _print_batch_info(batch_info: "BatchInfos") -> None:
    """Print one recorded batch at its finalize point (no request ids).

    The per-request tuples are (rid, extend_input_len, past_kv_len,
    output_len); the leading rid is dropped so no request id is printed.

    Writes directly to stdout's underlying buffer so that vLLM's
    decorate_logs prefix (e.g. [2/4,TP2][pid=...]) is bypassed.
    """
    row = asdict(batch_info)
    row["requests"] = [tuple(req[1:]) for req in row.get("requests", [])]
    line = "[sim_colletor] " + json.dumps(row, cls=CustomJsonEncoder) + "\n"
    sys.stdout.buffer.write(line.encode())
    sys.stdout.buffer.flush()


def _round6(value: Optional[float]) -> Optional[float]:
    """Round to 6 decimals to shrink the exported payload (None-safe)."""
    return round(value, 6) if value is not None else None


def _open_export(path: str):
    """Open a JSONL export file; gzip-compressed (level 1) when EXPORT_GZIP."""
    if EXPORT_GZIP:
        return gzip.open(path, "wt", compresslevel=1)
    return open(path, "w")


_LAST_EXPORT_THREAD: Optional[threading.Thread] = None


def _run_export(export_fn, is_start: bool) -> None:
    """Export on a side thread for start_profile (the engine keeps
    serving), blocking for stop_profile (data flushed before moving on)."""
    global _LAST_EXPORT_THREAD
    # Exports of one process write the same file: wait out the previous one.
    if _LAST_EXPORT_THREAD is not None:
        _LAST_EXPORT_THREAD.join()
        _LAST_EXPORT_THREAD = None
    if is_start:
        _LAST_EXPORT_THREAD = threading.Thread(target=export_fn, daemon=True)
        _LAST_EXPORT_THREAD.start()
    else:
        export_fn()


# Per-round export subdirectory --------------------------------------------
# EngineCore.profile and every Worker.profile fire exactly once per profile
# call (start_profile/stop_profile fan out via collective_rpc to all
# workers), so each process can keep its own call counter and the counters
# stay aligned 1:1 across the EngineCore and Worker processes. The run
# subdirectory below is therefore pairable across processes purely by the
# "run{NNN}" counter -- no cross-process coordination (shared files, extra
# RPC fields) is needed to match rank0.requests.jsonl with the
# rank{N}.schedule_batch.jsonl of the same round.
_PROFILE_ROUND = 0
# Readable wall-clock tag of this process, fixed once at import time. Only a
# human-friendly prefix; the run counter above is the pairing key.
_PROCESS_START_TS = time.strftime("%Y%m%d_%H%M%S")
# Run subdirectory minted by the latest start_profile; reused by stop_profile
# so one round's start/stop exports land in the same directory.
_CURRENT_RUN_SUBDIR: Optional[str] = None


# ---- HybridConnector KV transfer latency collection (load/store) ----
# Timing brackets V6dSwapHandler.async_swap: start at coroutine entry (the
# next stmt is `await client.async_get`, the real transfer kick-off), end
# right before return (after the inline `while not event.query()` loop, i.e.
# data truly landed). Wall clock on purpose: the path is CPU/RPC async, CUDA
# events cannot see it.
@dataclass(slots=True)
class TransferInfos:
    rid: str = ""               # request_id, joins with requests.jsonl
    direction: str = ""         # "load"/"store" (self._swap_in)
    latency_s: float = 0.0      # pure transfer latency (RPC pull + CUDA copy)
    num_blocks: int = 0         # len(keys)
    nbytes: int = 0             # num_blocks * block_bytes (0 if unknown)
    cross_node: Optional[bool] = None  # None = unknown (see _REQ_CROSS_NODE)
    queued_s: float = 0.0       # submit -> coroutine entry (event-loop queue)
    job_id: int = 0
    submit_ts: float = 0.0      # async_start_load_kv submit time (main thread)
    start_ts: float = 0.0       # async_swap coroutine entry (transfer begins)
    done_ts: float = 0.0        # async_swap pre-return (event.query done)
    error: str = ""


TRANSFER_INFOS: list[TransferInfos] = []
# req_id -> cross_node, stashed by C_V6dWorkerHook at submit time from
# metadata.external_tokens (>0 == remote hit), consumed by C_V6dSwapHook.
_REQ_CROSS_NODE: dict[str, bool] = {}
# req_id -> submit wall-clock, paired with the cross_node stash above.
_REQ_SUBMIT_TS: dict[str, float] = {}
# store-side queue anchor (async_start_store_kv submit time).
_REQ_STORE_SUBMIT_TS: dict[str, float] = {}


def get_output_dir(is_start: bool = True) -> str:
    """Export dir of the current profile round:
    $SIM_COLLECTOR_OUTPUT_DIR/engine{RANK}/{proc_start_ts}_run{NNN}/
    (pid{PID} instead of engine{RANK} when DS_LLM_PROC_RANK is unset).

    A fresh run subdirectory is minted on every start_profile (and lazily on
    a stop_profile that never saw a start), so successive profile rounds and
    service restarts no longer overwrite earlier exports -- previously the
    fixed directory plus "w"-mode files kept only the latest round.
    bandwidth_profile.json intentionally stays at the top level (node-level
    hardware property, calibrated once).
    """
    global _PROFILE_ROUND, _CURRENT_RUN_SUBDIR
    base = os.getenv("SIM_COLLECTOR_OUTPUT_DIR", os.getcwd())
    engine_rank = os.getenv("DS_LLM_PROC_RANK")
    if engine_rank is not None:
        sub_dir = f"engine{engine_rank}"
    else:
        sub_dir = f"pid{os.getpid()}"
    if is_start or _CURRENT_RUN_SUBDIR is None:
        _PROFILE_ROUND += 1
        _CURRENT_RUN_SUBDIR = f"{_PROCESS_START_TS}_run{_PROFILE_ROUND:03d}"
    output_dir = os.path.join(base, sub_dir, _CURRENT_RUN_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _export_engine_requests(is_start: bool) -> None:
    """Dump the engine-side REQUEST_INFOS of the current profile round.

    Called from the EngineCore profile hook (in-process / leader node, where
    a Scheduler lives). Kept as a module-level function so a second caller
    (e.g. an executor-level profile shim) can reuse it without duplicating
    the JSON shaping.
    """
    global REQUEST_INFOS
    output_dir = get_output_dir(is_start=is_start)

    request_infos = REQUEST_INFOS
    REQUEST_INFOS = defaultdict(RequestInfos)

    def _export():
        path = f"{output_dir}/rank0.requests.jsonl"
        if EXPORT_GZIP:
            path += ".gz"
        with _open_export(path) as f:
            for req_infos in request_infos.values():
                row = asdict(req_infos)
                if not EXPORT_TOKEN_IDS:
                    # Token ids dominate the payload; drop on request.
                    del row["input_ids"]
                    del row["output_ids"]
                f.write(json.dumps(row, cls=CustomJsonEncoder) + "\n")
        print(f"Request data has been saved to {path}")

    _run_export(_export, is_start)


class C_VLLMEngineArgsHook(BaseHook):
    """Hook EngineArgs to default the profiler to nsys (cudaProfilerApi)."""

    HOOK_CLASS_NAME = "EngineArgs"
    HOOK_MODULE_NAME = "vllm.engine.arg_utils"

    @classmethod
    def hook(cls, target):
        original_post_init = target.__post_init__

        def wrapped_post_init(self):
            original_post_init(self)
            profiler_config = getattr(self, "profiler_config", None)
            if isinstance(profiler_config, dict):
                if profiler_config.get("profiler") is None:
                    profiler_config["profiler"] = "cuda"
            elif profiler_config is not None:
                if profiler_config.profiler is None:
                    profiler_config.profiler = "cuda"

        target.__post_init__ = wrapped_post_init


class C_WorkerHook(BaseHook):
    HOOK_MODULE_NAME = "vllm.v1.worker.gpu_worker"
    HOOK_CLASS_NAME = "Worker"

    @classmethod
    def hook(cls, target) -> None:

        # Window RPC-1: bracket execute_model's forward with timing cuda events
        # (no synchronize). In the two-RPC architecture execute_model returns
        # None, so the elapsed time is read later -- paired with the
        # sample_tokens window under a single native get_output synchronize.
        original_execute_model = target.execute_model

        # *args/**kwargs: callers differ across vllm builds -- e.g. the
        # torchtpu PP executor passes intermediate_tensors positionally, so a
        # fixed arity here turns one extra argument into a TypeError that kills
        # EngineCore. Always forward verbatim to the original.
        def wrapped_execute_model(self, scheduler_output: "SchedulerOutput",
                                  *args, **kwargs):

            # SchedulerOutput shape is fork-dependent (torchtpu's PP chunk
            # scheduler builds its own), so parse defensively: a missing or
            # renamed field must never abort the forward -- an exception raised
            # before the original call would kill the engine busy loop.
            batch_req_infos = {}
            forward_mode = 0
            logprobs_req_count = 0
            logprobs_tokens = 0
            logprobs_n = 0
            try:
                for req_id, sched_token in scheduler_output.num_scheduled_tokens.items():
                    batch_req_infos[req_id] = {"extend_input_len": sched_token}

                for req_id, completed_token, output_token in zip(
                    scheduler_output.scheduled_cached_reqs.req_ids,
                    scheduler_output.scheduled_cached_reqs.num_computed_tokens,
                    scheduler_output.scheduled_cached_reqs.num_output_tokens
                ):
                    batch_req_infos[req_id]["past_kv_len"] = completed_token
                    batch_req_infos[req_id]["output_len"] = output_token

                for req in scheduler_output.scheduled_new_reqs:
                    batch_req_infos[req.req_id]["past_kv_len"] = req.num_computed_tokens
                    batch_req_infos[req.req_id]["output_len"] = 0

                forward_mode = 2 if all([num_token == 1 for num_token in scheduler_output.num_scheduled_tokens.values()]) else 1

                # Per-step logprobs form (for independent logprobs-compensation
                # training). Only new reqs carry sampling_params in SchedulerOutput;
                # cached reqs (decode steps) would need engine-side hook or a
                # differential stress test to capture. total_num_scheduled_tokens
                # is the step-level token count used as the MTP-compensation feature.
                for req in scheduler_output.scheduled_new_reqs:
                    sp = getattr(req, "sampling_params", None)
                    if sp is not None and getattr(sp, "logprobs", None):
                        logprobs_req_count += 1
                        logprobs_tokens += scheduler_output.num_scheduled_tokens.get(
                            req.req_id, 0
                        )
                        logprobs_n = max(logprobs_n, sp.logprobs)
            except Exception as e:
                print(
                    f"[sim_collector] step capture skipped: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                batch_req_infos = {}

            global _PENDING_EXEC

            # Bracket the forward with timing cuda events instead of syncing.
            exec_timer = _EventTimer()
            start_wall = time.time()
            exec_timer.record_start()
            ret = original_execute_model(self, scheduler_output, *args, **kwargs)
            exec_timer.record_end()

            batch_info = None
            if len(batch_req_infos):
                batch_info = BatchInfos(
                    start_timestamp=_round6(start_wall),
                    # end_timestamp / iter_latency are filled in once the timing
                    # events are read after vllm's own synchronize (see below).
                    forward_mode=forward_mode,
                    # Keep only the last 12 chars of rid.
                    requests=[
                        (
                            rid[-12:] if len(rid) > 12 else rid,
                            req_info["extend_input_len"],
                            req_info["past_kv_len"],
                            req_info["output_len"],
                        )
                        for rid, req_info in batch_req_infos.items()
                    ],
                    logprobs_req_count=logprobs_req_count,
                    logprobs_tokens=logprobs_tokens,
                    logprobs_n=logprobs_n,
                )
                BATCH_INFOS.append(batch_info)

            def _finalize_exec(sample_latency: Optional[float] = None,
                               seg: Optional[dict] = None,
                               deferred: bool = False):
                if batch_info is None:
                    return
                if _CUDA_EVENT_TIMING or not deferred:
                    # Same window as the GPU path: record_start/record_end
                    # bracket the forward call itself. On CPU/TPU the pair is
                    # host wall clock instead of device events, which is the
                    # closest available equivalent while the forward is
                    # synchronous.
                    iter_latency = exec_timer.elapsed_s()
                else:
                    # The forward returned before the work completed
                    # (AsyncModelRunnerOutput / two-RPC submission), so the
                    # bracket only covers submission. Measure to this finalize
                    # point instead -- it runs after vllm's own get_output.
                    iter_latency = time.time() - start_wall
                if iter_latency is None:
                    return
                batch_info.iter_latency = _round6(iter_latency)
                batch_info.end_timestamp = _round6(start_wall + iter_latency)
                if sample_latency is not None:
                    batch_info.sample_tokens_latency = _round6(sample_latency)
                if seg is not None:
                    st_ms = (sample_latency or 0.0) * 1e3
                    batch_info.st_segments_ms = {
                        **{k: round(v, 4) for k, v in seg.items()},
                        "other": round(st_ms - sum(seg.values()), 4),
                    }
                # Print the batch info at its recording (finalize) point.
                _print_batch_info(batch_info)

            # Two-RPC vllm: execute_model returns None and the real output
            # (carrying get_output) comes from sample_tokens -- defer to it so
            # both windows are read together after a single native synchronize.
            if ret is None:
                _PENDING_EXEC = (
                    lambda sl=None, sg=None: _finalize_exec(sl, sg, deferred=True),
                )
                return ret

            # Single-RPC vllm: this call already produced the output object.
            # If it exposes get_output the real work lands later (async), so the
            # timing has to be settled at that point too.
            _deferred = getattr(ret, "get_output", None) is not None
            return _finalize_on_get_output(
                ret, lambda: _finalize_exec(deferred=_deferred)
            )
        
        target.execute_model = wrapped_execute_model

        # Two-window collection (X1): if this vllm version has sample_tokens
        # (two-RPC architecture), time it with cuda events to measure the full
        # RPC-2 span (sampler + MTP draft forward + bookkeeping D2H + output
        # construction). It aligns 1:1 by call order with the execute_model
        # window (RPC-1, measured above).
        # Skipped automatically on older vllm without this method; the
        # iter_latency semantics remain unchanged.
        original_sample_tokens = getattr(target, "sample_tokens", None)
        if original_sample_tokens is not None:

            def wrapped_sample_tokens(self, *args, **kwargs):
                global _PENDING_EXEC

                # Bracket RPC-2 with timing cuda events instead of syncing.
                sample_timer = _EventTimer()
                sample_timer.record_start()
                sample_wall = time.time()
                # X2: open the per-segment timing window for this RPC-2 call.
                _SEG_STATE["cur"] = {k: 0.0 for k in SEG_KEYS}
                _SEG_STATE["active"] = True
                try:
                    ret = original_sample_tokens(self, *args, **kwargs)
                finally:
                    _SEG_STATE["active"] = False
                sample_timer.record_end()

                seg = _SEG_STATE["cur"]
                pending_exec = _PENDING_EXEC
                _PENDING_EXEC = None

                _deferred = getattr(ret, "get_output", None) is not None

                def _finalize():
                    if _CUDA_EVENT_TIMING or not _deferred:
                        # Bracket semantics of the GPU path (host clock here).
                        sample_latency = sample_timer.elapsed_s()
                    else:
                        # RPC-2 returned before completion: settle the window at
                        # get_output, after vllm's own synchronize.
                        sample_latency = time.time() - sample_wall
                    # Hand the paired sample latency and the per-segment
                    # breakdown to the execute_model window, which records
                    # them on the batch and prints it.
                    if pending_exec is not None:
                        pending_exec[0](sample_latency, seg)

                return _finalize_on_get_output(ret, _finalize)

            target.sample_tokens = wrapped_sample_tokens

        original_profile = getattr(target, "profile", None)

        # *args/**kwargs plus the optional trailing parameters: torchtpu's
        # TPUWorker.profile takes (is_start, profile_prefix, profiler_kwargs)
        # while the GPU Worker only takes is_start. A fixed arity here either
        # raises TypeError on the TPU path or breaks the older GPU path.
        def override_profile(self, is_start: bool = True, profile_prefix=None,
                             profiler_kwargs=None, *args, **kwargs):
            global BATCH_INFOS, TRANSFER_INFOS

            output_dir = get_output_dir(is_start=is_start)
            # The worker class differs across forks: vllm's Worker always has
            # .rank, torchtpu's TPUWorker may not -- fall back to the rank the
            # launcher exports so the file name stays rank{N}.
            rank = getattr(self, "rank", None)
            if rank is None:
                rank = os.getenv("DS_LLM_PROC_RANK", "0")
            rank_suffix = f"rank{rank}"

            # sample_tokens_latency is already recorded on each batch at its
            # finalize point (paired 1:1 with the sample_tokens call).
            batch_infos = BATCH_INFOS
            BATCH_INFOS = []

            def _export():
                path = f"{output_dir}/{rank_suffix}.schedule_batch.jsonl"
                if EXPORT_GZIP:
                    path += ".gz"
                with _open_export(path) as f:
                    for batch_info in batch_infos:
                        f.write(
                            json.dumps(batch_info, cls=CustomJsonEncoder) + "\n"
                        )
                print(f"Schedule batch data has been saved to {path}")

            transfer_infos = TRANSFER_INFOS
            TRANSFER_INFOS = []

            def _export_transfer():
                path = f"{output_dir}/{rank_suffix}.transfer.jsonl"
                if EXPORT_GZIP:
                    path += ".gz"
                with _open_export(path) as f:
                    for t in transfer_infos:
                        f.write(json.dumps(asdict(t), cls=CustomJsonEncoder) + "\n")
                print(f"Transfer data has been saved to {path}")

            _run_export(_export, is_start)
            _run_export(_export_transfer, is_start)

        target.profile = override_profile



class C_ModelRunnerSegHook(BaseHook):
    """Per-segment timing inside Worker.sample_tokens (X2).

    Wraps the four hookable boundaries of RPC-2 on GPUModelRunner:
    _sample / propose_draft_token_ids / _bookkeeping_sync / eplb_step +
    fluid_step (merged into "eplb"). Timing is active only inside the
    sample_tokens window (_SEG_STATE["active"]), so calls from warmup /
    dummy_run paths never pollute the step-aligned segments. Methods
    missing on this vllm version are skipped with a warning.
    """

    HOOK_MODULE_NAME = "vllm.v1.worker.gpu_model_runner"
    HOOK_CLASS_NAME = "GPUModelRunner"

    @classmethod
    def hook(cls, target) -> None:

        def _timed(attr, key):
            fn = getattr(target, attr, None)
            if fn is None:
                print(f"[seg-hook] {attr} not found on GPUModelRunner; skipped")
                return

            def wrapped(*args, **kwargs):
                if not _SEG_STATE["active"]:
                    return fn(*args, **kwargs)
                t0 = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    _SEG_STATE["cur"][key] += (time.perf_counter() - t0) * 1e3

            setattr(target, attr, wrapped)

        _timed("_sample", "sample")
        _timed("propose_draft_token_ids", "draft")
        _timed("_bookkeeping_sync", "bookkeep")
        _timed("eplb_step", "eplb")
        _timed("fluid_step", "eplb")



class C_EngineCoreHook(BaseHook):

    HOOK_MODULE_NAME = "vllm.v1.engine.core"
    HOOK_CLASS_NAME = "EngineCore"

    @classmethod
    def hook(cls, target) -> None:

        original_profile = target.profile

        # Profile signature differs across vllm builds; forward every field the
        # caller may have passed (the utility RPC dispatcher validates arity
        # against this signature before invoking us).
        def wrapped_profile(self, is_start: bool = True,
                            profile_prefix: Optional[str] = None,
                            *args, **kwargs):
            _export_engine_requests(is_start)

            # Call the original profile method to trigger the lower-level worker
            return original_profile(self, is_start, profile_prefix)

        target.profile = wrapped_profile



class C_TpuWorkerHook(C_WorkerHook):
    """Same worker-side collection, registered on the torchtpu worker class.

    Class hooks match (defining module, class name) exactly, so the GPU
    declaration (vllm.v1.worker.gpu_worker.Worker) never fires on a TPU build
    whose worker is vllm_torchtpu.worker.tpu_worker.TPUWorker. Subclassing
    reuses the identical hook body without touching the GPU path.
    """

    HOOK_MODULE_NAME = "vllm_torchtpu.worker.tpu_worker"
    HOOK_CLASS_NAME = "TPUWorker"


class C_MultiprocExecutorProfileHook(BaseHook):
    """Give MultiprocExecutor the start_profile/stop_profile pair.

    vllm only defines Executor.profile(is_start, profile_prefix). dashllm's
    multi-node PP branch, however, keeps a bare MultiprocExecutor as its
    backend engine (node_rank > 0 headless executor) and calls
    ``self._backend_engine.start_profile()`` directly, which raises
    AttributeError: 'MultiprocExecutor' object has no attribute
    'start_profile'. Forwarding into the worker-side profile hook keeps the
    collection window semantics identical to the leader node.

    The engine-side requests export is intentionally NOT duplicated here:
    this branch has no Scheduler, so REQUEST_INFOS is empty and an extra
    rank0.requests.jsonl would be misleading.
    """

    HOOK_MODULE_NAME = "vllm.v1.executor.multiproc_executor"
    HOOK_CLASS_NAME = "MultiprocExecutor"

    @classmethod
    def hook(cls, target) -> None:
        # A newer vllm may grow these methods itself; never shadow a real one.
        if getattr(target, "start_profile", None) is not None:
            return

        def _profile_prefix(args, kwargs):
            if "profile_prefix" in kwargs:
                return kwargs["profile_prefix"]
            return args[0] if args else None

        # dashllm calls these with no arguments; stay permissive anyway.
        def start_profile(self, *args, **kwargs):
            try:
                return self.collective_rpc(
                    "profile", args=(True, _profile_prefix(args, kwargs))
                )
            except Exception as e:
                print(
                    f"[sim_collector] MultiprocExecutor.start_profile forward "
                    f"failed: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                return None

        def stop_profile(self, *args, **kwargs):
            try:
                return self.collective_rpc("profile", args=(False, None))
            except Exception as e:
                print(
                    f"[sim_collector] MultiprocExecutor.stop_profile forward "
                    f"failed: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                return None

        target.start_profile = start_profile
        target.stop_profile = stop_profile


def _collect_remote_load_blocks(scheduler_output) -> dict:
    """req_id -> number of KV blocks this step pulls from remote (v6d).

    MultiKVConnectorMetadata.metadata aggregates one metadata object per child
    connector, and the v6d child carries
    ``reqs_to_load = {req_id: {group_id: (keys, gpu_block_ids)}}``. Having
    this inside the scheduler means the engine process can compute the remote
    split itself -- no cross-process hop through the workers.

    The per-request value is the MIN across the groups present in
    reqs_to_load, never the sum: ``num_computed_tokens`` is itself a minimum
    over groups, and a group absent from reqs_to_load needs nothing from
    remote so it does not constrain the length. Summing would double count
    the attention and mamba groups.

    Returns {} when the build exposes no such metadata, so callers keep the
    previous Request-level behaviour.
    """
    out: dict = {}
    meta = getattr(scheduler_output, "kv_connector_metadata", None)
    for child in (getattr(meta, "metadata", None) or ()):
        reqs = getattr(child, "reqs_to_load", None)
        if not reqs:
            continue
        for req_id, groups in reqs.items():
            counts = [
                len(keys) for keys, _block_ids in (groups or {}).values()
            ]
            if counts:
                out[req_id] = min(counts)
    return out


class C_SchedulerHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "vllm.v1.core.sched.scheduler"

    @classmethod
    def hook(cls, target) -> None:

        original_add_request = target.add_request
        original_schedule = target.schedule
        original_free_request = target._free_request

        def wrapped_add_request(self, request, *args, **kwargs):
            recv_time = time.time()
            req_info = REQUEST_INFOS[request.request_id]
            req_info.rid = request.request_id
            req_info.queue_start = _round6(recv_time)
            req_info.server_created_time = _round6(request.arrival_time)
            req_info.created_time = _round6(request.arrival_time)
            return original_add_request(self, request, *args, **kwargs)

        # Signature must stay call-site agnostic: vllm_torchtpu calls
        # scheduler.schedule(throttle_prefills) with one positional argument,
        # and a self-only wrapper raises TypeError inside the engine busy loop
        # (EngineCore fatal -> EngineDeadError -> pod restart).
        def wrapped_schedule(self, *args, **kwargs):
            # D1: timestamp when remote-KV readiness first becomes visible
            # to the scheduler (consumed from finished_recving this round).
            try:
                for _rid in getattr(self, "finished_recving_kv_req_ids", ()):
                    _ri = REQUEST_INFOS.get(_rid)
                    if _ri is not None and _ri.remote_kv_ready_ts == 0.0:
                        _ri.remote_kv_ready_ts = round(time.time(), 6)
            except Exception:
                pass
            scheduler_output = original_schedule(self, *args, **kwargs)

            # Remote-hit accounting for this step. Parsed defensively: the
            # metadata shape is connector/fork dependent and this runs on the
            # engine hot path.
            try:
                remote_load_blocks = _collect_remote_load_blocks(scheduler_output)
            except Exception as e:
                print(
                    f"[sim_collector] remote load meta skipped: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                remote_load_blocks = {}

            if scheduler_output.scheduled_new_reqs:
                prefill_timestamp = time.time()
                for new_req_data in scheduler_output.scheduled_new_reqs:
                    req_id = new_req_data.req_id
                    request = self.requests.get(req_id)
                    if request is None:
                        continue
                    req_info = REQUEST_INFOS[req_id]
                    if req_info.queue_end == 0:
                        req_info.queue_end = _round6(prefill_timestamp)
                        req_info.input_length = request.num_prompt_tokens
                        req_info.output_length = request.max_tokens

                        # Prefix-KV hit lengths. `Request.num_cached_tokens` is
                        # gone in current vllm (AttributeError on the first
                        # scheduled request would kill EngineCore). The hit
                        # count must come from the scheduler snapshot on
                        # NewRequestData: original_schedule() has already run,
                        # so Request.num_computed_tokens now includes this
                        # step's scheduled tokens. Same source as past_kv_len
                        # below, keeping both fields on one definition.
                        hit_len = getattr(new_req_data, "num_computed_tokens", None)
                        if hit_len is None:
                            # Older vllm: dedicated cache-hit field on Request.
                            hit_len = getattr(request, "num_cached_tokens", None)
                        if hit_len is None:
                            hit_len = getattr(request, "num_computed_tokens", 0)
                        ext_hit_len = getattr(
                            request, "num_external_computed_tokens", None
                        )
                        if ext_hit_len is None:
                            ext_hit_len = getattr(
                                new_req_data, "num_external_computed_tokens", 0
                            )
                        # Remote split. Preferred source: the v6d block
                        # accounting collected above -- exact whenever cache
                        # hits are block aligned, which is the only granularity
                        # vllm caches at. Falls back to the Request-level field
                        # for builds that expose it (this fork does not).
                        ext_blocks = remote_load_blocks.get(req_id)
                        block_size = getattr(self, "block_size", None) or getattr(
                            getattr(self, "cache_config", None), "block_size", None
                        )
                        if ext_blocks is not None and block_size:
                            ext_hit_len = min(ext_blocks * int(block_size), hit_len)
                        req_info.total_kv_hit_len = hit_len
                        req_info.ext_kv_hit_len = ext_hit_len
                        req_info.local_kv_hit_len = hit_len - ext_hit_len

            return scheduler_output

        def wrapped_free_request(self, request, *args, **kwargs):
            # Token ids dominate REQUEST_INFOS memory; skip collecting them
            # entirely when they are not exported.
            if EXPORT_TOKEN_IDS and request.is_finished():
                req_info = REQUEST_INFOS[request.request_id]
                req_info.input_ids = (
                    array("i", request.prompt_token_ids)
                    if request.prompt_token_ids is not None
                    else array("i")
                )
                req_info.output_ids = array("i", request.output_token_ids)
            return original_free_request(self, request, *args, **kwargs)

        target.add_request = wrapped_add_request
        target.schedule = wrapped_schedule
        target._free_request = wrapped_free_request


class C_TpuSchedulerHook(C_SchedulerHook):
    """Same scheduler recording, re-applied on torchtpu's scheduler subclass.

    Class hooks match (defining module, class name) exactly, so patching
    vllm.v1.core.sched.scheduler.Scheduler does not reach an override defined
    in a subclass. Verified in-container: TpuPipelineChunkScheduler has
    own add_request=False, own schedule=True, own _free_request=False -- i.e.
    only ``schedule`` needs a second patch, and without it every request row
    keeps input_length / output_length / *_kv_hit_len / queue_end at 0.

    Re-running the base hook body wraps the inherited pair a second time,
    which is harmless: both wrappers only stamp the same timestamps and
    delegate once to the real implementation.
    """

    HOOK_MODULE_NAME = "vllm_torchtpu.core.pp_chunk_scheduler"
    HOOK_CLASS_NAME = "TpuPipelineChunkScheduler"


class C_V6dWorkerHook(BaseHook):
    """Record per-request cross_node at transfer-submit time (worker side).

    async_start_load_kv sees metadata.external_tokens (>0 == remote hit) --
    the only spot the cross-node bit is visible on the worker side. Stash it
    so C_V6dSwapHook can tag each transfer without re-deriving from keys.
    """

    HOOK_MODULE_NAME = "vllm.distributed.kv_transfer.kv_connector.v1.v6d_object_connector"
    HOOK_CLASS_NAME = "V6dObjectConnectorWorker"

    @classmethod
    def hook(cls, target) -> None:
        original_async_start_load_kv = getattr(target, "async_start_load_kv", None)
        if original_async_start_load_kv is None:
            # Some connector builds (e.g. the kvs_connector package shipped on
            # the TPU image) expose only the synchronous submit path. Without
            # the async entry point there is nowhere to read
            # metadata.external_tokens, so cross_node simply stays unknown --
            # do not abort the hook installation.
            print(
                f"[sim_collector] {target.__name__}.async_start_load_kv not "
                f"found; cross_node tagging disabled",
                file=sys.stderr,
                flush=True,
            )
            return

        def wrapped_async_start_load_kv(self, metadata):
            ext = getattr(metadata, "external_tokens", None) or {}
            now = time.time()
            for req_id in (getattr(metadata, "reqs_to_load", None) or {}):
                _REQ_CROSS_NODE[req_id] = int(ext.get(req_id, 0) or 0) > 0
                _REQ_SUBMIT_TS[req_id] = now  # queue-wait anchor (T2/T3)
            return original_async_start_load_kv(self, metadata)

        target.async_start_load_kv = wrapped_async_start_load_kv


class C_V6dSwapHook(BaseHook):
    """Measure per-transfer latency inside V6dSwapHandler.async_swap.

    Start = coroutine entry (next stmt is `await client.async_get`, the real
    transfer kick-off). End = just before return (after the inline
    `while not event.query()` loop, i.e. data truly landed). load/store share
    async_swap; direction via self._swap_in. Signature-agnostic passthrough:
    async_swap kwargs differ across vllm versions (create_batch_id only on
    0827+). Positional order: (job_id, keys, gpu_block_ids, request_id, ...).
    """

    HOOK_MODULE_NAME = "vllm.distributed.kv_transfer.kv_connector.v1.v6d_object_connector"
    HOOK_CLASS_NAME = "V6dSwapHandler"

    @classmethod
    def hook(cls, target) -> None:
        # Both entry points are optional on a given connector build: this TPU
        # image's kvs_connector ships only the synchronous swap(), while the
        # GPU image ships async_swap(). Define both wrappers but install each
        # one only when the original exists -- never synthesize a method the
        # connector does not have, since callers use hasattr() to pick a path.
        original_async_swap = getattr(target, "async_swap", None)

        async def wrapped_async_swap(self, *args, **kwargs):
            job_id = args[0] if len(args) > 0 else kwargs.get("job_id", 0)
            keys = args[1] if len(args) > 1 else kwargs.get("keys") or []
            request_id = (args[3] if len(args) > 3
                          else kwargs.get("request_id", ""))
            start_ts = time.time()  # T4: transfer actually begins here
            submit_ts = _REQ_SUBMIT_TS.pop(request_id, None)
            err = None
            try:
                return await original_async_swap(self, *args, **kwargs)
            except Exception as e:  # record failed transfers too
                err = repr(e)
                raise
            finally:
                end_ts = time.time()
                TRANSFER_INFOS.append(TransferInfos(
                    rid=request_id,
                    direction=("load" if getattr(self, "_swap_in", True)
                               else "store"),
                    latency_s=round(end_ts - start_ts, 6),
                    num_blocks=len(keys) if keys else 0,
                    nbytes=0,
                    cross_node=_REQ_CROSS_NODE.get(request_id),
                    queued_s=(round(start_ts - submit_ts, 6)
                              if submit_ts else 0.0),
                    job_id=job_id,
                    submit_ts=(round(submit_ts, 6) if submit_ts else 0.0),
                    start_ts=round(start_ts, 6),
                    done_ts=round(end_ts, 6),
                    error=err or "",
                ))
                _REQ_CROSS_NODE.pop(request_id, None)

        if original_async_swap is not None:
            target.async_swap = wrapped_async_swap
        else:
            print(
                f"[sim_collector] {target.__name__}.async_swap not found; "
                f"async transfer path not measured",
                file=sys.stderr,
                flush=True,
            )

        # 同步 swap 路径：部分 vllm 版本/场景的 store（如 bypass 子步 store）
        # 走同步 swap() 而非 async_swap()，必须同时包裹才能全覆盖。
        original_swap = getattr(target, "swap", None)

        def wrapped_swap(self, *args, **kwargs):
            job_id = args[0] if len(args) > 0 else kwargs.get("job_id", 0)
            keys = args[1] if len(args) > 1 else kwargs.get("keys") or []
            request_id = (args[3] if len(args) > 3
                          else kwargs.get("request_id", ""))
            start_ts = time.time()
            swap_in = getattr(self, "_swap_in", True)
            submit_ts = (_REQ_SUBMIT_TS.pop(request_id, None) if swap_in
                         else _REQ_STORE_SUBMIT_TS.pop(request_id, None))
            err = None
            try:
                return original_swap(self, *args, **kwargs)
            except Exception as e:
                err = repr(e)
                raise
            finally:
                end_ts = time.time()
                TRANSFER_INFOS.append(TransferInfos(
                    rid=request_id,
                    direction=("load" if swap_in else "store"),
                    latency_s=round(end_ts - start_ts, 6),
                    num_blocks=len(keys) if keys else 0,
                    nbytes=0,
                    cross_node=_REQ_CROSS_NODE.get(request_id),
                    queued_s=(round(start_ts - submit_ts, 6)
                              if submit_ts else 0.0),
                    job_id=job_id,
                    submit_ts=(round(submit_ts, 6) if submit_ts else 0.0),
                    start_ts=round(start_ts, 6),
                    done_ts=round(end_ts, 6),
                    error=err or "",
                ))
                _REQ_CROSS_NODE.pop(request_id, None)

        if original_swap is not None:
            target.swap = wrapped_swap
        else:
            print(
                f"[sim_collector] {target.__name__}.swap not found; "
                f"sync transfer path not measured",
                file=sys.stderr,
                flush=True,
            )


class C_KvsV6dWorkerHook(C_V6dWorkerHook):
    """Submit-time stash for the external kvs_connector package.

    On this image the v6d connector ships as a top-level package
    (kvs_connector/v6d_object_connector.py) instead of vllm's kv_connector
    namespace, so the declaration above never matches.

    Two differences force a dedicated body instead of reusing the parent:
      1. only the synchronous entries exist (start_load_kv / start_store_kv),
         there is no async_start_load_kv;
      2. V6dObjectConnectorMetadata has no ``external_tokens`` field. The
         cross-node bit is therefore derived from reqs_to_load presence: a
         request that has remote blocks to pull is by definition a cross-node
         hit (same meaning as external_tokens > 0). Requests absent from
         reqs_to_load keep cross_node = None (unknown) -- we never default it
         to False, which downstream would read as "definitely local".
    """

    HOOK_MODULE_NAME = "kvs_connector.v6d_object_connector"
    HOOK_CLASS_NAME = "V6dObjectConnectorWorker"

    @classmethod
    def hook(cls, target) -> None:
        original_load = getattr(target, "start_load_kv", None)
        original_store = getattr(target, "start_store_kv", None)

        if original_load is None and original_store is None:
            print(
                f"[sim_collector] {target.__name__}: neither start_load_kv nor "
                f"start_store_kv found; transfer queue timing disabled",
                file=sys.stderr,
                flush=True,
            )
            return

        if original_load is not None:
            def wrapped_start_load_kv(self, metadata, *args, **kwargs):
                now = time.time()
                # reqs_to_load: {req_id: {group_id: (keys, gpu_block_ids)}}
                # Any non-empty key list means this request pulls KV from a
                # remote node -- the same fact external_tokens > 0 encoded.
                for req_id, groups_data in (
                    getattr(metadata, "reqs_to_load", None) or {}
                ).items():
                    _REQ_SUBMIT_TS[req_id] = now  # queue-wait anchor (T2/T3)
                    try:
                        has_remote = any(
                            len(keys) > 0
                            for keys, _ids in (groups_data or {}).values()
                        )
                    except Exception:
                        has_remote = True
                    _REQ_CROSS_NODE[req_id] = bool(has_remote)
                return original_load(self, metadata, *args, **kwargs)

            target.start_load_kv = wrapped_start_load_kv

        # Store side: keeps the store-direction queued_s anchor. Only the
        # request ids are read, so the (groups_data, is_last_save[, batch_id])
        # value shape of reqs_to_store does not matter.
        if original_store is not None:
            def wrapped_start_store_kv(self, metadata, *args, **kwargs):
                now = time.time()
                for req_id in (getattr(metadata, "reqs_to_store", None) or {}):
                    _REQ_STORE_SUBMIT_TS[req_id] = now
                return original_store(self, metadata, *args, **kwargs)

            target.start_store_kv = wrapped_start_store_kv


class C_KvsV6dSwapHook(C_V6dSwapHook):
    """Same swap timing, registered on the external connector package."""

    HOOK_MODULE_NAME = "kvs_connector.v6d_object_connector"
    HOOK_CLASS_NAME = "V6dSwapHandler"

