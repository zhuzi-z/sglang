import contextlib
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
# Suffix of a file that is still being written; removed by the atomic rename
# that publishes the final name.
_INPROGRESS_SUFFIX = ".profiling"


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
    # PP: wall clock when this request's first output token was produced
    # (update_from_output sees output_token_ids go from empty to non-empty).
    # TTFT = first_token_ts - created_time.
    first_token_ts: float = 0.0
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
    # PP wave (worker side, vllm_torchtpu.distributed.pp_wave.PPWave):
    # time rank-0 spent waiting for the previous hand-off to complete, and
    # the forward span its own stage ran afterwards. iter_latency covers
    # both; these two split it, so "pure forward" and "downstream backpressure"
    # can be told apart in one row.
    handoff_wait_s: float = 0.0
    stage0_forward_s: float = 0.0
    # Burst counters at the time this step ended (PPWave.forwards/launches).
    pp_forwards: int = 0
    pp_launches: int = 0
    # Position of this step inside its burst, read at start_forward (i.e. the
    # 0-based PPWave.forwards value, the same quantity called pos_in_burst in
    # the analysis), plus the burst serial (PPWave.bursts). Exported directly
    # so consumers never have to subtract one from pp_forwards.
    pos_in_burst: int = 0
    burst_id: int = 0
    # Wall clock at this stage's start_forward entry (PPWave hook fwd_t0):
    # the closest observable point to "this chunk starts computing on
    # stage 0" -- after the before_step hand-off wait, before the host
    # submits the forward. Paired with the EngineCore-side sampled reply
    # (engine_steps sampled.wall_ts) it yields the full per-chunk span
    # T_step = sampled.wall_ts - stage0_start_ts. Both processes live on
    # the leader host, so the two wall clocks are directly comparable.
    stage0_start_ts: float = 0.0


BATCH_INFOS: list[BatchInfos] = []

REQUEST_INFOS: dict[str, RequestInfos] = defaultdict(RequestInfos)

# --- PP step-level collection ------------------------------------------------
# EngineCore-side records (LazyPushStepper), one or more rows per scheduler
# step. Separate from BATCH_INFOS because it lives in the EngineCore process
# while BATCH_INFOS belongs to each worker process.
ENGINE_STEPS: list[dict] = []
_ENGINE_STEPS_LOCK = threading.Lock()
_ENGINE_STEP_SEQ = {"n": 0}

# Worker-side scratch for the current step: PPWave.before_step writes the
# hand-off wait, start_forward/end_forward write the forward span, and
# _finalize_exec drains it into the BatchInfos row. One worker process owns
# exactly one PPWave, so a module-level dict is enough.
_PP_STEP_STATE: dict = {}


def _record_engine_step(kind: str, stepper, slot=None,
                        forward_s: Optional[float] = None,
                        rids: Optional[list] = None) -> None:
    """Append one EngineCore-side step record (defensive: never raises).

    ``rids`` (dispatch rows only): full request ids from this step's
    SchedulerOutput.num_scheduled_tokens -- the same object the workers
    parse into BatchInfos.requests (which keeps rid[-12:]), so the rid set
    joins a dispatch row to the worker-side schedule_batch row of the same
    step. sampled/process rows carry no rids; they join to the dispatch
    row through slot_dispatched instead.
    """
    with _ENGINE_STEPS_LOCK:
        _ENGINE_STEP_SEQ["n"] += 1
        wave = getattr(stepper, "wave", None)
        row = {
            "seq": _ENGINE_STEP_SEQ["n"],
            "kind": kind,
            "wall_ts": round(time.time(), 6),
            "mono_ts": round(time.monotonic(), 6),
            "in_flight": len(getattr(stepper, "in_flight", None) or ()),
            "slots": len(getattr(stepper, "slots", None) or ()),
            "dropped": int(getattr(stepper, "dropped", 0) or 0),
            "measured_s": round(float(getattr(stepper, "measured", 0.0) or 0.0), 6),
            "ratio": round(float(getattr(stepper, "ratio", 1.0) or 1.0), 4),
            "pushes": int(getattr(wave, "pushes", 0) or 0),
            "settles": int(getattr(wave, "settles", 0) or 0),
            "real_steps": int(getattr(wave, "real_steps", 0) or 0),
            "pp_forwards": int(getattr(wave, "forwards", 0) or 0),
            "pp_bursts": int(getattr(wave, "bursts", 0) or 0),
        }
        if slot is not None:
            dispatched = getattr(slot, "dispatched", None)
            expected = getattr(slot, "expected", None)
            started = getattr(slot, "started_last", None)
            row["slot_dispatched"] = (
                round(float(dispatched), 6) if dispatched is not None else None
            )
            row["slot_expected"] = (
                round(float(expected), 6) if expected is not None else None
            )
            row["slot_modelled_ms"] = getattr(slot, "modelled_ms", None)
            row["slot_started_last"] = (
                round(float(started), 6) if started is not None else None
            )
        # Queue / busy snapshot: the training set needs the same quantities the
        # simulator can compute at inference time (waiting/running depth), so the
        # two sides stay on one definition. Read straight off the Scheduler that
        # lives in this process; missing attributes leave the keys out.
        scheduler = getattr(getattr(stepper, "core", None), "scheduler", None)
        if scheduler is not None:
            for key, attr in (
                ("num_waiting", "waiting"),
                ("num_running", "running"),
                ("num_skipped", "skipped_waiting"),
            ):
                queue = getattr(scheduler, attr, None)
                if queue is not None:
                    try:
                        row[key] = len(queue)
                    except TypeError:
                        pass
        if forward_s is not None:
            row["forward_s"] = round(float(forward_s), 6)
        if rids is not None:
            row["rids"] = rids
        ENGINE_STEPS.append(row)


def _warn_once(where: str, exc: BaseException) -> None:
    print(f"[sim_collector] {where} skipped: {exc}", file=sys.stderr, flush=True)


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


@contextlib.contextmanager
def _open_export(path: str):
    """Write into ``<path>.profiling`` and rename to ``path`` once complete.

    Every export goes through this context manager, so a reader globbing
    ``*.jsonl.gz`` never sees a file that is still being written: the final
    name only appears after the file is closed (gzip trailer included) and
    atomically renamed in the same directory. A leftover ``.profiling`` file
    therefore means "this export did not finish" (write error or process
    killed), which is exactly the signal the start-side async dump lacks.
    """
    tmp_path = path + _INPROGRESS_SUFFIX
    if EXPORT_GZIP:
        f = gzip.open(tmp_path, "wt", compresslevel=1)
    else:
        f = open(tmp_path, "w")
    try:
        yield f
    finally:
        f.close()
    # Only reached when the with-body and close() both succeeded; a raise
    # above leaves the .profiling file behind and never publishes the final
    # name.
    os.replace(tmp_path, path)


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


# Per-call export subdirectory ---------------------------------------------
# EngineCore.profile and every Worker.profile fire exactly once per profile
# call (/start_profile and /stop_profile both fan out via collective_rpc to
# all workers), so one call dumps one run directory shared by the EngineCore
# and every worker of the node:
#
#   $SIM_COLLECTOR_OUTPUT_DIR/engine{RANK}/run{NNN}/
#     rank0.requests.jsonl[.gz]         EngineCore
#     rank0.engine_steps.jsonl[.gz]     EngineCore
#     rank{N}.schedule_batch.jsonl[.gz] each worker
#     rank{N}.transfer.jsonl[.gz]       each worker
#
# The rank in the file name keeps the processes apart inside the shared
# directory. Every file is written as <name>.profiling and renamed to <name>
# only once it is complete, so a partial (or still running) dump is never
# mistaken for a finished one. The call number lives in a per-node marker
# (.current_run) so a worker joins the directory the EngineCore minted for
# this call instead of minting one of its own -- minting per process was what
# split one call's files across run001 (requests) and run002 (engine_steps).
# A worker that runs without a marker (node without an EngineCore) allocates
# it itself and the remaining workers join.
_RUN_MARKER = ".current_run"


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


def _run_dir_name(n: int) -> str:
    return f"run{n:03d}"


def _run_number(name: str) -> int:
    """Number of a ``runNNN`` directory; 0 for anything else (e.g. _old)."""
    if len(name) != 6 or not name.startswith("run"):
        return 0
    digits = name[3:]
    return int(digits) if digits.isdigit() else 0


def _output_root() -> str:
    base = os.getenv("SIM_COLLECTOR_OUTPUT_DIR", os.getcwd())
    engine_rank = os.getenv("DS_LLM_PROC_RANK")
    if engine_rank is not None:
        sub_dir = f"engine{engine_rank}"
    else:
        sub_dir = f"pid{os.getpid()}"
    return os.path.join(base, sub_dir)


def _read_run_marker(root: str) -> int:
    try:
        with open(os.path.join(root, _RUN_MARKER)) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 0


def _write_run_marker(root: str, n: int) -> None:
    path = os.path.join(root, _RUN_MARKER)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(f"{n}\n")
    os.replace(tmp_path, path)


def _highest_run(root: str) -> int:
    try:
        names = os.listdir(root)
    except OSError:
        return 0
    return max((_run_number(name) for name in names), default=0)


def _archive_run_dir(path: str) -> None:
    """Move a colliding run dir aside; the caller keeps the plain runNNN."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = f"{path}_old_{stamp}"
    n = 1
    while os.path.exists(target):
        target = f"{path}_old_{stamp}_{n}"
        n += 1
    os.rename(path, target)
    print(
        f"[sim_collector] {path} already exists, archived to {target}",
        file=sys.stderr,
        flush=True,
    )


def get_output_dir(join: bool = False) -> str:
    """Directory of the current profile call: engine{RANK}/run{NNN}/.

    The collector records continuously, so every profile call --
    /start_profile and /stop_profile alike -- dumps what accumulated since
    the previous call into its own run directory; nothing is overwritten.

    ``join=False`` (EngineCore, the allocating side on a node that has one):
    take the next free number and record it in the node's .current_run
    marker. The number is max(marker, existing runNNN) + 1, so a restart
    continues the sequence instead of reusing an existing directory; should
    the name still be taken (concurrent allocator, stale marker), the
    existing directory is archived as runNNN_old_<ts> rather than reused.

    ``join=True`` (workers): adopt the number the EngineCore wrote for this
    call -- the fan-out happens after the marker write, so it is the current
    call -- and allocate instead when no marker exists. All processes of one
    call therefore land in one directory, separated by the rank in the file
    names.
    """
    root = _output_root()
    os.makedirs(root, exist_ok=True)

    if join:
        n = _read_run_marker(root)
        if n > 0:
            path = os.path.join(root, _run_dir_name(n))
            os.makedirs(path, exist_ok=True)
            return path

    n = max(_read_run_marker(root), _highest_run(root)) + 1
    path = os.path.join(root, _run_dir_name(n))
    if os.path.exists(path):
        _archive_run_dir(path)
    os.makedirs(path, exist_ok=True)
    _write_run_marker(root, n)
    return path


def _export_or_warn(where: str, export_fn) -> None:
    """Run one export; a collection failure must never break the profile
    call (a raised wrapper would skip original_profile and leave the
    profiler running)."""
    try:
        export_fn()
    except Exception as e:
        _warn_once(where, e)


def _snapshot_engine_requests():
    """Swap the request buffer out; returns this call's slice."""
    global REQUEST_INFOS
    request_infos = REQUEST_INFOS
    REQUEST_INFOS = defaultdict(RequestInfos)
    return request_infos


def _write_engine_requests(output_dir: str, request_infos) -> None:
    """Write one cut of REQUEST_INFOS into this call's directory.

    Kept as a module-level function so a second caller (e.g. an executor-level
    profile shim) can reuse it without duplicating the JSON shaping.
    """
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


def _snapshot_engine_steps():
    """Swap the PP step buffer out; returns this call's slice."""
    global ENGINE_STEPS
    with _ENGINE_STEPS_LOCK:
        steps = ENGINE_STEPS
        ENGINE_STEPS = []
    return steps


def _write_engine_steps(output_dir: str, steps) -> None:
    """Write one cut of the EngineCore-side PP step records.

    Written by the LazyPushStepper hook (leader node only: the lazy-push step
    loop exists in the EngineCore process, not in a follower). One row per
    recorded event; ``kind`` says whether it marks a dispatch, a sampling
    reply (the only place a real per-step forward time is observable) or the
    processing of a finished step.
    """
    path = f"{output_dir}/rank0.engine_steps.jsonl"
    if EXPORT_GZIP:
        path += ".gz"
    with _open_export(path) as f:
        for row in steps:
            f.write(json.dumps(row, cls=CustomJsonEncoder) + "\n")
    print(f"Engine step data has been saved to {path}")


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
                    # The SchedulerOutput parse failed (batch_req_infos was
                    # cleared), so this step has no row to carry the PP split.
                    # Drop the scratch anyway: leaving it behind would attach
                    # this step's hand-off wait / forward span to the NEXT row.
                    _PP_STEP_STATE.clear()
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
                # PP wave split: what PPWave.before_step waited for the
                # previous hand-off, and what this stage's own forward took.
                # Missing keys simply mean the PPWave hook did not run on this
                # build (GPU path), leaving the zeros in place.
                try:
                    handoff_wait = _PP_STEP_STATE.pop("handoff_wait_s", None)
                    if handoff_wait is not None:
                        batch_info.handoff_wait_s = _round6(handoff_wait)
                    forward_s = _PP_STEP_STATE.pop("stage0_forward_s", None)
                    if forward_s is not None:
                        batch_info.stage0_forward_s = _round6(forward_s)
                    batch_info.pp_forwards = int(
                        _PP_STEP_STATE.pop("pp_forwards", 0) or 0
                    )
                    batch_info.pp_launches = int(
                        _PP_STEP_STATE.pop("pp_launches", 0) or 0
                    )
                    batch_info.pos_in_burst = int(
                        _PP_STEP_STATE.pop("pos_in_burst", 0) or 0
                    )
                    batch_info.burst_id = int(
                        _PP_STEP_STATE.pop("burst_id", 0) or 0
                    )
                    fwd_t0 = _PP_STEP_STATE.pop("fwd_t0", None)
                    if fwd_t0 is not None:
                        batch_info.stage0_start_ts = _round6(fwd_t0)
                except Exception as e:
                    _warn_once("pp step state", e)
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

            output_dir = get_output_dir(join=True)
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
            # Cut point first, write last: snapshotting both buffers before
            # the fan-out keeps this node's requests / engine_steps on the
            # same slice of traffic as the workers' schedule_batch / transfer
            # (the fan-out alone takes seconds), and every record that lands
            # while the files are being written goes to the next call.
            # A collection failure is a warning, never a broken profile call:
            # raising here would skip original_profile and leave the profiler
            # running.
            output_dir = snapshot = None
            try:
                output_dir = get_output_dir()
                snapshot = (_snapshot_engine_requests(), _snapshot_engine_steps())
            except Exception as e:
                _warn_once("engine snapshot", e)

            try:
                # Call the original profile method to trigger the lower-level
                # worker fan-out (and the real profiler, when configured).
                return original_profile(self, is_start, profile_prefix)
            finally:
                if snapshot is not None:
                    requests_cut, steps_cut = snapshot

                    def _export():
                        _export_or_warn(
                            "request export",
                            lambda: _write_engine_requests(output_dir, requests_cut),
                        )
                        _export_or_warn(
                            "engine step export",
                            lambda: _write_engine_steps(output_dir, steps_cut),
                        )

                    # start: side thread (the engine keeps serving);
                    # stop: synchronous, files are on disk before returning.
                    _run_export(_export, is_start)

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


class C_PPWaveHook(BaseHook):
    """Split iter_latency into "waited for the previous hand-off" and
    "this stage's own forward" on a TPU pipeline.

    PPWave.before_step() is where rank 0 blocks on the previous hand-off
    (_async_wait + _wait), so its wall span is the downstream backpressure
    that the worker-side iter_latency otherwise hides. start_forward /
    end_forward bracket the forward this stage runs afterwards. The three
    land in BatchInfos (schedule_batch.jsonl) through _finalize_exec.

    Missing methods leave the row's zeros in place, so the GPU path is
    unaffected.
    """

    HOOK_MODULE_NAME = "vllm_torchtpu.distributed.pp_wave"
    HOOK_CLASS_NAME = "PPWave"

    @classmethod
    def hook(cls, target) -> None:
        original_before_step = getattr(target, "before_step", None)
        original_start_forward = getattr(target, "start_forward", None)
        original_end_forward = getattr(target, "end_forward", None)

        if original_before_step is not None:

            def wrapped_before_step(self, num_tokens, *args, **kwargs):
                t0 = time.time()
                ret = original_before_step(self, num_tokens, *args, **kwargs)
                _PP_STEP_STATE["handoff_wait_s"] = time.time() - t0
                return ret

            target.before_step = wrapped_before_step

        if original_start_forward is not None:

            def wrapped_start_forward(self, rows, *args, **kwargs):
                _PP_STEP_STATE["fwd_t0"] = time.time()
                ret = original_start_forward(self, rows, *args, **kwargs)
                # _open_burst() runs inside the original call, so by now
                # `forwards` is this forward's 0-based position in the burst
                # and `bursts` is the burst serial.
                _PP_STEP_STATE["pos_in_burst"] = int(
                    getattr(self, "forwards", 0) or 0
                )
                _PP_STEP_STATE["burst_id"] = int(getattr(self, "bursts", 0) or 0)
                return ret

            target.start_forward = wrapped_start_forward

        if original_end_forward is not None:

            def wrapped_end_forward(self, tensors, rows, *args, **kwargs):
                ret = original_end_forward(self, tensors, rows, *args, **kwargs)
                t0 = _PP_STEP_STATE.pop("fwd_t0", None)
                if t0 is not None:
                    _PP_STEP_STATE["stage0_forward_s"] = time.time() - t0
                _PP_STEP_STATE["pp_forwards"] = int(
                    getattr(self, "forwards", 0) or 0
                )
                _PP_STEP_STATE["pp_launches"] = int(
                    getattr(self, "launches", 0) or 0
                )
                return ret

            target.end_forward = wrapped_end_forward


class C_LazyPushStepperHook(BaseHook):
    """Record one row per scheduler step of the lazy-push loop.

    LazyPushStepper replaces EngineCore.step when TPU_PP_LAZY_PUSH is set and
    never blocks on a result, so EngineCore.step's wall clock is meaningless
    on this path.

    ``forward_s`` on a ``kind="sampled"`` row spans from ``slot.started_last``
    to the sampling callback. ``started_last`` is stamped when the
    ``execute_model`` future completes -- i.e. when **stage 0** accepted the
    step (pp_lazy_step.py:293) -- and the sampling callback fires when the
    last stage's ``sample_tokens`` returns (pp_lazy_step.py:294). The value is
    therefore "everything after stage 0": the activations travelling through
    stages 1..N-1 plus the last stage's forward and sampling.

    It is NOT the last stage's own forward time, and stages 1..N-2 cannot be
    separated from it: they run in follower processes that carry no
    EngineCore, so there is nothing to hook there. Pairing it with the
    worker-side ``stage0_forward_s`` gives the whole step end to end.

    Rows are exported to rank0.engine_steps.jsonl by the EngineCore profile
    hook. The class only exists on a leader of a pipeline deeper than two
    stages, so the GPU path never installs it.
    """

    HOOK_MODULE_NAME = "vllm_torchtpu.core.pp_lazy_step"
    HOOK_CLASS_NAME = "LazyPushStepper"

    @classmethod
    def hook(cls, target) -> None:
        original_dispatch = getattr(target, "_dispatch", None)
        original_sampled = getattr(target, "_sampled", None)
        original_process = getattr(target, "_process", None)

        if original_dispatch is not None:

            def wrapped_dispatch(self, *args, **kwargs):
                ret = original_dispatch(self, *args, **kwargs)
                if ret:
                    try:
                        slot = self.slots[-1] if self.slots else None
                        # The just-dispatched _Step carries the same
                        # SchedulerOutput the workers receive; its
                        # num_scheduled_tokens keys are this step's rids.
                        # Empty for state-only steps without tokens.
                        rids = None
                        step = self.in_flight[-1] if self.in_flight else None
                        sched_out = getattr(step, "scheduler_output", None)
                        nst = getattr(sched_out, "num_scheduled_tokens", None)
                        if nst:
                            rids = list(nst.keys())
                        _record_engine_step("dispatch", self, slot=slot, rids=rids)
                    except Exception as e:
                        _warn_once("engine step dispatch", e)
                return ret

            target._dispatch = wrapped_dispatch

        if original_sampled is not None:

            def wrapped_sampled(self, slot, *args, **kwargs):
                started = getattr(slot, "started_last", None)
                forward_s = (
                    time.monotonic() - started if started is not None else None
                )
                ret = original_sampled(self, slot, *args, **kwargs)
                try:
                    _record_engine_step(
                        "sampled", self, slot=slot, forward_s=forward_s
                    )
                except Exception as e:
                    _warn_once("engine step sampled", e)
                return ret

            target._sampled = wrapped_sampled

        if original_process is not None:

            def wrapped_process(self, step, *args, **kwargs):
                ret = original_process(self, step, *args, **kwargs)
                try:
                    _record_engine_step(
                        "process", self, slot=getattr(step, "slot", None)
                    )
                except Exception as e:
                    _warn_once("engine step process", e)
                return ret

            target._process = wrapped_process


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
        original_update_from_output = getattr(target, "update_from_output", None)

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

        # PP: the first token of a request is only visible here -- the worker
        # that samples it is the last stage, and this EngineCore is the first
        # place the token meets the request object. update_from_output runs in
        # the same process as the Scheduler, so no cross-process join is needed
        # to turn `first_token_ts - created_time` into TTFT.
        def wrapped_update_from_output(self, scheduler_output,
                                       model_runner_output, *args, **kwargs):
            result = original_update_from_output(
                self, scheduler_output, model_runner_output, *args, **kwargs
            )
            try:
                ts = time.time()
                for _client_index, core_outputs in (result or {}).items():
                    for out in (getattr(core_outputs, "outputs", None) or ()):
                        rid = getattr(out, "request_id", None)
                        if not rid:
                            continue
                        new_ids = getattr(out, "new_token_ids", None)
                        if not new_ids:
                            continue
                        req_info = REQUEST_INFOS.get(rid)
                        if req_info is not None and req_info.first_token_ts == 0.0:
                            req_info.first_token_ts = _round6(ts)
            except Exception as e:
                _warn_once("first_token_ts", e)
            return result

        target.add_request = wrapped_add_request
        target.schedule = wrapped_schedule
        target._free_request = wrapped_free_request
        if original_update_from_output is not None:
            target.update_from_output = wrapped_update_from_output


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
    """Same swap timing, registered on the external connector package.

    Name matters here: on this build the real class is
    ``TorchTPUV6dSwapHandler`` and the module only binds
    ``V6dSwapHandler = TorchTPUV6dSwapHandler`` (a plain alias, ~line 3973).
    Class hooks match the name handed to ``__build_class__`` -- i.e. the real
    class name -- so declaring the alias meant the hook silently never
    installed and transfer.jsonl stayed empty. Match both spellings so the GPU
    build (whose class really is V6dSwapHandler) keeps working too.
    """

    REGEX = True
    # REGEX mode re.search()es the module name too, so anchor and escape it.
    HOOK_MODULE_NAME = r"^kvs_connector\.v6d_object_connector$"
    HOOK_CLASS_NAME = r"^(?:V6dSwapHandler|TorchTPUV6dSwapHandler)$"

