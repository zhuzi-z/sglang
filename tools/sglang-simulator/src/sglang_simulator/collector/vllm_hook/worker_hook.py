import gzip
import inspect
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
from contextlib import contextmanager
from functools import wraps

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
# DP>1 always collects the complete DP batch. All workers must use the same
# capacity; numeric tuples ride the existing all-reduce without another collective.
DP_REQUEST_CAPACITY = int(os.getenv("SIM_COLLECTOR_DP_MAX_REQUESTS", "256"))
if DP_REQUEST_CAPACITY < 1:
    raise ValueError("SIM_COLLECTOR_DP_MAX_REQUESTS must be positive")
_DP_COORD_CONTEXT = threading.local()


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
    # Numeric tuples (extend_input_len, past_kv_len, output_len), without IDs.
    requests: list[tuple[int, int, int]] = field(default_factory=list)
    # requests is DP-wide when requests_scope == "dp"; latency stays local.
    requests_local: list[tuple[int, int, int]] = field(default_factory=list)
    requests_by_dp: list[list[tuple[int, int, int]]] = field(default_factory=list)
    requests_scope: str = "local"
    dp_rank: int = 0
    dp_size: int = 1
    worker_rank: int = 0
    dp_step_id: Optional[int] = None
    local_forward_mode: int = 0
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


def _collect_model_runner_requests(runner, scheduler_output, num_scheduled_tokens):
    """Read actual local batch rows after ModelRunner's state/input preparation."""
    if not scheduler_output.num_scheduled_tokens:
        return []  # Dummy rows must not reuse a previous real input_batch.
    batch = runner.input_batch
    if len(batch.req_ids) != len(num_scheduled_tokens):
        raise RuntimeError("Collector ModelRunner request/token rows do not align")
    return [
        (int(extend), int(batch.num_computed_tokens_cpu[i]),
         len(runner.requests[rid].output_token_ids))
        for i, (rid, extend) in enumerate(zip(batch.req_ids, num_scheduled_tokens))
    ]


def _apply_dp_metadata(metadata, batch_info):
    """Decode ScheduledDPMetaData's collector payload after native synchronization."""
    payload = getattr(metadata, "_collector_request_payload", None)
    if payload is None:
        if batch_info.dp_size > 1:
            raise RuntimeError("Missing DP request payload; refusing to export an incomplete batch")
        return  # DP1: the local tuples already describe the complete batch.
    step_id = metadata._collector_dp_step_id
    counts, steps = payload[0].tolist(), payload[1].tolist()
    if any(step != step_id for step in steps):
        raise RuntimeError("Collector DP step counters are not aligned")
    if any(count < 0 or count > DP_REQUEST_CAPACITY for count in counts):
        raise RuntimeError("Collector DP request count exceeds its payload capacity")
    by_dp = [
        [tuple(req) for req in payload[2:2 + 3 * count, rank].reshape(count, 3).tolist()]
        for rank, count in enumerate(counts)
    ]
    if by_dp[batch_info.dp_rank] != metadata._collector_local_requests:
        raise RuntimeError("Collector DP payload does not match the local batch")
    batch_info.requests_by_dp = by_dp
    batch_info.requests = [req for requests in by_dp for req in requests]
    batch_info.requests_scope = "dp"
    batch_info.dp_step_id = step_id
    batch_info.forward_mode = 2 if all(req[0] == 1 for req in batch_info.requests) else 1


def _install_dp_coordination_hook(dp_utils):
    """Extend the native 10-row metadata all-reduce, keeping its return schema."""
    original = getattr(dp_utils, "_run_ar", None)
    if original is None:
        raise RuntimeError("Collector DP merging requires vLLM dp_utils._run_ar")
    if getattr(original, "_collector_dp_merge", False):
        return
    signature = inspect.signature(original)
    native_fields = (
        "orig_num_tokens_per_ubatch",
        "padded_num_tokens_per_ubatch",
        "should_ubatch",
        "should_dp_pad",
        "cudagraph_mode",
        "num_reqs",
        "num_context_tokens",
        "is_prompt_batch_local",
        "is_prefill_only_local",
        "input_fits_in_drafter",
    )
    if set(signature.parameters) != set(native_fields) | {
        "parallel_config",
        "device",
        "group",
    }:
        raise RuntimeError(
            "Unsupported vLLM DP _run_ar signature for collector merging"
        )

    @wraps(original)
    def run_ar(*args, **kwargs):
        context = getattr(_DP_COORD_CONTEXT, "current", None)
        # Context lasts only for the target ModelRunner's coordination call.
        if context is None or context.get("host_payload") is not None:
            return original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        parallel = values["parallel_config"]
        if (parallel.data_parallel_size, parallel.data_parallel_rank) != (
            context["dp_size"], context["dp_rank"]
        ):
            raise RuntimeError("Collector DP coordination topology mismatch")
        rank, requests = context["dp_rank"], context["requests"]
        tensor = torch.zeros(12 + 3 * DP_REQUEST_CAPACITY, context["dp_size"],
                             device=values["device"], dtype=torch.int32)
        for row, name in enumerate(native_fields):
            tensor[row, rank] = int(values[name])
        tensor[10, rank] = len(requests)
        tensor[11, rank] = context["step_id"]
        if requests:
            packed = torch.tensor(requests, device=tensor.device, dtype=tensor.dtype)
            tensor[12:12 + packed.numel(), rank] = packed.reshape(-1)
        # Same group, operation and collective count as the unmodified _run_ar.
        torch.distributed.all_reduce(tensor, group=values["group"])
        payload = tensor[10:]
        if tensor.device.type == "cuda":
            host = torch.empty(
                payload.shape, dtype=payload.dtype, device="cpu", pin_memory=True
            )
            # Stream-ordered D2H, no forced synchronization. The forward end
            # event covers this copy; decoding happens at native get_output().
            host.copy_(payload, non_blocking=True)
            context["host_payload"] = host
        else:
            context["host_payload"] = payload
        return tensor[:10]

    run_ar._collector_dp_merge = True
    dp_utils._run_ar = run_ar


@contextmanager
def _model_runner_forward_scope(runner):
    """Transient ModelRunner call state; durable metadata travels via outputs."""
    previous = getattr(runner, "_collector_forward_scope", None)
    scope = dict(coordinated=False, metadata=None)
    runner._collector_forward_scope = scope
    try:
        yield scope
    finally:
        runner._collector_forward_scope = previous


def _attach_output_metadata(output, metadata):
    """Pass batch metadata through normal and asynchronous native outputs."""
    if output is None or metadata is None:
        return output
    if getattr(output, "_collector_batch_metadata", None) is metadata:
        return output
    setattr(output, "_collector_batch_metadata", metadata)
    original_get_output = getattr(output, "get_output", None)
    if original_get_output is not None:
        def get_output(*args, **kwargs):
            result = original_get_output(*args, **kwargs)
            setattr(result, "_collector_batch_metadata", metadata)
            return result
        output.get_output = get_output
    return output


class M_DPBatchCoordinationHook(BaseHook):
    HOOK_MODULE_NAME = "vllm.v1.worker.dp_utils"

    @classmethod
    def hook(cls, target):
        _install_dp_coordination_hook(target)


class C_ModelRunnerBatchHook(BaseHook):
    """Collect DP batches in ModelRunner and propagate them through its outputs."""

    HOOK_MODULE_NAME = "vllm.v1.worker.gpu_model_runner"
    HOOK_CLASS_NAME = "GPUModelRunner"

    @classmethod
    def hook(cls, target):
        # Class-hook dispatch applies only one matching hook; retain RPC-2
        # segment collection as part of this ModelRunner hook.
        C_ModelRunnerSegHook.hook(target)
        original_determine = target._determine_batch_execution_and_padding
        original_execute = target.execute_model
        original_sample = getattr(target, "sample_tokens", None)
        original_dummy = getattr(target, "_dummy_run", None)
        signature = inspect.signature(original_determine)

        @wraps(original_determine)
        def determine(self, *args, **kwargs):
            scope = getattr(self, "_collector_forward_scope", None)
            if scope is None or scope["coordinated"]:
                return original_determine(self, *args, **kwargs)
            scope["coordinated"] = True
            values = signature.bind(self, *args, **kwargs).arguments
            output = values["scheduler_output"]
            requests = _collect_model_runner_requests(
                self, output, values["num_scheduled_tokens_np"])
            parallel = self.parallel_config
            context = None
            if parallel.data_parallel_size > 1:
                if output.scheduled_dp_metadata is not None:
                    raise RuntimeError("Collector requires ModelRunner DP coordination, not precomputed EngineCore metadata")
                if len(requests) > DP_REQUEST_CAPACITY:
                    raise ValueError("Batch exceeds SIM_COLLECTOR_DP_MAX_REQUESTS; set the same capacity on every DP worker")
                if any(value < 0 or value > 2**31 - 1 for req in requests for value in req):
                    raise ValueError("Collector DP request lengths must fit nonnegative int32")
                step = getattr(self, "_collector_dp_step_id", 0)
                self._collector_dp_step_id = step + 1
                context = dict(requests=requests, dp_rank=parallel.data_parallel_rank,
                               dp_size=parallel.data_parallel_size, step_id=step)
            previous = getattr(_DP_COORD_CONTEXT, "current", None)
            _DP_COORD_CONTEXT.current = context
            try:
                result = original_determine(self, *args, **kwargs)
            finally:
                _DP_COORD_CONTEXT.current = previous
            metadata = output.scheduled_dp_metadata
            if metadata is None:
                raise RuntimeError("ModelRunner did not produce ScheduledDPMetaData")
            setattr(metadata, "_collector_local_requests", requests)
            if context is not None:
                if context.get("host_payload") is None:
                    raise RuntimeError("No native DP _run_ar observed; coordination backend unsupported")
                setattr(metadata, "_collector_request_payload", context["host_payload"])
                setattr(metadata, "_collector_dp_step_id", context["step_id"])
            scope["metadata"] = metadata
            return result

        @wraps(original_execute)
        def execute_model(self, scheduler_output, *args, **kwargs):
            with _model_runner_forward_scope(self) as scope:
                result = original_execute(self, scheduler_output, *args, **kwargs)
            metadata = scope["metadata"]
            if scheduler_output.num_scheduled_tokens and metadata is None:
                raise RuntimeError("Target forward did not reach ModelRunner's batch/DP boundary")
            if result is None:
                # Two-RPC vLLM already stores this metadata in ExecuteModelState.
                if metadata is not None and scheduler_output.num_scheduled_tokens:
                    state = getattr(self, "execute_model_state", None)
                    if getattr(state, "scheduled_dp_metadata", None) is not metadata:
                        raise RuntimeError("Native ExecuteModelState did not retain batch metadata")
                return result
            if metadata is not None and metadata._collector_local_requests:
                return _attach_output_metadata(result, metadata)
            return result

        target._determine_batch_execution_and_padding = determine
        target.execute_model = execute_model

        if original_sample is not None:
            @wraps(original_sample)
            def sample_tokens(self, *args, **kwargs):
                state = getattr(self, "execute_model_state", None)
                metadata = getattr(state, "scheduled_dp_metadata", None)
                result = original_sample(self, *args, **kwargs)
                if metadata is not None and hasattr(metadata, "_collector_local_requests"):
                    return _attach_output_metadata(result, metadata)
                return result
            target.sample_tokens = sample_tokens

        if original_dummy is not None:
            dummy_signature = inspect.signature(original_dummy)
            @wraps(original_dummy)
            def dummy_run(self, *args, **kwargs):
                if getattr(self, "_collector_forward_scope", None) is not None:
                    return original_dummy(self, *args, **kwargs)
                values = dummy_signature.bind(self, *args, **kwargs)
                values.apply_defaults()
                values = values.arguments
                # Worker.execute_dummy_batch uses elastic_mask=True. Warmup and
                # graph capture must not contribute synthetic training requests.
                if (not values.get("elastic_mask", False) or values.get("is_profile", False)
                        or values.get("is_graph_capturing", False)):
                    return original_dummy(self, *args, **kwargs)
                with _model_runner_forward_scope(self):
                    return original_dummy(self, *args, **kwargs)
            target._dummy_run = dummy_run


# --- CUDA-event based timing -------------------------------------------------
# The collector must NOT call torch.cuda.synchronize(): a forced device sync
# serializes the pipeline, wrecks async-scheduling overlap and skews the very
# latencies we try to measure. Instead we bracket each measured region with a
# pair of timing cuda events recorded on the main stream (mirroring the
# async_event_pool pattern in vllm's GPUModelRunner). The elapsed time is read
# lazily from AsyncModelRunnerOutput.get_output() -- i.e. right after vllm's own
# synchronize has already drained the stream past our end event. DP merging
# extends the native collective and uses stream-ordered D2H, without extra sync.


class _EventTimer:
    """A start/end pair of timing cuda events for one measured region."""

    __slots__ = ("_start", "_end")

    def __init__(self) -> None:
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def record_start(self) -> None:
        self._start.record()

    def record_end(self) -> None:
        self._end.record()

    def elapsed_s(self) -> Optional[float]:
        """Elapsed seconds between the two events, or None if not ready yet.

        Event.elapsed_time() returns milliseconds and raises if either event
        has not completed. It is only safe to call after a native synchronize
        has passed self._end; we still guard it so a rare not-ready race can
        never crash the serving engine.
        """
        try:
            return self._start.elapsed_time(self._end) / 1000.0
        except RuntimeError:
            return None


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
        finalize(output)
        return output

    def wrapped_get_output(*args, **kwargs):
        result = original_get_output(*args, **kwargs)
        finalize(result)
        return result

    output.get_output = wrapped_get_output
    return output


def _print_batch_info(batch_info: "BatchInfos") -> None:
    """Print one recorded batch at its finalize point (no request ids).

    Numeric request tuples match the JSONL export: (extend, past, output).

    Writes directly to stdout's underlying buffer so that vLLM's
    decorate_logs prefix (e.g. [2/4,TP2][pid=...]) is bypassed.
    """
    row = asdict(batch_info)
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

        def wrapped_execute_model(self, scheduler_output: "SchedulerOutput"):

            # Connector-only calls do not produce a target-model batch.
            if not scheduler_output.num_scheduled_tokens:
                return original_execute_model(self, scheduler_output)
            parallel = self.vllm_config.parallel_config

            # Per-step logprobs form (for independent logprobs-compensation
            # training). Only new reqs carry sampling_params in SchedulerOutput;
            # cached reqs (decode steps) would need engine-side hook or a
            # differential stress test to capture. total_num_scheduled_tokens
            # is the step-level token count used as the MTP-compensation feature.
            logprobs_req_count = 0
            logprobs_tokens = 0
            logprobs_n = 0
            for req in scheduler_output.scheduled_new_reqs:
                sp = getattr(req, "sampling_params", None)
                if sp is not None and getattr(sp, "logprobs", None):
                    logprobs_req_count += 1
                    logprobs_tokens += scheduler_output.num_scheduled_tokens.get(
                        req.req_id, 0
                    )
                    logprobs_n = max(logprobs_n, sp.logprobs)

            global _PENDING_EXEC

            # Bracket the forward with timing cuda events instead of syncing.
            exec_timer = _EventTimer()
            start_wall = time.time()
            exec_timer.record_start()
            ret = original_execute_model(self, scheduler_output)
            exec_timer.record_end()

            def _finalize_exec(model_output, sample_latency: Optional[float] = None,
                               seg: Optional[dict] = None):
                metadata = getattr(model_output, "_collector_batch_metadata", None)
                if metadata is None:
                    raise RuntimeError("ModelRunner output has no collector batch metadata; check class-hook registration")
                # The collector consumes this private carrier before normal RPC
                # serialization; do not send its tensor payload to the scheduler.
                delattr(model_output, "_collector_batch_metadata")
                iter_latency = exec_timer.elapsed_s()
                if iter_latency is None:
                    return
                local_requests = metadata._collector_local_requests
                mode = 2 if all(req[0] == 1 for req in local_requests) else 1
                batch_info = BatchInfos(
                    start_timestamp=_round6(start_wall),
                    end_timestamp=_round6(start_wall + iter_latency),
                    forward_mode=mode,
                    local_forward_mode=mode,
                    requests=local_requests,
                    requests_local=local_requests,
                    dp_size=parallel.data_parallel_size,
                    dp_rank=parallel.data_parallel_rank,
                    worker_rank=self.rank,
                    iter_latency=_round6(iter_latency),
                    logprobs_req_count=logprobs_req_count,
                    logprobs_tokens=logprobs_tokens,
                    logprobs_n=logprobs_n,
                )
                _apply_dp_metadata(metadata, batch_info)
                if sample_latency is not None:
                    batch_info.sample_tokens_latency = _round6(sample_latency)
                if seg is not None:
                    st_ms = (sample_latency or 0.0) * 1e3
                    batch_info.st_segments_ms = {
                        **{k: round(v, 4) for k, v in seg.items()},
                        "other": round(st_ms - sum(seg.values()), 4),
                    }
                BATCH_INFOS.append(batch_info)
                # Print the batch info at its recording (finalize) point.
                _print_batch_info(batch_info)

            # Two-RPC vllm: execute_model returns None and the real output
            # (carrying get_output) comes from sample_tokens -- defer to it so
            # both windows are read together after a single native synchronize.
            if ret is None:
                _PENDING_EXEC = (_finalize_exec,)
                return ret

            # Single-RPC vllm: this call already produced the output object.
            return _finalize_on_get_output(ret, _finalize_exec)
        
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

                def _finalize(model_output):
                    sample_latency = sample_timer.elapsed_s()
                    # Metadata arrives through ModelRunner's returned output.
                    if pending_exec is not None:
                        pending_exec[0](model_output, sample_latency, seg)

                return _finalize_on_get_output(ret, _finalize)

            target.sample_tokens = wrapped_sample_tokens

        original_profile = getattr(target, "profile", None)

        def override_profile(self, is_start: bool = True):
            global BATCH_INFOS, TRANSFER_INFOS

            output_dir = get_output_dir(is_start=is_start)
            rank_suffix = f"rank{self.rank}"

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

        def wrapped_profile(self, is_start: bool = True):
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
            
            # Call the original profile method to trigger the lower-level worker
            return original_profile(self, is_start)

        target.profile = wrapped_profile



class C_SchedulerHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "vllm.v1.core.sched.scheduler"

    @classmethod
    def hook(cls, target) -> None:

        original_add_request = target.add_request
        original_schedule = target.schedule
        original_free_request = target._free_request

        def wrapped_add_request(self, request):
            recv_time = time.time()
            req_info = REQUEST_INFOS[request.request_id]
            req_info.rid = request.request_id
            req_info.queue_start = _round6(recv_time)
            req_info.server_created_time = _round6(request.arrival_time)
            req_info.created_time = _round6(request.arrival_time)
            return original_add_request(self, request)

        def wrapped_schedule(self):
            # D1: timestamp when remote-KV readiness first becomes visible
            # to the scheduler (consumed from finished_recving this round).
            try:
                for _rid in getattr(self, "finished_recving_kv_req_ids", ()):
                    _ri = REQUEST_INFOS.get(_rid)
                    if _ri is not None and _ri.remote_kv_ready_ts == 0.0:
                        _ri.remote_kv_ready_ts = round(time.time(), 6)
            except Exception:
                pass
            scheduler_output = original_schedule(self)

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

                        req_info.total_kv_hit_len = (
                            request.num_cached_tokens
                        )
                        req_info.ext_kv_hit_len = (
                            request.num_external_computed_tokens
                        )
                        req_info.local_kv_hit_len = (
                            request.num_cached_tokens
                            - request.num_external_computed_tokens
                        )

            return scheduler_output

        def wrapped_free_request(self, request):
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
            return original_free_request(self, request)

        target.add_request = wrapped_add_request
        target.schedule = wrapped_schedule
        target._free_request = wrapped_free_request


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
        original_async_start_load_kv = target.async_start_load_kv

        def wrapped_async_start_load_kv(self, metadata):
            ext = getattr(metadata, "external_tokens", None) or {}
            now = time.time()
            for req_id in (getattr(metadata, "reqs_to_load", None) or {}):
                _REQ_CROSS_NODE[req_id] = int(ext.get(req_id, 0) or 0) > 0
                _REQ_SUBMIT_TS[req_id] = now  # queue-wait anchor (T2/T3)
            return original_async_start_load_kv(self, metadata)

        target.async_start_load_kv = wrapped_async_start_load_kv

        # Store side: same submit-timestamp stash, so store rows get a
        # queued_s symmetric to load rows.
        original_async_start_store_kv = target.async_start_store_kv

        def wrapped_async_start_store_kv(self, metadata):
            now = time.time()
            for req_id in (getattr(metadata, "reqs_to_store", None) or {}):
                _REQ_STORE_SUBMIT_TS[req_id] = now
            return original_async_start_store_kv(self, metadata)

        target.async_start_store_kv = wrapped_async_start_store_kv


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
        original_async_swap = target.async_swap

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

        target.async_swap = wrapped_async_swap

        # 同步 swap 路径：部分 vllm 版本/场景的 store（如 bypass 子步 store）
        # 走同步 swap() 而非 async_swap()，必须同时包裹才能全覆盖。
        original_swap = target.swap

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

        target.swap = wrapped_swap

