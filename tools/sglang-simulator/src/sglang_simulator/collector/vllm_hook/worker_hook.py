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


def get_output_dir() -> str:
    base = os.getenv("SIM_COLLECTOR_OUTPUT_DIR", os.getcwd())
    engine_rank = os.getenv("DS_LLM_PROC_RANK")
    if engine_rank is not None:
        sub_dir = f"engine{engine_rank}"
    else:
        sub_dir = f"pid{os.getpid()}"
    output_dir = os.path.join(base, sub_dir)
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

            batch_req_infos = {}
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

            def _finalize_exec(sample_latency: Optional[float] = None):
                if batch_info is None:
                    return
                iter_latency = exec_timer.elapsed_s()
                if iter_latency is None:
                    return
                batch_info.iter_latency = _round6(iter_latency)
                batch_info.end_timestamp = _round6(start_wall + iter_latency)
                if sample_latency is not None:
                    batch_info.sample_tokens_latency = _round6(sample_latency)
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
                ret = original_sample_tokens(self, *args, **kwargs)
                sample_timer.record_end()

                pending_exec = _PENDING_EXEC
                _PENDING_EXEC = None

                def _finalize():
                    sample_latency = sample_timer.elapsed_s()
                    # Hand the paired sample latency to the execute_model
                    # window, which records it on the batch and prints it.
                    if pending_exec is not None:
                        pending_exec[0](sample_latency)

                return _finalize_on_get_output(ret, _finalize)

            target.sample_tokens = wrapped_sample_tokens

        original_profile = getattr(target, "profile", None)

        def override_profile(self, is_start: bool = True):
            global BATCH_INFOS

            output_dir = get_output_dir()
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

            _run_export(_export, is_start)

        target.profile = override_profile



class C_EngineCoreHook(BaseHook):

    HOOK_MODULE_NAME = "vllm.v1.engine.core"
    HOOK_CLASS_NAME = "EngineCore"

    @classmethod
    def hook(cls, target) -> None:

        original_profile = target.profile

        def wrapped_profile(self, is_start: bool = True):
            global REQUEST_INFOS

            output_dir = get_output_dir()

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
