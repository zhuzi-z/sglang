"""
vLLM engine-core pipeline hooks - the two class hooks that drive one
simulated engine step:

1. C_VLLMSchedulerHook (Scheduler.schedule): created_time-based request
   dispatch (SGLang-style future_queue) and per-request queue/prefix-hit
   stats.
2. C_VLLMExecutorHook (UniProcExecutor.execute_model): accounts the
   simulated GPU span (time prediction, sleep / virtual-clock advance,
   iteration stats, per-token latencies) at the engine's real execution
   seam (model_executor.execute_model, vllm/v1/engine/core.py).

Both hooks share the predictor / iteration-stats / sim-mode state carried
on C_VLLMSchedulerHook, which is why they live in one module.
"""

import heapq
import os
import time
from collections import deque

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import ConfigManager
from sglang_simulator.simulation.manager import StateManager
from sglang_simulator.simulation.manager.env import Envs
from sglang_simulator.simulation.req_stats_manager import request_stats_manager
from sglang_simulator.simulation.types import SimulationMode
from sglang_simulator.simulation.vllm.utils import (
    resolve_model_info,
    resolve_scheduler_config,
)
from sglang_simulator.time_predictor import InferTimePredictor
from sglang_simulator.time_predictor import ScheduleBatch
from sglang_simulator.time_predictor import ScheduleRequest
from sglang_simulator.utils import get_logger

logger = get_logger()

# Max decode steps (output length) from environment variable
# None means no override (use original sampling_params.max_tokens)
_MAX_DECODE_STEPS = int(v) if (v := os.environ.get("SGLANG_SIMULATOR_MAX_DECODE_STEPS")) is not None else None


class C_VLLMSchedulerHook(BaseHook):
    """Hook the vLLM Scheduler for created_time-based request dispatch
    and per-request queue/prefix-hit stats."""

    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "vllm.v1.core.sched.scheduler"

    INFERENCE_PREDICTOR: InferTimePredictor = None
    SIM_MODE: SimulationMode = SimulationMode(Envs.simulation_mode())

    # Per-request stats live in the shared request_stats_manager
    # (simulation/req_stats_manager.py), same as the SGLang backend.
    # Appended by C_VLLMExecutorHook, one record per forward step.  Keep the
    # schema aligned with the SGLang hook so step-level predictor validation
    # can consume either backend.
    ITERATION_STATS: list[dict] = []

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_add_request = target.add_request
        original_schedule = target.schedule
        original_get_num_unfinished = target.get_num_unfinished_requests

        # Future queue: heap of (created_time, seq_no, request)
        # Requests are held here until global_clock >= created_time
        future_queue: list[tuple[float, int, object]] = []
        seq_counter = 0
        total_expected = float("inf")
        all_received = False

        # Per-request created_time lookup (req_id -> created_time)
        req_created_time: dict[str, float] = {}

        def _new_request_stats(request, created_time, queue_start, last_event_time):
            """Initialize the shared RequestStats entry for a request so every
            dumped record carries length / hit fields (0 by default)."""
            input_length = getattr(request, "num_prompt_tokens", None)
            if input_length is None:
                input_length = len(getattr(request, "prompt_token_ids", None) or [])
            st = request_stats_manager.get_req_stats(request.request_id)
            st.created_time = created_time
            st.queue_start = queue_start
            st.queue_end = -1
            st.gen_token_latencies = []
            st.last_event_time = last_event_time
            st.input_length = input_length
            st.output_length = 0
            return st

        def wrapped_init(self, vllm_config, *args, **kwargs):
            """Hook __init__ to initialize AIConfigurator predictor
            from config.json."""
            nonlocal seq_counter, total_expected, all_received
            # Reset closure state for new engine instance
            future_queue.clear()
            seq_counter = 0
            total_expected = float("inf")
            all_received = False
            request_stats_manager.reset()
            cls.ITERATION_STATS.clear()
            req_created_time.clear()
            # Per-instance tracking to avoid cross-worker contamination
            self._sim_req_created_time = {}

            original_init(self, vllm_config, *args, **kwargs)

            try:
                model_config = vllm_config.model_config
                model = resolve_model_info(model_config)
                ConfigManager.set_model_info(model)

                hw = ConfigManager.get_accelerator_info()

                sched_config = resolve_scheduler_config(vllm_config)
                ConfigManager.set_scheduler_config(sched_config)

                cls.INFERENCE_PREDICTOR = ConfigManager.get_inference_time_predictor(
                    model, hw, sched_config
                )
                logger.info("AIConfigurator predictor initialized for vLLM.")
            except Exception as e:
                logger.error("Failed to initialize inference time predictor: %s", e)
                raise

        def wrapped_add_request(self, request):
            """Intercept add_request to divert simulation requests
            into future_queue based on created_time."""
            nonlocal seq_counter, total_expected, all_received

            # Force output length from environment variable (only if set)
            if _MAX_DECODE_STEPS is not None:
                if request.sampling_params is not None:
                    request.sampling_params.max_tokens = _MAX_DECODE_STEPS
                    request.sampling_params.ignore_eos = True
                request.max_tokens = _MAX_DECODE_STEPS

            created_time = None
            if request.sampling_params and request.sampling_params.extra_args:
                sim = request.sampling_params.extra_args.get("simulation")
                if sim:
                    created_time = sim.get("created_time")
                    total = sim.get("total_request")
                    if total is not None:
                        total_expected = total

            if cls.SIM_MODE == SimulationMode.BLOCKING:
                # BLOCKING mode: process immediately, record stats with real time
                now = time.time()
                _new_request_stats(
                    request,
                    created_time=created_time if created_time is not None else now,
                    queue_start=now,
                    last_event_time=now,
                )
                original_add_request(self, request)
            elif created_time is not None:
                # OFFLINE mode: hold in future_queue
                # Register request in self.requests (needed for engine tracking)
                # but do NOT enqueue to waiting yet
                if getattr(request, "resumable", False):
                    request.streaming_queue = deque()
                self.requests[request.request_id] = request

                heapq.heappush(future_queue, (created_time, seq_counter, request))
                seq_counter += 1
                req_created_time[request.request_id] = created_time

                # Check if all requests have been received
                if seq_counter >= total_expected:
                    all_received = True
                    logger.info(
                        "All %d requests received. Starting simulation.",
                        total_expected,
                    )
            else:
                # Non-simulation request - process normally
                original_add_request(self, request)

        def _dispatch_eligible(self):
            """Move requests from future_queue to self.waiting
            whose created_time <= global_clock."""
            current_time = StateManager.get_global_clock()
            while future_queue and future_queue[0][0] <= current_time:
                _, _, request = heapq.heappop(future_queue)
                if hasattr(self, "_enqueue_waiting_request"):
                    self._enqueue_waiting_request(request)
                else:
                    self.waiting.add_request(request)
                # Record queue_start = time when request enters the waiting queue
                ct = req_created_time.get(request.request_id, current_time)
                _new_request_stats(
                    request,
                    created_time=ct,
                    queue_start=current_time,
                    last_event_time=ct,  # starts at created_time
                )

        def wrapped_schedule(self):
            # --- Dispatch eligible requests from future_queue (OFFLINE mode) ---
            if cls.SIM_MODE == SimulationMode.OFFLINE and all_received:
                _dispatch_eligible(self)

                # Idle state: no waiting, no running, but future has items
                # Jump clock to next request's created_time
                if not self.waiting and not self.running and future_queue:
                    next_time = future_queue[0][0]
                    StateManager.set_global_clock(next_time + 1e-6)
                    _dispatch_eligible(self)

            scheduler_output = original_schedule(self)

            # Real-time per-request hit-rate log at inference end: vLLM reports
            # requests finished in the previous step via finished_req_ids.
            for fin_id in getattr(scheduler_output, "finished_req_ids", None) or ():
                st = request_stats_manager.stats.get(fin_id)
                if st is not None:
                    logger.info(
                        "[HitRate] rid=%s input_len=%d local_hit=%d ext_hit=%d",
                        st.rid,
                        st.input_length,
                        st.local_kv_hit_len,
                        st.ext_kv_hit_len,
                    )

            # --- Queue/hit stats for newly scheduled requests ---
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            if not num_scheduled_tokens:
                return scheduler_output

            # Record queue_end and prefix cache hit for newly scheduled requests
            # (first schedule), same scheme/fields as the real vllm_hook:
            #   final_device_hit_len = num_cached_tokens (total reused)
            #   ext_kv_hit_len       = num_external_computed_tokens (cross-node)
            #   local_kv_hit_len     = difference (local radix)
            queue_end_time = (
                time.time()
                if cls.SIM_MODE == SimulationMode.BLOCKING
                else StateManager.get_global_clock()
            )
            if scheduler_output.scheduled_new_reqs:
                for new_req_data in scheduler_output.scheduled_new_reqs:
                    req_id = new_req_data.req_id
                    request = self.requests.get(req_id)
                    if request is None:
                        continue
                    st = request_stats_manager.stats.get(req_id)
                    if st is None or st.queue_end != -1:
                        continue
                    st.queue_end = queue_end_time
                    st.input_length = request.num_prompt_tokens
                    st.output_length = request.max_tokens
                    cached = max(getattr(request, "num_cached_tokens", 0) or 0, 0)
                    ext = getattr(request, "num_external_computed_tokens", 0) or 0
                    st.final_device_hit_len = cached
                    st.ext_kv_hit_len = ext
                    st.local_kv_hit_len = cached - ext
                    # Legacy schema (vllm_worker / metric layer): host = external
                    st.final_host_hit_len = ext

            # The simulated GPU span (time prediction, sleep / virtual-clock
            # advance, iteration stats, per-token latencies) is accounted by
            # C_VLLMExecutorHook at model_executor.execute_model — the
            # engine's real execution seam.
            return scheduler_output

        def wrapped_get_num_unfinished(self):
            return original_get_num_unfinished(self) + len(future_queue)

        target.__init__ = wrapped_init
        target.add_request = wrapped_add_request
        target.schedule = wrapped_schedule
        target.get_num_unfinished_requests = wrapped_get_num_unfinished


class C_VLLMExecutorHook(BaseHook):
    """Hook UniProcExecutor.execute_model to consume the simulated GPU span.

    EngineCore.step() runs the model at
    ``self.model_executor.execute_model(scheduler_output, non_block=True)``.
    UniProcExecutor.collective_rpc executes the worker's method synchronously
    and wraps the result in a resolved Future, so accounting the predicted
    span here keeps the engine loop's timing identical to accounting inside
    schedule() while placing it where real hardware would be busy.

    ExecutorWithExternalLauncher does not override execute_model, so it
    inherits the hooked method.  The module regex also covers forks that
    keep uniproc_executor outside the ``.v1`` namespace.

    All inputs come from the SchedulerOutput itself — no reference to the
    Scheduler instance is needed.
    """

    HOOK_CLASS_NAME = "UniProcExecutor"
    HOOK_MODULE_NAME = r"vllm(\.v1)?\.executor\.uniproc_executor"
    REGEX = True

    _COLD_START_DONE = False

    @classmethod
    def hook(cls, target):
        original_execute_model = target.execute_model

        # req_id -> (prompt_tokens, has_logprobs).  Cached (decode) requests
        # carry neither on SchedulerOutput, so they are remembered from the
        # request's first (new-req) scheduling and dropped once the request
        # shows up in finished_req_ids.
        req_info: dict[str, tuple] = {}

        def wrapped_execute_model(self, scheduler_output, *args, **kwargs):
            num_scheduled_tokens = getattr(
                scheduler_output, "num_scheduled_tokens", None)
            if not num_scheduled_tokens:
                return original_execute_model(
                    self, scheduler_output, *args, **kwargs)

            # --- Per-request registry from new-req sightings ---
            new_reqs = (getattr(scheduler_output, "scheduled_new_reqs", None)
                        or ())
            for new_req in new_reqs:
                prompt_ids = getattr(new_req, "prompt_token_ids", None)
                sp = getattr(new_req, "sampling_params", None)
                req_info[new_req.req_id] = (
                    len(prompt_ids) if prompt_ids is not None else None,
                    bool(getattr(sp, "logprobs", None)),
                )
            for fin_id in (getattr(scheduler_output, "finished_req_ids", None)
                           or ()):
                req_info.pop(fin_id, None)

            # --- num_computed_tokens BEFORE this step (= past_kv_length) ---
            # SchedulerOutput is filled before _update_after_schedule()
            # advances the live requests, so these values are pre-step.
            num_computed = {r.req_id: r.num_computed_tokens for r in new_reqs}
            cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
            if cached is not None:
                num_computed.update(
                    zip(cached.req_ids, cached.num_computed_tokens))

            # --- Build ScheduleBatch + per-request token-emission flags ---
            # token_emitted is False for intermediate chunked-prefill forwards
            # (a token is emitted only once the forward covers the full
            # prompt).  Annotated onto scheduler_output so the sim worker can
            # build its mock ModelRunnerOutput without an executor reference —
            # the SchedulerOutput already travels executor -> worker natively.
            simulation_batch = ScheduleBatch(reqs=[])
            token_emitted: dict[str, bool] = {}
            logprobs_tokens = 0
            for req_id, num_tokens in num_scheduled_tokens.items():
                if req_id not in num_computed:
                    continue
                past_kv_length = num_computed[req_id]
                prompt_tokens, has_logprobs = req_info.get(req_id, (None, False))
                token_emitted[req_id] = (
                    prompt_tokens is None
                    or past_kv_length + num_tokens >= prompt_tokens
                )
                if has_logprobs:
                    logprobs_tokens += num_tokens
                simulation_batch.reqs.append(
                    ScheduleRequest(
                        extend_length=num_tokens,
                        past_kv_length=max(0, past_kv_length),
                    )
                )
            scheduler_output._sim_token_emitted = token_emitted

            result = original_execute_model(
                self, scheduler_output, *args, **kwargs)

            if simulation_batch.is_empty():
                return result

            # --- Predict and account this step's GPU span ---
            StateManager.inc_iteration()
            predictor = C_VLLMSchedulerHook.INFERENCE_PREDICTOR
            if predictor is not None:
                predicted_latency = float(
                    predictor.predict_infer_time(simulation_batch)
                )
            else:
                predicted_latency = 0.001  # fallback: 1ms per step

            # logprobs compensation (both modes): when the baseline was
            # collected without logprobs but the target deployment enables
            # them, the extra GPU work (topk / D2H / tolists) is
            # proportional to this step's scheduled tokens of logprobs
            # requests. Sim has no real GPU, so it must be added
            # explicitly. Only active with logprobs_cost_us_per_token > 0.
            lp_cost = getattr(predictor, "logprobs_cost_s_per_token", 0.0)
            logprobs_latency = lp_cost * logprobs_tokens

            # sample_tokens (RPC-2) compensation, kept as a SEPARATE term
            # from predicted_latency: the predictor's label covers only
            # RPC-1 (execute_model), while the real engine additionally
            # spends the RPC-2 span (sampler + speculative draft propose +
            # bookkeeping D2H + output construction) on GPU every step.
            # Deliberately not folded into predicted_latency so that the
            # iter_latency semantics of the trained label stay intact and
            # both components remain separately auditable in
            # iteration.jsonl. Driven by this step's scheduled token count,
            # mirroring the GPU-side ``total_tokens`` field.
            total_tokens = sum(num_scheduled_tokens.values())
            sample_tokens_latency = 0.0
            if predictor is not None and hasattr(
                predictor, "predict_sample_tokens_time"
            ):
                sample_tokens_latency = float(
                    predictor.predict_sample_tokens_time(total_tokens)
                )

            # Full GPU span actually occupied by this step, matching the
            # GPU-side ``full_step_latency`` = iter + sample_tokens.
            full_step_latency = (
                predicted_latency + sample_tokens_latency + logprobs_latency
            )

            if C_VLLMSchedulerHook.SIM_MODE == SimulationMode.BLOCKING:
                # One-time engine cold start (CUDA graph capture / kernel
                # JIT / first allocation). Measured on RTX PRO 6000: the
                # first non-empty iter of each fresh server process runs
                # ~1.4-1.5 s regardless of token count (6288 tok -> +1.48 s,
                # 256 tok -> +1.38 s), while the predictor only models the
                # per-token forward. Production pays this once per process
                # and it delays the earliest requests' queueing; the sim
                # never paid it, so its queue never built up. Env-gated:
                # unset -> 0 -> behaviour unchanged.
                if not cls._COLD_START_DONE:
                    cls._COLD_START_DONE = True
                    cold_start = float(os.environ.get(
                        "SGLANG_SIMULATOR_COLD_START_S", "0") or 0)
                    if cold_start > 0:
                        logger.info(
                            "[sim-coldstart] one-time cold-start "
                            "overhead %.3f s on first non-empty iter",
                            cold_start)
                        time.sleep(cold_start)
                # BLOCKING: real wall clock. Engine-side per-step CPU
                # overhead (schedule/update/dispatch/post_step) is
                # naturally consumed by the real engine loop, so only the
                # GPU span (RPC-1 + RPC-2 + logprobs term) is slept for.
                time.sleep(abs(full_step_latency))
                event_time = time.time()
            else:
                # OFFLINE: the virtual clock accumulates the full GPU span.
                # Known diff vs real, deliberately NOT compensated: the
                # engine residual (~2.5ms/step = RPC transfers +
                # EngineCore bookkeeping, outside the X1 label) is
                # unmodellable wall-clock time; back-filling it from real
                # measurements would smuggle in an unjustified
                # calibration. OFFLINE duration is expected to run
                # ~2.5ms/step shorter than real; treat it as a known diff
                # item when interpreting results, not a bug.
                StateManager.set_current_inference_dur(full_step_latency)
                StateManager.step_global_clock(full_step_latency)
                event_time = StateManager.get_global_clock()

            C_VLLMSchedulerHook.ITERATION_STATS.append(
                {
                    "requests": simulation_batch.request_info(),
                    # RPC-1 only, matching the predictor's training label.
                    "forward_latency": predicted_latency,
                    "sample_tokens_latency": sample_tokens_latency,
                    "logprobs_latency": logprobs_latency,
                    "full_step_latency": full_step_latency,
                    "total_tokens": total_tokens,
                    "l2_load_latency": 0.0,
                    "l2_backup_latency": 0.0,
                }
            )

            # Record per-token latency for all scheduled requests
            for req_id in num_scheduled_tokens:
                # Intermediate chunked-prefill forwards do not emit a
                # token.  Keep last_event_time unchanged so the first
                # recorded latency is the complete TTFT across all
                # prompt chunks, matching the SGLang hook semantics.
                if not token_emitted.get(req_id, True):
                    continue
                st = request_stats_manager.stats.get(req_id)
                if st is not None:
                    st.gen_token_latencies.append(
                        event_time - st.last_event_time
                    )
                    st.last_event_time = event_time

            return result

        target.execute_model = wrapped_execute_model
        logger.info("[vLLM Hijack] UniProcExecutor hook installed "
                    "(simulated GPU span at execute_model)")
