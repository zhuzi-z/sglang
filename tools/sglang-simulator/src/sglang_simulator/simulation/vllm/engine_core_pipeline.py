"""
vLLM engine-core pipeline hooks - the class hooks that drive one
simulated engine step:

1. C_VLLMEngineCoreHook (EngineCore init/add_request/step/profile):
   created_time-based request dispatch (ReqDispatcher future_queue),
   BLOCKING-mode request stats, profile stats export, predictor init.
   EngineCore.add_request is the single seam both engine paths funnel
   through (InprocClient and EngineCoreProc) and it sits BEFORE
   kvconn.on_add_req and Scheduler.add_request, so a simulation request is
   parked fully inert: no connector admission, no v6d lookup, no KV-block
   allocation until its created_time is reached and it is dispatched back
   through the native add_request path.
2. C_VLLMSchedulerHook (Scheduler unfinished-count accessors only):
   counts parked future-queue requests so the busy loop keeps stepping —
   the one thing with no EngineCore-level seam.
3. C_VLLMExecutorHook (UniProcExecutor.execute_model): accounts the
   simulated GPU span (time prediction, sleep / virtual-clock advance,
   iteration stats, per-token latencies) and per-request queue/hit stats
   at the engine's real execution seam (model_executor.execute_model,
   vllm/v1/engine/core.py).

The hooks share the predictor / iteration-stats / sim-mode state carried
on C_VLLMEngineCoreHook, which is why they live in one module.
"""

import heapq
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict

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


class ReqDispatcher:
    """Singleton holding the OFFLINE created_time replay state.

    Requests are parked at the EngineCore.add_request seam and released by
    dispatch() once global_clock >= created_time.  While parked they are
    completely inert: no connector admission, no v6d lookup, no KV-block
    allocation.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self.__class__._initialized:
            return
        # tuple(created_time, salt, engine_core, request, request_wave);
        # the salt makes entries comparable-free on identical created_time
        self.future_queue: list[tuple[float, int, object, object, int]] = []
        self.seq_counter = 0
        self.total_expected = float("inf")
        self.all_received = False
        # Per-request created_time lookup (req_id -> created_time)
        self.req_created_time: dict[str, float] = {}
        # Captured once at hook install (single engine per process)
        self.original_core_add_request = None
        self.__class__._initialized = True

    def reset(self):
        self.future_queue.clear()
        self.seq_counter = 0
        self.total_expected = float("inf")
        self.all_received = False
        self.req_created_time.clear()

    def __len__(self) -> int:
        return len(self.future_queue)

    def has_next(self) -> bool:
        return len(self.future_queue) > 0

    def next_req_created_time(self) -> float:
        return self.future_queue[0][0]

    def add(self, engine_core, request, request_wave: int, created_time: float):
        """Park a request fully inert until its created_time."""
        heapq.heappush(
            self.future_queue,
            (created_time, self.seq_counter, engine_core, request, request_wave),
        )
        self.req_created_time[request.request_id] = created_time
        self.seq_counter += 1

        # Check if all requests have been received
        if self.seq_counter >= self.total_expected:
            self.all_received = True
            logger.info(
                "All %d requests received. Starting simulation.",
                self.total_expected,
            )

    def dispatch(self, engine_core):
        """Release parked requests whose created_time <= global_clock back
        through the native EngineCore.add_request path (connector admission,
        scheduler enqueue) — KV-cache work starts exactly here, never earlier."""
        current_time = StateManager.get_global_clock()
        while self.future_queue and self.future_queue[0][0] <= current_time:
            _, _, core, request, request_wave = heapq.heappop(self.future_queue)
            ct = self.req_created_time.get(request.request_id, current_time)
            # Record queue_start = time when the request is dispatched
            _new_request_stats(
                request,
                created_time=ct,
                queue_start=current_time,
                last_event_time=ct,  # starts at created_time
            )
            self.original_core_add_request(core, request, request_wave)


class C_VLLMSchedulerHook(BaseHook):
    """Patch ONLY the scheduler's unfinished-count accessors.

    The engine busy loop reads them directly off the Scheduler instance
    (vllm/v1/engine/core.py: scheduler.has_unfinished_requests), so there
    is no EngineCore-level seam for them: parked future-queue requests
    must be counted here or the loop would go idle and never dispatch.
    Everything else the sim needs lives on the EngineCore / executor
    seams — do not grow this hook back.
    """

    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "vllm.v1.core.sched.scheduler"

    @classmethod
    def hook(cls, target):
        original_get_num_unfinished = target.get_num_unfinished_requests
        target.get_num_unfinished_requests = lambda self: (
            original_get_num_unfinished(self) + len(ReqDispatcher())
        )
        original_has_unfinished = getattr(target, "has_unfinished_requests", None)
        if original_has_unfinished is not None:
            target.has_unfinished_requests = lambda self: (
                original_has_unfinished(self) or len(ReqDispatcher()) > 0
            )


def _ext_computed_tokens(req_id) -> int:
    """External (cross-node/v6d) computed tokens of a newly scheduled
    request.  SchedulerOutput does not carry num_external_computed_tokens
    (it lives on the live Request), so read it through the connector's own
    scheduler accessor when a KV connector is configured — the only case
    where an external hit can exist anyway; 0 otherwise.
    """
    try:
        from vllm.v1.hybrid_connector.engine_proxy import _sched
        req = _sched().requests.get(req_id)
        return max(getattr(req, "num_external_computed_tokens", 0) or 0, 0)
    except Exception:
        return 0


def _finalize_step_stats(scheduler_output, event_time: float) -> None:
    """Publish one step's stats when its model-output future is complete."""
    if getattr(scheduler_output, "_sim_stats_finalized", False):
        return
    iteration_stat = getattr(scheduler_output, "_sim_iteration_stat", None)
    if iteration_stat is None:
        return
    scheduler_output._sim_stats_finalized = True
    C_VLLMEngineCoreHook.ITERATION_STATS.append(iteration_stat)

    token_emitted = getattr(scheduler_output, "_sim_token_emitted", {})
    for req_id in (getattr(scheduler_output, "num_scheduled_tokens", None) or {}):
        if not token_emitted.get(req_id, True):
            continue
        st = request_stats_manager.stats.get(req_id)
        if st is not None:
            st.gen_token_latencies.append(event_time - st.last_event_time)
            st.last_event_time = event_time


class C_VLLMEngineCoreHook(BaseHook):
    """Hook EngineCore for created_time-based request dispatch.

    EngineCore.add_request is the single seam both engine paths funnel
    through (InprocClient.add_request and the EngineCoreProc busy loop) and
    in dashllm vLLM it runs kvconn.on_add_req BEFORE Scheduler.add_request.
    Parking simulation requests here — before either — keeps them fully
    inert until created_time: no connector admission, no v6d lookup, no
    KV-block allocation.  Dispatch happens in EngineCore.step, back through
    the native add_request path, so all KV-cache work starts exactly at the
    request's created_time and in created_time order.

    Note: prefix block hashing still happens earlier in the engine's input
    thread (preprocess_add_request); it is stateless CPU prep that touches
    no cache state, and intercepting the IO thread would be far more
    invasive for no semantic gain.
    """

    HOOK_CLASS_NAME = "EngineCore"
    HOOK_MODULE_NAME = "vllm.v1.engine.core"

    INFERENCE_PREDICTOR: InferTimePredictor = None
    SIM_MODE: SimulationMode = SimulationMode(Envs.simulation_mode())

    # Per-request stats live in the shared request_stats_manager
    # (simulation/req_stats_manager.py), same as the SGLang backend.
    # C_VLLMExecutorHook attaches one record per forward step; OFFLINE publishes
    # it immediately, while BLOCKING publishes after the model-output future
    # completes. Keep the schema aligned with the SGLang hook.
    ITERATION_STATS: list[dict] = []

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_add_request = target.add_request
        original_step = target.step
        original_wait_model_output = getattr(
            target, "_wait_model_output_future", None
        )
        dispatcher = ReqDispatcher()
        dispatcher.original_core_add_request = original_add_request

        def wrapped_init(self, vllm_config, *args, **kwargs):
            """Reset shared replay state, init the AIConfigurator predictor,
            and instance-patch the Scheduler created by EngineCore."""
            ReqDispatcher().reset()
            request_stats_manager.reset()
            cls.ITERATION_STATS.clear()

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

        def wrapped_add_request(self, request, request_wave=0, *args, **kwargs):
            """Park simulation requests in the shared future queue; forward
            everything else to the native path immediately."""
            # Force output length from environment variable (only if set)
            if _MAX_DECODE_STEPS is not None:
                if request.sampling_params is not None:
                    request.sampling_params.max_tokens = _MAX_DECODE_STEPS
                    request.sampling_params.ignore_eos = True
                request.max_tokens = _MAX_DECODE_STEPS

            created_time = None
            sp = request.sampling_params
            if sp is not None and sp.extra_args:
                sim = sp.extra_args.get("simulation")
                if sim:
                    created_time = sim.get("created_time")
                    total = sim.get("total_request")
                    if total is not None:
                        dispatcher.total_expected = total

            if created_time is None or cls_sim_mode() == SimulationMode.BLOCKING:
                if cls_sim_mode() == SimulationMode.BLOCKING:
                    # BLOCKING mode: process immediately, record stats with
                    # real time.  Registered here at the engine entry so
                    # connector-eaten requests are covered too.
                    now = time.time()
                    _new_request_stats(
                        request,
                        created_time=created_time if created_time is not None else now,
                        queue_start=now,
                        last_event_time=now,
                    )
                # Non-simulation request (or BLOCKING mode): native path now
                original_add_request(self, request, request_wave, *args, **kwargs)
                return

            # OFFLINE simulation: park fully inert until created_time
            dispatcher.add(self, request, request_wave, created_time)

        def wrapped_step(self, *args, **kwargs):
            """Dispatch due future-queue requests before each engine step."""
            offline = cls_sim_mode() == SimulationMode.OFFLINE
            if offline and dispatcher.all_received:
                dispatcher.dispatch(self)

            result = original_step(self, *args, **kwargs)
            if offline and not result[1]:
                # The count hook includes parked arrivals; they are not active work.
                is_req_pending = (
                    self.scheduler.get_num_unfinished_requests() > len(dispatcher)
                    or _connector_has_pending(self.scheduler)
                )
                if is_req_pending:
                    # Match SGLang's idle tick. The connector observes the clock
                    # on its next native step and owns all transfer readiness.
                    StateManager.step_global_clock(0.005)
                    StateManager.set_current_inference_dur(0.005)
                elif dispatcher.all_received and dispatcher.has_next():
                    StateManager.set_global_clock(max(
                        StateManager.get_global_clock(), dispatcher.next_req_created_time()))
            return result

        def wrapped_wait_model_output(
            self, model_output_future, step_sout, *args, **kwargs
        ):
            result = original_wait_model_output(
                self, model_output_future, step_sout, *args, **kwargs
            )
            if cls.SIM_MODE == SimulationMode.BLOCKING:
                # The future returned only after _SimAsyncOutput.get_output()
                # has consumed the simulated GPU span. Use that real completion
                # time instead of maintaining a projected GPU clock.
                _finalize_step_stats(step_sout, time.time())
            return result

        target.__init__ = wrapped_init
        target.add_request = wrapped_add_request
        target.step = wrapped_step
        if original_wait_model_output is not None:
            target._wait_model_output_future = wrapped_wait_model_output

        # EngineCore.profile doubles as the benchmark round separator: dump
        # request / iteration stats to SGLANG_SIMULATOR_OUTPUT_DIR and reset
        # simulator-local accounting. Cache reset remains user-controlled via
        # the native API. The hook framework applies only the first hook
        # matching a class, so the profile patch is merged here instead of
        # living in its own hook class on EngineCore.
        def wrapped_profile(self, is_start: bool = True):
            req_stats = request_stats_manager.get_all_req_stats()
            output_dir = Envs.output_dir()
            with open(os.path.join(output_dir, "request.jsonl"), "w") as f:
                for item in req_stats:
                    f.write(json.dumps(asdict(item), default=str) + "\n")
            with open(os.path.join(output_dir, "iteration.jsonl"), "w") as f:
                for item in cls.ITERATION_STATS:
                    f.write(json.dumps(item, default=str) + "\n")

            logger.info(
                "[ProfileHook] Exported %d requests, %d iterations to %s",
                len(req_stats),
                len(cls.ITERATION_STATS),
                output_dir,
            )

            request_stats_manager.reset()
            cls.ITERATION_STATS.clear()
            # Global clock / iteration / future-queue state also reset at
            # round boundaries.  This hook runs in the engine-core process,
            # the only place the reset is reachable under mp (AsyncLLM).
            StateManager.reset()
            ReqDispatcher().reset()

        target.profile = wrapped_profile


def cls_sim_mode() -> SimulationMode:
    return C_VLLMEngineCoreHook.SIM_MODE


def _connector_has_pending(scheduler) -> bool:
    """Check native work interfaces, including loads and trailing stores."""
    get_conn = getattr(scheduler, "get_kv_connector", None)
    connector = get_conn() if get_conn is not None else None
    if connector is None:
        return False
    return bool(connector.has_requests() or getattr(connector, "pending_requests", None))


class C_VLLMExecutorHook(BaseHook):
    """Hook UniProcExecutor.execute_model to consume the simulated GPU span.

    EngineCore.step() runs the model at
    ``self.model_executor.execute_model(scheduler_output, non_block=True)``.
    The prediction remains at this executor seam. In BLOCKING mode its span is
    attached to SchedulerOutput and consumed by the worker's asynchronous
    sample output, keeping the EngineCore loop free to drive the connector. In
    OFFLINE mode the same prediction advances the virtual clock directly.

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
    # Sim-owned single-worker pool that consumes the modelled GPU span off the
    # EngineCore thread, so execute_model can return a still-pending future.
    # max_workers=1 serialises steps, matching a single GPU stream.
    _GPU_SPAN_EXECUTOR = None

    @classmethod
    def _gpu_span_pool(cls):
        if cls._GPU_SPAN_EXECUTOR is None:
            cls._GPU_SPAN_EXECUTOR = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="sim-gpu-span")
        return cls._GPU_SPAN_EXECUTOR

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

            # --- Per-request queue/hit stats (schedule seam, moved here):
            # queue_end at schedule-time clock (before this step is
            # accounted), hit fields per newly scheduled request ---
            queue_end_time = (
                time.time()
                if C_VLLMEngineCoreHook.SIM_MODE == SimulationMode.BLOCKING
                else StateManager.get_global_clock()
            )
            # --- One pass over newly scheduled requests: hit-rate log,
            # queue/hit stats, executor registry ---
            #   final_device_hit_len = num_computed_tokens (local + external)
            #   ext_kv_hit_len       = num_external_computed_tokens (cross-node)
            #   local_kv_hit_len     = difference (local radix)
            new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None) or ()
            for new_req_data in new_reqs:
                req_id = new_req_data.req_id
                st = request_stats_manager.stats.get(req_id)
                prompt_ids = getattr(new_req_data, "prompt_token_ids", None)
                sp = getattr(new_req_data, "sampling_params", None)

                # Queue/hit stats, recorded once at the first schedule
                if st is not None and st.queue_end == -1:
                    st.queue_end = queue_end_time
                    st.input_length = (
                        len(prompt_ids) if prompt_ids is not None else 0
                    )
                    st.output_length = getattr(sp, "max_tokens", 0) or 0
                    cached = max(
                        getattr(new_req_data, "num_computed_tokens", 0) or 0, 0
                    )
                    ext = _ext_computed_tokens(req_id)
                    st.final_device_hit_len = cached
                    st.ext_kv_hit_len = ext
                    st.local_kv_hit_len = cached - ext
                    # Legacy schema (vllm_worker / metric layer): host = external
                    st.final_host_hit_len = ext
                    # Hit values are fixed at admission and never change
                    # afterwards — log once here instead of at finish time
                    logger.info(
                        "[HitRate] rid=%s input_len=%d local_hit=%d ext_hit=%d",
                        st.rid,
                        st.input_length,
                        st.local_kv_hit_len,
                        st.ext_kv_hit_len,
                    )

                # Executor registry: (prompt_tokens, has_logprobs) remembered
                # from the first sighting; dropped on finish below.
                req_info[req_id] = (
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

            if simulation_batch.is_empty():
                return original_execute_model(
                    self, scheduler_output, *args, **kwargs)

            # --- Predict and account this step's GPU span ---
            # Prediction deliberately happens before original_execute_model so
            # BLOCKING can annotate scheduler_output before it reaches the
            # worker.  The prediction depends only on scheduler output state,
            # never on the model result, so this preserves the old semantics.
            StateManager.inc_iteration()
            predictor = C_VLLMEngineCoreHook.INFERENCE_PREDICTOR
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
            # iteration.jsonl. Driven by the same ScheduleBatch the iter
            # predictor sees: the mechanistic model consumes sum_extend
            # and sum(ext_i * past_i) from the per-request aggregation,
            # matching the GPU-side calibration features exactly.
            total_tokens = sum(num_scheduled_tokens.values())
            sample_tokens_latency = 0.0
            if predictor is not None and hasattr(
                predictor, "predict_sample_tokens_time"
            ):
                sample_tokens_latency = float(
                    predictor.predict_sample_tokens_time(simulation_batch)
                )

            # Full GPU span actually occupied by this step, matching the
            # GPU-side ``full_step_latency`` = iter + sample_tokens.
            full_step_latency = (
                predicted_latency + sample_tokens_latency + logprobs_latency
            )

            is_blocking = (
                C_VLLMEngineCoreHook.SIM_MODE == SimulationMode.BLOCKING
            )
            simulated_gpu_span = abs(full_step_latency)

            # Completion accounting travels on the native SchedulerOutput and is
            # published once the modelled span actually elapses (BLOCKING) or the
            # virtual clock advances (OFFLINE).
            scheduler_output._sim_iteration_stat = {
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

            if not is_blocking:
                # OFFLINE: the virtual clock accumulates the full GPU span.
                # Known diff vs real, deliberately NOT compensated: the
                # engine residual (~2.5ms/step = RPC transfers +
                # EngineCore bookkeeping, outside the X1 label) is
                # unmodellable wall-clock time; back-filling it from real
                # measurements would smuggle in an unjustified
                # calibration. OFFLINE duration is expected to run
                # ~2.5ms/step shorter than real; treat it as a known diff
                # item when interpreting results, not a bug.
                result = original_execute_model(
                    self, scheduler_output, *args, **kwargs)
                StateManager.set_current_inference_dur(full_step_latency)
                StateManager.step_global_clock(full_step_latency)
                _finalize_step_stats(
                    scheduler_output, StateManager.get_global_clock())
                return result

            # One-time engine cold start folded into the first modelled span.
            if not cls._COLD_START_DONE:
                cls._COLD_START_DONE = True
                cold_start = float(os.environ.get(
                    "SGLANG_SIMULATOR_COLD_START_S", "0") or 0)
                if cold_start > 0:
                    logger.info(
                        "[sim-coldstart] one-time cold-start "
                        "overhead %.3f s on first non-empty iter",
                        cold_start)
                    simulated_gpu_span += cold_start

            async_scheduling = bool(getattr(
                getattr(self, "scheduler_config", None),
                "async_scheduling", False))

            if async_scheduling:
                # Batch-queue path: EngineCore waits on the sample_tokens
                # future, so defer the span there via the worker's
                # AsyncModelRunnerOutput on vLLM's WorkerAsyncOutput thread.
                scheduler_output._sim_full_step_latency = simulated_gpu_span
                return original_execute_model(
                    self, scheduler_output, *args, **kwargs)

            # Non-async path (e.g. P/D-disagg P side, where vLLM forces async
            # scheduling off): step() waits on execute_model's own future.
            # Build the mock output synchronously (worker does NOT sleep since
            # _sim_full_step_latency is unset), then return a still-pending
            # future whose result is filled after the span elapses on a
            # background thread.  With _enable_bypass on, vLLM's native
            # _wait_model_output_future bypass loop then pumps ADD/ABORT input
            # and drives kvconn.step() during the wait -- reproducing the
            # real-GPU load/admission overlap with no hand-rolled time slicing.
            result = original_execute_model(
                self, scheduler_output, *args, **kwargs)
            model_output = result.result() if isinstance(result, Future) else result

            deferred: Future = Future()
            _span = simulated_gpu_span
            _sout = scheduler_output

            def _consume_gpu_span():
                try:
                    time.sleep(_span)
                    # Real completion timestamp; GPU steps serialise on this
                    # single-worker pool so stat writes never race the engine
                    # thread (which only reaches the next step after this
                    # future resolves).
                    _finalize_step_stats(_sout, time.time())
                    deferred.set_result(model_output)
                except BaseException as exc:  # pragma: no cover
                    deferred.set_exception(exc)

            cls._gpu_span_pool().submit(_consume_gpu_span)
            return deferred

        target.execute_model = wrapped_execute_model
        logger.info("[vLLM Hijack] UniProcExecutor hook installed "
                    "(GPU prediction at execute_model; BLOCKING span deferred)")
