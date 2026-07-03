"""
vLLM Scheduler Hook - Wraps the Scheduler to inject time prediction
and created_time-based request dispatch.

Similar to SGLang's future_queue pattern:
1. Requests with simulation.created_time are held in a future_queue
2. On each schedule() call, only requests whose created_time <= global_clock
   are dispatched to the actual waiting queue
3. Global clock advances by predicted inference latency each step
"""

import heapq
import time
from collections import deque

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import StateManager
from sglang_simulator.simulation.manager.env import Envs
from sglang_simulator.simulation.types import SimulationMode
from sglang_simulator.time_predictor import InferTimePredictor
from sglang_simulator.time_predictor import ScheduleBatch
from sglang_simulator.time_predictor import ScheduleRequest
from sglang_simulator.utils import get_logger

logger = get_logger()


class C_VLLMSchedulerHook(BaseHook):
    """Hook the vLLM Scheduler class to inject time prediction
    and created_time-based request dispatch."""

    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "vllm.v1.core.sched.scheduler"

    INFERENCE_PREDICTOR: InferTimePredictor = None
    SIM_MODE: SimulationMode = SimulationMode(Envs.simulation_mode())

    # Per-request stats collected during simulation.
    # Key: request_id, Value: dict with queue_start, queue_end,
    #   gen_token_latencies, last_event_time, created_time
    REQUEST_STATS: dict[str, dict] = {}

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
        # Track whether a request has been scheduled at least once
        req_first_scheduled: set[str] = set()

        def wrapped_init(self, vllm_config, *args, **kwargs):
            """Hook __init__ to initialize AIConfigurator predictor
            from config.json."""
            nonlocal seq_counter, total_expected, all_received
            # Reset closure state for new engine instance
            future_queue.clear()
            seq_counter = 0
            total_expected = float("inf")
            all_received = False
            req_created_time.clear()
            req_first_scheduled.clear()
            # Per-instance tracking to avoid cross-worker contamination
            self._sim_req_first_scheduled = set()
            self._sim_req_created_time = {}

            original_init(self, vllm_config, *args, **kwargs)
            try:
                from sglang_simulator.simulation.manager import ConfigManager
                from sglang_simulator.simulation.vllm.utils import (
                    resolve_model_info,
                    resolve_scheduler_config,
                )

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
                rid = request.request_id
                cls.REQUEST_STATS[rid] = {
                    "created_time": created_time if created_time is not None else now,
                    "queue_start": now,
                    "queue_end": -1,
                    "gen_token_latencies": [],
                    "last_event_time": now,
                }
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
                rid = request.request_id
                ct = req_created_time.get(rid, current_time)
                cls.REQUEST_STATS[rid] = {
                    "created_time": ct,
                    "queue_start": current_time,
                    "queue_end": -1,
                    "gen_token_latencies": [],
                    "last_event_time": ct,  # starts at created_time
                }

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

            # --- Call original schedule ---
            scheduler_output = original_schedule(self)

            # --- Build ScheduleBatch and predict inference time ---
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            if not num_scheduled_tokens:
                return scheduler_output

            # Record queue_end and prefix cache hit for newly scheduled requests (first schedule)
            queue_end_time = (
                time.time()
                if cls.SIM_MODE == SimulationMode.BLOCKING
                else StateManager.get_global_clock()
            )
            for req_id, num_tokens in num_scheduled_tokens.items():
                _inst_first = getattr(self, "_sim_req_first_scheduled", req_first_scheduled)
                if req_id not in _inst_first:
                    _inst_first.add(req_id)
                    if req_id in cls.REQUEST_STATS:
                        cls.REQUEST_STATS[req_id]["queue_end"] = queue_end_time
                    # Track prefix cache hit: final_device_hit_len = total reused (L1+L2),
                    # final_host_hit_len = L2 portion. Metric layer subtracts for per-level ratios.
                    request = self.requests.get(req_id)
                    if request is not None:
                        total_hit_len = request.num_computed_tokens - num_tokens
                        host_hit_len = 0
                        # Try prefill_stats (public vLLM) or num_external_computed_tokens (modified vLLM)
                        if (
                            hasattr(request, "prefill_stats")
                            and request.prefill_stats is not None
                        ):
                            host_hit_len = (
                                getattr(
                                    request.prefill_stats,
                                    "num_external_cached_tokens",
                                    0,
                                )
                                or 0
                            )
                        if host_hit_len == 0:
                            host_hit_len = getattr(
                                request, "num_external_computed_tokens", 0
                            ) or 0
                        if total_hit_len > 0 and req_id in cls.REQUEST_STATS:
                            cls.REQUEST_STATS[req_id]["final_device_hit_len"] = (
                                total_hit_len
                            )
                        if host_hit_len > 0 and req_id in cls.REQUEST_STATS:
                            cls.REQUEST_STATS[req_id]["final_host_hit_len"] = (
                                host_hit_len
                            )

            simulation_batch = ScheduleBatch(reqs=[])
            for req_id, num_tokens in num_scheduled_tokens.items():
                request = self.requests.get(req_id)
                if request is None:
                    continue
                # _update_after_schedule() already advanced num_computed_tokens,
                # so past_kv_length = num_computed_tokens - num_tokens.
                # extend_length = num_tokens directly (prefill > 1, decode == 1).
                past_kv_length = request.num_computed_tokens - num_tokens
                simulation_batch.reqs.append(
                    ScheduleRequest(
                        extend_length=num_tokens,
                        past_kv_length=max(0, past_kv_length),
                    )
                )

            if not simulation_batch.is_empty():
                StateManager.inc_iteration()
                predicted_latency = float(
                    cls.INFERENCE_PREDICTOR.predict_infer_time(simulation_batch)
                )

                if cls.SIM_MODE == SimulationMode.BLOCKING:
                    time.sleep(abs(predicted_latency))
                    event_time = time.time()
                else:
                    StateManager.set_current_inference_dur(predicted_latency)
                    StateManager.step_global_clock(predicted_latency)
                    event_time = StateManager.get_global_clock()

                # Record per-token latency for all scheduled requests
                for req_id in num_scheduled_tokens:
                    st = cls.REQUEST_STATS.get(req_id)
                    if st is not None:
                        st["gen_token_latencies"].append(
                            event_time - st["last_event_time"]
                        )
                        st["last_event_time"] = event_time

            return scheduler_output

        def wrapped_get_num_unfinished(self):
            return original_get_num_unfinished(self) + len(future_queue)

        target.__init__ = wrapped_init
        target.add_request = wrapped_add_request
        target.schedule = wrapped_schedule
        target.get_num_unfinished_requests = wrapped_get_num_unfinished
