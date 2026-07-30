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
import os
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

# Max decode steps (output length) from environment variable
# None means no override (use original sampling_params.max_tokens)
_MAX_DECODE_STEPS = int(v) if (v := os.environ.get("SGLANG_SIMULATOR_MAX_DECODE_STEPS")) is not None else None

# Idle-jump lookahead fix (OFFLINE cold-start batching).
# When the clock jumps to the next arrival, also open a lookahead window of
# one engine step's duration so requests arriving within that window are
# batched together, matching real vLLM accumulation during step execution.
# Set SGLANG_SIMULATOR_IDLE_JUMP_FIX=off to restore the legacy behavior.
_IDLE_JUMP_FIX_ENABLED = os.environ.get(
    "SGLANG_SIMULATOR_IDLE_JUMP_FIX", "on"
).lower() in ("1", "true", "on", "yes")
# Fallback lookahead window (seconds) for cold start when no step history.
# Robust parse: empty/invalid values fall back to 1.0 instead of raising at
# import time (which would disable the whole hook).
def _parse_idle_jump_lookahead_default() -> float:
    raw = os.environ.get("SGLANG_SIMULATOR_IDLE_JUMP_LOOKAHEAD_S", "1.0")
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid SGLANG_SIMULATOR_IDLE_JUMP_LOOKAHEAD_S=%r; "
            "falling back to 1.0",
            raw,
        )
        return 1.0


_IDLE_JUMP_LOOKAHEAD_DEFAULT = _parse_idle_jump_lookahead_default()


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
    # One record per scheduler forward step.  Keep the schema aligned with the
    # SGLang hook so step-level predictor validation can consume either backend.
    ITERATION_STATS: list[dict] = []

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_add_request = target.add_request
        original_schedule = target.schedule

        def override_update_from_output(self, scheduler_output, model_output):
            _kvo = getattr(model_output, "kv_connector_output", None)
            _fs = getattr(_kvo, "finished_sending", None) if _kvo else None
            import sys as _dbg3
            print(f"[DBG_SCH] update_from_output kv_connector_output={_kvo is not None} finished_sending={_fs}", file=_dbg3.stderr, flush=True)
            return original_update_from_output(self, scheduler_output, model_output)
        original_get_num_unfinished = target.get_num_unfinished_requests
        original_update_from_output = target.update_from_output

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
            cls.REQUEST_STATS.clear()
            cls.ITERATION_STATS.clear()
            req_created_time.clear()
            req_first_scheduled.clear()
            # Per-instance tracking to avoid cross-worker contamination
            self._sim_req_first_scheduled = set()
            self._sim_req_created_time = {}

            original_init(self, vllm_config, *args, **kwargs)
            native_v6d_control_plane = os.environ.get(
                'SGLANG_SIMULATOR_NATIVE_V6D_CONTROL_PLANE', ''
            ).strip().lower() in {'1', 'true', 'yes', 'on'}
            if native_v6d_control_plane:
                logger.info(
                    '[Scheduler Hook] Native V6D control-plane mode: '
                    'keeping connector=%s',
                    type(self.connector).__name__ if self.connector is not None else None,
                )
            else:
                # Ensure the simulator always owns the scheduler connector.
                # PAI-vLLM may eagerly create a native HybridConnector before the
                # factory hook is reached; in CPU simulation that path will not
                # produce MockHybridConnector request-level ownership logs.  Replace
                # any non-Mock connector so request_finished/get_num_new_matched_tokens
                # consistently go through V6DCacheStorage(etcd).
                try:
                    from sglang_simulator.simulation.vllm.kv_connector import MockHybridConnector
                    kv_cache_config = None
                    if hasattr(self, 'kv_cache_config'):
                        kv_cache_config = self.kv_cache_config
                    if not isinstance(self.connector, MockHybridConnector):
                        original_connector = type(self.connector).__name__ if self.connector is not None else None
                        self.connector = MockHybridConnector(vllm_config, None, kv_cache_config)
                        logger.info(
                            '[Scheduler Hook] Installed MockHybridConnector '
                            '(replaced connector=%s)',
                            original_connector,
                        )
                except Exception as e:
                    logger.warning('[Scheduler Hook] Failed to install MockHybridConnector: %s', e)

            # Always register MockHybridConnector with SupportsHMA regardless
            # of how the connector was created (factory hook or fallback above).
            # This ensures _connector_finished uses request_finished_all_groups
            # on hybrid models with multiple kv_cache_groups (e.g. Qwen3.5).
            try:
                from sglang_simulator.simulation.vllm.kv_connector import MockHybridConnector
                from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
                SupportsHMA.register(MockHybridConnector)
            except Exception:
                pass
            # Share scheduler reference with MockHybridConnector
            try:
                from sglang_simulator.simulation.vllm.kv_connector import set_scheduler_ref
                set_scheduler_ref(self)
            except Exception:
                pass
            # Patch _mamba_block_aligned_split to allow external computed tokens
            # Required for MockHybridConnector v2 (uses scheduler's
            # get_num_new_matched_tokens -> WAITING_FOR_REMOTE_KVS path)
            try:
                if hasattr(self, '_mamba_block_aligned_split'):
                    import types, textwrap
                    orig_fn = self._mamba_block_aligned_split
                    import inspect as _inspect
                    src_lines = _inspect.getsource(orig_fn).split('\n')
                    # Remove the assertion lines
                    filtered = []
                    skip_next = 0
                    for line in src_lines:
                        if skip_next > 0:
                            skip_next -= 1
                            continue
                        if 'num_external_computed_tokens == 0' in line:
                            # Skip this line and next 2 (the error msg + close paren)
                            skip_next = 2
                            filtered.append(line.split('assert')[0] + 'pass  # assertion removed')
                            continue
                        filtered.append(line)
                    new_src = '\n'.join(filtered)
                    new_src = textwrap.dedent(new_src)
                    ns = orig_fn.__globals__.copy()
                    exec(compile(new_src, '<patched_mamba_split>', 'exec'), ns)
                    fn_name = orig_fn.__name__
                    if fn_name in ns:
                        self._mamba_block_aligned_split = ns[fn_name].__get__(self)
                        logger.info('[Scheduler Hook] Patched _mamba_block_aligned_split')
            except Exception as e:
                logger.warning('[Scheduler Hook] Could not patch mamba assertion: %s', e)

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

        # Count of clamped negative token latencies (idle-jump pre-pulled reqs)
        neg_latency_clamps = 0

        def _dispatch_eligible(self, window_end=None):
            """Move requests from future_queue to self.waiting whose
            created_time <= global_clock (or <= window_end when given,
            used by the idle-jump lookahead fix to pre-pull requests
            without advancing the physical clock)."""
            current_time = StateManager.get_global_clock()
            threshold = window_end if window_end is not None else current_time
            while future_queue and future_queue[0][0] <= threshold:
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
                    # Pre-pulled requests (ct > clock) cannot be queued before
                    # they arrive; clamp to ct.  Steady-state path always has
                    # ct <= current_time, so this is a no-op there.
                    "queue_start": max(current_time, ct),
                    "queue_end": -1,
                    "gen_token_latencies": [],
                    "last_event_time": ct,  # starts at created_time
                }

        def wrapped_schedule(self):
            nonlocal neg_latency_clamps
            # --- Dispatch eligible requests from future_queue (OFFLINE mode) ---
            if cls.SIM_MODE == SimulationMode.OFFLINE and all_received:
                _dispatch_eligible(self)

                # Idle state: no waiting, no running, but future has items
                if not self.waiting and not self.running and future_queue:
                    next_time = future_queue[0][0]
                    # Jump clock to next request's created_time (both modes:
                    # keeps the physical timeline truthful for the first
                    # request; TTFT is unaffected by the fix).
                    StateManager.set_global_clock(next_time + 1e-6)
                    if _IDLE_JUMP_FIX_ENABLED:
                        # Pre-pull all requests arriving within one engine
                        # step's duration after next_time, so they are batched
                        # together as real vLLM would accumulate them while
                        # executing a step.  The window only widens dispatch
                        # eligibility; the clock is NOT advanced by lookahead.
                        # The most recent step's predicted latency is stored
                        # via set_current_inference_dur() at each step's end.
                        lookahead = StateManager.get_current_inference_dur()
                        if lookahead <= 0:
                            lookahead = _IDLE_JUMP_LOOKAHEAD_DEFAULT
                        _dispatch_eligible(
                            self, window_end=next_time + lookahead
                        )
                    else:
                        # Legacy behavior: dispatch only requests whose
                        # created_time <= clock (serializes cold-start steps).
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
                        # Pre-pulled requests can be scheduled at a clock
                        # earlier than their queue_start (= created_time);
                        # clamp so queue_end - queue_start >= 0.  Steady-state
                        # path always has queue_end_time >= queue_start.
                        cls.REQUEST_STATS[req_id]["queue_end"] = max(
                            queue_end_time,
                            cls.REQUEST_STATS[req_id]["queue_start"],
                        )
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
                if cls.INFERENCE_PREDICTOR is not None:
                    predicted_latency = float(
                        cls.INFERENCE_PREDICTOR.predict_infer_time(simulation_batch)
                    )
                else:
                    predicted_latency = 0.001  # fallback: 1ms per step

                if cls.SIM_MODE == SimulationMode.BLOCKING:
                    time.sleep(abs(predicted_latency))
                    event_time = time.time()
                else:
                    StateManager.set_current_inference_dur(predicted_latency)
                    StateManager.step_global_clock(predicted_latency)
                    event_time = StateManager.get_global_clock()

                cls.ITERATION_STATS.append(
                    {
                        "requests": simulation_batch.request_info(),
                        "forward_latency": predicted_latency,
                        "l2_load_latency": 0.0,
                        "l2_backup_latency": 0.0,
                    }
                )

                # Record per-token latency for all scheduled requests
                for req_id in num_scheduled_tokens:
                    request = self.requests.get(req_id)
                    if request is not None:
                        prompt_tokens = getattr(request, "num_prompt_tokens", None)
                        computed_tokens = getattr(
                            request, "num_computed_tokens", 0
                        )
                        # Intermediate chunked-prefill forwards do not emit a
                        # token.  Keep last_event_time unchanged so the first
                        # recorded latency is the complete TTFT across all
                        # prompt chunks, matching the SGLang hook semantics.
                        if (
                            prompt_tokens is not None
                            and computed_tokens < prompt_tokens
                        ):
                            continue
                    st = cls.REQUEST_STATS.get(req_id)
                    if st is not None:
                        token_latency = event_time - st["last_event_time"]
                        if token_latency < 0:
                            # Pre-pulled request (idle-jump lookahead): the
                            # first token event can precede created_time when
                            # the step latency is shorter than the arrival
                            # offset.  Clamp to 0; keeping the later timestamp
                            # as last_event_time preserves e2e = sum(latencies)
                            # = last_token_time - created_time.
                            neg_latency_clamps += 1
                            logger.debug(
                                "Clamped negative token latency %.6fs for "
                                "req %s (idle-jump pre-pull, total clamps=%d)",
                                token_latency,
                                req_id,
                                neg_latency_clamps,
                            )
                            token_latency = 0.0
                        st["gen_token_latencies"].append(token_latency)
                        st["last_event_time"] = max(
                            event_time, st["last_event_time"]
                        )

            return scheduler_output

        def wrapped_get_num_unfinished(self):
            return original_get_num_unfinished(self) + len(future_queue)



        target.__init__ = wrapped_init
        target.add_request = wrapped_add_request
        target.schedule = wrapped_schedule
        target.update_from_output = override_update_from_output
        target.get_num_unfinished_requests = wrapped_get_num_unfinished
