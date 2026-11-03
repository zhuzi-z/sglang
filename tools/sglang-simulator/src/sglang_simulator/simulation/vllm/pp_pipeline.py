"""Logical PP in-flight scheduler for CPU simulation.

The physical engine stays a single UniProcExecutor. This stepper reproduces
PP's independent dispatch and completion events instead of sleeping the whole
step inside UniProcExecutor. A full-span prediction determines ``finish_at``;
stage0 occupancy is deliberately a separate policy/interface.
"""

import time
from collections import deque
from dataclasses import dataclass

from sglang_simulator.simulation.manager import StateManager
from sglang_simulator.simulation.types import SimulationMode
from sglang_simulator.utils import get_logger


logger = get_logger()


@dataclass
class _PipelineStep:
    scheduler_output: object
    model_output: object
    finish_at: float
    real: bool


class SimulatedPipelineStepper:
    """Event-driven PP stepper installed as ``EngineCore.step_fn``.

    ``span_share`` is a bring-up-only occupancy fallback: it creates overlap
    without pretending to be a calibrated stage0 model. A predictor can later
    implement ``predict_stage0_occupancy(batch)`` to replace it. ``serial`` is
    provided as a debug control and collapses dispatch to one real step.
    """

    def __init__(self, core, stages: int, occupancy_mode: str = "span_share"):
        if stages < 2:
            raise ValueError(f"PP stepper requires stages >= 2, got {stages}")
        if occupancy_mode not in ("span_share", "serial", "predictor"):
            raise ValueError(f"Unknown PP occupancy mode: {occupancy_mode}")
        self.core = core
        self.stages = int(stages)
        self.occupancy_mode = occupancy_mode
        self.in_flight: deque[_PipelineStep] = deque()
        self.stage0_free_at = 0.0
        self.burst_id = 0
        self.pos_in_burst = -1
        self.burst_open = False
        if occupancy_mode == "span_share":
            logger.warning(
                "[PP Hijack] using uncalibrated span/stages occupancy for "
                "bring-up; TTFT/throughput is not an accuracy baseline"
            )

    def reset(self):
        self.in_flight.clear()
        self.stage0_free_at = self._now()
        self.burst_id = 0
        self.pos_in_burst = -1
        self.burst_open = False

    def _now(self) -> float:
        from sglang_simulator.simulation.vllm.engine_core_pipeline import (
            C_VLLMEngineCoreHook,
        )

        if C_VLLMEngineCoreHook.SIM_MODE == SimulationMode.BLOCKING:
            return time.time()
        return StateManager.get_global_clock()

    def _real_in_flight(self) -> int:
        return sum(1 for step in self.in_flight if step.real)

    def _next_finish(self) -> float | None:
        return self.in_flight[0].finish_at if self.in_flight else None

    def _can_dispatch(self, now: float) -> bool:
        return (
            self._real_in_flight() < self.stages
            and len(self.in_flight) < 2 * self.stages
            and now >= self.stage0_free_at
            and self.core.scheduler.has_requests()
        )

    @staticmethod
    def _carries_state(scheduler_output) -> bool:
        return bool(
            getattr(scheduler_output, "kv_connector_metadata", None) is not None
            or getattr(scheduler_output, "finished_req_ids", None)
        )

    def _occupancy(self, scheduler_output, span: float) -> float:
        if self.occupancy_mode == "serial":
            return span
        from sglang_simulator.simulation.vllm.engine_core_pipeline import (
            C_VLLMEngineCoreHook,
        )

        predictor = C_VLLMEngineCoreHook.INFERENCE_PREDICTOR
        if hasattr(predictor, "predict_stage0_occupancy"):
            batch = getattr(scheduler_output, "_sim_batch", None)
            return max(0.0, float(predictor.predict_stage0_occupancy(batch)))
        if self.occupancy_mode == "predictor":
            raise NotImplementedError(
                "pp_occupancy_mode='predictor' requires "
                "predict_stage0_occupancy(batch) on the configured predictor"
            )
        # Bring-up fallback only. Do not use the full span as stage0 occupancy:
        # that would silently revert the implementation to serial execution.
        return span / self.stages

    def _dispatch(self) -> bool:
        core = self.core
        scheduler = core.scheduler
        throttle = (
            core._should_throttle_prefills()
            if hasattr(core, "_should_throttle_prefills")
            else False
        )
        scheduler_output = scheduler.schedule(throttle)
        total_tokens = getattr(scheduler_output, "total_num_scheduled_tokens", 0)
        real = total_tokens > 0
        if not real and not self._carries_state(scheduler_output):
            return False

        now = self._now()
        if real:
            pos = self.pos_in_burst + 1 if self.burst_open else 0
            scheduler_output._sim_pp_context = {
                "pos_in_burst": pos,
                # Collector dispatch rows include the step just dispatched.
                "in_flight": self._real_in_flight() + 1,
                "num_waiting": len(getattr(scheduler, "waiting", ()) or ()),
                "num_running": len(getattr(scheduler, "running", ()) or ()),
                "num_skipped": len(
                    getattr(scheduler, "skipped_waiting", ()) or ()
                ),
            }

        exec_future = core.model_executor.execute_model(
            scheduler_output, non_block=True
        )
        if real:
            get_grammar = getattr(scheduler, "get_grammar_bitmask", None)
            grammar_output = get_grammar(scheduler_output) if get_grammar else None
            future = core.model_executor.sample_tokens(
                grammar_output, non_block=True
            )
        else:
            future = exec_future
        model_output = future.result()

        if not real:
            self.in_flight.append(
                _PipelineStep(scheduler_output, model_output, now, False)
            )
            return True

        span = getattr(scheduler_output, "_sim_full_step_latency", None)
        if span is None:
            raise RuntimeError(
                "PP executor did not annotate _sim_full_step_latency; "
                "check that C_VLLMExecutorHook is installed"
            )
        span = max(float(span), 0.0)
        occupancy = min(self._occupancy(scheduler_output, span), span)
        start_at = max(now, self.stage0_free_at)
        finish_at = start_at + span
        self.stage0_free_at = start_at + occupancy
        self.in_flight.append(
            _PipelineStep(scheduler_output, model_output, finish_at, True)
        )

        if not self.burst_open:
            self.burst_open = True
            self.burst_id += 1
        self.pos_in_burst = scheduler_output._sim_pp_context["pos_in_burst"]
        scheduler_output._sim_pp_timing = {
            "stage0_start_at": start_at,
            "stage0_free_at": self.stage0_free_at,
            "finish_at": finish_at,
            "occupancy_s": occupancy,
            "span_s": span,
            "burst_id": self.burst_id,
        }
        return True

    def _process(self):
        if not self.in_flight:
            return None
        now = self._now()
        step = self.in_flight[0]
        if step.finish_at > now:
            return None
        self.in_flight.popleft()

        core = self.core
        scheduler_output = step.scheduler_output
        core._process_aborts_queue()
        with (
            core.capture_iteration_details(scheduler_output) as iteration_details,
            core.log_error_detail(scheduler_output),
        ):
            outputs = core.scheduler.update_from_output(
                scheduler_output, step.model_output
            )
        core._attach_iteration_details(outputs, iteration_details)

        from sglang_simulator.simulation.vllm.engine_core_pipeline import (
            _finalize_step_stats,
        )

        _finalize_step_stats(scheduler_output, self._now())
        return outputs

    def _advance_offline(self, now: float) -> None:
        from sglang_simulator.simulation.vllm.engine_core_pipeline import (
            ReqDispatcher,
            _connector_has_pending,
        )

        dispatcher = ReqDispatcher()
        candidates = []
        if self._can_dispatch(now) or (
            self._real_in_flight() < self.stages
            and len(self.in_flight) < 2 * self.stages
            and self.core.scheduler.has_requests()
        ):
            candidates.append(self.stage0_free_at)
        if self.in_flight:
            candidates.append(self.in_flight[0].finish_at)
        if dispatcher.all_received and dispatcher.has_next():
            candidates.append(dispatcher.next_req_created_time())
        future = [value for value in candidates if value > now]
        if future:
            next_at = min(future)
            StateManager.set_current_inference_dur(next_at - now)
            StateManager.set_global_clock(next_at)
            return

        is_pending = (
            self.core.scheduler.get_num_unfinished_requests() > len(dispatcher)
            or _connector_has_pending(self.core.scheduler)
        )
        if is_pending:
            StateManager.step_global_clock(0.005)
            StateManager.set_current_inference_dur(0.005)

    def step(self):
        """Process at most one completion, then dispatch at most one step."""
        from sglang_simulator.simulation.vllm.engine_core_pipeline import (
            C_VLLMEngineCoreHook,
            ReqDispatcher,
        )

        offline = C_VLLMEngineCoreHook.SIM_MODE == SimulationMode.OFFLINE
        if offline and ReqDispatcher().all_received:
            ReqDispatcher().dispatch(self.core)

        outputs = self._process()
        progressed = outputs is not None
        now = self._now()
        if self._can_dispatch(now):
            progressed = self._dispatch() or progressed

        # Match LazyPushStepper: dispatch happens before settle. Completing
        # the previous step and admitting the next one in the same call keeps
        # the current burst open.
        if self._real_in_flight() == 0:
            self.burst_open = False
            self.pos_in_burst = -1

        if offline and not progressed:
            self._advance_offline(self._now())
        elif not offline and not progressed and self.in_flight:
            # Yield to the EngineCore input loop without sleeping through the
            # next dispatch/completion event.
            time.sleep(0.001)
        return outputs, self._real_in_flight() > 0
