import heapq
import importlib
import json
import os
import time
from dataclasses import asdict
from typing import Any

from sglang_simulator.hook import BaseHook
from sglang_simulator.hook.utils import get_obj_from_args
from sglang_simulator.simulation.manager import ConfigManager, Envs, StateManager
from sglang_simulator.simulation.manager.d2h_log import D2HLog
from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager
from sglang_simulator.simulation.sglang.utils import (
    resolve_model_info,
    resolve_scheduler_config,
)
from sglang_simulator.simulation.types import (
    RequestStats,
    SimulationMode,
)
from sglang_simulator.simulation.utils import (
    calc_metrics,
)
from sglang_simulator.time_predictor import InferTimePredictor
from sglang_simulator.time_predictor import ScheduleBatch as SimulationScheduleBatch
from sglang_simulator.time_predictor import ScheduleRequest
from sglang_simulator.utils import get_logger
from sglang_simulator.utils.json import CustomJsonEncoder

logger = get_logger("sgl_simulator")


class C_SglangPrefillAdderHook(BaseHook):
    HOOK_CLASS_NAME = "PrefillAdder"
    HOOK_MODULE_NAME = "sglang.srt.managers.schedule_policy"

    @classmethod
    def hook(cls, target):
        original_add_one_req = target.add_one_req

        def wrapped_add_one_req(self, *args, **kwargs):
            req = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.Req",
                *args,
                **kwargs,
            )
            req_infos = request_stats_manager.get_req_stats(req.rid)
            req_infos.before_adder_device_hit_len = len(req.prefix_indices)
            req_infos.final_host_hit_len = req.host_hit_length

            return original_add_one_req(self, *args, **kwargs)

        target.add_one_req = wrapped_add_one_req


class ReqDispatcher:
    _instance = None
    _initialized = False

    def __new__(cls, mode):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, mode: SimulationMode):
        if self.__class__._initialized:
            return

        self.mode = mode
        # If the simulation mode is `BLOCKING`, all requests are released immediately.
        # If the simulation mode is `OFFLINE`, only control requests, such as `flush_cache`
        # and `server_info`, are released immediately.
        self.immediate_release_requests = []
        self.future_queue: list[tuple[float, int, Any]] = (
            []
        )  # tuple(created time, salt, request)
        self.offline_recv_all_requests = False

    def has_next(self) -> bool:
        return len(self.future_queue) > 0

    def next_req_from_future_ts(self) -> float:
        return self.future_queue[0][0]

    def add(self, reqs: list):
        if self.mode == SimulationMode.BLOCKING:
            self.immediate_release_requests.extend(reqs)
        elif self.mode == SimulationMode.OFFLINE:
            if self.offline_recv_all_requests:
                self.immediate_release_requests.extend(reqs)
                return

            gen_requests = []
            time.sleep(0.05)  # waiting requests

            for req in reqs:
                if req.__class__.__name__ == "TokenizedGenerateReqInput":
                    gen_requests.append(req)
                else:
                    # Such as: /profile_start, /flush_cache, etc.
                    self.immediate_release_requests.append(req)

            # Add requests to future queue
            for req in gen_requests:
                sim_params = None
                if req.sampling_params.custom_params is not None:
                    sim_params = req.sampling_params.custom_params.get("simulation")
                if sim_params is None:
                    # There are some warm-up requests when starting the server without --skip-server-warmup.
                    self.immediate_release_requests.append(req)
                    logger.warning(
                        "Failed to extract the simulation parameters required for simulation from the request. Ignore this warning if the request is a warm-up request."
                    )
                    continue
                if sim_params.get("queue_start"):
                    logger.debug(
                        "Add request to waiting queue with custom queue start timestamp."
                    )

                self.future_queue.append(
                    (
                        sim_params.get("queue_start") or sim_params["created_time"],
                        time.time_ns(),  # The request is not comparable, so add the salt to avoid comparison.
                        req,
                    )
                )

            if len(self.future_queue) != 0:
                _, _, gen_req = self.future_queue[-1]
                total_request = gen_req.sampling_params.custom_params["simulation"][
                    "total_request"
                ]

                if len(self.future_queue) == total_request:
                    self.offline_recv_all_requests = True
                    heapq.heapify(self.future_queue)
                    logger.info("All requests received. Starting simulation now.")
                else:
                    logger.info(
                        f"Offline simulation mode enabled. {total_request} requests expected in total. Received {len(self.future_queue)} requests so far."
                    )

    def dispatch(self) -> list:
        recv_reqs = []

        recv_reqs.extend(self.immediate_release_requests)
        self.immediate_release_requests.clear()

        if self.mode == SimulationMode.OFFLINE and self.offline_recv_all_requests:
            # Process the arrived requests only after all requests have been added to the future queue
            current_timestamp = StateManager.get_global_clock()
            while len(self.future_queue) > 0:
                enqueue_time, _, req = self.future_queue[0]
                if enqueue_time > current_timestamp:
                    break
                recv_reqs.append(req)
                heapq.heappop(self.future_queue)

        now = time.time()
        for req in recv_reqs:
            if req.__class__.__name__ in [
                "BatchTokenizedGenerateReqInput",
                "TokenizedGenerateReqInput",
            ]:
                simulation_args = None
                if req.sampling_params.custom_params is not None:
                    simulation_args = req.sampling_params.custom_params.get(
                        "simulation"
                    )
                # The warm-up request might not include any simulation arguments.
                if simulation_args is None:
                    continue
                req_stats = request_stats_manager.get_req_stats(req.rid)
                req_stats.rid = req.rid
                req_stats.input_length = len(req.input_ids)
                req_stats.output_length = req.sampling_params.max_new_tokens

                if self.mode == SimulationMode.BLOCKING:
                    if "server_created_time" not in simulation_args:
                        logger.warning(
                            "The request's creation time is missing, which may cause the TTFT to be inaccurate."
                        )
                    req_stats.created_time = simulation_args.get(
                        "server_created_time", now
                    )
                    req_stats.last_event_time = req_stats.created_time
                    req_stats.queue_start = now
                elif self.mode == SimulationMode.OFFLINE:
                    req_stats.created_time = simulation_args["created_time"]
                    req_stats.last_event_time = req_stats.created_time
                    # Align with the real queue start timestamp if queue_start is not None. For debugging only.
                    queue_start = simulation_args.get("queue_start")
                    if queue_start is not None:
                        StateManager.set_global_clock(queue_start)
                    req_stats.queue_start = StateManager.get_global_clock()

        if recv_reqs and StateManager.get_last_real_time_ts() == 0:
            StateManager.set_last_real_time_ts(time.time())
            StateManager.set_global_clock(0)

        return recv_reqs


class C_SchedulerRequestReceiver(BaseHook):
    HOOK_CLASS_NAME = "SchedulerRequestReceiver"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler_components.request_receiver"

    REQ_DISPATCHER: ReqDispatcher = ReqDispatcher(
        SimulationMode(Envs.simulation_mode())
    )

    @classmethod
    def hook(cls, target):
        original_recv_requests = target.recv_requests

        def wrapped_recv_requests(self, *args, **kwargs):
            recv_reqs = original_recv_requests(self, *args, **kwargs)
            C_SchedulerRequestReceiver.REQ_DISPATCHER.add(recv_reqs)
            return C_SchedulerRequestReceiver.REQ_DISPATCHER.dispatch()

        target.recv_requests = wrapped_recv_requests


class C_SchedulerHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler"

    INFERENCE_PREDICTOR: InferTimePredictor = None

    ITERATION_STATS: list[dict] = []
    TOTAL_PREDICTOR_TIME_COST = 0
    SIMULATION_BATCH: SimulationScheduleBatch = None
    OVERLAP_SCHEDULE: bool = False
    SIM_MODE = SimulationMode(Envs.simulation_mode())
    # Shared singleton instance with `C_SchedulerRequestReceiver.REQ_DISPATCHER`.
    REQ_DISPATCHER = ReqDispatcher(SIM_MODE)

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_recv_requests = getattr(target, "recv_requests", None)
        original_prefetch_kvcache = target._prefetch_kvcache
        original_get_new_batch_prefill = target.get_new_batch_prefill
        original_run_batch = target.run_batch
        original_process_batch_result = target.process_batch_result
        original_event_loop_normal = target.event_loop_normal
        original_init_request_dispatcher = target.init_request_dispatcher

        def override_event_loop_overlap(self, *args, **kwargs):
            # To reduce the complexity of the simulation, the overlapping schedule is not needed.
            return original_event_loop_normal(self, *args, **kwargs)

        def wrapped_init(self, *args, **kwargs):
            # Disable overlap schedule
            server_args = get_obj_from_args(
                "sglang.srt.server_args.ServerArgs", *args, **kwargs
            )
            C_SchedulerHook.OVERLAP_SCHEDULE = not getattr(
                server_args, "disable_overlap_schedule", False
            )
            setattr(server_args, "disable_overlap_schedule", True)
            logger.debug(
                f"Overlap schedule simulation mode: {C_SchedulerHook.OVERLAP_SCHEDULE}."
            )
            # Use `torch_native` as the attention backend to avoid Triton exceptions during memory operations.
            setattr(server_args, "attention_backend", "torch_native")
            setattr(server_args, "prefill_attention_backend", "torch_native")
            setattr(server_args, "decode_attention_backend", "torch_native")

            original_init(self, *args, **kwargs)

            try:
                if ConfigManager.get_model_info() is None:
                    model = resolve_model_info(self.model_config)
                    ConfigManager.set_model_info(model)

                model = ConfigManager.get_model_info()

                hw = ConfigManager.get_accelerator_info()

                if ConfigManager.get_scheduler_config() is None:
                    sched_config = resolve_scheduler_config(
                        server_args=self.server_args,
                    )
                    ConfigManager.set_scheduler_config(sched_config)
                sched_config = ConfigManager.get_scheduler_config()

                C_SchedulerHook.INFERENCE_PREDICTOR = (
                    ConfigManager.get_inference_time_predictor(model, hw, sched_config)
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize inference time predictor. Error: {e}"
                )
                raise e

        def wrapped_recv_requests(self, *args, **kwargs) -> list:
            recv_reqs = original_recv_requests(self, *args, **kwargs)
            C_SchedulerHook.REQ_DISPATCHER.add(recv_reqs)
            return C_SchedulerHook.REQ_DISPATCHER.dispatch()

        def wrapped_get_new_batch_prefill(self, *args, **kwargs):
            new_batch = original_get_new_batch_prefill(self, *args, **kwargs)
            now = time.time()
            if new_batch is not None:
                for req in new_batch.reqs:
                    req_stats = request_stats_manager.get_req_stats(req.rid)
                    req_stats.final_device_hit_len = req.cached_tokens
                    if req_stats.queue_end == -1:
                        if C_SchedulerHook.SIM_MODE == SimulationMode.BLOCKING:
                            req_stats.queue_end = now
                        else:
                            req_stats.queue_end = StateManager.get_global_clock()
                    else:
                        # Chunked request
                        pass
            elif len(self.running_batch.reqs) == 0 and len(self.waiting_queue) > 0:
                # Prefetching
                StateManager.step_global_clock(0.005)
                StateManager.set_current_inference_dur(0.005)
            else:
                # Idle stage, there are some requests pendding in the future queue.
                if C_SchedulerHook.SIM_MODE == SimulationMode.OFFLINE and (
                    C_SchedulerHook.REQ_DISPATCHER.has_next()
                    and len(self.running_batch.reqs) == 0
                ):
                    next_created_time = (
                        C_SchedulerHook.REQ_DISPATCHER.next_req_from_future_ts()
                    )
                    StateManager.set_global_clock(next_created_time + 1e-6)
            logger.debug(
                f"Get new batch prefill: global iteration={StateManager.get_iteration()}, "
                f"new batch={new_batch.batch_size() if new_batch is not None else 0}, "
                f"waiting queue={len(self.waiting_queue)}"
            )

            return new_batch

        def wrapped_prefetch_kvcache(self, *args, **kwargs):
            original_prefetch_kvcache(self, *args, **kwargs)

            req = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.Req",
                *args,
                **kwargs,
            )
            req_stats = request_stats_manager.get_req_stats(req.rid)
            req_stats.recv_device_hit_len = len(req.prefix_indices)
            req_stats.recv_host_hit_len = req.host_hit_length

        def wrapped_run_batch(self, *args, **kwargs):
            ret = original_run_batch(self, *args, **kwargs)

            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )

            if ret.__class__.__name__ == "GenerationBatchResult":
                simulation_batch = SimulationScheduleBatch(reqs=[])
                if batch.forward_mode.is_extend():
                    for req in batch.reqs:
                        simulation_batch.reqs.append(
                            ScheduleRequest(
                                extend_length=req.extend_input_len,
                                past_kv_length=len(req.prefix_indices)
                                + len(req.output_ids),
                            )
                        )
                elif batch.forward_mode.is_decode():
                    for req in batch.reqs:
                        simulation_batch.reqs.append(
                            ScheduleRequest(
                                extend_length=1,
                                past_kv_length=len(req.prefix_indices)
                                + len(req.output_ids),
                            )
                        )

                if not simulation_batch.is_empty():
                    StateManager.inc_iteration()
                    pred_start = time.perf_counter()
                    predicted_latency = (
                        C_SchedulerHook.INFERENCE_PREDICTOR.predict_infer_time(
                            simulation_batch
                        )
                    )
                    # Accumulate predictor execution time for performance analysis.
                    C_SchedulerHook.TOTAL_PREDICTOR_TIME_COST += (
                        time.perf_counter() - pred_start
                    )
                    predicted_latency = float(predicted_latency)

                    forward_latency = 0
                    if C_SchedulerHook.SIM_MODE == SimulationMode.BLOCKING:
                        time.sleep(abs(predicted_latency))
                        now = time.time()
                        forward_latency = now - StateManager.get_last_real_time_ts()
                        StateManager.set_last_real_time_ts(now)
                    else:
                        forward_latency = predicted_latency

                    StateManager.set_current_inference_dur(forward_latency)

                C_SchedulerHook.SIMULATION_BATCH = simulation_batch

            return ret

        def wrapped_process_batch_result(self, *args, **kwargs):
            ret = original_process_batch_result(self, *args, **kwargs)

            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )
            if batch is not None:
                if len(batch.reqs) == 0:
                    return ret

                hicache_l2_load_dur = StateManager.pop_hicache_l2_load_dur()
                hicache_l2_backup_dur = StateManager.pop_hicache_l2_backup_dur()
                current_inference_dur = StateManager.get_current_inference_dur()

                if C_SchedulerHook.OVERLAP_SCHEDULE:
                    StateManager.step_global_clock(
                        max(
                            hicache_l2_load_dur - StateManager.get_last_inference_dur(),
                            0,
                        )
                    )
                    StateManager.step_global_clock(current_inference_dur)
                    # D2H (write_through backup) runs async on HiCacheController's
                    # backup_queue / write_stream; stream_output returns the first
                    # token without synchronizing on it (verified in real sglang's
                    # scheduler_output_processor_mixin.py:process_batch_result_prefill).
                    # In overlap mode, backup also runs concurrent with subsequent
                    # inference on a separate stream, so it doesn't advance the
                    # wall clock either.
                    request_response_time = StateManager.get_global_clock()
                else:
                    # Serial mode: H2D + forward block the first token return.
                    # D2H still happens after forward but the token is already returned.
                    StateManager.step_global_clock(
                        hicache_l2_load_dur + current_inference_dur
                    )
                    request_response_time = StateManager.get_global_clock()
                    # D2H delays the next iteration's start (no overlap to hide it),
                    # so advance global_clock but DO NOT include it in this request's
                    # response_time.
                    StateManager.step_global_clock(hicache_l2_backup_dur)
                # Request statistics
                for req in batch.reqs:
                    if len(req.output_ids) != 0:  # not chunked
                        req_stats = request_stats_manager.get_req_stats(req.rid)
                        req_stats.gen_token_latencies.append(
                            request_response_time
                            - req_stats.last_event_time  # queue duration
                        )
                        req_stats.last_event_time = request_response_time
                    else:
                        # Chunked request: nothing to do
                        pass
                # Iteration statistics
                C_SchedulerHook.ITERATION_STATS.append(
                    {
                        "requests": C_SchedulerHook.SIMULATION_BATCH.request_info(),
                        "forward_latency": current_inference_dur,
                        "l2_load_latency": hicache_l2_load_dur,
                        "l2_backup_latency": hicache_l2_backup_dur,
                    }
                )
            StateManager.set_last_real_time_ts(time.time())
            return ret

        def override_profile(req, *args, **kwargs):
            stats: list[RequestStats] = []
            for item in request_stats_manager.get_all_req_stats():
                if item.rid is not None and item.input_length > 0:
                    stats.append(item)

            stats = sorted(stats, key=lambda req: req.created_time)

            output_dir = Envs.output_dir()
            os.makedirs(output_dir, exist_ok=True)

            if len(stats) > 0:
                # Remove warmup requests.
                if len(stats) > Envs.num_warmup():
                    metrics_stats = stats[Envs.num_warmup() :]
                else:
                    metrics_stats = stats

                min_created_time = metrics_stats[0].created_time
                # Align timestamps
                for item in stats:
                    item.created_time -= min_created_time
                    item.queue_start -= min_created_time
                    item.queue_end -= min_created_time
                    item.last_event_time -= min_created_time

                metrics = calc_metrics(metrics_stats)
                metrics["time_cost"] = (
                    time.time() - StateManager.get_last_flush_time_ts()
                )
                metrics["predictor_time_cost"] = (
                    C_SchedulerHook.TOTAL_PREDICTOR_TIME_COST
                )

                try:
                    with open(f"{output_dir}/metrics.json", "w") as f:
                        f.write(json.dumps(metrics, cls=CustomJsonEncoder) + "\n")

                    with open(f"{output_dir}/iteration.jsonl", "w") as f:
                        for item in C_SchedulerHook.ITERATION_STATS:
                            f.write(json.dumps(item) + "\n")

                    with open(f"{output_dir}/request.jsonl", "w") as f:
                        for item in stats:
                            f.write(json.dumps(asdict(item)) + "\n")

                    # D2H (L1->L2 backup) IO log — one record per
                    # backup_from_device_all_layer call. Only rank 0 writes to
                    # disk to avoid races when tp_size > 1.
                    if getattr(self, "tp_rank", 0) == 0:
                        with open(f"{output_dir}/d2h.jsonl", "w") as f:
                            for item in D2HLog.drain():
                                f.write(json.dumps(item) + "\n")

                    logger.info(f"Simulation results saved to {output_dir}.")

                except Exception as e:
                    logger.error(f"Failed to dump results. Error: {e}")
            else:
                logger.warning("No request statistics available.")

            StateManager.reset()
            StateManager.set_last_flush_time_ts(time.time())
            request_stats_manager.reset()
            C_SchedulerHook.ITERATION_STATS.clear()
            D2HLog.reset()
            C_SchedulerHook.TOTAL_PREDICTOR_TIME_COST = 0

            ProfileReqOutput = getattr(
                importlib.import_module("sglang.srt.managers.io_struct"),
                "ProfileReqOutput",
            )
            result = {
                "total_request": len(stats),
                "output_directory": output_dir,
            }

            return ProfileReqOutput(True, json.dumps(result))

        def wrapped_init_request_dispatcher(self, *args, **kwargs):
            ret = original_init_request_dispatcher(self, *args, **kwargs)

            _request_dispatcher = getattr(self, "_request_dispatcher", None)

            if _request_dispatcher is not None:
                for ty in _request_dispatcher._mapping.keys():
                    if ty.__name__ == "ProfileReq":
                        _request_dispatcher._mapping[ty] = override_profile
            return ret

        target.event_loop_overlap = override_event_loop_overlap
        target.__init__ = wrapped_init
        target.get_new_batch_prefill = wrapped_get_new_batch_prefill
        target.run_batch = wrapped_run_batch
        target.process_batch_result = wrapped_process_batch_result
        target._prefetch_kvcache = wrapped_prefetch_kvcache
        target.init_request_dispatcher = wrapped_init_request_dispatcher

        if original_recv_requests:
            # version <= 0.5.12.post1
            target.recv_requests = wrapped_recv_requests
