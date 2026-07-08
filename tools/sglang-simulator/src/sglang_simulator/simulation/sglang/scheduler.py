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

        self._num_expected_new_reqs: int = -1
        self._num_received_new_reqs = 0
        self._can_dispatch = False
        self.clock_aligned_with_first_req = False

    def set_num_new_reqs(self, num: int):
        self._num_expected_new_reqs = num

    def disable_dispatch(self):
        self._num_received_new_reqs = 0
        self._can_dispatch = False

    def enable_dispatch(self):
        pass
    
    def reset(self):
        self.immediate_release_requests.clear()
        self.future_queue.clear()
        self._num_expected_new_reqs = -1
        self._num_received_new_reqs = 0
        self._can_dispatch = False
        self.clock_aligned_with_first_req = False

    def has_next(self) -> bool:
        return len(self.future_queue) > 0

    def next_req_from_future_ts(self) -> float:
        return self.future_queue[0][0]

    def add(self, reqs: list):
        if self.mode == SimulationMode.BLOCKING:
            self.immediate_release_requests.extend(reqs)
        elif self.mode == SimulationMode.OFFLINE:
            if self._can_dispatch:
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

                self._num_received_new_reqs += 1
                self.future_queue.append(
                    (
                        sim_params.get("queue_start") or sim_params["created_time"],
                        time.time_ns(),  # The request is not comparable, so add the salt to avoid comparison.
                        req,
                    )
                )

            if (not self._can_dispatch) and len(self.future_queue) != 0:
                _, _, gen_req = self.future_queue[-1]
                # If the number of requests is not provided, the `_can_dispatch` will be triggered by user input.
                total_request = gen_req.sampling_params.custom_params["simulation"].get(
                    "total_request"
                )

                if total_request:
                    self._num_expected_new_reqs = total_request

                if self._num_received_new_reqs == self._num_expected_new_reqs:
                    self._can_dispatch = True
                    logger.info("All requests received. Starting simulation now.")
                else:
                    logger.info(
                        f"Offline simulation mode enabled. {self._num_received_new_reqs} requests expected in total. Received {len(self.future_queue)} requests so far."
                    )

    def dispatch(self) -> list:
        recv_reqs = []

        recv_reqs.extend(self.immediate_release_requests)
        self.immediate_release_requests.clear()

        if self._can_dispatch:
            if not self.clock_aligned_with_first_req and len(self.future_queue) > 0:
                # Adjust the global clock to the first request's enqueue time, 
                # since requests created in the decoding instance during PD disaggregation 
                # do not start with a zero creation time.
                heapq.heapify(self.future_queue)
                StateManager.set_global_clock(self.future_queue[0][0])
                self.clock_aligned_with_first_req = True
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
        original_get_new_prebuilt_batch = target.get_new_prebuilt_batch
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

            if hasattr(self, "send_to_detokenizer"):
                hijack_send_to_detokenizer_send_output(getattr(self, "send_to_detokenizer"))
            elif hasattr(self, "ipc_channels") and hasattr(self.ipc_channels, "send_to_detokenizer"):
                hijack_send_to_detokenizer_send_output(getattr(self.ipc_channels, "send_to_detokenizer"))
            else:
                logger.error("Fail to hijack the send_to_detokenizer's send_output, which return request's statistic information.")

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
            if self.server_args.disaggregation_mode == "decode" and recv_reqs:
                pass
            C_SchedulerHook.REQ_DISPATCHER.add(recv_reqs)
            return C_SchedulerHook.REQ_DISPATCHER.dispatch()
        
        def statistics_new_batch(self, new_batch, is_req_pending: bool):
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
            elif is_req_pending:
                # When requests are pendding in the waiting queue, the new batch is empty.
                # Then the `run_batch` may not be called and the global clock is not advanced.
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

        def wrapped_get_new_batch_prefill(self, *args, **kwargs):
            new_batch = original_get_new_batch_prefill(self, *args, **kwargs)
            is_req_pendding = len(self.running_batch.reqs) == 0 and len(self.waiting_queue) > 0
            statistics_new_batch(self, new_batch, is_req_pending=is_req_pendding)
            return new_batch

        def wrapped_get_new_prebuilt_batch(self, *args, **kwargs):
            new_batch = original_get_new_prebuilt_batch(self, *args, **kwargs)
            is_req_pendding = len(self.running_batch.reqs) == 0 and (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
            ) > 0
            statistics_new_batch(self, new_batch, is_req_pending=is_req_pendding)
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

                now = time.time()
                if not ConfigManager.ignore_cpu_overhead():
                    cpu_overhead = now - StateManager.get_last_real_time_ts()
                    StateManager.step_global_clock(cpu_overhead)
                StateManager.set_last_real_time_ts(now)

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
            return ret

        def override_profile(req, *args, **kwargs):

            from sglang.srt.managers.io_struct import ProfileReqType
            from sglang.srt.managers.io_struct import ProfileReqOutput

            if req.type == ProfileReqType.START_PROFILE and req.profile_prefix is not None:
                try:
                    config = json.loads(req.profile_prefix)
                    if config["type"] == "config":
                        if config.get("num_new_reqs"):
                            ReqDispatcher(C_SchedulerHook.SIM_MODE).set_num_new_reqs(config.get("num_new_reqs"))
                    return ProfileReqOutput(True, "Configured")
                except Exception:
                    logger.warning(f"Fail to get configuration from req's attr `profile_prefix={req.profile_prefix}`")

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

                    logger.info(f"Simulation results saved to {output_dir}.")

                except Exception as e:
                    logger.error(f"Failed to dump results. Error: {e}")
            else:
                logger.warning("No request statistics available.")

            StateManager.reset()
            StateManager.set_last_flush_time_ts(time.time())
            request_stats_manager.reset()
            C_SchedulerHook.ITERATION_STATS.clear()
            C_SchedulerHook.TOTAL_PREDICTOR_TIME_COST = 0
            C_SchedulerHook.REQ_DISPATCHER.reset()

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

        def override_pause_generation(self, *args, **kwargs):
            C_SchedulerHook.REQ_DISPATCHER.disable_dispatch()

        def override_continue_generation(self, *args, **kwargs):
            C_SchedulerHook.REQ_DISPATCHER.enable_dispatch()

        def hijack_send_to_detokenizer_send_output(send_to_detokenizer):
            original_send_output = send_to_detokenizer.send_output

            def dummy_send_output(output, recv_obj=None):
                for rid, finish_reason in zip(output.rids, output.finished_reasons):
                    # When the request is finished, response the simulation statistics via finish_reason
                    if finish_reason is not None:
                        req_stat = request_stats_manager.get_req_stats(rid)
                        finish_reason["simulation_stat"] = {
                            "gen_token_latencies": req_stat.gen_token_latencies,
                            "last_event_time": req_stat.last_event_time,
                            "queue_start": req_stat.queue_start,
                            "queue_end": req_stat.queue_end,
                            "final_device_hit_len": req_stat.final_device_hit_len,
                            "final_host_hit_len": req_stat.final_host_hit_len,
                            "final_storage_hit_len": req_stat.final_storage_hit_len,
                            "input_length": req_stat.input_length,
                            "output_length": req_stat.output_length,
                            "kv_cache_transfer_queue_start_time": req_stat.kv_cache_transfer_queue_start_time,
                            "kv_cache_transfer_start_time": req_stat.kv_cache_transfer_start_time,
                            "kv_cache_transfer_duration": req_stat.kv_cache_transfer_duration,
                        }

                original_send_output(output, recv_obj)

            send_to_detokenizer.send_output = dummy_send_output

        target.event_loop_overlap = override_event_loop_overlap
        target.__init__ = wrapped_init
        target.get_new_batch_prefill = wrapped_get_new_batch_prefill
        target.get_new_prebuilt_batch = wrapped_get_new_prebuilt_batch
        target.run_batch = wrapped_run_batch
        target.process_batch_result = wrapped_process_batch_result
        target._prefetch_kvcache = wrapped_prefetch_kvcache
        target.init_request_dispatcher = wrapped_init_request_dispatcher
        target.pause_generation = override_pause_generation
        target.continue_generation = override_continue_generation

        if original_recv_requests:
            # version <= 0.5.12.post1
            target.recv_requests = wrapped_recv_requests
