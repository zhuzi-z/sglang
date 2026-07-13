import os
import time
import torch
import json
from typing import Optional
from types import MethodType
from collections import defaultdict
from dataclasses import dataclass, field, asdict

from sglang_simulator.hook import BaseHook


@dataclass
class RequestInfos:
    rid: str = ""
    created_time: Optional[float] = None
    client_created_time: Optional[float] = None  # client created time
    server_created_time: Optional[float] = None
    queue_start: float = 0.0
    queue_end: float = 0.0
    output_length: int = 0
    input_length: int = 0
    final_device_hit_len: int = 0
    local_kv_hit_len: int = 0
    ext_kv_hit_len: int = 0
    input_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)


SCHEDULE_INFOS: list[dict] = []
REQUEST_INFOS: dict[str, RequestInfos] = defaultdict(RequestInfos)



class C_WorkerWrapperBaseHook(BaseHook):

    HOOK_CLASS_NAME = "WorkerWrapperBase"    
    HOOK_MODULE_NAME = "vllm.v1.worker.worker_base"

    @classmethod
    def hook(cls, target) -> None:

        original_execute_model = target.execute_model

        def wrapped_execute_model(self, scheduler_output: "SchedulerOutput"):

            request_infos = {}
            for req_id, sched_token in scheduler_output.num_scheduled_tokens.items():
                request_infos[req_id] = {"extend_input_len": sched_token}
            
            for req_id, completed_token, output_token in zip(
                scheduler_output.scheduled_cached_reqs.req_ids, 
                scheduler_output.scheduled_cached_reqs.num_computed_tokens,
                scheduler_output.scheduled_cached_reqs.num_output_tokens
            ):
                request_infos[req_id]["prefix_indices_len"] = completed_token - output_token
                request_infos[req_id]["output_ids_len"] = output_token
            
            for req in scheduler_output.scheduled_new_reqs:
                request_infos[req.req_id]["prefix_indices_len"] = req.num_computed_tokens
                request_infos[req.req_id]["output_ids_len"] = 0

            forward_mode = 2 if all([num_token == 1 for num_token in scheduler_output.num_scheduled_tokens.values()]) else 1

            torch.cuda.synchronize()
            start = time.time()
            ret = original_execute_model(self, scheduler_output)
            torch.cuda.synchronize()
            end = time.time()

            if len(request_infos):
                for rid, req_info in request_infos.items():
                    req_info["rid"] = rid

                SCHEDULE_INFOS.append(
                    {
                        "start_timestamp": start,
                        "end_timestamp": end,
                        "forward_mode": forward_mode,
                        "request_infos": list(request_infos.values()),
                        "iter_latency": end - start,
                    }
                )

            return ret
        
        target.execute_model = wrapped_execute_model



class C_WorkerHook(BaseHook):
    HOOK_MODULE_NAME = "vllm.v1.worker.gpu_worker"
    HOOK_CLASS_NAME = "Worker"

    @classmethod
    def hook(cls, target) -> None:

        def override_profile(self, *args, **kwrags):
            SGL_HOOK_REQ_INFO_DIR = os.getenv("SGL_HOOK_REQ_INFO_DIR", os.getcwd())

            rank_suffix = f"rank{self.rank}"

            with open(
                f"{SGL_HOOK_REQ_INFO_DIR}/{rank_suffix}.schedule_batch.jsonl",
                "w",
            ) as f:
                for batch_infos in SCHEDULE_INFOS:
                    f.write(json.dumps(batch_infos) + "\n")
            
            with open(
                f"{SGL_HOOK_REQ_INFO_DIR}/{rank_suffix}.requests.jsonl",
                "w",
            ) as f:
                for req_infos in REQUEST_INFOS.values():
                    f.write(json.dumps(asdict(req_infos)) + "\n")

            print(f"Data has been saved to {SGL_HOOK_REQ_INFO_DIR}")
            SCHEDULE_INFOS.clear()
            REQUEST_INFOS.clear()

        target.profile = override_profile



class C_SchedulerHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "vllm.v1.core.sched.scheduler"

    @classmethod
    def hook(cls, target) -> None:

        original_init = target.__init__
        original_add_request = target.add_request
        original_schedule = target.schedule
        original_free_request = target._free_request

        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)

            original_kv_cache_manager_get_computed_blocks = (
                self.kv_cache_manager.get_computed_blocks
            )
            if self.connector is not None:
                original_connector_get_num_new_matched_tokens = (
                    self.connector.get_num_new_matched_tokens
                )

            def wrapped_get_computed_blocks(kv_self, request):
                ret = original_kv_cache_manager_get_computed_blocks(request)
                num_local_computed = ret[1]
                req_info = REQUEST_INFOS[request.request_id]
                req_info.local_kv_hit_len = num_local_computed
                # req_info.final_device_hit_len = num_local_computed
                return ret

            def wrapped_get_num_new_matched_tokens(
                conn_self, request, num_new_local_computed_tokens
            ):
                ret = original_connector_get_num_new_matched_tokens(
                    request, num_new_local_computed_tokens
                )
                REQUEST_INFOS[request.request_id].ext_kv_hit_len = ret[0]
                return ret

            self.kv_cache_manager.get_computed_blocks = MethodType(
                wrapped_get_computed_blocks, self.kv_cache_manager
            )
            if self.connector is not None:
                self.connector.get_num_new_matched_tokens = MethodType(
                    wrapped_get_num_new_matched_tokens, self.connector
                )

        def wrapped_add_request(self, request):
            recv_time = time.time()
            req_info = REQUEST_INFOS[request.request_id]
            req_info.rid = request.request_id
            req_info.queue_start = recv_time
            req_info.server_created_time = request.arrival_time
            req_info.created_time = request.arrival_time
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
                        req_info.queue_end = prefill_timestamp
                        req_info.input_length = request.num_prompt_tokens
                        req_info.output_length = request.max_tokens
                        req_info.final_device_hit_len = request.num_cached_tokens

            return scheduler_output

        def wrapped_free_request(self, request):
            if request.is_finished():
                req_info = REQUEST_INFOS[request.request_id]
                req_info.input_ids = (
                    list(request.prompt_token_ids)
                    if request.prompt_token_ids is not None
                    else []
                )
                req_info.output_ids = list(request.output_token_ids)
            return original_free_request(self, request)

        target.__init__ = wrapped_init
        target.add_request = wrapped_add_request
        target.schedule = wrapped_schedule
        target._free_request = wrapped_free_request

