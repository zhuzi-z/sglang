import os
import time
import torch
import json
from typing import Optional
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
# RPC-2 (sample_tokens) window timings; aligns 1:1 with SCHEDULE_INFOS
# strictly by call order.
SAMPLE_TOKENS_LATENCIES: list[float] = []



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
                        "total_tokens": scheduler_output.total_num_scheduled_tokens,
                        "logprobs_req_count": logprobs_req_count,
                        "logprobs_tokens": logprobs_tokens,
                        "logprobs_n": logprobs_n,
                    }
                )

            return ret
        
        target.execute_model = wrapped_execute_model



class C_WorkerHook(BaseHook):
    HOOK_MODULE_NAME = "vllm.v1.worker.gpu_worker"
    HOOK_CLASS_NAME = "Worker"

    @classmethod
    def hook(cls, target) -> None:

        # Two-window collection (X1): if this vllm version has sample_tokens
        # (two-RPC architecture), clamp it with cuda.sync to measure the full
        # RPC-2 span (sampler + MTP draft forward + bookkeeping D2H + output
        # construction). It aligns 1:1 by call order with the execute_model
        # window (RPC-1, measured by C_WorkerWrapperBaseHook).
        # Skipped automatically on older vllm without this method; the
        # iter_latency semantics remain unchanged.
        original_sample_tokens = getattr(target, "sample_tokens", None)
        if original_sample_tokens is not None:

            def wrapped_sample_tokens(self, *args, **kwargs):
                torch.cuda.synchronize()
                start = time.time()
                ret = original_sample_tokens(self, *args, **kwargs)
                torch.cuda.synchronize()
                SAMPLE_TOKENS_LATENCIES.append(time.time() - start)
                return ret

            target.sample_tokens = wrapped_sample_tokens

        def override_profile(self, *args, **kwrags):
            SGL_HOOK_REQ_INFO_DIR = os.getenv("SGL_HOOK_REQ_INFO_DIR", os.getcwd())

            rank_suffix = f"rank{self.rank}"

            n_st = len(SAMPLE_TOKENS_LATENCIES)
            with open(
                f"{SGL_HOOK_REQ_INFO_DIR}/{rank_suffix}.schedule_batch.jsonl",
                "w",
            ) as f:
                for i, batch_infos in enumerate(SCHEDULE_INFOS):
                    # New fields, backward compatible: under the old hook /
                    # old vllm, i >= n_st and no extra field is written.
                    if i < n_st:
                        st_lat = SAMPLE_TOKENS_LATENCIES[i]
                        batch_infos["sample_tokens_latency"] = st_lat
                        il = batch_infos.get("iter_latency")
                        if il is not None:
                            # X1 label: full GPU span = RPC-1 + RPC-2
                            batch_infos["full_step_latency"] = il + st_lat
                    f.write(json.dumps(batch_infos) + "\n")

            print(f"Schedule batch data has been saved to {SGL_HOOK_REQ_INFO_DIR}/{rank_suffix}.schedule_batch.jsonl")
            SCHEDULE_INFOS.clear()
            SAMPLE_TOKENS_LATENCIES.clear()

        target.profile = override_profile



class C_EngineCoreHook(BaseHook):

    HOOK_MODULE_NAME = "vllm.v1.engine.core"
    HOOK_CLASS_NAME = "EngineCore"

    @classmethod
    def hook(cls, target) -> None:

        original_profile = target.profile

        def wrapped_profile(self, is_start: bool = True):
            if not is_start:
                # stop_profile: export REQUEST_INFOS before delegating to
                # the executor (which forwards profile to workers).
                SGL_HOOK_REQ_INFO_DIR = os.getenv(
                    "SGL_HOOK_REQ_INFO_DIR", os.getcwd()
                )

                with open(
                    f"{SGL_HOOK_REQ_INFO_DIR}/rank0.requests.jsonl", "w"
                ) as f:
                    for req_infos in REQUEST_INFOS.values():
                        f.write(json.dumps(asdict(req_infos)) + "\n")

                print(
                    f"Request data has been saved to "
                    f"{SGL_HOOK_REQ_INFO_DIR}/rank0.requests.jsonl"
                )
                REQUEST_INFOS.clear()

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

                        req_info.final_device_hit_len = (
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
            if request.is_finished():
                req_info = REQUEST_INFOS[request.request_id]
                req_info.input_ids = (
                    list(request.prompt_token_ids)
                    if request.prompt_token_ids is not None
                    else []
                )
                req_info.output_ids = list(request.output_token_ids)
            return original_free_request(self, request)

        target.add_request = wrapped_add_request
        target.schedule = wrapped_schedule
        target._free_request = wrapped_free_request

