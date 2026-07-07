import os
import time
import torch
import json

from sglang_simulator.hook import BaseHook


SCHEDULE_INFOS: list[dict] = []


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

            print(f"Data has been saved to {SGL_HOOK_REQ_INFO_DIR}")
            SCHEDULE_INFOS.clear()

        target.profile = override_profile
