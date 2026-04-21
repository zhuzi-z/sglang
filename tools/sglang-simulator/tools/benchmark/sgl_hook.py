import importlib
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
from sglang_simulator.hook import BaseHook
from sglang_simulator.hook.utils import get_obj_from_args

SGL_HOOK_FETCH_BATCH_INFO = os.getenv("SGL_HOOK_FETCH_BATCH_INFO", "0") == "1"


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
    input_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)


REQUEST_INFOS: dict[str, RequestInfos] = defaultdict(RequestInfos)
SCHEDULE_INFOS: list[dict] = []


class C_TokenizerManagerHook(BaseHook):
    HOOK_CLASS_NAME = "TokenizerManager"
    HOOK_MODULE_NAME = "sglang.srt.managers.tokenizer_manager"

    @classmethod
    def hook(cls, target):
        original_send_one_request = target._send_one_request

        def wrapped_send_one_request(self, obj, tokenized_obj, created_time):
            if tokenized_obj.sampling_params.custom_params is None:
                tokenized_obj.sampling_params.custom_params = {}
            tokenized_obj.sampling_params.custom_params.update(
                {"server_created_time": created_time}
            )
            return original_send_one_request(self, obj, tokenized_obj, created_time)

        target._send_one_request = wrapped_send_one_request


class C_SglangSchedulerReqHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler"

    LAST_PROCESS_RESULT_END: float = 0
    LAST_GET_NEW_BATCH_DUR: float = 0
    CUR_GET_NEW_BATCH_DUR: float = 0

    @classmethod
    def hook(cls, target_class):
        original_get_new_batch_prefill = target_class.get_new_batch_prefill
        original_recv_requests = target_class.recv_requests
        original_process_batch_result = target_class.process_batch_result
        original_run_batch = target_class.run_batch

        def wrapped_recv_requests(self, *args, **kwargs):
            reqs = original_recv_requests(self, *args, **kwargs)

            recv_time = time.time()
            for req in reqs:
                if not hasattr(req, "rid"):
                    # control request
                    continue
                if req.rid:
                    req_infos = REQUEST_INFOS[req.rid]
                    req_infos.rid = req.rid
                    req_infos.queue_start = recv_time
                    req_infos.last_event_time = recv_time

                    custom_params = req.sampling_params.custom_params
                    req_infos.server_created_time = custom_params.get(
                        "server_created_time"
                    )
                    req_infos.client_created_time = custom_params.get(
                        "client_created_time"
                    )
                    req_infos.created_time = (
                        req_infos.client_created_time or req_infos.server_created_time
                    )

            return reqs

        def wrapped_get_new_batch_prefill(self, *args, **kwargs):
            start = time.time()
            batch = original_get_new_batch_prefill(self, *args, **kwargs)
            C_SglangSchedulerReqHook.LAST_GET_NEW_BATCH_DUR = (
                C_SglangSchedulerReqHook.CUR_GET_NEW_BATCH_DUR
            )
            C_SglangSchedulerReqHook.CUR_GET_NEW_BATCH_DUR = time.time() - start

            if batch is not None and not batch.is_empty():
                prefill_timestamp = time.time()

                for req in batch.reqs:
                    if REQUEST_INFOS[req.rid].queue_end == 0:
                        req_info = REQUEST_INFOS[req.rid]
                        req_info.queue_end = prefill_timestamp
                        req_info.input_length = len(req.origin_input_ids)
                        req_info.output_length = req.sampling_params.max_new_tokens
                        req_info.final_device_hit_len = len(req.prefix_indices)

            if not self.last_batch:
                # Idle
                C_SglangSchedulerReqHook.LAST_PROCESS_RESULT_END = time.time()

            return batch

        def wrapped_run_batch(self, *args, **kwargs):
            torch.cuda.synchronize()
            start = time.time()
            result = original_run_batch(self, *args, **kwargs)
            torch.cuda.synchronize()  # synchronize
            end = time.time()

            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch",
                *args,
                **kwargs,
            )

            if batch is not None:
                request_infos = []
                for req in batch.reqs:
                    request_infos.append(
                        {
                            "rid": req.rid,
                            "extend_input_len": (
                                1
                                if batch.forward_mode.is_decode()
                                else req.extend_input_len
                            ),
                            "prefix_indices_len": len(req.prefix_indices),
                            "output_ids_len": len(req.output_ids),
                        }
                    )

                SCHEDULE_INFOS.append(
                    {
                        "start_timestamp": start,
                        "end_timestamp": end,
                        "forward_mode": int(batch.forward_mode),
                        "request_infos": request_infos,
                        "iter_latency": end - start,
                    }
                )

            return result

        def wrapped_process_batch_result(self, *args, **kwargs):
            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )
            result = original_process_batch_result(self, *args, **kwargs)
            if batch.reqs is None:
                # dummy first batch while overlap schedule is enable.
                return result

            for req in batch.reqs:
                req_stats = REQUEST_INFOS[req.rid]
                if req.finished():
                    req_stats.input_ids = req.origin_input_ids
                    req_stats.output_ids = req.output_ids

            return result

        def wrapped_profile(self, *args, **kwargs):
            SGL_HOOK_REQ_INFO_DIR = os.getenv("SGL_HOOK_REQ_INFO_DIR", os.getcwd())
            if REQUEST_INFOS:
                filename_prefix = f"TP{self.tp_rank}"

                # Only add other ranks if parallelism is enabled (size > 1)
                if getattr(self, "dp_size", 1) > 1:
                    filename_prefix += f"-DP{getattr(self, 'dp_rank', 0)}"
                if getattr(self, "pp_size", 1) > 1:
                    filename_prefix += f"-PP{getattr(self, 'pp_rank', 0)}"
                if getattr(self, "moe_ep_size", 1) > 1:
                    filename_prefix += f"-EP{getattr(self, 'moe_ep_rank', 0)}"

                os.makedirs(SGL_HOOK_REQ_INFO_DIR, exist_ok=True)

                with open(
                    f"{SGL_HOOK_REQ_INFO_DIR}/{filename_prefix}.request.jsonl", "w"
                ) as f:
                    for req_infos in REQUEST_INFOS.values():
                        f.write(json.dumps(asdict(req_infos)) + "\n")

                with open(
                    f"{SGL_HOOK_REQ_INFO_DIR}/{filename_prefix}.schedule_batch.jsonl",
                    "w",
                ) as f:
                    for batch_infos in SCHEDULE_INFOS:
                        f.write(json.dumps(batch_infos) + "\n")

            REQUEST_INFOS.clear()
            SCHEDULE_INFOS.clear()
            # There is not need to call the real profiling api.

            ProfileReqOutput = getattr(
                importlib.import_module("sglang.srt.managers.io_struct"),
                "ProfileReqOutput",
            )
            return ProfileReqOutput(True, "Success")

        target_class.recv_requests = wrapped_recv_requests
        target_class.get_new_batch_prefill = wrapped_get_new_batch_prefill
        target_class.run_batch = wrapped_run_batch
        target_class.process_batch_result = wrapped_process_batch_result
        target_class.profile = wrapped_profile
        return target_class
