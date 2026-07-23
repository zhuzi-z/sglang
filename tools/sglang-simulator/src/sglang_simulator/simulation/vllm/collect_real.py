"""Real-GPU collection hooks (observer-only) for schedule_batch / requests capture.

Ported from the legacy standalone benchmark hook
(sgl-simulator/.../benchmark/vllm_hook/worker_hook.py) but routed through the shared
`schedule_collector`, so the real-GPU path and the CPU-sim path emit identical-schema
files that can be diffed directly.

These hooks DO NOT replace any vLLM behavior — they only wrap methods to observe. They
must never be installed together with the CPU-sim mock hooks (mock worker / platform /
engine_args), which is enforced by the mode branch in startup.init_hook.

Class-hook constraint reminder: only one hook per class name is applied, so this module
hooks WorkerWrapperBase (iteration timing) and Scheduler (request stats) — both distinct
from the Worker class used by C_WorkerProfileDumpHook and from EngineCore.
"""

import time
from types import MethodType

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.vllm import schedule_collector
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


class C_RealWorkerWrapperHook(BaseHook):
    """Measure real per-iteration GPU latency and record schedule_batch composition."""

    HOOK_CLASS_NAME = "WorkerWrapperBase"
    HOOK_MODULE_NAME = "vllm.v1.worker.worker_base"

    @classmethod
    def hook(cls, target):
        import torch

        original_execute_model = target.execute_model

        def wrapped_execute_model(self, scheduler_output):
            # Build per-request (extend_input_len, prefix_indices_len) from the single
            # broadcast SchedulerOutput (identical on every TP rank).
            extend = {}
            prefix = {}
            for req_id, sched_tokens in scheduler_output.num_scheduled_tokens.items():
                extend[req_id] = sched_tokens

            cached = scheduler_output.scheduled_cached_reqs
            for req_id, completed, output in zip(
                cached.req_ids,
                cached.num_computed_tokens,
                cached.num_output_tokens,
            ):
                prefix[req_id] = completed - output

            for new_req in scheduler_output.scheduled_new_reqs:
                prefix[new_req.req_id] = new_req.num_computed_tokens

            num_scheduled = scheduler_output.num_scheduled_tokens
            forward_mode = (
                2 if num_scheduled and all(n == 1 for n in num_scheduled.values()) else 1
            )

            torch.cuda.synchronize()
            start = time.time()
            ret = original_execute_model(self, scheduler_output)
            torch.cuda.synchronize()
            end = time.time()

            if num_scheduled:
                pairs = [
                    (extend.get(rid, 0), max(0, prefix.get(rid, 0)))
                    for rid in num_scheduled
                ]
                schedule_collector.record_iteration(pairs, forward_mode, end - start)

            return ret

        target.execute_model = wrapped_execute_model


class C_RealSchedulerHook(BaseHook):
    """Collect request-level stats (queue times, prefix hit, lengths) for requests.jsonl."""

    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "vllm.v1.core.sched.scheduler"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_add_request = target.add_request
        original_schedule = target.schedule
        original_free_request = target._free_request

        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)

            orig_get_computed_blocks = self.kv_cache_manager.get_computed_blocks
            orig_get_num_new_matched = (
                self.connector.get_num_new_matched_tokens
                if self.connector is not None
                else None
            )

            def wrapped_get_computed_blocks(_kv_self, request):
                ret = orig_get_computed_blocks(request)
                schedule_collector.record_request(
                    request.request_id, local_kv_hit_len=ret[1]
                )
                return ret

            self.kv_cache_manager.get_computed_blocks = MethodType(
                wrapped_get_computed_blocks, self.kv_cache_manager
            )

            if orig_get_num_new_matched is not None:
                def wrapped_get_num_new_matched(_conn_self, request, num_local):
                    ret = orig_get_num_new_matched(request, num_local)
                    schedule_collector.record_request(
                        request.request_id, ext_kv_hit_len=ret[0]
                    )
                    return ret

                self.connector.get_num_new_matched_tokens = MethodType(
                    wrapped_get_num_new_matched, self.connector
                )

        def wrapped_add_request(self, request):
            recv_time = time.time()
            schedule_collector.record_request(
                request.request_id,
                queue_start=recv_time,
                created_time=getattr(request, "arrival_time", None),
                server_created_time=getattr(request, "arrival_time", None),
            )
            return original_add_request(self, request)

        def wrapped_schedule(self):
            scheduler_output = original_schedule(self)
            if scheduler_output.scheduled_new_reqs:
                prefill_ts = time.time()
                for new_req in scheduler_output.scheduled_new_reqs:
                    request = self.requests.get(new_req.req_id)
                    if request is None:
                        continue
                    schedule_collector.record_request(
                        new_req.req_id,
                        queue_end=prefill_ts,
                        input_length=request.num_prompt_tokens,
                        output_length=request.max_tokens,
                        final_device_hit_len=getattr(request, "num_cached_tokens", 0),
                    )
            return scheduler_output

        def wrapped_free_request(self, request):
            if request.is_finished():
                schedule_collector.record_request(
                    request.request_id,
                    input_ids=list(request.prompt_token_ids)
                    if request.prompt_token_ids is not None
                    else [],
                    output_ids=list(request.output_token_ids),
                )
            return original_free_request(self, request)

        target.__init__ = wrapped_init
        target.add_request = wrapped_add_request
        target.schedule = wrapped_schedule
        target._free_request = wrapped_free_request
