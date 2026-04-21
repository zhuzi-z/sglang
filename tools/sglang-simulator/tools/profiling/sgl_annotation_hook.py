import time

import torch
from sglang_simulator.hook import BaseHook
from sglang_simulator.hook.utils import get_obj_from_args


class C_SglangSchedulerRunBatchAnnotationHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler"

    @classmethod
    def hook(cls, target_class):
        original_recv_requests = target_class.recv_requests
        original_run_batch = target_class.run_batch

        def wrapped_recv_requests(self, *args, **kwargs):
            recv_reqs: list = original_recv_requests(self, *args, **kwargs)
            # Force waiting all requests
            if len(recv_reqs) > 0:
                time.sleep(0.02)
                recv_reqs.extend(original_recv_requests(self, *args, **kwargs))
            return recv_reqs

        def wrapped_run_batch(self, *args, **kwargs):
            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch",
                *args,
                **kwargs,
            )

            if batch is not None:
                if batch.forward_mode.is_decode():
                    request_infos = [
                        (1, len(req.prefix_indices) + len(req.output_ids))
                        for req in batch.reqs
                    ]
                else:
                    # TODO: mixed chunk and other mode
                    request_infos = [
                        (req.extend_input_len, len(req.prefix_indices))
                        for req in batch.reqs
                    ]

                msg = f"Scheduler.run_batch: {request_infos}"
                with torch.profiler.record_function(msg):
                    with torch.cuda.nvtx.range(msg):
                        return original_run_batch(self, *args, **kwargs)

            return original_run_batch(self, *args, **kwargs)

        target_class.recv_requests = wrapped_recv_requests
        target_class.run_batch = wrapped_run_batch


class C_SglangModelForwardAnnotationHook(BaseHook):
    HOOK_CLASS_NAME = r".+"
    HOOK_MODULE_NAME = r"^sglang\.srt\.(models|layers)\..*"
    REGEX = True

    @classmethod
    def hook(cls, target_class):
        if not hasattr(target_class, "forward") and not issubclass(
            torch.nn.Module, target_class
        ):
            return target_class

        original_forward = target_class.forward

        def wrapped_forward(self, *args, **kwargs):
            msg = f"{self.__class__.__name__}.forward"
            with torch.profiler.record_function(msg):
                with torch.cuda.nvtx.range(msg):
                    result = original_forward(self, *args, **kwargs)
                    return result

        target_class.forward = wrapped_forward

        return target_class
