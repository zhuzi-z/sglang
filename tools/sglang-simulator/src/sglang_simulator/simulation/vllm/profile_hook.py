"""vLLM Profile Hook — export REQUEST_STATS / ITERATION_STATS on profile() call."""

import json
import os

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager.env import Envs
from sglang_simulator.utils import get_logger

logger = get_logger()


class C_VLLMProfileHook(BaseHook):
    HOOK_CLASS_NAME = "EngineCore"
    HOOK_MODULE_NAME = "vllm.v1.engine.core"

    @classmethod
    def hook(cls, target):
        def wrapped_profile(self, is_start: bool = True):
            from sglang_simulator.simulation.vllm.scheduler import (
                C_VLLMSchedulerHook,
            )

            output_dir = Envs.output_dir()
            with open(os.path.join(output_dir, "request.jsonl"), "w") as f:
                for item in C_VLLMSchedulerHook.REQUEST_STATS.values():
                    f.write(json.dumps(item, default=str) + "\n")
            with open(os.path.join(output_dir, "iteration.jsonl"), "w") as f:
                for item in C_VLLMSchedulerHook.ITERATION_STATS:
                    f.write(json.dumps(item, default=str) + "\n")

            logger.info(
                "[ProfileHook] Exported %d requests, %d iterations to %s",
                len(C_VLLMSchedulerHook.REQUEST_STATS),
                len(C_VLLMSchedulerHook.ITERATION_STATS),
                output_dir,
            )

            C_VLLMSchedulerHook.REQUEST_STATS.clear()
            C_VLLMSchedulerHook.ITERATION_STATS.clear()

        target.profile = wrapped_profile
