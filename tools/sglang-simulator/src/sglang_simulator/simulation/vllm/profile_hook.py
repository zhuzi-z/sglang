"""vLLM Profile Hook — export request / iteration stats on profile() call."""

import json
import os
from dataclasses import asdict

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager.env import Envs
from sglang_simulator.simulation.req_stats_manager import request_stats_manager
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

            req_stats = request_stats_manager.get_all_req_stats()
            output_dir = Envs.output_dir()
            with open(os.path.join(output_dir, "request.jsonl"), "w") as f:
                for item in req_stats:
                    f.write(json.dumps(asdict(item), default=str) + "\n")
            with open(os.path.join(output_dir, "iteration.jsonl"), "w") as f:
                for item in C_VLLMSchedulerHook.ITERATION_STATS:
                    f.write(json.dumps(item, default=str) + "\n")

            logger.info(
                "[ProfileHook] Exported %d requests, %d iterations to %s",
                len(req_stats),
                len(C_VLLMSchedulerHook.ITERATION_STATS),
                output_dir,
            )

            request_stats_manager.reset()
            C_VLLMSchedulerHook.ITERATION_STATS.clear()

        target.profile = wrapped_profile
