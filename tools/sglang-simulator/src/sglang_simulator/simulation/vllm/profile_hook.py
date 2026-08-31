"""vLLM Profile Hook — export request / iteration stats on profile() call.

``profile(is_start=True)`` doubles as the round separator: it resets the local
prefix cache so each benchmark round starts from a cold engine. This is the
only reset seam available under dashllm_cmd, whose dashserving control server
exposes /start_profile but has no /reset_prefix_cache route (vLLM's own HTTP
server never starts on that path). The vllm-serve path is unaffected in
practice: clear_cache.py already resets there, and the reset is idempotent.
"""

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
            from sglang_simulator.simulation.vllm.engine_core_pipeline import (
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

            if is_start:
                cls._reset_prefix_cache(self)

        target.profile = wrapped_profile

    @staticmethod
    def _reset_prefix_cache(engine_core):
        """Drop the local prefix cache so the next round starts cold.

        reset_connector stays False: the external v6d tier is evicted out of
        band by clear_cache.py against the v6d daemon, and HybridConnector's
        reset_cache() does not reach it anyway.
        """
        reset = getattr(engine_core, "reset_prefix_cache", None)
        if not callable(reset):
            logger.warning(
                "[ProfileHook] EngineCore has no reset_prefix_cache; "
                "local prefix cache carries over between rounds"
            )
            return
        try:
            ok = reset(reset_running_requests=True, reset_connector=False)
        except Exception as e:
            logger.warning("[ProfileHook] reset_prefix_cache failed: %s", e)
            return
        logger.info("[ProfileHook] Local prefix cache reset (ok=%s)", ok)
