"""
vLLM EngineArgs Hook - Forces all parallelism parameters to 1.

Similar to SGLang's C_ServerArgsHook, this ensures the simulator runs with
a single worker regardless of user-supplied --tp / --pp / --dp flags.
The actual parallelism is captured in the simulator's scheduler config
(from SGLANG_SIMULATOR_CONFIG_PATH) and used by the time predictor.
"""

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger()


class C_VLLMEngineArgsHook(BaseHook):
    """Hook EngineArgs to force parallelism to 1 for simulation."""

    HOOK_CLASS_NAME = "EngineArgs"
    HOOK_MODULE_NAME = "vllm.engine.arg_utils"

    @classmethod
    def hook(cls, target):
        original_post_init = target.__post_init__

        def wrapped_post_init(self):
            parallel_attrs = [
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "data_parallel_size_local",
                "expert_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            ]
            for attr in parallel_attrs:
                if hasattr(self, attr):
                    setattr(self, attr, 1)

            # Disable expert parallelism flag
            if hasattr(self, "enable_expert_parallel"):
                self.enable_expert_parallel = False

            # Keep the only worker in this process.  A spawned ``mp`` worker
            # starts a fresh interpreter before the simulator hooks are
            # installed and therefore tries to construct/load the real GPU
            # model.  ``uni`` preserves the hooked Worker class.
            if hasattr(self, "distributed_executor_backend"):
                self.distributed_executor_backend = "uni"

            original_post_init(self)
            logger.info(
                "[vLLM Hijack] EngineArgs: forced parallelism to 1 "
                "(tp=%d, pp=%d)",
                self.tensor_parallel_size,
                self.pipeline_parallel_size,
            )

        target.__post_init__ = wrapped_post_init
