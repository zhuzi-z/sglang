"""
vLLM EngineArgs Hook - Forces all parallelism parameters to 1.

Similar to SGLang's C_ServerArgsHook, this ensures the simulator runs with
a single worker regardless of user-supplied --tp / --pp / --dp flags.
The actual parallelism is captured in the simulator's scheduler config
(from SGLANG_SIMULATOR_CONFIG_PATH) and used by the time predictor.
"""

import os

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
                # Multi-node PP is represented logically by the simulator;
                # never spawn real headless followers on CPU.
                "nnodes",
            ]
            for attr in parallel_attrs:
                if hasattr(self, attr):
                    setattr(self, attr, 1)

            if hasattr(self, "node_rank"):
                self.node_rank = 0

            # Disable expert parallelism flag
            if hasattr(self, "enable_expert_parallel"):
                self.enable_expert_parallel = False
            if hasattr(self, "enable_ep_weight_filter"):
                self.enable_ep_weight_filter = False

            # Do not instantiate the TPU compiler/attention backend while the
            # logical PP/PCP configuration is preserved for the predictor.
            if "CUSTOM" in str(getattr(self, "attention_backend", "")).upper():
                self.attention_backend = None
            compilation = getattr(self, "compilation_config", None)
            if isinstance(compilation, dict) and "vllm_torchtpu" in str(
                compilation.get("backend", "")
            ):
                self.compilation_config = {"cudagraph_mode": "NONE"}

            speculative = getattr(self, "speculative_config", None)
            method = (
                speculative.get("method")
                if isinstance(speculative, dict)
                else getattr(speculative, "method", None)
            )
            if (
                getattr(self, "disaggregation_mode", None) == "prefill"
                and method == "mtp"
            ):
                os.environ["TPU_MTP_PREFILL_ONLY"] = "1"

            # Keep the only worker in this process.  A spawned ``mp`` worker
            # starts a fresh interpreter before the simulator hooks are
            # installed and therefore tries to construct/load the real GPU
            # model.  ``uni`` preserves the hooked Worker class.
            if hasattr(self, "distributed_executor_backend"):
                self.distributed_executor_backend = "uni"

            original_post_init(self)

            profiler_config = getattr(self, "profiler_config", None)
            if isinstance(profiler_config, dict):
                if profiler_config.get("profiler") is None:
                    profiler_config["profiler"] = "cuda"
            elif profiler_config is not None:
                if profiler_config.profiler is None:
                    profiler_config.profiler = "cuda"

            logger.info(
                "[vLLM Hijack] EngineArgs: forced parallelism to 1 "
                "(tp=%d, pp=%d)",
                self.tensor_parallel_size,
                self.pipeline_parallel_size,
            )

        target.__post_init__ = wrapped_post_init
