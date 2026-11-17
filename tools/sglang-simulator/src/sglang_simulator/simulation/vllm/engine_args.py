"""
vLLM EngineArgs Hook - Forces all parallelism parameters to 1.

Similar to SGLang's C_ServerArgsHook, this ensures the simulator runs with
a single worker regardless of user-supplied --tp / --pp / --dp flags.
The actual parallelism is captured in the simulator's scheduler config
(from SGLANG_SIMULATOR_CONFIG_PATH) and used by the time predictor.
"""

import os
import sys
from functools import wraps

from sglang_simulator.hook import BaseHook
from sglang_simulator.hook.class_hook_entry import emit_hook_event
from sglang_simulator.utils import get_logger

logger = get_logger()


class C_VLLMEngineArgsHook(BaseHook):
    """Hook EngineArgs to force parallelism to 1 for simulation."""

    HOOK_CLASS_NAME = "EngineArgs"
    HOOK_MODULE_NAME = "vllm.engine.arg_utils"

    @classmethod
    def hook(cls, target):
        original_post_init = target.__post_init__
        original_create_engine_config = target.create_engine_config

        def wrapped_post_init(self):
            emit_hook_event(
                "ENGINE_CONFIG", stage="engine_args_input",
                backend=getattr(self, "distributed_executor_backend", None),
                worker_cls=getattr(self, "worker_cls", None),
            )
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

            # 逻辑 PP 由本地 stage workers 表达，内层执行器保持单进程。
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

            emit_hook_event(
                "ENGINE_CONFIG", stage="engine_args",
                backend=getattr(self, "distributed_executor_backend", None),
                worker_cls=getattr(self, "worker_cls", None),
                tp=self.tensor_parallel_size, pp=self.pipeline_parallel_size,
            )
            logger.info(
                "[vLLM Hijack] EngineArgs: forced parallelism to 1 "
                "(tp=%d, pp=%d)",
                self.tensor_parallel_size,
                self.pipeline_parallel_size,
            )

        @wraps(original_create_engine_config)
        def wrapped_create_engine_config(self, *args, **kwargs):
            config = original_create_engine_config(self, *args, **kwargs)
            parallel = config.parallel_config
            platform = getattr(sys.modules.get("vllm.platforms"), "current_platform", None)
            emit_hook_event(
                "ENGINE_CONFIG", stage="create_engine_config",
                platform=type(platform).__module__ + "." + type(platform).__qualname__,
                backend=parallel.distributed_executor_backend, worker_cls=parallel.worker_cls,
                tp=parallel.tensor_parallel_size, pp=parallel.pipeline_parallel_size,
            )
            return config

        target.__post_init__ = wrapped_post_init
        # DashLLM backports EngineArgs by inspecting its original method source.
        # Keep __post_init__ hooked, but let its backport see the native method.
        if os.environ.get("HISIM_TPU_PP_PRESERVE_DASHLLM_ENGINE_ARGS") != "1":
            target.create_engine_config = wrapped_create_engine_config
