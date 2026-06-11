import torch
from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import ConfigManager
from sglang_simulator.simulation.sglang.utils import (
    resolve_model_info,
    resolve_scheduler_config,
)
from sglang_simulator.simulation.utils import profile_device_available_bytes
from sglang_simulator.utils import get_logger

logger = get_logger()


class C_ModelRunnerHook(BaseHook):
    HOOK_CLASS_NAME = "ModelRunner"
    HOOK_MODULE_NAME = "sglang.srt.model_executor.model_runner"

    @classmethod
    def hook(cls, target):

        original_init_pools = target._init_pools

        def override_load_model(self):
            class MockModel:
                def __init__(self):
                    self.start_layers = 0

                def modules(self) -> list:
                    return []
                
                def forward(self, *args, **kwargs):
                    pass
            
            self.model = MockModel()
            self.dtype = self.model_config.dtype

            # Parse other args
            self.sliding_window_size = None
            if (
                self.model_config.is_hybrid_swa
                and self.model_config.sliding_window_size is not None
            ):
                # sliding window field in model config may have different meaning for different kinds of models (e.g., dllm), here we only consider the sliding window in SWA model
                self.sliding_window_size = self.model_config.sliding_window_size
            elif self.model_config.attention_chunk_size is not None:
                self.sliding_window_size = self.model_config.attention_chunk_size

        def override_profile_available_bytes(self, *args, **kwargs):
            # return the available hbm capacity after model loading
            model = resolve_model_info(self.model_config)
            hw = ConfigManager.get_accelerator_info()
            scheduler_config = resolve_scheduler_config(
                server_args=self.server_args,
            )
            rest_memory = profile_device_available_bytes(
                model=model,
                device=hw,
                scheduler_config=scheduler_config,
            ) / (1 << 30)
            if self.mambaish_config is not None:
                rest_memory = self.handle_max_mamba_cache(rest_memory)
            
            return rest_memory * (1 << 30)

        
        def wrapped_init_pools(self, *args, **kwargs):
            kv_cache_attrs = [
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "index_head_dim",
                "kv_lora_rank",
                "head_dim",
                "v_head_dim",
                # mamba2
                "linear_value_head_dim",
                "linear_key_head_dim",
                "linear_conv_kernel_dim"
            ]
            
            # set all model_config keywords about kv cache pool allocation to 1 to reduce memory usage
            modified_model_attrs = {}
            for attr in kv_cache_attrs:
                if hasattr(self.model_config, attr):
                    modified_model_attrs[attr] = getattr(self.model_config, attr)
                    setattr(self.model_config, attr, 1)

            ret = original_init_pools(self, *args, **kwargs)

            for attr, v in modified_model_attrs.items():
                setattr(self.model_config, attr, v)

                # restore the modified model_config keywords, which will be used in the `HostKVCache.get_size_per_token()`
                if hasattr(self.token_to_kv_pool, attr):
                    setattr(self.token_to_kv_pool, attr, v)
                else:
                    logger.warning(f"{self.token_to_kv_pool} does not have attribute {attr}, which has been modified while init_pools")

            from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
            if isinstance(self.token_to_kv_pool, MLATokenToKVPool):
                if getattr(self.token_to_kv_pool, "kv_cache_dim") == 2:
                    setattr(self.token_to_kv_pool, "kv_cache_dim", self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)

            return ret

        def wrapped_forward(self, *args, **kwargs):
            batch = args[0]
            from sglang.srt.layers.logits_processor import LogitsProcessorOutput

            output = LogitsProcessorOutput(
                next_token_logits=torch.empty(
                    size=(batch.batch_size, self.model_config.vocab_size),
                    device=self.device,
                )
            )
            from sglang.srt.model_executor.model_runner import ModelRunnerOutput

            return ModelRunnerOutput(
                logits_output=output,
                can_run_graph=False,
                expert_distribution_metrics=None,
            )

        def wrapped_sample(self, *args, **kwargs):
            logits = args[0]
            ids = torch.ones(
                size=(logits.next_token_logits.shape[0],),
                device=self.device,
                dtype=torch.int64,
            )
            return ids

        def wrapped_compute_logprobs_only(*args, **kwargs):
            return None

        def override_init_attention_backend(self, *args, **kwargs):
            # This might cause a CUDA exception.
            self.attn_backend = None

        target.load_model = override_load_model
        target._profile_available_bytes = override_profile_available_bytes
        target._init_pools = wrapped_init_pools
        target.forward = wrapped_forward
        target.sample = wrapped_sample
        target.compute_logprobs_only = wrapped_compute_logprobs_only
        target.init_attention_backend = override_init_attention_backend
