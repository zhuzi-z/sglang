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

        def override_profile_available_bytes(self, *args, **kwargs):
            # return the available hbm capacity after model loading
            model = resolve_model_info(self.model_config)
            hw = ConfigManager.get_accelerator_info()
            scheduler_config = resolve_scheduler_config(
                server_args=self.server_args,
            )
            return profile_device_available_bytes(
                model=model,
                device=hw,
                scheduler_config=scheduler_config,
            )

        
        def wrapped_init_pools(self, *args, **kwargs):
            model_config_keywords = [
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "index_head_dim",
                "kv_lora_rank",
                "head_dim",
                "v_head_dim"
            ]
            
            # set all model_config keywords about kv cache pool allocation to 1 to reduce memory usage
            for kw in model_config_keywords:
                if hasattr(self.model_config, kw):
                    setattr(self.model_config, kw, 1)

            return original_init_pools(self, *args, **kwargs)


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

        target.load_model = override_load_model
        target._profile_available_bytes = override_profile_available_bytes
        target._init_pools = wrapped_init_pools
        target.forward = wrapped_forward
        target.sample = wrapped_sample
        target.compute_logprobs_only = wrapped_compute_logprobs_only
