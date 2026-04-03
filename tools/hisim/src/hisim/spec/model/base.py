from dataclasses import dataclass
from typing import Optional


from hisim.utils import get_logger


logger = get_logger("hisim")


@dataclass
class ModelInfo:
    hf_config: Optional[dict] = None
    model_path: Optional[str] = None

    @property
    def torch_dtype(self) -> str:
        return self.hf_config.get("dtype")

    @property
    def kv_lora_rank(self) -> int:
        return self.hf_config.get("kv_lora_rank", 0)

    @property
    def qk_rope_head_dim(self) -> int:
        return self.hf_config.get("qk_rope_head_dim", 0)

    @property
    def num_key_value_heads(self) -> int:
        return self.hf_config.get("num_key_value_heads", 0)

    @property
    def head_dim(self) -> int:
        return self.hf_config.get("head_dim", 0)

    @property
    def num_hidden_layers(self) -> int:
        return self.hf_config.get("num_hidden_layers", 0)
