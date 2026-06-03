from dataclasses import dataclass
from typing import List, Optional

from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


@dataclass
class ModelInfo:
    hf_config: Optional[dict] = None
    model_path: Optional[str] = None

    attention_arch: Optional[str] = None  # MLA | MHA
    context_len: Optional[int] = None
    hidden_size: Optional[int] = None
    head_dim: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    v_head_dim: Optional[int] = None
    vocab_size: Optional[int] = None

    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None

    torch_dtype: Optional[str] = None

    # deepseek v4 model config
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    indexer_head_dim: Optional[int] = None

    # Hybrid SSM fields (Qwen3.5 / Qwen3-Next / ...).
    # When `is_hybrid_ssm=True`, `num_hidden_layers` still reports the total layer
    # count, but only `num_full_attention_layers` of them carry a per-token KV
    # cache; the remaining `num_mamba_layers` use mamba state (a fixed per-request
    # blob of `mamba_bytes_per_req` bytes, counted across ALL mamba layers).
    is_hybrid_ssm: bool = False
    num_full_attention_layers: Optional[int] = None
    num_mamba_layers: Optional[int] = None
    mamba_bytes_per_req: Optional[int] = None
    full_attention_layer_ids: Optional[List[int]] = None
    mamba_layer_ids: Optional[List[int]] = None

    def is_mla(self) -> bool:
        return self.attention_arch == "MLA"

    def num_kv_layers(self, pp_size: int = 1) -> int:
        """Number of layers that carry per-token KV cache on this PP rank.
        For hybrid-SSM models this counts ONLY full_attention layers."""
        total = (
            self.num_full_attention_layers
            if (self.is_hybrid_ssm and self.num_full_attention_layers is not None)
            else (self.num_hidden_layers or 0)
        )
        return max(total // max(pp_size, 1), 0)
