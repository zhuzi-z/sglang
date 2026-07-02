"""
vLLM utility functions for resolving model info and scheduler config
from vLLM's internal configuration objects.

Similar to sglang_simulator.simulation.sglang.utils but adapted for vLLM.
"""

import typing

from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec import DataType, ModelInfo

if typing.TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.model import ModelConfig


def resolve_model_info(model_config: "ModelConfig") -> ModelInfo:
    """Convert vLLM's ModelConfig to simulator ModelInfo."""
    hf_config = model_config.hf_text_config
    head_dim = model_config.get_head_size()
    torch_dtype = str(model_config.dtype).replace("torch.", "")

    kwargs = dict(
        hf_config=hf_config,
        model_path=model_config.model,
        attention_arch="MLA" if model_config.is_deepseek_mla else "MHA",
        context_len=model_config.max_model_len,
        hidden_size=model_config.get_hidden_size(),
        head_dim=head_dim,
        num_attention_heads=(
            model_config.model_arch_config.total_num_attention_heads
            if hasattr(model_config, "model_arch_config")
            else hf_config.num_attention_heads
        ),
        num_hidden_layers=model_config.get_total_num_hidden_layers(),
        num_key_value_heads=model_config.get_total_num_kv_heads(),
        v_head_dim=getattr(hf_config, "v_head_dim", head_dim),
        vocab_size=model_config.get_vocab_size(),
        torch_dtype=torch_dtype,
    )
    if model_config.is_deepseek_mla:
        kwargs.update(
            qk_rope_head_dim=getattr(hf_config, "qk_rope_head_dim", None),
            qk_nope_head_dim=getattr(hf_config, "qk_nope_head_dim", None),
            kv_lora_rank=getattr(hf_config, "kv_lora_rank", None),
        )
    return ModelInfo(**kwargs)


def resolve_scheduler_config(vllm_config: "VllmConfig") -> SchedulerConfig:
    """Create simulator SchedulerConfig from vLLM's VllmConfig."""
    import vllm

    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config

    torch_dtype = str(model_config.dtype).replace("torch.", "")
    data_type = DataType.from_torch_dtype(torch_dtype)

    # KV cache dtype
    cache_dtype = cache_config.cache_dtype
    if cache_dtype == "auto":
        kv_cache_data_type = data_type
    else:
        kv_cache_data_type = DataType.from_torch_dtype(cache_dtype) or data_type

    return SchedulerConfig(
        data_type=data_type,
        kv_cache_data_type=kv_cache_data_type,
        mem_fraction_static=cache_config.gpu_memory_utilization,
        tp_size=parallel_config.tensor_parallel_size,
        pp_size=parallel_config.pipeline_parallel_size,
        dp_size=getattr(parallel_config, "data_parallel_size", 1),
        backend_name="vllm",
        backend_version=vllm.__version__,
    )
