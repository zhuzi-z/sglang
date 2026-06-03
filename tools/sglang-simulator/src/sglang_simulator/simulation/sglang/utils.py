import typing

from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec import DataType, ModelInfo
from sglang_simulator.utils import get_logger

if typing.TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = get_logger("sgl_simulator")


def resolve_scheduler_config(
    server_args: "ServerArgs",
) -> SchedulerConfig:
    from sglang.version import __version__

    dtype = server_args.dtype
    if dtype == "auto":
        dtype = str(server_args.model_config.dtype).strip("torch.")
    data_type = DataType.from_torch_dtype(dtype)
    # Pick up mamba_full_memory_ratio from server_args when sglang exposes it,
    # otherwise fall back to the SchedulerConfig default (0.9).
    mamba_ratio = getattr(server_args, "mamba_full_memory_ratio", None)
    cfg = SchedulerConfig(
        data_type=data_type,
        kv_cache_data_type=DataType.from_torch_dtype(server_args.kv_cache_dtype)
        or data_type,
        mem_fraction_static=server_args.mem_fraction_static,
        max_total_tokens=server_args.max_total_tokens,
        tp_size=server_args.tp_size,
        ep_size=server_args.ep_size,
        dp_size=server_args.dp_size,
        pp_size=server_args.pp_size,
        backend_name="sglang",
        backend_version=__version__,
    )
    if mamba_ratio is not None:
        cfg.mamba_full_memory_ratio = float(mamba_ratio)
    return cfg


def _resolve_hybrid_ssm_fields(model_config: "ModelConfig") -> dict:
    """Detect hybrid-SSM models (Qwen3.5 / Qwen3-Next / GraniteMoeHybrid / ...)
    and return the extra ModelInfo fields needed for capacity estimation.

    Returns an empty dict for non-hybrid (dense / pure MLA) models.

    Detection order (sglang exposes these in different ways across configs):
      1. `full_attention_layer_ids` / `linear_layer_ids` properties on the text
         config (set by Qwen3NextConfig / Qwen3_5TextConfig and friends).
      2. `layers_block_type` (sglang's internal `HybridLayerType` enum names —
         "attention" for full, "linear_attention" for mamba).
      3. Raw HF `layer_types` attribute ("full_attention" / "linear_attention").
    """
    text_cfg = getattr(model_config, "hf_text_config", None) or getattr(
        model_config, "hf_config", None
    )
    if text_cfg is None:
        return {}

    full_ids = getattr(text_cfg, "full_attention_layer_ids", None)
    mamba_ids = getattr(text_cfg, "linear_layer_ids", None)
    if full_ids is not None and mamba_ids is not None and len(list(mamba_ids)) > 0:
        full_ids = list(full_ids)
        mamba_ids = list(mamba_ids)
    else:
        # Fallback: parse layer_types / layers_block_type. Both "attention"
        # (sglang internal) and "full_attention" (HF raw) map to full KV.
        layer_types = (
            getattr(text_cfg, "layer_types", None)
            or getattr(text_cfg, "layers_block_type", None)
        )
        if not layer_types:
            return {}
        layer_types = list(layer_types)
        if "linear_attention" not in layer_types:
            return {}
        full_ids = [
            i for i, t in enumerate(layer_types) if t in ("full_attention", "attention")
        ]
        mamba_ids = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
        if not mamba_ids:
            return {}

    # Per-request mamba state bytes — same formula as MambaPoolHost uses for
    # its host buffer. Reads the per-layer mamba dims from the text config.
    mamba_bytes = 0
    try:
        # temporal state: (num_value_heads, head_dim) tensor of state_size each
        # Actually shape is determined by Mamba2StateShape.create(); per-token
        # element count = num_value_heads * head_dim * state_size (where
        # state_size = linear_key_head_dim). dtype is fp32 by default for state.
        temporal_elem = (
            int(text_cfg.linear_num_value_heads)
            * int(text_cfg.linear_value_head_dim)
            * int(text_cfg.linear_key_head_dim)
        )
        # conv state shape: (intermediate_size + 2*n_groups*state_size,
        #                    conv_kernel - 1). We approximate as the same
        # geometric prefix the host pool uses. dtype matches model dtype (bf16/fp16).
        intermediate = int(text_cfg.linear_num_value_heads) * int(text_cfg.linear_value_head_dim)
        n_groups = int(text_cfg.linear_num_key_heads)
        state_size = int(text_cfg.linear_key_head_dim)
        conv_kernel = int(text_cfg.linear_conv_kernel_dim)
        conv_elem = (intermediate + 2 * n_groups * state_size) * (conv_kernel - 1)
        # Bytes per layer: temporal fp32 + conv bf16 (model_dtype)
        # We treat both as bf16/fp32 conservatively per sglang's defaults.
        per_layer_bytes = temporal_elem * 4 + conv_elem * 2
        mamba_bytes = per_layer_bytes * len(mamba_ids)
    except (AttributeError, TypeError, ValueError) as e:
        logger.warning(f"Failed to compute mamba_bytes_per_req: {e}; using 0")
        mamba_bytes = 0

    return dict(
        is_hybrid_ssm=True,
        num_full_attention_layers=len(full_ids),
        num_mamba_layers=len(mamba_ids),
        mamba_bytes_per_req=mamba_bytes,
        full_attention_layer_ids=full_ids,
        mamba_layer_ids=mamba_ids,
    )


def resolve_model_info(model_config: "ModelConfig") -> ModelInfo:
    from sglang.srt.configs.model_config import AttentionArch

    torch_dtype = str(model_config.dtype).strip("torch.")
    hybrid_fields = _resolve_hybrid_ssm_fields(model_config)

    if model_config.attention_arch == AttentionArch.MHA:
        return ModelInfo(
            hf_config=model_config.hf_text_config,
            model_path=model_config.model_path,
            attention_arch="MHA",
            context_len=model_config.context_len,
            hidden_size=model_config.hidden_size,
            head_dim=model_config.head_dim,
            num_attention_heads=model_config.num_attention_heads,
            num_hidden_layers=model_config.num_hidden_layers,
            num_key_value_heads=model_config.num_key_value_heads,
            v_head_dim=model_config.v_head_dim,
            vocab_size=model_config.vocab_size,
            torch_dtype=torch_dtype,
            **hybrid_fields,
        )
    elif model_config.attention_arch == AttentionArch.MLA:
        return ModelInfo(
            hf_config=model_config.hf_text_config,
            model_path=model_config.model_path,
            attention_arch="MLA",
            context_len=model_config.context_len,
            hidden_size=model_config.hidden_size,
            head_dim=model_config.head_dim,
            num_attention_heads=model_config.num_attention_heads,
            num_hidden_layers=model_config.num_hidden_layers,
            num_key_value_heads=model_config.num_key_value_heads,
            v_head_dim=model_config.v_head_dim,
            vocab_size=model_config.vocab_size,
            qk_rope_head_dim=model_config.qk_rope_head_dim,
            qk_nope_head_dim=model_config.qk_nope_head_dim,
            kv_lora_rank=model_config.kv_lora_rank,
            torch_dtype=torch_dtype,
            **hybrid_fields,
        )
    else:
        raise ValueError(
            f"The attention type of `{model_config.attention_arch}` is not supported now."
        )
