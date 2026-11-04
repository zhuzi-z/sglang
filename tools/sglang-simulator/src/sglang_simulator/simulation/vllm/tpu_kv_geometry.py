"""Qwen GDN unified-pool 的纯 CPU 元数据几何，不导入 TPU 内核。

依据 vllm_torchtpu 的 kernels/gdn/head_geometry.py、gdn_pool_layout.py、
platforms/tpu_block_size_utils.py。这里只描述一个目标 TP/PCP rank 的页，
不把 PP stage 数乘到内存预算，也不修改 CPU 引擎的执行拓扑或 token 粒度。
"""

import dataclasses
import math
import os

import torch


_QWEN_GDN_ARCHITECTURES = frozenset({
    "Qwen3NextForCausalLM", "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
})


def is_tpu_target(raw_config):
    """由模拟目标选择适配器，不使用已被 CPU bootstrap 清除的设备变量。"""
    accelerator = raw_config.get("platform", {}).get("accelerator", {})
    return str(accelerator.get("name", "")).lower().startswith("tpu")


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _local_heads(key_heads, value_heads, parallel_size):
    # 与原生 GDN 一致：V 完整分片；Q/K head 不够时复制，不能把总字节除以 PCP。
    unique_shards = min(key_heads, parallel_size)
    if (value_heads % parallel_size or key_heads % unique_shards
            or parallel_size % unique_shards):
        raise ValueError(
            f"Invalid TPU GDN sharding: key_heads={key_heads}, "
            f"value_heads={value_heads}, parallel_size={parallel_size}"
        )
    local_key, local_value = key_heads // unique_shards, value_heads // parallel_size
    if local_value % local_key:
        raise ValueError("Each local Q/K head must own an integer number of V heads")
    return local_key, local_value


def _dtype(value):
    if isinstance(value, torch.dtype):
        return value
    aliases = {"fp8": "float8_e4m3fn", "fp8_e4m3": "float8_e4m3fn",
               "fp8_e5m2": "float8_e5m2", "half": "float16"}
    name = str(value).removeprefix("torch.")
    result = getattr(torch, aliases.get(name, name), None)
    if not isinstance(result, torch.dtype):
        raise ValueError(f"Unsupported TPU state/cache dtype: {value!r}")
    return result


@dataclasses.dataclass(frozen=True)
class TPUPooledPage:
    shapes: tuple
    dtypes: tuple
    tp_size: int
    pcp_size: int
    token_bytes: int
    kv_dtype: torch.dtype
    raw_state_bytes: int
    required_state_bytes: int
    slot_bytes: int
    full_state_bytes: int


def derive_tpu_pooled_page(vllm_config, raw_config, shapes, dtypes):
    """把 CPU 模型元数据转换到目标 rank，保留 Conv taps/MTP 扩展长度。

    当前覆盖 Qwen GDN 的 unified pool。对未核验的 attention packing
    明确拒绝，不借用其他模型/芯片的数值，也不以增大 block_size 绕过不变量。
    """
    model = vllm_config.model_config
    cache = vllm_config.cache_config
    hf = model.hf_text_config
    scheduler = raw_config.get("scheduler", {})
    if model.architecture not in _QWEN_GDN_ARCHITECTURES:
        raise ValueError(f"TPU pooled GDN geometry is not adapted for {model.architecture}")
    override = os.environ.get("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL")
    if override is not None and override.lower() not in ("1", "true"):
        raise ValueError("TPU non-unified KV layout requires a separate geometry adapter")
    if (getattr(cache, "attn_pack_size", 1) or 1) != 1:
        raise ValueError("TPU pooled pages cannot reuse the GPU attn_pack_size rule")
    # 模型类在 CPU 配置下提供完整状态；不对已分片状态二次分片。
    if vllm_config.parallel_config.tensor_parallel_size != 1:
        raise ValueError("TPU geometry adapter expects the physical CPU TP size to be 1")

    tp = _positive_int(scheduler.get("tp_size", 1), "scheduler.tp_size")
    pcp = _positive_int(scheduler.get("pcp_size", 1), "scheduler.pcp_size")
    key_heads = _positive_int(hf.linear_num_key_heads, "linear_num_key_heads")
    value_heads = _positive_int(hf.linear_num_value_heads, "linear_num_value_heads")
    key_dim = _positive_int(hf.linear_key_head_dim, "linear_key_head_dim")
    value_dim = _positive_int(hf.linear_value_head_dim, "linear_value_head_dim")
    if len(shapes) != 2 or len(dtypes) != 2:
        raise ValueError("TPU GDN requires exactly two state shapes/dtypes (conv, SSM)")
    conv, ssm = tuple(shapes[0]), tuple(shapes[1])
    for shape in (conv, ssm):
        for dimension in shape:
            _positive_int(dimension, "state dimension")
    full_width = 2 * key_heads * key_dim + value_heads * value_dim
    if len(conv) not in (2, 3) or len(ssm) != 3 or ssm[0] != value_heads:
        raise ValueError(f"Unexpected full GDN state shapes: {shapes}")
    if math.prod(ssm[1:]) != key_dim * value_dim:
        raise ValueError(f"GDN SSM dimensions disagree with model heads: {ssm}")
    # SD/DS 的通道轴由全量 head 宽度唯一确定，只转换元数据、不搬运张量。
    axes = [i for i, size in enumerate(conv) if size == full_width]
    if len(axes) != 1:
        raise ValueError(f"Cannot identify GDN conv channel axis: {conv}, width={full_width}")
    taps = math.prod(conv) // full_width
    local_key, local_value = _local_heads(key_heads, value_heads, tp)
    local_key, local_value = _local_heads(local_key, local_value, pcp)
    local_width = 2 * local_key * key_dim + local_value * value_dim
    local_shapes = ((taps, 1, local_width), (local_value, *ssm[1:]))

    # unified pool 的 Conv 固定 BF16，SSM 保留模型声明（不能被 fp8 KV 覆盖）。
    local_dtypes = (torch.bfloat16, _dtype(dtypes[1]))
    if local_dtypes[1] not in (torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported pooled SSM dtype: {local_dtypes[1]}")
    conv_bytes, ssm_bytes = (
        math.prod(shape) * dtype.itemsize
        for shape, dtype in zip(local_shapes, local_dtypes)
    )

    total_kv_heads = _positive_int(model.get_total_num_kv_heads(), "total KV heads")
    if total_kv_heads % tp and tp % total_kv_heads:
        raise ValueError("Attention KV heads are incompatible with target TP")
    kv_heads = max(total_kv_heads // tp, 1)
    head_size = _positive_int(model.get_head_size(), "attention head_size")
    cache_dtype = model.dtype if cache.cache_dtype == "auto" else cache.cache_dtype
    cache_dtype = _dtype(cache_dtype)
    if cache_dtype not in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"Unsupported TPU attention KV dtype: {cache_dtype}")
    packing = 4 // cache_dtype.itemsize
    # 此子集在 batched RPA 的 NHD/HND 下具有相同页字节数；需要额外 lane/head
    # padding 的形状必须另行适配，不能按通用 FullAttentionSpec 猜测物理页。
    if head_size % 128 or (2 * kv_heads) % packing:
        raise ValueError(
            f"TPU attention packing needs an explicit adapter: "
            f"head_size={head_size}, kv_heads={kv_heads}, packing={packing}"
        )
    token_bytes = 2 * kv_heads * head_size * cache_dtype.itemsize
    block_size = _positive_int(cache.block_size, "physical block_size")
    if block_size % 128:
        raise ValueError("TPU batched-RPA physical block_size must be a multiple of 128")
    ssm_rows = (ssm_bytes + token_bytes - 1) // token_bytes
    conv_rows = 1
    while conv_rows * token_bytes < conv_bytes:
        conv_rows *= 2
    required = (ssm_rows + conv_rows) * token_bytes
    slot_bytes = block_size * token_bytes
    if required > slot_bytes:
        raise ValueError(
            f"TPU pooled state does not fit the configured physical block: "
            f"tp={tp}, pcp={pcp}, shapes={local_shapes}, dtypes={local_dtypes}, "
            f"raw_state_bytes={conv_bytes + ssm_bytes}, required_state_bytes={required}, "
            f"slot_bytes={slot_bytes}, block_size={block_size}; "
            "verify target topology/dtypes/block_size against the native TPU deployment"
        )
    return TPUPooledPage(
        shapes=local_shapes, dtypes=local_dtypes, tp_size=tp, pcp_size=pcp,
        token_bytes=token_bytes, kv_dtype=cache_dtype,
        raw_state_bytes=conv_bytes + ssm_bytes,
        required_state_bytes=required, slot_bytes=slot_bytes,
        full_state_bytes=sum(math.prod(s) * _dtype(d).itemsize for s, d in zip(shapes, dtypes)),
    )


def normalize_tpu_pooled_specs(full_spec, mamba_spec, page):
    """同时规范化两种 spec，保留原生布局、预算和块数校验。"""
    fa_fields = {f.name for f in dataclasses.fields(full_spec)}
    if not {"page_size_padded", "num_head_slots", "state_content_bytes"} <= fa_fields:
        raise ValueError("This vLLM version lacks the TPU physical-page spec fields")
    full_spec = dataclasses.replace(
        full_spec, dtype=page.kv_dtype, page_size_padded=page.slot_bytes,
        num_head_slots=1, state_content_bytes=page.token_bytes,
    )
    mamba_spec = dataclasses.replace(
        mamba_spec, shapes=page.shapes, dtypes=page.dtypes,
        page_size_padded=page.slot_bytes,
    )
    # 主动触发不变量，让错误发生在构造处而不是后续 layout resolver 中。
    if full_spec.page_size_bytes != page.slot_bytes or mamba_spec.page_size_bytes != page.slot_bytes:
        raise ValueError("TPU attention/Mamba specs disagree on the physical slot size")
    return full_spec, mamba_spec
