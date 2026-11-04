"""将 TPU PP/PCP 的虚拟设备容量折叠到单个 CPU 调度器。

虚拟 stage 的分组、预算校验和共同 block 数全部交给原生 vLLM；此处
只做层归属与 token 单位映射，不启动虚拟 worker，也不分配 KV 张量。
"""

import copy
import dataclasses
import json
import os
import re

from sglang_simulator.utils import get_logger
from sglang_simulator.simulation.vllm.tpu_kv_geometry import _positive_int

logger = get_logger()


@dataclasses.dataclass(frozen=True)
class TPUCapacityPlan:
    specs: dict
    stage_layers: tuple
    stage_budgets: tuple
    stage_block_limits: tuple
    num_blocks: int
    physical_block_size: int
    logical_block_size: int
    hash_block_size: int
    virtual_memory_bytes: int
    layout_name: str


def _parse_partition(scheduler, num_layers, stages):
    value = scheduler.get("pp_layer_partition")
    if value is None:
        value = os.environ.get("SGLANG_SIMULATOR_PP_LAYER_PARTITION")
    if value is None:
        if stages == 1:
            return (num_layers,)
        raise ValueError(
            "PP capacity planning requires scheduler.pp_layer_partition or "
            "SGLANG_SIMULATOR_PP_LAYER_PARTITION; supply the native stage layer counts"
        )
    if isinstance(value, str):
        value = json.loads(value) if value.strip().startswith("[") else [
            int(part.strip()) for part in value.split(",")
        ]
    if not isinstance(value, (list, tuple)):
        raise ValueError("pp_layer_partition must be a list or comma-separated layer counts")
    partition = tuple(_positive_int(n, "PP stage layers") for n in value)
    if len(partition) != stages or sum(partition) != num_layers:
        raise ValueError(
            f"Invalid pp_layer_partition={partition}: expected {stages} stages "
            f"covering exactly {num_layers} model layers"
        )
    return partition


def _project_layers(specs, partition):
    owners = [stage for stage, count in enumerate(partition) for _ in range(count)]
    projected = [{} for _ in partition]
    for name, spec in specs.items():
        match = re.match(r"^model\.layers\.(\d+)(?:\.|$)", name)
        if match:
            layer = int(match.group(1))
            if layer >= len(owners):
                raise ValueError(f"KV layer lies outside PP partition: {name}")
            stage = owners[layer]
        elif name.startswith(("model.mtp.layers.", "mtp.")):
            # MTP 属于最后 stage，不按全模型层数平均分摊。
            stage = len(partition) - 1
        else:
            raise ValueError(f"Cannot determine PP stage ownership for KV layer {name}")
        projected[stage][name] = spec
    if any(not stage_specs for stage_specs in projected):
        raise ValueError("Every virtual PP stage must own at least one KV layer")
    return projected


def _logical_specs(physical_specs, physical_block, pcp):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    result = {}
    for name, spec in physical_specs.items():
        if spec.block_size != physical_block:
            raise ValueError(f"Mixed physical block sizes are not supported: {name}")
        if isinstance(spec, FullAttentionSpec):
            # 一份 rank-local 页覆盖 PCP 个 rank 的逻辑 token。状态数/页字节
            # 不变，只把每个物理 token 对应的逻辑跨度乘 PCP，避免 padded 失效。
            logical = dataclasses.replace(
                spec, block_size=physical_block * pcp,
                tokens_per_state=spec.tokens_per_state * pcp,
            )
        elif isinstance(spec, MambaSpec):
            logical = dataclasses.replace(spec, block_size=physical_block * pcp)
        else:
            raise ValueError(f"Unsupported TPU pooled spec type: {type(spec).__name__}")
        if logical.page_size_bytes != spec.page_size_bytes:
            raise ValueError(f"PCP folding changed physical page bytes for {name}")
        result[name] = logical
    return result


def plan_tpu_kv_capacity(vllm_config, raw_config, physical_specs):
    """保留每设备预算，投影全部 stages；失败时原生容量错误直接上抛。"""
    from vllm.v1.attention.backends.utils import (
        get_supported_kv_cache_layouts, resolve_kv_cache_layout,
    )
    from vllm.v1.core.kv_cache_utils import (
        _pool_bytes_per_block, _project_kv_cache_groups_to_worker,
        get_kv_cache_groups, get_kv_cache_configs, resolve_kv_cache_block_sizes,
    )
    from sglang_simulator.simulation.vllm.attention_backend import SimulatedAttentionBackend

    scheduler = raw_config.get("scheduler", {})
    stages = _positive_int(scheduler.get("pp_size", 1), "scheduler.pp_size")
    pcp = _positive_int(scheduler.get("pcp_size", 1), "scheduler.pcp_size")
    physical_block = _positive_int(scheduler.get("physical_block_size"), "scheduler.physical_block_size")
    if vllm_config.cache_config.block_size != physical_block:
        raise ValueError("Input KV specs/config must still use the physical block size")
    partition = _parse_partition(
        scheduler, vllm_config.model_config.hf_text_config.num_hidden_layers, stages
    )
    budgets = scheduler.get("pp_stage_kv_cache_memory_bytes")
    if budgets is None:
        per_device = _positive_int(
            vllm_config.cache_config.kv_cache_memory_bytes, "kv_cache_memory_bytes"
        )
        budgets = [per_device] * stages
    if not isinstance(budgets, (list, tuple)) or len(budgets) != stages:
        raise ValueError("pp_stage_kv_cache_memory_bytes must give one per-device budget per stage")
    budgets = tuple(_positive_int(b, "stage KV budget") for b in budgets)
    parallel = vllm_config.parallel_config
    if any(getattr(parallel, key, 1) != 1 for key in (
        "tensor_parallel_size", "pipeline_parallel_size",
        "prefill_context_parallel_size", "decode_context_parallel_size",
    )):
        raise ValueError("TPU capacity folding requires single-worker physical parallelism")

    # 使用独立配置视图，原生规划过程不得改变实际引擎的预算/拓扑。
    planning = copy.deepcopy(vllm_config)
    logical_block = physical_block * pcp
    planning.cache_config.block_size = logical_block
    planning.cache_config.mamba_block_size = logical_block
    override = planning.cache_config.num_gpu_blocks_override
    planning.cache_config.num_gpu_blocks_override = None
    if getattr(planning.model_config, "original_max_model_len", None) == -1:
        raise ValueError("TPU capacity folding requires an explicit max_model_len")
    logical_specs = _logical_specs(physical_specs, physical_block, pcp)
    stage_specs = _project_layers(logical_specs, partition)
    from vllm.config import set_current_vllm_config

    # 原生辅助函数可能读取当前 config；只在本段暂存独立规划视图。
    with set_current_vllm_config(planning):
        layout_names = [layout.name for layout in get_supported_kv_cache_layouts([SimulatedAttentionBackend])]
        layout = resolve_kv_cache_layout(planning, [layout_names], logical_specs.values())
        groups = get_kv_cache_groups(planning, dict(logical_specs))
        projected_groups = [_project_kv_cache_groups_to_worker(groups, s) for s in stage_specs]
        stage_limits = tuple(b // _pool_bytes_per_block(g) for b, g in zip(budgets, projected_groups))

        # get_kv_cache_configs 原生执行逐 stage 最长请求检查、null block 预留，
        # 再把 block 数统一到瓶颈 stage；PCP 已由 logical spec 表达，不能再乘8。
        configs = get_kv_cache_configs(planning, stage_specs, list(budgets))
        native_limit = min(c.num_blocks for c in configs)
        if override is not None:
            _positive_int(override, "num_gpu_blocks_override")
            if override > native_limit:
                raise ValueError(
                    f"num_gpu_blocks_override={override} exceeds virtual stage capacity={native_limit}"
                )
            planning.cache_config.num_gpu_blocks_override = override
            # 降低容量也必须重新验证最长请求，不能通过 override 跳过检查。
            configs = get_kv_cache_configs(planning, stage_specs, list(budgets))
        num_blocks = min(c.num_blocks for c in configs)
        virtual_bytes = num_blocks * _pool_bytes_per_block(groups)

        # 单 CPU 持有全模型 metadata，以同样的 block 个数构造等价记账预算。
        # 不是把真实内存预算随意乘 PP/PCP；两侧都仍经过原生请求容量校验。
        cpu_config = get_kv_cache_configs(planning, [logical_specs], [virtual_bytes])[0]
        if cpu_config.num_blocks != num_blocks:
            raise ValueError("CPU equivalent cache capacity differs from virtual TPU stages")
        scheduler_block, hash_block = resolve_kv_cache_block_sizes(cpu_config, planning)
        if scheduler_block != logical_block:
            raise ValueError("CPU scheduler block size differs from PCP logical block size")

    plan = TPUCapacityPlan(
        specs=logical_specs, stage_layers=partition, stage_budgets=budgets,
        stage_block_limits=stage_limits, num_blocks=num_blocks,
        physical_block_size=physical_block, logical_block_size=logical_block,
        hash_block_size=hash_block, virtual_memory_bytes=virtual_bytes, layout_name=layout.name,
    )
    logger.info(
        "[TPU KV capacity] stage_layers=%s per_device_budgets=%s stage_block_limits=%s "
        "common_num_blocks=%d physical_block_size=%d logical_block_size=%d "
        "hash_block_size=%d cpu_metadata_budget_bytes=%d",
        partition, budgets, stage_limits, num_blocks, physical_block, logical_block,
        hash_block, virtual_bytes,
    )
    return plan


def apply_tpu_kv_capacity(vllm_config, plan):
    """在 Scheduler、hasher 和 connector 创建前一次性发布 CPU 逻辑单位。"""
    cache = vllm_config.cache_config
    cache.block_size = plan.logical_block_size
    cache.mamba_block_size = plan.logical_block_size
    cache.kv_cache_layout = plan.layout_name
    cache.prefix_match_unit = plan.hash_block_size
    # kv_cache_memory_bytes 保留每设备的配置值；Worker 返回的是 plan 的
    # cpu_metadata_budget_bytes。不能把一次折叠的结果当成下一次的物理输入。
