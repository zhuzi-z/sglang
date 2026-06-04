import numpy as np
from sglang_simulator.simulation.types import RequestStats, SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.aiconfigurator import get_perf_model


def calc_kv_cache_cell_elems(model_info: ModelInfo, tp_size: int, pp_size: int) -> int:
    """Per-token KV cache element count across ALL full-KV layers (this PP rank).
    Hybrid-SSM models count only `full_attention` layers (mamba layers carry a
    fixed per-request state, not per-token)."""
    num_kv_layers = model_info.num_kv_layers(pp_size)
    if model_info.is_mla():
        return (model_info.kv_lora_rank + model_info.qk_rope_head_dim) * num_kv_layers
    num_kv_heads = max(model_info.num_key_value_heads // tp_size, 1)
    return num_kv_heads * model_info.head_dim * num_kv_layers * 2


def calc_kv_cache_per_layer_elems(
    model_info: ModelInfo, tp_size: int, pp_size: int
) -> int:
    if model_info.is_mla():
        return model_info.kv_lora_rank + model_info.qk_rope_head_dim
    num_kv_heads = max(model_info.num_key_value_heads // tp_size, 1)
    return num_kv_heads * model_info.head_dim * 2


def estimate_kv_cache_pool_capacity(
    model: ModelInfo, device: AcceleratorInfo, scheduler_config: SchedulerConfig
) -> int:
    """Available token slots in the GPU KV cache pool.

    Memory split for the 3 model families:
      * Dense MHA / MLA: full post-weights HBM goes to KV.
      * Hybrid SSM (Qwen3.5 / Qwen3-Next): post-weights HBM is split between
        mamba state and full KV per `mamba_full_memory_ratio`
        (mamba = R/(1+R), KV = 1/(1+R)). The chosen mamba pool size is also
        written back to `scheduler_config.max_mamba_cache_size` so
        HybridReqToTokenPool can pick it up downstream.
      * DSv4 / hybrid-SWA: not handled here — that path bypasses this function
        and uses `_resolve_memory_pool_config` directly from sglang.
    """
    perf_model = get_perf_model(scheduler_config, model)
    weights = 0
    for op in perf_model.context_ops:
        weights += op.get_weights()
    # Count weights on a single GPU
    weights /= perf_model.config.pp_size
    framework_reserved_mem_gb = 1.4
    rest_memory = (
        scheduler_config.mem_fraction_static * device.hbm_capacity_gb
        - framework_reserved_mem_gb
    ) * (1 << 30) - weights

    # Reserve mamba state pool for hybrid-SSM models BEFORE computing KV slots.
    if (
        getattr(model, "is_hybrid_ssm", False)
        and getattr(model, "mamba_bytes_per_req", 0)
    ):
        ratio = float(getattr(scheduler_config, "mamba_full_memory_ratio", 0.9))
        mamba_state_mem = rest_memory * ratio / (1.0 + ratio)
        rest_memory -= mamba_state_mem
        # Expose the chosen mamba pool size so HybridReqToTokenPool can use it.
        scheduler_config.max_mamba_cache_size = max(
            int(mamba_state_mem // model.mamba_bytes_per_req), 1
        )

    kv_cache_space_per_token = (
        calc_kv_cache_cell_elems(
            model, scheduler_config.tp_size, scheduler_config.pp_size
        )
        * scheduler_config.kv_cache_data_type.bytes
    )
    return int(rest_memory / kv_cache_space_per_token)


def calc_metrics(requests: list[RequestStats]) -> dict:
    ttfts = []
    tpots = []
    itls = []
    e2e_latencies = []
    total_dur_s = 1e-9
    total_input = 0
    total_output = 0
    completed = 0
    total_reused_tokens = 0
    total_device_hit_tokens = 0
    total_host_hit_tokens = 0
    total_storage_hit_tokens = 0
    queue_durs = []
    # TTFT compensation: sim under-estimates real client-side TTFT due to:
    #   (a) gtl[0] is just first-iter compute time, missing HTTP/SSE, scheduler
    #       overhead, kernel launch tail, remaining prefill chunks
    #   (b) per-iter compute is ~80% of real, so queue wait estimate is also
    #       ~25% short (queue grows linearly with per-iter time)
    # Two-term compensation, tunable via env:
    #   ttft = gtl[0] + base + queue_scale * queue_dur
    # Defaults calibrated from Qwen3.5-9B no_cache/l1/l2 fit (base=200ms, queue_scale=0.241).
    import os as _os
    _first_token_comp_s = float(_os.environ.get("SIM_FIRST_TOKEN_COMPENSATION_MS", "200")) / 1000.0
    _queue_scale = float(_os.environ.get("SIM_TTFT_QUEUE_SCALE", "0.241"))
    for req in requests:
        if not req.is_complete():
            continue
        completed += 1
        _q = req.queue_end - req.queue_start
        ttfts.append(req.gen_token_latencies[0] + _first_token_comp_s + _queue_scale * _q)
        queue_durs.append(_q)
        if len(req.gen_token_latencies) > 1:
            # output length > 1
            tpots.append(np.mean(req.gen_token_latencies[1:]))
        itls.extend(req.gen_token_latencies[1:])
        e2e_latencies.append(sum(req.gen_token_latencies))
        total_dur_s = max(total_dur_s, req.last_event_time)
        total_input += req.input_length
        total_output += req.output_length
        total_reused_tokens += req.final_device_hit_len + req.final_storage_hit_len
        total_device_hit_tokens += max(0, req.final_device_hit_len - max(req.final_host_hit_len, req.final_storage_hit_len))
        total_host_hit_tokens += max(0, req.final_host_hit_len - req.final_storage_hit_len)

        total_storage_hit_tokens += req.final_storage_hit_len
    return {
        "num_requests": len(requests),
        "completed": completed,
        "total_input": total_input,
        "total_output": total_output,
        "duration": total_dur_s,
        "request_throughput": len(requests) / total_dur_s,
        "input_throughput": total_input / total_dur_s,
        "output_throughput": total_output / total_dur_s,
        "total_throughput": (total_input + total_output) / total_dur_s,
        "prefix_cache_reused_ratio": (
            0 if total_input == 0 else total_reused_tokens / total_input
        ),
        "kv_cache_storage_hit_ratio": (
            0 if total_input == 0 else total_storage_hit_tokens / total_input
        ),
        "kv_cache_host_hit_ratio": (
            0 if total_input == 0 else total_host_hit_tokens / total_input
        ),
        "kv_cache_device_hit_ratio": (
            0 if total_input == 0 else total_device_hit_tokens / total_input
        ),
        "mean_ttft_ms": np.mean(ttfts or 0) * 1000,
        "median_ttft_ms": np.median(ttfts or 0) * 1000,
        "std_ttft_ms": np.std(ttfts or 0) * 1000,
        "p90_ttft_ms": np.percentile(ttfts or 0, 90) * 1000,
        "p95_ttft_ms": np.percentile(ttfts or 0, 95) * 1000,
        "p99_ttft_ms": np.percentile(ttfts or 0, 99) * 1000,
        "mean_queue_ms": np.mean(queue_durs or 0) * 1000,
        "mean_tpot_ms": np.mean(tpots or 0) * 1000,
        "median_tpot_ms": np.median(tpots or 0) * 1000,
        "std_tpot_ms": np.std(tpots or 0) * 1000,
        "p90_tpot_ms": np.percentile(tpots or 0, 90) * 1000,
        "p95_tpot_ms": np.percentile(tpots or 0, 95) * 1000,
        "p99_tpot_ms": np.percentile(tpots or 0, 99) * 1000,
        "mean_itl_ms": np.mean(itls or 0) * 1000,
        "median_itl_ms": np.median(itls or 0) * 1000,
        "std_itl_ms": np.std(itls or 0) * 1000,
        "p90_itl_ms": np.percentile(itls or 0, 90) * 1000,
        "p95_itl_ms": np.percentile(itls or 0, 95) * 1000,
        "p99_itl_ms": np.percentile(itls or 0, 99) * 1000,
        "max_itl_ms": np.max(itls or 0) * 1000,
        "mean_e2e_latency_ms": np.mean(e2e_latencies) * 1000,
        "median_e2e_latency_ms": np.median(e2e_latencies) * 1000,
        "std_e2e_latency_ms": np.std(e2e_latencies) * 1000,
        "p90_e2e_latency_ms": np.percentile(e2e_latencies or 0, 90) * 1000,
        "p95_e2e_latency_ms": np.percentile(e2e_latencies or 0, 95) * 1000,
        "p99_e2e_latency_ms": np.percentile(e2e_latencies or 0, 99) * 1000,
        "time_cost": -1,  # Updated by external benchmark caller
    }
