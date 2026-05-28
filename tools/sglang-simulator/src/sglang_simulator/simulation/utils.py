import numpy as np
from sglang_simulator.simulation.types import RequestStats, SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.aiconfigurator import get_perf_model


def calc_kv_cache_cell_elems(model_info: ModelInfo, tp_size: int, pp_size: int) -> int:
    num_layers = model_info.num_hidden_layers // pp_size
    if model_info.is_mla():
        return (model_info.kv_lora_rank + model_info.qk_rope_head_dim) * num_layers
    else:
        num_kv_heads = max(model_info.num_key_value_heads // tp_size, 1)
        return num_kv_heads * model_info.head_dim * num_layers * 2


def calc_kv_cache_per_layer_elems(
    model_info: ModelInfo, tp_size: int, pp_size: int
) -> int:
    if model_info.is_mla():
        return model_info.kv_lora_rank + model_info.qk_rope_head_dim
    else:
        num_kv_heads = max(model_info.num_key_value_heads // tp_size, 1)
        return num_kv_heads * model_info.head_dim * 2


def estimate_kv_cache_pool_capacity(
    model: ModelInfo, device: AcceleratorInfo, scheduler_config: SchedulerConfig
) -> int:
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
    kv_cache_space_per_token = (
        calc_kv_cache_cell_elems(
            model, scheduler_config.tp_size, scheduler_config.pp_size
        )
        * scheduler_config.kv_cache_data_type.bytes
    )
    return int(rest_memory / kv_cache_space_per_token)


def kv_bytes_per_full_token_per_gpu(
    model_info: ModelInfo, scheduler_config: SchedulerConfig
) -> float:
    """Single-GPU (per TP shard) KV-cache bytes/token.

    Three-tier resolution (highest priority first):

      1. ``scheduler_config.kv_bytes_per_token_per_gpu`` — explicit override
         provided by the user (e.g. read from sglang server's
         "KV Cache is allocated. #tokens=N, KV size=G GB" log line). Use
         this when the sglang KV pool path isn't reachable from a static
         ModelConfig (e.g. GlmMoeDsa, where sglang's config_backup mechanism
         or env-driven calculator branching makes derivation unreliable).
      2. sglang ``DSv4MemoryCalculator`` — for DSv4 architectures (per-layer
         compression ratios + sparse indexer + SWA).
      3. ``calc_kv_cache_cell_elems × kv_cache_data_type.bytes`` — generic
         MHA/MLA formula.

    Matches sglang's reported ``bytes_per_full_token`` for DSv4 (which is
    per-GPU; ``pool_size = available_bytes_on_one_gpu / bytes_per_full_token``).
    """
    if scheduler_config.kv_bytes_per_token_per_gpu is not None:
        return scheduler_config.kv_bytes_per_token_per_gpu

    if model_info.is_dsv4():
        from sglang.srt.model_executor.memory_profiler import DSv4MemoryCalculator

        class _Cfg:
            qk_nope_head_dim = model_info.qk_nope_head_dim
            qk_rope_head_dim = model_info.qk_rope_head_dim
            index_head_dim = model_info.indexer_head_dim
            compress_ratios = model_info.compression_ratios
            window_size = model_info.window_size

        if scheduler_config.swa_full_tokens_ratio is None:
            raise ValueError(
                "DSv4 KV cache calculation requires scheduler_config.swa_full_tokens_ratio "
                "to be set explicitly. Add `swa_full_tokens_ratio` to your server_args.json "
                "(match the value of your real sglang server --swa-full-tokens-ratio). "
                "Inferring from defaults is unsafe: a wrong value silently breaks bytes_per_token "
                "by up to 8x (e.g. swa=0.8 vs real 0.1 -> 6.4x error)."
            )
        if scheduler_config.page_size is None:
            raise ValueError(
                "DSv4 KV cache calculation requires scheduler_config.page_size to be set "
                "explicitly. Add `page_size` to your server_args.json."
            )
        calc = DSv4MemoryCalculator(
            model_config=_Cfg(),
            page_size=scheduler_config.page_size,
            swa_ratio=scheduler_config.swa_full_tokens_ratio,
        )
        return calc.bytes_per_full_token

    cell_elems = calc_kv_cache_cell_elems(
        model_info, scheduler_config.tp_size, scheduler_config.pp_size
    )
    return cell_elems * scheduler_config.kv_cache_data_type.bytes


def kv_bytes_per_full_token(
    model_info: ModelInfo, scheduler_config: SchedulerConfig
) -> float:
    """Cluster-total KV-cache bytes/token across all TP * PP shards.

    A "logical token" (sglang request-level count) lives on every TP shard of
    every PP rank, so its true KV footprint = per-GPU value * tp * pp.
    This is the value tier metrics multiply by token counts to get GB.

    Reproduces the legacy hardcoded behavior:
      - GLM-5 (MHA, tp=8): 53.93 KiB/GPU * 8 = 431.45 KiB/token (cluster)
      - DSv4-Pro (tp=4):   23.19 KiB/GPU * 4 = 92.77 KiB/token (cluster)
    """
    per_gpu = kv_bytes_per_full_token_per_gpu(model_info, scheduler_config)
    return per_gpu * scheduler_config.tp_size * scheduler_config.pp_size


# KV cache bytes per token, model-specific.
# Sources: sglang server log "DSV4 memory calculation" line (dsv4-pro);
#          original GLM-5 calibration (glm-5).
#
# TODO (proper fix): replace this lookup with derivation from model/config:
#   bytes_per_token = calc_kv_cache_cell_elems(model_info, tp, pp) *
#                     scheduler_config.kv_cache_data_type.bytes
# `calc_kv_cache_cell_elems` already exists above. To wire through,
# C_SchedulerHook (sglang/scheduler.py) would need to pass model_info +
# scheduler_config (both available via ConfigManager.get_*) into calc_metrics.
_KB_PER_TOKEN_BY_MODEL: dict[str, float] = {
    # Cluster-total KiB/token (sum across all TP*PP shards) — same convention
    # as kv_bytes_per_full_token() above. Used only when ConfigManager isn't
    # populated; main path always derives the value live.
    "glm-5":    431.45,                  # = 53.93 KiB/GPU * tp=8
    "dsv4-pro": 23749.84 / 1024 * 4,     # ~92.77 KiB; 23.19 KiB/GPU * tp=4
}


def lookup_kb_per_token(model_path: str) -> float:
    """Fallback for legacy callers without ModelInfo/SchedulerConfig.

    Matches model identifier substrings inside ``model_path``.
    """
    p = (model_path or "").lower()
    if "deepseek-v4-pro" in p or "dsv4-pro" in p or "dpskv4pro" in p:
        return _KB_PER_TOKEN_BY_MODEL["dsv4-pro"]
    if "glm-5" in p or "glm5" in p:
        return _KB_PER_TOKEN_BY_MODEL["glm-5"]
    raise ValueError(f"No KV bytes/token mapping for model_path={model_path!r}")


# Back-compat alias for callers still importing the old constant name.
_GLM5_KB_PER_TOKEN = _KB_PER_TOKEN_BY_MODEL["glm-5"]


def _tokens_to_gb(tokens: float, kb_per_token: float) -> float:
    return tokens * kb_per_token / (1024 * 1024)


def calc_kv_cache_tier_metrics(
    total_input: int,
    total_reused_tokens: int,
    total_host_hit_tokens: int,
    total_storage_hit_tokens: int,
    total_dur_s: float,
    kb_per_token: float = _GLM5_KB_PER_TOKEN,
) -> dict:
    """Compute KV cache metrics across the L1/L2/L3 storage tiers.

    Tier mapping (write-through policy):
        L1 -> device HBM, L2 -> host DRAM, L3 -> storage DISK.

    Traffic model:
        * new input tokens (total_input - total_reused_tokens) are
          written through to L1/L2/L3 simultaneously.
        * L3 -> L2 write-back moves the tokens that were hit on L3
          (``total_storage_hit_tokens``) up to L2.
        * L2 -> L1 write-back moves the tokens that were hit on L2 or
          pulled up from L3 (``total_host_hit_tokens +
          total_storage_hit_tokens``) up to L1.

    Occupancy model:
        Under write-through, every new input token occupies one slot on
        each tier, so L1/L2/L3 occupancy all equal the new input tokens.
    """
    dur_s = max(total_dur_s, 1e-9)

    total_new_input_tokens = total_input - total_reused_tokens
    total_new_input_gb = _tokens_to_gb(total_new_input_tokens, kb_per_token)

    # Write-through traffic for new input tokens (identical on L1/L2/L3).
    new_input_write_thr_tokens = total_new_input_tokens / dur_s
    new_input_write_thr_gb = total_new_input_gb / dur_s

    # L3 -> L2 write-back: tokens that were hit at the storage tier.
    l3_to_l2_tokens = total_storage_hit_tokens
    l3_to_l2_gb = _tokens_to_gb(l3_to_l2_tokens, kb_per_token)
    l3_to_l2_thr_tokens = l3_to_l2_tokens / dur_s
    l3_to_l2_thr_gb = l3_to_l2_gb / dur_s

    # L2 -> L1 write-back: tokens hit at host, which already include the
    # portion pulled up from storage.
    l2_to_l1_tokens = total_host_hit_tokens + total_storage_hit_tokens
    l2_to_l1_gb = _tokens_to_gb(l2_to_l1_tokens, kb_per_token)
    l2_to_l1_thr_tokens = l2_to_l1_tokens / dur_s
    l2_to_l1_thr_gb = l2_to_l1_gb / dur_s

    return {
        "kv_cache_kb_per_token": kb_per_token,
        # Keep legacy keys for backward compatibility.
        "total_new_input": total_new_input_tokens,
        "total_new_input_GB": total_new_input_gb,
        # Write-through traffic for new input (same on L1/L2/L3).
        "new_input_write_throughput_tokens_per_s": new_input_write_thr_tokens,
        "new_input_write_throughput_GB_per_s": new_input_write_thr_gb,
        "L3_write_throughput_tokens_per_s": new_input_write_thr_tokens / 8,
        "L3_write_throughput_GB_per_s": new_input_write_thr_gb / 8,
        # L3 -> L2 write-back totals and throughput.
        "l3_to_l2_tokens": l3_to_l2_tokens,
        "l3_to_l2_GB": l3_to_l2_gb,
        "l3_to_l2_throughput_tokens_per_s": l3_to_l2_thr_tokens,
        "l3_to_l2_throughput_GB_per_s": l3_to_l2_thr_gb,
        # L2 -> L1 write-back totals and throughput (includes L3 portion).
        "l2_to_l1_tokens": l2_to_l1_tokens,
        "l2_to_l1_GB": l2_to_l1_gb,
        "l2_to_l1_throughput_tokens_per_s": l2_to_l1_thr_tokens,
        "l2_to_l1_throughput_GB_per_s": l2_to_l1_thr_gb,
    }


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
    for req in requests:
        if not req.is_complete():
            continue
        completed += 1
        ttfts.append(req.gen_token_latencies[0])
        queue_durs.append(req.queue_end - req.queue_start)
        if len(req.gen_token_latencies) > 1:
            # output length > 1
            tpots.append(np.mean(req.gen_token_latencies[1:]))
        itls.extend(req.gen_token_latencies[1:])
        e2e_latencies.append(sum(req.gen_token_latencies))
        total_dur_s = max(total_dur_s, req.last_event_time)
        total_input += req.input_length
        total_output += req.output_length
        total_reused_tokens += req.final_device_hit_len
        total_device_hit_tokens += req.final_device_hit_len - req.final_host_hit_len
        total_host_hit_tokens += req.final_host_hit_len - req.final_storage_hit_len
        total_storage_hit_tokens += req.final_storage_hit_len

    # Derive accurate kb_per_token via a real 3-tier fallback chain.
    # Historical bug: the chain used to be `if/elif/else` inside a single
    # try/except — if path 1 (kv_bytes_per_full_token) RAISED, the outer except
    # jumped straight to glm-5 default, skipping path 2 (lookup_kb_per_token).
    # That silently gave 431.45 KB/token for DSv4-Pro instead of 92.77,
    # inflating all *_GB metrics by ~4.65x. The 3 paths are now independent
    # try/except'd so each can fail back to the next.
    kb_per_token = None
    model_info = None
    sched_config = None
    try:
        from sglang_simulator.simulation.manager import ConfigManager
        model_info = ConfigManager.get_model_info()
        sched_config = ConfigManager.get_scheduler_config()
    except Exception as e:
        print(f"[utils.calc_metrics] ConfigManager unavailable: {type(e).__name__}: {e}")

    # Path 1: live derivation from model + scheduler config (most accurate).
    if kb_per_token is None and model_info is not None and sched_config is not None:
        try:
            kb_per_token = kv_bytes_per_full_token(model_info, sched_config) / 1024
        except Exception as e:
            print(
                f"[utils.calc_metrics] kv_bytes_per_full_token failed "
                f"({type(e).__name__}: {e}); trying per-model lookup"
            )

    # Path 2: model_path substring lookup (works without scheduler_config).
    if kb_per_token is None and model_info is not None and getattr(model_info, "model_path", None):
        try:
            kb_per_token = lookup_kb_per_token(model_info.model_path)
        except Exception as e:
            print(
                f"[utils.calc_metrics] lookup_kb_per_token({model_info.model_path!r}) "
                f"failed: {e}; using glm-5 default"
            )

    # Path 3: hardcoded glm-5 default (last resort — gives WRONG GB metrics
    # for any non-glm-5 model; will print a warning above).
    if kb_per_token is None:
        kb_per_token = _KB_PER_TOKEN_BY_MODEL["glm-5"]

    kv_cache_tier_metrics = calc_kv_cache_tier_metrics(
        total_input=total_input,
        total_reused_tokens=total_reused_tokens,
        total_host_hit_tokens=total_host_hit_tokens,
        total_storage_hit_tokens=total_storage_hit_tokens,
        total_dur_s=total_dur_s,
        kb_per_token=kb_per_token,
    )

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
        **kv_cache_tier_metrics,
        "mean_ttft_ms": np.mean(ttfts or 0) * 1000,
        "median_ttft_ms": np.median(ttfts or 0) * 1000,
        # "std_ttft_ms": np.std(ttfts or 0) * 1000,
        "p90_ttft_ms": np.percentile(ttfts or 0, 90) * 1000,
        # "p95_ttft_ms": np.percentile(ttfts or 0, 95) * 1000,
        # "p99_ttft_ms": np.percentile(ttfts or 0, 99) * 1000,
        "mean_queue_ms": np.mean(queue_durs or 0) * 1000,
        "mean_tpot_ms": np.mean(tpots or 0) * 1000,
        "median_tpot_ms": np.median(tpots or 0) * 1000,
        # "std_tpot_ms": np.std(tpots or 0) * 1000,
        "p90_tpot_ms": np.percentile(tpots or 0, 90) * 1000,
        # "p95_tpot_ms": np.percentile(tpots or 0, 95) * 1000,
        # "p99_tpot_ms": np.percentile(tpots or 0, 99) * 1000,
        "mean_itl_ms": np.mean(itls or 0) * 1000,
        "median_itl_ms": np.median(itls or 0) * 1000,
        # "std_itl_ms": np.std(itls or 0) * 1000,
        "p90_itl_ms": np.percentile(itls or 0, 90) * 1000,
        # "p95_itl_ms": np.percentile(itls or 0, 95) * 1000,
        # "p99_itl_ms": np.percentile(itls or 0, 99) * 1000,
        "max_itl_ms": np.max(itls or 0) * 1000,
        "mean_e2e_latency_ms": np.mean(e2e_latencies) * 1000,
        "median_e2e_latency_ms": np.median(e2e_latencies) * 1000,
        # "std_e2e_latency_ms": np.std(e2e_latencies) * 1000,
        "p90_e2e_latency_ms": np.percentile(e2e_latencies or 0, 90) * 1000,
        # "p95_e2e_latency_ms": np.percentile(e2e_latencies or 0, 95) * 1000,
        # "p99_e2e_latency_ms": np.percentile(e2e_latencies or 0, 99) * 1000,
        "time_cost": -1,  # Updated by external benchmark caller
    }
