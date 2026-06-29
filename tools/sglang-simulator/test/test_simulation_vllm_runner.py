"""Test vLLM framework hijacking via sglang_simulator hooks.

Follows the same pattern as test_simulation_sglang_runner:
1. Creates VLLMWorker + MultiInstanceBenchmarkRunner
2. Builds random dataset aligned to block_size
3. Validates completions, throughput, and created_time
4. Tests L1 (device) prefix cache hit (>95%)
5. Tests L2 (host/CPU) cache hit via native kv_offloading (>95%)
"""

import os
import random
from copy import deepcopy

import numpy as np
from transformers import AutoTokenizer

from sglang_simulator.dataset import DatasetArgs, SimpleDataset, get_dataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig, MultiInstanceBenchmarkRunner
from sglang_simulator.simulation.vllm.vllm_worker import VLLMWorker, EngineArgs

os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = (
    os.path.dirname(__file__) + "/assets/config_vllm.json"
)
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def run_vllm_benchmark(engine_args: dict):
    """Run vLLM benchmark: basic throughput + L1 prefix cache hit + L2 host cache hit."""
    random.seed(0)
    np.random.seed(0)

    worker = VLLMWorker(engine_args=EngineArgs(**engine_args))
    runner = MultiInstanceBenchmarkRunner(workers=[worker])

    # Benchmark settings
    benchmark_config = BenchmarkConfig(request_rate=10, ignore_request_timestamp=True)

    # Build random requests aligned to block_size=16
    tokenizer = AutoTokenizer.from_pretrained(engine_args["model"])
    dataset_args = DatasetArgs(
        "random_ids",
        num_prompts=40,
        min_input_len=65,  # [:64] => align with block_size=16 (4 full blocks)
        max_input_len=66,
        min_output_len=1,
        max_output_len=2,
    )
    dataset = get_dataset(dataset_args, tokenizer=tokenizer)

    # Split requests for cache tests
    cached_ds = SimpleDataset(reqs=dataset[:8])
    evict_l1_ds = SimpleDataset(reqs=dataset[10:30])

    # First run: warm up cache (populates both L1 GPU and L2 CPU)
    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["completed"] == len(cached_ds)

    request_stats = runner.get_request_stats()
    for idx, req in enumerate(request_stats):
        assert (
            idx == 0 or req["created_time"] != 0
        ), "created_time should not be zero when request_rate=10"

    assert metrics["prefix_cache_reused_ratio"] == 0

    # Second run: hit device cache (L1)
    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["kv_cache_device_hit_ratio"] > 0.95, (
        f"Expected device cache hit ratio > 0.95, got {metrics['kv_cache_device_hit_ratio']}"
    )

    # Evict from device cache by filling GPU with other requests
    _ = runner.benchmark(benchmark_config, dataset=evict_l1_ds)

    # Third run: should hit host (L2 CPU) cache
    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["kv_cache_host_hit_ratio"] > 0.95, (
        f"Expected host cache hit ratio > 0.95, got {metrics['kv_cache_host_hit_ratio']}"
    )

    runner.shutdown()


def test_vllm_benchmark():
    model_engine_args_list = [
        {"model": "/host/models/Qwen/Qwen3-0.6B/"},       # MHA
        {"model": "/host/models/Qwen/Qwen3.5-0.8B/"},     # Hybrid (full_attention + linear_attention)
    ]

    common_args = {
        "block_size": 16,
        "gpu_memory_utilization": 0.9,
        "kv_offloading_size": 4.0,
        "num_gpu_blocks_override": 100,
        "max_model_len": 128,
        "enable_prefix_caching": True,
    }

    for model_args in model_engine_args_list:
        engine_args = deepcopy(common_args)
        engine_args.update(model_args)
        run_vllm_benchmark(engine_args)


if __name__ == "__main__":
    test_vllm_benchmark()
