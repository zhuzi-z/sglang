import os
import random
import numpy as np

from sglang_simulator.dataset import DatasetArgs, SimpleDataset, get_dataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig
from transformers import AutoTokenizer

os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = (
    os.path.dirname(__file__) + "/assets/config.json"
)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sglang_simulator.simulation.sglang.bench_runner import (
    SGLangBenchmarkRunner,
)

random.seed(0)
np.random.seed(0)


def test_benchmark_sglang():
    from sglang.srt.server_args import ServerArgs  # noqa

    model_path = "Qwen/Qwen3-8B"
    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=model_path,
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=True,
            hicache_ratio=2,
            hicache_storage_backend="file",
            hicache_storage_prefetch_policy="wait_complete",
            max_total_tokens=10000,
            page_size=2,
        )
    )
    runner.clear_hicache_storage()

    # Benchmark settings
    benchmark_config = BenchmarkConfig(request_rate=10, ignore_request_timestamp=True)

    # Build random requests
    dataset_args = DatasetArgs(
        "random_ids",
        num_prompts=100,
        min_input_len=1000,
        max_input_len=1001,
        min_output_len=1,
        max_output_len=2,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = get_dataset(dataset_args, tokenizer=tokenizer)

    # Split requests for cache tests
    cached_ds = SimpleDataset(reqs=dataset[:8])
    evict_l1_ds = SimpleDataset(reqs=dataset[10:20])
    evict_l2_ds = SimpleDataset(reqs=dataset[20:40])

    # First run: warm up cache
    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["completed"] == len(cached_ds)

    request_stats = runner.get_request_stats()
    for idx, req in enumerate(request_stats):
        assert (
            idx == 0 or req["created_time"] != 0
        ), "created_time should not be zero when request_rate=10"

    assert metrics["prefix_cache_reused_ratio"] == 0

    # Second run: hit device cache
    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["kv_cache_device_hit_ratio"] > 0.95

    # Evict from device cache, then hit host cache
    _ = runner.benchmark(benchmark_config, dataset=evict_l1_ds)
    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["kv_cache_host_hit_ratio"] > 0.95

    # Evict from host cache, then hit storage cache
    _ = runner.benchmark(benchmark_config, dataset=evict_l2_ds)
    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["kv_cache_storage_hit_ratio"] > 0.95

    print(metrics["mean_ttft_ms"])  # -56.48228185033968

    runner.shutdown()


if __name__ == "__main__":
    test_benchmark_sglang()
