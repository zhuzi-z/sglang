import os

from sglang_simulator.dataset import DatasetArgs, get_dataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig
from transformers import AutoTokenizer

os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = (
    os.path.dirname(__file__) + "/assets/config.json"
)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sglang_simulator.simulation.benchmark import MultiInstanceBenchmarkRunner
from sglang_simulator.simulation.benchmark.load_balance import RoundRobinPolicy
from sglang_simulator.simulation.sglang.worker import SGLangWorker


def test_benchmark_sglang():
    from sglang.srt.server_args import ServerArgs  # noqa

    model_path = "Qwen/Qwen3-8B"
    workers = []
    for idx in range(3):
        workers.append(
            SGLangWorker(
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
                ),
                name=f"worker{idx}",
            )
        )

    runner = MultiInstanceBenchmarkRunner(workers=workers, lb_proxy=RoundRobinPolicy())

    # Benchmark settings
    benchmark_config = BenchmarkConfig(ignore_request_timestamp=False)

    # Build random requests
    dataset_args = DatasetArgs(
        "random_ids",
        num_prompts=len(workers) * 10,
        min_input_len=1000,
        max_input_len=1001,
        min_output_len=1,
        max_output_len=2,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = get_dataset(dataset_args, tokenizer=tokenizer)
    for idx, req in enumerate(dataset):
        req.custom_params["created_time"] = idx

    metrics = runner.benchmark(benchmark_config, dataset=dataset)
    assert metrics["completed"] == len(dataset)

    runner.shutdown()


if __name__ == "__main__":
    test_benchmark_sglang()
