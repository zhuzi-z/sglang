import os
import random

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


def _create_workers(model_path, num_workers=3):
    from sglang.srt.server_args import ServerArgs  # noqa

    workers = []
    for idx in range(num_workers):
        workers.append(
            SGLangWorker(
                server_args=ServerArgs(
                    model_path=model_path,
                    load_format="dummy",
                    device="cpu",
                    max_total_tokens=100000,
                ),
                name=f"worker{idx}",
            )
        )
    return workers


def _create_dataset(model_path, num_workers):
    dataset_args = DatasetArgs(
        "random_ids",
        num_prompts=num_workers * 10,
        min_input_len=1000,
        max_input_len=1001,
        min_output_len=1,
        max_output_len=2,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = get_dataset(dataset_args, tokenizer=tokenizer)
    for idx, req in enumerate(dataset):
        req.custom_params["created_time"] = idx
    return dataset


def test_benchmark_inner_round_robin():
    model_path = "Qwen/Qwen3-8B"
    workers = _create_workers(model_path)

    runner = MultiInstanceBenchmarkRunner(workers=workers, lb_proxy=RoundRobinPolicy())

    benchmark_config = BenchmarkConfig(ignore_request_timestamp=False)
    dataset = _create_dataset(model_path, len(workers))

    metrics = runner.benchmark(benchmark_config, dataset=dataset)
    assert metrics["completed"] == len(dataset)

    request_stats = runner.get_request_stats()
    request_stats = sorted(request_stats, key=lambda x: x["created_time"])
    for idx, stats in enumerate(request_stats):
        assert stats["worker"] == workers[idx % len(workers)].name

    runner.shutdown()


def test_benchmark_sglang_router_cache_aware():
    from sglang_router.sglang_router_rs import PyWorkerInfo

    from sglang_simulator.dataset import SimpleDataset, GenericRequest
    from sglang_simulator.simulation.benchmark.load_balance import SGLangRouterPolicy

    model_path = "Qwen/Qwen3-8B"
    num_workers = 3
    workers = _create_workers(model_path, num_workers)

    gateway_policy = SGLangRouterPolicy("cache_aware", cache_threshold=0.3)
    runner = MultiInstanceBenchmarkRunner(workers=workers, lb_proxy=gateway_policy)

    # Use worker names as prefixes for straightforward assertion
    prefixes = [f"{workers[i].name} " * 10 for i in range(num_workers)]

    # Warmup: seed each prefix to a distinct worker via load manipulation
    prefix_to_worker = {}
    for i, prefix in enumerate(prefixes):
        loads = [100 if k != i else 0 for k in range(num_workers)]
        worker_infos = [
            PyWorkerInfo(url=workers[k].name, load=loads[k])
            for k in range(num_workers)
        ]
        idx = gateway_policy._policy.select_worker(worker_infos, request_text=prefix)
        assert idx == i, f"warmup failed: prefix {i} routed to worker {idx}"
        prefix_to_worker[i] = workers[i].name

    # Build dataset: each prefix extended with different suffixes
    reqs = []
    for i, prefix in enumerate(prefixes):
        for j in range(5):
            reqs.append(GenericRequest(
                prompt=prefix + f"suffix-{j}",
                output_length=1,
                custom_params={"created_time": random.random()},
            ))

    dataset = SimpleDataset(reqs=reqs)
    benchmark_config = BenchmarkConfig(ignore_request_timestamp=True)
    metrics = runner.benchmark(benchmark_config, dataset=dataset)
    assert metrics["completed"] == len(dataset)

    # Verify routing: each worker only receives requests starting with its own name
    routing_records = runner.get_lb_routing_records()
    for worker_name, routed_reqs in routing_records.items():
        for req in routed_reqs:
            assert req.prompt.startswith(worker_name), (
                f"worker {worker_name}: got '{req.prompt[:30]}...'"
            )

    runner.shutdown()


if __name__ == "__main__":
    test_benchmark_inner_round_robin()
    test_benchmark_sglang_router_cache_aware()
