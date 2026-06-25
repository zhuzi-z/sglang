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

    prefixes = [f"{workers[i].name} " * 10 for i in range(num_workers)]

    # ── Warmup: 将每个 prefix 绑定到对应 worker ──────────────────────────
    for i, prefix in enumerate(prefixes):
        loads = [100 if k != i else 0 for k in range(num_workers)]
        worker_infos = [
            PyWorkerInfo(url=workers[k].name, load=loads[k])
            for k in range(num_workers)
        ]
        idx = gateway_policy._policy.select_worker(worker_infos, request_text=prefix)
        assert idx == i, f"warmup failed: prefix {i} routed to worker {idx}"

    print("\n[Warmup done] prefix->worker binding:")
    for i in range(num_workers):
        print(f"  prefix[{i}] ('{prefixes[i][:20]}...') -> worker{i}")

    # ── （3 workers × 5 suffixes，各去各自节点）──────────────
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

    routing_records = runner.get_lb_routing_records()
    for worker_name, routed_reqs in routing_records.items():
        for req in routed_reqs:
            assert req.prompt.startswith(worker_name), (
                f"worker {worker_name}: got '{req.prompt[:30]}...'"
            )

    runner.shutdown()

    # ──  所有节点空闲，同时发 req_num 个 worker0-prefix 请求 ──────────
    req_num =36
    print("\n" + "=" * 60)
    print(f"[New Test] {req_num} concurrent requests all with worker0's prefix")
    print("  Expected behavior: cache_aware should prefer worker0")
    print("  Question: does load awareness spill overflow to worker1/worker2?")
    print("=" * 60)

    workers2 = _create_workers(model_path, num_workers)
    gateway_policy2 = SGLangRouterPolicy("cache_aware", cache_threshold=0.3)
    runner2 = MultiInstanceBenchmarkRunner(workers=workers2, lb_proxy=gateway_policy2)

    # Warmup：只 seed worker0 的 prefix
    w0_prefix = f"{workers2[0].name} " * 10
    for i in range(num_workers):
        loads_warmup = [100 if k != i else 0 for k in range(num_workers)]
        winfos = [PyWorkerInfo(url=workers2[k].name, load=loads_warmup[k]) for k in range(num_workers)]
        idx = gateway_policy2._policy.select_worker(winfos, request_text=f"{workers2[i].name} " * 10)
        assert idx == i

    print(f"\n[Warmup2 done] worker0's prefix seeded to worker0")

    # 构造 req_num 条全部带 worker0 prefix 的请求
    reqs2 = []
    for j in range(req_num):
        reqs2.append(GenericRequest(
            prompt=w0_prefix + f"query-{j}",
            output_length=1,
            custom_params={"created_time": 0.0},
        ))

    # Patch select_worker，记录每条请求的路由决策
    route_decisions = []
    _orig = gateway_policy2.select_worker

    def _patched(workers_arg, req_arg):
        chosen = _orig(workers_arg, req_arg)
        route_decisions.append(chosen.name if chosen else "None")
        return chosen

    gateway_policy2.select_worker = _patched

    dataset2 = SimpleDataset(reqs=reqs2)
    benchmark_config2 = BenchmarkConfig(ignore_request_timestamp=True)
    metrics2 = runner2.benchmark(benchmark_config2, dataset=dataset2)
    assert len(route_decisions) == len(dataset2), (
        f"Expected {len(dataset2)} routing decisions, got {len(route_decisions)}"
    )

    # 统计每个 worker 收到了多少请求
    from collections import Counter
    counter = Counter(route_decisions)

    print(f"\n[Routing decisions for {req_num} worker0-prefix requests]")
    for seq_idx, (prompt, dest) in enumerate(
        zip([r.prompt for r in reqs2], route_decisions)
    ):
        mark = "✓" if dest == "worker0" else "↪ SPILL"
        print(f"  req[{seq_idx:02d}] -> {dest}  {mark}  (prompt: '{prompt[20:40]}')")

    print(f"\n[Summary]")
    for wname in [w.name for w in workers2]:
        cnt = counter.get(wname, 0)
        bar = "█" * cnt
        print(f"  {wname}: {cnt:2d} requests  {bar}")

    all_to_w0 = all(d == "worker0" for d in route_decisions)
    if all_to_w0:
        print(f"\n[Conclusion] ALL {req_num} requests went to worker0.")
        print("  => cache_aware ignores actual load (no load feedback).")
        print("  => No spill to worker1/worker2, load imbalance not detected.")
    else:
        spill_count = sum(1 for d in route_decisions if d != "worker0")
        print(f"\n[Conclusion] {spill_count}/{req_num} requests spilled to other workers.")
        print("  => cache_aware detected worker0 congestion and redistributed load.")

    runner2.shutdown()

if __name__ == "__main__":
    # test_benchmark_inner_round_robin()
    test_benchmark_sglang_router_cache_aware()
