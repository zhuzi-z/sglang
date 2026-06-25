import asyncio
from typing import Iterator

import numpy as np
from sglang_simulator.dataset import (
    BaseDataset,
    GenericRequest,
)
from sglang_simulator.simulation.benchmark.base_runner import (
    BaseBenchmarkRunner,
    BaseWorker,
)
from sglang_simulator.simulation.benchmark.bench_config import BenchmarkConfig
from sglang_simulator.simulation.benchmark.load_balance import (
    LoadBalancingPolicy,
    RoundRobinPolicy,
)
from sglang_simulator.simulation.utils import calc_metrics
from sglang_simulator.utils.logger import get_logger

logger = get_logger("sglang_simulator")


class MultiInstanceBenchmarkRunner(BaseBenchmarkRunner):
    def __init__(
        self, workers: list[BaseWorker], lb_proxy: LoadBalancingPolicy | None = None
    ):
        self.workers = workers
        self.lb_proxy = lb_proxy if lb_proxy is not None else RoundRobinPolicy()
        self.lb_proxy.init_workers(workers)
        self.loop = asyncio.new_event_loop()

        # Worker's name -> requests
        self.lb_routing_records: dict[str, list[GenericRequest]] = {}

    def get_request(
        self,
        dataset: BaseDataset,
        ignore_timestamp: bool = False,
        request_rate: float = float("inf"),
    ) -> Iterator[GenericRequest]:
        yield_delay = 0
        for req in dataset:
            if ignore_timestamp:
                created_time = yield_delay
                yield_delay += np.random.exponential(1.0 / request_rate)
            else:
                created_time = req.custom_params.get("created_time", 0)

            req.custom_params.update(
                {
                    "total_request": len(dataset),  # include the warmup requests.
                    "created_time": created_time,
                }
            )

            yield req

    async def async_benchmark(
        self,
        benchmark_config: BenchmarkConfig,
        dataset: BaseDataset,
    ):

        for worker in self.workers:
            await worker.trigger_simulation()
            await worker.pause_generation()

        tasks = []
        logger.info(f"Created {len(dataset)} request tasks.")
        for req in self.get_request(
            dataset,
            ignore_timestamp=benchmark_config.ignore_request_timestamp,
            request_rate=benchmark_config.request_rate,
        ):
            worker = self.lb_proxy.select_worker(self.workers, req)
            if worker.name not in self.lb_routing_records:
                self.lb_routing_records[worker.name] = []
            self.lb_routing_records[worker.name].append(req)

            async def _generate_with_callback(w=worker, r=req):
                try:
                    result = await w.async_generate(r)
                    self.lb_proxy.on_request_complete(w.name, True, req)
                    return result
                except Exception:
                    self.lb_proxy.on_request_complete(w.name, False, req)
                    raise

            task = asyncio.create_task(_generate_with_callback())
            tasks.append(task)

        for worker in self.workers:
            await worker.continue_generation()

        _ = await asyncio.gather(*tasks)

        # dump result
        for worker in self.workers:
            await worker.trigger_simulation()

        request_stats = self.get_request_stats()
        metrics = calc_metrics(request_stats)

        return metrics

    def set_lb_proxy(self, lb_proxy: LoadBalancingPolicy):
        self.lb_proxy = lb_proxy

    def benchmark(self, benchmark_config: BenchmarkConfig, dataset: BaseDataset):

        return self.loop.run_until_complete(
            self.async_benchmark(benchmark_config, dataset)
        )

    def get_request_stats(self) -> list[dict]:
        result = []
        for worker in self.workers:
            request_stats = worker.get_request_stats()
            for item in request_stats:
                item["worker"] = worker.name
            result.extend(request_stats)
        return result

    def get_iteration_stats(self):
        result = []
        for worker in self.workers:
            iteration_stats = worker.get_iteration_stats()
            for item in iteration_stats:
                item["worker"] = worker.name
            result.extend(iteration_stats)
        return result

    def get_lb_routing_records(self) -> dict[str, list[GenericRequest]]:
        return self.lb_routing_records
    
    def shutdown(self):
        for worker in self.workers:
            worker.shutdown()

    def flush_cache(self):
        for worker in self.workers:
            worker.flush_cache()
