import asyncio
import json
import os
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
from sglang_simulator.utils.logger import get_logger

logger = get_logger("sglang_simulator")


class MultiInstanceBenchmarkRunner(BaseBenchmarkRunner):
    def __init__(self, workers: list[BaseWorker]):
        self.workers = workers[0]
        self.loop = asyncio.new_event_loop()

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

        await self.workers.trigger_simulation()

        tasks = []
        logger.info(f"Created {len(dataset)} request tasks.")
        for req in self.get_request(
            dataset,
            ignore_timestamp=benchmark_config.ignore_request_timestamp,
            request_rate=benchmark_config.request_rate,
        ):
            task = asyncio.create_task(self.workers.async_generate(req))
            tasks.append(task)

        _ = await asyncio.gather(*tasks)

        # dump result
        await self.workers.trigger_simulation()

        if os.path.exists(self.workers.output_dir + "/metrics.json"):
            with open(self.workers.output_dir + "/metrics.json", "r") as f:
                metrics = json.load(f)
        else:
            logger.error(
                f"Failed to load metrics from serving backend. The metrics file should be loaded from {self.workers.output_dir}."
            )
            return None

        return metrics

    def benchmark(self, benchmark_config: BenchmarkConfig, dataset: BaseDataset):

        return self.loop.run_until_complete(
            self.async_benchmark(benchmark_config, dataset)
        )

    def get_request_stats(self) -> dict[str, list[dict]]:
        return self.workers.get_request_stats()

    def get_iteration_stats(self):
        return self.workers.get_iteration_stats()

    def shutdown(self):
        self.workers.shutdown()

    def flush_cache(self):
        self.workers.flush_cache()
