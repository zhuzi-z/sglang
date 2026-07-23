import asyncio
import random
from typing import Iterator
from collections import defaultdict

import numpy as np
from sglang_simulator.dataset import BaseDataset, GenericRequest
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


class PDDisaggBenchmarkRunner(BaseBenchmarkRunner):
    def __init__(
        self,
        prefill_workers: list[BaseWorker],
        decode_workers: list[BaseWorker],
        prefill_lb_proxy: LoadBalancingPolicy | None = None,
        decode_lb_proxy: LoadBalancingPolicy | None = None,
    ):
        self.prefill_workers = prefill_workers
        self.decode_workers = decode_workers
        assert len(set(w.name for w in prefill_workers + decode_workers)) == \
                len(prefill_workers) + len(decode_workers), "The worker's names should be unique."
        self.worker_request_count: dict[str, int] = defaultdict(int)

        self.prefill_lb_proxy = prefill_lb_proxy or RoundRobinPolicy()
        self.decode_lb_proxy = decode_lb_proxy or RoundRobinPolicy()
        self.prefill_lb_proxy.init_workers(prefill_workers)
        self.decode_lb_proxy.init_workers(decode_workers)
        self.loop = asyncio.new_event_loop()

        self.lb_routing_records: dict[str, list[GenericRequest]] = {}
        self.request_stats: list[dict] = []

    @property
    def all_workers(self) -> list[BaseWorker]:
        return self.prefill_workers + self.decode_workers

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
                    # "total_request": len(dataset),
                    "created_time": created_time,
                }
            )
            yield req

    def _pair_request(self, req: GenericRequest):
        """Pair a request with its prefill and decode workers."""
        prefill_worker = self.prefill_lb_proxy.select_worker(
            self.prefill_workers, req
        )
        decode_worker = self.decode_lb_proxy.select_worker(
            self.decode_workers, req
        )
        if prefill_worker:
            self.lb_routing_records.setdefault(prefill_worker.name, []).append(req)
            self.worker_request_count[prefill_worker.name] += 1
        if decode_worker:
            self.lb_routing_records.setdefault(decode_worker.name, []).append(req)
            self.worker_request_count[decode_worker.name] += 1

        req.extra_args.update({
            "bootstrap_port": 30500,
            "bootstrap_room": random.randint(0, 2**63 - 1),
        })
        req.custom_params["prefill_worker"] = prefill_worker
        req.custom_params["decode_worker"] = decode_worker

    async def _run_prefill_request(self, req: GenericRequest):
        """Run prefill phase for a single request."""
        prefill_worker = req.custom_params.get("prefill_worker")
        if not prefill_worker:
            return {}

        prefill_response = await prefill_worker.async_generate(req)
        prefill_sim_stat = prefill_response['meta_info']['finish_reason'].get("simulation_stat")
        if not prefill_sim_stat:
            logger.error("Prefill response does not contain simulation result.")
            return {}

        # update the created_time which should be the end of prefilling.
        req.custom_params["created_time"] = prefill_sim_stat["last_event_time"]
        return prefill_sim_stat

    async def _run_decode_request(self, req: GenericRequest):
        """Run decode phase for a single request."""
        decode_worker = req.custom_params.get("decode_worker")
        if not decode_worker:
            return {}

        decode_response = await decode_worker.async_generate(req)
        decode_sim_stat = decode_response['meta_info']['finish_reason'].get("simulation_stat")
        return decode_sim_stat or {}

    def _collect_request_stats(
        self,
        req: GenericRequest,
        prefill_sim_stat: dict | None,
        decode_sim_stat: dict | None,
    ):
        """Collect per-request stats from prefill and decode results."""
        prefill_sim_stat = prefill_sim_stat or {}
        decode_sim_stat = decode_sim_stat or {}

        prefill_worker = req.custom_params.get("prefill_worker")
        decode_worker = req.custom_params.get("decode_worker")

        if not prefill_sim_stat and not decode_sim_stat:
            return

        def pick_first_not_none(key, *stats_dicts):
            """Return the first non-None value for the given key."""
            for stats in stats_dicts:
                value = stats.get(key)
                if value is not None:
                    return value
            return None

        self.request_stats.append({
            "prefill_worker": prefill_worker.name if prefill_worker else None,
            "decode_worker": decode_worker.name if decode_worker else None,
            "last_event_time": pick_first_not_none(
                "last_event_time", decode_sim_stat, prefill_sim_stat
            ),
            "queue_start": pick_first_not_none(
                "queue_start", prefill_sim_stat, decode_sim_stat
            ),
            "queue_end": pick_first_not_none(
                "queue_end", prefill_sim_stat, decode_sim_stat
            ),
            # The prefix cache only considers the prefill worker
            "final_device_hit_len": prefill_sim_stat.get("final_device_hit_len"),
            "final_host_hit_len": prefill_sim_stat.get("final_host_hit_len"),
            "final_storage_hit_len": prefill_sim_stat.get("final_storage_hit_len"),
            "input_length": pick_first_not_none(
                "input_length", prefill_sim_stat, decode_sim_stat
            ),
            "output_length": pick_first_not_none(
                "output_length", prefill_sim_stat, decode_sim_stat
            ),
            "gen_token_latencies": (
                (prefill_sim_stat.get("gen_token_latencies") or [])
                + (decode_sim_stat.get("gen_token_latencies") or [])
            ),
            # KV cache transfer only happens on the decode side.
            "kv_cache_transfer_queue_start_time": decode_sim_stat.get(
                "kv_cache_transfer_queue_start_time", -1
            ),
            "kv_cache_transfer_start_time": decode_sim_stat.get(
                "kv_cache_transfer_start_time", -1
            ),
            "kv_cache_transfer_duration": decode_sim_stat.get(
                "kv_cache_transfer_duration", 0.0
            ),
        })

    async def async_benchmark(
        self,
        benchmark_config: BenchmarkConfig,
        dataset: BaseDataset,
    ):
        # Reset per-round state: benchmark() may be called multiple times on the
        # same runner (e.g. sweeping slowdown factors). Without this, request
        # counts accumulate and continue_generation(num_new_reqs=...) exceeds the
        # actual number of requests, so the worker-side ReqDispatcher never
        # releases the round's requests.
        self.worker_request_count.clear()
        self.lb_routing_records.clear()
        self.request_stats.clear()

        for worker in self.all_workers:
            await worker.trigger_simulation()
            await worker.pause_generation()

        # Collect requests
        requests = list(self.get_request(
            dataset,
            ignore_timestamp=benchmark_config.ignore_request_timestamp,
            request_rate=benchmark_config.request_rate,
        ))
        logger.info(f"Created {len(requests)} PD disagg request tasks.")

        # Phase 1: Pair P/D workers for each request
        for req in requests:
            self._pair_request(req)

        # Phase 2: P workers execute, wait for all results
        prefill_gen_tasks = []
        for req in requests:
            prefill_gen_tasks.append(asyncio.create_task(
                self._run_prefill_request(req)
            ))

        for worker in self.prefill_workers:
            await worker.continue_generation(num_new_reqs=self.worker_request_count[worker.name])
        
        prefill_stats_list = await asyncio.gather(*prefill_gen_tasks)

        decode_gen_tasks = []
        for req in requests:
            decode_gen_tasks.append(asyncio.create_task(
                self._run_decode_request(req)
            ))

        for worker in self.decode_workers:
            await worker.continue_generation(num_new_reqs=self.worker_request_count[worker.name])
        
        decode_stats_list = await asyncio.gather(*decode_gen_tasks)

        # Collect stats
        for req, p_stat, d_stat in zip(requests, prefill_stats_list, decode_stats_list):
            self._collect_request_stats(req, p_stat, d_stat)

        for worker in self.all_workers:
            await worker.trigger_simulation()

        request_stats = self.get_request_stats()
        metrics = calc_metrics(request_stats)
        return metrics

    def benchmark(self, benchmark_config: BenchmarkConfig, dataset: BaseDataset):
        return self.loop.run_until_complete(
            self.async_benchmark(benchmark_config, dataset)
        )

    def get_request_stats(self) -> list[dict]:
        if self.request_stats:
            return self.request_stats
        result = []
        for worker in self.decode_workers:
            request_stats = worker.get_request_stats()
            for item in request_stats:
                item["worker"] = worker.name
            result.extend(request_stats)
        return result

    def get_iteration_stats(self) -> list[dict]:
        result = []
        for worker in self.all_workers:
            iteration_stats = worker.get_iteration_stats()
            for item in iteration_stats:
                item["worker"] = worker.name
            result.extend(iteration_stats)
        return result

    def get_lb_routing_records(self) -> dict[str, list[GenericRequest]]:
        return self.lb_routing_records

    def shutdown(self):
        for worker in self.all_workers:
            worker.shutdown()

    def flush_cache(self):
        for worker in self.all_workers:
            worker.flush_cache()
