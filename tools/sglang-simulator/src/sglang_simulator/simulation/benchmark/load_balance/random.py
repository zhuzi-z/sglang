import random
from typing import Optional

from sglang_simulator.dataset import GenericRequest
from sglang_simulator.simulation.benchmark.base_runner import BaseWorker
from sglang_simulator.simulation.benchmark.load_balance.base import LoadBalancingPolicy


class RandomPolicy(LoadBalancingPolicy):
    """Uniform random selection among workers."""

    def select_worker(
        self, workers: list[BaseWorker], req: GenericRequest
    ) -> Optional[BaseWorker]:
        if not workers:
            return None
        return random.choice(workers)

    def name(self) -> str:
        return "random"
