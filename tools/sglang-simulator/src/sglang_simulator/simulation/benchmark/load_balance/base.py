from abc import ABC, abstractmethod
from typing import Optional

from sglang_simulator.dataset import GenericRequest
from sglang_simulator.simulation.benchmark.base_runner import BaseWorker


class LoadBalancingPolicy(ABC):
    @abstractmethod
    def select_worker(
        self, workers: list[BaseWorker], req: GenericRequest
    ) -> Optional[BaseWorker]: ...

    @abstractmethod
    def name(self) -> str: ...

    def on_request_complete(self, worker_name: str, success: bool, token_cost: int = 1):
        pass

    def update_loads(self, loads: dict[str, int]):
        pass

    def init_workers(self, workers: list[BaseWorker]):
        pass

    def reset(self):
        pass
