from typing import Optional

from sglang_simulator.dataset import GenericRequest
from sglang_simulator.simulation.benchmark.base_runner import BaseWorker
from sglang_simulator.simulation.benchmark.load_balance.base import LoadBalancingPolicy


class GatewayPolicy(LoadBalancingPolicy):
    """Adapter that delegates to sglang_router's PyPolicy."""

    def __init__(self, policy_name: str, **kwargs):
        from sglang_router.sglang_router_rs import PyPolicy, PyWorkerInfo

        self._policy = PyPolicy(policy_name, **kwargs)
        self._PyWorkerInfo = PyWorkerInfo

    def select_worker(
        self, workers: list[BaseWorker], req: GenericRequest
    ) -> Optional[BaseWorker]:
        if not workers:
            return None
        worker_infos = [
            self._PyWorkerInfo(url=w.name) for w in workers
        ]
        request_text = req.prompt if req is not None else None
        tokens = req.token_ids if req is not None else None
        idx = self._policy.select_worker(worker_infos, request_text=request_text, tokens=tokens)
        return workers[idx]

    def init_workers(self, workers):
        worker_infos = [self._PyWorkerInfo(url=w.name) for w in workers]
        self._policy.init_workers(worker_infos)

    def name(self) -> str:
        return self._policy.name()

    def reset(self):
        self._policy.reset()
