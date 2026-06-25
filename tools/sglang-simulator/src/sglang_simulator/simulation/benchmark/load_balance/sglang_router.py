from typing import Optional

from sglang_simulator.dataset import GenericRequest
from sglang_simulator.simulation.benchmark.base_runner import BaseWorker
from sglang_simulator.simulation.benchmark.load_balance.base import LoadBalancingPolicy


def _estimate_tokens(req: GenericRequest) -> int:
    """Estimate request load cost for sglang_router's cache_aware scoring.

    sglang_router's cache_aware formula: score = prefix_matched_tokens - load.
    The load unit is "running request count", not raw token count.
    We use output_length as a lightweight cost that:
      - Scales with expected decode time (longer decode = higher cost)
      - Keeps input-only prefix requests (output=0) at cost=1
      - Allows spillover to surface realistically under burst load
    """
    if req is None:
        return 1
    output = req.output_length if req.output_length > 0 else 0
    return output


class GatewayPolicy(LoadBalancingPolicy):
    """Adapter that delegates to sglang_router's PyPolicy."""

    def __init__(self, policy_name: str, **kwargs):
        from sglang_router.sglang_router_rs import PyPolicy, PyWorkerInfo

        self._policy = PyPolicy(policy_name, **kwargs)
        self._PyWorkerInfo = PyWorkerInfo
        # Track in-flight token load per worker so PyWorkerInfo.load uses the
        # same unit (tokens) that sglang_router's cache_aware scoring expects.
        self._inflight: dict[str, int] = {}

    def select_worker(
        self, workers: list[BaseWorker], req: GenericRequest
    ) -> Optional[BaseWorker]:
        if not workers:
            return None
        worker_infos = [
            self._PyWorkerInfo(url=w.name, load=self._inflight.get(w.name, 0))
            for w in workers
        ]
        request_text = req.prompt if req is not None else None
        tokens = req.token_ids if req is not None else None
        idx = self._policy.select_worker(worker_infos, request_text=request_text, tokens=tokens)
        chosen = workers[idx]
        token_cost = _estimate_tokens(req)
        self._inflight[chosen.name] = self._inflight.get(chosen.name, 0) + token_cost
        return chosen

    def on_request_complete(self, worker_name: str, success: bool, req: GenericRequest | None  = None):
        if req is not None:
            token_cost = req.output_length if req.output_length > 0 else 0
            if worker_name in self._inflight:
                self._inflight[worker_name] = max(0, self._inflight[worker_name] - token_cost)

    def init_workers(self, workers):
        worker_infos = [self._PyWorkerInfo(url=w.name) for w in workers]
        self._policy.init_workers(worker_infos)
        self._inflight = {w.name: 0 for w in workers}

    def name(self) -> str:
        return self._policy.name()

    def reset(self):
        self._policy.reset()
        self._inflight = {k: 0 for k in self._inflight}
