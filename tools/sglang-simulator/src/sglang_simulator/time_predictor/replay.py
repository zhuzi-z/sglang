"""Oracle lookup predictor — replays real GPU iter_latency from a pre-built table.

Used for diagnostics: validates whether sim discrepancy comes from predictor inaccuracy
vs structural sim bugs. When sim hit ratio still differs from real after using replay
(oracle latency), the gap is NOT from predictor.

Build the table from real bench's TP0.schedule_batch.jsonl:
    docker exec mry-dpsk-v4 python3 -c "
    import json, statistics
    from collections import defaultdict
    batches = [json.loads(l) for l in open('/host/oss_pull/.../TP0.schedule_batch.jsonl')]
    lookup = defaultdict(list)
    for b in batches:
        if b.get('forward_mode') in (1,2):
            lookup[json.dumps(sorted([(ri['extend_input_len'], ri['prefix_indices_len']) for ri in b['request_infos']]))].append(b['iter_latency'])
    json.dump({k:statistics.median(v) for k,v in lookup.items()}, open('/path/to/replay_table.json','w'))
    "

hisim_config.json usage:
    "predictor": {
        "name": "replay",
        "database_path": "/path/to/replay_table.json"
    }
"""
import json
import os

from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.base import InferTimePredictor, ScheduleBatch
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


class ReplayTimePredictor(InferTimePredictor):
    """Oracle lookup predictor: returns real GPU iter_latency for matching batch compositions.

    Compositions are matched exactly by sorted (extend_len, past_kv_len) tuples. On lookup
    miss the predictor returns 0.0 (and emits a debug log). For workloads where sim batch
    composition diverges from real (e.g., bursty max-tps), miss rate can be high.
    """

    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        database_path: str,
        miss_fallback_seconds: float = 0.0,
        **kwargs,
    ):
        super().__init__(model, hw, config)
        if not database_path or not os.path.exists(database_path):
            raise FileNotFoundError(
                f"ReplayTimePredictor database_path not found: {database_path}"
            )
        with open(database_path) as f:
            self._table = json.load(f)
        self._miss_fallback = miss_fallback_seconds
        self._hits = 0
        self._misses = 0
        logger.info(
            "ReplayTimePredictor loaded %d unique compositions from %s (miss fallback=%.4fs)",
            len(self._table), database_path, miss_fallback_seconds,
        )

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0
        key = json.dumps(sorted(
            [req.extend_length, req.past_kv_length] for req in batch.reqs
        ))
        v = self._table.get(key)
        if v is None:
            self._misses += 1
            return self._miss_fallback
        self._hits += 1
        return float(v)
