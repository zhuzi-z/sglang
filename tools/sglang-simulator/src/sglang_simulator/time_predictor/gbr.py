"""GradientBoostingRegressor per-iter latency predictor (port of insight_benchmark gbr_predictor.py).

Trains GBR on schedule_batch JSONL data (forward_mode==1 = Prefill).
Decode returns a fixed small value (matches insight_benchmark behavior).

hisim_config.json usage:
    "predictor": {
        "name": "gbr",
        "database_path": "/path/to/schedule_batch1.jsonl,/path/to/schedule_batch2.jsonl",
        "model_path": "/optional/cache.joblib"
    }
"""
import glob
import json
import os
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.base import InferTimePredictor, ScheduleBatch
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


class GBRTimePredictor(InferTimePredictor):
    FEATURE_NAMES = [
        "batch_size",
        "total_extend_input_len",
        "total_prefix_indices_len",
        "max_extend_input_len",
        "min_extend_input_len",
        "mean_extend_input_len",
        "max_prefix_indices_len",
        "min_prefix_indices_len",
        "mean_prefix_indices_len",
        "total_tokens",
        "max_single_req_tokens",
    ]

    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        database_path: str,
        model_path: Optional[str] = None,
        decode_latency: float = 0.02,
        **kwargs,
    ):
        super().__init__(model, hw, config)
        self._decode_latency = float(decode_latency)

        if model_path and os.path.exists(model_path):
            import joblib
            self._regressor = joblib.load(model_path)
            logger.info(f"GBR loaded pre-trained model from {model_path}")
        else:
            paths = self._expand_paths(database_path)
            if not paths:
                raise FileNotFoundError(
                    f"GBR database_path expanded to 0 files: {database_path}"
                )
            X, y = self._load_features(paths)
            self._regressor = GradientBoostingRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
            self._regressor.fit(X, y)
            logger.info(f"GBR trained on {len(X)} samples from {len(paths)} files")
            if model_path:
                import joblib
                os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
                joblib.dump(self._regressor, model_path)
                logger.info(f"GBR cached to {model_path}")

    @staticmethod
    def _expand_paths(database_path: str) -> list:
        out = []
        for part in database_path.split(","):
            part = part.strip()
            if not part:
                continue
            if any(c in part for c in "*?["):
                out.extend(sorted(glob.glob(part)))
            elif os.path.exists(part):
                out.append(part)
        return out

    def _load_features(self, paths):
        X, y = [], []
        for p in paths:
            with open(p) as f:
                for line in f:
                    item = json.loads(line)
                    if item.get("forward_mode") != 1:
                        continue
                    reqs = item["request_infos"]
                    extend_lens = [r["extend_input_len"] for r in reqs]
                    prefix_lens = [r["prefix_indices_len"] for r in reqs]
                    X.append(self._agg(extend_lens, prefix_lens))
                    y.append(item["iter_latency"])
        if not X:
            raise RuntimeError(f"No Prefill samples in {paths}")
        return np.array(X), np.array(y)

    @staticmethod
    def _agg(extend_lens, prefix_lens):
        return [
            len(extend_lens),
            sum(extend_lens),
            sum(prefix_lens),
            max(extend_lens),
            min(extend_lens),
            float(np.mean(extend_lens)),
            max(prefix_lens),
            min(prefix_lens),
            float(np.mean(prefix_lens)),
            sum(extend_lens) + sum(prefix_lens),
            max(e + p for e, p in zip(extend_lens, prefix_lens)),
        ]

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0
        if batch.is_decode():
            return self._decode_latency
        extend_lens = [req.extend_length for req in batch.reqs]
        prefix_lens = [req.past_kv_length for req in batch.reqs]
        feats = np.array(self._agg(extend_lens, prefix_lens)).reshape(1, -1)
        return float(max(self._regressor.predict(feats)[0], 0.0))
