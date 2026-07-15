"""ML-trained per-iter latency predictor.

Loads a joblib pickle of a sklearn-compatible regressor and predicts forward latency
from batch composition features. Train one with `train_latency_model.py`.

Supports two bundle formats:
  - Single model:  {"model": regressor, "features": [...]}
  - Dual model:    {"prefill": regressor, "decode": regressor, "features": [...]}
    The dual format routes decode batches (all extend==1) to the decode model
    and everything else to the prefill model, avoiding scale-mismatch issues.

hisim_config.json usage:
    "predictor": {
        "name": "ml",
        "database_path": "/path/to/latency_model.pkl"
    }
"""
import math
import os
from typing import Optional

import joblib

from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.base import InferTimePredictor, ScheduleBatch
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


class MLTimePredictor(InferTimePredictor):
    """Per-iter latency predictor backed by an offline-trained sklearn regressor.

    Features (18 dim) extracted from ScheduleBatch:
        batch_size, sum/max/min(extend), sum/max/min(past),
        sum(extend*past), sum(extend^2), sum(past^2),
        sum_attn_flops (= sum(e*(p+e/2))),
        sum(extend × max_past), log1p(sum_past), log1p(sum_attn_flops),
        batch_size × sum_extend, max_past - min_past,
        is_decode, is_prefill
    """

    FEATURE_NAMES = [
        "batch_size", "sum_extend", "max_extend", "min_extend",
        "sum_past", "max_past", "min_past",
        "sum_extend_x_past", "sum_extend_squared", "sum_past_squared",
        "sum_attn_flops",
        "sum_extend_x_max_past",
        "log1p_sum_past",
        "log1p_sum_attn_flops",
        "batch_size_x_sum_extend",
        "max_past_minus_min_past",
        "is_decode",
        "is_prefill",
    ]

    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        database_path: str,
        latency_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(model, hw, config)
        if not database_path or not os.path.exists(database_path):
            raise FileNotFoundError(
                f"MLTimePredictor database_path not found: {database_path}. "
                "Train one with `train_latency_model.py` first."
            )

        bundle = joblib.load(database_path)
        self._prefill_model = None
        self._decode_model = None

        if isinstance(bundle, dict) and "prefill" in bundle and "decode" in bundle:
            # Dual-model bundle: separate prefill/decode regressors
            self._prefill_model = bundle["prefill"]
            self._decode_model = bundle["decode"]
            saved_features = bundle.get("features", self.FEATURE_NAMES)
            logger.info(
                "MLTimePredictor dual-model loaded from %s "
                "(prefill=%s, decode=%s, n_features=%d, latency_scale=%.4f)",
                database_path,
                type(self._prefill_model).__name__,
                type(self._decode_model).__name__,
                len(saved_features), float(latency_scale),
            )
        elif isinstance(bundle, dict) and "model" in bundle:
            # Single-model bundle (backward compatible)
            self._prefill_model = bundle["model"]
            self._decode_model = bundle["model"]
            saved_features = bundle.get("features", self.FEATURE_NAMES)
            logger.info(
                "MLTimePredictor loaded from %s (model=%s, n_features=%d, latency_scale=%.4f)",
                database_path, type(self._prefill_model).__name__,
                len(saved_features), float(latency_scale),
            )
        else:
            # Bare regressor (backward compatible)
            self._prefill_model = bundle
            self._decode_model = bundle
            saved_features = self.FEATURE_NAMES
            logger.info(
                "MLTimePredictor loaded from %s (model=%s, n_features=%d, latency_scale=%.4f)",
                database_path, type(self._prefill_model).__name__,
                len(saved_features), float(latency_scale),
            )

        if list(saved_features) != list(self.FEATURE_NAMES):
            logger.warning(
                "MLTimePredictor: feature list mismatch — saved=%d, expected=%d. "
                "Using saved order; train_latency_model.py and this class must stay in sync.",
                len(saved_features), len(self.FEATURE_NAMES),
            )
        self._features = saved_features
        self._call_count = 0
        self._latency_scale = float(latency_scale)

    @staticmethod
    def _extract_features(batch: ScheduleBatch) -> list[float]:
        """Extract 18-dim feature vector from a ScheduleBatch."""
        exts = [req.extend_length for req in batch.reqs]
        pasts = [req.past_kv_length for req in batch.reqs]

        bs = len(exts)
        sum_e = sum(exts); sum_p = sum(pasts)
        sum_ep = sum(e * p for e, p in zip(exts, pasts))
        sum_e2 = sum(e * e for e in exts)
        sum_p2 = sum(p * p for p in pasts)
        sum_attn = sum(e * (p + e / 2) for e, p in zip(exts, pasts))
        max_e = max(exts); max_p = max(pasts)
        min_e = min(exts); min_p = min(pasts)

        return [
            bs, sum_e, max_e, min_e,
            sum_p, max_p, min_p,
            sum_ep, sum_e2, sum_p2,
            sum_attn,
            sum_e * max_p,
            math.log1p(sum_p),
            math.log1p(sum_attn),
            bs * sum_e,
            max_p - min_p,
            int(all(e == 1 for e in exts)),
            int(any(e > 1 for e in exts)),
        ]

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0

        feats = self._extract_features(batch)
        is_decode = feats[-2] == 1  # is_decode feature
        model = self._decode_model if is_decode else self._prefill_model

        self._call_count += 1
        return float(model.predict([feats])[0]) * self._latency_scale