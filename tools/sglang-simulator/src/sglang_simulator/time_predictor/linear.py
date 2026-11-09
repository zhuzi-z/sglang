"""Deterministic step latency for smoke tests; not a calibrated performance model."""

import math

from sglang_simulator.time_predictor.base import InferTimePredictor


class LinearTimePredictor(InferTimePredictor):
    def __init__(self, model, hw, config, base_ms=0.0, per_token_ms=0.0):
        super().__init__(model, hw, config)
        values = (float(base_ms), float(per_token_ms))
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Linear predictor latencies must be finite and nonnegative")
        self.base_s, self.per_token_s = (value / 1000 for value in values)

    def predict_infer_time(self, batch):
        if batch.is_empty():
            return 0.0
        return self.base_s + self.per_token_s * batch.num_context_tokens
