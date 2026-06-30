"""GBR (GradientBoostingRegressor) inference time predictor.

Migrated from insight_benchmark (zhouhaizhu.zhz) to sglang_simulator.
Only predicts Prefill latency; returns 0.02s for Decode batches.
"""

import json
import os

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from sglang_simulator.time_predictor.base import (
    InferTimePredictor,
    ScheduleBatch,
)
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


class GBRTimePredictor(InferTimePredictor):
    """
    GradientBoostingRegressor-based inference time predictor using aggregated features.

    Only predicts Prefill latency. Returns 0.02s for Decode batches.

    Features extracted from ScheduleBatch (11 dims):
        - batch_size
        - total_extend_input_len (sum of input_length)
        - total_prefix_indices_len (sum of past_kv_length)
        - max_extend_input_len
        - min_extend_input_len
        - mean_extend_input_len
        - max_prefix_indices_len
        - min_prefix_indices_len
        - mean_prefix_indices_len
        - total_tokens (total_extend + total_prefix)
        - max_single_req_tokens (max of input_length + past_kv_length per req)

    The model is trained on schedule_batch JSONL data (forward_mode==1 for Prefill).
    """

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
        model,
        hw,
        config,
        database_path: str,
        model_path: str | None = None,
        gbr_params: dict | None = None,
        decode_latency: float = 0.02,
        *args,
        **kwargs,
    ):
        """
        Args:
            model: ModelInfo instance.
            hw: AcceleratorInfo instance.
            config: SchedulerConfig instance.
            database_path: Comma-separated paths to schedule_batch JSONL files for training.
            model_path: Optional path to a pre-trained model (.joblib).
                        If provided and exists, load directly without training.
            gbr_params: Optional GradientBoostingRegressor parameters override.
            decode_latency: Fixed decode latency in seconds (default 0.02).
        """
        super().__init__(model, hw, config, *args, **kwargs)

        self.decode_latency = decode_latency
        self.gbr_params = gbr_params or {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "random_state": 42,
        }

        if model_path and os.path.exists(model_path):
            import joblib

            self.regressor = joblib.load(model_path)
            logger.info(f"Loaded pre-trained GBR model from {model_path}")
        else:
            X, y = self._load_and_extract_features(database_path)
            self.regressor = self._train(X, y)
            if model_path:
                import joblib

                joblib.dump(self.regressor, model_path)
                logger.info(f"Saved trained GBR model to {model_path}")

    def _load_and_extract_features(
        self, database_path: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load JSONL data and extract aggregated features for Prefill batches."""
        records_X = []
        records_y = []

        for db_path in database_path.split(","):
            db_path = db_path.strip()
            if not os.path.exists(db_path):
                raise RuntimeError(f"{db_path} does not exist.")
            with open(db_path) as f:
                for line in f:
                    item = json.loads(line)
                    # Filter: only Prefill (forward_mode == 1)
                    if item.get("forward_mode") != 1:
                        continue
                    reqs = item["request_infos"]
                    features = self._extract_features_from_raw(reqs)
                    records_X.append(features)
                    records_y.append(item["iter_latency"])

        if not records_X:
            raise RuntimeError(
                f"No Prefill samples found in {database_path}. "
                "Ensure the data contains forward_mode==1 records."
            )

        logger.info(
            f"Loaded {len(records_X)} Prefill samples from {database_path} for GBR training."
        )
        return np.array(records_X), np.array(records_y)

    def _extract_features_from_raw(self, request_infos: list[dict]) -> list[float]:
        """Extract aggregated features from raw JSONL request_infos."""
        extend_lens = [r["extend_input_len"] for r in request_infos]
        prefix_lens = [r["prefix_indices_len"] for r in request_infos]
        return self._compute_aggregated_features(extend_lens, prefix_lens)

    def _extract_features_from_batch(self, batch: ScheduleBatch) -> list[float]:
        """Extract aggregated features from a ScheduleBatch."""
        extend_lens = [req.extend_length for req in batch.reqs]
        prefix_lens = [req.past_kv_length for req in batch.reqs]
        return self._compute_aggregated_features(extend_lens, prefix_lens)

    @staticmethod
    def _compute_aggregated_features(
        extend_lens: list[int], prefix_lens: list[int]
    ) -> list[float]:
        """Compute the 11 aggregated features from extend/prefix length lists."""
        batch_size = len(extend_lens)
        total_extend = sum(extend_lens)
        total_prefix = sum(prefix_lens)
        return [
            batch_size,
            total_extend,
            total_prefix,
            max(extend_lens),
            min(extend_lens),
            float(np.mean(extend_lens)),
            max(prefix_lens) if prefix_lens else 0,
            min(prefix_lens) if prefix_lens else 0,
            float(np.mean(prefix_lens)) if prefix_lens else 0.0,
            total_extend + total_prefix,
            max(e + p for e, p in zip(extend_lens, prefix_lens)),
        ]

    def _train(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
        """Train the GBR regressor."""
        regressor = GradientBoostingRegressor(**self.gbr_params)
        regressor.fit(X, y)
        logger.info(
            f"GBR model training completed. "
            f"Samples={len(X)}, Features={X.shape[1]}."
        )
        return regressor

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        """
        Predict inference time (in seconds) for a given ScheduleBatch.
        Only supports Prefill batches. Returns fixed decode_latency for Decode.
        """
        if batch.is_empty():
            return 0.0

        if batch.is_decode():
            return self.decode_latency

        features = self._extract_features_from_batch(batch)
        X = np.array(features).reshape(1, -1)
        latency = self.regressor.predict(X)[0]
        return float(max(latency, 0.0))

    def save_model(self, path: str):
        """Save the trained GBR model to file."""
        import joblib
        joblib.dump(self.regressor, path)
        logger.info(f"GBR model saved to {path}")

    def load_model(self, path: str):
        """Load a pre-trained GBR model from file."""
        import joblib
        self.regressor = joblib.load(path)
        logger.info(f"GBR model loaded from {path}")
