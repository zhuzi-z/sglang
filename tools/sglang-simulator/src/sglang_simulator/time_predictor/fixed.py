from sglang_simulator.time_predictor.base import (
    InferTimePredictor,
    ScheduleBatch,
)
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


class FixedTimePredictor(InferTimePredictor):
    """Constant-time predictor used to validate the simulator end-to-end without
    a real per-op latency database.

    Latency model:
        prefill: prefill_overhead_s + prefill_s_per_token * total_extend_tokens
        decode:  decode_s_per_request * batch_size

    All values are in seconds. The defaults below are rough estimates for a
    9B-class model on H20; calibrate per-deployment via the JSON config.
    """

    def __init__(
        self,
        model,
        hw,
        config,
        *,
        prefill_ms_per_token: float = 0.05,
        prefill_overhead_ms: float = 5.0,
        decode_ms_per_request: float = 8.0,
        decode_overhead_ms: float = 0.0,
        **kwargs,
    ):
        super().__init__(model=model, hw=hw, config=config)
        self.prefill_s_per_token = float(prefill_ms_per_token) / 1000.0
        self.prefill_overhead_s = float(prefill_overhead_ms) / 1000.0
        self.decode_s_per_request = float(decode_ms_per_request) / 1000.0
        self.decode_overhead_s = float(decode_overhead_ms) / 1000.0
        logger.info(
            "FixedTimePredictor: prefill=%.3f ms/token + %.3f ms overhead; "
            "decode=%.3f ms/request + %.3f ms overhead",
            prefill_ms_per_token,
            prefill_overhead_ms,
            decode_ms_per_request,
            decode_overhead_ms,
        )

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0
        if batch.is_decode():
            return self.decode_overhead_s + self.decode_s_per_request * batch.batch_size
        # prefill / extend
        return (
            self.prefill_overhead_s
            + self.prefill_s_per_token * batch.num_context_tokens
        )

    def reset_state(self):
        pass
