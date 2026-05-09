from copy import deepcopy

from sglang_simulator.time_predictor.base import InferTimePredictor


class InterpolatedDecodeTimePredictor(InferTimePredictor):
    """
    Speed up decode-time latency prediction by reducing calls to the inner predictor.

    Core idea:
    - During continuous decode, latency usually changes smoothly as KV cache grows.
    - Instead of invoking the inner predictor on every decode step, this predictor:
        1. samples the latency at the current decode position
        2. samples the latency after `sample_decode_steps` more decode tokens
        3. uses linear interpolation for intermediate decode steps

    This is effective because decode-phase execution is typically dominated by
    attention over an incrementally growing KV cache, and adjacent decode steps
    often have very similar latency characteristics.
    """

    def __init__(
        self,
        model,
        hw,
        config,
        predictor: InferTimePredictor,
        sample_decode_steps: int,
        *args,
        **kwargs,
    ):
        super().__init__(model, hw, config, *args, **kwargs)

        self._base_predictor = predictor
        self.sample_decode_steps = sample_decode_steps

        # Cached state for the current interpolation window
        self._last_batch_is_prefill = True
        self._cached_batch_size = 0
        self._window_start_total_kv = 0
        self._window_end_total_kv = 0
        self._window_start_latency = 0.0
        self._window_end_latency = 0.0
        self._window_step_index = 0

    def predict_infer_time(self, batch):
        """
        Predict inference latency for a batch.

        For decode batches, this method avoids frequent calls to the base predictor
        by reusing a cached interpolation window when possible.
        """
        if not batch.is_decode():
            self._last_batch_is_prefill = True
            return self._base_predictor.predict_infer_time(batch)

        current_total_kv = batch.total_past_kv_length

        if self._should_refresh_window(batch.batch_size, current_total_kv):
            self._refresh_interpolation_window(batch, current_total_kv)
            self._last_batch_is_prefill = False
            return self._window_start_latency

        self._window_step_index += 1
        return self._interpolate_latency()

    def _should_refresh_window(self, batch_size: int, current_total_kv: int) -> bool:
        """
        Decide whether the interpolation window must be rebuilt.

        A refresh is required when:
        - batch size changes
        - KV cache length moves outside the cached interpolation range
        """
        return (
            self._last_batch_is_prefill
            or batch_size != self._cached_batch_size
            or current_total_kv < self._window_start_total_kv
            or current_total_kv > self._window_end_total_kv
        )

    def _refresh_interpolation_window(self, batch, current_total_kv: int) -> None:
        """
        Rebuild the interpolation window using two exact predictions:
        - latency at current decode position
        - latency after `sample_decode_steps` additional decode tokens
        """
        self._cached_batch_size = batch.batch_size
        self._window_start_total_kv = current_total_kv
        self._window_start_latency = self._base_predictor.predict_infer_time(batch)

        future_batch = deepcopy(batch)
        for req in future_batch.reqs:
            req.past_kv_length += self.sample_decode_steps

        self._window_end_latency = self._base_predictor.predict_infer_time(future_batch)
        self._window_end_total_kv = future_batch.total_past_kv_length
        self._window_step_index = 0

    def _interpolate_latency(self) -> float:
        """
        Estimate latency inside the current interpolation window.

        Linear interpolation is sufficient here because decode latency usually
        evolves smoothly across nearby decode steps.
        """
        progress = self._window_step_index / self.sample_decode_steps
        return (
            self._window_start_latency
            + (self._window_end_latency - self._window_start_latency) * progress
        )
