"""ML-trained per-iteration latency predictor.

The predictor consumes the 18-feature sklearn bundle produced by the
Qwen3.7-Max baseline training pipeline.  It intentionally has no dependency on
the serving backend beyond the simulator's common ``ScheduleBatch`` interface.
"""

import math
import os
import warnings

import joblib
from sklearn.exceptions import InconsistentVersionWarning

from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.base import InferTimePredictor, ScheduleBatch
from sglang_simulator.utils import get_logger


logger = get_logger("sgl_simulator")


class MLTimePredictor(InferTimePredictor):
    """Predict a vLLM/SGLang step latency from batch-composition features."""

    FEATURE_NAMES = [
        "batch_size",
        "sum_extend",
        "max_extend",
        "min_extend",
        "sum_past",
        "max_past",
        "min_past",
        "sum_extend_x_past",
        "sum_extend_squared",
        "sum_past_squared",
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
        logprobs_cost_us_per_token: float = 0.0,
        sample_tokens_base_ms: float = 0.0,
        # Form A (legacy): single-variable hinge on total_tokens.
        sample_tokens_lo_us_per_token: float = 0.0,
        sample_tokens_hi_us_per_token: float = 0.0,
        sample_tokens_breakpoint_tokens: int = 0,
        # Form B (mechanistic): multi-variable linear on per-request sums.
        sample_tokens_a_ms_per_1k_ext: float = 0.0,
        sample_tokens_b_ms_per_1g_ext_past: float = 0.0,
        **kwargs,
    ):
        super().__init__(model, hw, config)
        if not database_path or not os.path.exists(database_path):
            raise FileNotFoundError(
                f"MLTimePredictor database_path not found: {database_path}"
            )

        # A sklearn ensemble can emit one version warning per tree.  Preserve
        # the compatibility signal without flooding a 500-tree simulation log.
        with warnings.catch_warnings(record=True) as version_warnings:
            warnings.simplefilter("always", InconsistentVersionWarning)
            bundle = joblib.load(database_path)
        if version_warnings:
            logger.warning(
                "MLTimePredictor loaded a model produced by another sklearn "
                "version (%d compatibility warnings suppressed)",
                len(version_warnings),
            )
        if isinstance(bundle, dict) and "prefill" in bundle and "decode" in bundle:
            self._prefill_model = bundle["prefill"]
            self._decode_model = bundle["decode"]
            saved_features = bundle.get("features", self.FEATURE_NAMES)
        elif isinstance(bundle, dict) and "model" in bundle:
            self._prefill_model = bundle["model"]
            self._decode_model = bundle["model"]
            saved_features = bundle.get("features", self.FEATURE_NAMES)
        else:
            self._prefill_model = bundle
            self._decode_model = bundle
            saved_features = self.FEATURE_NAMES

        if list(saved_features) != self.FEATURE_NAMES:
            raise ValueError(
                "MLTimePredictor feature schema mismatch: "
                f"saved={list(saved_features)!r}, expected={self.FEATURE_NAMES!r}"
            )

        self._latency_scale = float(latency_scale)
        # Extra linear cost for requests that enable logprobs (GPU topk /
        # D2H / tolists, proportional to tokens). Only needed when the
        # baseline was collected without logprobs but the target deployment
        # enables them; must be calibrated from independent measurement.
        self.logprobs_cost_s_per_token = float(logprobs_cost_us_per_token) / 1e6

        # RPC-2 (sample_tokens) compensation. The trained label above covers
        # only the first RPC (execute_model / main-model forward), so the
        # second RPC -- sampler + speculative (MTP) draft propose +
        # bookkeeping D2H + output construction -- is missing from the
        # predicted step time. Two mutually-exclusive calibrated forms are
        # supported; the caller selects one by passing its parameters:
        #
        # Form A -- legacy single-variable hinge on this step's total_tokens
        # (the scheduled token count written by the GPU-side collection
        # hook):
        #
        #   sample_tokens_s = base + a_lo * tt
        #                    + (a_hi - a_lo) * max(0, tt - breakpoint)
        #
        # Form B -- mechanistic multi-variable linear on the per-request
        # sums of the step (matches the 18-feature aggregation exactly):
        #
        #   sample_tokens_s = base + a * sum_ext / 1e3
        #                    + b * sum(ext_i * past_i) / 1e9
        #
        # ``base`` is the token-independent RPC-2 floor (sampler + small-step
        # kernel launches + D2H + RPC); in form B, ``a`` is the draft-model
        # linear-layer cost per 1k scheduled tokens and ``b`` the draft-model
        # attention cost per 1e9 sum(ext_i * past_i) -- the draft forward
        # attends the per-request KV history, so its cost scales with the
        # per-request sum of extend*past, NOT with the aggregate product
        # sum_ext * sum_past.
        #
        # Both forms are kept as a SEPARATE term: never folded into the
        # predictor's own output so the iter_latency label semantics stay
        # intact. Passing parameters of both forms at once is rejected (they
        # would silently combine); all-zero parameters disable the
        # compensation.
        _hinge_keys = {
            "sample_tokens_lo_us_per_token": float(sample_tokens_lo_us_per_token),
            "sample_tokens_hi_us_per_token": float(sample_tokens_hi_us_per_token),
            "sample_tokens_breakpoint_tokens": int(sample_tokens_breakpoint_tokens),
        }
        _mech_keys = {
            "sample_tokens_a_ms_per_1k_ext": float(sample_tokens_a_ms_per_1k_ext),
            "sample_tokens_b_ms_per_1g_ext_past": float(
                sample_tokens_b_ms_per_1g_ext_past
            ),
        }
        _hinge_set = any(v != 0 for v in _hinge_keys.values())
        _mech_set = any(v != 0 for v in _mech_keys.values())
        if _hinge_set and _mech_set:
            raise ValueError(
                "MLTimePredictor: hinge-form and mechanistic-form "
                "sample_tokens parameters are mutually exclusive -- pass one "
                "calibrated set only (hinge: sample_tokens_lo_us_per_token/"
                "sample_tokens_hi_us_per_token/"
                "sample_tokens_breakpoint_tokens; mechanistic: "
                "sample_tokens_a_ms_per_1k_ext/"
                "sample_tokens_b_ms_per_1g_ext_past)"
            )
        self.sample_tokens_form = "hinge" if _hinge_set else (
            "mechanistic" if _mech_set else None
        )
        self.sample_tokens_base_s = float(sample_tokens_base_ms) / 1e3
        if self.sample_tokens_form == "hinge":
            self.sample_tokens_lo_s_per_token = float(
                sample_tokens_lo_us_per_token
            ) / 1e6
            self.sample_tokens_hi_s_per_token = float(
                sample_tokens_hi_us_per_token
            ) / 1e6
            self.sample_tokens_breakpoint_tokens = max(
                0, int(sample_tokens_breakpoint_tokens)
            )
        else:
            self.sample_tokens_lo_s_per_token = 0.0
            self.sample_tokens_hi_s_per_token = 0.0
            self.sample_tokens_breakpoint_tokens = 0
        if self.sample_tokens_form == "mechanistic":
            self.sample_tokens_a_ms_per_1k_ext = float(sample_tokens_a_ms_per_1k_ext)
            self.sample_tokens_b_ms_per_1g_ext_past = float(
                sample_tokens_b_ms_per_1g_ext_past
            )
        else:
            self.sample_tokens_a_ms_per_1k_ext = 0.0
            self.sample_tokens_b_ms_per_1g_ext_past = 0.0
        self._sample_tokens_enabled = self.sample_tokens_form is not None or (
            self.sample_tokens_base_s > 0.0
        )
        logger.info(
            "MLTimePredictor loaded from %s (prefill=%s, decode=%s, "
            "n_features=%d, latency_scale=%.4f)",
            database_path,
            type(self._prefill_model).__name__,
            type(self._decode_model).__name__,
            len(saved_features),
            self._latency_scale,
        )
        if self._sample_tokens_enabled:
            if self.sample_tokens_form == "hinge":
                logger.info(
                    "MLTimePredictor sample_tokens compensation enabled "
                    "(hinge form): base=%.3fms a_lo=%.4fus/token "
                    "a_hi=%.4fus/token breakpoint=%d tokens",
                    self.sample_tokens_base_s * 1e3,
                    self.sample_tokens_lo_s_per_token * 1e6,
                    self.sample_tokens_hi_s_per_token * 1e6,
                    self.sample_tokens_breakpoint_tokens,
                )
            else:
                logger.info(
                    "MLTimePredictor sample_tokens compensation enabled "
                    "(mechanistic form): base=%.3fms a=%.4fms/1k_ext "
                    "b=%.5fms/G_ext_past",
                    self.sample_tokens_base_s * 1e3,
                    self.sample_tokens_a_ms_per_1k_ext,
                    self.sample_tokens_b_ms_per_1g_ext_past,
                )

    @staticmethod
    def _extract_features(batch: ScheduleBatch) -> list[float]:
        exts = [req.extend_length for req in batch.reqs]
        pasts = [req.past_kv_length for req in batch.reqs]

        batch_size = len(exts)
        sum_extend = sum(exts)
        sum_past = sum(pasts)
        max_extend = max(exts)
        min_extend = min(exts)
        max_past = max(pasts)
        min_past = min(pasts)
        sum_attn_flops = sum(
            extend * (past + extend / 2) for extend, past in zip(exts, pasts)
        )

        return [
            batch_size,
            sum_extend,
            max_extend,
            min_extend,
            sum_past,
            max_past,
            min_past,
            sum(extend * past for extend, past in zip(exts, pasts)),
            sum(extend * extend for extend in exts),
            sum(past * past for past in pasts),
            sum_attn_flops,
            sum_extend * max_past,
            math.log1p(sum_past),
            math.log1p(sum_attn_flops),
            batch_size * sum_extend,
            max_past - min_past,
            int(all(extend == 1 for extend in exts)),
            int(any(extend > 1 for extend in exts)),
        ]

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0

        features = self._extract_features(batch)
        is_decode = features[-2] == 1
        model = self._decode_model if is_decode else self._prefill_model
        latency = float(model.predict([features])[0]) * self._latency_scale
        return max(latency, 0.0)

    def predict_sample_tokens_time(self, batch: ScheduleBatch) -> float:
        """Return the RPC-2 (sample_tokens) duration in seconds.

        Dispatches on the calibrated form selected at construction time:

        * hinge (legacy) -- single-variable piecewise linear on the step's
          total scheduled tokens (``total_tokens``, same field the GPU-side
          collection hook writes)::

              st = base + a_lo*tt + (a_hi - a_lo)*max(0, tt - breakpoint)

        * mechanistic -- multi-variable linear on the per-request sums,
          re-derived from the SAME aggregation used by ``_extract_features``
          (``sum_extend`` and ``sum_extend_x_past``), so the compensation
          input always matches the GPU-side calibration口径 exactly. The b
          term is the PER-REQUEST sum of extend*past; the aggregate product
          ``sum_ext * sum_past`` inflates multi-request steps (up to ~9x)
          and is the exact bug the v1 calibration was refuted on::

              st = base + a*sum_ext/1e3 + b*sum(ext_i*past_i)/1e9

        Returns 0.0 when the compensation is not configured.

        Both calibrations were fitted on mixed (chunked-prefill) steps only,
        so applying them to a hypothetical decode-only deployment
        extrapolates the token-independent floor beyond its measured domain.
        """
        if not self._sample_tokens_enabled or batch.is_empty():
            return 0.0

        if self.sample_tokens_form == "hinge":
            tt = float(sum(req.extend_length for req in batch.reqs))
            latency = (
                self.sample_tokens_base_s
                + self.sample_tokens_lo_s_per_token * tt
            )
            if tt > self.sample_tokens_breakpoint_tokens:
                latency += (
                    self.sample_tokens_hi_s_per_token
                    - self.sample_tokens_lo_s_per_token
                ) * (tt - self.sample_tokens_breakpoint_tokens)
            return max(latency, 0.0)

        features = self._extract_features(batch)
        sum_extend = features[1]  # FEATURE_NAMES[1] == "sum_extend"
        sum_extend_x_past = features[7]  # FEATURE_NAMES[7] == "sum_extend_x_past"
        latency_ms = (
            self.sample_tokens_base_s * 1e3
            + self.sample_tokens_a_ms_per_1k_ext * sum_extend / 1e3
            + self.sample_tokens_b_ms_per_1g_ext_past * sum_extend_x_past / 1e9
        )
        return max(latency_ms / 1e3, 0.0)
