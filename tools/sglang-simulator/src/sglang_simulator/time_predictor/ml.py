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
    PP_FEATURE_NAMES = [
        "total_extend", "total_past", "batch_size",
        "max_extend", "min_extend", "mean_extend",
        "max_past", "min_past", "mean_past",
        "max_single_req_tokens", "total_tokens",
        "pos_in_burst", "in_flight", "num_waiting", "num_running",
        "logprobs_req_count", "logprobs_tokens", "logprobs_n",
    ]
    SUPPORTED_FEATURE_NAMES = frozenset(FEATURE_NAMES + PP_FEATURE_NAMES)

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
        # Form C (two-branch plateau, F-B): constant plateau of L ms below
        # the threshold, Form B's line at or above it. Mutually exclusive
        # with the hinge keys ONLY -- a/b are shared by Forms B and C.
        sample_tokens_plateau_ms: float = 0.0,
        sample_tokens_threshold_tokens: int = 0,
        full_span: bool = False,
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

        self._features = list(saved_features)
        unknown = [name for name in self._features if name not in self.SUPPORTED_FEATURE_NAMES]
        if unknown:
            raise ValueError(
                "MLTimePredictor unknown feature schema entries: "
                f"{unknown!r}; supported={sorted(self.SUPPORTED_FEATURE_NAMES)!r}"
            )
        # full_span labels already include last-stage sampling/MTP/logprobs.
        # The executor uses this flag to suppress additive RPC-2 compensations.
        self.full_span = bool(full_span)

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
        # predicted step time. Three calibrated forms are supported; the
        # caller selects one by passing the parameters of that form:
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
        # Form C -- two-branch plateau (F-B), the form measured on gen2 /
        # aarch64 / GB300 date-code runtimes:
        #
        #   sample_tokens_s = L                            if sum_ext <  T
        #                   = base + a * sum_ext / 1e3     if sum_ext >= T
        #                          + b * sum(ext_i*past_i) / 1e9
        #
        # The plateau branch is a CONSTANT: on gen2 steps below the
        # threshold, st shows no response to sum_ext, batch size, sum_past,
        # iter time or logprobs. ``L`` is therefore the WHOLE RPC-2 duration
        # below the threshold, NOT an additive offset on top of the line.
        # Migrating an F-A ``P`` value (the superseded additive plateau
        # term, ~18-22ms) into ``L`` underestimates the plateau by the line
        # value at T. Setting ``sample_tokens_plateau_ms = 0`` makes Form C
        # degenerate into Form B, which is what gen1 / x86 runtimes need
        # (their measured plateau is ~1ms).
        #
        # Every form is kept as a SEPARATE term: never folded into the
        # predictor's own output so the iter_latency label semantics stay
        # intact. Mixing the hinge triple with either other form is rejected
        # (they would silently combine the two trajectories); all-zero
        # parameters disable the compensation.
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
        _plateau_ms = float(sample_tokens_plateau_ms)
        _threshold_tokens = max(0, int(sample_tokens_threshold_tokens))
        if _plateau_ms < 0.0:
            raise ValueError(
                "MLTimePredictor: sample_tokens_plateau_ms must be >= 0 "
                f"(got {_plateau_ms}); a negative plateau has no physical "
                "meaning. Set 0 to disable the plateau branch."
            )
        _plateau_set = _plateau_ms != 0.0
        _hinge_set = any(v != 0 for v in _hinge_keys.values())
        _mech_set = any(v != 0 for v in _mech_keys.values())
        if _plateau_set and _hinge_set:
            raise ValueError(
                "MLTimePredictor: plateau-form and hinge-form sample_tokens "
                "parameters are mutually exclusive -- the plateau form "
                "(sample_tokens_plateau_ms/sample_tokens_threshold_tokens) "
                "already carries its own line via sample_tokens_base_ms/"
                "sample_tokens_a_ms_per_1k_ext/"
                "sample_tokens_b_ms_per_1g_ext_past."
            )
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
        self.sample_tokens_form = (
            "plateau_line"
            if _plateau_set
            else ("hinge" if _hinge_set else ("mechanistic" if _mech_set else None))
        )
        self.sample_tokens_base_s = float(sample_tokens_base_ms) / 1e3
        self.sample_tokens_plateau_s = _plateau_ms / 1e3
        self.sample_tokens_threshold_tokens = _threshold_tokens
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
        if self.sample_tokens_form in ("mechanistic", "plateau_line"):
            # plateau_line shares these two as its above-threshold line, so
            # they must survive the form switch; only the hinge form owns
            # its own lo/hi/breakpoint triple.
            self.sample_tokens_a_ms_per_1k_ext = float(sample_tokens_a_ms_per_1k_ext)
            self.sample_tokens_b_ms_per_1g_ext_past = float(
                sample_tokens_b_ms_per_1g_ext_past
            )
        else:
            self.sample_tokens_a_ms_per_1k_ext = 0.0
            self.sample_tokens_b_ms_per_1g_ext_past = 0.0
        if self.sample_tokens_form == "plateau_line" and (
            self.sample_tokens_base_s <= 0.0
            and self.sample_tokens_a_ms_per_1k_ext <= 0.0
        ):
            logger.warning(
                "MLTimePredictor: plateau_line form selected but its line is "
                "all-zero (base<=0 and a<=0); every step with sum_ext >= "
                "threshold will be compensated with 0ms. Fill "
                "sample_tokens_base_ms/sample_tokens_a_ms_per_1k_ext unless "
                "this is intentional."
            )
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
            elif self.sample_tokens_form == "mechanistic":
                logger.info(
                    "MLTimePredictor sample_tokens compensation enabled "
                    "(mechanistic form): base=%.3fms a=%.4fms/1k_ext "
                    "b=%.5fms/G_ext_past",
                    self.sample_tokens_base_s * 1e3,
                    self.sample_tokens_a_ms_per_1k_ext,
                    self.sample_tokens_b_ms_per_1g_ext_past,
                )
            elif self.sample_tokens_form == "plateau_line":
                logger.info(
                    "MLTimePredictor sample_tokens compensation enabled "
                    "(plateau_line form, F-B): plateau=%.3fms "
                    "threshold=%d tokens, line base=%.3fms "
                    "a=%.4fms/1k_ext b=%.5fms/G_ext_past",
                    self.sample_tokens_plateau_s * 1e3,
                    self.sample_tokens_threshold_tokens,
                    self.sample_tokens_base_s * 1e3,
                    self.sample_tokens_a_ms_per_1k_ext,
                    self.sample_tokens_b_ms_per_1g_ext_past,
                )
            else:
                logger.info(
                    "MLTimePredictor sample_tokens compensation enabled "
                    "(base only, no form selected): base=%.3fms",
                    self.sample_tokens_base_s * 1e3,
                )

    @staticmethod
    def _extract_features(batch: ScheduleBatch, feature_names=None) -> list[float]:
        """Extract features by name, preserving each bundle's own schema."""
        names = list(feature_names or MLTimePredictor.FEATURE_NAMES)
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
        per_req_tokens = [extend + past for extend, past in zip(exts, pasts)]
        values = {
            "batch_size": batch_size,
            "sum_extend": sum_extend,
            "total_extend": sum_extend,
            "max_extend": max_extend,
            "min_extend": min_extend,
            "mean_extend": sum_extend / batch_size,
            "sum_past": sum_past,
            "total_past": sum_past,
            "max_past": max_past,
            "min_past": min_past,
            "mean_past": sum_past / batch_size,
            "sum_extend_x_past": sum(e * p for e, p in zip(exts, pasts)),
            "sum_extend_squared": sum(e * e for e in exts),
            "sum_past_squared": sum(p * p for p in pasts),
            "sum_attn_flops": sum_attn_flops,
            "sum_extend_x_max_past": sum_extend * max_past,
            "log1p_sum_past": math.log1p(sum_past),
            "log1p_sum_attn_flops": math.log1p(sum_attn_flops),
            "batch_size_x_sum_extend": batch_size * sum_extend,
            "max_past_minus_min_past": max_past - min_past,
            "max_single_req_tokens": max(per_req_tokens),
            "total_tokens": sum_extend + sum_past,
            "is_decode": int(all(e == 1 for e in exts)),
            "is_prefill": int(any(e > 1 for e in exts)),
            "pos_in_burst": batch.pos_in_burst,
            "in_flight": batch.in_flight,
            "num_waiting": batch.num_waiting,
            "num_running": batch.num_running,
            "logprobs_req_count": batch.logprobs_req_count,
            "logprobs_tokens": batch.logprobs_tokens,
            "logprobs_n": batch.logprobs_n,
        }
        unknown = [name for name in names if name not in values]
        if unknown:
            raise ValueError(f"Unsupported predictor features: {unknown!r}")
        return [values[name] for name in names]

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0

        features = self._extract_features(batch, self._features)
        is_decode = all(req.extend_length == 1 for req in batch.reqs)
        model = self._decode_model if is_decode else self._prefill_model
        latency = float(model.predict([features])[0]) * self._latency_scale
        return max(latency, 0.0)

    def predict_sample_tokens_time(self, batch: ScheduleBatch) -> float:
        """Return the RPC-2 (sample_tokens) duration in seconds.

        Dispatches on the calibrated form selected at construction time:

        * plateau_line (F-B, gen2) -- a constant plateau of ``L`` ms while
          the step's ``sum_extend`` is below the threshold, then the
          mechanistic line above it::

              st = L                                    if se <  T
              st = base + a*sum_ext/1e3 + b*sum(ext_i*past_i)/1e9   if se >= T

          ``L`` is the WHOLE RPC-2 duration in the plateau, not an offset
          added on top of the line (that was the superseded F-A form). A
          step with ``sum_extend == T`` already belongs to the line: the
          measured shape is a step down AT T, not just before it. Setting
          ``L = 0`` degenerates into the mechanistic form.

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

        All calibrations were fitted on mixed (chunked-prefill) steps only,
        so applying them to a hypothetical decode-only deployment
        extrapolates the token-independent floor beyond its measured domain.
        The plateau form was additionally calibrated on gen2 / aarch64 /
        GB300 data with L in [29.5, 39.6] ms and T in [4736, 6656] tokens;
        both drift from pod to pod, so BOTH must be calibrated per serving
        instance, while the line slope may be shared across a generation.
        """
        if not self._sample_tokens_enabled or batch.is_empty():
            return 0.0

        if self.sample_tokens_form == "plateau_line":
            # F-B. Compare on the SAME sum_extend feature the calibration
            # used, then either return the constant plateau or evaluate the
            # mechanistic line, reusing the identical per-request
            # aggregation as _extract_features' "sum_extend_x_past".
            sum_extend = float(sum(req.extend_length for req in batch.reqs))
            if sum_extend < self.sample_tokens_threshold_tokens:
                return max(self.sample_tokens_plateau_s, 0.0)
            sum_extend_x_past = float(
                sum(req.extend_length * req.past_kv_length for req in batch.reqs)
            )
            latency_ms = (
                self.sample_tokens_base_s * 1e3
                + self.sample_tokens_a_ms_per_1k_ext * sum_extend / 1e3
                + self.sample_tokens_b_ms_per_1g_ext_past * sum_extend_x_past / 1e9
            )
            return max(latency_ms / 1e3, 0.0)

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

        features = self._extract_features(batch, self.FEATURE_NAMES)
        sum_extend = features[1]  # FEATURE_NAMES[1] == "sum_extend"
        sum_extend_x_past = features[7]  # FEATURE_NAMES[7] == "sum_extend_x_past"
        latency_ms = (
            self.sample_tokens_base_s * 1e3
            + self.sample_tokens_a_ms_per_1k_ext * sum_extend / 1e3
            + self.sample_tokens_b_ms_per_1g_ext_past * sum_extend_x_past / 1e9
        )
        return max(latency_ms / 1e3, 0.0)
