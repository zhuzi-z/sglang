"""TPU PP×PCP 逐 stage 前向时延预测器。

特征口径与采集训练一致：tpu_stage_22_v1（18 个批次组成特征 + 4 个原生 PCP
布局特征），最后一个 stage 额外带 4 个尾部特征（tpu_stage3_tail_26_v1）。
PCP 布局按 vllm_torchtpu ``_evaluate_streaming_query`` 计算：本批 query
行拼成一条连续 token 流按 interleave 轮转分给各 PCP rank，因此各 rank 的
有效 token 数只取决于本批 token 总数。
"""

import bisect
import math
import warnings
from dataclasses import dataclass

from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")

LEGACY_FEATURES = [
    "batch_size", "sum_extend", "max_extend", "min_extend", "sum_past",
    "max_past", "min_past", "sum_extend_x_past", "sum_extend_squared",
    "sum_past_squared", "sum_attn_flops", "sum_extend_x_max_past",
    "log1p_sum_past", "log1p_sum_attn_flops", "batch_size_x_sum_extend",
    "max_past_minus_min_past", "is_decode", "is_prefill",
]
TPU_FEATURES = [
    "global_padded_tokens", "local_padded_tokens",
    "pcp_max_valid_tokens", "pcp_min_valid_tokens",
]
FEATURE_NAMES = LEGACY_FEATURES + TPU_FEATURES
TAIL_FEATURES = [
    "prompt_complete_request_count", "prompt_complete_query_tokens",
    "tail_logprobs_request_count", "tail_logprobs_query_tokens",
]
MODEL_KIND = "effective_host_native_return_diagnostic"


@dataclass(frozen=True)
class StageSpan:
    """一个请求在本 step 的调度量；prompt_tokens 为 None 表示未知。"""

    extend_length: int
    past_kv_length: int
    prompt_tokens: int | None = None
    logprobs: bool = False


@dataclass(frozen=True)
class PcpLayout:
    pcp_size: int
    interleave_size: int
    num_tokens_paddings: tuple
    dp_target_bucket: int | None = None
    max_num_tokens: int | None = None

    def __post_init__(self):
        if self.pcp_size < 1 or self.interleave_size <= 0:
            raise ValueError(f"invalid PCP layout: {self}")
        if not self.num_tokens_paddings:
            raise ValueError("num_tokens_paddings is required for PCP padding")
        object.__setattr__(
            self, "num_tokens_paddings",
            tuple(sorted(int(p) for p in self.num_tokens_paddings)))

    def local_counts(self, total_tokens: int) -> list[int]:
        # 连续 token 流 [0, total) 中，rank r 拥有 [k*cycle + r*il, +il)
        il, cycle = self.interleave_size, self.pcp_size * self.interleave_size
        full, rest = divmod(int(total_tokens), cycle)
        return [full * il + min(max(rest - r * il, 0), il)
                for r in range(self.pcp_size)]

    def evaluate(self, total_tokens: int) -> dict:
        counts = self.local_counts(total_tokens)
        required = max(counts)
        paddings = self.num_tokens_paddings
        index = bisect.bisect_left(paddings, required)
        if index == len(paddings):
            raise ValueError(
                f"no PCP padding bucket >= {required}; paddings={paddings}")
        local = paddings[index]
        if self.dp_target_bucket is not None and self.dp_target_bucket > local:
            local = int(self.dp_target_bucket)
        if self.max_num_tokens is not None and local > self.max_num_tokens:
            raise ValueError(
                f"PCP local padded {local} exceeds max_num_tokens "
                f"{self.max_num_tokens}")
        return {
            "global_padded_tokens": local * self.pcp_size,
            "local_padded_tokens": local,
            "pcp_max_valid_tokens": required,
            "pcp_min_valid_tokens": min(counts),
        }


def stage_features(spans: list[StageSpan], layout: PcpLayout) -> dict:
    q = [int(s.extend_length) for s in spans]
    p = [int(s.past_kv_length) for s in spans]
    if not q or any(x <= 0 for x in q) or any(x < 0 for x in p):
        raise ValueError(f"invalid spans for stage features: {spans}")
    n, se, sp = len(q), sum(q), sum(p)
    attn = sum(x * (y + x / 2) for x, y in zip(q, p))
    values = [
        n, se, max(q), min(q), sp, max(p), min(p),
        sum(x * y for x, y in zip(q, p)), sum(x * x for x in q),
        sum(x * x for x in p), attn, se * max(p), math.log1p(sp),
        math.log1p(attn), n * se, max(p) - min(p),
        int(all(x == 1 for x in q)), int(any(x > 1 for x in q)),
    ]
    features = dict(zip(LEGACY_FEATURES, values))
    features.update(layout.evaluate(se))
    return features


def tail_features(spans: list[StageSpan]) -> dict:
    if any(s.prompt_tokens is None for s in spans):
        raise ValueError("prompt length is required for tail features")
    complete = [s for s in spans
                if s.past_kv_length + s.extend_length >= s.prompt_tokens]
    logs = [s for s in spans if s.logprobs]
    return dict(zip(TAIL_FEATURES, [
        len(complete), sum(s.extend_length for s in complete),
        len(logs), sum(s.extend_length for s in logs),
    ]))


class StageModel:
    """单个 stage 的前向窗口模型，契约校验与训练侧 StageObservationPredictor 一致。"""

    def __init__(self, bundle: dict, stage: int, last_stage: int):
        if (bundle.get("model_kind") != MODEL_KIND
                or bundle.get("production_usable") is not False):
            raise ValueError("unsupported TPU stage model contract")
        if bundle["stage"] != stage:
            raise ValueError(f"stage mismatch: bundle={bundle['stage']} want={stage}")
        expected = FEATURE_NAMES + (TAIL_FEATURES if stage == last_stage else [])
        if bundle["features"] != expected or bundle.get("time_unit") != "seconds":
            raise ValueError(f"stage {stage} feature/unit mismatch")
        self.stage, self.names, self.models = stage, expected, bundle["models"]

    @classmethod
    def from_file(cls, path: str, stage: int, last_stage: int) -> "StageModel":
        import joblib
        from sklearn.exceptions import InconsistentVersionWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            return cls(joblib.load(path), stage, last_stage)

    def predict(self, features: dict) -> float:
        x = [float(features[k]) for k in self.names]
        if not all(math.isfinite(v) for v in x):
            raise ValueError(f"non-finite stage features: {features}")
        return sum(max(float(m.predict([x])[0]), 0.0) for m in self.models)


def host_prep_s(config: dict | None, spans: list[StageSpan]) -> float | None:
    """各 stage 在 before_step 返回后构建本步输入的主机耗时；None 表示不建模。"""
    if not config:
        return None
    new = sum(1 for s in spans if s.past_kv_length == 0)
    ms = (config["intercept"] + config["per_request"] * len(spans)
          + config["per_new_request"] * new)
    return max(0.0, ms) / 1e3


def empty_step_host_s(config: dict | None, stages: int) -> tuple | None:
    """带状态的空 step 在各 stage worker 上 before_step 之后的主机耗时；None 表示不建模。"""
    ms = (config or {}).get("empty_step_ms")
    if ms is None:
        return None
    if len(ms) != stages or not all(math.isfinite(float(x)) for x in ms):
        raise ValueError(f"empty_step_ms needs {stages} finite values: {ms}")
    return tuple(max(0.0, float(x)) / 1e3 for x in ms)
