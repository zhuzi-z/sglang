"""TPU PP×PCP 逐 stage 前向时延预测器（HiSim 配置入口）。

特征与模型实现在 simulation.vllm.tpu_pp.stage_model，stage worker 进程直接
从那里导入，避免经 time_predictor 包连带导入其他预测器的依赖。
"""

import math

from sglang_simulator.simulation.vllm.tpu_pp.stage_model import (
    PcpLayout, StageModel, StageSpan, stage_features, tail_features,
)
from sglang_simulator.time_predictor.base import InferTimePredictor, ScheduleBatch


class TpuPPStageTimePredictor(InferTimePredictor):
    """只承载配置；逐 stage 推理在 stage worker 进程内完成。"""

    tpu_pp_stage = True

    def __init__(self, model, hw, config, stage_model_paths: list[str],
                 pcp_size: int, interleave_size: int,
                 num_tokens_paddings: list[int], communication_s: float,
                 host_prep: dict | None = None,
                 dp_target_bucket: int | None = None,
                 max_num_tokens: int | None = None,
                 async_handoff: bool = True):
        super().__init__(model, hw, config)
        if len(stage_model_paths) != config.pp_size:
            raise ValueError(
                f"need {config.pp_size} stage models, got {len(stage_model_paths)}")
        if config.pcp_size != pcp_size:
            raise ValueError(
                f"predictor pcp_size={pcp_size} != deployment pcp_size={config.pcp_size}")
        if not math.isfinite(communication_s) or communication_s < 0:
            raise ValueError(f"invalid communication_s={communication_s}")
        self.stage_model_paths = list(stage_model_paths)
        self.layout = PcpLayout(pcp_size, interleave_size,
                                tuple(num_tokens_paddings), dp_target_bucket,
                                max_num_tokens)
        self.communication_s = float(communication_s)
        self.host_prep = host_prep
        self.async_handoff = bool(async_handoff)
        # 主进程只做一次加载校验，worker 进程各自加载本 stage 模型
        last = config.pp_size - 1
        for stage, path in enumerate(self.stage_model_paths):
            StageModel.from_file(path, stage, last)

    def features_for(self, spans: list[StageSpan]) -> tuple[dict, dict | None]:
        base = stage_features(spans, self.layout)
        tail = tail_features(spans)
        return base, tail

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        raise NotImplementedError(
            "tpu_pp_stage predicts per PP stage inside the TPU PP runtime; "
            "it has no single-span latency")
