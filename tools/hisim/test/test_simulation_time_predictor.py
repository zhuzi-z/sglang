from hisim.spec.accelerator import AcceleratorInfo
from hisim.spec.model import ModelInfo

from hisim.time_predictor import (
    AIConfiguratorTimePredictor,
    ScheduleBatch,
    FakeRequest,
)
from hisim.simulation.types import (
    SchedulerConfig,
)


def test_time_predictor():
    model = ModelInfo(model_path="Qwen/Qwen3-8B")
    hw = AcceleratorInfo(
        name="a100_sxm",
        vendor="NVIDIA",
        hbm_capacity_gb=80,
        hbm_bandwidth_gb=2039,
        inter_node_bandwidth_gb=300,
        intra_node_bandwidth_gb=25,
    )
    config = SchedulerConfig(backend_name="sglang", backend_version="0.5.9")
    for clz in [
        AIConfiguratorTimePredictor,
    ]:
        predictor = clz(model, hw, config)

        # Prefill
        reqs = [
            FakeRequest(512, 512),
            FakeRequest(1024, 0),
            FakeRequest(512, 0),
        ]

        latency = predictor.predict_infer_time(ScheduleBatch(reqs))
        assert latency > 0

        # Decode
        reqs = [
            FakeRequest(1, 1024),
            FakeRequest(1, 1024),
            FakeRequest(1, 1024),
        ]

        latency = predictor.predict_infer_time(ScheduleBatch(reqs))
        assert latency > 0


if __name__ == "__main__":
    test_time_predictor()
