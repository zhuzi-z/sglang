from sglang_simulator.simulation.sglang.utils import resolve_model_info
from sglang_simulator.simulation.types import (
    SchedulerConfig,
)
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor import (
    AIConfiguratorTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)

from sglang.srt.configs.model_config import ModelConfig


def test_time_predictor():
    model: ModelInfo = resolve_model_info(ModelConfig(model_path="Qwen/Qwen3-8B"))
    hw = AcceleratorInfo(
        name="a100_sxm",
        vendor="NVIDIA",
        hbm_capacity_gb=80,
        hbm_bandwidth_gb=2039,
        inter_node_bandwidth_gb=25,
        intra_node_bandwidth_gb=300,
    )
    config = SchedulerConfig(backend_name="sglang", backend_version="0.5.9")
    for clz in [
        AIConfiguratorTimePredictor,
    ]:
        predictor = clz(model, hw, config)

        # Prefill
        reqs = [
            ScheduleRequest(512, 512),
            ScheduleRequest(1024, 0),
            ScheduleRequest(512, 0),
        ]

        latency = predictor.predict_infer_time(ScheduleBatch(reqs))
        assert latency > 0

        # Decode
        reqs = [
            ScheduleRequest(1, 1024),
            ScheduleRequest(1, 1024),
            ScheduleRequest(1, 1024),
        ]

        latency = predictor.predict_infer_time(ScheduleBatch(reqs))
        assert latency > 0


def test_pp_feature_schema_and_legacy_compatibility():
    from sglang_simulator.time_predictor.ml import MLTimePredictor

    batch = ScheduleBatch(
        reqs=[ScheduleRequest(4, 8), ScheduleRequest(2, 4)],
        pos_in_burst=3,
        in_flight=2,
        num_waiting=5,
        num_running=6,
        logprobs_req_count=1,
        logprobs_tokens=4,
        logprobs_n=2,
    )
    legacy = MLTimePredictor._extract_features(batch)
    assert legacy == MLTimePredictor._extract_features(
        batch, MLTimePredictor.FEATURE_NAMES
    )
    assert legacy[:7] == [2, 6, 4, 2, 12, 8, 4]
    assert legacy[-2:] == [0, 1]

    pp = MLTimePredictor._extract_features(
        batch, MLTimePredictor.PP_FEATURE_NAMES
    )
    assert pp == [
        6, 12, 2, 4, 2, 3.0, 8, 4, 6.0, 12, 18,
        3, 2, 5, 6, 1, 4, 2,
    ]


def test_tpu_external_hit_metadata_uses_min_over_groups():
    from types import SimpleNamespace
    from sglang_simulator.simulation.vllm.engine_core_pipeline import (
        _tpu_external_hit_tokens,
    )

    metadata = SimpleNamespace(
        metadata=(
            SimpleNamespace(
                reqs_to_load={
                    "r": {
                        "attn": ([1, 2, 3], [10, 11, 12]),
                        "mamba": ([4, 5], [20, 21]),
                    }
                }
            ),
        )
    )
    output = SimpleNamespace(kv_connector_metadata=metadata)
    assert _tpu_external_hit_tokens(output, "r", 1000, 384) == 768
    assert _tpu_external_hit_tokens(output, "missing", 1000, 384) is None


def test_pp_stepper_dispatches_concurrently_and_completes_in_order():
    from concurrent.futures import Future
    from contextlib import contextmanager
    from types import SimpleNamespace
    from sglang_simulator.simulation.manager import StateManager
    from sglang_simulator.simulation.types import SimulationMode
    from sglang_simulator.simulation.vllm.engine_core_pipeline import (
        C_VLLMEngineCoreHook,
    )
    from sglang_simulator.simulation.vllm.pp_pipeline import (
        SimulatedPipelineStepper,
    )

    def done(value):
        future = Future()
        future.set_result(value)
        return future

    class Scheduler:
        def __init__(self):
            self.remaining = 5
            self.waiting = [1, 2, 3]
            self.running = [4]
            self.skipped_waiting = []
            self.outputs = []

        def has_requests(self):
            return self.remaining > 0

        def get_num_unfinished_requests(self):
            return self.remaining

        def schedule(self, _throttle):
            self.remaining -= 1
            rid = f"r{5 - self.remaining}"
            return SimpleNamespace(
                total_num_scheduled_tokens=1,
                num_scheduled_tokens={rid: 1},
                kv_connector_metadata=None,
                finished_req_ids=set(),
            )

        def get_grammar_bitmask(self, _output):
            return None

        def update_from_output(self, output, model_output):
            self.outputs.append((output, model_output))
            return {"completed": len(self.outputs)}

    class Executor:
        def execute_model(self, output, non_block=False):
            output._sim_full_step_latency = 4.0
            return done(None)

        def sample_tokens(self, _grammar, non_block=False):
            return done({"mock": True})

    class Core:
        def __init__(self):
            self.scheduler = Scheduler()
            self.model_executor = Executor()

        def _should_throttle_prefills(self):
            return False

        def _process_aborts_queue(self):
            pass

        @contextmanager
        def capture_iteration_details(self, _output):
            yield None

        @contextmanager
        def log_error_detail(self, _output):
            yield

        def _attach_iteration_details(self, _outputs, _details):
            pass

    StateManager.reset()
    C_VLLMEngineCoreHook.SIM_MODE = SimulationMode.OFFLINE
    core = Core()
    stepper = SimulatedPipelineStepper(core, stages=4)
    contexts = []
    completed = []
    for now in range(4):
        StateManager.set_global_clock(now)
        outputs, _ = stepper.step()
        assert outputs is None
        latest = core.scheduler and stepper.in_flight[-1].scheduler_output
        contexts.append(latest._sim_pp_context.copy())

    assert [c["pos_in_burst"] for c in contexts] == [0, 1, 2, 3]
    assert [c["in_flight"] for c in contexts] == [1, 2, 3, 4]
    assert stepper._real_in_flight() == 4

    StateManager.set_global_clock(4)
    outputs, _ = stepper.step()
    completed.append(outputs)
    assert outputs == {"completed": 1}
    # The fifth step is dispatched in the same call, so its burst continues.
    assert stepper._real_in_flight() == 4
    assert stepper.in_flight[-1].scheduler_output._sim_pp_context["pos_in_burst"] == 4

    for now in (5, 6, 7, 8):
        StateManager.set_global_clock(now)
        outputs, _ = stepper.step()
        completed.append(outputs)
    assert completed == [
        {"completed": 1}, {"completed": 2}, {"completed": 3},
        {"completed": 4}, {"completed": 5},
    ]
    assert not stepper.in_flight
    assert not stepper.burst_open


def test_pcp_chunk_budget_alignment_only_caps_when_enabled():
    from types import SimpleNamespace
    from sglang_simulator.simulation.manager import ConfigManager
    from sglang_simulator.simulation.vllm.engine_core_pipeline import (
        C_VLLMSchedulerHook,
    )

    class Scheduler:
        def get_num_unfinished_requests(self):
            return 0

        def has_unfinished_requests(self):
            return False

        def _mamba_block_aligned_split(self, _request, tokens, *_args):
            return tokens

    C_VLLMSchedulerHook.hook(Scheduler)
    scheduler = Scheduler()
    scheduler.cache_config = SimpleNamespace(block_size=384)
    scheduler.scheduler_config = SimpleNamespace(max_num_batched_tokens=8192)

    ConfigManager._scheduler_config = SchedulerConfig(pcp_size=8)
    assert scheduler._mamba_block_aligned_split(None, 8192) == 6144
    ConfigManager._scheduler_config = SchedulerConfig(pcp_size=1)
    assert scheduler._mamba_block_aligned_split(None, 8192) == 8192


def test_pp_executor_uses_full_span_without_duplicate_compensation():
    from types import SimpleNamespace
    from sglang_simulator.simulation.types import SimulationMode
    from sglang_simulator.simulation.vllm.engine_core_pipeline import (
        C_VLLMEngineCoreHook,
        C_VLLMExecutorHook,
    )

    class Predictor:
        full_span = True
        logprobs_cost_s_per_token = 10.0

        def predict_infer_time(self, _batch):
            return 2.0

        def predict_sample_tokens_time(self, _batch):
            return 100.0

    class Executor:
        def __init__(self):
            self.vllm_config = SimpleNamespace(
                cache_config=SimpleNamespace(block_size=384)
            )

        def execute_model(self, _output):
            return "mock-output"

    C_VLLMExecutorHook.hook(Executor)
    C_VLLMEngineCoreHook.INFERENCE_PREDICTOR = Predictor()
    C_VLLMEngineCoreHook.SIM_MODE = SimulationMode.OFFLINE
    output = SimpleNamespace(
        num_scheduled_tokens={"r": 4},
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="r",
                num_computed_tokens=8,
                prompt_token_ids=list(range(12)),
                sampling_params=SimpleNamespace(logprobs=2, max_tokens=1),
            )
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], num_computed_tokens=[]
        ),
        finished_req_ids=set(),
        kv_connector_metadata=None,
        _sim_pp_context={
            "pos_in_burst": 2,
            "in_flight": 3,
            "num_waiting": 4,
            "num_running": 5,
            "num_skipped": 1,
        },
    )
    assert Executor().execute_model(output) == "mock-output"
    assert output._sim_full_step_latency == 2.0
    stat = output._sim_iteration_stat
    assert stat["sample_tokens_latency"] == 0.0
    assert stat["logprobs_latency"] == 0.0
    assert stat["pos_in_burst"] == 2
    assert stat["in_flight"] == 3
    assert stat["num_waiting"] == 4
    assert stat["num_running"] == 5
    assert output._sim_batch.logprobs_req_count == 1
    assert output._sim_batch.logprobs_tokens == 4
    assert output._sim_batch.logprobs_n == 2


def test_tpu_connector_load_store_lifecycle_drains_completion_sets():
    from types import SimpleNamespace
    from sglang_simulator.simulation.vllm.v6d.tpu_connector import (
        C_TPUV6dObjectConnectorWorkerHook,
    )

    class Worker:
        def start_load_kv(self, _metadata):
            raise AssertionError("real data path must be replaced")

        def start_store_kv(self, _metadata):
            raise AssertionError("real data path must be replaced")

        def get_finished(self, _finished_req_ids=None):
            raise AssertionError("real completion path must be replaced")

        def has_pending_push_work(self):
            return True

        def register_kv_caches(self, _caches):
            self.native_register_called = True

    C_TPUV6dObjectConnectorWorkerHook.hook(Worker)
    worker = Worker()
    worker.register_kv_caches(["metadata-only"])
    assert not hasattr(worker, "native_register_called")
    assert worker.kv_caches == ["metadata-only"]
    assert worker.get_finished() == (set(), set())

    worker.start_load_kv(
        SimpleNamespace(
            reqs_to_load={
                "r": {"attn": (["key"], [1])},
                "empty": {"attn": ([], [])},
            }
        )
    )
    assert worker.has_pending_push_work()
    assert worker.get_finished() == (set(), {"r"})
    assert not worker.has_pending_push_work()

    worker.start_store_kv(
        SimpleNamespace(reqs_to_store={"r": ([], True, "batch")})
    )
    assert worker.has_pending_push_work()
    assert worker.get_finished() == ({"r"}, set())
    assert worker.get_finished() == (set(), set())


def test_raiden_process_send_load_uses_producer_and_consumer_contracts():
    from types import SimpleNamespace
    from sglang_simulator.simulation.vllm.v6d.tpu_connector import (
        C_TPURaidenConnectorWorkerHook,
    )

    class Worker:
        def __init__(self, is_producer):
            self.is_producer = is_producer

        def process_send_load(self, _metadata, **_kwargs):
            raise AssertionError("real Raiden engine must not be called")

        def get_finished(self, _finished_req_ids=None):
            raise AssertionError("real completion path must be replaced")

        def has_pending_push_work(self):
            return True

    C_TPURaidenConnectorWorkerHook.hook(Worker)
    producer = Worker(True)
    producer.process_send_load(SimpleNamespace(reqs_to_send={"p": object()}))
    assert producer.get_finished() == ({"p"}, set())

    consumer = Worker(False)
    consumer.process_send_load(
        SimpleNamespace(
            reqs_to_load={
                "c": SimpleNamespace(report_completion=True),
                "suppressed": SimpleNamespace(report_completion=False),
            }
        )
    )
    assert consumer.get_finished() == (set(), {"c"})
    consumer.process_send_load(
        SimpleNamespace(
            reqs_to_load={"ignored": SimpleNamespace(report_completion=True)}
        ),
        report_completion=False,
    )
    assert consumer.get_finished() == (set(), set())


def test_qwen_mtp_prefix_boundary_preserves_last_hash_when_drop_disabled():
    import os
    from types import SimpleNamespace
    from sglang_simulator.simulation.manager import ConfigManager
    from sglang_simulator.simulation.vllm.engine_core_pipeline import (
        C_VLLMSchedulerHook,
    )

    class Scheduler:
        def get_num_unfinished_requests(self):
            return 0

        def has_unfinished_requests(self):
            return False

        def _mamba_block_aligned_split(self, _request, tokens, *_args):
            return tokens

    C_VLLMSchedulerHook.hook(Scheduler)
    scheduler = Scheduler()
    scheduler.cache_config = SimpleNamespace(
        block_size=384, enable_prefix_caching=True
    )
    scheduler.scheduler_config = SimpleNamespace(max_num_batched_tokens=8192)
    scheduler.hash_block_size = 384
    ConfigManager._scheduler_config = SchedulerConfig(pcp_size=1)
    request = SimpleNamespace(num_prompt_tokens=1000, num_computed_tokens=0)
    old = {key: os.environ.get(key) for key in (
        "TPU_MTP_PREFILL_ONLY", "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP"
    )}
    try:
        os.environ["TPU_MTP_PREFILL_ONLY"] = "1"
        os.environ["VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP"] = "1"
        assert scheduler._mamba_block_aligned_split(request, 1000) == 768
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_cpu_startup_sanitizes_tpu_platform_but_keeps_connector_plugin():
    import os
    from sglang_simulator.simulation.vllm.startup import (
        _sanitize_tpu_environment_for_cpu,
    )

    keys = (
        "VLLM_TARGET_DEVICE", "VLLM_PLUGINS", "PJRT_DEVICE",
        "VLLM_PP_LAYER_PARTITION", "TPU_VISIBLE_CHIPS",
        "VLLM_NNODES", "RANK_SIZE",
    )
    old = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["VLLM_TARGET_DEVICE"] = "tpu"
        os.environ["VLLM_PLUGINS"] = "torchtpu,torchtpu_layers,kvs_connector"
        os.environ["PJRT_DEVICE"] = "TPU"
        os.environ["VLLM_PP_LAYER_PARTITION"] = "24,24,24,20"
        os.environ["TPU_VISIBLE_CHIPS"] = "0,1,2,3"
        os.environ["VLLM_NNODES"] = "4"
        os.environ["RANK_SIZE"] = "4"
        _sanitize_tpu_environment_for_cpu()
        assert "VLLM_TARGET_DEVICE" not in os.environ
        assert os.environ["VLLM_PLUGINS"] == "kvs_connector"
        assert "PJRT_DEVICE" not in os.environ
        assert "VLLM_PP_LAYER_PARTITION" not in os.environ
        assert os.environ["TPU_VISIBLE_CHIPS"] == ""
        assert os.environ["VLLM_NNODES"] == "1"
        assert os.environ["RANK_SIZE"] == "1"
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_tpu_connector_names_register_without_tpu_platform(monkeypatch):
    import sys
    from types import SimpleNamespace
    from sglang_simulator.simulation.vllm.engine_core_pipeline import (
        _register_tpu_connector_names,
    )

    registrations = []

    class Factory:
        @staticmethod
        def register_connector(name, module_path, class_name):
            registrations.append((name, module_path, class_name))

    module_name = "vllm.distributed.kv_transfer.kv_connector.factory"
    monkeypatch.setitem(
        sys.modules, module_name, SimpleNamespace(KVConnectorFactory=Factory)
    )
    _register_tpu_connector_names()
    assert {item[0] for item in registrations} == {
        "TPURaidenConnector", "TPUMultiConnector", "TPUV6dObjectConnector"
    }


if __name__ == "__main__":
    test_time_predictor()
