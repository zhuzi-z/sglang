"""TPU PP 逻辑流水线运行时单测：原生控制面加载、PCP 布局特征与 4 stage 计时闭环。"""

import shutil
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field

import pytest

from sglang_simulator.simulation.vllm.tpu_pp import native_control
from sglang_simulator.simulation.vllm.tpu_pp.stage_model import (
    FEATURE_NAMES, MODEL_KIND, TAIL_FEATURES, PcpLayout, StageSpan,
    empty_step_host_s, host_prep_s, stage_features, tail_features,
)

LAYOUT = PcpLayout(8, 128, (256, 512, 1024))


class TestStageFeatures:
    def test_pcp_interleave_layout(self):
        # 1000 token 不足一个 8x128 周期：前 7 个 rank 各 128，最后一个 104
        assert LAYOUT.local_counts(1000) == [128] * 7 + [104]
        assert LAYOUT.evaluate(1000) == {
            "global_padded_tokens": 2048, "local_padded_tokens": 256,
            "pcp_max_valid_tokens": 128, "pcp_min_valid_tokens": 104}

    def test_pcp_padding_overflow_rejected(self):
        with pytest.raises(ValueError):
            LAYOUT.evaluate(8 * 1024 + 1)

    def test_feature_names_and_tail(self):
        spans = [StageSpan(512, 0, 512), StageSpan(256, 1024, 4096, True)]
        assert list(stage_features(spans, LAYOUT)) == FEATURE_NAMES
        assert tail_features(spans) == dict(zip(TAIL_FEATURES, [1, 512, 1, 256]))

    def test_host_prep(self):
        cfg = {"intercept": 3.773523, "per_request": 4.896685,
               "per_new_request": -1.80922}
        spans = [StageSpan(512, 0, 512), StageSpan(256, 1024, 4096)]
        assert host_prep_s(cfg, spans) == pytest.approx(11.757673e-3)
        assert host_prep_s(None, spans) is None

    def test_empty_step_host(self):
        assert empty_step_host_s(None, 4) is None
        assert empty_step_host_s({"intercept": 1.0}, 4) is None
        assert empty_step_host_s({"empty_step_ms": [1, 2, -3, 4]}, 4) == pytest.approx(
            (1e-3, 2e-3, 0.0, 4e-3))
        with pytest.raises(ValueError):
            empty_step_host_s({"empty_step_ms": [1, 2]}, 4)
        with pytest.raises(ValueError):
            empty_step_host_s({"empty_step_ms": [1, 2, float("nan"), 4]}, 4)


class TestNativeControl:
    def test_load_exposes_native_control(self):
        native = native_control.load(True, 300.0)
        assert "execute_model" in native.STEP_METHODS
        assert native.LazyPushStepper and native.PPWave and native.FutureWrapper

    def test_tampered_vendor_rejected(self, tmp_path, monkeypatch):
        vendor = tmp_path / "vendor"
        shutil.copytree(native_control.VENDOR, vendor)
        with open(vendor / "pp_lazy_step.py", "a") as f:
            f.write("\n# tampered\n")
        monkeypatch.setattr(native_control, "VENDOR", vendor)
        monkeypatch.setattr(native_control, "MANIFEST", vendor / "SOURCE_MANIFEST.json")
        with pytest.raises(RuntimeError, match="fingerprint mismatch"):
            native_control.load(True, 300.0)


# ---- 4 stage 闭环：常数 stage 模型 + 原生 LazyPushStepper ----

STAGE_S = 0.03
COMM_S = 0.0008


@dataclass
class CachedReqs:
    req_ids: list = field(default_factory=list)
    num_computed_tokens: list = field(default_factory=list)


@dataclass
class NewReq:
    req_id: str
    prompt_token_ids: range
    sampling_params: object
    num_computed_tokens: int


@dataclass
class SchedOut:
    scheduled_new_reqs: list = field(default_factory=list)
    scheduled_cached_reqs: CachedReqs = field(default_factory=CachedReqs)
    num_scheduled_tokens: dict = field(default_factory=dict)
    total_num_scheduled_tokens: int = 0
    finished_req_ids: set = field(default_factory=set)
    kv_connector_metadata: object = None
    tag: int = 0


class _Sampling:
    logprobs = None


class _Inner:
    kv_output_aggregator = None

    def __init__(self):
        self.last = None

    def execute_model(self, so, non_block=False):
        self.last = so
        return None if so.total_num_scheduled_tokens else {"tag": so.tag}

    def sample_tokens(self, grammar, non_block=False):
        return {"tag": self.last.tag}

    def collective_rpc(self, method, *a, **k):
        return None

    def shutdown(self):
        pass


class _Scheduler:
    def __init__(self, steps):
        self.pending, self.completed = deque(steps), []

    def has_requests(self):
        return bool(self.pending)

    def schedule(self, throttle=False):
        return self.pending.popleft() if self.pending else SchedOut()

    def get_grammar_bitmask(self, so):
        return None

    def update_from_output(self, so, output):
        assert output["tag"] == so.tag
        self.completed.append(so.tag)
        return {0: output}


class _Core:
    async_scheduling = False

    def __init__(self, ex, sch):
        self.model_executor, self.scheduler = ex, sch

    def capture_iteration_details(self, so):
        return nullcontext(None)

    def log_error_detail(self, so):
        return nullcontext()

    def _process_aborts_queue(self):
        pass

    def _should_throttle_prefills(self):
        return False

    def _attach_iteration_details(self, o, d):
        pass


def _stage_models(tmp_path, stages):
    joblib = pytest.importorskip("joblib")
    dummy = pytest.importorskip("sklearn.dummy")
    paths = []
    for k in range(stages):
        names = FEATURE_NAMES + (TAIL_FEATURES if k == stages - 1 else [])
        model = dummy.DummyRegressor(strategy="constant", constant=STAGE_S)
        model.fit([[0.0] * len(names)], [STAGE_S])
        path = tmp_path / f"stage-{k}.pkl"
        joblib.dump(dict(model_kind=MODEL_KIND, production_usable=False, stage=k,
                         features=names, time_unit="seconds", models=[model]), path)
        paths.append(str(path))
    return paths


class _Predictor:
    async_handoff = True
    host_prep = None
    communication_s = COMM_S

    def __init__(self, paths):
        self.stage_model_paths = paths

    def features_for(self, spans):
        return stage_features(spans, LAYOUT), tail_features(spans)


def _steps(n):
    out = []
    for i in range(n):
        so = SchedOut(tag=i + 1)
        rid = f"r{i}"
        so.scheduled_new_reqs.append(NewReq(rid, range(512), _Sampling(), 0))
        so.num_scheduled_tokens[rid] = 512
        so.total_num_scheduled_tokens = 512
        out.append(so)
    # 携带状态的空 step：原生路径照常下发但不触发前向
    empty = SchedOut(tag=n + 1, kv_connector_metadata=object())
    return out + [empty]


def _run_pipeline(predictor, steps):
    from sglang_simulator.simulation.vllm.tpu_pp.stage_runtime import VirtualPPExecutor

    done = []
    ex = VirtualPPExecutor(_Inner(), predictor, 4,
                           on_step_done=lambda so, t: done.append(so.tag))
    sch = _Scheduler(steps)
    stepper = ex.native.LazyPushStepper(_Core(ex, sch))
    deadline = time.monotonic() + 60
    try:
        while sch.pending or stepper.in_flight or stepper.wave.open or ex.futures_queue:
            assert time.monotonic() < deadline, "pipeline did not drain"
            assert all(p.exitcode is None for p in ex._workers), "stage worker exited"
            stepper.step()
    finally:
        ex.shutdown()
        ex._collector.join(5)
    assert sch.completed == [s.tag for s in steps]
    assert not [e for e in ex.stage_events if e["kind"] == "worker_error"]
    return ex, done


def test_virtual_pp_pipeline_timing(tmp_path):
    ex, done = _run_pipeline(_Predictor(_stage_models(tmp_path, 4)), _steps(6))
    assert done == list(range(1, 7))
    computes = [e for e in ex.stage_events if e["kind"] == "compute"]
    assert len(computes) == 6 * 4
    by_step = {}
    for e in computes:
        by_step.setdefault(e["step_id"], {})[e["stage"]] = e
    for step_id, stages in by_step.items():
        assert sorted(stages) == [0, 1, 2, 3]
        for k in range(1, 4):
            # 下游 stage 不早于上游前向结束加一次 PP 通信
            assert stages[k]["start_at"] >= stages[k - 1]["end_at"] + COMM_S - 1e-6
    first = ex.step_records[1]
    latency = first["sample_ready_at"] - first["dispatched_at"]
    assert 4 * STAGE_S + 3 * COMM_S <= latency < 4 * STAGE_S + 0.1


@pytest.mark.parametrize("empty_ms", [None, 20.0])
def test_empty_step_occupies_every_stage_worker(tmp_path, empty_ms):
    predictor = _Predictor(_stage_models(tmp_path, 4))
    if empty_ms is not None:
        predictor.host_prep = {"intercept": 0.0, "per_request": 0.0, "per_new_request": 0.0,
                               "empty_step_ms": [empty_ms] * 4}
    ex, _ = _run_pipeline(predictor, _steps(2))
    empty = {e["stage"]: e for e in ex.stage_events if e["kind"] == "empty"}
    # 空 step 原生下发到全部 worker，即使只有末 stage 回复
    assert sorted(empty) == [0, 1, 2, 3]
    for e in empty.values():
        held = e["end_at"] - e["start_at"]
        if empty_ms is None:
            assert held < 5e-3
        else:
            assert empty_ms / 1e3 - 1e-3 <= held < empty_ms / 1e3 + 0.05
