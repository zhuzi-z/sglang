"""Unit tests for V6D simulation hooks.

These tests verify the hook components in isolation without requiring
a V6D daemon or vLLM engine. They test:
1. DummyStream/DummyEvent behavior
2. head_dim=1 injection logic
3. KV cache spec construction with real num_kv_heads
4. Hook class installation mechanics
"""

import os
import sys
import ctypes
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import torch
import pytest

from sglang_simulator.simulation.manager import StateManager
from sglang_simulator.simulation.vllm.v6d.v6d_backend import SimulatedKVController


def make_controller():
    from collections import defaultdict
    from vllm.v1.core.block_pool import BlockPool

    cache = SimpleNamespace(_block_pool=BlockPool(8, True, 4224),
                            _swap_protected_blocks=defaultdict(list), mamba_group_ids={0})
    cache._release_protected_blocks = lambda rid: cache._block_pool.free_blocks(
        cache._swap_protected_blocks.pop(rid, []))
    req = SimpleNamespace(request_id="r", is_finished=lambda: True)
    hybrid = SimpleNamespace(_backend=SimpleNamespace(_scheduler=cache),
                             _saving={"r": SimpleNamespace(_req=req)},
                             _saved=[], _loaded=[], pending_requests=[],
                             _step_saved=MagicMock(), loop=None, _tp_size=lambda: 2)
    return SimulatedKVController(hybrid)


@pytest.fixture
def controller(monkeypatch):
    from sglang_simulator.simulation.vllm.v6d.bandwidth import BandwidthModel

    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE")
    monkeypatch.setattr(StateManager, "_global_clock", 0.0)
    monkeypatch.setattr(BandwidthModel, "get", lambda: SimpleNamespace(
        enabled=True, store_completion_latency=lambda n: 0.5 if n else 0.0,
        latency_for=lambda n, load: 0.5 if n else 0.0, seg1_latency=lambda n: 0.0))
    return make_controller()


def store_meta(controller, last=False, empty=False):
    groups = {}
    block = None
    if not empty:
        block = controller.cache._block_pool.get_new_blocks(1)[0]
        block.ref_cnt += 1
        controller.cache._swap_protected_blocks["r"].append(block)
        groups = {0: (["hash"], [block.block_id])}
    meta = SimpleNamespace(reqs=SimpleNamespace(v6d=SimpleNamespace(
        inner=SimpleNamespace(reqs_to_store={"r": (groups, last)}))))
    return meta, block


# Ensure hooks are importable
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


# ============================================================
# Test 1: DummyStream / DummyEvent
# ============================================================

class TestDummyPrimitives:
    """Test DummyStream and DummyEvent mock CUDA primitives."""

    def setup_method(self):
        from sglang_simulator.simulation.vllm.cpu_stubs import (
            DummyStream, DummyEvent,
        )
        self.DummyStream = DummyStream
        self.DummyEvent = DummyEvent

    def test_dummy_event_query_always_true(self):
        event = self.DummyEvent()
        assert event.query() is True

    def test_dummy_event_record_noop(self):
        event = self.DummyEvent()
        # Should not raise
        event.record()
        event.record(stream=self.DummyStream())

    def test_dummy_event_synchronize_noop(self):
        event = self.DummyEvent()
        event.synchronize()

    def test_dummy_event_elapsed_time_zero(self):
        e1 = self.DummyEvent()
        e2 = self.DummyEvent()
        assert e1.elapsed_time(e2) == 0.0

    def test_dummy_stream_context_manager(self):
        stream = self.DummyStream()
        with stream as s:
            assert s is stream

    def test_dummy_stream_wait_stream_noop(self):
        s1 = self.DummyStream()
        s2 = self.DummyStream()
        # Should not raise
        s1.wait_stream(s2)

    def test_dummy_stream_synchronize_noop(self):
        stream = self.DummyStream()
        stream.synchronize()

    def test_dummy_stream_record_event_returns_event(self):
        stream = self.DummyStream()
        event = stream.record_event()
        assert event.query() is True

    def test_dummy_stream_query_always_true(self):
        stream = self.DummyStream()
        assert stream.query() is True


# ============================================================
# Test 3: _build_kv_cache_spec
# ============================================================

class TestBuildKVCacheSpec:
    """Test KV cache spec construction with the real head_dim."""

    def _make_vllm_config(
        self, num_layers=28, num_kv_heads=4, block_size=16, tp_size=1,
        head_dim=128,
    ):
        hf_config = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_key_value_heads=num_kv_heads,
            num_attention_heads=32,
            head_dim=head_dim,  # real model head_dim, no injection
        )
        model_config = SimpleNamespace(
            hf_text_config=hf_config,
            dtype=torch.float16,
            get_total_num_kv_heads=lambda: num_kv_heads,
            get_total_num_hidden_layers=lambda: num_layers,
        )
        cache_config = SimpleNamespace(
            block_size=block_size,
            mamba_block_size=None,
            cache_dtype="auto",
        )
        parallel_config = SimpleNamespace(tensor_parallel_size=tp_size)
        return SimpleNamespace(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
        )

    def test_pure_mha_model_spec_count(self):
        """Pure MHA: should have num_layers entries."""
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec

        vllm_config = self._make_vllm_config(num_layers=28)
        spec = _build_kv_cache_spec(vllm_config)
        assert len(spec) == 28

    def test_spec_uses_real_num_kv_heads(self):
        """Spec should use real num_kv_heads and real head_size."""
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec

        vllm_config = self._make_vllm_config(num_kv_heads=8, tp_size=1)
        spec = _build_kv_cache_spec(vllm_config)

        first_spec = list(spec.values())[0]
        assert first_spec.num_kv_heads == 8
        assert first_spec.head_size == 128

    def test_spec_with_tp_sharding(self):
        """With TP=4 and num_kv_heads=8, per-TP should be 2."""
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec

        vllm_config = self._make_vllm_config(num_kv_heads=8, tp_size=4)
        # Mock ConfigManager to return tp_size=4
        mock_scheduler_cfg = SimpleNamespace(tp_size=4)
        with patch(
            "sglang_simulator.simulation.manager.ConfigManager.get_scheduler_config",
            return_value=mock_scheduler_cfg,
        ):
            spec = _build_kv_cache_spec(vllm_config)

        first_spec = list(spec.values())[0]
        assert first_spec.num_kv_heads == 2  # 8 / 4

    def test_spec_page_size_is_real(self):
        """With real head_size, page_size should match the real layout."""
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec

        vllm_config = self._make_vllm_config(
            num_kv_heads=8, block_size=16, tp_size=1
        )
        spec = _build_kv_cache_spec(vllm_config)

        first_spec = list(spec.values())[0]
        # page_size = 2 * block_size * num_kv_heads * head_size * dtype_size
        # = 2 * 16 * 8 * 128 * 2 (fp16) = 65536 bytes
        assert first_spec.page_size_bytes == 65536

    def test_spec_layer_name_format(self):
        """Layer names should follow model.layers.{i} format."""
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec

        vllm_config = self._make_vllm_config(num_layers=3)
        spec = _build_kv_cache_spec(vllm_config)

        expected_names = [
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
        ]
        assert list(spec.keys()) == expected_names

    def test_hybrid_model_spec(self):
        """Hybrid model with layer_types should produce mixed specs."""
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec

        hf_config = SimpleNamespace(
            num_hidden_layers=4,
            num_key_value_heads=4,
            num_attention_heads=32,
            head_dim=128,
            layer_types=[
                "full_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
            ],
        )
        model_config = SimpleNamespace(
            hf_text_config=hf_config,
            dtype=torch.float16,
            get_total_num_kv_heads=lambda: 4,
            get_total_num_hidden_layers=lambda: 4,
        )
        cache_config = SimpleNamespace(
            block_size=16,
            mamba_block_size=None,
            mamba_cache_mode="none",
            cache_dtype="auto",
        )
        parallel_config = SimpleNamespace(tensor_parallel_size=1)
        vllm_config = SimpleNamespace(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
        )

        spec = _build_kv_cache_spec(vllm_config)
        assert len(spec) == 4
        # Check layer name patterns
        assert "model.layers.0" in spec
        assert "model.layers.1" in spec


class TestTPURaidenLocalCompletion:
    """覆盖四组断言、原生 MultiConnector 聚合和 Scheduler 延迟释放。"""

    @pytest.fixture
    def raiden_case(self, monkeypatch):
        from sglang_simulator.simulation.vllm.v6d.tpu_connector import (
            C_TPURaidenConnectorHook,
        )

        class Metadata:
            def __init__(self):
                self.reqs_to_send = {}
                self.reqs_to_load = {}

        module = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(TPUConnectorMetadata=Metadata))

        class NativeConnector:
            def __init__(self, config, role, cache):
                self.calls = []
                self.connector_scheduler = SimpleNamespace(reqs_to_send={}, reqs_to_load={})

            def on_new_request(self, req):
                self.calls.append("admit")
                req.prompt_token_ids.pop()  # 模拟 stage3 的末 token 截断

            def get_num_new_matched_tokens(self, req, computed):
                self.calls.append("match")
                return 8, True

            def update_state_after_alloc(self, req, blocks, external):
                self.calls.append("alloc")
                self.connector_scheduler.reqs_to_load[req.request_id] = blocks

            def build_connector_meta(self, output):
                self.calls.append("meta")
                return "native-meta"

            def request_finished(self, req, blocks):
                self.calls.append("finish")
                self.connector_scheduler.reqs_to_send[req.request_id] = blocks
                return True, {"remote_block_ids": blocks}

            def request_finished_all_groups(self, req, blocks):
                assert len(blocks) == 1, "Non-HMA TPUConnector expects a single kv-cache group"
                return self.request_finished(req, blocks[0])

        class Raiden(NativeConnector):
            pass

        C_TPURaidenConnectorHook.hook(Raiden)
        C_TPURaidenConnectorHook.hook(Raiden)  # 重复安装不能二次包裹
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1, pipeline_parallel_size=1,
                prefill_context_parallel_size=1, decode_context_parallel_size=1,
                data_parallel_size=1, nnodes=1,
            ),
        )
        cache = SimpleNamespace(kv_cache_groups=[object() for _ in range(4)])

        class Request:
            request_id = "r"
            client_index = 0
            kv_transfer_params = None
            num_computed_tokens = 62
            status = "FINISHED_LENGTH_CAPPED"

            def __init__(self):
                self.prompt_token_ids = list(range(62))

            def is_finished(self):
                return True

        return Raiden, NativeConnector, config, cache, Request()

    @pytest.mark.parametrize("transport", ["zmq", "raiden"])
    @pytest.mark.parametrize("tokens", [62, 3072, 6144])
    def test_multigroup_finish_never_publishes_real_handoff(self, raiden_case, monkeypatch, transport, tokens):
        Raiden, Native, config, cache, req = raiden_case
        monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", transport)
        req.num_computed_tokens = tokens
        groups = ([11], [], [22], [33, 34])
        native = Native(config, None, cache)
        with pytest.raises(AssertionError, match="Non-HMA"):
            native.request_finished_all_groups(req, groups)
        connector = Raiden(config, None, cache)
        prompt = req.prompt_token_ids.copy()
        connector.on_new_request(req)
        assert req.prompt_token_ids == prompt
        assert connector.get_num_new_matched_tokens(req, 0) == (0, False)
        connector.update_state_after_alloc(req, groups, 0)
        for status in ("FINISHED_LENGTH_CAPPED", "FINISHED_STOPPED", "FINISHED_ABORTED"):
            req.status = status
            for _ in range(2):
                assert connector.request_finished_all_groups(req, groups) == (False, None)
        assert connector.request_finished(req, groups[0]) == (False, None)
        for _ in range(2):
            metadata = connector.build_connector_meta(None)
            assert metadata.reqs_to_send == metadata.reqs_to_load == {}
        assert connector.connector_scheduler.reqs_to_send == {}
        assert connector.connector_scheduler.reqs_to_load == {}
        assert connector.calls == []
        assert groups == ([11], [], [22], [33, 34])

    @pytest.mark.parametrize("role,group_count", [("kv_consumer", 4), ("kv_both", 4), ("kv_producer", 1)])
    def test_other_routes_keep_native_contract(self, raiden_case, role, group_count):
        Raiden, _, config, cache, req = raiden_case
        config.kv_transfer_config.kv_role = role
        cache.kv_cache_groups = cache.kv_cache_groups[:group_count]
        connector = Raiden(config, None, cache)
        connector.on_new_request(req)
        assert len(req.prompt_token_ids) == 61
        assert connector.get_num_new_matched_tokens(req, 0) == (8, True)
        connector.update_state_after_alloc(req, ([1],), 8)
        assert connector.build_connector_meta(None) == "native-meta"
        assert connector.request_finished_all_groups(req, ([1],)) == (True, {"remote_block_ids": [1]})
        assert connector.calls == ["admit", "match", "alloc", "meta", "finish"]

    @pytest.mark.parametrize("params", [
        {"do_remote_decode": True}, {"do_remote_prefill": True},
        {"src_controller_address": "test-controller"}, {"uuid": 1}, "invalid",
    ])
    def test_explicit_remote_requests_fail_closed(self, raiden_case, params):
        Raiden, _, config, cache, req = raiden_case
        connector = Raiden(config, None, cache)
        req.kv_transfer_params = params
        with pytest.raises(ValueError, match="Local Raiden simulation"):
            connector.on_new_request(req)
        assert connector.calls == []

    def test_bad_group_count_or_remote_allocation_is_not_hidden(self, raiden_case):
        Raiden, _, config, cache, req = raiden_case
        connector = Raiden(config, None, cache)
        with pytest.raises(ValueError, match="KV group count mismatch"):
            connector.request_finished_all_groups(req, ([1],))
        with pytest.raises(ValueError, match="cannot load remote"):
            connector.update_state_after_alloc(req, ([1],) * 4, 1)

    @pytest.mark.parametrize("dimension", ["tensor_parallel_size", "pipeline_parallel_size", "prefill_context_parallel_size", "nnodes"])
    def test_physical_distributed_execution_is_not_silently_stubbed(self, raiden_case, dimension):
        Raiden, _, config, cache, _ = raiden_case
        setattr(config.parallel_config, dimension, 2)
        with pytest.raises(ValueError, match="single-worker topology"):
            Raiden(config, None, cache)

    @pytest.mark.parametrize("v6d_delays", [False, True])
    def test_native_aggregation_preserves_v6d_and_scheduler_free(self, raiden_case, v6d_delays):
        from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
        from vllm.v1.core.sched.scheduler import Scheduler

        Raiden, _, config, cache, req = raiden_case
        connector = Raiden(config, None, cache)
        groups = ([11], [], [22], [33, 34])
        sibling = SimpleNamespace(
            request_finished_all_groups=MagicMock(return_value=(v6d_delays, {"v6d": "kept"})),
        )
        multi = object.__new__(MultiConnector)
        multi._connectors = [connector, sibling]
        multi._all_support_hma = True
        multi._requests_to_connector = {req.request_id: 1}
        multi._extra_async_saves = {}
        scheduler = object.__new__(Scheduler)
        scheduler._inflight_prefills = set()
        scheduler._connector_finished = lambda request: multi.request_finished_all_groups(request, groups)
        scheduler.ec_connector = None
        scheduler.encoder_cache_manager = SimpleNamespace(free=MagicMock())
        scheduler.finished_req_ids = set()
        scheduler.finished_req_ids_dict = None
        scheduler._free_blocks = MagicMock()

        assert scheduler._free_request(req) == ({"v6d": "kept"}, None)
        sibling.request_finished_all_groups.assert_called_once_with(req, groups)
        assert sibling.request_finished_all_groups.call_args.args[1] is groups
        assert multi._requests_to_connector == multi._extra_async_saves == {}
        assert scheduler.finished_req_ids == {req.request_id}
        if v6d_delays:
            scheduler._free_blocks.assert_not_called()
        else:
            scheduler._free_blocks.assert_called_once_with(req)
        assert connector.connector_scheduler.reqs_to_send == {}

    def test_native_multi_still_selects_v6d_for_remote_cache_hits(self, raiden_case):
        from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

        Raiden, _, config, cache, req = raiden_case
        connector = Raiden(config, None, cache)
        v6d_meta = SimpleNamespace(reqs_to_load={req.request_id: {3: (["key"], [33])}})
        sibling = SimpleNamespace(
            on_new_request=MagicMock(),
            get_num_new_matched_tokens=MagicMock(return_value=(3072, True)),
            update_state_after_alloc=MagicMock(),
            build_connector_meta=MagicMock(return_value=v6d_meta),
        )
        multi = object.__new__(MultiConnector)
        multi._connectors = [connector, sibling]
        multi._requests_to_connector = {}
        multi._extra_async_saves = {}
        multi.on_new_request(req)
        assert multi.get_num_new_matched_tokens(req, 0) == (3072, True)
        assert multi._requests_to_connector == {req.request_id: 1}
        blocks = object()
        multi.update_state_after_alloc(req, blocks, 3072)
        sibling.update_state_after_alloc.assert_called_once_with(req, blocks, 3072)
        meta = multi.build_connector_meta(None)
        assert meta.metadata[0].reqs_to_send == meta.metadata[0].reqs_to_load == {}
        assert meta.metadata[1] is v6d_meta
        assert connector.calls == []

    def test_scheduler_hook_is_registered(self):
        from sglang_simulator.simulation.vllm.v6d.hooks import register_v6d_hooks
        from sglang_simulator.simulation.vllm.v6d.tpu_connector import C_TPURaidenConnectorHook

        hooks = []
        register_v6d_hooks(hooks)
        assert hooks.count(C_TPURaidenConnectorHook) == 1
        assert C_TPURaidenConnectorHook.HOOK_CLASS_NAME == "TPURaidenConnector"
        assert C_TPURaidenConnectorHook.HOOK_MODULE_NAME == "vllm_torchtpu.distributed.kv_transfer.tpu_connector"


class TestTPUPooledKVGeometry:
    """以混合 spec 实际消费链覆盖页不足，不再传空 specs 绕过校验。"""

    @pytest.fixture
    def tpu_case(self, monkeypatch):
        from vllm.model_executor.models import ModelRegistry
        from sglang_simulator.simulation.manager import ConfigManager

        # 合成 Qwen GDN fixture：PCP8 后的单页大小与参考日志一致；
        # 这些数值只属于回归输入，不作为实现中的模型常量。
        hf = SimpleNamespace(
            num_hidden_layers=4, num_key_value_heads=4, num_attention_heads=32,
            head_dim=256, hidden_size=8192, linear_num_key_heads=16,
            linear_num_value_heads=128, linear_key_head_dim=128,
            linear_value_head_dim=128, mtp_num_hidden_layers=0,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
        )
        model = SimpleNamespace(
            architecture="Qwen3NextForCausalLM", hf_text_config=hf,
            dtype=torch.bfloat16, get_total_num_kv_heads=lambda: hf.num_key_value_heads,
            get_head_size=lambda: hf.head_dim,
        )
        class Cache(SimpleNamespace):
            def get_resolved_kv_cache_layout(self):
                from vllm.v1.kv_cache_interface import KVCacheLayout
                return KVCacheLayout[self.kv_cache_layout]

        cache = Cache(
            block_size=384, mamba_block_size=384, mamba_cache_mode="align",
            mamba_ssm_cache_dtype="bfloat16", cache_dtype="fp8", attn_pack_size=1,
            kv_cache_memory_bytes=22548578304, num_gpu_blocks_override=None,
            kv_cache_layout=None, user_specified_block_size=True,
            prefix_cache_retention_interval=0, prefix_match_unit=None,
            enable_prefix_caching=True,
        )
        model.max_model_len = model.original_max_model_len = 1048576
        config = SimpleNamespace(
            model_config=model, cache_config=cache,
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1, pipeline_parallel_size=1,
                prefill_context_parallel_size=1, decode_context_parallel_size=1,
            ),
            speculative_config=SimpleNamespace(
                method="mtp", num_speculative_tokens=1, use_eagle=lambda: False,
                use_multi_module_mtp=lambda: False,
            ),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            kv_transfer_config=None,
        )
        raw = {
            "platform": {"accelerator": {"name": "tpu7x-32"}},
            "scheduler": {"tp_size": 1, "pcp_size": 8, "pp_size": 4,
                          "physical_block_size": 384},
        }

        class Model:
            @staticmethod
            def get_mamba_state_shape_from_config(cfg):
                taps = 3 + cfg.speculative_config.num_speculative_tokens
                width = (2 * hf.linear_num_key_heads * hf.linear_key_head_dim
                         + hf.linear_num_value_heads * hf.linear_value_head_dim)
                return ((taps, width), (hf.linear_num_value_heads, 128, 128))

            @staticmethod
            def get_mamba_state_dtype_from_config(cfg):
                return (torch.float32, getattr(torch, cfg.cache_config.mamba_ssm_cache_dtype))

        monkeypatch.setattr(ModelRegistry, "resolve_model_cls", lambda *a, **kw: (Model, None))
        monkeypatch.setattr(ConfigManager, "_get_raw_config", lambda: raw)
        monkeypatch.setattr(ConfigManager, "get_scheduler_config", lambda: None)
        monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
        monkeypatch.delenv("SGLANG_SIMULATOR_PP_LAYER_PARTITION", raising=False)
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
        return config, raw, Model

    def test_pcp_local_page_fits_native_mamba_invariant(self, tpu_case):
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec
        from sglang_simulator.simulation.vllm.tpu_kv_geometry import derive_tpu_pooled_page

        config, raw, model = tpu_case
        shapes = model.get_mamba_state_shape_from_config(config)
        dtypes = model.get_mamba_state_dtype_from_config(config)
        page = derive_tpu_pooled_page(config, raw, shapes, dtypes)
        assert page.full_state_bytes > 786432
        assert page.shapes == ((4, 1, 2560), (16, 128, 128))
        assert page.dtypes == (torch.bfloat16, torch.bfloat16)
        assert page.raw_state_bytes == 544768
        assert page.required_state_bytes == 557056
        assert page.slot_bytes == 786432
        specs = _build_kv_cache_spec(config)
        assert len(specs) == 4
        mamba, attention = specs["model.layers.0"], specs["model.layers.3"]
        assert mamba.shapes == page.shapes
        assert mamba.dtypes == page.dtypes
        assert mamba.page_size_bytes == attention.page_size_bytes == page.slot_bytes
        assert config.cache_config.mamba_page_size_padded == page.slot_bytes
        assert attention.num_head_slots == 1
        assert attention.state_content_bytes == page.token_bytes
        # 不改变 384 粒度、真实执行拓扑或为绕过断言扩大总内存预算。
        assert config.cache_config.block_size == config.cache_config.mamba_block_size == 384
        assert config.cache_config.kv_cache_memory_bytes == 22548578304
        assert config.cache_config.num_gpu_blocks_override is None
        assert config.parallel_config.prefill_context_parallel_size == 1
        assert config.parallel_config.tensor_parallel_size == 1

    def test_mixed_specs_reach_native_layout_and_budget_validation(self, tpu_case, monkeypatch):
        from vllm.v1.attention.backends.utils import (
            get_supported_kv_cache_layouts, resolve_kv_cache_layout,
        )
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_groups, get_kv_cache_config_from_groups,
        )
        from sglang_simulator.simulation.vllm.attention_backend import SimulatedAttentionBackend
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec

        config, _, _ = tpu_case
        specs = _build_kv_cache_spec(config)
        names = [layout.name for layout in get_supported_kv_cache_layouts([SimulatedAttentionBackend])]
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
        layout = resolve_kv_cache_layout(config, [names], specs.values())
        assert layout.name == "LBHNC"
        assert config.cache_config.kv_cache_layout == layout.name
        # 不用空 specs：新版本 HMA 分组也必须能读取每种页的实际大小。
        config.cache_config.get_resolved_kv_cache_layout = lambda: layout
        config.scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=False)
        groups = get_kv_cache_groups(config, specs)
        assert groups
        assert all(group.kv_cache_spec.page_size_bytes == 786432 for group in groups)
        budget = config.cache_config.kv_cache_memory_bytes
        allocated = get_kv_cache_config_from_groups(config, groups, budget)
        smaller = get_kv_cache_config_from_groups(config, groups, budget // 2)
        assert 0 < smaller.num_blocks <= allocated.num_blocks
        # 各组别名共享同一 backing allocation，不能按 tensor 个数重复累加。
        assert all(tensor.size <= budget for tensor in allocated.kv_cache_tensors)
        assert config.cache_config.kv_cache_memory_bytes == budget
        assert config.cache_config.num_gpu_blocks_override is None
        config.cache_config.kv_cache_layout = None
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "INVALID_HISIM_LAYOUT")
        with pytest.raises(ValueError):
            resolve_kv_cache_layout(config, [names], specs.values())

    def test_qk_replication_is_not_total_bytes_divided_by_pcp(self, tpu_case):
        from sglang_simulator.simulation.vllm.tpu_kv_geometry import derive_tpu_pooled_page

        config, raw, model = tpu_case
        hf = config.model_config.hf_text_config
        hf.linear_num_key_heads, hf.linear_num_value_heads = 4, 32
        shapes = model.get_mamba_state_shape_from_config(config)
        page = derive_tpu_pooled_page(config, raw, shapes, (torch.bfloat16, torch.bfloat16))
        # PCP8 下 Q/K 保留一个完整 head，V 切为4个。
        assert page.shapes == ((4, 1, 768), (4, 128, 128))
        assert page.raw_state_bytes == 137216
        assert page.raw_state_bytes != page.full_state_bytes // 8

    def test_logical_tp_applied_before_pcp_without_config_mutation(self, tpu_case):
        from sglang_simulator.simulation.vllm.tpu_kv_geometry import derive_tpu_pooled_page

        config, raw, model = tpu_case
        raw["scheduler"].update(tp_size=2, pcp_size=4, physical_block_size=768)
        config.cache_config.block_size = 768
        page = derive_tpu_pooled_page(
            config, raw, model.get_mamba_state_shape_from_config(config),
            model.get_mamba_state_dtype_from_config(config),
        )
        assert page.shapes == ((4, 1, 2560), (16, 128, 128))
        assert page.token_bytes == 1024
        assert page.slot_bytes == 786432
        assert config.parallel_config.tensor_parallel_size == 1
        assert config.parallel_config.prefill_context_parallel_size == 1

    @pytest.mark.parametrize("change", ["pcp1", "ssm_fp32", "small_block"])
    def test_insufficient_target_slot_fails_with_byte_diagnostics(self, tpu_case, change):
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec

        config, raw, _ = tpu_case
        if change == "pcp1":
            raw["scheduler"]["pcp_size"] = 1
        elif change == "ssm_fp32":
            config.cache_config.mamba_ssm_cache_dtype = "float32"
        else:
            raw["scheduler"]["physical_block_size"] = 128
        with pytest.raises(ValueError, match="raw_state_bytes=.*required_state_bytes=.*slot_bytes="):
            _build_kv_cache_spec(config)

    @pytest.mark.parametrize("taps", [3, 4, 9])
    def test_conv_layout_and_speculative_width_are_preserved(self, tpu_case, taps):
        from sglang_simulator.simulation.vllm.tpu_kv_geometry import derive_tpu_pooled_page

        config, raw, model = tpu_case
        config.speculative_config.num_speculative_tokens = taps - 3
        shapes = model.get_mamba_state_shape_from_config(config)
        dtypes = model.get_mamba_state_dtype_from_config(config)
        sd = derive_tpu_pooled_page(config, raw, shapes, dtypes)
        ds = derive_tpu_pooled_page(config, raw, (shapes[0][::-1], shapes[1]), dtypes)
        assert sd == ds
        assert sd.shapes[0] == (taps, 1, 2560)

    @pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3", "fp8_e5m2", "auto"])
    def test_kv_dtype_does_not_quantize_mamba_ssm(self, tpu_case, cache_dtype):
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec

        config, _, _ = tpu_case
        config.cache_config.cache_dtype = cache_dtype
        specs = _build_kv_cache_spec(config)
        assert specs["model.layers.0"].dtypes == (torch.bfloat16, torch.bfloat16)
        expected = 1572864 if cache_dtype == "auto" else 786432
        assert all(spec.page_size_bytes == expected for spec in specs.values())

    @pytest.mark.parametrize("field,value", [("pcp_size", 3), ("tp_size", 0), ("pcp_size", True)])
    def test_invalid_topology_is_rejected(self, tpu_case, field, value):
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec

        config, raw, _ = tpu_case
        raw["scheduler"][field] = value
        with pytest.raises(ValueError):
            _build_kv_cache_spec(config)

    def test_missing_model_shapes_does_not_fall_back_to_placeholder(self, tpu_case, monkeypatch):
        from vllm.model_executor.models import ModelRegistry
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec

        def unavailable(*args, **kwargs):
            raise RuntimeError("model API unavailable")

        monkeypatch.setattr(ModelRegistry, "resolve_model_cls", unavailable)
        with pytest.raises(ValueError, match="without real Mamba shapes/dtypes"):
            _build_kv_cache_spec(tpu_case[0])

    def test_non_unified_target_is_not_silently_treated_as_unified(self, tpu_case, monkeypatch):
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec

        monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "0")
        with pytest.raises(ValueError, match="non-unified"):
            _build_kv_cache_spec(tpu_case[0])

    def test_legacy_gpu_pack_rule_is_preserved(self, tpu_case):
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec

        config, raw, _ = tpu_case
        raw["platform"]["accelerator"]["name"] = "a100_sxm"
        config.cache_config.attn_pack_size = 8
        specs = _build_kv_cache_spec(config)
        attention, mamba = specs["model.layers.3"], specs["model.layers.0"]
        assert attention.page_size_bytes == 786432
        assert mamba.page_size_padded == attention.page_size_bytes * 8
        assert mamba.dtypes == (torch.float32, torch.bfloat16)
        assert mamba.shapes == ((4, 20480), (128, 128, 128))


class TestTPUVirtualKVCapacity(TestTPUPooledKVGeometry):
    """复用真实 spec fixture，执行原生容量规划而非仅核验自编公式。"""

    @pytest.fixture
    def pipeline_case(self, tpu_case):
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec

        config, raw, _ = tpu_case
        hf = config.model_config.hf_text_config
        hf.num_hidden_layers = 92
        hf.layer_types = (["linear_attention"] * 3 + ["full_attention"]) * 23
        raw["scheduler"]["pp_layer_partition"] = [24, 24, 24, 20]
        return config, raw, _build_kv_cache_spec(config)

    def test_native_virtual_stages_and_cpu_capacity_agree(self, pipeline_case):
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
        from sglang_simulator.simulation.vllm.tpu_kv_capacity import (
            apply_tpu_kv_capacity, plan_tpu_kv_capacity,
        )

        config, raw, specs = pipeline_case
        original_budget = config.cache_config.kv_cache_memory_bytes
        plan = plan_tpu_kv_capacity(config, raw, specs)
        assert plan.stage_layers == (24, 24, 24, 20)
        assert plan.stage_block_limits[:3] == (4778, 4778, 4778)
        assert plan.stage_block_limits[3] > 4778
        assert plan.num_blocks == 4778
        assert plan.logical_block_size == plan.hash_block_size == 3072
        assert plan.virtual_memory_bytes == 4778 * 23 * 786432
        assert config.cache_config.block_size == 384  # 规划未修改原对象
        apply_tpu_kv_capacity(config, plan)
        assert config.cache_config.block_size == config.cache_config.mamba_block_size == 3072
        assert config.cache_config.prefix_match_unit is None
        assert config.parallel_config.pipeline_parallel_size == 1
        assert config.parallel_config.prefill_context_parallel_size == 1
        assert config.cache_config.kv_cache_memory_bytes == original_budget
        assert config.cache_config.num_gpu_blocks_override is None
        assert config.model_config.max_model_len == 1048576
        actual = get_kv_cache_configs(config, [plan.specs], [plan.virtual_memory_bytes])[0]
        assert actual.num_blocks == plan.num_blocks
        assert all(s.page_size_bytes == 786432 for s in plan.specs.values())
        assert specs["model.layers.3"].block_size == 384
        assert plan.specs["model.layers.3"].tokens_per_state == 8
        assert plan.specs["model.layers.0"].shapes == specs["model.layers.0"].shapes

    def test_smallest_stage_budget_limits_whole_pipeline(self, pipeline_case):
        from sglang_simulator.simulation.vllm.tpu_kv_capacity import plan_tpu_kv_capacity

        config, raw, specs = pipeline_case
        budget = config.cache_config.kv_cache_memory_bytes
        raw["scheduler"]["pp_stage_kv_cache_memory_bytes"] = [budget] * 3 + [budget // 2]
        plan = plan_tpu_kv_capacity(config, raw, specs)
        assert plan.num_blocks == min(plan.stage_block_limits)
        assert plan.num_blocks == plan.stage_block_limits[3] < plan.stage_block_limits[0]

    @pytest.mark.parametrize("budgets", [[100] * 4, [22548578304] * 3 + [100]])
    def test_insufficient_virtual_stage_still_raises(self, pipeline_case, budgets):
        from sglang_simulator.simulation.vllm.tpu_kv_capacity import plan_tpu_kv_capacity

        config, raw, specs = pipeline_case
        raw["scheduler"]["pp_stage_kv_cache_memory_bytes"] = budgets
        with pytest.raises(ValueError):
            plan_tpu_kv_capacity(config, raw, specs)
        assert config.cache_config.block_size == 384
        assert config.model_config.max_model_len == 1048576

    @pytest.mark.parametrize("override", [1, 100000])
    def test_override_cannot_bypass_request_or_physical_capacity(self, pipeline_case, override):
        from sglang_simulator.simulation.vllm.tpu_kv_capacity import plan_tpu_kv_capacity

        config, raw, specs = pipeline_case
        config.cache_config.num_gpu_blocks_override = override
        with pytest.raises(ValueError):
            plan_tpu_kv_capacity(config, raw, specs)

    def test_valid_override_can_reduce_capacity(self, pipeline_case):
        from sglang_simulator.simulation.vllm.tpu_kv_capacity import plan_tpu_kv_capacity

        config, raw, specs = pipeline_case
        config.cache_config.num_gpu_blocks_override = 4000
        assert plan_tpu_kv_capacity(config, raw, specs).num_blocks == 4000

    def test_explicit_fine_hash_unit_is_preserved(self, pipeline_case):
        from sglang_simulator.simulation.vllm.tpu_kv_capacity import plan_tpu_kv_capacity

        config, raw, specs = pipeline_case
        config.cache_config.prefix_match_unit = 384
        assert plan_tpu_kv_capacity(config, raw, specs).hash_block_size == 384
        config.cache_config.prefix_match_unit = 100
        with pytest.raises(ValueError):
            plan_tpu_kv_capacity(config, raw, specs)

    @pytest.mark.parametrize("requested", [None, 384, 3072])
    def test_apply_does_not_overwrite_user_prefix_match_unit(self, requested):
        from sglang_simulator.simulation.vllm.tpu_kv_capacity import apply_tpu_kv_capacity

        config = SimpleNamespace(cache_config=SimpleNamespace(prefix_match_unit=requested))
        plan = SimpleNamespace(logical_block_size=3072, hash_block_size=3072, layout_name="LBNHC")
        apply_tpu_kv_capacity(config, plan)
        assert config.cache_config.prefix_match_unit == requested

    def test_apply_accepts_unset_read_only_v6d_prefix_override(self):
        from sglang_simulator.simulation.vllm.tpu_kv_capacity import apply_tpu_kv_capacity

        # 只检查 V6D 的 None 契约，不启动真实 connector/外部连接。
        class Cache:
            @property
            def prefix_match_unit(self):
                return None

        config = SimpleNamespace(cache_config=Cache())
        plan = SimpleNamespace(logical_block_size=3072, hash_block_size=3072, layout_name="LBNHC")
        apply_tpu_kv_capacity(config, plan)
        assert config.cache_config.prefix_match_unit is None
        assert config.cache_config.block_size == 3072

    def test_missing_partition_requires_explicit_target_input(self, pipeline_case, monkeypatch):
        from sglang_simulator.simulation.vllm.tpu_kv_capacity import plan_tpu_kv_capacity

        config, raw, specs = pipeline_case
        del raw["scheduler"]["pp_layer_partition"]
        with pytest.raises(ValueError, match="PP capacity planning requires"):
            plan_tpu_kv_capacity(config, raw, specs)
        monkeypatch.setenv("SGLANG_SIMULATOR_PP_LAYER_PARTITION", "24,24,24,20")
        assert plan_tpu_kv_capacity(config, raw, specs).stage_layers == (24, 24, 24, 20)

    @pytest.mark.parametrize("partition", [[24, 24, 24, 24], [24, 24], [0, 24, 24, 44]])
    def test_invalid_stage_partition_rejected(self, pipeline_case, partition):
        from sglang_simulator.simulation.vllm.tpu_kv_capacity import plan_tpu_kv_capacity

        config, raw, specs = pipeline_case
        raw["scheduler"]["pp_layer_partition"] = partition
        with pytest.raises(ValueError, match="PP stage layers|Invalid pp_layer_partition"):
            plan_tpu_kv_capacity(config, raw, specs)

    def test_mtp_ownership_is_last_stage(self, pipeline_case):
        from sglang_simulator.simulation.vllm.tpu_kv_capacity import _project_layers

        _, _, specs = pipeline_case
        specs = dict(specs, **{"model.mtp.layers.92": specs["model.layers.3"]})
        projected = _project_layers(specs, (24, 24, 24, 20))
        assert [len(s) for s in projected] == [24, 24, 24, 21]
        assert "model.mtp.layers.92" in projected[-1]
        assert all("model.mtp.layers.92" not in s for s in projected[:-1])

    def test_block_table_and_hash_units_use_single_pcp_factor(self, pipeline_case):
        from vllm.v1.core.kv_cache_utils import (
            BlockHashListWithBlockSize, get_kv_cache_configs, resolve_kv_cache_block_sizes,
        )
        from sglang_simulator.simulation.vllm.tpu_kv_capacity import (
            apply_tpu_kv_capacity, plan_tpu_kv_capacity,
        )

        config, raw, specs = pipeline_case
        plan = plan_tpu_kv_capacity(config, raw, specs)
        apply_tpu_kv_capacity(config, plan)
        # 保持 V6D 要求的 None，仍由原生 resolver 从逻辑块推导3072。
        assert config.cache_config.prefix_match_unit is None
        native = get_kv_cache_configs(config, [plan.specs], [plan.virtual_memory_bytes])[0]
        assert resolve_kv_cache_block_sizes(native, config) == (3072, 3072)
        assert config.cache_config.prefix_match_unit is None
        assert plan.specs["model.layers.3"].max_num_blocks_per_req(config, 6144) == 2
        hashes = list(range(16))  # 16 个384-token hash，对应两个3072-token块
        assert list(BlockHashListWithBlockSize(hashes, 384, 3072)) == [7, 15]

    @pytest.mark.parametrize("budget,expected", [(8192, 6144), (32768, 30720)])
    def test_scheduler_alignment_does_not_multiply_pcp_twice(self, tpu_case, budget, expected):
        from sglang_simulator.simulation.vllm.engine_core_pipeline import _pcp_budget_granule

        _, raw, _ = tpu_case
        granule = _pcp_budget_granule(3072, 8)
        assert granule == 3072
        assert budget // granule * granule == expected
        assert _pcp_budget_granule(384, 8) == 3072
        raw["platform"]["accelerator"]["name"] = "a100_sxm"
        assert _pcp_budget_granule(384, 8) == 3072

    def test_tensor_layer_names_support_native_and_legacy_configs(self):
        from sglang_simulator.simulation.vllm.worker import _kv_tensor_layer_names

        assert _kv_tensor_layer_names(SimpleNamespace(layers=["a"])) == ["a"]
        assert _kv_tensor_layer_names(SimpleNamespace(shared_by=["b"])) == ["b"]
        with pytest.raises(ValueError, match="layers/shared_by"):
            _kv_tensor_layer_names(SimpleNamespace(size=4096))


# ============================================================
# Test 7: V6dObjectConnectorWorker hook mechanics
# ============================================================

class TestV6dObjectConnectorWorkerHook:
    """Test V6dObjectConnectorWorker hook skips CUDA operations."""

    def test_hook_replaces_register_host_memory(self):
        from sglang_simulator.simulation.vllm.v6d.v6d_worker import (
            C_V6dObjectConnectorWorkerHook,
        )

        class MockConnectorWorker:
            def _register_v6d_host_memory(self):
                raise RuntimeError("Should not call CUDA host register")

            def _start_async_v6d_init(self):
                pass

            def register_kv_caches(self, kv_caches):
                pass

        C_V6dObjectConnectorWorkerHook.hook(MockConnectorWorker)

        worker = MockConnectorWorker()
        # Should NOT raise (replaced with no-op)
        worker._register_v6d_host_memory()


# ============================================================
# Test 8: V6dObjectBackend hook mechanics
# ============================================================

class TestV6dObjectBackendHook:
    """Test V6dObjectBackend hook replaces CUDA Event pools."""

    def test_hook_replaces_event_pool(self):
        from sglang_simulator.simulation.vllm.v6d.v6d_backend import (
            C_V6dObjectBackendHook,
        )
        from sglang_simulator.simulation.vllm.cpu_stubs import DummyEvent

        class MockV6dObjectBackend:
            def __init__(self):
                self._save_event_pool = ["real_event_1", "real_event_2"]
                self._load_event_pool = ["real_event_3"]

            async def async_load_kv(self, m):
                # Hook wraps this with an instant-completion override
                yield None

        C_V6dObjectBackendHook.hook(MockV6dObjectBackend)

        backend = MockV6dObjectBackend()
        # Event pools should be replaced with DummyEvent instances
        assert all(
            isinstance(e, DummyEvent) for e in backend._save_event_pool
        )
        assert all(
            isinstance(e, DummyEvent) for e in backend._load_event_pool
        )
        assert len(backend._save_event_pool) == 8
        assert len(backend._load_event_pool) == 8


# ============================================================
# Test 9: Platform hook additions
# ============================================================

class TestPlatformHookAdditions:
    """验证 Platform 的调用签名、可导入返回值和真实 layout 消费链路。"""

    @pytest.fixture
    def cpu_platform(self, monkeypatch):
        import vllm.platforms
        from vllm.platforms.interface import Platform
        from sglang_simulator.simulation.vllm.platform import C_VLLMPlatformHook

        # Hook 会替换全局变量，测试结束时恢复，避免依赖执行顺序。
        monkeypatch.setattr(
            vllm.platforms, "current_platform", vllm.platforms.current_platform
        )
        C_VLLMPlatformHook.hook(Platform)
        return vllm.platforms.current_platform

    def test_set_device_is_noop(self, cpu_platform):
        cpu_platform.set_device(torch.device("cpu"))
        cpu_platform.set_device(torch.device("cuda:0"))

    def test_get_device_total_memory(self, cpu_platform):
        assert cpu_platform.get_device_total_memory() == 80 * (1 << 30)

    @pytest.mark.parametrize("use_mla", [False, True])
    @pytest.mark.parametrize("style", [
        "legacy_positional", "legacy_keyword",
        "selector_positional", "selector_keyword",
    ])
    def test_attention_backend_signature_and_import(self, cpu_platform, style, use_mla):
        from importlib import import_module

        selector = SimpleNamespace(use_mla=use_mla)
        if style == "legacy_positional":
            path = cpu_platform.get_attn_backend_cls(
                None, 128, torch.bfloat16, "fp8", 384, True, use_mla
            )
        elif style == "legacy_keyword":
            path = cpu_platform.get_attn_backend_cls(
                selected_backend=None, head_size=128, dtype=torch.bfloat16,
                kv_cache_dtype="fp8", block_size=384, use_v1=True, use_mla=use_mla,
            )
        elif style == "selector_positional":
            path = cpu_platform.get_attn_backend_cls(None, selector, 16)
        else:
            path = cpu_platform.get_attn_backend_cls(
                None, attn_selector_config=selector, num_heads=16
            )

        assert isinstance(path, str) and path
        module_name, class_name = path.rsplit(".", 1)
        backend = getattr(import_module(module_name), class_name)
        assert not backend.__abstractmethods__
        assert backend.full_cls_name() == (module_name, class_name)
        assert backend.is_mla() is use_mla
        assert backend.is_ssm() is False
        assert backend.supported_kv_cache_layouts() is None
        expected_shape = (2, 8, 384, 4, 128) if not use_mla else (8, 384, 128)
        assert backend.get_kv_cache_shape(8, 384, 4, 128) == expected_shape
        # 元数据 backend 不能被当成真实计算实现静默使用。
        with pytest.raises(NotImplementedError, match="no attention kernel"):
            backend.get_impl_cls()
        with pytest.raises(NotImplementedError, match="no attention metadata builder"):
            backend.get_builder_cls()

    def test_attention_backend_path_is_lazy(self, cpu_platform, monkeypatch):
        import builtins

        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.startswith(("vllm", "flashinfer", "flash_attn")) or "attention_backend" in name:
                raise AssertionError(f"不应在 Platform 选择阶段导入 {name}")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        path = cpu_platform.get_attn_backend_cls(
            None, attn_selector_config=SimpleNamespace(use_mla=False), num_heads=16
        )
        assert path.endswith(".SimulatedAttentionBackend")

    @pytest.mark.parametrize("use_mla", [False, True])
    def test_native_v029_worker_layout_contract(self, cpu_platform, monkeypatch, use_mla):
        """直接走原生 WorkerBase → selector → backend → layout resolver。"""
        from vllm.v1.worker.worker_base import WorkerBase

        if not hasattr(WorkerBase, "get_supported_kv_cache_layouts"):
            pytest.skip("此项针对 vLLM 0.29 的 layout 协商接口")
        from vllm.v1.attention.selector import (
            AttentionSelectorConfig, _cached_get_attn_backend,
        )
        from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

        # 仅提供配置数据，不替换被测 WorkerBase、selector 或 resolver。
        config = SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={}),
            model_config=SimpleNamespace(
                get_head_size=lambda: 128, dtype=torch.bfloat16, use_mla=use_mla,
            ),
            cache_config=SimpleNamespace(
                user_specified_block_size=True, block_size=384,
                cache_dtype="fp8", kv_cache_layout=None,
            ),
            attention_config=SimpleNamespace(
                backend=None, backend_per_kind={}, use_non_causal=False,
            ),
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=1, decode_context_parallel_size=1,
            ),
            kv_transfer_config=None,
            speculative_config=None,
        )
        worker = object.__new__(WorkerBase)
        worker.vllm_config = config

        def no_cuda(*args, **kwargs):
            raise AssertionError("layout 协商不应访问 CUDA")

        monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
        monkeypatch.setattr(torch.cuda, "get_device_properties", no_cuda)
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
        _cached_get_attn_backend.cache_clear()
        try:
            selector = AttentionSelectorConfig(128, torch.bfloat16, "fp8", 384, use_mla=use_mla)
            backend = _cached_get_attn_backend(
                None, attn_selector_config=selector, num_heads=16
            )
            assert backend.is_mla() is use_mla
            layouts = worker.get_supported_kv_cache_layouts()
            assert layouts and "LBHNC" in layouts
            layout = resolve_kv_cache_layout(config, [layouts], [])
            assert layout.name == "LBHNC"
            worker.set_kv_cache_layout(layout.name)
            assert config.cache_config.kv_cache_layout == layout.name

            # 不得为了绕过启动错误屏蔽原生布局校验。
            config.cache_config.kv_cache_layout = None
            monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "INVALID_HISIM_LAYOUT")
            with pytest.raises(ValueError):
                resolve_kv_cache_layout(config, [layouts], [])
        finally:
            _cached_get_attn_backend.cache_clear()


# ============================================================
# Test: HybridConnector hook — mamba block protection release
# (regression for the block-pool leak livelock, see
#  docs/v6d_mamba_block_leak_hang_report.md)
# ============================================================

class TestLocalKVController:
    @pytest.mark.parametrize("mode", ["OFFLINE", "BLOCKING"])
    def test_store_waits_for_compute_then_copy(self, controller, monkeypatch, mode):
        monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_MODE", mode)
        with patch("sglang_simulator.simulation.vllm.v6d.v6d_backend.time.perf_counter",
                   return_value=0.0) as wall, patch.object(controller, "_store_done") as done:
            meta, block = store_meta(controller, last=True)
            controller.capture_stores(meta)
            controller.cache._block_pool.free_blocks([block])
            assert controller._stores == []
            assert block.ref_cnt == 1
            StateManager.set_global_clock(2.0)
            wall.return_value = 2.0
            controller.progress()
            assert controller.next_wakeup() == 2.5
            for _ in range(3):
                controller.progress()
            assert controller.next_wakeup() == 2.5
            if mode == "OFFLINE":
                wall.return_value = 100.0
            else:
                StateManager.set_global_clock(100.0)
            controller.progress()
            done.assert_not_called()
            StateManager.set_global_clock(2.5)
            wall.return_value = 2.5
            controller.progress()
            done.assert_called_once()
            assert block.ref_cnt == 0
            assert not controller.has_pending()

    def test_completed_chunk_does_not_release_new_chunk(self, controller):
        meta, first = store_meta(controller)
        controller.capture_stores(meta)
        controller.cache._block_pool.free_blocks([first])
        controller.progress()
        StateManager.set_global_clock(0.25)
        meta, second = store_meta(controller, last=True)
        controller.capture_stores(meta)
        controller.cache._block_pool.free_blocks([second])
        controller.progress()
        with patch.object(controller, "_store_done") as done:
            StateManager.set_global_clock(0.5)
            controller.progress()
            assert (first.ref_cnt, second.ref_cnt) == (0, 1)
            done.assert_not_called()
            StateManager.set_global_clock(1.0)
            controller.progress()
            done.assert_called_once()
            assert second.ref_cnt == 0

    def test_empty_last_marker_waits_for_prior_copy(self, controller):
        meta, _block = store_meta(controller)
        controller.capture_stores(meta)
        controller.progress()
        StateManager.set_global_clock(0.1)
        meta, _ = store_meta(controller, last=True, empty=True)
        controller.capture_stores(meta)
        with patch.object(controller, "_store_done") as done:
            controller.progress()
            done.assert_not_called()
            StateManager.set_global_clock(0.5)
            controller.progress()
            controller.progress()
            done.assert_called_once()

    def test_abort_cleanup_cannot_release_staged_or_inflight_pins(self, controller):
        meta, block = store_meta(controller)
        controller.capture_stores(meta)
        controller.cache._block_pool.free_blocks([block])
        controller.cache._release_protected_blocks("r")
        assert block.ref_cnt == 1
        controller.progress()
        controller.cache._release_protected_blocks("r")
        assert block.ref_cnt == 1
        StateManager.set_global_clock(0.5)
        controller.progress()
        assert block.ref_cnt == 0

    def test_long_prefill_with_tiny_pool(self, controller):
        pool = controller.cache._block_pool
        with patch.object(controller, "_store_done") as done:
            for chunk in range(256):
                meta, block = store_meta(controller, last=chunk == 255)
                controller.capture_stores(meta)
                pool.free_blocks([block])
                StateManager.step_global_clock(1.0)
                controller.progress()
                done.assert_not_called()
            controller.reset()
            done.assert_called_once_with(
                controller.hybrid._saving["r"]._req, failed=True)
        assert pool.get_num_free_blocks() == 7
        assert not controller.has_pending()
        controller.hybrid._step_saved.assert_called_once()

    def test_load_deadline_is_not_spent_twice(self, controller):
        req = SimpleNamespace(request_id="load")
        groups = {0: (["key"], [1])}
        controller.queue_load(req, 4224, groups)
        StateManager.set_global_clock(0.25)
        controller.queue_load(req, 4224, groups)
        with patch.object(controller, "_load_done") as done:
            for _ in range(3):
                controller.progress()
            done.assert_not_called()
            assert controller.next_wakeup() == 0.5
            StateManager.set_global_clock(0.5)
            controller.progress()
            controller.progress()
            done.assert_called_once_with("load", 4224)

    @pytest.mark.parametrize("last", [False, True])
    @pytest.mark.parametrize("started", [False, True])
    def test_abort_waits_for_copies_and_reports_failure_once(self, controller, last, started):
        meta, block = store_meta(controller, last=last)
        controller.capture_stores(meta)
        controller.cache._block_pool.free_blocks([block])
        if started:
            controller.progress()
        abort = SimpleNamespace(reqs=SimpleNamespace(v6d=SimpleNamespace(
            aborted_save_ids=["r"], inner=None)))
        controller.capture_stores(abort)
        controller.capture_stores(abort)
        with patch.object(controller, "_store_done") as done:
            controller.progress()
            done.assert_not_called()
            assert block.ref_cnt == 1
            assert controller.next_wakeup() == 0.5
            StateManager.set_global_clock(0.5)
            controller.progress()
            controller.progress()
            done.assert_called_once_with(controller.hybrid._saving["r"]._req, failed=True)
        assert block.ref_cnt == 0
        assert not controller.has_pending()

    def test_reset_fails_unfinished_saves_and_releases_pins(self, controller):
        controller.capture_stores(SimpleNamespace(aborted_save_ids=["r", "missing"]))
        assert controller.has_pending()
        assert controller.next_wakeup() == 0.0
        with patch.object(controller, "_store_done") as done:
            controller.reset()
            done.assert_called_once_with(controller.hybrid._saving["r"]._req, failed=True)
        assert not controller.has_pending()
        assert StateManager.get_global_clock() == 0.0

    def test_reset_refused_while_requests_active(self, controller):
        meta, block = store_meta(controller, last=True)
        controller.capture_stores(meta)
        controller.cache._block_pool.free_blocks([block])
        controller.hybrid._saving["r"]._req.is_finished = lambda: False
        with patch.object(controller, "_store_done") as done:
            assert not controller.reset()
            done.assert_not_called()
        assert controller.has_pending()
        assert block.ref_cnt == 1

    def test_controllers_do_not_share_state(self, controller):
        second = make_controller()
        meta, _ = store_meta(controller)
        controller.capture_stores(meta)
        assert controller.has_pending()
        assert not second.has_pending()
        assert second.backend._sim_controller is second

    @pytest.mark.parametrize("failed", [False, True])
    def test_native_source_rank_acknowledgements(self, controller, failed):
        import asyncio
        calls = []
        async def save(rank, ret):
            calls.append((rank, ret.reqid, ret.source, ret.n))
        controller.hybrid._do_save_done = save
        with patch.object(controller, "_run_control", side_effect=asyncio.run), \
                patch("vllm.v1.hybrid_connector.get_param", return_value=("v6d_object", "kvt")):
            controller._store_done(SimpleNamespace(request_id="r"), failed=failed)
        n = 0 if failed else None
        assert calls == [(0, "r", "v6d_object", n), (1, "r", "v6d_object", n),
                         (0, "r", "kvt", n), (1, "r", "kvt", n)]


class TestConnectorAdapters:
    def test_worker_clear_and_bypass_do_not_publish_completion(self):
        from sglang_simulator.simulation.vllm.v6d.v6d_backend import (
            C_HybridConnectorHook, C_V6dObjectBackendHook,
        )
        class Backend:
            def clear_backend_metadata(self):
                raise AssertionError("worker must not launch stores")
            def bypass_bind(self, metadata):
                raise AssertionError("worker must not schedule abort timers")
        class Connector:
            pass
        C_V6dObjectBackendHook.hook(Backend)
        C_HybridConnectorHook.hook(Connector)
        backend = Backend()
        metadata = SimpleNamespace(aborted_save_ids=["r"])
        backend._bound_meta = metadata
        backend.bypass_bind(metadata)
        assert backend._bound_meta is metadata
        worker = SimpleNamespace(_meta=metadata)
        def clear():
            backend.clear_backend_metadata()
            worker._meta = None
        worker.clear_connector_metadata = clear
        connector = Connector()
        connector._worker = worker
        connector.clear_connector_metadata()
        assert worker._meta is None
        assert backend._bound_meta is None

    @pytest.mark.parametrize("tokens,result", [(0, None), (16, "fallback"), (16, None)])
    def test_backend_load_adapter(self, controller, tokens, result):
        import asyncio
        from sglang_simulator.simulation.vllm.v6d.v6d_backend import C_V6dObjectBackendHook
        class Backend:
            def __init__(self):
                self._scheduler = SimpleNamespace(_reqs_to_load={"r": {0: (["key"], [1])}})
                self._sim_controller = controller
            async def async_update_state_after_alloc(self, request, blocks, n):
                return result
        C_V6dObjectBackendHook.hook(Backend)
        backend = Backend()
        assert asyncio.run(backend.async_update_state_after_alloc(
            SimpleNamespace(request_id="r"), None, tokens)) == result
        assert controller.has_pending() == (tokens > 0 and result is None)
        async def worker_load():
            return [ret async for ret in backend.async_load_kv(None)]
        assert asyncio.run(worker_load()) == []

    @pytest.mark.parametrize("failure", [False, True])
    def test_preparation_joins_control_without_waiting_for_transfer(self, controller, failure):
        import asyncio
        import threading
        from sglang_simulator.simulation.vllm.v6d.v6d_backend import C_HybridControlPlaneHook
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever)
        thread.start()
        class Hybrid:
            def __init__(self):
                self._backend = controller.backend
                self._backend._sim_sync_prepare = True
                self.loop = loop
            async def _on_add_req(self, req, blocks):
                await asyncio.sleep(0.01)
                if failure:
                    raise ValueError("lookup failed")
                self._sim_controller.queue_load(req, 4224, {0: (["key"], [1])})
            def _step_waiting(self):
                self.task = asyncio.run_coroutine_threadsafe(
                    self._on_add_req(SimpleNamespace(request_id="r"), None), loop)
            def step(self):
                self._step_waiting()
        C_HybridControlPlaneHook.hook(Hybrid)
        hybrid = Hybrid()
        try:
            if failure:
                with pytest.raises(ValueError, match="lookup failed"):
                    hybrid.step()
                with pytest.raises(ValueError, match="lookup failed"):
                    hybrid.task.result(timeout=1)
            else:
                hybrid.step()
                assert hybrid._sim_controller.next_wakeup() == 0.5
                assert StateManager.get_global_clock() == 0.0
                hybrid.task.result(timeout=1)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=1)
            loop.close()

    def test_connector_build_captures_stores_and_keeps_native_finish(self, controller):
        from sglang_simulator.simulation.vllm.v6d.v6d_backend import C_HybridConnectorHook
        metadata, _block = store_meta(controller)
        class Connector:
            def build_connector_meta(self, output):
                return metadata
            def has_requests(self):
                return False
            def bind_connector_metadata(self, metadata):
                self.meta = metadata
            def request_finished_all_groups(self, req, blocks):
                return False, None
        original_bind = Connector.bind_connector_metadata
        original_finish = Connector.request_finished_all_groups
        C_HybridConnectorHook.hook(Connector)
        connector = Connector()
        connector._sched = SimpleNamespace(_sim_controller=controller)
        output = SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(req_id="r")])
        assert connector.build_connector_meta(output) is metadata
        assert not hasattr(output, "_sim_external_computed_tokens")
        assert connector.has_requests()
        assert not hasattr(connector, "simulation_next_wakeup")
        assert connector.get_finished(set()) == (set(), set())
        assert Connector.bind_connector_metadata is original_bind
        assert Connector.request_finished_all_groups is original_finish

    @pytest.mark.parametrize(
        "mode,settle_calls,reap_calls", [
            ("OFFLINE", 1, 1),
            ("BLOCKING", 0, 2),
        ]
    )
    def test_control_plane_barrier_is_offline_only(
            self, controller, monkeypatch, mode, settle_calls, reap_calls):
        from sglang_simulator.simulation.vllm.v6d.v6d_backend import (
            C_HybridControlPlaneHook,
        )

        monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_MODE", mode)

        class Hybrid:
            def __init__(self):
                self._backend = controller.backend
                self._backend._sim_sync_prepare = True
                self.loop = None

            async def _on_add_req(self, req, blocks):
                return None

            def _step_waiting(self):
                return "waiting"

            def step(self):
                return self._step_waiting()

        C_HybridControlPlaneHook.hook(Hybrid)
        hybrid = Hybrid()
        control = hybrid._sim_controller
        control.settle_preparations = MagicMock()
        control.reap_preparations = MagicMock()
        control.progress = MagicMock()

        assert hybrid.step() == "waiting"
        assert control.settle_preparations.call_count == settle_calls
        assert control.reap_preparations.call_count == reap_calls
        assert control.progress.call_count == 2

    def test_control_wait_rejects_its_own_loop(self, controller):
        import asyncio
        async def complete():
            return None
        async def attempt():
            controller.hybrid.loop = asyncio.get_running_loop()
            with pytest.raises(RuntimeError, match="current connector loop"):
                controller._run_control(complete())
        asyncio.run(attempt())


class TestEngineClockBoundary:
    @pytest.mark.parametrize(
        "value, expected",
        [(None, False), ("1", True), ("true", True), ("0", False), ("false", False)],
    )
    def test_record_stats_env(self, monkeypatch, value, expected):
        from sglang_simulator.simulation.manager.env import Envs

        if value is None:
            monkeypatch.delenv("SGLANG_SIMULATOR_RECORD_STATS", raising=False)
        else:
            monkeypatch.setenv("SGLANG_SIMULATOR_RECORD_STATS", value)
        assert Envs.record_stats() is expected

    @pytest.fixture(autouse=True)
    def engine(self, monkeypatch):
        from sglang_simulator.simulation.manager import StateManager
        from sglang_simulator.simulation.vllm import engine_core_pipeline as pipeline
        from sglang_simulator.simulation.types import SimulationMode

        class Engine:
            def __init__(self):
                pass
            def add_request(self, request, request_wave=0):
                self.admitted.append((request.request_id, StateManager.get_global_clock()))
            def step(self):
                return self.native_step()
            def _wait_model_output_future(self, future, step_sout):
                return future.result(), {}

        monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE")
        monkeypatch.setattr(StateManager, "_global_clock", 1.0)
        monkeypatch.setattr(pipeline.ReqDispatcher, "_instance", None)
        monkeypatch.setattr(pipeline.ReqDispatcher, "_initialized", False)
        monkeypatch.setattr(pipeline.C_VLLMEngineCoreHook, "SIM_MODE", SimulationMode.OFFLINE)
        pipeline.C_VLLMEngineCoreHook.hook(Engine)
        self.engine = object.__new__(Engine)
        self.engine.admitted = []
        self.engine.native_step = lambda: ({}, False, None)
        self.dispatcher = pipeline.ReqDispatcher()
        self.dispatcher.all_received = True
        self.active_requests = 0
        # Public work counts include parked arrivals via C_VLLMSchedulerHook.
        # No waiting/running queues or backend implementation are exposed.
        self.engine.scheduler = SimpleNamespace(
            get_kv_connector=lambda: None,
            get_num_unfinished_requests=lambda: self.active_requests + len(self.dispatcher))
        monkeypatch.setattr(StateManager, "_current_inference_dur", 0.0)
        monkeypatch.setattr(StateManager, "_last_inference_dur", 0.0)
        self.state = StateManager

    def park(self, when):
        req = SimpleNamespace(request_id=f"future-{when}", prompt_token_ids=[1])
        self.dispatcher.add(self.engine, req, 0, when)

    @pytest.mark.parametrize("owner", ["scheduler", "connector", "connector_loading"])
    def test_pending_work_advances_fixed_tick(self, owner):
        if owner == "scheduler":
            self.active_requests = 1
        else:
            connector = SimpleNamespace(has_requests=lambda: owner == "connector",
                                        pending_requests=[object()] if owner == "connector_loading" else [])
            self.engine.scheduler.get_kv_connector = lambda: connector
        self.engine.step()
        assert StateManager.get_global_clock() == pytest.approx(1.005)
        assert StateManager.get_current_inference_dur() == 0.005

    def test_pending_work_does_not_jump_to_distant_arrival(self):
        self.park(100.0)
        self.engine.scheduler.get_kv_connector = lambda: SimpleNamespace(has_requests=lambda: True)
        self.engine.step()
        assert StateManager.get_global_clock() == pytest.approx(1.005)
        assert self.engine.admitted == []

    def test_arrival_within_tick_is_dispatched_on_next_pass(self):
        self.park(1.003)
        self.active_requests = 1
        self.engine.step()
        self.engine.step()
        assert self.engine.admitted[0][0] == "future-1.003"
        assert self.engine.admitted[0][1] == pytest.approx(1.005)

    def test_no_pending_work_does_not_tick(self):
        self.engine.step()
        assert StateManager.get_global_clock() == 1.0
        assert StateManager.get_current_inference_dur() == 0.0

    def test_future_only_replay_waits_until_all_received(self):
        self.park(100.0)
        self.dispatcher.all_received = False
        self.engine.step()
        assert StateManager.get_global_clock() == 1.0

    def test_no_connector_idle_replay_advances_to_arrival(self):
        self.park(3.0)
        self.engine.step()
        self.engine.step()
        assert self.engine.admitted == [("future-3.0", 3.0)]

    def test_model_execution_does_not_poll_pending_work_or_tick(self):
        get_connector = MagicMock(side_effect=AssertionError("unexpected idle poll"))
        self.engine.scheduler.get_kv_connector = get_connector
        self.engine.native_step = lambda: ({}, True, None)
        self.engine.step()
        get_connector.assert_not_called()
        assert StateManager.get_global_clock() == 1.0
        assert StateManager.get_current_inference_dur() == 0.0

    @pytest.mark.parametrize("transfer", ["load", "store"])
    def test_idle_ticks_publish_transfer_only_after_ready_time(self, controller, monkeypatch, transfer):
        from sglang_simulator.simulation.vllm.v6d.bandwidth import BandwidthModel

        monkeypatch.setattr(BandwidthModel, "get", lambda: SimpleNamespace(
            latency_for=lambda n, load: 0.012, seg1_latency=lambda n: 0.0,
            store_completion_latency=lambda n: 0.012))
        StateManager.set_global_clock(1.0)
        if transfer == "load":
            controller.queue_load(SimpleNamespace(request_id="r"), 4224, {0: (["key"], [1])})
            method = "_load_done"
        else:
            metadata, block = store_meta(controller, last=True)
            controller.capture_stores(metadata)
            controller.cache._block_pool.free_blocks([block])
            method = "_store_done"
        # The engine sees only a native has_requests interface, not a deadline.
        self.engine.scheduler.get_kv_connector = lambda: SimpleNamespace(
            has_requests=controller.has_pending)
        def native_step():
            controller.progress()
            return {}, False, None
        self.engine.native_step = native_step
        with patch.object(controller, method) as done:
            for _ in range(3):
                self.engine.step()
                done.assert_not_called()
            assert StateManager.get_global_clock() == pytest.approx(1.015)
            self.engine.step()
            done.assert_called_once()
        assert not controller.has_pending()
        assert StateManager.get_global_clock() == pytest.approx(1.015)
        if transfer == "store":
            assert block.ref_cnt == 0

    def test_profile_does_not_implicitly_reset_connector(self, monkeypatch, tmp_path):
        reset_connector_cache = MagicMock()
        self.engine.scheduler.get_kv_connector = lambda: SimpleNamespace()
        self.engine.scheduler.reset_connector_cache = reset_connector_cache
        monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_DIR", str(tmp_path))
        self.engine.profile(False)
        reset_connector_cache.assert_not_called()
        assert StateManager.get_global_clock() == 0.0

    def test_blocking_step_never_polls_idle_work_or_advances_virtual_time(self, monkeypatch):
        from sglang_simulator.simulation.vllm import engine_core_pipeline as pipeline
        from sglang_simulator.simulation.types import SimulationMode

        monkeypatch.setattr(pipeline.C_VLLMEngineCoreHook, "SIM_MODE", SimulationMode.BLOCKING)
        get_connector = MagicMock(side_effect=AssertionError("BLOCKING polled idle work"))
        self.engine.scheduler.get_kv_connector = get_connector
        self.active_requests = 1
        self.park(2.0)
        assert self.engine.step() == ({}, False, None)
        get_connector.assert_not_called()
        assert StateManager.get_global_clock() == 1.0
        assert StateManager.get_current_inference_dur() == 0.0
        assert self.engine.admitted == []

    def test_blocking_stats_recording_can_be_disabled(self, monkeypatch, tmp_path):
        from sglang_simulator.simulation.req_stats_manager import request_stats_manager
        from sglang_simulator.simulation.types import SimulationMode
        from sglang_simulator.simulation.vllm import engine_core_pipeline as pipeline

        monkeypatch.setattr(pipeline.C_VLLMEngineCoreHook, "SIM_MODE", SimulationMode.BLOCKING)
        monkeypatch.setattr(pipeline, "_RECORD_STATS", False)
        monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_DIR", str(tmp_path))
        request_stats_manager.reset()
        pipeline.C_VLLMEngineCoreHook.ITERATION_STATS.clear()

        req = SimpleNamespace(request_id="r", prompt_token_ids=[1])
        assert pipeline._new_request_stats(req, 0.0, 0.0, 0.0) is None
        assert request_stats_manager.stats == {}

        output = SimpleNamespace(
            _sim_iteration_stat={"forward_latency": 1.0},
            num_scheduled_tokens={"r": 1},
        )
        pipeline._finalize_step_stats(output, 1.0)
        assert pipeline.C_VLLMEngineCoreHook.ITERATION_STATS == []

        self.engine.profile(False)
        assert (tmp_path / "request.jsonl").read_text() == ""
        assert (tmp_path / "iteration.jsonl").read_text() == ""

    def test_stats_switch_does_not_disable_offline_recording(self, monkeypatch):
        from sglang_simulator.simulation.req_stats_manager import request_stats_manager
        from sglang_simulator.simulation.types import SimulationMode
        from sglang_simulator.simulation.vllm import engine_core_pipeline as pipeline

        monkeypatch.setattr(pipeline.C_VLLMEngineCoreHook, "SIM_MODE", SimulationMode.OFFLINE)
        monkeypatch.setattr(pipeline, "_RECORD_STATS", False)
        request_stats_manager.reset()
        pipeline.C_VLLMEngineCoreHook.ITERATION_STATS.clear()

        req = SimpleNamespace(request_id="r", prompt_token_ids=[1])
        assert pipeline._new_request_stats(req, 0.0, 0.0, 0.0) is not None
        output = SimpleNamespace(
            _sim_iteration_stat={"forward_latency": 1.0},
            num_scheduled_tokens={"r": 1},
            _sim_token_emitted={"r": True},
        )
        pipeline._finalize_step_stats(output, 1.0)
        assert len(pipeline.C_VLLMEngineCoreHook.ITERATION_STATS) == 1

    @pytest.mark.parametrize("is_start", [False, True])
    def test_blocking_profile_does_not_reset_connector(self, monkeypatch, tmp_path, is_start):
        from sglang_simulator.simulation.vllm import engine_core_pipeline as pipeline
        from sglang_simulator.simulation.types import SimulationMode

        monkeypatch.setattr(pipeline.C_VLLMEngineCoreHook, "SIM_MODE", SimulationMode.BLOCKING)
        reset = MagicMock(side_effect=AssertionError("BLOCKING profile reset connector"))
        self.engine.scheduler.get_kv_connector = lambda: SimpleNamespace()
        self.engine.scheduler.reset_connector_cache = reset
        monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_DIR", str(tmp_path))
        self.engine.profile(is_start)
        reset.assert_not_called()

    def test_blocking_transfer_deadline_survives_profile_reset(self, controller, monkeypatch, tmp_path):
        from sglang_simulator.simulation.vllm import engine_core_pipeline as pipeline
        from sglang_simulator.simulation.types import SimulationMode

        monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_MODE", "BLOCKING")
        monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_DIR", str(tmp_path))
        monkeypatch.setattr(pipeline.C_VLLMEngineCoreHook, "SIM_MODE", SimulationMode.BLOCKING)
        self.engine.scheduler.get_kv_connector = lambda: SimpleNamespace(
            reset_cache=controller.reset)
        self.engine.scheduler.reset_connector_cache = MagicMock(
            side_effect=AssertionError("BLOCKING profile reset connector"))
        with patch("sglang_simulator.simulation.vllm.v6d.v6d_backend.time.perf_counter",
                   return_value=1.0) as wall, patch.object(controller, "_load_done") as done:
            controller.queue_load(SimpleNamespace(request_id="load"), 4224, {0: (["key"], [1])})
            self.engine.profile(False)
            assert StateManager.get_global_clock() == 0.0
            assert controller.next_wakeup() == 1.5
            wall.return_value = 1.49
            controller.progress()
            done.assert_not_called()
            wall.return_value = 1.5
            controller.progress()
            done.assert_called_once_with("load", 4224)
        assert not controller.has_pending()

    def test_executor_preserves_timing_and_existing_hit_statistics(self, monkeypatch):
        from sglang_simulator.simulation.vllm import engine_core_pipeline as pipeline
        from sglang_simulator.simulation.req_stats_manager import request_stats_manager

        class Executor:
            def execute_model(self, output):
                assert not hasattr(output, "_sim_step_duration")
                assert StateManager.get_global_clock() == 1.0
                assert output._sim_token_emitted == {"r": True}
                return "native-output"
        predictor = SimpleNamespace(predict_infer_time=lambda batch: 1.0,
                                    predict_sample_tokens_time=lambda tokens: 0.25)
        monkeypatch.setattr(pipeline.C_VLLMEngineCoreHook, "INFERENCE_PREDICTOR", predictor)
        monkeypatch.setattr(pipeline.C_VLLMEngineCoreHook, "ITERATION_STATS", [])
        monkeypatch.setattr(request_stats_manager, "stats", {})
        req = SimpleNamespace(request_id="r", prompt_token_ids=[1] * 16)
        st = pipeline._new_request_stats(req, 0.0, 0.0, 0.0)
        output = SimpleNamespace(
            num_scheduled_tokens={"r": 4}, finished_req_ids=set(),
            scheduled_new_reqs=[SimpleNamespace(
                req_id="r", prompt_token_ids=req.prompt_token_ids,
                num_computed_tokens=12, sampling_params=SimpleNamespace(max_tokens=1))])
        monkeypatch.setattr(pipeline, "_ext_computed_tokens", lambda rid: 8)
        pipeline.C_VLLMExecutorHook.hook(Executor)
        assert Executor().execute_model(output) == "native-output"
        assert StateManager.get_global_clock() == 2.25
        assert (st.ext_kv_hit_len, st.local_kv_hit_len) == (8, 4)

    @pytest.mark.parametrize("tokens", [0, 4])
    @pytest.mark.parametrize("mode", ["OFFLINE", "BLOCKING"])
    def test_executor_preserves_worker_future_and_execution_order(self, monkeypatch, tokens, mode):
        from concurrent.futures import Future
        from sglang_simulator.simulation.vllm import engine_core_pipeline as pipeline
        from sglang_simulator.simulation.types import SimulationMode

        calls = []
        worker_future = Future()
        worker_future.set_result("model-output")
        class Executor:
            # Fake executor with no scheduler_config -> async_scheduling reads
            # False, exercising the P-side (non-async) deferred-future path.
            def execute_model(self, output):
                calls.append("execute")
                return worker_future
        def predict(batch):
            calls.append("predict")
            return 0.5
        monkeypatch.setattr(pipeline.C_VLLMEngineCoreHook, "SIM_MODE", SimulationMode(mode))
        monkeypatch.setattr(pipeline, "_RECORD_STATS", True)
        monkeypatch.setattr(pipeline.C_VLLMExecutorHook, "_COLD_START_DONE", True)
        monkeypatch.setattr(pipeline.C_VLLMEngineCoreHook, "INFERENCE_PREDICTOR", SimpleNamespace(
            predict_infer_time=predict))
        monkeypatch.setattr(pipeline.C_VLLMEngineCoreHook, "ITERATION_STATS", [])
        pipeline.C_VLLMExecutorHook.hook(Executor)
        output = SimpleNamespace(
            num_scheduled_tokens={"r": tokens} if tokens else {},
            scheduled_new_reqs=[SimpleNamespace(req_id="r", prompt_token_ids=[1] * tokens,
                                               num_computed_tokens=0, sampling_params=None)])
        with patch.object(pipeline.time, "sleep") as sleep:
            returned = Executor().execute_model(output)
            if not tokens:
                # Empty sim batch: native passthrough, no accounting/deferral.
                assert returned is worker_future
                assert calls == ["execute"]
                sleep.assert_not_called()
                assert StateManager.get_global_clock() == 1.0
                return
            # Prediction runs before the original call in both modes.
            assert calls == ["predict", "execute"]
            # Non-async path never annotates the worker-side latency carrier.
            assert not hasattr(output, "_sim_full_step_latency")
            if mode == "OFFLINE":
                # Synchronous virtual-clock path returns the worker future as-is.
                assert returned is worker_future
                sleep.assert_not_called()
                assert StateManager.get_global_clock() == 1.5
                assert len(pipeline.C_VLLMEngineCoreHook.ITERATION_STATS) == 1
                return
            # BLOCKING (async off): a NEW deferred future is returned; the span
            # is slept on the sim GPU pool and stats publish only after it.
            assert returned is not worker_future
            assert isinstance(returned, Future)
            assert returned.result(timeout=5) == "model-output"
            sleep.assert_called_once_with(pytest.approx(0.5))
            assert len(pipeline.C_VLLMEngineCoreHook.ITERATION_STATS) == 1
            assert StateManager.get_global_clock() == 1.0

    @pytest.mark.parametrize("async_scheduling", [False, True])
    @pytest.mark.parametrize("mode", ["OFFLINE", "BLOCKING"])
    @pytest.mark.parametrize("tokens", [0, 4])
    def test_worker_lifecycle_leaves_transfer_progress_to_connector(
            self, controller, monkeypatch, mode, tokens, async_scheduling):
        from sglang_simulator.simulation.vllm import engine_core_pipeline as pipeline
        from sglang_simulator.simulation.vllm.worker import C_VLLMWorkerHook
        from sglang_simulator.simulation.types import SimulationMode
        import vllm.distributed.kv_transfer as transfer

        monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_MODE", mode)
        monkeypatch.setattr(pipeline.C_VLLMEngineCoreHook, "SIM_MODE", SimulationMode(mode))
        monkeypatch.setattr(pipeline.C_VLLMExecutorHook, "_COLD_START_DONE", True)
        monkeypatch.setattr(pipeline.C_VLLMEngineCoreHook, "ITERATION_STATS", [])
        monkeypatch.setattr(pipeline.C_VLLMEngineCoreHook, "INFERENCE_PREDICTOR", SimpleNamespace(
            predict_infer_time=lambda batch: 0.5))
        StateManager.set_global_clock(1.0)
        metadata, block = store_meta(controller, last=True)
        controller.capture_stores(metadata)
        controller.cache._block_pool.free_blocks([block])
        calls = []
        def record(phase):
            calls.append((phase, controller.now()))
        connector = SimpleNamespace(
            bind_connector_metadata=lambda metadata: record("bind"),
            start_load_kv=lambda context: record("load"),
            wait_for_save=lambda: record("save"),
            get_finished=lambda ids: (set(), set()),
            clear_connector_metadata=lambda: record("clear"))
        monkeypatch.setattr(transfer, "has_kv_transfer_group", lambda: True)
        monkeypatch.setattr(transfer, "get_kv_transfer_group", lambda: connector)
        class Worker:
            pass
        C_VLLMWorkerHook.hook(Worker)
        worker = Worker()
        worker.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=async_scheduling)
        )
        class Executor:
            # Real UniProcExecutor exposes scheduler_config; mirror it so the
            # executor-side async_scheduling read matches the worker-side.
            scheduler_config = SimpleNamespace(async_scheduling=async_scheduling)
            def execute_model(self, output):
                result = worker.execute_model(output)
                actual = getattr(result, "_output", result)
                assert calls == [("bind", 1.0), ("load", 1.0),
                                 ("save", 1.0), ("clear", 1.0)]
                assert actual.kv_connector_output is not None
                assert controller._stores == []
                return result
        pipeline.C_VLLMExecutorHook.hook(Executor)
        new_reqs = [SimpleNamespace(req_id="r", prompt_token_ids=[1] * tokens,
                                   num_computed_tokens=0, sampling_params=None)] if tokens else []
        output = SimpleNamespace(num_scheduled_tokens={"r": tokens} if tokens else {},
                                 scheduled_new_reqs=new_reqs, kv_connector_metadata=object())
        with patch("sglang_simulator.simulation.vllm.v6d.v6d_backend.time.perf_counter",
                   return_value=1.0) as wall, \
                patch.object(pipeline.time, "sleep") as sleep:
            sleep.side_effect = lambda duration: setattr(wall, "return_value", wall.return_value + duration)
            result = Executor().execute_model(output)
            if mode == "BLOCKING" and tokens:
                if async_scheduling:
                    result = worker.sample_tokens(None).get_output()
                else:
                    # Non-async path returns a deferred future; the span sleep
                    # runs on the sim GPU pool. Block on it to synchronize.
                    result = result.result(timeout=5)
            end = 1.5 if tokens else 1.0
            assert controller.now() == end
            assert controller._stores == []
            controller.progress()
            assert controller.next_wakeup() == end + 0.5
            assert block.ref_cnt == 1
            with patch.object(controller, "_store_done") as done:
                StateManager.set_global_clock(end + 0.5)
                wall.return_value = end + 0.5
                controller.progress()
                done.assert_called_once()
            assert block.ref_cnt == 0
        assert calls == [("bind", 1.0), ("load", 1.0), ("save", 1.0), ("clear", 1.0)]
        assert result.kv_connector_output is not None
        assert not hasattr(output, "_sim_step_duration")
        assert not hasattr(connector, "_sim_step_duration")
        if mode == "BLOCKING":
            assert sleep.call_count == int(bool(tokens))

    def test_inference_scheduler_hook_only_changes_counts(self):
        from sglang_simulator.simulation.vllm.engine_core_pipeline import C_VLLMSchedulerHook

        class Scheduler:
            def schedule(self):
                return "native"
            def update_from_output(self):
                return "native"
            def get_num_unfinished_requests(self):
                return 0
            def has_unfinished_requests(self):
                return False
        original = dict(Scheduler.__dict__)
        C_VLLMSchedulerHook.hook(Scheduler)
        changed = {key for key, value in Scheduler.__dict__.items()
                   if value is not original.get(key)}
        assert changed == {"get_num_unfinished_requests", "has_unfinished_requests"}
        self.park(2.0)
        assert Scheduler().get_num_unfinished_requests() == 1
        assert Scheduler().has_unfinished_requests()


class TestChunkStoreReferenceOwnership:
    def test_abort_cleanup_cannot_release_event_owned_references(self, controller):
        scheduler = controller.cache
        pool = scheduler._block_pool
        meta, first = store_meta(controller)
        controller.capture_stores(meta)
        _, second = store_meta(controller)
        pool.free_blocks([first, second])  # Scheduler drops its own references.
        scheduler._release_protected_blocks("r")  # Abort on the async thread.
        assert first.ref_cnt == 1
        assert second.ref_cnt == 0
        assert pool.get_num_free_blocks() == 6
        controller.progress()
        StateManager.set_global_clock(0.5)
        controller.progress()
        scheduler._release_protected_blocks("r")
        assert pool.get_num_free_blocks() == 7

    def test_repeated_reference_is_transferred_once_per_chunk(self, controller):
        scheduler = controller.cache
        pool = scheduler._block_pool
        meta, block = store_meta(controller)
        block.ref_cnt += 1
        scheduler._swap_protected_blocks["r"].append(block)
        controller.capture_stores(meta)
        assert controller._staged[0].blocks == [block]
        assert scheduler._swap_protected_blocks["r"] == [block]
        scheduler._release_protected_blocks("r")
        pool.free_blocks([block])
        assert block.ref_cnt == 1
        controller.progress()
        StateManager.set_global_clock(0.5)
        controller.progress()
        assert pool.get_num_free_blocks() == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
