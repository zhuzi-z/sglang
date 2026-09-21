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
    """Test the new platform mock methods."""

    def test_set_device_is_noop(self):
        """set_device should not raise or do anything."""
        # We test the _MockCudaPlatform via the hook mechanism
        import sglang_simulator.hook as sgl_hook
        from sglang_simulator.simulation.vllm.platform import C_VLLMPlatformHook

        # Simulate: create a base class, apply hook
        class FakePlatform:
            pass

        C_VLLMPlatformHook.hook(FakePlatform)

        # After hook, sys.modules should have the mock
        import vllm.platforms
        platform = vllm.platforms.current_platform

        # set_device should be a no-op
        platform.set_device(torch.device("cpu"))
        platform.set_device(torch.device("cuda:0"))

    def test_get_device_total_memory(self):
        """Should return 80 GiB."""
        import vllm.platforms
        platform = vllm.platforms.current_platform
        mem = platform.get_device_total_memory()
        assert mem == 80 * (1 << 30)


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
        future = Future()
        future.result = MagicMock(side_effect=AssertionError("extra Future wait"))
        class Executor:
            def execute_model(self, output):
                calls.append("execute")
                return future
        def predict(batch):
            calls.append("predict")
            return 0.5
        monkeypatch.setattr(pipeline.C_VLLMEngineCoreHook, "SIM_MODE", SimulationMode(mode))
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
            assert Executor().execute_model(output) is future
            sleep.assert_not_called()
        assert calls == (["predict", "execute"] if tokens else ["execute"])
        if mode == "BLOCKING" and tokens:
            assert output._sim_full_step_latency == pytest.approx(0.5)
            # Completion accounting happens only after EngineCore has waited
            # for the deferred model-output future.
            assert pipeline.C_VLLMEngineCoreHook.ITERATION_STATS == []
            completed = Future()
            completed.set_result("model-output")
            with patch.object(pipeline.time, "time", return_value=123.0):
                assert self.engine._wait_model_output_future(
                    completed, output
                ) == ("model-output", {})
            assert len(pipeline.C_VLLMEngineCoreHook.ITERATION_STATS) == 1
        else:
            assert not hasattr(output, "_sim_full_step_latency")
        future.result.assert_not_called()
        assert StateManager.get_global_clock() == (1.5 if tokens and mode == "OFFLINE" else 1.0)

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
                    result = result.get_output()
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
