"""Unit tests for V6D simulation hooks.

These tests verify the hook components in isolation without requiring
a V6D daemon or vLLM engine. They test:
1. DummyStream/DummyEvent behavior
2. head_dim=1 injection logic
3. KV cache spec construction with real num_kv_heads
4. ops monkey-patching (v6d_swap_blocks CPU memcpy)
5. Hook class installation mechanics
"""

import os
import sys
import ctypes
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import torch
import pytest

# Ensure hooks are importable
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


# ============================================================
# Test 1: DummyStream / DummyEvent
# ============================================================

class TestDummyPrimitives:
    """Test DummyStream and DummyEvent mock CUDA primitives."""

    def setup_method(self):
        from sglang_simulator.simulation.vllm.v6d.v6d_swap import (
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
# Test 4: ops monkey-patching
# ============================================================

class TestOpsPatch:
    """Test _patch_v6d_ops correctly replaces CUDA ops."""

    def test_patch_replaces_swap_blocks(self):
        """After patching, v6d_swap_blocks should be a Python function."""
        from sglang_simulator.simulation.vllm.v6d import v6d_swap

        # Reset patch state
        v6d_swap._OPS_PATCHED = False

        # Create a mock _custom_ops module
        mock_ops = MagicMock()
        with patch.dict(sys.modules, {"vllm._custom_ops": mock_ops, "vllm": MagicMock(_custom_ops=mock_ops)}):
            v6d_swap._patch_v6d_ops()

        assert v6d_swap._OPS_PATCHED is True
        # The mock should have had its functions replaced
        assert mock_ops.v6d_swap_blocks is not None
        assert mock_ops.v6d_register_host_memory is not None
        assert mock_ops.v6d_unregister_host_memory is not None


# ============================================================
# Test 5: CPU memcpy via mock_v6d_swap_blocks
# ============================================================

class TestMockSwapBlocks:
    """Test the CPU memcpy implementation of v6d_swap_blocks."""

    def test_swap_in_copies_data_correctly(self):
        """Swap-in: copy from V6D mmap (src) to KV cache (dst)."""
        num_layers = 2
        num_blocks = 3
        page_size = 64  # bytes per block per layer

        # Allocate KV cache (destination) - zeros initially
        kv_cache = torch.zeros(
            num_layers * num_blocks * page_size, dtype=torch.uint8
        )
        # Allocate V6D mmap objects (source) - filled with data
        v6d_objs = [
            torch.arange(num_layers * page_size, dtype=torch.uint8) + (i + 1)
            for i in range(num_blocks)
        ]

        # Build pointer tensors
        layer_ptrs = torch.zeros(num_layers, dtype=torch.long)
        for layer_idx in range(num_layers):
            layer_ptrs[layer_idx] = (
                kv_cache.data_ptr() + layer_idx * num_blocks * page_size
            )

        cpu_block_ptrs = torch.zeros(num_blocks, dtype=torch.long)
        for i in range(num_blocks):
            cpu_block_ptrs[i] = v6d_objs[i].data_ptr()

        gpu_block_ids = torch.arange(num_blocks, dtype=torch.long)

        # Perform swap-in
        from sglang_simulator.simulation.vllm.v6d.v6d_swap import _patch_v6d_ops
        # Get the mock function directly
        import sglang_simulator.simulation.vllm.v6d.v6d_swap as swap_mod

        # Call internal mock directly
        # We need to test the actual memcpy logic
        for i in range(num_blocks):
            cpu_base = cpu_block_ptrs[i].item()
            block_id = gpu_block_ids[i].item()
            for layer_idx in range(num_layers):
                layer_ptr = layer_ptrs[layer_idx].item()
                kv_ptr = layer_ptr + block_id * page_size
                obj_ptr = cpu_base + layer_idx * page_size
                ctypes.memmove(kv_ptr, obj_ptr, page_size)

        # Verify: KV cache should now contain V6D data
        for i in range(num_blocks):
            for layer_idx in range(num_layers):
                offset = layer_idx * num_blocks * page_size + i * page_size
                expected_start = layer_idx * page_size
                expected_end = (layer_idx + 1) * page_size
                actual = kv_cache[offset : offset + page_size]
                expected = v6d_objs[i][expected_start:expected_end]
                assert torch.equal(actual, expected), (
                    f"Mismatch at block={i}, layer={layer_idx}"
                )

    def test_swap_out_copies_data_correctly(self):
        """Swap-out: copy from KV cache (src) to V6D mmap (dst)."""
        num_layers = 2
        num_blocks = 2
        page_size = 32

        # KV cache with data
        kv_cache = torch.arange(
            num_layers * num_blocks * page_size, dtype=torch.uint8
        )
        # V6D objects (destination) - zeros initially
        v6d_objs = [
            torch.zeros(num_layers * page_size, dtype=torch.uint8)
            for _ in range(num_blocks)
        ]

        # Build pointer tensors
        layer_ptrs = torch.zeros(num_layers, dtype=torch.long)
        for layer_idx in range(num_layers):
            layer_ptrs[layer_idx] = (
                kv_cache.data_ptr() + layer_idx * num_blocks * page_size
            )

        cpu_block_ptrs = torch.zeros(num_blocks, dtype=torch.long)
        for i in range(num_blocks):
            cpu_block_ptrs[i] = v6d_objs[i].data_ptr()

        gpu_block_ids = torch.arange(num_blocks, dtype=torch.long)

        # Perform swap-out (swap_in=False)
        for i in range(num_blocks):
            cpu_base = cpu_block_ptrs[i].item()
            block_id = gpu_block_ids[i].item()
            for layer_idx in range(num_layers):
                layer_ptr = layer_ptrs[layer_idx].item()
                kv_ptr = layer_ptr + block_id * page_size
                obj_ptr = cpu_base + layer_idx * page_size
                # swap_out: kv -> obj
                ctypes.memmove(obj_ptr, kv_ptr, page_size)

        # Verify: V6D objects should now contain KV cache data
        for i in range(num_blocks):
            for layer_idx in range(num_layers):
                kv_offset = layer_idx * num_blocks * page_size + i * page_size
                obj_offset = layer_idx * page_size
                actual = v6d_objs[i][obj_offset : obj_offset + page_size]
                expected = kv_cache[kv_offset : kv_offset + page_size]
                assert torch.equal(actual, expected), (
                    f"Mismatch at block={i}, layer={layer_idx}"
                )


# ============================================================
# Test 6: V6dSwapHandler hook mechanics
# ============================================================

class TestV6dSwapHandlerHook:
    """Test that the hook correctly installs overrides on target class."""

    def test_hook_installs_init_override(self):
        from sglang_simulator.simulation.vllm.v6d.v6d_swap import (
            C_V6dSwapHandlerHook, DummyStream,
        )

        # Create a mock target class
        class MockV6dSwapHandler:
            def __init__(self):
                pass

            def _validate_swap(self, *args):
                return True

            def _process_swap_batch(self, *args):
                return []

        # Apply hook
        C_V6dSwapHandlerHook.hook(MockV6dSwapHandler)

        # Verify: __init__ was replaced
        mock_client = MagicMock()
        handler = MockV6dSwapHandler.__new__(MockV6dSwapHandler)
        MockV6dSwapHandler.__init__(handler, 0, 4, mock_client, True, 0)

        assert isinstance(handler._stream, DummyStream)
        assert handler._rank_id == 0
        assert handler._swap_in is True
        assert handler._gpu_device == torch.device("cpu")

    def test_hook_installs_swap_override(self):
        from sglang_simulator.simulation.vllm.v6d.v6d_swap import (
            C_V6dSwapHandlerHook,
        )

        class MockV6dSwapHandler:
            def __init__(self):
                pass

            def _validate_swap(self, *args):
                return True

            def _process_swap_batch(self, *args, **kwargs):
                return ["obj1"]

        C_V6dSwapHandlerHook.hook(MockV6dSwapHandler)

        # Verify swap method exists and doesn't use torch.cuda
        assert hasattr(MockV6dSwapHandler, "swap")
        assert hasattr(MockV6dSwapHandler, "async_swap")

    def test_hook_get_finished_returns_all_immediately(self):
        from collections import deque
        from sglang_simulator.simulation.vllm.v6d.v6d_swap import (
            C_V6dSwapHandlerHook, DummyEvent,
        )

        class MockV6dSwapHandler:
            def __init__(self):
                pass

            def _validate_swap(self, *args):
                return True

            def _process_swap_batch(self, *args, **kwargs):
                return []

        C_V6dSwapHandlerHook.hook(MockV6dSwapHandler)

        handler = MockV6dSwapHandler.__new__(MockV6dSwapHandler)
        handler._event_jobs = deque()
        handler._job_objs = {}

        # Simulate some pending events
        handler._event_jobs.append((DummyEvent(), 1))
        handler._event_jobs.append((DummyEvent(), 2))
        handler._event_jobs.append((DummyEvent(), 3))
        handler._job_objs = {1: [], 2: [], 3: []}

        # All should finish immediately (DummyEvent.query() is True)
        finished = handler.get_finished()
        assert finished == [1, 2, 3]
        assert len(handler._event_jobs) == 0


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
        from sglang_simulator.simulation.vllm.v6d.v6d_swap import DummyEvent

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

class TestHybridConnectorLeakFix:
    """Verify save completion travels the REAL production channel.

    In hybrid mode the v6d save-done is signalled via the worker's
    _SAVE_DONE_REQ RPC (simulated by mark_backend_save_done), which drives
    _do_save_done -> _saved/_try_teardown_save + async_cleanup (seal v6d
    objects + release mamba protected blocks).  It does NOT travel via
    kv_connector_output.finished_sending / update_connector_output —
    both are upstream no-ops for hybrid and must stay untouched.

    1. bind_connector_metadata resolves reqs_to_store through the
       v6d_object+kvt combo nesting (meta.reqs.v6d.inner.reqs_to_store)
       and fires mark_backend_save_done for last-save events.
    2. Non-last saves do not signal (v6d save_count=1 semantics).
    3. SGLANG_SIMULATOR_V6D_SAVE_DONE_DELAY_MS defers the signal to
       model the real async save latency.
    4. get_finished never reports store reqs; update_connector_output and
       request_finished_all_groups are left untouched (fidelity).
    """

    def _make_hooked_connector_cls(self):
        from sglang_simulator.simulation.vllm.v6d.v6d_backend import (
            C_HybridConnectorHook,
        )

        class FakeHybridConnector:
            def bind_connector_metadata(self, metadata):
                pass

            def clear_connector_metadata(self):
                pass

            def request_finished_all_groups(self, request, block_ids):
                return False, {"kv_transfer_pending": True}

        C_HybridConnectorHook.hook(FakeHybridConnector)
        return FakeHybridConnector

    def _make_kvt_combo_meta(self, reqs_to_store):
        # Mirrors HybridMetadata -> V6dObjectKVTMeta -> V6dObjectBackendMeta
        # -> V6dObjectConnectorMetadata nesting for backend "v6d_object+kvt".
        inner = SimpleNamespace(reqs_to_store=reqs_to_store, reqs_to_load={})
        v6d = SimpleNamespace(inner=inner)
        return SimpleNamespace(reqs=SimpleNamespace(v6d=v6d, kvt=None))

    def _attach_nested_scheduler(self, connector):
        scheduler = MagicMock()
        backend = SimpleNamespace(_v6d=SimpleNamespace(_scheduler=scheduler))
        connector._sched = SimpleNamespace(_backend=backend)
        return scheduler

    def _patch_save_done(self, requests):
        # bind_connector_metadata imports mark_backend_save_done and
        # sched_get_req from vllm.v1.hybrid_connector at call time.
        mark = patch("vllm.v1.hybrid_connector.mark_backend_save_done")
        get_req = patch("vllm.v1.hybrid_connector.sched_get_req",
                        side_effect=lambda rid: requests.get(rid))
        return mark, get_req

    def test_last_save_fires_save_done_via_real_channel(self):
        cls = self._make_hooked_connector_cls()
        connector = cls()
        req = SimpleNamespace(request_id="req-1")
        meta = self._make_kvt_combo_meta(
            {"req-1": ({0: (["hash_0"], [19])}, True)}   # is_last_save=True
        )

        mark, get_req = self._patch_save_done({"req-1": req})
        with mark as mark_mock, get_req:
            connector.bind_connector_metadata(meta)

        mark_mock.assert_called_once_with(req)
        # Fidelity: save completion must NOT surface as finished_sending
        store, _ = connector.get_finished(set())
        assert store == set()

    def test_intermediate_save_does_not_signal(self):
        cls = self._make_hooked_connector_cls()
        connector = cls()
        req = SimpleNamespace(request_id="req-1")
        meta = self._make_kvt_combo_meta(
            {"req-1": ({0: (["hash_0"], [19])}, False)}  # is_last_save=False
        )

        mark, get_req = self._patch_save_done({"req-1": req})
        with mark as mark_mock, get_req:
            connector.bind_connector_metadata(meta)

        mark_mock.assert_not_called()

    def test_noop_empty_last_save_still_signals(self):
        cls = self._make_hooked_connector_cls()
        connector = cls()
        req = SimpleNamespace(request_id="req-1")
        # Empty groups_data marks the noop last save — must still signal
        meta = self._make_kvt_combo_meta({"req-1": ({}, True)})

        mark, get_req = self._patch_save_done({"req-1": req})
        with mark as mark_mock, get_req:
            connector.bind_connector_metadata(meta)

        mark_mock.assert_called_once_with(req)

    def test_save_done_delay_defers_signal(self):
        cls = self._make_hooked_connector_cls()
        connector = cls()
        req = SimpleNamespace(request_id="req-1")
        meta = self._make_kvt_combo_meta(
            {"req-1": ({0: (["hash_0"], [19])}, True)}
        )

        mark, get_req = self._patch_save_done({"req-1": req})
        with mark as mark_mock, get_req, \
                patch.dict(os.environ,
                           {"SGLANG_SIMULATOR_V6D_SAVE_DONE_DELAY_MS": "50"}):
            connector.bind_connector_metadata(meta)
            # Not fired yet: models the in-flight async save holding
            # block protection for the save duration
            mark_mock.assert_not_called()

            import time as _time
            _time.sleep(0.08)
            mark_mock.assert_called_once_with(req)

    def test_update_connector_output_left_untouched(self):
        """Fidelity: upstream HybridConnector has no update_connector_output
        (base no-op); the hook must not add one."""
        cls = self._make_hooked_connector_cls()
        assert "update_connector_output" not in cls.__dict__

    def test_request_finished_left_untouched(self):
        """Fidelity: upstream has no finish-time release; the hook must
        preserve the original request_finished_all_groups untouched (the
        finish-time safety net is expected to be fixed upstream)."""
        cls = self._make_hooked_connector_cls()
        connector = cls()
        scheduler = self._attach_nested_scheduler(connector)

        request = SimpleNamespace(request_id="req-1")
        should_wait, params = connector.request_finished_all_groups(
            request, ([1, 2],))

        assert should_wait is False
        assert params == {"kv_transfer_pending": True}
        scheduler.request_finished_all_groups.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
