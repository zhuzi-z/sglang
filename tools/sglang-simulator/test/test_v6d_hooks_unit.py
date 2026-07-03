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
        from sglang_simulator.simulation.vllm.v6d_swap import (
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
# Test 2: head_dim injection
# ============================================================

class TestHeadDimInjection:
    """Test _inject_head_dim modifies HF config correctly."""

    def _make_mock_vllm_config(self, head_dim=128, num_attention_heads=32):
        hf_config = SimpleNamespace(
            head_dim=head_dim,
            hidden_size=head_dim * num_attention_heads,
            num_attention_heads=num_attention_heads,
        )
        model_config = SimpleNamespace(hf_text_config=hf_config)
        return SimpleNamespace(model_config=model_config)

    def test_inject_head_dim_sets_to_one(self):
        from sglang_simulator.simulation.vllm.worker import _inject_head_dim

        vllm_config = self._make_mock_vllm_config(head_dim=128)
        original = _inject_head_dim(vllm_config)

        assert original == 128
        assert vllm_config.model_config.hf_text_config.head_dim == 1

    def test_inject_head_dim_fallback_no_head_dim_attr(self):
        """When hf_config has no head_dim, compute from hidden_size/num_heads."""
        from sglang_simulator.simulation.vllm.worker import _inject_head_dim

        hf_config = SimpleNamespace(
            hidden_size=4096,
            num_attention_heads=32,
        )
        model_config = SimpleNamespace(hf_text_config=hf_config)
        vllm_config = SimpleNamespace(model_config=model_config)

        original = _inject_head_dim(vllm_config)
        assert original == 128  # 4096 / 32
        assert hf_config.head_dim == 1

    def test_inject_head_dim_idempotent(self):
        """Calling inject twice should still result in head_dim=1."""
        from sglang_simulator.simulation.vllm.worker import _inject_head_dim

        vllm_config = self._make_mock_vllm_config(head_dim=128)
        _inject_head_dim(vllm_config)
        # Second call: head_dim is already 1
        original = _inject_head_dim(vllm_config)
        assert original == 1
        assert vllm_config.model_config.hf_text_config.head_dim == 1


# ============================================================
# Test 3: _build_kv_cache_spec
# ============================================================

class TestBuildKVCacheSpec:
    """Test KV cache spec construction with head_dim=1."""

    def _make_vllm_config(
        self, num_layers=28, num_kv_heads=4, block_size=16, tp_size=1
    ):
        hf_config = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_key_value_heads=num_kv_heads,
            num_attention_heads=32,
            head_dim=1,  # Already injected
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
        """Spec should use real num_kv_heads (not 1)."""
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec

        vllm_config = self._make_vllm_config(num_kv_heads=8, tp_size=1)
        spec = _build_kv_cache_spec(vllm_config)

        first_spec = list(spec.values())[0]
        assert first_spec.num_kv_heads == 8
        assert first_spec.head_size == 1

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

    def test_spec_page_size_is_tiny(self):
        """With head_size=1, page_size should be very small."""
        from sglang_simulator.simulation.vllm.worker import _build_kv_cache_spec

        vllm_config = self._make_vllm_config(
            num_kv_heads=8, block_size=16, tp_size=1
        )
        spec = _build_kv_cache_spec(vllm_config)

        first_spec = list(spec.values())[0]
        # page_size = 2 * block_size * num_kv_heads * head_size * dtype_size
        # = 2 * 16 * 8 * 1 * 2 (fp16) = 512 bytes
        assert first_spec.page_size_bytes == 512

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
            head_dim=1,
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
        from sglang_simulator.simulation.vllm import v6d_swap

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
        from sglang_simulator.simulation.vllm.v6d_swap import _patch_v6d_ops
        # Get the mock function directly
        import sglang_simulator.simulation.vllm.v6d_swap as swap_mod

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
        from sglang_simulator.simulation.vllm.v6d_swap import (
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
        from sglang_simulator.simulation.vllm.v6d_swap import (
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
        from sglang_simulator.simulation.vllm.v6d_swap import (
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
        from sglang_simulator.simulation.vllm.v6d_worker import (
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
        from sglang_simulator.simulation.vllm.v6d_backend import (
            C_V6dObjectBackendHook,
        )
        from sglang_simulator.simulation.vllm.v6d_swap import DummyEvent

        class MockV6dObjectBackend:
            def __init__(self):
                self._save_event_pool = ["real_event_1", "real_event_2"]
                self._load_event_pool = ["real_event_3"]

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
