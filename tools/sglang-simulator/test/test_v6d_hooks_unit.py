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
    3. BandwidthModel.store_completion_latency defers the signal to
       model the real async save latency (max(DMA, poll) + rank_sync).
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
        _bw_mock = MagicMock()
        _bw_mock.store_completion_latency.return_value = 0.05  # 50ms
        with mark as mark_mock, get_req, \
                patch(
                    "sglang_simulator.simulation.vllm.v6d.v6d_backend"
                    ".BandwidthModel.get",
                    return_value=_bw_mock,
                ):
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


# ============================================================
# Test: V6dObjectManager hook — write-path dedup (P0-A) and
# short-circuit holder registration (P0-B)
# ============================================================

class TestV6dManagerRealLookupPath:
    """The manager hook must NOT override the real connector methods.

    After the real-path alignment, lookup/async_lookup/get_key/
    async_get_key/prepare_batch_allocate/batch_allocate are the production
    connector methods (batch ``client.get``, ``_cached_objs`` skip, holder
    registration); the hook only touches ``_async_connect`` (skip
    ``_on_connected``).  Daemon-down semantics are production's: the
    connect failure is logged and ``client`` stays unset.
    """

    def _make_hooked_manager_cls(self):
        from sglang_simulator.simulation.vllm.v6d.v6d_manager import (
            C_V6dObjectManagerHook,
        )

        class FakeV6dObjectManager:
            def __init__(self):
                self._group_id = 0
                self._pending_objs = {}
                self._cached_objs = {}
                self._cached_objs_reqs = {}
                self.client = None

            def _make_key(self, h):
                return f"key-{h}"

            def lookup(self, block_hashes, request_id=None,
                       unfetched_objs=None):
                raise NotImplementedError

            async def async_lookup(self, block_hashes, request_id=None,
                                   unfetched_objs=None):
                raise NotImplementedError

            def get_key(self, block_hash, request_id=None):
                raise NotImplementedError

            async def async_get_key(self, block_hash, request_id=None):
                raise NotImplementedError

            def prepare_batch_allocate(self, block_hashes):
                raise NotImplementedError

            def batch_allocate(self, *args, **kwargs):
                raise NotImplementedError

        originals = {
            name: getattr(FakeV6dObjectManager, name)
            for name in ("lookup", "async_lookup", "get_key",
                         "async_get_key", "prepare_batch_allocate",
                         "batch_allocate")
        }
        C_V6dObjectManagerHook.hook(FakeV6dObjectManager)
        return FakeV6dObjectManager, originals

    def test_read_path_methods_not_overridden(self):
        cls, originals = self._make_hooked_manager_cls()
        for name, fn in originals.items():
            assert getattr(cls, name) is fn, (
                f"{name} must stay the real connector method")

    def test_async_connect_failure_leaves_client_unset(self, monkeypatch):
        # Production semantics: connect failure is logged and the manager
        # keeps client=None (the first lookup then fails loudly, exactly
        # like production with a dead daemon) — no fallback client.
        import sys as _sys

        def _boom(url):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(_sys.modules[__name__], "_connect_v6d_with_retry",
                            _boom, raising=False)
        cls, _ = self._make_hooked_manager_cls()
        mgr = cls()
        mgr._v6d_url = "fake://0"
        mgr._v6d_backend = None
        mgr._async_connect()

        assert mgr.client is None

    def test_async_connect_success_marks_connected(self, monkeypatch):
        import sys as _sys

        sentinel_client = object()
        monkeypatch.setattr(_sys.modules[__name__], "_connect_v6d_with_retry",
                            lambda url: sentinel_client, raising=False)
        cls, _ = self._make_hooked_manager_cls()
        mgr = cls()
        mgr._v6d_url = "fake://0"

        class FakeBackend:
            marked = []

            def mark_manager_connected(self, group_id):
                self.marked.append(group_id)

        mgr._v6d_backend = FakeBackend()
        mgr._async_connect()

        assert mgr.client is sentinel_client
        assert mgr._v6d_backend.marked == [0]


class TestSchedulerRealLookupPath:
    """The scheduler hook must NOT override the real lookup entry points."""

    def test_get_num_new_matched_tokens_not_overridden(self):
        from sglang_simulator.simulation.vllm.v6d.v6d_manager import (
            C_V6dObjectConnectorSchedulerHook,
        )

        class FakeScheduler:
            def get_num_new_matched_tokens(self, request, n):
                raise NotImplementedError

            async def async_get_num_new_matched_tokens(self, request, n):
                raise NotImplementedError

            def request_finished(self, request, block_ids):
                return False, None

            def request_finished_all_groups(self, request, block_ids):
                return False, None

        orig_sync = FakeScheduler.get_num_new_matched_tokens
        orig_async = FakeScheduler.async_get_num_new_matched_tokens
        C_V6dObjectConnectorSchedulerHook.hook(FakeScheduler)

        assert FakeScheduler.get_num_new_matched_tokens is orig_sync
        assert FakeScheduler.async_get_num_new_matched_tokens is orig_async

    def test_request_finished_wrapped_for_cpu_noop_store(self):
        from sglang_simulator.simulation.vllm.v6d.v6d_manager import (
            C_V6dObjectConnectorSchedulerHook,
        )

        class FakeScheduler:
            def request_finished(self, request, block_ids):
                return False, None

            def request_finished_all_groups(self, request, block_ids):
                return False, None

        C_V6dObjectConnectorSchedulerHook.hook(FakeScheduler)
        sched = FakeScheduler()

        class _Req:
            request_id = "r1"

        # should_wait=False -> no store completion, passthrough.
        assert sched.request_finished(_Req(), []) == (False, None)
        assert sched.request_finished_all_groups(_Req(), []) == (False, None)


# ============================================================
# Test: sim fetch = seal-only (no data movement)
# ============================================================

class TestSimFetchSealOnly:
    """C_V6dObjectFetchHelperHook.start_fetch completes lazy placeholders
    via the real seal protocol (set_seal_target + complete -> client.seal)
    without touching BlockReceiver or blob data."""

    def _make_hooked_helper_cls(self):
        from sglang_simulator.simulation.vllm.v6d.v6d_manager import (
            C_V6dObjectFetchHelperHook,
        )

        class FakeFetchHelper:
            def start_fetch(self, objs):
                raise NotImplementedError

        C_V6dObjectFetchHelperHook.hook(FakeFetchHelper)
        return FakeFetchHelper

    @staticmethod
    def _make_objs(specs):
        """specs: list of (key, location); one shared lease + client."""

        class FakeLease:
            def __init__(self):
                self.seal_count = 0
                self.seal_target_count = 0
                self.completed_object_keys = []

        class FakeClient:
            def __init__(self):
                self.seal_calls = []

            def seal(self, lease_id, seal_object_keys=None):
                self.seal_calls.append((lease_id, list(seal_object_keys)))

        class FakeObj:
            def __init__(self, key, location, lease, client):
                self.key = key
                self.meta = {"location": location}
                self._lease = lease
                self._lease_id = "L1"
                self._client = client
                self.seal_target_set = None

            def set_seal_target(self, count):
                self.seal_target_set = count
                self._lease.seal_target_count = count

            def complete(self):
                # Mirror v6d Object.complete(): mark local, bump the shared
                # lease counter, and fire client.seal when the target is met.
                self.meta["location"] = "local"
                self._lease.seal_count += 1
                self._lease.completed_object_keys.append(self.key)
                if self._lease.seal_count == self._lease.seal_target_count:
                    self._client.seal(
                        self._lease_id,
                        seal_object_keys=self._lease.completed_object_keys)

        lease = FakeLease()
        client = FakeClient()
        objs = [FakeObj(k, loc, lease, client) for k, loc in specs]
        return objs, lease, client

    def test_lazy_placeholders_sealed(self):
        helper = self._make_hooked_helper_cls()()
        objs, lease, client = self._make_objs(
            [("k1", "10.0.0.2:9600"), ("k2", "sharedfs")])

        assert helper.start_fetch(objs) is None

        # Seal target narrowed to the fetched subset, then both completed
        # and the lease sealed once with both keys.
        assert objs[0].seal_target_set == 2
        assert client.seal_calls == [("L1", ["k1", "k2"])]
        assert all(o.meta["location"] == "local" for o in objs)

    def test_local_objs_untouched(self):
        helper = self._make_hooked_helper_cls()()
        objs, lease, client = self._make_objs([("k1", "local")])

        assert helper.start_fetch(objs) is None
        assert client.seal_calls == []
        assert objs[0].seal_target_set is None

    def test_empty_and_none(self):
        helper = self._make_hooked_helper_cls()()
        assert helper.start_fetch([]) is None
        assert helper.start_fetch(None) is None
        assert helper.start_fetch([None]) is None


# ============================================================
# Test: daemon [V6D HitSource] — port of production v6d_hitsource_patch
# ============================================================

class TestAcquireTieredReadHitSource:
    """One [V6D HitSource] line per _acquire_tiered_read batch, with the
    daemon's own record_*_hit classification feeding the bag."""

    @staticmethod
    def _cap_sim_logs(caplog, level):
        """Capture sglang_simulator records even when propagate=False."""
        import contextlib
        import logging

        @contextlib.contextmanager
        def _ctx():
            sim_logger = logging.getLogger("sglang_simulator")
            sim_logger.addHandler(caplog.handler)
            try:
                with caplog.at_level(level, logger="sglang_simulator"):
                    yield
            finally:
                sim_logger.removeHandler(caplog.handler)

        return _ctx()

    @staticmethod
    def _make_peer(plan):
        """Hooked fake peer whose read batch records hits per *plan*."""
        from sglang_simulator.simulation.vllm.v6d import v6d_capacity as cap

        class FakeStats:
            def record_local_vineyard_hit(self, size=0):
                pass

            def record_remote_vineyard_hit(self, size=0):
                pass

            def record_sharedfs_hit(self, size=0):
                pass

            def record_tair_kvcm_hit(self, size=0):
                pass

        cap.C_HitRateStatsHook.hook(FakeStats)

        class FakePeer:
            def __init__(self):
                self._hit_stats = FakeStats()

            async def _acquire_tiered_read(self, object_keys, term,
                                           peer=None, request_id=None,
                                           best_effort=False):
                stats = self._hit_stats
                for _ in range(plan.get("local", 0)):
                    stats.record_local_vineyard_hit(1)
                for _ in range(plan.get("p2p", 0)):
                    stats.record_remote_vineyard_hit(1)
                for _ in range(plan.get("sharedfs", 0)):
                    stats.record_sharedfs_hit(1)
                for _ in range(plan.get("tair_kvcm", 0)):
                    stats.record_tair_kvcm_hit(1)
                if plan.get("raise"):
                    raise RuntimeError("boom")
                return None, []

        cap.C_TieredVineyardPeerHook.hook(FakePeer)
        return FakePeer()

    @staticmethod
    def _hit_source_lines(caplog):
        return [r.getMessage() for r in caplog.records
                if "[V6D HitSource]" in r.getMessage()]

    def test_per_batch_source_split(self, caplog):
        import asyncio
        import logging

        peer = self._make_peer({"local": 2, "p2p": 1})
        with self._cap_sim_logs(caplog, logging.INFO):
            asyncio.run(peer._acquire_tiered_read(
                ["a", "b", "c", "d", "e"], None, request_id="r1"))
        lines = self._hit_source_lines(caplog)
        assert len(lines) == 1
        assert lines[0] == (
            "[V6D HitSource] request_id=r1 queried=5 local=2 p2p=1 "
            "sharedfs=0 tair_kvcm=0 miss=2")

    def test_error_suffix_and_reraise(self, caplog):
        import asyncio
        import logging

        peer = self._make_peer({"local": 1, "raise": True})
        with self._cap_sim_logs(caplog, logging.INFO):
            with pytest.raises(RuntimeError):
                asyncio.run(peer._acquire_tiered_read(
                    ["a", "b"], None, request_id="r2"))
        lines = self._hit_source_lines(caplog)
        assert len(lines) == 1
        assert "error=RuntimeError" in lines[0]
        assert "miss=1" in lines[0]

    def test_disabled_by_env(self, caplog, monkeypatch):
        import asyncio
        import logging

        monkeypatch.setenv("V6D_HITSOURCE_LOG", "0")
        peer = self._make_peer({"local": 1})
        with self._cap_sim_logs(caplog, logging.INFO):
            asyncio.run(peer._acquire_tiered_read(
                ["a"], None, request_id="r3"))
        assert self._hit_source_lines(caplog) == []

    def test_record_outside_read_not_counted(self, caplog):
        import logging

        peer = self._make_peer({})
        with self._cap_sim_logs(caplog, logging.INFO):
            # No bag active outside _acquire_tiered_read: no line, no error.
            peer._hit_stats.record_local_vineyard_hit(1)
        assert self._hit_source_lines(caplog) == []


# ============================================================
# Test: remote data-plane stubs (metadata only, no transfer)
# ============================================================

# ============================================================
# Test: daemon argv/options hooks (reserve_memory, sparse mmap)
# ============================================================


class TestVineyardPeerArgvHook:
    """C_VineyardPeerHook: swap reserve_memory=true->false, keep 2M
    alignment (real SRPC), and make init_srpc soft-fail."""

    @staticmethod
    def _fake_transfer_module(monkeypatch, **attrs):
        """Substitute v6d.common.transfer with a fake module.

        Must patch BOTH sys.modules and the parent package attribute:
        ``import v6d.common.transfer as t`` resolves via getattr on the
        parent package, which still points at the real module otherwise.
        """
        import sys
        import types

        import v6d.common

        fake_transfer = types.SimpleNamespace(**attrs)
        monkeypatch.setitem(sys.modules, "v6d.common.transfer", fake_transfer)
        monkeypatch.setattr(v6d.common, "transfer", fake_transfer,
                            raising=False)
        return fake_transfer

    @staticmethod
    def _make_fake_peer(monkeypatch, init_srpc):
        from sglang_simulator.simulation.vllm.v6d import v6d_capacity as cap

        fake_transfer = TestVineyardPeerArgvHook._fake_transfer_module(
            monkeypatch, init_srpc_transfer=init_srpc)

        class FakePeer:
            def __init__(self, argc=0, argv=None, *args, **kwargs):
                self.argv = argv

        cap.C_VineyardPeerHook.hook(FakePeer)
        return FakePeer, fake_transfer

    def test_reserve_memory_swapped_alignment_untouched(self, monkeypatch):
        calls = []
        FakePeer, _ = self._make_fake_peer(
            monkeypatch, lambda *a, **k: calls.append((a, k)))

        peer = FakePeer(3, ["v6d", "--reserve_memory=true",
                            "-2M_alignment=true"])

        assert "--reserve_memory=false" in peer.argv
        assert "--reserve_memory=true" not in peer.argv
        assert "-2M_alignment=true" in peer.argv  # 2M alignment kept: real SRPC

    def test_srpc_init_soft_fail(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("MEM_ALIGN")

        FakePeer, fake_transfer = self._make_fake_peer(monkeypatch, _boom)
        FakePeer(1, ["v6d"])

        # init_srpc_transfer was replaced by the soft-fail wrapper: calling
        # it logs and returns None instead of raising.
        assert fake_transfer.init_srpc_transfer(1, 2) is None

    def test_srpc_init_success_passthrough(self, monkeypatch):
        calls = []
        FakePeer, fake_transfer = self._make_fake_peer(
            monkeypatch, lambda addr, size: calls.append((addr, size)) or "ok")
        FakePeer(1, ["v6d"])

        assert fake_transfer.init_srpc_transfer(123, 4096) == "ok"
        assert calls == [(123, 4096)]


class TestV6dMmapManagerHook:
    """C_V6dMmapManagerHook: skip the populating C++ init_mmap on client
    connect, keep the real handshake socket."""

    def test_create_mmap_skips_population(self, monkeypatch):
        import os

        from sglang_simulator.simulation.vllm.v6d import v6d_manager as mgr

        # Import the real module chain BEFORE faking v6d.common.transfer —
        # v6d.client.peers.vineyard.__init__ pulls blob_view, which imports
        # transfer.load_data and would choke on the fake module.
        import v6d.client.peers.vineyard.mmap_manager  # noqa: F401

        r_fd, w_fd = os.pipe()
        connect_calls = []

        def _fake_connect(socket_path):
            connect_calls.append(socket_path)
            return r_fd, 500 << 30, 0, object()  # real fd so os.close works

        TestVineyardPeerArgvHook._fake_transfer_module(
            monkeypatch, _vineyard_connect=_fake_connect)

        class FakeMgr:
            pass

        mgr.C_V6dMmapManagerHook.hook(FakeMgr)
        info = FakeMgr._create_mmap(FakeMgr(), "/tmp/x.sock", True)
        os.close(w_fd)

        # Handshake happened, but no mmap / no init_srpc, zero base.
        assert connect_calls == ["/tmp/x.sock"]
        assert info.base_addr == 0
        assert info.map_size == 500 << 30
        assert info.fd == -1


# ============================================================
# Test: remote data-plane stubs (payload copies only; meta probe is real)
# ============================================================


class TestTransferStubs:

    def test_load_data_noop(self):
        from sglang_simulator.simulation.vllm.v6d.v6d_capacity import (
            _sim_async_load_data_noop,
            _sim_load_data_noop,
        )
        assert _sim_load_data_noop([[1]], [[2]], [[0]], [[4]], "x:1") is None
        assert _sim_async_load_data_noop([[1]], [[2]], [[0]], [[4]], "x:1") == []

    def test_meta_probe_not_stubbed(self, monkeypatch):
        """The tiered hook stubs only the payload copies; the SRPC meta probe
        must stay real."""
        from sglang_simulator.simulation.vllm.v6d import v6d_capacity as cap

        sentinel = object()
        fake_transfer = TestVineyardPeerArgvHook._fake_transfer_module(
            monkeypatch,
            get_metas_by_names=sentinel,
            load_data=lambda *a, **k: "orig",
            async_load_data=lambda *a, **k: "orig",
        )

        class FakeTiered:
            async def _acquire_tiered_read(self, *args, **kwargs):
                return None, []

        cap.C_TieredVineyardPeerHook.hook(FakeTiered)

        assert fake_transfer.get_metas_by_names is sentinel  # real SRPC probe
        assert fake_transfer.load_data is cap._sim_load_data_noop
        assert fake_transfer.async_load_data is cap._sim_async_load_data_noop


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
