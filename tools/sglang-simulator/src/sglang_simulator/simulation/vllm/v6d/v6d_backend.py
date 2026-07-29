"""
Runtime hooks for native V6D/KVT control-plane simulation on CPU.

These hooks are active only when native V6D control-plane mode is enabled.
They keep the real vLLM V6D/KVT control-plane classes in use while replacing
CUDA-only synchronization and transport pieces with no-op CPU shims:

- V6dObjectBackend keeps scheduling/save/load decisions but uses DummyEvent.
- PBackend keeps operation selection but bypasses blade_kvt GPU transport.

No DashServing/vLLM source file is modified on disk; all changes are installed
through class monkey-patches at interpreter startup.
"""

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.vllm.cpu_stubs import DummyEvent
from sglang_simulator.utils import get_logger

logger = get_logger()


class C_HybridBackendHook(BaseHook):
    """Bypass GPU-oriented HybridBackend group layout validation on CPU."""

    HOOK_CLASS_NAME = "HybridBackend"
    HOOK_MODULE_NAME = "vllm.v1.hybrid_connector"

    @classmethod
    def hook(cls, target):
        def override_validate_group_ordering(self):
            logger.info(
                "[V6D Hijack] HybridBackend._validate_group_ordering: "
                "skipped CPU native control-plane validation"
            )
            return None

        target._validate_group_ordering = override_validate_group_ordering
        logger.info("[V6D Hijack] HybridBackend hook installed")


class C_HybridConnectorHook(BaseHook):
    """Report HybridConnector CPU no-op save/load completions."""

    HOOK_CLASS_NAME = "HybridConnector"
    HOOK_MODULE_NAME = "vllm.v1.hybrid_connector"

    @classmethod
    def hook(cls, target):
        original_bind = target.bind_connector_metadata
        original_clear = target.clear_connector_metadata

        def _collect_req_ids(obj, attr_name, seen=None):
            if obj is None:
                return set()
            if seen is None:
                seen = set()
            obj_id = id(obj)
            if obj_id in seen:
                return set()
            seen.add(obj_id)
            if isinstance(obj, dict):
                ids = set()
                for value in obj.values():
                    ids.update(_collect_req_ids(value, attr_name, seen))
                return ids
            if isinstance(obj, (list, tuple, set)):
                ids = set()
                for value in obj:
                    ids.update(_collect_req_ids(value, attr_name, seen))
                return ids
            value = getattr(obj, attr_name, None)
            if isinstance(value, dict):
                return set(value)
            ids = set()
            for value in vars(obj).values() if hasattr(obj, "__dict__") else ():
                ids.update(_collect_req_ids(value, attr_name, seen))
            return ids

        def override_bind_connector_metadata(self, metadata):
            original_bind(self, metadata)
            backend_meta = getattr(metadata, "reqs", None)
            reqs_to_store = getattr(metadata, "reqs_to_store", None)
            if reqs_to_store is None and backend_meta is not None:
                reqs_to_store = getattr(backend_meta, "reqs_to_store", None)
            if reqs_to_store is None and backend_meta is not None:
                inner_meta = getattr(backend_meta, "inner", None)
                if inner_meta is not None:
                    reqs_to_store = getattr(inner_meta, "reqs_to_store", None)
            reqs_to_store = reqs_to_store or {}
            noop_store_reqs = {
                req_id
                for req_id, (groups_data, _is_last_save) in reqs_to_store.items()
                if not groups_data
            }
            if noop_store_reqs:
                try:
                    # sched_get_req looks up the live Scheduler's requests
                    # dict (engine_proxy); HybridConnector itself has no
                    # get_request() — same pattern as v6d_manager.py.
                    from vllm.v1.hybrid_connector import (
                        mark_backend_save_done,
                        sched_get_req,
                    )
                    for req_id in sorted(noop_store_reqs):
                        req = sched_get_req(req_id)
                        if req is not None:
                            mark_backend_save_done(req)
                    logger.info(
                        "[V6D Hijack] completed noop last_save via mark_backend_save_done: %s",
                        sorted(noop_store_reqs),
                    )
                except Exception:
                    logger.exception(
                        "[V6D Hijack] failed to complete noop last_save: %s",
                        sorted(noop_store_reqs),
                    )
            store_reqs = {
                req_id
                for req_id, (groups_data, _is_last_save) in reqs_to_store.items()
                if groups_data
            }
            load_reqs = _collect_req_ids(metadata, "reqs_to_load")
            self._sim_finished_store_reqs = (
                getattr(self, "_sim_finished_store_reqs", set()) | store_reqs
            )
            self._sim_finished_load_reqs = (
                getattr(self, "_sim_finished_load_reqs", set()) | load_reqs
            )
            if store_reqs or load_reqs:
                logger.info(
                    "[V6D Hijack] HybridConnector bind metadata: "
                    "store=%s load=%s",
                    sorted(store_reqs),
                    sorted(load_reqs),
                )

        def override_wait_for_save(self):
            return None

        def override_get_finished(self, finished_req_ids):
            store_reqs = set(getattr(self, "_sim_finished_store_reqs", set()))
            load_reqs = set(getattr(self, "_sim_finished_load_reqs", set()))
            self._sim_finished_store_reqs = set()
            self._sim_finished_load_reqs = set()
            if store_reqs or load_reqs:
                logger.info(
                    "[V6D Hijack] get_finished: store=%s load=%s "
                    "finished_req_ids=%s",
                    sorted(store_reqs),
                    sorted(load_reqs),
                    sorted(finished_req_ids or []),
                )
            return store_reqs, load_reqs

        def override_clear_connector_metadata(self):
            logger.debug(
                "[V6D Hijack] HybridConnector.clear_connector_metadata: "
                "skipped CUDA backend clear in CPU mode"
            )
            if getattr(self, "_worker", None) is not None:
                setattr(self._worker, "_meta", None)
            return None

        def override_start_load_kv(self, forward_context=None, **kwargs):
            # Do not no-op: let HybridWorker.start_load_kv schedule _async_load_kv.
            # V6dObjectBackend.async_load_kv is overridden to yield fake IoRet
            # immediately (simulating instant load completion in CPU mode).
            assert self._worker is not None
            self._worker.start_load_kv()

        target.bind_connector_metadata = override_bind_connector_metadata
        target.wait_for_save = override_wait_for_save
        target.get_finished = override_get_finished
        target.clear_connector_metadata = override_clear_connector_metadata
        target.start_load_kv = override_start_load_kv

        def override_update_connector_output(self, connector_output):
            # Delegate to scheduler-side backend (V6dObjectConnectorScheduler)
            sched = getattr(self, "_sched", None)
            if sched is not None:
                backend = getattr(sched, "_backend", None)
                scheduler = getattr(backend, "_scheduler", None) if backend is not None else None
                if scheduler is not None and hasattr(scheduler, "update_connector_output"):
                    scheduler.update_connector_output(connector_output)
                    logger.info("[V6D Hijack] update_connector_output delegated to backend: finished_sending=%s",
                               getattr(connector_output, "finished_sending", None))
        target.update_connector_output = override_update_connector_output
        logger.info("[V6D Hijack] HybridConnector hook installed")


class C_V6dObjectBackendHook(BaseHook):
    """Hook V6dObjectBackend to replace CUDA Event pool.

    The V6dObjectBackend coordinates save/load operations and uses
    CUDA Events to track when GPU operations complete. In CPU simulation,
    all events are immediately complete (DummyEvent.query() → True).
    """

    HOOK_CLASS_NAME = "V6dObjectBackend"
    HOOK_MODULE_NAME = "vllm.v1.hybrid_connector.v6d_object_backend"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def override_init(self, *args, **kwargs):
            """Call original init, then replace CUDA Event pool."""
            original_init(self, *args, **kwargs)
            # Replace Event pools with DummyEvent lists
            if hasattr(self, "_save_event_pool"):
                self._save_event_pool = [DummyEvent() for _ in range(8)]
            if hasattr(self, "_load_event_pool"):
                self._load_event_pool = [DummyEvent() for _ in range(8)]
            if getattr(self, "_scheduler", None) is not None and hasattr(self, "_v6d_ready"):
                self._v6d_ready = True
                logger.info(
                    "[V6D Hijack] V6dObjectBackend scheduler marked ready "
                    "for CPU native control-plane"
                )
            logger.info(
                "[V6D Hijack] V6dObjectBackend.__init__: "
                "replaced CUDA Event pools with DummyEvent"
            )

        target.__init__ = override_init

        # Patch _get_event / _new_event if they exist
        def override_new_event(self):
            """Return DummyEvent instead of torch.cuda.Event."""
            return DummyEvent()

        if hasattr(target, "_new_event"):
            target._new_event = override_new_event

        # Override _record_event if exists
        def override_record_event(self, event=None):
            """No-op: DummyEvent.record() does nothing."""
            if event is None:
                event = DummyEvent()
            event.record()
            return event

        if hasattr(target, "_record_event"):
            target._record_event = override_record_event

        # Override async_load_kv to simulate instant load completion.
        # In CPU simulation, no actual KV data transfer is needed.
        # The original async_load_kv calls self._worker.async_start_load_kv(meta)
        # which requires real V6D data-plane operations (CUDA events, DMA, etc.).
        # We bypass that and directly yield IoRet for each reqs_to_load entry.
        original_async_load_kv = target.async_load_kv

        async def override_async_load_kv(self, m):
            from vllm.v1.hybrid_connector import IoRet
            meta = m.inner
            if not meta or not getattr(meta, "reqs_to_load", None):
                return
            for req_id in meta.reqs_to_load:
                n = (m.external_tokens or {}).get(req_id, 0)
                logger.info(
                    "[V6D Hijack] async_load_kv: simulating instant load "
                    "completion for req=%s n=%d", req_id, n)
                yield IoRet(reqid=req_id, n=n)

        target.async_load_kv = override_async_load_kv

        logger.info("[V6D Hijack] V6dObjectBackend hook installed")


class _DummyBladeKVTModule:
    """Minimal blade_kvt shim used before the real CUDA extension is loaded."""

    @staticmethod
    def set_envs(*args, **kwargs):
        logger.info("[KVT Hijack] skip blade_kvt.set_envs")
        return None

    @staticmethod
    def is_nv_gpu(*args, **kwargs):
        return True


class _DummyKVTClient:
    """No-op blade_kvt client for CPU native-control-plane simulation."""

    def record_event(self, *args, **kwargs):
        return None

    def start_req_send(self, *args, **kwargs):
        return None

    def start_send_substep(self, *args, **kwargs):
        return None

    def submit_delta_send(self, *args, **kwargs):
        return None

    def submit_req_send2(self, *args, **kwargs):
        return None

    def start_send_step(self, *args, **kwargs):
        return None

    def flush_send_step(self, *args, **kwargs):
        return None

    async def send_error_done_req(self, *args, **kwargs):
        return None


class C_KVTPBackendHook(BaseHook):
    """Hook KVT PBackend worker-side data transport for CPU simulation."""

    HOOK_CLASS_NAME = "PBackend"
    HOOK_MODULE_NAME = "vllm.v1.hybrid_connector.kvtbackend"

    @classmethod
    def hook(cls, target):
        import sys

        module = sys.modules.get(target.__module__)
        if module is not None:
            def _noop_generate_nic_affinity(*args, **kwargs):
                logger.info("[KVT Hijack] skip generate_nic_affinity")
                return None

            module.generate_nic_affinity = _noop_generate_nic_affinity
            module.blade_kvt = _DummyBladeKVTModule

        original_init = target.__init__

        def override_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if getattr(self, "_bladkv_cli", None) is None:
                self._bladkv_cli = _DummyKVTClient()
            logger.info("[KVT Hijack] PBackend initialized with dummy transport")

        def override_register_kv_caches(self, kv_caches):
            self._bladkv_cli = _DummyKVTClient()
            logger.info(
                "[KVT Hijack] PBackend.register_kv_caches skipped "
                "blade_kvt client creation; layers=%d",
                len(kv_caches),
            )
            return None

        def override_async_save_kv_layer(self, *args, **kwargs):
            return None

        def override_bind_backend_metadata(self, *args, **kwargs):
            return None

        def override_bypass_bind(self, *args, **kwargs):
            return None

        def override_clear_backend_metadata(self, *args, **kwargs):
            return None

        def override_bypass_clear(self, *args, **kwargs):
            return None

        target.__init__ = override_init
        target.register_kv_caches = override_register_kv_caches
        target.async_save_kv_layer = override_async_save_kv_layer
        target.bind_backend_metadata = override_bind_backend_metadata
        target.bypass_bind = override_bypass_bind
        target.clear_backend_metadata = override_clear_backend_metadata
        target.bypass_clear = override_bypass_clear

        logger.info("[KVT Hijack] PBackend hook installed")
