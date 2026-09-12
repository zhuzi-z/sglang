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

import asyncio
import sys
import threading
import time
from collections import Counter
from concurrent.futures import Future
from dataclasses import dataclass

from sglang_simulator.simulation.manager import Envs, StateManager
from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.vllm.cpu_stubs import DummyEvent
from sglang_simulator.simulation.vllm.v6d.bandwidth import BandwidthModel
from sglang_simulator.utils import get_logger

logger = get_logger()

class SimulatedKVController:
    """One owner for KV progress, protection, and completion visibility.

    Like the SGLang HiCacheController hook, native connector polling drives this
    controller synchronously. Metadata generated before a forward is staged;
    the next connector step observes the executor's updated clock and starts
    those copies. Nothing is called back into the executor or worker.
    """

    @dataclass
    class Store:
        request: object
        blocks: list
        num_blocks: int
        last: bool
        ready_at: float = 0.0

    def __init__(self, hybrid):
        self.hybrid = hybrid
        self.backend = getattr(hybrid._backend, "_v6d", hybrid._backend)
        self.cache = self.backend._scheduler
        self.backend._sim_controller = self
        self._staged = []
        self._stores = []
        self._loads = {}
        self._store_tails = {}
        self._aborted_stores = {}
        self._preparing = []
        self._lock = threading.RLock()
        # Native abort cleanup runs on the connector loop. Its remaining pins
        # must not race with our transfer of ownership into a staged copy.
        original_release = self.cache._release_protected_blocks

        def release(req_id):
            with self._lock:
                return original_release(req_id)

        self.cache._release_protected_blocks = release

    @staticmethod
    def now():
        return (StateManager.get_global_clock() if Envs.simulation_mode() == "OFFLINE"
                else time.perf_counter())

    @staticmethod
    def _metadata_objects(metadata):
        seen, stack = set(), [metadata]
        while stack:
            obj = stack.pop()
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            yield obj
            stack.extend(getattr(obj, attr, None) for attr in ("reqs", "inner", "v6d"))

    def capture_stores(self, metadata):
        """Take this chunk's pins, but do not spend future compute time."""
        stores = {}
        for obj in self._metadata_objects(metadata):
            stores.update(getattr(obj, "reqs_to_store", None) or {})
            for rid in getattr(obj, "aborted_save_ids", ()):
                state = self.hybrid._saving.get(rid)
                if state is not None:
                    self._aborted_stores[rid] = state._req
        for rid, (groups, last) in stores.items():
            state = self.hybrid._saving.get(rid)
            if state is None:
                raise RuntimeError(f"Store request {rid} is missing")
            wanted = Counter(bid for gid, (_keys, ids) in groups.items()
                             if gid in self.cache.mamba_group_ids for bid in ids)
            with self._lock:
                taken, remaining = [], []
                for block in self.cache._swap_protected_blocks.get(rid, ()):
                    if wanted[block.block_id]:
                        wanted[block.block_id] -= 1
                        taken.append(block)
                    else:
                        remaining.append(block)
                if remaining:
                    self.cache._swap_protected_blocks[rid] = remaining
                else:
                    self.cache._swap_protected_blocks.pop(rid, None)
            self._staged.append(self.Store(
                state._req, taken, sum(len(keys) for keys, _ids in groups.values()), last))

    def prepare(self, coroutine):
        future = Future()
        self._preparing.append(future)

        async def tracked():
            try:
                result = await coroutine
            except BaseException as exc:
                future.set_exception(exc)
                raise
            else:
                future.set_result(result)
                return result
        return tracked()

    def settle_preparations(self):
        preparing, self._preparing = self._preparing, []
        for future in preparing:
            future.result(timeout=30.0)

    def queue_load(self, request, num_tokens, groups):
        nblocks = sum(len(keys) for keys, _ids in groups.values())
        bw = BandwidthModel.get()
        deadline = self.now() + bw.latency_for(nblocks, True) + bw.seg1_latency(nblocks)
        with self._lock:
            self._loads.setdefault(request.request_id, (deadline, num_tokens))

    def _run_control(self, coroutine):
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if current is self.hybrid.loop:
            coroutine.close()
            raise RuntimeError("Cannot synchronously wait on the current connector loop")
        return asyncio.run_coroutine_threadsafe(coroutine, self.hybrid.loop).result(timeout=30.0)

    def _load_done(self, rid, num_tokens):
        from vllm.v1.hybrid_connector import IoRet

        async def complete():
            for rank in range(self.hybrid._tp_size()):
                await self.hybrid._do_load_done(rank, IoRet(reqid=rid, n=num_tokens))
        self._run_control(complete())

    def _store_done(self, req, failed=False):
        from vllm.v1.hybrid_connector import HB_SAVE_SOURCES, IoRet, get_param

        sources = tuple(get_param(req, HB_SAVE_SOURCES, ()) or ())
        if not sources:
            raise RuntimeError(f"Missing save sources for {req.request_id}")

        async def complete():
            for source in sources:
                for rank in range(self.hybrid._tp_size()):
                    await self.hybrid._do_save_done(
                        rank, IoRet(reqid=req.request_id, source=source,
                                    n=0 if failed else None))
        self._run_control(complete())

    def progress(self):
        """Settle work through this clock; future readiness stays private."""
        now = self.now()
        bw = BandwidthModel.get()
        staged, self._staged = self._staged, []
        for store in staged:
            rid = store.request.request_id
            store.ready_at = max(now, self._store_tails.get(rid, now))
            store.ready_at += bw.store_completion_latency(store.num_blocks)
            self._store_tails[rid] = store.ready_at
            self._stores.append(store)
        due = [store for store in self._stores if store.ready_at <= now]
        self._stores = [store for store in self._stores if store.ready_at > now]
        for store in due:
            if store.blocks:
                with self._lock:
                    self.cache._block_pool.free_blocks(store.blocks)
            rid = store.request.request_id
            if self._store_tails.get(rid) == store.ready_at:
                self._store_tails.pop(rid, None)
            if store.last and rid not in self._aborted_stores:
                self._store_done(store.request)
        aborted = [rid for rid in self._aborted_stores if rid not in self._store_tails]
        for rid in aborted:
            self._store_done(self._aborted_stores.pop(rid), failed=True)
        with self._lock:
            loads = [(rid, n) for rid, (deadline, n) in self._loads.items() if deadline <= now]
            for rid, _n in loads:
                del self._loads[rid]
        for rid, n in loads:
            self._load_done(rid, n)
        return len(due) + len(loads) + len(aborted)

    def next_wakeup(self):
        if self._staged or any(rid not in self._store_tails for rid in self._aborted_stores):
            return self.now()
        with self._lock:
            deadlines = [deadline for deadline, _n in self._loads.values()]
        deadlines.extend(store.ready_at for store in self._stores)
        return min(deadlines, default=None)

    def has_pending(self):
        return bool(self._preparing) or self.next_wakeup() is not None

    def reset(self):
        """Settle controller-owned state at the native cache-reset boundary.

        A round/profile reset starts a fresh clock epoch, so pending modeled
        copies are released immediately (no GPU DMA exists to wait for) and
        unfinished saves are failed through the native source/rank
        acknowledgement chain — unsealed objects are discarded and block
        references freed. No virtual time is advanced and nothing sleeps.
        Refused while requests are still active.
        """
        if (self.hybrid.pending_requests or self.hybrid._loaded
                or any(not state._req.is_finished()
                       for state in self.hybrid._saving.values())):
            logger.warning("[V6D Hijack] controller reset refused: requests active")
            return False
        self.settle_preparations()
        with self._lock:
            dropped_loads = len(self._loads)
            self._loads.clear()
            for store in self._staged + self._stores:
                if store.blocks:
                    self.cache._block_pool.free_blocks(store.blocks)
            self._staged.clear()
            self._stores.clear()
            self._store_tails.clear()
            self._aborted_stores.clear()
        if dropped_loads:
            logger.warning("[V6D Hijack] controller reset dropped %d pending load(s)",
                           dropped_loads)
        saved = set(self.hybrid._saved)
        for rid, state in list(self.hybrid._saving.items()):
            if rid not in saved:
                self._store_done(state._req, failed=True)
        self.hybrid._step_saved()
        return not self.has_pending()


class C_HybridControlPlaneHook(BaseHook):
    """Join native V6D metadata preparation inside the connector, not inference.

    HybridScheduler is the connector's internal controller. Its native waiting
    loop still performs admission/allocation; only already-submitted lookup and
    metadata preparation are synchronized before its normal loaded-queue pass.
    """

    HOOK_CLASS_NAME = "HybridScheduler"
    HOOK_MODULE_NAME = "vllm.v1.hybrid_connector"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_prepare = target._on_add_req
        original_waiting = target._step_waiting
        original_step = target.step

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            backend = getattr(self._backend, "_v6d", self._backend)
            self._sim_controller = (SimulatedKVController(self)
                                    if getattr(backend, "_sim_sync_prepare", False) else None)

        def prepare(self, req, blocks):
            coroutine = original_prepare(self, req, blocks)
            controller = self._sim_controller
            return controller.prepare(coroutine) if controller is not None else coroutine

        def step_waiting(self):
            result = original_waiting(self)
            if self._sim_controller is not None:
                self._sim_controller.settle_preparations()
                self._sim_controller.progress()
            return result

        def step(self):
            if self._sim_controller is not None:
                self._sim_controller.progress()
            return original_step(self)

        target.__init__ = init
        target._on_add_req = prepare
        target._step_waiting = step_waiting
        target.step = step


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
    """Own modeled transfers while preserving native completion semantics."""

    HOOK_CLASS_NAME = "HybridConnector"
    HOOK_MODULE_NAME = "vllm.v1.hybrid_connector"

    @classmethod
    def hook(cls, target):
        original_has_requests = getattr(target, "has_requests", None)
        original_build_meta = getattr(target, "build_connector_meta", None)

        def build_connector_meta(self, scheduler_output):
            metadata = original_build_meta(self, scheduler_output)
            controller = self._sched._sim_controller
            if controller is not None:
                controller.capture_stores(metadata)
            return metadata

        if original_build_meta is not None:
            target.build_connector_meta = build_connector_meta

        def has_requests(self):
            controller = self._sched._sim_controller
            return original_has_requests(self) or (
                controller is not None and controller.has_pending())

        if original_has_requests is not None:
            target.has_requests = has_requests

        def override_wait_for_save(self):
            return None

        def override_get_finished(self, finished_req_ids):
            # Hybrid load/save completion uses native acknowledgement handlers,
            # not the base connector's finished_recving/finished_sending path.
            return set(), set()

        def override_clear_connector_metadata(self):
            if getattr(self, "_worker", None) is not None:
                self._worker.clear_connector_metadata()
            return None

        def override_start_load_kv(self, forward_context=None, **kwargs):
            # Preserve worker lifecycle dispatch. The backend's no-op async
            # generator leaves acknowledgement to the modeled load event.
            assert self._worker is not None
            self._worker.start_load_kv()

        target.wait_for_save = override_wait_for_save
        target.get_finished = override_get_finished
        target.clear_connector_metadata = override_clear_connector_metadata
        target.start_load_kv = override_start_load_kv

        # Request finish/seal semantics remain native. The simulation-specific
        # correction is GPU snapshot ownership: completed chunk copies no longer
        # pin Mamba blocks until the entire request's last save. The installed
        # native hybrid backend retains those pins until async_cleanup, which
        # can exhaust the pool during long prefills even in BLOCKING mode.
        # Events own detached references, so native final/abort cleanup cannot
        # free them twice or release a newer, still-in-flight chunk.

        def override_reset_cache(self):
            # Settle the simulated-transfer controller first (release
            # copy-owned block pins, fail unfinished saves), then forward to
            # the v6d managers.  Upstream HybridConnector lacks reset_cache
            # forwarding (falls back to the KVConnectorBase_V1 no-op), so the
            # official /reset_prefix_cache?reset_external=true path never
            # reaches the v6d managers.  Bridge it: walk
            # _sched._backend[._v6d]._scheduler to the
            # V6dObjectConnectorScheduler and reuse its reset_cache()
            # (which resets every V6dObjectManager).
            controller = getattr(self._sched, "_sim_controller", None)
            if controller is not None and not controller.reset():
                logger.warning(
                    "[V6D Hijack] HybridConnector.reset_cache: controller "
                    "still has active requests; caches NOT reset")
                return False
            backend = getattr(getattr(self, "_sched", None), "_backend", None)
            for candidate in (backend, getattr(backend, "_v6d", None)):
                scheduler = getattr(candidate, "_scheduler", None)
                if scheduler is not None and hasattr(scheduler, "reset_cache"):
                    result = scheduler.reset_cache()
                    logger.info(
                        "[V6D Hijack] HybridConnector.reset_cache: forwarded "
                        "to %s -> %s", type(scheduler).__name__, result)
                    return result
            logger.warning(
                "[V6D Hijack] HybridConnector.reset_cache: no backend "
                "scheduler with reset_cache found (backend=%s); connector "
                "caches NOT reset", type(backend).__name__ if backend else None)
            return False

        target.reset_cache = override_reset_cache
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

        original_prepare = getattr(target, "async_update_state_after_alloc", None)

        async def prepare_load(self, request, blocks, num_external_tokens):
            # Complete real metadata lookup/allocation first. A modeled DMA is
            # only scheduled here, never awaited by the preparation barrier.
            result = await original_prepare(self, request, blocks, num_external_tokens)
            if result is not None or num_external_tokens <= 0:
                return result
            groups = self._scheduler._reqs_to_load.get(request.request_id, {})
            self._sim_controller.queue_load(request, num_external_tokens, groups)
            return result

        if original_prepare is not None:
            target.async_update_state_after_alloc = prepare_load
            target._sim_sync_prepare = True

        async def override_async_load_kv(self, m):
            # Preparation schedules the native load acknowledgement at its
            # modeled ready time. Yielding here would send an early/duplicate
            # LOAD_DONE RPC and let the native scheduler run the request early.
            return
            yield  # Preserve the native async-generator interface.

        target.async_load_kv = override_async_load_kv

        def clear_backend_metadata(self):
            # The controller captured stores before forward and owns completion.
            self._bound_meta = None

        def bypass_bind(self, metadata):
            # Aborts are captured on the scheduler side, including idle substeps.
            # No worker timer may publish completion ahead of the modeled copy.
            return None

        target.clear_backend_metadata = clear_backend_metadata
        target.bypass_bind = bypass_bind

        _bwm = BandwidthModel.get()
        logger.info("[V6D Hijack] V6dObjectBackend hook installed "
                    "(transfer-latency model enabled=%s)", _bwm.enabled)


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
