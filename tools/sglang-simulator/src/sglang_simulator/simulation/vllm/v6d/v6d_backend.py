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
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import Future
from dataclasses import dataclass

from sglang_simulator.simulation.manager import Envs, StateManager
from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.vllm.cpu_stubs import DummyEvent
from sglang_simulator.simulation.vllm.v6d.bandwidth import BandwidthModel, validate_load_timing
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
        keys: list
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
        self._native_locality = {}
        self._preparing_requests = Counter()
        self._cancelled_preparations = set()
        self._store_tails = {}
        self._aborted_stores = {}
        self._local_keys: set = set()
        # 原生命中快照为主；本机存储 registry 仅用于缺失信息兜底。
        self._seg1_scope = os.environ.get("SIMULATOR_SEG1_SCOPE",
                                          "local_registry")
        self._preparing = []
        self._lock = threading.RLock()
        # Native abort cleanup runs on the connector loop. Its remaining pins
        # must not race with our transfer of ownership into a staged copy.
        original_release = self.cache._release_protected_blocks

        def release(req_id):
            with self._lock:
                return original_release(req_id)

        self.cache._release_protected_blocks = release
        self._install_locality_hooks()


    def _capture_native_key(self, rid, gid, key, obj, cached=False):
        # 只读复制 fetch 前的原生位置/对象字节数，不保留对象或租约。#
        if rid is None or obj is None:
            return
        meta = getattr(obj, "meta", {}) or {}
        location = meta.get("location")
        note = meta.get("_sim_native_read_origin")
        # 完整 request_id（含原生 -0 后缀）严格匹配，历史标签不跨请求。
        matched_note = (isinstance(note, dict) and note.get("version") == 1
                        and note.get("request_id") == rid
                        and note.get("key") == key
                        and isinstance(note.get("peer"), str) and bool(note["peer"]))
        if matched_note:
            location = note["peer"]
        scope = getattr(getattr(obj, "_lease", None), "scope", None)
        scope = getattr(scope, "value", scope)
        # READ 缓存租约是原生驻留证据；历史 remote 属性不能推翻它。
        if not matched_note and cached and str(scope).lower() == "read":
            location = "local"
        if not isinstance(location, str) or not location:
            return
        try:
            size = int(note.get("size", 0) if matched_note else meta.get("size", 0))
        except (ValueError, TypeError, OverflowError):
            size = 0
        with self._lock:
            if rid in self._cancelled_preparations:
                return
            self._native_locality.setdefault(rid, {}).setdefault(
                (gid, key), (location == "local", size if size > 0 else None))

    def _forget_native_request(self, rid):
        with self._lock:
            if self._preparing_requests.get(rid):
                self._cancelled_preparations.add(rid)
            self._native_locality.pop(rid, None)
            self._loads.pop(rid, None)

    def _install_locality_hooks(self):
        cache = self.cache
        def capture_fetch(request, hashes, unfetched):
            objs, cached, promoted = cache._prepare_fetch(hashes, unfetched)
            by_key = {o.key: o for o in objs if o is not None}
            for manager, key in cached + promoted:
                self._capture_native_key(request.request_id, manager._group_id,
                                         key, by_key.get(key), (manager, key) in cached)
        def wrap_fetch(original, asynchronous):
            if asynchronous:
                async def fetch(request, hit_blocks_hashes, unfetched_objs=None):
                    capture_fetch(request, hit_blocks_hashes, unfetched_objs)
                    result = await original(request, hit_blocks_hashes, unfetched_objs)
                    if not result:
                        self._forget_native_request(request.request_id)
                    return result
            else:
                def fetch(request, hit_blocks_hashes, unfetched_objs=None):
                    capture_fetch(request, hit_blocks_hashes, unfetched_objs)
                    result = original(request, hit_blocks_hashes, unfetched_objs)
                    if not result:
                        self._forget_native_request(request.request_id)
                    return result
            return fetch

        for name, asynchronous in (("_fetch_intersection_blocks", False),
                                   ("_async_fetch_intersection_blocks", True)):
            original = getattr(cache, name, None)
            if original is not None:
                setattr(cache, name, wrap_fetch(original, asynchronous))
        def wrap_key(manager, original, asynchronous):
            def capture(key, rid):
                if key is not None:
                    self._capture_native_key(rid, manager._group_id, key,
                                             manager._cached_objs.get(key), True)
            if asynchronous:
                async def get_key(block_hash, request_id=None):
                    key = await original(block_hash, request_id=request_id)
                    capture(key, request_id)
                    return key
            else:
                def get_key(block_hash, request_id=None):
                    key = original(block_hash, request_id=request_id)
                    capture(key, request_id)
                    return key
            return get_key

        def wrap_cleanup(original):
            def cleanup(*args, **kwargs):
                rid = (args[0] if args else kwargs.get("reqid",
                       kwargs.get("req_id", kwargs.get("request_id"))))
                try:
                    return original(*args, **kwargs)
                finally:
                    self._forget_native_request(rid)
            return cleanup

        for manager in getattr(cache, "managers", {}).values():
            for name, asynchronous in (("get_key", False), ("async_get_key", True)):
                original = getattr(manager, name, None)
                if original is not None:
                    setattr(manager, name, wrap_key(manager, original, asynchronous))
            original = getattr(manager, "cleanup_request", None)
            if original is not None:
                manager.cleanup_request = wrap_cleanup(original)
        for name in ("on_abort_req", "_cleanup_req_tracking"):
            original = getattr(cache, name, None)
            if original is not None:
                setattr(cache, name, wrap_cleanup(original))

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
        # reqs_to_store value arity depends on the engine baseline:
        #   nightly_20260724_1ab8ff0e: (groups_data, is_last_save)
        #   release_20260827_c9032a29: (groups_data, is_last_save,
        #                               create_batch_id)
        # create_batch_id only names the pre-submitted async CREATE batch
        # for the real data plane (async_swap(create_batch_id=...)), which
        # the sim replaces with a CPU no-op, so it carries no meaning here
        # -- the engine's own sync start_store_kv discards it as well.
        # Index positionally rather than unpacking a fixed arity: a future
        # trailing field must not disable store/save-done modelling again
        # (that silently stalls the engine -- no save-done means v6d
        # objects are never sealed and protected blocks never released).
        _store_entries = tuple(stores.items())
        if _store_entries and not getattr(
                self, "_sim_store_arity_logged", False):
            _entry0 = _store_entries[0][1]
            if not isinstance(_entry0, (tuple, list)):
                raise TypeError(
                    "[V6D Hijack] reqs_to_store entry is "
                    f"{type(_entry0).__name__}, expected a sequence "
                    "(groups_data, is_last_save[, create_batch_id])"
                )
            _arity = len(_entry0)
            if _arity < 2:
                raise ValueError(
                    "[V6D Hijack] reqs_to_store entry arity "
                    f"{_arity} < 2: engine metadata contract changed, "
                    "cannot model store completion / save-done"
                )
            logger.info(
                "[V6D Hijack] reqs_to_store entry arity=%d "
                "(groups_data, is_last_save%s), indexing positionally",
                _arity,
                ", create_batch_id" if _arity == 3 else (
                    f", +{_arity - 2} trailing fields" if _arity > 3
                    else ""),
            )
            self._sim_store_arity_logged = True
        for rid, _entry in _store_entries:
            groups, last = _entry[0], _entry[1]
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
            _s_keys = [k for keys, _ids in groups.values() for k in keys]
            self._staged.append(self.Store(
                state._req, taken, len(_s_keys), _s_keys, last))

    def prepare(self, coroutine, request_id=None):
        future = Future()
        self._preparing.append(future)
        with self._lock:
            if request_id is not None:
                self._preparing_requests[request_id] += 1

        async def tracked():
            try:
                result = await coroutine
            except BaseException as exc:
                self._forget_native_request(request_id)
                future.set_exception(exc)
                raise
            else:
                future.set_result(result)
                return result
            finally:
                with self._lock:
                    if request_id is not None:
                        self._preparing_requests[request_id] -= 1
                        if not self._preparing_requests[request_id]:
                            self._preparing_requests.pop(request_id, None)
                            self._cancelled_preparations.discard(request_id)
        return tracked()

    def settle_preparations(self):
        """Wait for control-plane RPCs (OFFLINE deterministic barrier)."""
        preparing, self._preparing = self._preparing, []
        for future in preparing:
            future.result(timeout=30.0)

    def reap_preparations(self):
        """Publish completed RPCs without blocking the EngineCore thread."""
        pending = []
        for future in self._preparing:
            if future.done():
                # Surface connector-loop failures on the engine thread.
                future.result()
            else:
                pending.append(future)
        self._preparing = pending

    def queue_load(self, request, num_tokens, groups):
        nblocks = sum(len(keys) for keys, _ids in groups.values())
        bw = BandwidthModel.get()
        latency = bw.latency_for(nblocks, True)
        rid = request.request_id
        with self._lock:
            if rid in self._cancelled_preparations:
                return
            facts = self._native_locality.pop(rid, {})
            remote_blocks = remote_bytes = missing_bytes = 0
            sources = Counter()
            for gid, (keys, _ids) in groups.items():
                for key in keys:
                    fact = facts.get((gid, key))
                    if fact is not None:
                        local, size = fact
                        sources["native"] += 1
                    else:
                        sources["fallback"] += 1
                        local = self._seg1_scope == "local_registry" and key in self._local_keys
                        size = None
                    if not local:
                        remote_blocks += 1
                        if size is None:
                            missing_bytes += 1
                        else:
                            remote_bytes += size
            remote_bytes += bw.seg2_bytes(missing_bytes)
            if remote_blocks:
                latency += bw.seg1_latency(remote_blocks, nbytes=remote_bytes)
            self._loads.setdefault(rid, (self.now() + latency, num_tokens))
        source = "native" if not sources["fallback"] else "fallback"
        logger.info("[V6D locality] req=%s source=%s native=%d fallback=%d remote_keys=%d seg1_bytes=%d bytes_fallback=%d",
                    rid, source, sources["native"], sources["fallback"],
                    remote_blocks, remote_bytes, missing_bytes)

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
            self._local_keys.update(store.keys or ())
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
            self._native_locality.clear()
            self._local_keys.clear()
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
            return controller.prepare(coroutine, req.request_id) if controller is not None else coroutine

        def step_waiting(self):
            result = original_waiting(self)
            controller = self._sim_controller
            if controller is not None:
                if Envs.simulation_mode() == "OFFLINE":
                    # Virtual time cannot advance while a real control-plane
                    # coroutine is unresolved, so OFFLINE retains the explicit
                    # synchronization barrier.
                    controller.settle_preparations()
                else:
                    # BLOCKING mirrors production: lookup/allocation RPCs stay
                    # on the connector asyncio loop while EngineCore continues
                    # polling and overlapping them with the simulated GPU span.
                    controller.reap_preparations()
                controller.progress()
            return result

        def step(self):
            if self._sim_controller is not None:
                self._sim_controller.reap_preparations()
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
        # 入口在注册前同样校验；这里保护绕过标准入口直接安装的调用方。
        validate_load_timing()
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
                self._sim_controller._forget_native_request(request.request_id)
                return result
            groups = self._scheduler._reqs_to_load.get(request.request_id, {})
            if groups:
                self._sim_controller.queue_load(request, num_external_tokens, groups)
            if not groups:
                self._sim_controller._forget_native_request(request.request_id)
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
