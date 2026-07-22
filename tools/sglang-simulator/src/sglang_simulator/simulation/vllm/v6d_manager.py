"""
V6D Object Manager hooks for native control-plane ownership simulation.

In native V6D control-plane mode this module keeps the real vLLM
V6dObjectManager/V6dObjectConnectorScheduler classes in the request path, but
bypasses the unavailable CPU-only data plane:

1. Tag each manager with `_SIM_V6D_ACTIVE_WORKER_ID`.
2. Register allocated/sealed block ownership in the shared V6DCacheStorage
   metadata store when CPU data transfer is skipped.
3. Resolve lookup hits through the same shared metadata store and classify
   local vs remote ownership.
4. Bypass scheduler-side `client.create()` while preserving cross-group
   allocation semantics.

The fallback `/dev/shm` tracker is retained only for local development when
etcd is unavailable; dual-node validation should use the shared metastore.
"""

from __future__ import annotations
import fcntl
import json
import os
import shutil
from typing import Any, Iterable

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# File-backed cross-process ownership tracker
# ---------------------------------------------------------------------------

_TRACKER_DIR = "/dev/shm/v6d_sim_tracker"
_OWNERSHIP_FILE = os.path.join(_TRACKER_DIR, "ownership.json")
_STATS_FILE = os.path.join(_TRACKER_DIR, "stats.json")
_LOCK_FILE = os.path.join(_TRACKER_DIR, ".lock")


class V6dBlockOwnershipTracker:
    """Ownership tracker used by the native V6D CPU bypass.

    The preferred storage is shared V6DCacheStorage(etcd), which provides
    physical cross-node visibility between node1 and node2.  The file-backed
    `/dev/shm` fallback is used only when that shared store is unavailable.
    """

    @classmethod
    def _ensure_dir(cls):
        os.makedirs(_TRACKER_DIR, exist_ok=True)

    @classmethod
    def _lock(cls):
        """Acquire exclusive file lock. Returns lock file descriptor."""
        cls._ensure_dir()
        fd = open(_LOCK_FILE, "w")
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        return fd

    @classmethod
    def _unlock(cls, fd):
        """Release file lock."""
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        fd.close()

    @classmethod
    def _read_json(cls, path: str) -> dict:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return {}

    @classmethod
    def _write_json(cls, path: str, data: dict):
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def reset(cls):
        """Reset all tracking data. Call at test setup."""
        if os.path.exists(_TRACKER_DIR):
            shutil.rmtree(_TRACKER_DIR)
        cls._ensure_dir()

    @classmethod
    def _storage(cls):
        try:
            from sglang_simulator.simulation.vllm.v6d_cache_storage import (
                V6DCacheStorage,
            )
            storage = V6DCacheStorage.get_instance()
            if storage.connected:
                return storage
        except Exception:
            logger.debug("[V6D RPC Bypass] etcd tracker unavailable", exc_info=True)
        return None

    @classmethod
    def record_seal(cls, key: str, worker_id: str) -> None:
        """Record that worker_id sealed the given block key."""
        storage = cls._storage()
        if storage is not None and storage.register_block(key, worker_id):
            return
        lock = cls._lock()
        try:
            data = cls._read_json(_OWNERSHIP_FILE)
            data[key] = worker_id
            cls._write_json(_OWNERSHIP_FILE, data)
        finally:
            cls._unlock(lock)

    @classmethod
    def get_owner(cls, key: str) -> str | None:
        """Get the owner worker_id of a block key, or None if unknown."""
        storage = cls._storage()
        if storage is not None:
            owner = storage.lookup_block(key)
            if owner is not None:
                return owner
        lock = cls._lock()
        try:
            data = cls._read_json(_OWNERSHIP_FILE)
            return data.get(key)
        finally:
            cls._unlock(lock)

    @classmethod
    def classify_hit(cls, key: str, current_worker_id: str) -> str:
        """Classify a cache hit as local, remote, or unknown."""
        lock = cls._lock()
        try:
            data = cls._read_json(_OWNERSHIP_FILE)
            owner = data.get(key)
        finally:
            cls._unlock(lock)

        if owner is None:
            return "unknown"
        return "local" if owner == current_worker_id else "remote"

    @classmethod
    def record_hit(cls, worker_id: str, hit_type: str, count: int = 1) -> None:
        """Record hit classification for a worker."""
        lock = cls._lock()
        try:
            stats = cls._read_json(_STATS_FILE)
            if worker_id not in stats:
                stats[worker_id] = {}
            stat_key = f"{hit_type}_hits"
            stats[worker_id][stat_key] = stats[worker_id].get(stat_key, 0) + count
            cls._write_json(_STATS_FILE, stats)
        finally:
            cls._unlock(lock)

    @classmethod
    def get_stats(cls, worker_id: str) -> dict:
        """Get hit stats for a specific worker (safe from any process)."""
        lock = cls._lock()
        try:
            stats = cls._read_json(_STATS_FILE)
            return stats.get(worker_id, {})
        finally:
            cls._unlock(lock)

    @classmethod
    def get_all_stats(cls) -> dict:
        """Get all worker stats."""
        lock = cls._lock()
        try:
            return cls._read_json(_STATS_FILE)
        finally:
            cls._unlock(lock)

    @classmethod
    def get_ownership_count(cls) -> int:
        """Get total number of tracked ownership entries."""
        lock = cls._lock()
        try:
            data = cls._read_json(_OWNERSHIP_FILE)
            return len(data)
        finally:
            cls._unlock(lock)


# ---------------------------------------------------------------------------
# V6dObjectManager Hook
# ---------------------------------------------------------------------------

class C_V6dObjectManagerHook(BaseHook):
    """Hook the real V6dObjectManager for CPU native control-plane mode.

    The hook preserves lookup/allocation ownership semantics while bypassing
    scheduler-side v6d client data-plane calls that require SRPC/CUDA.
    """

    HOOK_CLASS_NAME = "V6dObjectManager"
    HOOK_MODULE_NAME = (
        "vllm.distributed.kv_transfer.kv_connector.v1.v6d_object_connector"
    )

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def override_async_connect(self) -> None:
            """Connect to real v6d daemon for P2P discovery.

            IPC hook patches transfer.init_srpc_transfer to no-op,
            so SRPC BAREX init is skipped and client connects without CUDA.
            """
            try:
                import sys as _sys
                _mod = _sys.modules.get(target.__module__)
                _fn = getattr(_mod, "_connect_v6d_with_retry", None)
                if _fn is None:
                    raise RuntimeError("_connect_v6d_with_retry not found")
                self.client = _fn(self._v6d_url)
                self._sim_fallback_mode = False
                logger.info(
                    "[V6D P2P] Manager group=%d connected to v6d daemon "
                    "at %s (real P2P path)",
                    self._group_id, self._v6d_url,
                )
            except Exception as e:
                self.client = None
                self._sim_fallback_mode = True
                logger.warning(
                    "[V6D P2P] *** FALLBACK TO ETCD MODE *** "
                    "Manager group=%d cannot connect to v6d daemon at %s: %s. "
                    "Cross-node ownership queries will use etcd HTTP API "
                    "instead of real v6d P2P discovery (redis tracker). "
                    "This path DIFFERS from production and may not reproduce "
                    "all inflight/prefill issues.",
                    self._group_id, getattr(self, "_v6d_url", "?"), e,
                )
                if self._v6d_backend is not None:
                    self._v6d_backend.mark_manager_connected(self._group_id)
                return
            # Skip _on_connected: it starts schedrpcserver which conflicts
            # with LAZY_INITIALIZE_KV_TRANSFER_OUTSIDE_VLLM mechanism
            if self._v6d_backend is not None:
                self._v6d_backend.mark_manager_connected(self._group_id)

        target._async_connect = override_async_connect

        def override_init(self, *args, **kwargs):
            """Tag manager with active worker_id from environment."""
            original_init(self, *args, **kwargs)
            worker_id = get_active_worker_id()
            if worker_id:
                self._sim_worker_id = worker_id
                logger.info(
                    f"[V6D RPC Bypass] Manager group={self._group_id} "
                    f"tagged with worker_id={worker_id}")
            else:
                self._sim_worker_id = None
                logger.debug(
                    f"[V6D RPC Bypass] Manager group={self._group_id} "
                    f"no active worker_id")
            self._sim_fallback_mode = False

        target.__init__ = override_init

        # ---- Override seal() to record ownership ----
        original_seal = target.seal

        def override_seal(self, block_hashes: Iterable, request_id=None):
            """Record sealed block ownership without v6d client data-plane."""
            if not getattr(self, "_sim_fallback_mode", False):
                return original_seal(self, block_hashes, request_id=request_id)
            block_hashes_list = list(block_hashes)
            worker_id = getattr(self, "_sim_worker_id", None)
            if worker_id:
                for h in block_hashes_list:
                    key = self._make_key(h)
                    V6dBlockOwnershipTracker.record_seal(key, worker_id)
                logger.info(
                    f"[V6D RPC Bypass] seal: {len(block_hashes_list)} blocks "
                    f"by {worker_id} group={self._group_id}")
            for h in block_hashes_list:
                self._pending_objs.pop(h, None)

        target.seal = override_seal

        # ---- Override _process_lookup() to classify hits ----
        original_process_lookup = target._process_lookup

        def override_process_lookup(
            self,
            block_hashes,
            got_objs: dict[str, Any],
            request_id: str | None,
            unfetched_objs: dict[str, Any] | None = None,
        ) -> int:
            """Process lookup using file-backed ownership tracker."""
            if not getattr(self, "_sim_fallback_mode", False):
                return original_process_lookup(
                    self, block_hashes, got_objs, request_id,
                    unfetched_objs=unfetched_objs)
            worker_id = getattr(self, "_sim_worker_id", None)
            hits = 0
            local_count = 0
            remote_count = 0
            unknown_count = 0

            for h in block_hashes:
                key = self._make_key(h)
                owner = V6dBlockOwnershipTracker.get_owner(key)
                if owner is None:
                    break
                hits += 1
                self._cached_objs[key] = key
                if request_id is not None:
                    self._hold_key_for_req(key, request_id)
                if worker_id is None:
                    unknown_count += 1
                elif owner == worker_id:
                    local_count += 1
                else:
                    remote_count += 1

            if worker_id and hits > 0:
                if local_count:
                    V6dBlockOwnershipTracker.record_hit(
                        worker_id, "local", local_count)
                if remote_count:
                    V6dBlockOwnershipTracker.record_hit(
                        worker_id, "remote", remote_count)
                if unknown_count:
                    V6dBlockOwnershipTracker.record_hit(
                        worker_id, "unknown", unknown_count)

                logger.info(
                    f"[V6D RPC Bypass] Worker {worker_id}: lookup group={self._group_id} "
                    f"hits={hits} local={local_count} remote={remote_count} "
                    f"unknown={unknown_count} (v6d client bypassed)")
                if remote_count > 0:
                    logger.info(
                        f"[V6D RPC Bypass] Worker {worker_id}: "
                        f"cross-node hit! local={local_count} remote={remote_count} "
                        f"unknown={unknown_count} (RPC bypassed)")

            return hits

        target._process_lookup = override_process_lookup

        def override_lookup(self, block_hashes, request_id=None,
                            unfetched_objs=None):
            if getattr(self, "_sim_fallback_mode", True):
                return self._process_lookup(
                    list(block_hashes), {}, request_id,
                    unfetched_objs=unfetched_objs)
            if getattr(self, "client", None) is None:
                return 0
            hits = 0
            for h in block_hashes:
                key = self._make_key(h)
                if key in self._cached_objs:
                    hits += 1
                    continue
                try:
                    if self.client.exists(key):
                        self._cached_objs[key] = key
                        if request_id is not None:
                            self._hold_key_for_req(key, request_id)
                        hits += 1
                    else:
                        break
                except Exception:
                    break
            return hits * getattr(self, "_group_block_size", 1)

        async def override_async_lookup(self, block_hashes, request_id=None,
                                        unfetched_objs=None):
            if getattr(self, "_sim_fallback_mode", True):
                return self._process_lookup(
                    list(block_hashes), {}, request_id,
                    unfetched_objs=unfetched_objs)
            if getattr(self, "client", None) is None:
                return 0
            import asyncio
            loop = asyncio.get_event_loop()
            hits = 0
            for h in block_hashes:
                key = self._make_key(h)
                if key in self._cached_objs:
                    hits += 1
                    continue
                try:
                    exists = await loop.run_in_executor(None, self.client.exists, key)
                    if exists:
                        self._cached_objs[key] = key
                        if request_id is not None:
                            self._hold_key_for_req(key, request_id)
                        hits += 1
                    else:
                        break
                except Exception:
                    break
            return hits * getattr(self, "_group_block_size", 1)

        def override_get_key(self, block_hash, request_id=None):
            if not getattr(self, "_sim_fallback_mode", False):
                return original_get_key(self, block_hash, request_id=request_id)
            key = self._make_key(block_hash)
            if V6dBlockOwnershipTracker.get_owner(key) is None:
                return None
            self._cached_objs[key] = key
            if request_id is not None:
                self._hold_key_for_req(key, request_id)
            return key

        async def override_async_get_key(self, block_hash, request_id=None):
            return override_get_key(self, block_hash, request_id=request_id)

        def override_reset(self):
            if not getattr(self, "_sim_fallback_mode", False):
                return original_reset(self)
            self._cached_objs.clear()
            self._cached_objs_reqs.clear()
            self._pending_objs.clear()

        original_lookup = target.lookup
        original_async_lookup = target.async_lookup
        original_get_key = target.get_key
        original_async_get_key = target.async_get_key
        original_reset = target.reset
        target.lookup = override_lookup
        target.async_lookup = override_async_lookup
        target.get_key = override_get_key
        target.async_get_key = override_async_get_key
        target.reset = override_reset

        # ---- Override batch_allocate to record ownership ----
        if hasattr(target, 'batch_allocate'):
            original_batch_allocate = target.batch_allocate

            def override_batch_allocate(self, block_hashes, size, shape,
                                        dtype, request_id=None):
                """Allocate simulated V6D keys without v6d client data-plane."""
                if not getattr(self, "_sim_fallback_mode", False):
                    return original_batch_allocate(
                        self, block_hashes, size, shape, dtype,
                        request_id=request_id)
                result = {}
                worker_id = getattr(self, "_sim_worker_id", None)
                for h in block_hashes:
                    key = self._make_key(h)
                    self._pending_objs[h] = key
                    result[h] = key
                    if worker_id:
                        V6dBlockOwnershipTracker.record_seal(key, worker_id)
                logger.info(
                    f"[V6D RPC Bypass] batch_allocate: {len(result)} blocks "
                    f"group={self._group_id} req={request_id} owner={worker_id}")
                return result

            target.batch_allocate = override_batch_allocate

        logger.info("[V6D Hijack] V6dObjectManager hook installed "
                    "(real v6d P2P + etcd fallback)")


class C_V6dObjectConnectorSchedulerHook(BaseHook):
    """Hook scheduler cross-group allocation to avoid real v6d client.create."""

    HOOK_CLASS_NAME = "V6dObjectConnectorScheduler"
    HOOK_MODULE_NAME = (
        "vllm.distributed.kv_transfer.kv_connector.v1.v6d_object_connector"
    )

    @classmethod
    def hook(cls, target):
        import sys

        module = sys.modules.get(target.__module__)
        block_hash_list_with_block_size = getattr(
            module, "BlockHashListWithBlockSize", None
        )

        def _group_block_hashes(self, block_hashes, group_block_size):
            if group_block_size == self.hash_block_size:
                return block_hashes
            if block_hash_list_with_block_size is None:
                return block_hashes
            return block_hash_list_with_block_size(
                block_hashes, self.hash_block_size, group_block_size
            )

        def _num_hash_blocks(self, request):
            spec_cfg = getattr(self.vllm_config, "speculative_config", None)
            use_eagle = bool(spec_cfg and spec_cfg.use_eagle())
            if getattr(self, "_is_hybrid_backend", False) or use_eagle:
                return max((request.num_tokens - 1) // self.hash_block_size, 0)
            return max(request.num_tokens // self.hash_block_size, 0)

        def _all_group_ids(self):
            return sorted(
                set(getattr(self, "full_attention_group_ids", set()))
                | set(getattr(self, "mamba_group_ids", set()))
            )

        def _finalize_hit(self, request, num_computed_tokens, num_hash_blocks,
                          hit_length):
            lcm_block_size = getattr(self, "lcm_block_size", self.hash_block_size)
            if lcm_block_size > 0:
                hit_length = hit_length // lcm_block_size * lcm_block_size
            if hit_length <= 0:
                return 0, False
            if hasattr(self, "_to_load_token_idx"):
                self._to_load_token_idx[request.request_id] = num_computed_tokens
            if hasattr(self, "_hit_stats"):
                self._hit_stats.record(
                    total_tokens=num_hash_blocks * self.hash_block_size,
                    hit_tokens=hit_length,
                )
            logger.info(
                "[V6D RPC Bypass] Request %s: scheduler lookup hit %d "
                "tokens from num_computed_tokens=%d",
                request.request_id,
                hit_length,
                num_computed_tokens,
            )
            return hit_length, True

        def _mamba_validate(self, block_hashes, num_computed_tokens, fa_hit_length):
            """Validate mamba state at aligned boundaries.

            Mirrors the original V6dObjectConnectorScheduler logic:
            search aligned boundaries from right to left, find the first
            boundary where ALL mamba groups have a matching block in etcd.
            If no boundary matches, return 0.
            """
            mamba_group_ids = sorted(getattr(self, "mamba_group_ids", set()))
            if not mamba_group_ids:
                return fa_hit_length

            lcm_block_size = getattr(self, "lcm_block_size", self.hash_block_size)
            if lcm_block_size <= 0:
                return fa_hit_length

            max_mamba_hit_length = (
                fa_hit_length // lcm_block_size * lcm_block_size)
            if max_mamba_hit_length <= 0:
                return 0

            aligned_hit_lengths = list(range(
                max_mamba_hit_length, 0, -lcm_block_size))

            for aligned_hit_length in aligned_hit_lengths:
                all_match = True
                for group_id in mamba_group_ids:
                    manager = self.managers[group_id]
                    group_block_size = self.group_block_sizes[group_id]
                    group_hashes = _group_block_hashes(
                        self, block_hashes, group_block_size)
                    abs_idx = (
                        (num_computed_tokens + aligned_hit_length)
                        // group_block_size - 1
                    )
                    if abs_idx < 0 or abs_idx >= len(group_hashes):
                        all_match = False
                        break
                    key = manager._make_key(group_hashes[abs_idx])
                    if V6dBlockOwnershipTracker.get_owner(key) is None:
                        all_match = False
                        break
                if all_match:
                    return aligned_hit_length

            return 0

        def override_get_num_new_matched_tokens(
            self, request, num_computed_tokens
        ):
            num_hash_blocks = _num_hash_blocks(self, request)
            block_hashes = request.block_hashes[:num_hash_blocks]
            if not block_hashes:
                return 0, False

            # Step 1: full attention groups — take min
            fa_group_ids = sorted(
                getattr(self, "full_attention_group_ids", set()))
            hit_length = num_hash_blocks * self.hash_block_size
            for group_id in fa_group_ids:
                manager = self.managers[group_id]
                group_block_size = self.group_block_sizes[group_id]
                group_hashes = _group_block_hashes(
                    self, block_hashes, group_block_size
                )
                start_block_idx = num_computed_tokens // group_block_size
                group_hits = manager.lookup(
                    group_hashes[start_block_idx:],
                    request_id=request.request_id,
                    unfetched_objs={},
                )
                hit_length = min(hit_length, group_hits * group_block_size)

            # Step 2: mamba aligned boundary validation (matches original vLLM)
            hit_length = _mamba_validate(
                self, block_hashes, num_computed_tokens, hit_length)

            return _finalize_hit(
                self, request, num_computed_tokens, num_hash_blocks, hit_length
            )

        async def override_async_get_num_new_matched_tokens(
            self, request, num_computed_tokens
        ):
            num_hash_blocks = _num_hash_blocks(self, request)
            block_hashes = request.block_hashes[:num_hash_blocks]
            if not block_hashes:
                return 0, False

            # Step 1: full attention groups — take min
            fa_group_ids = sorted(
                getattr(self, "full_attention_group_ids", set()))
            hit_length = num_hash_blocks * self.hash_block_size
            for group_id in fa_group_ids:
                manager = self.managers[group_id]
                group_block_size = self.group_block_sizes[group_id]
                group_hashes = _group_block_hashes(
                    self, block_hashes, group_block_size
                )
                start_block_idx = num_computed_tokens // group_block_size
                group_hits = await manager.async_lookup(
                    group_hashes[start_block_idx:],
                    request_id=request.request_id,
                    unfetched_objs={},
                )
                hit_length = min(hit_length, group_hits * group_block_size)

            # Step 2: mamba aligned boundary validation (matches original vLLM)
            hit_length = _mamba_validate(
                self, block_hashes, num_computed_tokens, hit_length)

            return _finalize_hit(
                self, request, num_computed_tokens, num_hash_blocks, hit_length
            )

        target.get_num_new_matched_tokens = override_get_num_new_matched_tokens
        target.async_get_num_new_matched_tokens = override_async_get_num_new_matched_tokens

        original_request_finished = getattr(target, "request_finished", None)
        original_request_finished_all_groups = getattr(
            target, "request_finished_all_groups", None
        )

        def _complete_cpu_store_noop(self, req_id):
            if hasattr(self, "_pending_store_reqs"):
                self._pending_store_reqs.discard(req_id)
            if hasattr(self, "_finished_pending_store_reqs"):
                self._finished_pending_store_reqs.discard(req_id)
            if hasattr(self, "_storing_block_hashes"):
                self._storing_block_hashes.pop(req_id, None)
            try:
                from vllm.v1.hybrid_connector import (
                    mark_backend_save_done, sched_get_req,
                )
                req = sched_get_req(req_id)
                if req is not None:
                    mark_backend_save_done(req)
            except Exception:
                pass
            if hasattr(self, "_release_protected_blocks"):
                self._release_protected_blocks(req_id)
            logger.info(
                "[V6D RPC Bypass] Request %s: completed scheduler-side "
                "CPU no-op store without async wait",
                req_id,
            )

        def override_request_finished(self, request, block_ids):
            if original_request_finished is None:
                return False, None
            should_wait, params = original_request_finished(self, request, block_ids)
            # Always clear _saving for CPU no-op store
            if hasattr(self, "_saving"):
                self._saving.pop(request.request_id, None)
            if should_wait:
                _complete_cpu_store_noop(self, request.request_id)
                return False, params
            return should_wait, params

        def override_request_finished_all_groups(self, request, block_ids):
            if original_request_finished_all_groups is None:
                return False, None
            should_wait, params = original_request_finished_all_groups(
                self, request, block_ids
            )
            if should_wait:
                _complete_cpu_store_noop(self, request.request_id)
                return False, params
            return should_wait, params

        target.request_finished = override_request_finished
        target.request_finished_all_groups = override_request_finished_all_groups

        def override_cross_group_batch_allocate(
            self,
            group_candidates,
            request_id=None,
        ):
            result = {}
            for group_id, candidate_hashes in group_candidates.items():
                if not candidate_hashes:
                    continue
                manager = self.managers[group_id]
                block_bytes, block_shape = self._group_block_bytes[group_id]
                torch_dtype = self._group_torch_dtype[group_id]
                result[group_id] = manager.batch_allocate(
                    candidate_hashes,
                    block_bytes,
                    block_shape,
                    torch_dtype,
                    request_id=request_id,
                )
            logger.info(
                "[V6D RPC Bypass] cross_group_batch_allocate: groups=%s req=%s",
                sorted(result),
                request_id,
            )
            return result

        target._cross_group_batch_allocate = override_cross_group_batch_allocate

        original_update_output = target.update_connector_output
        def override_update_connector_output(self, connector_output):
            _fs = getattr(connector_output, "finished_sending", None)
            return original_update_output(self, connector_output)
        target.update_connector_output = override_update_connector_output
        logger.info(
            "[V6D Hijack] V6dObjectConnectorScheduler hook installed "
            "(cross-group allocation bypass)"
        )


# ---------------------------------------------------------------------------
# Active worker context for automatic manager tagging
# ---------------------------------------------------------------------------

_ENV_KEY = "_SIM_V6D_ACTIVE_WORKER_ID"


def set_active_worker_id(worker_id: str | None) -> None:
    """Set the active worker ID for subsequent V6D manager creation.

    Uses environment variable to survive across process boundaries
    (EngineCore subprocess inherits env from parent).
    """
    if worker_id:
        os.environ[_ENV_KEY] = worker_id
        logger.info(f"[V6D RPC Bypass] Active worker set to: {worker_id}")
    else:
        os.environ.pop(_ENV_KEY, None)


def get_active_worker_id() -> str | None:
    """Get active worker id, deriving a stable per-instance id when absent."""
    explicit = os.environ.get(_ENV_KEY)
    if explicit:
        return explicit

    pod_name = os.environ.get("POD_NAME") or os.environ.get("HOSTNAME")
    worker_name = os.environ.get("WORKER_NAME")
    if pod_name and worker_name:
        return f"{pod_name}:{worker_name}"
    if pod_name:
        return pod_name
    for key in ("SPECTRUM_INSTANCE_NAME", "POD_IP", "ALIYUN_ECI_ETH0_IP"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def set_manager_worker_id(connector, worker_id: str) -> None:
    """Set the worker_id on all V6dObjectManager instances in a connector."""

    if hasattr(connector, 'managers'):
        for gid, manager in connector.managers.items():
            manager._sim_worker_id = worker_id
            logger.info(
                f"[V6D RPC Bypass] Manager group={gid} tagged with "
                f"worker_id={worker_id}")


class C_HybridSchedulerHook(BaseHook):
    """Hook HybridScheduler.step() to flush _saving into _saved.

    In CPU simulation, start_store_kv is a no-op and never sends
    _SAVE_DONE_REQ RPC.  As a result, _step_saved() has an empty _saved
    deque and never clears _saving, causing block exhaustion.

    This hook flushes all _saving keys into _saved before each step(),
    so _step_saved() processes them naturally (releases blocks + clears
    _saving).  This is safe because all saves are no-ops in CPU mode.
    """

    HOOK_CLASS_NAME = "HybridScheduler"
    HOOK_MODULE_NAME = "vllm.v1.hybrid_connector"

    @classmethod
    def hook(cls, target):
        original_step = target.step

        def override_step(self):
            saving = getattr(self, "_saving", None)
            saved = getattr(self, "_saved", None)
            if saving and saved is not None:
                flushed = []
                for req_id in list(saving.keys()):
                    if req_id not in saved:
                        saved.append(req_id)
                        flushed.append(req_id)
                if flushed:
                    logger.info(
                        "[V6D Hijack] Flushed %d saving entries into _saved "
                        "for _step_saved processing: %s",
                        len(flushed), flushed,
                    )
            return original_step(self)

        target.step = override_step
        logger.info(
            "[V6D Hijack] HybridScheduler.step hook installed "
            "(_saving flush for CPU no-op store)")
