"""
V6D Object Manager Hook - Implements RPC bypass for cross-node simulation.

In a multi-node V6D cluster with real GPUs, cross-node KV cache sharing
works via RPC (SRPC over GPU DMA). In CPU-only simulation, we bypass RPC
by tracking block ownership at the simulation level:

1. Each V6dObjectManager instance is tagged with a worker_id (node identity)
2. On seal(): record which worker owns each block_hash key
3. On lookup(): detect cross-node hits (owner != current worker)
4. Remote data access works via shared vineyardd IPC (no RPC needed)

This preserves the full V6D query/match semantics while accurately
distinguishing local hits from cross-node hits for metrics reporting.

CROSS-PROCESS DESIGN:
Since vLLM runs EngineCore in subprocesses, we use file-backed storage
under /dev/shm/ (tmpfs) for cross-process state sharing. File locking
ensures correctness under concurrent access.
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
    """Cross-process tracker for block ownership across all workers.

    Uses file-backed storage under /dev/shm/ (tmpfs) so that data
    persists across forked subprocesses. File locking ensures atomicity.
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
    def record_seal(cls, key: str, worker_id: str) -> None:
        """Record that worker_id sealed (owns) the given block key."""
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
    """Hook V6dObjectManager to track block ownership and classify hits.

    Intercepts seal() and _process_lookup() to implement RPC bypass:
    - seal(): records which worker owns each block
    - _process_lookup(): classifies hits as local vs remote (cross-node)
    """

    HOOK_CLASS_NAME = "V6dObjectManager"
    HOOK_MODULE_NAME = (
        "vllm.distributed.kv_transfer.kv_connector.v1.v6d_object_connector"
    )

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

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

        target.__init__ = override_init

        # ---- Override seal() to record ownership ----
        original_seal = target.seal

        def override_seal(self, block_hashes: Iterable, request_id=None):
            """Seal blocks and record ownership in the tracker."""
            block_hashes_list = list(block_hashes)
            original_seal(self, block_hashes_list, request_id=request_id)

            worker_id = getattr(self, "_sim_worker_id", None)
            if worker_id:
                for h in block_hashes_list:
                    key = self._make_key(h)
                    V6dBlockOwnershipTracker.record_seal(key, worker_id)
                logger.debug(
                    f"[V6D RPC Bypass] seal: {len(block_hashes_list)} blocks "
                    f"by {worker_id}")

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
            """Process lookup and classify hits as local/remote."""
            hits = original_process_lookup(
                self, block_hashes, got_objs, request_id,
                unfetched_objs=unfetched_objs)

            worker_id = getattr(self, "_sim_worker_id", None)
            if worker_id and hits > 0:
                local_count = 0
                remote_count = 0
                unknown_count = 0

                for i, h in enumerate(block_hashes):
                    if i >= hits:
                        break
                    key = self._make_key(h)
                    hit_type = V6dBlockOwnershipTracker.classify_hit(
                        key, worker_id)
                    if hit_type == "local":
                        local_count += 1
                    elif hit_type == "remote":
                        remote_count += 1
                    else:
                        unknown_count += 1

                if local_count:
                    V6dBlockOwnershipTracker.record_hit(
                        worker_id, "local", local_count)
                if remote_count:
                    V6dBlockOwnershipTracker.record_hit(
                        worker_id, "remote", remote_count)
                if unknown_count:
                    V6dBlockOwnershipTracker.record_hit(
                        worker_id, "unknown", unknown_count)

                if remote_count > 0:
                    logger.info(
                        f"[V6D RPC Bypass] Worker {worker_id}: "
                        f"cross-node hit! "
                        f"local={local_count} remote={remote_count} "
                        f"unknown={unknown_count} "
                        f"(RPC bypassed)")

            return hits

        target._process_lookup = override_process_lookup

        # ---- Override batch_allocate to record ownership ----
        if hasattr(target, 'batch_allocate'):
            original_batch_allocate = target.batch_allocate

            def override_batch_allocate(self, block_hashes, size, shape,
                                        dtype, request_id=None):
                """Batch allocate and record ownership."""
                result = original_batch_allocate(
                    self, block_hashes, size, shape, dtype,
                    request_id=request_id)
                worker_id = getattr(self, "_sim_worker_id", None)
                if worker_id and result:
                    for h in block_hashes:
                        key = self._make_key(h)
                        V6dBlockOwnershipTracker.record_seal(key, worker_id)
                return result

            target.batch_allocate = override_batch_allocate

        logger.info("[V6D Hijack] V6dObjectManager hook installed "
                    "(RPC bypass + ownership tracking)")


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
    """Get the currently active worker ID from environment."""
    return os.environ.get(_ENV_KEY)


def set_manager_worker_id(connector, worker_id: str) -> None:
    """Set the worker_id on all V6dObjectManager instances in a connector."""
    if hasattr(connector, 'managers'):
        for gid, manager in connector.managers.items():
            manager._sim_worker_id = worker_id
            logger.info(
                f"[V6D RPC Bypass] Manager group={gid} tagged with "
                f"worker_id={worker_id}")
