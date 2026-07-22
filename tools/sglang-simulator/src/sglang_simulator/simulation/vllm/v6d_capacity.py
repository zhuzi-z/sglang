"""
V6D Virtual Capacity Manager — C_VineyardServerHook + runtime patches.

This module consolidates ALL v6d-side simulation hooks needed to run a
vineyardd process on CPU without RDMA:

1. **SRPC bypass** — ``init_srpc*`` → no-op, ``init_transfer_engine_client``
   → ``init_mmap`` only (shared-memory IPC, no RDMA).
2. **VineyardPeer rpc=false** — inject ``-rpc=false`` into argv so the C++
   barex SRPC init is skipped, while P2P tracker (Redis) discovery is
   preserved.
3. **Virtual capacity control** — ``C_VineyardServerHook`` patches four
   ``VineyardServer`` methods to enforce a fixed virtual memory capacity
   (1 GB / 2 MB per blob) for deterministic eviction simulation.

Call ``install_v6d_runtime_hooks()`` once at vineyardd startup (before
``v6d.cli.cli.main()``) to apply all three.  Individual functions are also
exposed for fine-grained control.

Simplified model for capacity simulation:
  - Virtual capacity is always **1 GB** (regardless of real server config).
  - Every blob allocation counts as **2 MB** (regardless of actual size).
  - This gives ~512 blobs before eviction triggers — deterministic and
    predictable for simulation.

Enables both eviction paths in ``TieredVineyardPeer``:

1. **Proactive** — ``AsyncEvictor`` polls ``get_memory_usage()`` → virtual
   values trigger eviction at ``memory_usage_max`` / ``memory_usage_critical``
   thresholds.
2. **Reactive** — ``create_blobs()`` raises ``NotEnoughMemoryException``
   when virtual capacity exceeded → ``TieredVineyardPeer`` catches →
   ``_trigger_emergency_eviction()`` → ``drop_names()`` decrements counter →
   retry succeeds.

Configuration:

  ``SGLANG_SIMULATOR_V6D_CAPACITY_CONTROL=1``
      Enable virtual capacity control (SRPC bypass and VineyardPeer patch
      are always applied by ``install_v6d_runtime_hooks()``).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from sglang_simulator.hook import BaseHook

logger = logging.getLogger("sglang_simulator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENV_ENABLE = "SGLANG_SIMULATOR_V6D_CAPACITY_CONTROL"

_VIRTUAL_CAPACITY = 1 << 30   # 1 GB
_BLOB_SIZE = 2 << 20          # 2 MB per blob (fixed for simulation)


# ---------------------------------------------------------------------------
# VirtualCapacityManager
# ---------------------------------------------------------------------------

class VirtualCapacityManager:
    """Thread-safe virtual memory capacity tracker.

    Uses fixed values: 1 GB total, 2 MB per blob.
    A single instance is shared across all ``VineyardServer`` method calls
    within the vineyardd process.
    """

    _instance: Optional["VirtualCapacityManager"] = None
    _singleton_lock = threading.Lock()

    def __init__(self):
        self._total: int = _VIRTUAL_CAPACITY
        self._used: int = 0
        self._blob_ids: set[int] = set()  # tracked allocated blob IDs
        self._lock = threading.Lock()

    # -- singleton ----------------------------------------------------------

    @classmethod
    def get_instance(cls) -> Optional["VirtualCapacityManager"]:
        return cls._instance

    @classmethod
    def get_or_create(cls) -> "VirtualCapacityManager":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for test isolation)."""
        with cls._singleton_lock:
            cls._instance = None

    # -- allocation tracking ------------------------------------------------

    def try_allocate(self, blob_count: int) -> bool:
        """Return True if *blob_count* blobs (each 2 MB) fit in capacity."""
        with self._lock:
            return self._used + blob_count * _BLOB_SIZE <= self._total

    def record_allocate(self, payloads: list):
        """Record successful blob allocation (each blob = 2 MB)."""
        with self._lock:
            for payload in payloads:
                blob_id = payload.object_id.id
                self._blob_ids.add(blob_id)
                self._used += _BLOB_SIZE
            logger.debug(
                "[V6D Capacity] allocate: +%d blobs (+%d bytes), "
                "used=%d/%d (%.1f%%)",
                len(payloads), len(payloads) * _BLOB_SIZE,
                self._used, self._total,
                self._used / self._total * 100 if self._total > 0 else 0,
            )

    def record_free(self, ids: list) -> int:
        """Record blob deletion by blob IDs.  Returns total freed bytes.

        Only decrements for IDs that were previously tracked.  This is used
        for the direct ``del_blob`` path.
        """
        freed_count = 0
        with self._lock:
            for oid in ids:
                int_id = oid.id if hasattr(oid, "id") else oid
                if int_id in self._blob_ids:
                    self._blob_ids.discard(int_id)
                    self._used -= _BLOB_SIZE
                    freed_count += 1
            if freed_count > 0:
                logger.debug(
                    "[V6D Capacity] free: %d blobs (-%d bytes), "
                    "used=%d/%d (%.1f%%)",
                    freed_count, freed_count * _BLOB_SIZE,
                    self._used, self._total,
                    self._used / self._total * 100 if self._total > 0 else 0,
                )
        return freed_count * _BLOB_SIZE

    def record_free_by_count(self, count: int) -> int:
        """Free *count* blobs (each 2 MB) by count, not by ID.

        Used by ``drop_names`` hook: during eviction the DELETE lease has
        empty ``blob_ids``, so ``del_blob`` receives 0 IDs and cannot
        decrement the counter.  ``drop_names`` is always called with the
        object keys, so we decrement by object count here.
        """
        if count <= 0:
            return 0
        with self._lock:
            actual = min(count * _BLOB_SIZE, self._used)
            self._used -= actual
            freed_blobs = actual // _BLOB_SIZE
            # Trim tracked blob IDs set to stay consistent
            for _ in range(freed_blobs):
                if self._blob_ids:
                    self._blob_ids.pop()
            logger.debug(
                "[V6D Capacity] free_by_count: %d blobs (-%d bytes), "
                "used=%d/%d (%.1f%%)",
                freed_blobs, actual,
                self._used, self._total,
                self._used / self._total * 100 if self._total > 0 else 0,
            )
        return actual

    def get_usage(self) -> tuple[int, int]:
        """Return (used_bytes, total_bytes)."""
        with self._lock:
            return self._used, self._total

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "used": self._used,
                "total": self._total,
                "usage_percent": (
                    self._used / self._total if self._total > 0 else 0.0
                ),
                "tracked_blobs": len(self._blob_ids),
            }


# ---------------------------------------------------------------------------
# 1. SRPC bypass — no-op all SRPC init entrypoints
# ---------------------------------------------------------------------------

def patch_srpc_bypass():
    """Replace all SRPC init functions with no-ops.

    Patches ``v6d.common.transfer`` and ``v6d.lite.common.transfer_engine``
    so that ``init_srpc*`` → no-op and ``init_transfer_engine_client`` →
    ``init_mmap`` (shared-memory IPC only, no RDMA).
    """
    import v6d.common.transfer as transfer

    def _skip_srpc(*args, **kwargs):
        logger.info("[v6d-sim] skip SRPC init")
        return None

    transfer.init_srpc_transfer = _skip_srpc
    if hasattr(transfer, "init_srpc"):
        transfer.init_srpc = _skip_srpc
    if hasattr(transfer, "init_srpc_"):
        transfer.init_srpc_ = _skip_srpc

    if hasattr(transfer, "init_mmap") and hasattr(
        transfer, "init_transfer_engine_client"
    ):
        def _init_client(fd: int, size: int):
            return transfer.init_mmap(fd, size)

        transfer.init_transfer_engine_client = _init_client

    logger.info("[v6d-sim] patched v6d.common.transfer SRPC entrypoints")

    try:
        import v6d.lite.common.transfer_engine as te
        if hasattr(te, "init_srpc"):
            te.init_srpc = _skip_srpc
        if hasattr(te, "init_srpc_"):
            te.init_srpc_ = _skip_srpc
        if hasattr(te, "init_srpc_transfer"):
            te.init_srpc_transfer = _skip_srpc
        logger.info("[v6d-sim] patched v6d.lite transfer_engine SRPC entrypoints")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 2a. C_VineyardPeerHook — inject -rpc=false at class-definition time
# ---------------------------------------------------------------------------

class C_VineyardPeerHook(BaseHook):
    """Hook ``VineyardPeer`` to inject ``-rpc=false`` into argv.

    Skips the C++ SRPC barex initialization while preserving P2P
    tracker (Redis) discovery capability.
    """

    HOOK_CLASS_NAME = "VineyardPeer"
    HOOK_MODULE_NAME = "v6d.server.peers.vineyard.peer"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def patched_init(
            self,
            argc=0,
            argv=None,
            tracker_url=None,
            tracker_key_prefix=None,
            lazy_load=True,
        ):
            patched_argv = list(argv) if argv is not None else None
            if patched_argv is not None:
                if not any(a.startswith("-rpc=") for a in patched_argv):
                    patched_argv.append("-rpc=false")
                    argc = len(patched_argv)
            return original_init(
                self, argc, patched_argv, tracker_url,
                tracker_key_prefix, lazy_load,
            )

        target.__init__ = patched_init
        logger.info("[v6d-sim] C_VineyardPeerHook installed: rpc=false")


# ---------------------------------------------------------------------------
# 3. C_VineyardServerHook — virtual memory capacity control
# ---------------------------------------------------------------------------

def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


class C_VineyardServerHook(BaseHook):
    """Hook ``VineyardServer`` to enforce virtual memory capacity.

    Intercepts ``VineyardServer`` at class-definition time via the
    class-hook mechanism.  When ``SGLANG_SIMULATOR_V6D_CAPACITY_CONTROL``
    is enabled, patches four methods:

      - ``get_memory_usage()`` → return virtual ``(used, total)``
      - ``create_blobs(sizes)`` → check capacity (each blob = 2 MB), raise
        ``NotEnoughMemoryException`` if exceeded, else call original and
        record allocation
      - ``create_blob(size)`` → same for single blob
      - ``drop_names(names)`` → decrement virtual counter (each object =
        1 blob = 2 MB), then call original

    ``drop_names`` is used instead of ``del_blob`` because during eviction
    the DELETE lease has empty ``blob_ids``, causing ``del_blob`` to receive
    0 IDs.  ``drop_names`` is always called with the object keys.

    Fixed model: 1 GB capacity, 2 MB per blob → ~512 blobs max.
    """

    HOOK_CLASS_NAME = "VineyardServer"
    HOOK_MODULE_NAME = "v6d.lite.server.vineyard_lite"

    @classmethod
    def hook(cls, target):
        if not _env_enabled(_ENV_ENABLE):
            logger.debug("[V6D Capacity] %s not set, skipping hook", _ENV_ENABLE)
            return

        try:
            from v6d.common.exceptions import NotEnoughMemoryException
        except ImportError:
            logger.warning(
                "[V6D Capacity] Cannot import NotEnoughMemoryException, "
                "skipping hook"
            )
            return

        logger.info(
            "[V6D Capacity] Installing C_VineyardServerHook: "
            "capacity=%d MB, blob_size=%d MB",
            _VIRTUAL_CAPACITY // (1 << 20),
            _BLOB_SIZE // (1 << 20),
        )

        manager = VirtualCapacityManager.get_or_create()

        # Save original methods
        original_create_blob = target.create_blob
        original_create_blobs = target.create_blobs
        original_drop_names = target.drop_names

        # -- patched methods -------------------------------------------------

        def patched_get_memory_usage(self) -> tuple[int, int]:
            """Return virtual (used, total) — always 1 GB capacity."""
            return manager.get_usage()

        def patched_create_blobs(
            self, sizes: list[int], request_id: str = ""
        ) -> list:
            """Check virtual capacity before real allocation.

            Each blob counts as 2 MB regardless of actual size.
            """
            blob_count = len(sizes)
            if not manager.try_allocate(blob_count):
                used, total = manager.get_usage()
                raise NotEnoughMemoryException(
                    f"Virtual capacity exceeded: used={used} "
                    f"({used // _BLOB_SIZE} blobs), "
                    f"requested={blob_count} blobs ({blob_count * _BLOB_SIZE}), "
                    f"total={total} ({total // _BLOB_SIZE} blobs), "
                    f"deficit={used + blob_count * _BLOB_SIZE - total}"
                )
            payloads = original_create_blobs(self, sizes, request_id)
            manager.record_allocate(payloads)
            return payloads

        def patched_create_blob(self, size: int, request_id: str = ""):
            """Check virtual capacity for single blob (counts as 2 MB)."""
            if not manager.try_allocate(1):
                used, total = manager.get_usage()
                raise NotEnoughMemoryException(
                    f"Virtual capacity exceeded: used={used} "
                    f"({used // _BLOB_SIZE} blobs), "
                    f"requested=1 blob ({_BLOB_SIZE}), "
                    f"total={total} ({total // _BLOB_SIZE} blobs), "
                    f"deficit={used + _BLOB_SIZE - total}"
                )
            payload = original_create_blob(self, size, request_id)
            manager.record_allocate([payload])
            return payload

        def patched_drop_names(self, names, request_id: str = ""):
            """Decrement virtual counter (1 object = 1 blob = 2 MB).

            Called during both eviction (via ``_batch_discard``) and normal
            deletion (via ``discard``).  ``del_blob`` cannot be used because
            the eviction DELETE lease has empty ``blob_ids``.
            """
            if names:
                manager.record_free_by_count(len(names))
            return original_drop_names(self, names, request_id)

        # Apply patches
        target.get_memory_usage = patched_get_memory_usage
        target.create_blob = patched_create_blob
        target.create_blobs = patched_create_blobs
        target.drop_names = patched_drop_names

        logger.info(
            "[V6D Capacity] C_VineyardServerHook installed: "
            "get_memory_usage, create_blob, create_blobs, drop_names "
            "(capacity=%d MB, blob=%d MB)",
            _VIRTUAL_CAPACITY // (1 << 20),
            _BLOB_SIZE // (1 << 20),
        )


# ---------------------------------------------------------------------------
# Public API for external inspection
# ---------------------------------------------------------------------------

def get_capacity_stats() -> dict:
    """Return current virtual capacity statistics (for monitoring)."""
    mgr = VirtualCapacityManager.get_instance()
    if mgr is None:
        return {"enabled": False}
    stats = mgr.get_stats()
    stats["enabled"] = True
    return stats


# ---------------------------------------------------------------------------
# Unified entry point — call once at vineyardd startup
# ---------------------------------------------------------------------------

def install_v6d_runtime_hooks():
    """Apply all v6d-side simulation hooks.

    Always applies:
      1. C_VineyardPeerHook (inject ``-rpc=false`` via class hook)
      2. SRPC bypass (``patch_srpc_bypass``)

    Conditionally applies (when ``SGLANG_SIMULATOR_V6D_CAPACITY_CONTROL=1``):
      3. C_VineyardServerHook (virtual memory capacity control)

    Class hooks MUST be installed before any import that triggers
    ``VineyardServer`` / ``VineyardPeer`` class definitions.  SRPC bypass
    patches module-level functions and runs after class hook installation.
    """
    from sglang_simulator.hook import install_class_hooks

    hooks = [C_VineyardPeerHook]
    if _env_enabled(_ENV_ENABLE):
        hooks.append(C_VineyardServerHook)
        logger.info("[v6d-sim] C_VineyardServerHook enabled")

    install_class_hooks(hooks)

    patch_srpc_bypass()
