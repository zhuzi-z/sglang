"""v6d daemon-side simulation hooks — run vineyardd on CPU without RDMA.

Applied by ``install_v6d_runtime_hooks()`` at vineyardd startup.  Core idea:
``VirtualCapacityManager`` tracks declared blob sizes against the configured
``--vineyard-size`` and drives ``TieredVineyardPeer`` eviction (AsyncEvictor
polling + ``NotEnoughMemoryException`` → emergency eviction), while physical
allocation stays minimal (1G mmap, 4K per blob).

``SGLANG_SIMULATOR_V6D_LOGICAL_CAPACITY`` (bytes) overrides the capacity
derived from ``--vineyard-size``.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from typing import Optional

from sglang_simulator.hook import BaseHook, install_class_hooks

logger = logging.getLogger("sglang_simulator")

# TCP-only SRPC engine: skips barex/RDMA and its libcuda/ibverbs deps.
os.environ.setdefault("SRPC_STREAM_DISABLE_RDMA", "1")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENV_LOGICAL_CAPACITY = "SGLANG_SIMULATOR_V6D_LOGICAL_CAPACITY"

_DEFAULT_CAPACITY = 1 << 30   # 1 GB (default virtual capacity)

_REAL_MEMORY_BYTES = 1 << 30  # Real mmap size passed to C++ backend (1G)
_REAL_BLOB_SIZE = 4096        # Real allocation per blob in C++ backend (4K)


def _env_int(name: str, default: int) -> int:
    """Read an integer from env var, fall back to *default*."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("[v6d-sim] invalid %s=%r, using default %d", name, raw, default)
        return default


# ---------------------------------------------------------------------------
# VirtualCapacityManager
# ---------------------------------------------------------------------------

class VirtualCapacityManager:
    """Thread-safe virtual capacity tracker driving eviction decisions.

    Tracks declared blob sizes from ``create_blobs(sizes)``.  ``drop_names``
    only carries object names, so freeing pops a FIFO queue of recorded
    sizes — exact when all blobs share one size (the common KV-cache case).
    """

    _instance: Optional["VirtualCapacityManager"] = None
    _singleton_lock = threading.Lock()

    def __init__(
        self,
        total_capacity: int = _DEFAULT_CAPACITY,
    ):
        self._total: int = total_capacity
        self._used: int = 0
        self._lock = threading.Lock()
        self._size_queue: deque[int] = deque()
        logger.info(
            "[V6D Capacity] capacity=%d MB (actual-size tracking)",
            total_capacity // (1 << 20),
        )

    # -- singleton ----------------------------------------------------------

    @classmethod
    def get_or_create(
        cls,
        total_capacity: Optional[int] = None,
    ) -> "VirtualCapacityManager":
        with cls._singleton_lock:
            if cls._instance is None:
                if total_capacity is None:
                    total_capacity = _env_int(
                        _ENV_LOGICAL_CAPACITY, _DEFAULT_CAPACITY
                    )
                cls._instance = cls(total_capacity)
            return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for test isolation)."""
        with cls._singleton_lock:
            cls._instance = None

    # -- allocation tracking ------------------------------------------------

    def try_allocate(self, total_bytes: int) -> bool:
        """Return True if *total_bytes* fits in virtual capacity."""
        with self._lock:
            return self._used + total_bytes <= self._total

    def record_allocate(self, sizes: list[int]):
        """Record successful blob allocation with actual sizes."""
        total = sum(sizes)
        with self._lock:
            self._used += total
            self._size_queue.extend(sizes)
            logger.debug(
                "[V6D Capacity] allocate: +%d blobs (+%d bytes), "
                "used=%d/%d (%.1f%%)",
                len(sizes), total,
                self._used, self._total,
                self._used / self._total * 100 if self._total > 0 else 0,
            )

    def record_free_by_count(self, count: int) -> int:
        """Free *count* blobs by popping the FIFO size queue."""
        if count <= 0:
            return 0
        with self._lock:
            freed = 0
            n = min(count, len(self._size_queue))
            for _ in range(n):
                freed += self._size_queue.popleft()
            # If queue exhausted but more to free, use average of popped
            remaining = count - n
            if remaining > 0 and n > 0:
                avg = freed // n
                freed += remaining * avg
            self._used = max(0, self._used - freed)
            logger.debug(
                "[V6D Capacity] free_by_count: %d blobs (-%d bytes), "
                "used=%d/%d (%.1f%%)",
                count, freed,
                self._used, self._total,
                self._used / self._total * 100 if self._total > 0 else 0,
            )
        return freed

    def set_total_capacity(self, total: int) -> None:
        """Update total capacity at runtime (e.g. from --vineyard-size)."""
        with self._lock:
            old = self._total
            self._total = total
        logger.info(
            "[V6D Capacity] total_capacity: %d MB → %d MB",
            old // (1 << 20), total // (1 << 20),
        )

    def get_usage(self) -> tuple[int, int]:
        """Return (used_bytes, total_bytes) — drives eviction decisions."""
        with self._lock:
            return self._used, self._total


# ---------------------------------------------------------------------------
# C_VineyardPeerHook — argv patch
# ---------------------------------------------------------------------------

class C_VineyardPeerHook(BaseHook):
    """Force 4K page alignment via ``VineyardPeer.__init__`` argv."""

    HOOK_CLASS_NAME = "VineyardPeer"
    HOOK_MODULE_NAME = "v6d.server.peers.vineyard.peer"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def patched_init(self, argc=0, argv=None, *args, **kwargs):

            import v6d.common.transfer as transfer

            def _skip_srpc_transfer(*args, **kwargs):
                logger.info("[v6d-sim] skip SRPC memory registration (4K-aligned mmap)")
                return None

            # Registers the mmap unconditionally; the 4K-aligned mmap would fail
            # with SRPC_STREAM_ERROR_MEM_ALIGN_ERROR.
            transfer.init_srpc_transfer = _skip_srpc_transfer

            if argv is not None:
                # 4K alignment maximises physical block count; a C++ gflag
                # absent from the resolved options, so swap at argv level.
                argv = [
                    "-2M_alignment=false" if a == "-2M_alignment=true" else a
                    for a in argv
                ]
            return original_init(self, argc, argv, *args, **kwargs)

        target.__init__ = patched_init
        logger.info("[v6d-sim] C_VineyardPeerHook installed: 4K alignment")


# ---------------------------------------------------------------------------
# C_VineyardRunnerHook — resolved-options patch
# ---------------------------------------------------------------------------

class C_VineyardRunnerHook(BaseHook):
    """Extract virtual capacity and shrink real memory via the resolved
    options passed to ``VineyardRunner.get(dict_options)``."""

    HOOK_CLASS_NAME = "VineyardRunner"
    HOOK_MODULE_NAME = "v6d.lite.server.vineyard_lite"

    @classmethod
    def hook(cls, target):
        original_get = target.get

        def patched_get(dict_options: dict):
            bulkstore = dict_options.setdefault("bulkstore_spec", {})
            capacity = bulkstore.get("memory_size", 0)

            mgr = VirtualCapacityManager.get_or_create()
            env_capacity = _env_int(_ENV_LOGICAL_CAPACITY, 0)
            if env_capacity > 0:
                mgr.set_total_capacity(env_capacity)
            elif capacity > 0:
                mgr.set_total_capacity(capacity)

            # Configured size only drives virtual accounting; real mmap is 1G.
            bulkstore["memory_size"] = _REAL_MEMORY_BYTES
            # 4K-aligned mmap cannot be SRPC-registered; tracker P2P unaffected.
            dict_options.setdefault("rpc_spec", {})["rpc"] = False
            logger.info(
                "[v6d-sim] memory_size=%d MB → %d MB (real mmap), rpc=off, "
                "virtual_capacity=%d MB",
                capacity // (1 << 20),
                _REAL_MEMORY_BYTES // (1 << 20),
                mgr.get_usage()[1] // (1 << 20),
            )
            return original_get(dict_options)

        target.get = staticmethod(patched_get)
        logger.info(
            "[v6d-sim] C_VineyardRunnerHook installed: "
            "virtual capacity from resolved options, rpc=off"
        )


# ---------------------------------------------------------------------------
# C_TieredVineyardPeerHook — LRU touch on exists() local hit
# ---------------------------------------------------------------------------

class C_TieredVineyardPeerHook(BaseHook):
    """Refresh LRU order on ``exists()`` local hits.

    The real daemon touches the LRU in the read data plane after every
    lookup hit; the sim connector never loads, so hits must touch here or
    eviction runs in insertion order and inflates replay hit rates.
    """

    HOOK_CLASS_NAME = "TieredVineyardPeer"
    HOOK_MODULE_NAME = "v6d.server.peers.tiered_vineyard.peer"

    @classmethod
    def hook(cls, target):
        original_exists = target.exists

        async def patched_exists(self, object_key, peer=None):
            if self._is_in_vineyard(object_key):
                self._touch_access(object_key)
                return True
            return await original_exists(self, object_key, peer)

        target.exists = patched_exists
        logger.info(
            "[v6d-sim] C_TieredVineyardPeerHook installed: "
            "LRU touch on exists() local hit"
        )


# ---------------------------------------------------------------------------
# C_VineyardServerHook — virtual memory capacity control
# ---------------------------------------------------------------------------

class C_VineyardServerHook(BaseHook):
    """Enforce virtual memory capacity on ``VineyardServer``: declared sizes
    are checked/tracked against ``VirtualCapacityManager`` while the C++
    backend allocates only 4K per blob."""

    HOOK_CLASS_NAME = "VineyardServer"
    HOOK_MODULE_NAME = "v6d.lite.server.vineyard_lite"

    @classmethod
    def hook(cls, target):
        try:
            from v6d.common.exceptions import NotEnoughMemoryException
        except ImportError:
            logger.warning(
                "[V6D Capacity] Cannot import NotEnoughMemoryException, "
                "skipping hook"
            )
            return

        logger.info("[V6D Capacity] Installing C_VineyardServerHook")

        # Default capacity here; C_VineyardRunnerHook updates it from the
        # resolved options.
        manager = VirtualCapacityManager.get_or_create()

        original_create_blob = target.create_blob
        original_create_blobs = target.create_blobs
        original_drop_names = target.drop_names

        def patched_get_memory_usage(self) -> tuple[int, int]:
            return manager.get_usage()

        def patched_create_blobs(
            self, sizes: list[int], request_id: str = ""
        ) -> list:
            total_bytes = sum(sizes)
            if not manager.try_allocate(total_bytes):
                used, total = manager.get_usage()
                raise NotEnoughMemoryException(
                    f"Virtual capacity exceeded: used={used}, "
                    f"requested={total_bytes} ({len(sizes)} blobs), "
                    f"total={total}, "
                    f"deficit={used + total_bytes - total}"
                )
            real_sizes = [_REAL_BLOB_SIZE] * len(sizes)
            payloads = original_create_blobs(self, real_sizes, request_id)
            manager.record_allocate(sizes)
            logger.debug(
                "[V6D Capacity] create_blobs: %d blobs, declared=%d, "
                "total_used=%d/%d",
                len(sizes), total_bytes,
                manager.get_usage()[0], manager.get_usage()[1],
            )
            return payloads

        def patched_create_blob(self, size: int, request_id: str = ""):
            if not manager.try_allocate(size):
                used, total = manager.get_usage()
                raise NotEnoughMemoryException(
                    f"Virtual capacity exceeded: used={used}, "
                    f"requested={size} (1 blob), "
                    f"total={total}, "
                    f"deficit={used + size - total}"
                )
            payload = original_create_blob(self, _REAL_BLOB_SIZE, request_id)
            manager.record_allocate([size])
            return payload

        def patched_drop_names(self, names, request_id: str = ""):
            # Release hook is drop_names, not del_blob: eviction DELETE
            # leases carry empty blob_ids.
            if names:
                manager.record_free_by_count(len(names))
            return original_drop_names(self, names, request_id)

        target.get_memory_usage = patched_get_memory_usage
        target.create_blob = patched_create_blob
        target.create_blobs = patched_create_blobs
        target.drop_names = patched_drop_names

        logger.info(
            "[V6D Capacity] C_VineyardServerHook installed: "
            "get_memory_usage, create_blob, create_blobs, drop_names "
            "(capacity=%d MB, declared-size tracking)",
            manager._total // (1 << 20),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def install_v6d_runtime_hooks():
    """Apply all v6d-side simulation hooks; call once before v6d.cli main
    (class hooks must precede the imports that define the target classes).
    """
    install_class_hooks([
        C_VineyardPeerHook,
        C_VineyardRunnerHook,
        C_TieredVineyardPeerHook,
        C_VineyardServerHook,
    ])