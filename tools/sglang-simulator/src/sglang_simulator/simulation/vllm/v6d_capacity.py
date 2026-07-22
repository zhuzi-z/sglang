"""
V6D Virtual Capacity Manager — C_VineyardServerHook + runtime patches.

This module consolidates ALL v6d-side simulation hooks needed to run a
vineyardd process on CPU without RDMA:

1. **SRPC bypass** — ``init_srpc*`` → no-op, ``init_transfer_engine_client``
   → ``init_mmap`` only (shared-memory IPC, no RDMA).
2. **VineyardPeer rpc=false + 4K alignment + capacity extraction** —
   inject ``-rpc=false`` and replace ``-2M_alignment=true`` with
   ``-2M_alignment=false`` in argv.  Also extracts ``--vineyard-size``
   from argv to initialise virtual capacity.  The C++ barex SRPC init
   is skipped while P2P tracker (Redis) discovery is preserved.
3. **Virtual capacity control** — ``C_VineyardServerHook`` patches four
   ``VineyardServer`` methods.  ``get_memory_usage`` returns virtual
   ``(used, total)``; ``create_blobs`` / ``create_blob`` check virtual
   capacity using **actual blob sizes** passed from the interface call
   (not a fixed per-blob size); ``drop_names`` decrements via a FIFO
   size queue (count-based) or ``record_free_by_sizes`` (exact sizes).
   The real C++ backend still handles actual memory allocation and
   freeing underneath.

Call ``install_v6d_runtime_hooks()`` once at vineyardd startup (before
``v6d.cli.cli.main()``) to apply all three.

Capacity model:

  ``VirtualCapacityManager`` tracks a **virtual** capacity that drives
  eviction decisions.  Total capacity is resolved at runtime from the
  ``--vineyard-size`` CLI argument (e.g. ``--vineyard-size=500G``),
  extracted by ``C_VineyardPeerHook`` when ``VineyardPeer.__init__`` is
  called with ``argv``.  No env var or programmatic parameter is needed.

  Priority (for backward compat with tests):
    1. ``SGLANG_SIMULATOR_V6D_LOGICAL_CAPACITY`` env var (override)
    2. ``--vineyard-size`` from argv (primary source)
    3. Default: 1 GB

  Allocation uses **actual blob sizes** from the ``create_blobs(sizes)``
  / ``create_blob(size)`` interface calls — no fixed ``alloc_size``.
  Release via ``drop_names`` uses a FIFO size queue (count-based); for
  exact-size release, ``record_free_by_sizes()`` is available.
  ``set_total_capacity()`` allows runtime reconfiguration.

  The hook calls the **original** ``create_blobs`` / ``drop_names`` after
  the virtual check, so the real C++ backend still allocates and frees
  real memory.

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

  ``SGLANG_SIMULATOR_V6D_LOGICAL_CAPACITY``
      Optional override for virtual total capacity in bytes.  When unset,
      capacity is derived from ``--vineyard-size``.  Default: 1 GB.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections import deque
from typing import Optional

from sglang_simulator.hook import BaseHook

logger = logging.getLogger("sglang_simulator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENV_ENABLE = "SGLANG_SIMULATOR_V6D_CAPACITY_CONTROL"
_ENV_LOGICAL_CAPACITY = "SGLANG_SIMULATOR_V6D_LOGICAL_CAPACITY"

_DEFAULT_CAPACITY = 1 << 30   # 1 GB (default virtual capacity)

_REAL_MEMORY_ARG = "1G"       # Real mmap size passed to C++ backend
_REAL_BLOB_SIZE = 4096        # Real allocation per blob in C++ backend (4K)

_SIZE_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*([KMGTP]?B?)$", re.IGNORECASE)
_SIZE_MULTIPLIERS = {
    "": 1, "B": 1,
    "K": 1 << 10, "KB": 1 << 10,
    "M": 1 << 20, "MB": 1 << 20,
    "G": 1 << 30, "GB": 1 << 30,
    "T": 1 << 40, "TB": 1 << 40,
    "P": 1 << 50, "PB": 1 << 50,
}


def _parse_size_str(raw: str) -> Optional[int]:
    """Convert a size string like '500G' or '256MB' to bytes."""
    m = _SIZE_PATTERN.match(raw.strip())
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).upper()
    return int(val * _SIZE_MULTIPLIERS.get(unit, 1))


def _vineyard_size_from_argv(argv: Optional[list[str]]) -> Optional[int]:
    """Read vineyard size from argv (read-only, does not modify argv).

    v6d's ``serve_command`` already parses ``--vineyard-size`` via argparse
    and converts it to ``-size=<value>`` in the C-style argv passed to
    ``VineyardPeer.__init__``.  We look for ``-size=`` here — no need to
    re-implement v6d's argument parsing.
    """
    if not argv:
        return None
    for arg in argv:
        if arg.startswith("-size="):
            return _parse_size_str(arg.split("=", 1)[1])
    return None


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
# VirtualCapacityManager — virtual capacity tracker (actual-size tracking)
# ---------------------------------------------------------------------------

class VirtualCapacityManager:
    """Thread-safe virtual memory capacity tracker.

    Tracks **actual** blob sizes from ``create_blobs(sizes)`` — no fixed
    per-blob size.  Capacity (total) is configurable via constructor or
    env var.  The eviction logic (``AsyncEvictor``, ``TieredVineyardPeer``)
    queries this layer via ``get_memory_usage()`` to make eviction decisions.

    For ``drop_names``, which receives object names (not sizes), a FIFO
    queue of blob sizes is maintained.  When objects are created, their
    sizes are pushed; when dropped, sizes are popped from the front.
    This is exact when all blobs are the same size (the common KV-cache
    case) and approximately correct for LRU eviction otherwise.

    The hook still calls the original ``create_blobs`` / ``drop_names``
    (C++ backend) for real memory management.  This class only tracks
    the virtual counter that drives eviction thresholds.
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
    def get_instance(cls) -> Optional["VirtualCapacityManager"]:
        return cls._instance

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
        """Free *count* blobs by popping from the FIFO size queue.

        Used during eviction: the DELETE lease has empty ``blob_ids``, so
        ``del_blob`` receives 0 IDs.  ``drop_names`` is always called with
        the object keys, so we decrement by object count here.

        When all blobs are the same size (common KV-cache case), the FIFO
        queue gives exact results regardless of eviction order.
        """
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

    def record_free_by_sizes(self, sizes: list[int]) -> int:
        """Free blobs by actual sizes passed from the interface call.

        When the caller knows the actual sizes of freed blobs (e.g. from a
        name→size mapping), use this instead of ``record_free_by_count``
        for exact accounting.  Entries are popped from the FIFO queue to
        keep ``blob_count`` consistent; the *sizes* argument controls the
        byte count, not the queue.
        """
        if not sizes:
            return 0
        total = sum(sizes)
        with self._lock:
            n = min(len(sizes), len(self._size_queue))
            for _ in range(n):
                self._size_queue.popleft()
            self._used = max(0, self._used - total)
            logger.debug(
                "[V6D Capacity] free_by_sizes: %d blobs (-%d bytes), "
                "used=%d/%d (%.1f%%)",
                len(sizes), total,
                self._used, self._total,
                self._used / self._total * 100 if self._total > 0 else 0,
            )
        return total

    def set_total_capacity(self, total: int) -> None:
        """Update total capacity at runtime.

        Useful when the capacity is determined after startup (e.g. from
        the v6d serve ``--vineyard-size`` argument parsed at run time).
        """
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

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "used": self._used,
                "total": self._total,
                "blob_count": len(self._size_queue),
                "usage_percent": (
                    self._used / self._total if self._total > 0 else 0.0
                ),
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
# 2a. C_VineyardPeerHook — inject -rpc=false + 4K alignment
# ---------------------------------------------------------------------------

class C_VineyardPeerHook(BaseHook):
    """Hook ``VineyardPeer`` to inject ``-rpc=false``, 4K alignment, and
    extract ``--vineyard-size`` for virtual capacity initialization.

    Skips the C++ SRPC barex initialization while preserving P2P
    tracker (Redis) discovery capability.  Replaces ``-2M_alignment=true``
    with ``-2M_alignment=false`` so the C++ mmap uses 4K page alignment,
    maximising the number of physical blocks available.  This ensures
    virtual capacity (our hook) is always the eviction bottleneck, not
    real memory.

    **Capacity extraction**: ``--vineyard-size`` is read from ``argv``
    during ``VineyardPeer.__init__`` and used to set the
    ``VirtualCapacityManager`` total capacity.  This happens at runtime
    (when the peer instance is created), before any ``create_blobs``
    calls.  If ``SGLANG_SIMULATOR_V6D_LOGICAL_CAPACITY`` env var is set,
    it takes priority (for tests where virtual ≠ real memory).
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
                # Inject -rpc=false (skip C++ barex SRPC init)
                if not any(a.startswith("-rpc=") for a in patched_argv):
                    patched_argv.append("-rpc=false")
                    argc = len(patched_argv)
                # Replace 2M alignment with 4K for maximum block count
                for i, a in enumerate(patched_argv):
                    if a == "-2M_alignment=true":
                        patched_argv[i] = "-2M_alignment=false"
                        logger.info("[v6d-sim] replaced -2M_alignment=true → false (4K)")

                # Extract -size=<value> for virtual capacity, then replace
                # with 1G so the C++ backend doesn't mmap the full virtual
                # size.  VirtualCapacityManager tracks the user's intended
                # capacity (e.g. 500G); real memory stays at 1G.
                for i, a in enumerate(patched_argv):
                    if a.startswith("-size="):
                        raw_size = a.split("=", 1)[1]
                        capacity = _parse_size_str(raw_size)
                        mgr = VirtualCapacityManager.get_or_create()
                        env_val = os.environ.get(_ENV_LOGICAL_CAPACITY, "").strip()
                        if env_val:
                            try:
                                mgr.set_total_capacity(int(env_val))
                            except ValueError:
                                pass
                        elif capacity and capacity > 0:
                            mgr.set_total_capacity(capacity)
                        patched_argv[i] = f"-size={_REAL_MEMORY_ARG}"
                        logger.info(
                            "[v6d-sim] -size=%s → 1G (real mmap), "
                            "virtual_capacity=%d MB",
                            raw_size,
                            capacity // (1 << 20) if capacity else 0,
                        )
                        break

            return original_init(
                self, argc, patched_argv, tracker_url,
                tracker_key_prefix, lazy_load,
            )

        target.__init__ = patched_init
        logger.info("[v6d-sim] C_VineyardPeerHook installed: rpc=false, 4K alignment, vineyard-size extraction")


# ---------------------------------------------------------------------------
# 3. C_VineyardServerHook — virtual memory capacity control
# ---------------------------------------------------------------------------

def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


class C_VineyardServerHook(BaseHook):
    """Hook ``VineyardServer`` to enforce virtual memory capacity.

    Patches four methods.  Only the virtual capacity counter is
    simulated; the original C++ backend still handles real memory:

      - ``get_memory_usage()`` → return virtual ``(used, total)``
      - ``create_blobs(sizes)`` → check virtual capacity using **actual
        sizes** from the interface call, raise
        ``NotEnoughMemoryException`` if exceeded, else call original
        (C++ allocates real memory) and record actual sizes
      - ``create_blob(size)`` → same for single blob
      - ``drop_names(names)`` → decrement virtual counter via FIFO size
        queue (count-based), then call original (C++ frees real memory).
        For exact-size release, callers can use
        ``manager.record_free_by_sizes()`` instead.

    ``drop_names`` is used instead of ``del_blob`` because during eviction
    the DELETE lease has empty ``blob_ids``, causing ``del_blob`` to receive
    0 IDs.  ``drop_names`` is always called with the object keys.

    Total capacity is initialised from ``--vineyard-size`` by
    ``C_VineyardPeerHook`` at runtime (when ``VineyardPeer.__init__`` is
    called with ``argv``).  At hook-install time the manager is created
    with a default; the capacity is updated before any allocation calls.
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

        logger.info("[V6D Capacity] Installing C_VineyardServerHook")

        # Create manager with default capacity; C_VineyardPeerHook will
        # update it from --vineyard-size at runtime (VineyardPeer.__init__).
        manager = VirtualCapacityManager.get_or_create()

        # Save original methods
        original_create_blob = target.create_blob
        original_create_blobs = target.create_blobs
        original_drop_names = target.drop_names

        # -- patched methods -------------------------------------------------

        def patched_get_memory_usage(self) -> tuple[int, int]:
            """Return virtual (used, total) — drives eviction decisions."""
            return manager.get_usage()

        def patched_create_blobs(
            self, sizes: list[int], request_id: str = ""
        ) -> list:
            """Check virtual capacity with actual sizes; allocate 4K per blob in C++."""
            total_bytes = sum(sizes)
            if not manager.try_allocate(total_bytes):
                used, total = manager.get_usage()
                raise NotEnoughMemoryException(
                    f"Virtual capacity exceeded: used={used}, "
                    f"requested={total_bytes} ({len(sizes)} blobs), "
                    f"total={total}, "
                    f"deficit={used + total_bytes - total}"
                )
            # C++ backend allocates 4K per blob (not the virtual size)
            real_sizes = [_REAL_BLOB_SIZE] * len(sizes)
            payloads = original_create_blobs(self, real_sizes, request_id)
            manager.record_allocate(sizes)
            return payloads

        def patched_create_blob(self, size: int, request_id: str = ""):
            """Check virtual capacity with actual size; allocate 4K in C++."""
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
            """Decrement virtual counter via FIFO size queue.

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
            "(capacity=%d MB, actual-size tracking)",
            manager._total // (1 << 20),
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
      1. C_VineyardPeerHook (inject ``-rpc=false`` + ``-2M_alignment=false``
         via class hook; also extracts ``--vineyard-size`` from argv for
         virtual capacity initialization)
      2. SRPC bypass (``patch_srpc_bypass``)

    Conditionally applies (when ``SGLANG_SIMULATOR_V6D_CAPACITY_CONTROL=1``):
      3. C_VineyardServerHook (virtual memory capacity control)

    Virtual capacity is resolved at runtime from ``--vineyard-size`` in
    the v6d serve CLI args (extracted in ``C_VineyardPeerHook`` when
    ``VineyardPeer.__init__`` is called).  No env var or programmatic
    parameter is required for capacity.

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
