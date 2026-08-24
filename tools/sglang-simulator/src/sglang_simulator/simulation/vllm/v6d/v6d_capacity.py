"""v6d daemon-side simulation hooks — run vineyardd on CPU without RDMA.

Applied by ``install_v6d_runtime_hooks()`` at vineyardd startup.  Core idea:
``VirtualCapacityManager`` tracks declared blob sizes against the configured
``--vineyard-size`` and drives ``TieredVineyardPeer`` eviction (AsyncEvictor
polling + ``NotEnoughMemoryException`` → emergency eviction), while physical
allocation stays minimal (1G mmap, 4K per blob).

The read path is the REAL ``TieredVineyardPeer._acquire_tiered_read``
(reached via ``client.get`` from the connector): only the two remote SRPC
entry points in ``v6d.common.transfer`` are stubbed so P2P reads move
metadata only, and a ``[V6D HitSource]`` line per read batch is logged
(port of the production v6d_hitsource_patch; see tmp.out/bugfix/
hit_stats.sh).  Only ``--peer=tiered_vineyard`` daemon mode is supported.

``SGLANG_SIMULATOR_V6D_LOGICAL_CAPACITY`` (bytes) overrides the capacity
derived from ``--vineyard-size``.
"""

from __future__ import annotations

import contextvars
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

# ---------------------------------------------------------------------------
# [V6D HitSource] observability — sim port of the production daemon patch
# ---------------------------------------------------------------------------
#
# Byte-identical port of ``v6d_hitsource_patch`` (tmp.out/bugfix/hit_stats.sh):
# a contextvar counter bag is pushed around
# ``TieredVineyardPeer._acquire_tiered_read`` and bumped from
# ``HitRateStats.record_*_hit``; one line is logged per read batch:
#
#   [V6D HitSource] request_id=<rid> queried=N local=N p2p=N \
#       sharedfs=N tair_kvcm=N miss=N[ error=E]
#
# Only Scope.READ acquires reach ``_acquire_tiered_read``, so store-side
# creates stay silent.  Keys short-circuited by the connector's
# ``_cached_objs`` never reach the daemon and are not counted — same as
# production.  Contextvars are asyncio-task-local, so per-batch attribution
# stays exact even when the daemon interleaves concurrent acquires.

_ENV_HITSOURCE_LOG = "V6D_HITSOURCE_LOG"

_HITSOURCE_COUNTERS = ("local", "p2p", "sharedfs", "tair_kvcm")

# None -> outside a tracked read; dict -> current read's counter bag.
_hit_bag: contextvars.ContextVar = contextvars.ContextVar(
    "v6d_sim_hitsource_bag", default=None)


def _hitsource_enabled() -> bool:
    return os.environ.get(_ENV_HITSOURCE_LOG, "1") not in ("0", "false", "False")


# ---------------------------------------------------------------------------
# Remote data-plane stubs (no real data movement in simulation)
# ---------------------------------------------------------------------------
#
# A P2P read in the real daemon resolves to two SRPC calls into the remote
# peer: ``transfer.get_metas_by_names`` (metadata probe, also used by
# ``VineyardPeer.get_remote_object_sizes`` for the eviction check) and
# ``transfer.load_data`` (the actual blob copy).  Sim daemons run with the
# SRPC engine disabled, so both are stubbed:
#
#   * ``get_metas_by_names`` fabricates the meta locally.  The declared size
#     is the cluster-wide uniform page size learned from local stores, which
#     keeps ``_check_and_evict`` and the follow-up local ``create_blobs``
#     (virtual capacity accounting) exactly as in production.  The tracker
#     lookup in ``_acquire_tiered_read`` (the real metadata P2P probe) is
#     untouched.
#   * ``load_data``/``async_load_data`` are no-ops.  They are only reached by
#     eager fetch strategies; the default lazy strategy never calls them
#     (data load is client-side, and the sim connector never loads).


def _sim_get_metas_by_names(names, endpoint, trace_id=""):
    """Fabricate remote object metas in place of the SRPC meta probe."""
    size = VirtualCapacityManager.get_or_create().page_size()
    if size <= 0:
        # No local store yet: the uniform page size is unknown, so no
        # faithful meta can be fabricated.  Fail like a real remote miss.
        raise KeyError(
            f"[v6d-sim] remote meta probe {endpoint}: page size unknown "
            f"(no local store yet), {len(names)} keys treated as unfetchable"
        )
    from v6d.lite.common.type import ObjectID, ObjectMeta
    blob_meta = {
        "id": ObjectID.invalid_object_id().to_string(),
        "length": size,
        "nbytes": size,
        "transient": False,
        "instance_id": 0,
        "typename": "vineyard::Blob",
    }
    return [
        ObjectMeta.from_dict({
            "size": size,
            # Mirror the field defaults _check_meta_data() applies to local
            # stores — the C++ create_data behind seal() rejects metas that
            # lack them ("Metatree invalid: No 'typename' field").
            "typename": "vineyard::Object",
            "instance_id": 0,
            "transient": False,
            "user_name": "",
            "buffer_num": 1,
            "buffer_0": dict(blob_meta),
        })
        for _ in names
    ]


def _sim_load_data_noop(*args, **kwargs):
    """Skip the remote blob copy (eager strategies only; lazy never calls)."""
    logger.debug("[v6d-sim] skip remote blob data transfer (no data plane)")
    return None


def _sim_async_load_data_noop(*args, **kwargs):
    return []


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
        # First declared blob size seen, kept sticky. vLLM derives one uniform
        # page size per deployment (worker.py pads the mamba spec to the
        # attention page size), so this is the footprint a remote-fetched
        # object takes locally. Sticky rather than "last seen" so it stays
        # valid even if a remote hit arrives before any local store.
        self._page_size: int = 0
        self._page_size_warned: bool = False
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
            for sz in sizes:
                if self._page_size == 0:
                    self._page_size = sz
                    logger.info(
                        "[V6D Capacity] page_size learned from connector: "
                        "%d bytes (%.3f MiB)", sz, sz / (1 << 20),
                    )
                elif sz != self._page_size and not self._page_size_warned:
                    # vLLM is expected to use one uniform page size; a second
                    # size means either a config change mid-run or a genuinely
                    # heterogeneous layout, and remote-fetch admission would
                    # then charge the wrong footprint.
                    self._page_size_warned = True
                    logger.warning(
                        "[V6D Capacity] non-uniform blob size: got %d bytes, "
                        "page_size was %d. Remote-fetch admission assumes a "
                        "uniform page size and may misaccount.", sz,
                        self._page_size,
                    )
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

    def page_size(self) -> int:
        """Uniform v6d object size declared by the connector (0 if none seen).

        Derived, never hardcoded. The connector declares
        ``effective_ranks * num_layers * page_size_bytes`` per object (see the
        "V6D object layout" log in v6d_object_connector), and all three
        factors come from the model/runtime config, so switching model or
        dtype updates this automatically.

        The declared size is TP-invariant: ``page_size_bytes`` is already the
        per-rank post-shard value, so multiplying it back by
        ``effective_ranks`` conserves the total. tp=2 (2 x 12 x 2,146,304) and
        tp=1 (1 x 12 x 4,292,608) both declare 51,511,296 bytes, which is why
        production and simulation agree on capacity without any scaling.

        This is the *capacity* footprint of one object. It is not the per-rank
        byte count a single worker moves over PCIe -- that is
        ``num_layers * page_size_bytes``, half of this under tp=2.
        """
        with self._lock:
            return self._page_size

    def last_blob_size(self) -> int:
        """Deprecated alias for :meth:`page_size`."""
        return self.page_size()


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
# C_TieredVineyardPeerHook — real read path + hit-source stats + data stubs
# ---------------------------------------------------------------------------

class C_TieredVineyardPeerHook(BaseHook):
    """Keep the real ``_acquire_tiered_read`` on the CPU read path.

    Two pieces, both daemon-process only (this class is defined only where
    the tiered peer module is imported, i.e. the daemon):

    1. Observability: wrap ``_acquire_tiered_read`` with a contextvar
       counter bag and log one ``[V6D HitSource]`` line per read batch — a
       direct port of the production ``v6d_hitsource_patch`` (hit_stats.sh).
    2. Data-plane stubs: replace the two remote SRPC entry points in
       ``v6d.common.transfer`` so P2P reads move only metadata.  Everything
       else on the read path (LRU touch, lease pin, tracker probe, eviction
       check, local admit + re-ANNOUNCE, ``record_*_hit``) runs unmodified.
    """

    HOOK_CLASS_NAME = "TieredVineyardPeer"
    HOOK_MODULE_NAME = "v6d.server.peers.tiered_vineyard.peer"

    @classmethod
    def hook(cls, target):
        original_acquire_tiered_read = target._acquire_tiered_read

        async def wrapped_acquire_tiered_read(self, *args, **kwargs):
            if not _hitsource_enabled():
                return await original_acquire_tiered_read(self, *args, **kwargs)
            object_keys = args[0] if args else kwargs.get("object_keys") or ()
            request_id = kwargs.get("request_id")
            bag = dict.fromkeys(_HITSOURCE_COUNTERS, 0)
            token = _hit_bag.set(bag)
            error = None
            try:
                return await original_acquire_tiered_read(self, *args, **kwargs)
            except Exception as exc:
                error = type(exc).__name__
                raise
            finally:
                _hit_bag.reset(token)
                queried = len(object_keys)
                if queried:
                    hits = sum(bag.values())
                    logger.info(
                        "[V6D HitSource] request_id=%s queried=%d local=%d p2p=%d "
                        "sharedfs=%d tair_kvcm=%d miss=%d%s",
                        request_id, queried, bag["local"], bag["p2p"],
                        bag["sharedfs"], bag["tair_kvcm"],
                        max(queried - hits, 0),
                        f" error={error}" if error else "",
                    )

        target._acquire_tiered_read = wrapped_acquire_tiered_read

        # The tiered peer module already imports v6d.common.transfer (through
        # v6d.server.peers.vineyard.peer), so this import is a sys.modules
        # lookup; the stubs apply process-wide, which is fine: the client
        # side never transfers data either (fetch is stubbed there).
        import v6d.common.transfer as transfer
        transfer.get_metas_by_names = _sim_get_metas_by_names
        transfer.load_data = _sim_load_data_noop
        transfer.async_load_data = _sim_async_load_data_noop

        logger.info(
            "[v6d-sim] C_TieredVineyardPeerHook installed: real "
            "_acquire_tiered_read + [V6D HitSource] stats + remote "
            "data-plane stubs (meta fabricate / load no-op)"
        )


# ---------------------------------------------------------------------------
# C_HitRateStatsHook — feed the per-read hit-source bag
# ---------------------------------------------------------------------------

class C_HitRateStatsHook(BaseHook):
    """Bump the contextvar bag from ``HitRateStats.record_*_hit``.

    Direct port of the production ``v6d_hitsource_patch`` stats hook; the
    daemon's own classification (local / remote / sharedfs / tair_kvcm) is
    reused unchanged, so the sim's split is defined by the same code as
    production.
    """

    HOOK_CLASS_NAME = "HitRateStats"
    HOOK_MODULE_NAME = "v6d.server.peers.tiered_vineyard.stats"

    @classmethod
    def hook(cls, target):
        def wrap(name, counter):
            original = getattr(target, name)

            def wrapped(self, size: int = 0):
                bag = _hit_bag.get()
                if bag is not None:
                    bag[counter] += 1
                return original(self, size)

            setattr(target, name, wrapped)

        wrap("record_local_vineyard_hit", "local")
        wrap("record_remote_vineyard_hit", "p2p")
        wrap("record_sharedfs_hit", "sharedfs")
        wrap("record_tair_kvcm_hit", "tair_kvcm")
        logger.info(
            "[v6d-sim] C_HitRateStatsHook installed: record_*_hit -> "
            "[V6D HitSource] bag"
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
        C_HitRateStatsHook,
        C_VineyardServerHook,
    ])