"""V6D Object Manager hooks for native control-plane P2P ownership simulation.

In native V6D control-plane mode this module keeps the real vLLM
V6dObjectManager/V6dObjectConnectorScheduler classes in the request path.
The lookup path is the REAL one — ``lookup``/``get_key`` issue
``client.get()`` reads that run the daemon's real
``TieredVineyardPeer._acquire_tiered_read`` (LRU touch, lease pin, tracker
P2P probe, tiered fetch admission, re-ANNOUNCE).  Only the data plane is
stubbed:

1. Daemon side (v6d_capacity.py): remote payload copies (load_data)
   no-oped; the SRPC meta probe runs for real over TCP.
2. Connector side (here): ``V6dObjectFetchHelper.start_fetch`` skips
   BlockReceiver (SRPC -> GPU) and only completes the lazy-seal protocol,
   and ``ClientV6dMmapManager._create_mmap`` skips the C++ ``init_mmap``
   (which materializes the whole bulkstore in tmpfs — fatal with a large
   sparse arena): the sim never dereferences client-side blob memory.
3. Worker side (v6d_worker.py): load/store handlers stay CPU no-ops.

NOTE: The legacy etcd-backed V6dBlockOwnershipTracker (removed) and
V6DCacheStorage have been removed. The P2P path (client.get via Redis
tracker) is the only supported cross-node matching mechanism.
"""

from __future__ import annotations
import os
import sys

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger()


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
        def override_async_connect(self) -> None:
            """Connect to the real v6d daemon.

            Same semantics as the production ``_async_connect``: on failure
            it logs and returns with ``client`` unset (the first lookup then
            fails loudly, exactly like production with a dead daemon).  The
            only difference is that ``_on_connected`` is skipped — it starts
            schedrpcserver, which conflicts with
            LAZY_INITIALIZE_KV_TRANSFER_OUTSIDE_VLLM.
            """
            try:
                _mod = sys.modules.get(target.__module__)
                _fn = getattr(_mod, "_connect_v6d_with_retry", None)
                if _fn is None:
                    raise RuntimeError("_connect_v6d_with_retry not found")
                self.client = _fn(self._v6d_url)
                logger.info(
                    "[V6D P2P] Manager group=%d connected to v6d daemon "
                    "at %s (real P2P path)",
                    self._group_id, self._v6d_url,
                )
            except Exception:
                logger.exception(
                    "[V6D P2P] Manager group=%d cannot connect to v6d "
                    "daemon at %s",
                    self._group_id, getattr(self, "_v6d_url", "?"))
                return
            if self._v6d_backend is not None:
                self._v6d_backend.mark_manager_connected(self._group_id)

        target._async_connect = override_async_connect

        # NOTE: lookup/async_lookup/get_key/async_get_key,
        # prepare_batch_allocate and batch_allocate are NOT overridden — the
        # real connector methods run unchanged.  ``client.get()`` is
        # metadata-only RPC plus an Object wrapper (no payload movement), so
        # it is CPU-safe; the daemon side is stubbed at the transfer layer
        # instead (v6d_capacity.py).  This also retires the sim's P0-A/P0-B
        # fixes: the real ``prepare_batch_allocate`` already skips
        # ``_cached_objs`` and the real ``_process_lookup``/``get_key``
        # already register holders on the short-circuit path.

        logger.info("[V6D Hijack] V6dObjectManager hook installed "
                    "(real v6d P2P path)")


class C_V6dObjectFetchHelperHook(BaseHook):
    """Replace the BlockReceiver data plane with the bare seal protocol.

    Real ``start_fetch`` hands the hit objects to ``BlockReceiver``, which
    SRPC-loads remote/sharedfs payloads into the local blobs and then seals
    the lazy placeholders (``set_seal_target`` + ``Object.complete()`` per
    external obj).  In simulation the local blobs are sparse-arena stubs
    whose pages are never touched, so only the seal protocol runs — no
    bytes move.  Returning None makes the real ``wait()`` report zero tier
    counts; ``_promote_fetched_objs`` then runs unchanged.
    """

    HOOK_CLASS_NAME = "V6dObjectFetchHelper"
    HOOK_MODULE_NAME = (
        "vllm.distributed.kv_transfer.kv_connector.v1.v6d_object_connector"
    )

    @classmethod
    def hook(cls, target):
        def override_start_fetch(self, objs):
            objs = [o for o in (objs or ()) if o is not None]
            if not objs:
                return None
            by_lease: dict = {}
            for o in objs:
                lease_id = getattr(o, "_lease_id", None)
                if lease_id:
                    by_lease.setdefault(lease_id, []).append(o)
            for lease_id, group in by_lease.items():
                pending = [
                    o for o in group
                    if getattr(o, "meta", {}).get("location", "local") != "local"
                ]
                if not pending:
                    continue
                try:
                    # Narrow the lease's seal target to the fetched subset —
                    # the same lease may also cover probe-only objects that
                    # were dropped after mamba boundary selection.
                    pending[0].set_seal_target(len(pending))
                except Exception as e:
                    logger.warning(
                        "[V6D Hijack] sim fetch: set_seal_target failed "
                        "lease=%s: %r; %d placeholder(s) left to the "
                        "daemon's release/discard path",
                        lease_id, e, len(pending))
                    continue
                for o in pending:
                    try:
                        o.complete()
                    except Exception as e:
                        logger.warning(
                            "[V6D Hijack] sim fetch: complete failed "
                            "key=%s: %r", getattr(o, "key", "?"), e)
            return None

        target.start_fetch = override_start_fetch
        logger.info(
            "[V6D Hijack] V6dObjectFetchHelper hook installed "
            "(seal-only fetch, no data movement)"
        )


class C_V6dMmapManagerHook(BaseHook):
    """Skip the client-side bulkstore mmap (which populates the arena).

    Real ``_create_mmap`` calls the C++ ``init_mmap(fd, map_size)`` that
    materializes the entire bulkstore mapping (with a sparse 500G arena
    this hangs the connect and inflates tmpfs/page-cache until OOM).
    The sim never dereferences client-side blob memory — fetch is stubbed
    and ``VineyardBlobView`` only does pointer arithmetic — so the hook
    keeps the real connection handshake (``_vineyard_connect``: its socket
    is the refcounted keepalive) but skips ``init_mmap``/``init_srpc`` and
    reports a zero base address with the real map size.
    """

    HOOK_CLASS_NAME = "ClientV6dMmapManager"
    HOOK_MODULE_NAME = "v6d.client.peers.vineyard.mmap_manager"

    @classmethod
    def hook(cls, target):
        from v6d.client.peers.vineyard.mmap_manager import MmapInfo

        def override_create_mmap(self, socket_path: str, is_lazy_strategy: bool):
            from v6d.common.transfer import _vineyard_connect

            fd, map_size, offset, socket = _vineyard_connect(socket_path)
            os.close(fd)  # no mmap follows; nothing dereferences blob data
            logger.info(
                "[V6D Hijack] sim mmap: skip bulkstore mmap for %s "
                "(map_size=%d MB, no client data-plane dereference)",
                socket_path, map_size // (1 << 20))
            return MmapInfo(
                socket_path=socket_path,
                socket=socket,
                fd=-1,
                base_addr=0,
                map_size=map_size,
                refcount=1,
            )

        target._create_mmap = override_create_mmap
        logger.info(
            "[V6D Hijack] ClientV6dMmapManager hook installed "
            "(no bulkstore mmap population)"
        )


class C_V6dObjectConnectorSchedulerHook(BaseHook):
    """Hook scheduler request-finished for CPU no-op store completion.

    ``get_num_new_matched_tokens``/``async_get_num_new_matched_tokens`` and
    ``_cross_group_batch_allocate`` are NOT overridden: the real scheduler
    lookup (batch ``client.get`` + ``_fetch_intersection_blocks`` + promote)
    and the real merged cross-group create run unchanged, with the data
    plane removed by C_V6dObjectFetchHelperHook.
    """

    HOOK_CLASS_NAME = "V6dObjectConnectorScheduler"
    HOOK_MODULE_NAME = (
        "vllm.distributed.kv_transfer.kv_connector.v1.v6d_object_connector"
    )

    @classmethod
    def hook(cls, target):
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
            logger.debug(
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

        logger.info(
            "[V6D Hijack] V6dObjectConnectorScheduler hook installed "
            "(CPU no-op store completion)"
        )


# NOTE (fidelity): the former C_HybridSchedulerHook (_saving -> _saved flush
# before each HybridScheduler.step) has been removed.  It pre-emptively tore
# down _saving so the later _do_save_done found no state (try_advance -> None)
# and silently skipped backend.async_cleanup — leaving v6d objects unsealed
# and mamba protected blocks unreleased.  Save completion is now signalled
# through the real channel (mark_backend_save_done, equivalent to the
# worker's _SAVE_DONE_REQ RPC) from C_HybridConnectorHook in v6d_backend.py,
# so _saved/_try_teardown_save and async_cleanup run on the native path.
