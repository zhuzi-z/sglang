"""V6D Object Manager hooks for native control-plane P2P ownership simulation.

In native V6D control-plane mode this module keeps the real vLLM
V6dObjectManager/V6dObjectConnectorScheduler classes in the request path.
The lookup path is the REAL one — ``lookup``/``get_key`` issue
``client.get()`` reads that run the daemon's real
``TieredVineyardPeer._acquire_tiered_read`` (LRU touch, lease pin, tracker
P2P probe, tiered fetch admission, re-ANNOUNCE).  Only the data plane is
stubbed:

1. Daemon side (v6d_capacity.py): remote SRPC meta/data transfer replaced
   by fabricated metas + no-op loads.
2. Connector side (here): ``V6dObjectFetchHelper.start_fetch`` skips
   BlockReceiver (SRPC -> GPU) and only completes the lazy-seal protocol.
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


class _DeadClient:
    """Stand-in for a v6d client whose daemon never came up.

    Keeps the real read path alive instead of crashing on
    ``assert self.client is not None``: every read misses (``get`` returns
    None, exactly like a daemon that has no data), so the scheduler simply
    observes 0 external hits.  Store calls are routed to the manager's
    local-key fallback by ``override_batch_allocate``.
    """

    def __init__(self, url):
        self._url = url

    def get(self, object_key, **kwargs):
        return None

    async def async_get(self, object_key, **kwargs):
        return None



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
            except Exception as e:
                # Non-None stand-in: the real lookup path asserts
                # ``self.client is not None``; a _DeadClient misses every
                # read, which is exactly what a dead-but-connected daemon
                # looks like to the scheduler.
                self.client = _DeadClient(getattr(self, "_v6d_url", "?"))
                logger.warning(
                    "[V6D P2P] Manager group=%d cannot connect to v6d daemon "
                    "at %s: %s. Cross-node lookup will return 0 hits.",
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

        target.__init__ = override_init

        # NOTE: lookup/async_lookup/get_key/async_get_key and
        # prepare_batch_allocate are NOT overridden — the real connector
        # methods run unchanged.  ``client.get()`` is metadata-only RPC plus
        # an Object wrapper (no payload movement), so it is CPU-safe; the
        # daemon side is stubbed at the transfer layer instead
        # (v6d_capacity.py).  This also retires the sim's P0-A/P0-B fixes:
        # the real ``prepare_batch_allocate`` already skips ``_cached_objs``
        # and the real ``_process_lookup``/``get_key`` already register
        # holders on the short-circuit path.

        # ---- Override batch_allocate to record ownership ----
        if hasattr(target, 'batch_allocate'):
            original_batch_allocate = target.batch_allocate

            def override_batch_allocate(self, block_hashes, size, shape,
                                        dtype, request_id=None):
                """Allocate simulated V6D keys without v6d client data-plane."""
                client = getattr(self, "client", None)
                if client is not None and not isinstance(client, _DeadClient):
                    return original_batch_allocate(
                        self, block_hashes, size, shape, dtype,
                        request_id=request_id)
                logger.warning(
                    "[V6D RPC Bypass] batch_allocate: client is None "
                    "(daemon not connected), using local key allocation. "
                    "group=%s req=%s",
                    self._group_id, request_id)
                result = {}
                for h in block_hashes:
                    key = self._make_key(h)
                    self._pending_objs[h] = key
                    result[h] = key
                logger.info(
                    f"[V6D RPC Bypass] batch_allocate: {len(result)} blocks "
                    f"group={self._group_id} req={request_id}")
                return result

            target.batch_allocate = override_batch_allocate

        logger.info("[V6D Hijack] V6dObjectManager hook installed "
                    "(real v6d P2P path)")


class C_V6dObjectFetchHelperHook(BaseHook):
    """Replace the BlockReceiver data plane with the bare seal protocol.

    Real ``start_fetch`` hands the hit objects to ``BlockReceiver``, which
    SRPC-loads remote/sharedfs payloads into the local blobs and then seals
    the lazy placeholders (``set_seal_target`` + ``Object.complete()`` per
    external obj).  In simulation the local blobs are 4K stubs that already
    occupy the right virtual capacity, so only the seal protocol runs — no
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


class C_V6dObjectConnectorSchedulerHook(BaseHook):
    """Hook scheduler cross-group allocation to avoid real v6d client.create.

    ``get_num_new_matched_tokens``/``async_get_num_new_matched_tokens`` are
    NOT overridden: the real scheduler lookup (batch ``client.get`` +
    ``_fetch_intersection_blocks`` + promote) runs unchanged, with the data
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

        original_cross_group_batch_allocate = getattr(
            target, "_cross_group_batch_allocate", None
        )

        def override_cross_group_batch_allocate(
            self,
            group_candidates,
            request_id=None,
        ):
            # Prefer the upstream merged path: all groups collected into ONE
            # client.create() call (the daemon then logs a single
            # BATCH_CREATE n=6 per request, matching the real server).
            client = None
            if getattr(self, "managers", None):
                client = next(iter(self.managers.values())).client
            if (client is not None and not isinstance(client, _DeadClient)
                    and original_cross_group_batch_allocate is not None):
                result = original_cross_group_batch_allocate(
                    self, group_candidates, request_id=request_id
                )
                logger.debug(
                    "[V6D RPC Bypass] cross_group_batch_allocate(merged): "
                    "groups=%s blobs=%d req=%s",
                    sorted(result),
                    sum(len(v) for v in result.values()),
                    request_id,
                )
                return result

            # Fallback: daemon not connected — per-group loop so the
            # manager-level batch_allocate bypass can hand out local keys.
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
                "[V6D RPC Bypass] cross_group_batch_allocate(fallback): "
                "groups=%s req=%s",
                sorted(result),
                request_id,
            )
            return result

        target._cross_group_batch_allocate = override_cross_group_batch_allocate

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


# NOTE (fidelity): the former C_HybridSchedulerHook (_saving -> _saved flush
# before each HybridScheduler.step) has been removed.  It pre-emptively tore
# down _saving so the later _do_save_done found no state (try_advance -> None)
# and silently skipped backend.async_cleanup — leaving v6d objects unsealed
# and mamba protected blocks unreleased.  Save completion is now signalled
# through the real channel (mark_backend_save_done, equivalent to the
# worker's _SAVE_DONE_REQ RPC) from C_HybridConnectorHook in v6d_backend.py,
# so _saved/_try_teardown_save and async_cleanup run on the native path.
