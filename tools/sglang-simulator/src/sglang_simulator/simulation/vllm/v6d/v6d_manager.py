"""
V6D Object Manager hooks for native control-plane P2P ownership simulation.

In native V6D control-plane mode this module keeps the real vLLM
V6dObjectManager/V6dObjectConnectorScheduler classes in the request path, but
bypasses the unavailable CPU-only data plane:

1. Tag each manager with `_SIM_V6D_ACTIVE_WORKER_ID`.
2. Resolve lookup hits through real v6d daemon P2P path (client.exists →
   Redis tracker), without attempting SRPC/CUDA data transfer.
3. Bypass scheduler-side `client.create()` while preserving cross-group
   allocation semantics.

NOTE: The legacy etcd-backed # V6dBlockOwnershipTracker (removed) and V6DCacheStorage
have been removed. The P2P path (client.exists via Redis tracker) is the
only supported cross-node matching mechanism.
"""

from __future__ import annotations
import asyncio
import os
import sys

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger()


def _sim_client_exists(client, key: str, request_id=None) -> bool:
    """exists() RPC that threads request_id to the sim-patched daemon.

    The stock ``ExistsRequest`` dataclass has no request_id field; the
    simulator's daemon-side hook (C_V6dDaemonExistsHook) parses it from
    the raw JSON body for per-request hit classification and answers
    with an extra ``location`` field.  Against a stock daemon the extra
    field is rejected (HTTP 500) — fall back to plain client.exists().
    """
    if request_id:
        try:
            data = client.rpc.call("exists", {
                "object_key": key,
                "peer": None,
                "request_id": request_id,
            })
            location = data.get("location")
            if location:
                logger.debug(
                    f"[V6D P2P] exists req={request_id} key={key} -> {location}")
            return bool(data.get("exists"))
        except Exception as e:
            logger.warning(
                "[V6D DIAG] exists(rpc) failed req=%s key=%s: %r; "
                "falling back to plain exists()", request_id, key, e)
    try:
        return bool(client.exists(key))
    except Exception as e:
        logger.warning(
            "[V6D DIAG] exists() failed req=%s key=%s: %r -> treated as MISS",
            request_id, key, e)
        return False





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
                self.client = None
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

        def _lookup_loop(self, block_hashes, request_id, exists_fn):
            """Shared hit-counting loop for lookup/async_lookup.

            *exists_fn* performs the (possibly awaited-elsewhere) exists
            check for one key; the loop truncates at the first miss.
            """
            hits = 0
            total = 0
            stop_reason = "all_hit"
            for h in block_hashes:
                total += 1
                key = self._make_key(h)
                if key in self._cached_objs:
                    hits += 1
                    continue
                try:
                    if exists_fn(key):
                        self._cached_objs[key] = key
                        if request_id is not None:
                            self._hold_key_for_req(key, request_id)
                        hits += 1
                    else:
                        stop_reason = f"miss@{hits} key={key}"
                        break
                except Exception as e:
                    stop_reason = f"exception@{hits} key={key}: {e!r}"
                    break
            return hits, total, stop_reason

        def override_lookup(self, block_hashes, request_id=None,
                            unfetched_objs=None):
            if getattr(self, "client", None) is None:
                return 0
            hits, total, stop_reason = _lookup_loop(
                self, block_hashes, request_id,
                lambda key: _sim_client_exists(self.client, key, request_id))
            if stop_reason != "all_hit":
                logger.info(
                    "[V6D DIAG] lookup group=%s req=%s hits=%d/%d stop=%s",
                    self._group_id, request_id, hits, total, stop_reason)
            return hits * getattr(self, "_group_block_size", 1)

        async def override_async_lookup(self, block_hashes, request_id=None,
                                        unfetched_objs=None):
            if getattr(self, "client", None) is None:
                return 0
            loop = asyncio.get_running_loop()
            hits = 0
            total = 0
            stop_reason = "all_hit"
            for h in block_hashes:
                total += 1
                key = self._make_key(h)
                if key in self._cached_objs:
                    hits += 1
                    continue
                try:
                    exists = await loop.run_in_executor(
                        None, _sim_client_exists, self.client, key, request_id)
                    if exists:
                        self._cached_objs[key] = key
                        if request_id is not None:
                            self._hold_key_for_req(key, request_id)
                        hits += 1
                    else:
                        stop_reason = f"miss@{hits} key={key}"
                        break
                except Exception as e:
                    stop_reason = f"exception@{hits} key={key}: {e!r}"
                    break
            if stop_reason != "all_hit":
                logger.info(
                    "[V6D DIAG] async_lookup group=%s req=%s hits=%d/%d stop=%s",
                    self._group_id, request_id, hits, total, stop_reason)
            return hits * getattr(self, "_group_block_size", 1)

        def override_get_key(self, block_hash, request_id=None):
            key = self._make_key(block_hash)
            if key in self._cached_objs:
                if request_id is not None:
                    self._hold_key_for_req(key, request_id)
                return key
            # P2P mode: use client.exists() to check tracker without
            # attempting client.get() which fails under -rpc=false.
            # Simulate "data transferred" by returning a fake key;
            # actual KV data is not needed in CPU simulation.
            try:
                if self.client is not None and self.client.exists(key):
                    self._cached_objs[key] = key
                    if request_id is not None:
                        self._hold_key_for_req(key, request_id)
                    return key
            except Exception:
                pass
            return None

        async def override_async_get_key(self, block_hash, request_id=None):
            return override_get_key(self, block_hash, request_id=request_id)

        original_lookup = target.lookup
        original_async_lookup = target.async_lookup
        original_get_key = target.get_key
        original_async_get_key = target.async_get_key
        target.lookup = override_lookup
        target.async_lookup = override_async_lookup
        target.get_key = override_get_key
        target.async_get_key = override_async_get_key

        # ---- Override batch_allocate to record ownership ----
        if hasattr(target, 'batch_allocate'):
            original_batch_allocate = target.batch_allocate

            def override_batch_allocate(self, block_hashes, size, shape,
                                        dtype, request_id=None):
                """Allocate simulated V6D keys without v6d client data-plane."""
                if getattr(self, "client", None) is not None:
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


class C_V6dObjectConnectorSchedulerHook(BaseHook):
    """Hook scheduler cross-group allocation to avoid real v6d client.create."""

    HOOK_CLASS_NAME = "V6dObjectConnectorScheduler"
    HOOK_MODULE_NAME = (
        "vllm.distributed.kv_transfer.kv_connector.v1.v6d_object_connector"
    )

    @classmethod
    def hook(cls, target):
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
                logger.info(
                    "[V6D DIAG] Request %s: scheduler lookup MISS "
                    "(final hit_length=%d) from num_computed_tokens=%d",
                    request.request_id, hit_length, num_computed_tokens)
                return 0, False
            if hasattr(self, "_to_load_token_idx"):
                self._to_load_token_idx[request.request_id] = num_computed_tokens
            if hasattr(self, "_hit_stats"):
                self._hit_stats.record(
                    total_tokens=num_hash_blocks * self.hash_block_size,
                    hit_tokens=hit_length,
                )
            logger.debug(
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

            first_fail_reason = None
            for aligned_hit_length in aligned_hit_lengths:
                all_match = True
                fail_reason = None
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
                        fail_reason = (
                            f"idx_out_of_range group={group_id} "
                            f"abs_idx={abs_idx} n={len(group_hashes)}")
                        all_match = False
                        break
                    key = manager._make_key(group_hashes[abs_idx])
                    # P2P mode: use client.exists() to check Redis tracker
                    try:
                        client = getattr(manager, "client", None)
                        if client is None or not client.exists(key):
                            fail_reason = (
                                f"exists=False group={group_id} key={key}")
                            all_match = False
                            break
                    except Exception as e:
                        fail_reason = (
                            f"exists_exception group={group_id} "
                            f"key={key}: {e!r}")
                        all_match = False
                        break
                if all_match:
                    if aligned_hit_length != max_mamba_hit_length:
                        logger.info(
                            "[V6D DIAG] mamba_validate: degraded "
                            "%d -> %d (first_fail: %s)",
                            max_mamba_hit_length, aligned_hit_length,
                            first_fail_reason)
                    return aligned_hit_length
                if first_fail_reason is None:
                    first_fail_reason = (
                        f"boundary={aligned_hit_length}: {fail_reason}")

            logger.info(
                "[V6D DIAG] mamba_validate: ALL boundaries failed, "
                "fa_hit=%d max_mamba=%d num_computed=%d first_fail: %s",
                fa_hit_length, max_mamba_hit_length, num_computed_tokens,
                first_fail_reason)
            return 0

        def _fa_lookup_plan(self, block_hashes, num_computed_tokens):
            """Yield (manager, group_block_size, tail_hashes) per FA group.

            Shared between the sync and async get_num_new_matched_tokens
            overrides — only the lookup call itself differs.
            """
            fa_group_ids = sorted(
                getattr(self, "full_attention_group_ids", set()))
            for group_id in fa_group_ids:
                manager = self.managers[group_id]
                group_block_size = self.group_block_sizes[group_id]
                group_hashes = _group_block_hashes(
                    self, block_hashes, group_block_size
                )
                start_block_idx = num_computed_tokens // group_block_size
                yield manager, group_block_size, group_hashes[start_block_idx:]

        def override_get_num_new_matched_tokens(
            self, request, num_computed_tokens
        ):
            num_hash_blocks = _num_hash_blocks(self, request)
            block_hashes = request.block_hashes[:num_hash_blocks]
            if not block_hashes:
                return 0, False

            # Step 1: full attention groups — take min
            hit_length = num_hash_blocks * self.hash_block_size
            for manager, group_block_size, tail_hashes in _fa_lookup_plan(
                self, block_hashes, num_computed_tokens
            ):
                group_hits = manager.lookup(
                    tail_hashes,
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
            hit_length = num_hash_blocks * self.hash_block_size
            for manager, group_block_size, tail_hashes in _fa_lookup_plan(
                self, block_hashes, num_computed_tokens
            ):
                group_hits = await manager.async_lookup(
                    tail_hashes,
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
            if client is not None and original_cross_group_batch_allocate is not None:
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
                    logger.debug(
                        "[V6D Hijack] Flushed %d saving entries into _saved "
                        "for _step_saved processing: %s",
                        len(flushed), flushed,
                    )
            return original_step(self)

        target.step = override_step
        logger.info(
            "[V6D Hijack] HybridScheduler.step hook installed "
            "(_saving flush for CPU no-op store)")
