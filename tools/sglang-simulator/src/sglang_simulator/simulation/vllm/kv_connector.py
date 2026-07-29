"""
MockHybridConnector v2 — Simulates cross-node KV cache sharing via V6D/etcd.

Design principle: NEVER allocate blocks in on_add_req.  All block allocation
happens inside the scheduler's scheduling loop (one request at a time) via
get_num_new_matched_tokens, ensuring proper LRU ordering and preventing
batch-arrival cache thrashing.

Flow (aligned with scheduler's connector path):
  1. on_add_req(req) → always returns False (request enters scheduler normally)
  2. Scheduler scheduling loop (per-request, sequential):
     a. get_computed_blocks(req) → local prefix hits
     b. get_num_new_matched_tokens(req, local_hit) → etcd lookup for remote
     c. allocate_slots(req, ...) → allocates all needed blocks at once
     d. cache_blocks → caches everything into local prefix cache
  3. request_finished(req) → registers block hashes to etcd for cross-node
     visibility (replaces V6D seal)

This eliminates the "batch allocation storm" bug where on_add_req allocated
blocks for ALL queued requests at once (during _process_input_queue), causing
earlier requests' cached prefixes to be evicted before later requests in the
same batch could benefit from them.
"""

from __future__ import annotations
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger()

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


# Module-level scheduler reference (set by C_VLLMSchedulerHook)
_scheduler_ref = None


def set_scheduler_ref(scheduler):
    """Called by scheduler hook after init to share the reference."""
    global _scheduler_ref
    _scheduler_ref = scheduler
    logger.info("[MockHybridConnector] Scheduler reference acquired")


def get_scheduler_ref():
    return _scheduler_ref


# ---------------------------------------------------------------------------
# Metadata (minimal, no real KV transfer in CPU sim)
# ---------------------------------------------------------------------------


@dataclass
class MockHybridMetadata:
    """Minimal metadata placeholder."""
    pass


# ---------------------------------------------------------------------------
# MockHybridConnector v2
# ---------------------------------------------------------------------------


# DEPRECATED: MockHybridConnector is legacy code from the non-native CPU simulation path.
# It is only used when SGLANG_SIMULATOR_NATIVE_V6D_CONTROL_PLANE is NOT set (default off).
# The current production path uses NATIVE_V6D_CONTROL_PLANE=1 which keeps the real
# HybridConnector and does not use this class.
# V6DCacheStorage (etcd backend) has been deleted; this code will NOT work if activated.
# Retained for backward compatibility reference only. Do NOT enable in new deployments.
class MockHybridConnector:
    """Simulates HybridConnector's cross-node KV cache sharing for CPU.

    Key behavioral contract:
    - on_add_req() → False: NEVER eats request; let scheduler handle
    - get_num_new_matched_tokens() → (remote_tokens, False): returns V6D
      remote hit count; False = no async load (CPU sim, instant)
    - request_finished() → registers block_hashes to etcd
    """

    def __init__(self, config: "VllmConfig", role, kv_cache_config):
        # DEPRECATED: v6d_cache_storage deleted
        self._config = config
        self._kv_cache_config = kv_cache_config
        self._v6d_storage = V6DCacheStorage.get_instance()
        from sglang_simulator.simulation.vllm.v6d.v6d_manager import (
            get_active_worker_id,
        )
        self._worker_id = get_active_worker_id()

        # Derive hash_block_size from config (same as real connector)
        if kv_cache_config and hasattr(kv_cache_config, "hash_block_size"):
            self.hash_block_size = kv_cache_config.hash_block_size
        else:
            self.hash_block_size = 544  # Qwen3.5 hybrid default

        # Track registered blocks for stats
        self._registered_blocks = 0
        self._remote_hits_total = 0

        logger.info(
            "[MockHybridConnector] Initialized: worker_id=%s, "
            "hash_block_size=%d, etcd=%s",
            self._worker_id, self.hash_block_size,
            "connected" if self._v6d_storage.connected else "disconnected",
        )

    # ------------------------------------------------------------------
    # Core interface: on_add_req — NEVER eats, always returns False
    # ------------------------------------------------------------------

    def on_add_req(self, req: "Request") -> bool:
        """Always returns False — request enters scheduler normally.

        We do NOT allocate blocks or query etcd here. All work is deferred
        to get_num_new_matched_tokens (called by scheduler in its scheduling
        loop, one request at a time).
        """
        return False

    # ------------------------------------------------------------------
    # get_num_new_matched_tokens: V6D etcd lookup (called by scheduler)
    # ------------------------------------------------------------------

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """Query V6D/etcd for remote prefix hits beyond local cache.

        Called by scheduler INSIDE the scheduling loop (one request at a time).
        This ensures block allocation happens sequentially, avoiding the batch
        cache thrashing bug.

        When remote hits are found, returns load_kv_async=True to trigger the
        WAITING_FOR_REMOTE_KVS path. This path uses block-aligned cache_blocks
        (via _update_waiting_for_remote_kv), which is critical for mamba hybrid
        models where cache_blocks requires block-aligned num_tokens.

        We simultaneously signal "transfer complete" by adding the request_id
        to scheduler.finished_recving_kv_req_ids, so the request becomes ready
        on the very next scheduling step (simulating instant CPU transfer).

        Args:
            request: The request being scheduled.
            num_computed_tokens: Number of tokens already locally cached.

        Returns:
            (num_external_tokens, load_kv_async):
            - num_external_tokens: additional tokens available via V6D
            - load_kv_async: True if remote hit found (triggers async path)
        """
        if not self._v6d_storage.connected:
            return 0, False

        # Skip short prompts (matches VLLM_KVS_ON_MIN_LENGTH behavior)
        min_length = int(os.environ.get("VLLM_KVS_ON_MIN_LENGTH", "0"))
        if request.num_prompt_tokens <= min_length + 1:
            return 0, False

        # Query etcd for remote hits starting after local prefix
        num_skipped_blocks = num_computed_tokens // self.hash_block_size
        remaining_hashes = request.block_hashes[num_skipped_blocks:]

        remote_hit_blocks = 0
        if remaining_hashes:
            for h in remaining_hashes:
                key = h.hex() if isinstance(h, bytes) else str(h)
                owner = self._v6d_storage.lookup_block(key)
                if owner is None:
                    break  # prefix continuity: stop at first miss
                remote_hit_blocks += 1

        remote_hit_tokens = remote_hit_blocks * self.hash_block_size

        if remote_hit_tokens > 0:
            # Classify hits for logging
            same_worker_count = 0
            cross_node_count = 0
            for h in remaining_hashes[:remote_hit_blocks]:
                key = h.hex() if isinstance(h, bytes) else str(h)
                owner = self._v6d_storage.lookup_block(key)
                if owner == self._worker_id:
                    same_worker_count += 1
                elif owner is not None:
                    cross_node_count += 1

            self._remote_hits_total += remote_hit_blocks
            total_computed = num_computed_tokens + remote_hit_tokens
            remaining = request.num_tokens - total_computed

            logger.info(
                "[MockHybridConnector] Request %s: "
                "local_prefix=%d tokens, remote_hit=%d tokens (%d blocks), "
                "total_computed=%d, remaining=%d | "
                "owners: same_worker=%d, cross_node=%d",
                request.request_id,
                num_computed_tokens, remote_hit_tokens, remote_hit_blocks,
                total_computed, remaining,
                same_worker_count, cross_node_count,
            )

            # Signal "instant transfer complete" so _update_waiting_for_remote_kv
            # finds this request ready on the next scheduling step.
            scheduler = get_scheduler_ref()
            if scheduler is not None:
                scheduler.finished_recving_kv_req_ids.add(request.request_id)
            else:
                logger.warning(
                    "[MockHybridConnector] No scheduler ref! "
                    "Cannot signal transfer complete for %s",
                    request.request_id,
                )

            # load_kv_async=True triggers WAITING_FOR_REMOTE_KVS path
            # which uses block-aligned cache_blocks in
            # _update_waiting_for_remote_kv
            return remote_hit_tokens, True

        return 0, False

    # ------------------------------------------------------------------
    # request_finished: register blocks to etcd (replaces V6D seal)
    # ------------------------------------------------------------------

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Register all block hashes to etcd for cross-node visibility.

        After prefill completes, blocks become discoverable by other nodes.
        """
        if request.block_hashes:
            # Use worker_id directly (may be None — consistent with lookup)
            worker_id = self._worker_id
            registered = 0
            for h in request.block_hashes:
                key = h.hex() if isinstance(h, bytes) else str(h)
                # Store worker_id as-is (None stored as "None" string in JSON)
                value = worker_id if worker_id is not None else "__self__"
                if self._v6d_storage.register_block(key, value):
                    registered += 1
            self._registered_blocks += registered
            logger.debug(
                "[MockHybridConnector] Registered %d/%d blocks for req %s "
                "(worker=%s)",
                registered, len(request.block_hashes),
                request.request_id, worker_id,
            )
        return False, None

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, [])

    # ------------------------------------------------------------------
    # Scheduler interface stubs (matching KVConnectorBase_V1)
    # ------------------------------------------------------------------

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks",
        num_external_tokens: int
    ) -> None:
        """No-op: scheduler handles allocation directly."""
        pass

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> MockHybridMetadata:
        return MockHybridMetadata()

    def step(self):
        """No async work in CPU simulation."""
        return None

    def has_requests(self) -> bool:
        """No pending async requests."""
        return False

    def reset_cache(self) -> bool | None:
        # DEPRECATED: v6d_cache_storage deleted
        V6DCacheStorage.reset_instance()
        return True

    def take_events(self) -> Iterable["KVCacheEvent"]:
        return ()

    def on_abort_req(self, reqid: str, reason: str = "",
                     output: bool = True, iscore=True):
        pass

    def on_new_request(self, request: "Request") -> None:
        pass

    def update_connector_output(self, connector_output) -> None:
        pass

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        pass

    def has_pending_transfers(self) -> bool:
        return False

    def set_xfer_handshake_metadata(self, metadata) -> None:
        pass

    def set_xfer_handshake_metadata_pp_aware(self, metadata) -> None:
        pass

    def set_block_pool(self, block_pool) -> None:
        pass

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return False

    @classmethod
    def build_kv_connector_stats(cls, data=None):
        return None

    @classmethod
    def build_prom_metrics(cls, *args, **kwargs):
        return None

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config=None) -> bool:
        return False

    def get_kv_connector_stats(self):
        return None

    def get_finished_count(self):
        """Return None to use default world_size for output aggregation."""
        return None

    def get_finished(self, finished_req_ids=None):
        """No-op: no async transfers to track."""
        return set(), set()

    def get_block_ids_with_load_errors(self):
        return set()

    def shutdown(self):
        pass

    def get_kv_connector_kv_cache_events(self):
        return None

    def get_handshake_metadata(self):
        return None

    def register_kv_caches(self, kv_caches):
        pass

    def register_cross_layers_kv_cache(self, *args, **kwargs):
        pass

    def set_host_xfer_buffer_ops(self, copy_operation):
        pass

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, *args, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def bind_connector_metadata(self, connector_metadata):
        pass

    def clear_connector_metadata(self):
        pass

    def has_connector_metadata(self):
        return False


# ---------------------------------------------------------------------------
# Factory Hook
# ---------------------------------------------------------------------------


class C_KVConnectorFactoryHook(BaseHook):
    """Hook KVConnectorFactory to return MockHybridConnector."""

    HOOK_CLASS_NAME = "KVConnectorFactory"
    HOOK_MODULE_NAME = "vllm.distributed.kv_transfer.kv_connector.factory"

    @classmethod
    def hook(cls, target):

        @classmethod
        def override_create_connector(klass, config, role, kv_cache_config):
            connector = MockHybridConnector(config, role, kv_cache_config)
            logger.info(
                "[KVConnector Hook] Created MockHybridConnector "
                "(simulates real HybridConnector path)")
            return connector

        target.create_connector = override_create_connector
