"""
MockHybridConnector — Simulates the real HybridConnector's cross-node KV
cache loading flow WITHOUT CUDA/GPU dependencies.

Real HybridConnector flow:
  1. on_add_req(req) → intercepts request (returns True = "eaten")
  2. sched_allocate_slots() → finds local prefix hits + allocates blocks
  3. async_get_num_new_matched_tokens() → V6D lookup for remote hits
  4. Worker-side DMA transfer (CUDA ops) → loads remote KV data
  5. _step_loaded() → sets req.num_computed_tokens += remote_hits
  6. sched_add_req(req) → adds to scheduler waiting queue

CPU Simulation:
  Steps 3-4 are replaced with synchronous etcd lookup (no real data to
  transfer). The net effect is the same: req enters scheduler with
  num_computed_tokens already reflecting both local and remote hits.

This ensures the scheduler takes the SAME code path as the real system:
  - request.num_computed_tokens > 0 → enters "else" branch
  - num_external_computed_tokens = 0 in scheduling loop
  - No _mamba_block_aligned_split assertion triggered
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
# MockHybridConnector
# ---------------------------------------------------------------------------


class MockHybridConnector:
    """Simulates HybridConnector's cross-node KV cache flow for CPU.

    Key behavioral contract (same as real HybridConnector):
    - on_add_req() → True: intercepts request, performs synchronous
      local prefix lookup + etcd remote lookup, then adds request to
      scheduler with updated num_computed_tokens
    - get_num_new_matched_tokens() → (0, False): never used because
      requests arrive at scheduler with num_computed_tokens > 0
    - request_finished() → registers block_hashes to etcd for
      cross-node visibility (replaces seal() in real V6D)
    """

    def __init__(self, config: "VllmConfig", role, kv_cache_config):
        from sglang_simulator.simulation.vllm.v6d_cache_storage import (
            V6DCacheStorage,
        )
        self._config = config
        self._kv_cache_config = kv_cache_config
        self._v6d_storage = V6DCacheStorage.get_instance()
        self._worker_id = os.environ.get("_SIM_V6D_ACTIVE_WORKER_ID", None)

        # Derive hash_block_size from config (same as real connector)
        if kv_cache_config and hasattr(kv_cache_config, "hash_block_size"):
            self.hash_block_size = kv_cache_config.hash_block_size
        else:
            self.hash_block_size = 544  # Qwen3.5 hybrid default

        # Track registered blocks for stats
        self._registered_blocks = 0
        self._remote_hits_total = 0
        self._local_hits_total = 0

        logger.info(
            "[MockHybridConnector] Initialized: worker_id=%s, "
            "hash_block_size=%d, etcd=%s",
            self._worker_id, self.hash_block_size,
            "connected" if self._v6d_storage.connected else "disconnected",
        )

    # ------------------------------------------------------------------
    # Core interface: on_add_req (mirrors HybridConnector.on_add_req)
    # ------------------------------------------------------------------

    def on_add_req(self, req: "Request") -> bool:
        """Intercept request and perform synchronous cross-node lookup.

        Mirrors real HybridConnector flow:
        1. Find local prefix cache hits (via kv_cache_manager)
        2. Query etcd for remote hits (replaces V6D async_lookup)
        3. Set req.num_computed_tokens = local + remote
        4. Add request to scheduler waiting queue
        5. Return True (request "eaten")

        If etcd is not connected or no remote hits, returns False to
        let the scheduler handle normally (pure local prefix matching).
        """
        scheduler = get_scheduler_ref()
        if scheduler is None:
            return False

        if not self._v6d_storage.connected:
            return False

        # Skip short prompts (matches VLLM_KVS_ON_MIN_LENGTH behavior)
        min_length = int(os.environ.get("VLLM_KVS_ON_MIN_LENGTH", "0"))
        if req.num_prompt_tokens <= min_length + 1:
            return False

        # Step 1: Find local prefix hits
        # (same as sched_allocate_slots does in real HybridConnector)
        #
        # NOTE: suppress prefix_cache_stats recording for this internal probe.
        # If there is no remote hit we return False and the scheduler will
        # re-run get_computed_blocks (which records stats there). Recording
        # here too would double-count the local prefix stats for every request
        # NOT eaten by the connector, polluting "Prefix cache hit rate".
        kv_cache_manager = scheduler.kv_cache_manager
        prev_log_stats = kv_cache_manager.log_stats
        try:
            kv_cache_manager.log_stats = False
            computed_blocks, num_local_computed = (
                kv_cache_manager.get_computed_blocks(req)
            )
        except Exception as e:
            logger.warning(
                "[MockHybridConnector] get_computed_blocks failed: %s", e)
            return False
        finally:
            kv_cache_manager.log_stats = prev_log_stats

        # Step 2: Query etcd for remote hits starting after local prefix
        num_skipped_blocks = num_local_computed // self.hash_block_size
        remaining_hashes = req.block_hashes[num_skipped_blocks:]

        remote_hit_blocks = 0
        if remaining_hashes:
            for h in remaining_hashes:
                key = h.hex() if isinstance(h, bytes) else str(h)
                owner = self._v6d_storage.lookup_block(key)
                if owner is None:
                    break  # prefix continuity: stop at first miss
                remote_hit_blocks += 1

        remote_hit_tokens = remote_hit_blocks * self.hash_block_size

        # If no remote hits, let scheduler handle normally (local-only path)
        if remote_hit_tokens == 0:
            return False

        # Step 3: Allocate blocks for ONLY the connector-loaded prefix.
        #
        # The real HybridConnector allocates just the remote-loaded prefix
        # (block-aligned) with delay_cache_blocks=True, moves the request to
        # WAITING_FOR_REMOTE_KV, and once the async load finishes the scheduler
        # (_update_waiting_for_remote_kv) caches those blocks into the LOCAL
        # prefix cache and lets normal scheduling prefill the rest.
        #
        # This mock has no async load, so we replicate that completion path
        # synchronously (Step 3b). Allocating ONLY remote_hit_tokens keeps the
        # new allocation block-aligned, which is REQUIRED for cache_blocks to
        # cache the mamba blocks (mamba "light" mode caches one aligned block
        # per call). The remaining prompt tokens are prefilled + cached later by
        # the normal scheduler loop (request enters with num_computed_tokens>0).
        total_computed = num_local_computed + remote_hit_tokens
        num_new_tokens = req.num_tokens - total_computed

        try:
            new_blocks = scheduler.kv_cache_manager.allocate_slots(
                req,
                remote_hit_tokens,
                num_local_computed,
                computed_blocks,
                delay_cache_blocks=True,
            )
        except Exception as e:
            logger.warning(
                "[MockHybridConnector] allocate_slots failed: %s", e)
            return False

        if new_blocks is None:
            # Block allocation failed, let scheduler handle (may retry later)
            logger.debug(
                "[MockHybridConnector] Block allocation failed for req %s",
                req.request_id)
            return False

        # Step 3b: Cache the connector-loaded prefix into the LOCAL prefix
        # cache, mirroring Scheduler._update_waiting_for_remote_kv (the real
        # HybridConnector completion path). Without this the served prefix is
        # never locally cached, so every later request with the same prefix
        # keeps going through V6D -- collapsing the local prefix hit rate.
        try:
            kv_cache_manager.cache_blocks(req, total_computed)
        except Exception as e:
            logger.warning(
                "[MockHybridConnector] local cache_blocks failed: %s", e)

        # Step 4: Update request state (mirrors _step_loaded)
        req.num_computed_tokens = total_computed
        req.num_external_computed_tokens = remote_hit_tokens

        # Record the LOCAL prefix hit exactly once for this eaten request.
        # (We suppressed recording during the probe above; the scheduler will
        # NOT re-run get_computed_blocks for eaten requests because they enter
        # with num_computed_tokens > 0 -> the scheduler "else" branch.)
        if prev_log_stats and kv_cache_manager.prefix_cache_stats is not None:
            kv_cache_manager.prefix_cache_stats.record(
                num_tokens=req.num_tokens,
                num_hits=num_local_computed,
                preempted=req.num_preemptions > 0,
            )

        # Track stats
        self._remote_hits_total += remote_hit_blocks
        self._local_hits_total += num_local_computed // self.hash_block_size

        # Classify hits for logging
        local_owner_count = 0
        remote_owner_count = 0
        for i, h in enumerate(remaining_hashes[:remote_hit_blocks]):
            key = h.hex() if isinstance(h, bytes) else str(h)
            owner = self._v6d_storage.lookup_block(key)
            if owner == self._worker_id:
                local_owner_count += 1
            else:
                remote_owner_count += 1

        logger.info(
            "[MockHybridConnector] Request %s: "
            "local_prefix=%d tokens, remote_hit=%d tokens (%d blocks), "
            "total_computed=%d, remaining=%d | "
            "owners: same_worker=%d, cross_node=%d",
            req.request_id,
            num_local_computed, remote_hit_tokens, remote_hit_blocks,
            total_computed, num_new_tokens,
            local_owner_count, remote_owner_count,
        )

        # Step 5: Add request to scheduler (mirrors sched_add_req)
        scheduler.add_request(req)
        return True

    # ------------------------------------------------------------------
    # get_num_new_matched_tokens: ALWAYS returns (0, False)
    # This matches real HybridConnector behavior exactly.
    # ------------------------------------------------------------------

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """Always returns (0, False) — same as real HybridConnector.

        Remote hits are handled in on_add_req(), not here.
        """
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

        Replaces V6D's seal() operation in real system. After prefill
        completes, blocks become discoverable by other nodes via etcd.
        """
        if request.block_hashes:
            worker_id = self._worker_id or "unknown"
            registered = 0
            for h in request.block_hashes:
                key = h.hex() if isinstance(h, bytes) else str(h)
                if self._v6d_storage.register_block(key, worker_id):
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
        """No-op: blocks are pre-allocated in on_add_req."""
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
        from sglang_simulator.simulation.vllm.v6d_cache_storage import (
            V6DCacheStorage,
        )
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
