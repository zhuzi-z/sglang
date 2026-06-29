"""
MockOffloadConnector — implements KVConnectorBase_V1 with pluggable
BaseCacheStorage for simulating external KV cache offloading (L2/L3)
without CUDA or network dependencies.

Also provides C_KVConnectorFactoryHook to intercept connector creation.
"""

from __future__ import annotations
import torch
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger()

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorRole,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


# ---------------------------------------------------------------------------
# Metadata passed from scheduler to worker
# ---------------------------------------------------------------------------


@dataclass
class MockConnectorMetadata:
    """Minimal metadata: just load request IDs to report as finished."""

    load_req_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MockOffloadConnector
# ---------------------------------------------------------------------------


class MockOffloadConnector:
    """
    Simulates external KV cache offloading via BaseCacheStorage.

    Scheduler role:
      - get_num_new_matched_tokens: query storage for prefix hits
      - request_finished: store block hashes for completed requests

    Worker role:
      - All transfer methods are no-op
      - get_finished: report loads as immediately completed
    """

    # Shared storage instance (module-level singleton for in-process simulation)
    _storage: set[str] = set()

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig",
    ):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
            SupportsHMA,
        )

        # Register as SupportsHMA so scheduler uses request_finished_all_groups
        # (required for hybrid models with multiple kv_cache_groups)
        SupportsHMA.register(MockOffloadConnector)

        self._vllm_config = vllm_config
        self._role = role
        self._kv_cache_config = kv_cache_config
        self._connector_metadata: MockConnectorMetadata | None = None

        # Resolve hash_block_size for proper block_hashes indexing
        self.block_size = vllm_config.cache_config.block_size
        if kv_cache_config is not None:
            from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

            _, self.hash_block_size = resolve_kv_cache_block_sizes(
                kv_cache_config, vllm_config
            )
        else:
            self.hash_block_size = self.block_size

        # Scheduler state
        self._pending_loads: set[str] = set()

        if role == KVConnectorRole.SCHEDULER:
            # Clear stale storage from previous engine instance
            MockOffloadConnector._storage.clear()
            logger.info(
                "[MockOffloadConnector] SCHEDULER role, hash_block_size=%d",
                self.hash_block_size,
            )
        elif role == KVConnectorRole.WORKER:
            logger.info("[MockOffloadConnector] WORKER role (all transfers no-op)")

    @property
    def role(self):
        return self._role

    # ------------------------------------------------------------------
    # Worker-side methods (all transfers are no-op in simulation)
    # ------------------------------------------------------------------

    def bind_connector_metadata(self, connector_metadata) -> None:
        self._connector_metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def has_connector_metadata(self) -> bool:
        return self._connector_metadata is not None

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Report all pending loads as immediately finished."""
        meta = self._connector_metadata
        finished_recving = (
            set(meta.load_req_ids) if meta is not None and meta.load_req_ids else None
        )
        return None, finished_recving

    # No-op worker stubs (no actual KV tensors or transfers)
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        pass

    def register_cross_layers_kv_cache(self, kv_cache, attn_backend) -> None:
        pass

    def set_host_xfer_buffer_ops(self, copy_operation) -> None:
        pass

    def handle_preemptions(self, kv_connector_metadata) -> None:
        pass

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def wait_for_save(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        pass

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    def get_kv_connector_stats(self):
        return None

    def get_kv_connector_kv_cache_events(self):
        return None

    def get_handshake_metadata(self):
        return None

    def build_connector_worker_meta(self):
        return None

    def get_finished_count(self) -> int | None:
        return None

    # ------------------------------------------------------------------
    # Scheduler-side methods
    # ------------------------------------------------------------------
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Query storage for prefix cache hits beyond num_computed_tokens."""
        num_skipped = num_computed_tokens // self.hash_block_size
        remaining_hashes = request.block_hashes[num_skipped:]

        if not remaining_hashes:
            return 0, False

        # Max tokens we can report (must leave at least 1 token to compute)
        max_hit_tokens = request.num_tokens - 1 - num_computed_tokens
        if max_hit_tokens <= 0:
            return 0, False

        # Convert block hashes to string keys and do prefix match
        keys = [h.hex() if isinstance(h, bytes) else str(h) for h in remaining_hashes]
        hit_blocks = 0
        for key in keys:
            if key not in self._storage:
                break
            hit_blocks += 1
        hit_tokens = hit_blocks * self.hash_block_size

        # Cap to max allowed
        hit_tokens = min(hit_tokens, max_hit_tokens)
        # Align down to hash_block_size
        hit_tokens = (hit_tokens // self.hash_block_size) * self.hash_block_size

        if hit_tokens > 0:
            # NOTE: Do NOT add to _pending_loads here. The scheduler may fail
            # to allocate blocks for this request (e.g. budget exhausted).
            # _pending_loads is populated in update_state_after_alloc() which
            # is called only after successful block allocation.
            return hit_tokens, True
        return 0, False

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> MockConnectorMetadata:
        """Return metadata with pending load request IDs."""
        meta = MockConnectorMetadata(load_req_ids=list(self._pending_loads))
        self._pending_loads.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Store all block hashes for the finished request into cache storage."""
        if request.block_hashes:
            keys = [
                h.hex() if isinstance(h, bytes) else str(h)
                for h in request.block_hashes
            ]
            self._storage.update(keys)
        return False, None

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, [])

    def reset_cache(self) -> bool | None:
        MockOffloadConnector._storage.clear()
        return True

    @classmethod
    def reset_storage(cls):
        """Reset shared storage (for test isolation)."""
        cls._storage = set()

    # No-op scheduler stubs
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        """Called by scheduler after successful block allocation.

        Only requests that successfully allocate blocks should be reported
        as pending loads. This prevents the assertion error when allocation
        fails for later requests in the same scheduling step.
        """
        if num_external_tokens > 0:
            self._pending_loads.add(request.request_id)

    def on_new_request(self, request: "Request") -> None:
        pass

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        pass

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        pass

    def has_pending_transfers(self) -> bool:
        return False

    def take_events(self) -> Iterable["KVCacheEvent"]:
        return ()

    def set_xfer_handshake_metadata(self, metadata) -> None:
        pass

    def set_xfer_handshake_metadata_pp_aware(self, metadata) -> None:
        pass

    # Class-level stubs
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


# ---------------------------------------------------------------------------
# Factory Hook
# ---------------------------------------------------------------------------


class C_KVConnectorFactoryHook(BaseHook):
    """Hook KVConnectorFactory to return MockOffloadConnector."""

    HOOK_CLASS_NAME = "KVConnectorFactory"
    HOOK_MODULE_NAME = "vllm.distributed.kv_transfer.kv_connector.factory"

    @classmethod
    def hook(cls, target):
        @classmethod
        def override_create_connector(cls, config, role, kv_cache_config):
            return MockOffloadConnector(config, role, kv_cache_config)

        target.create_connector = override_create_connector
