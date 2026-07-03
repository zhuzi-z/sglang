"""
V6D ObjectConnectorWorker Hook - Skips CUDA-specific initialization.

Preserves:
- V6D client connection (real V6D daemon)
- Handler creation and KV cache setup
- Async load/store logic

Mocks:
- _register_v6d_host_memory() → No-op (CPU can access mmap directly)
- torch.cuda.current_device() → cpu device
- torch.cuda.set_device() → No-op
- pin_memory → False (no CUDA pinning needed)
"""

import torch

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.vllm.v6d_swap import DummyEvent, DummyStream
from sglang_simulator.utils import get_logger

logger = get_logger()


class C_V6dObjectConnectorWorkerHook(BaseHook):
    """Hook V6dObjectConnectorWorker to skip CUDA host memory registration.

    The V6dObjectConnectorWorker is responsible for:
    1. Connecting to V6D daemon (PRESERVED)
    2. Registering V6D mmap as CUDA pinned memory (SKIPPED - CPU direct access)
    3. Creating load/store handlers (PRESERVED, but handlers use DummyStream)
    4. Managing async swap operations (PRESERVED)
    """

    HOOK_CLASS_NAME = "V6dObjectConnectorWorker"
    HOOK_MODULE_NAME = (
        "vllm.distributed.kv_transfer.kv_connector.v1.v6d_object_connector"
    )

    @classmethod
    def hook(cls, target):
        # Override _register_v6d_host_memory → No-op
        def noop_register_v6d_host_memory(self):
            """Skip cudaHostRegister — CPU can directly access V6D mmap."""
            logger.info(
                "[V6D Hijack] _register_v6d_host_memory: skipped "
                "(CPU direct mmap access)"
            )

        target._register_v6d_host_memory = noop_register_v6d_host_memory

        # Override _start_async_v6d_init to avoid torch.cuda.current_device()
        original_start_async = target._start_async_v6d_init

        def override_start_async_v6d_init(self):
            """Connect to V6D without CUDA device setup."""
            from vllm.distributed.kv_transfer.kv_connector.v1.v6d_object_connector import (
                _connect_v6d_with_retry,
            )

            # Skip VLLM_V6D_ASYNC_REGISTER branch - do sync connect
            # This avoids torch.cuda.current_device() in the async path
            self._device = torch.device("cpu")
            self.client = _connect_v6d_with_retry(self.v6d_url)
            # Skip _register_v6d_host_memory (already overridden to no-op)
            self._ensure_handlers()
            self._setup_handler_kv_caches()
            logger.info(
                "[V6D Hijack] _start_async_v6d_init: connected to %s (CPU mode)",
                self.v6d_url,
            )

        target._start_async_v6d_init = override_start_async_v6d_init

        # Override register_kv_caches to allocate on CPU instead of GPU
        original_register = target.register_kv_caches

        def override_register_kv_caches(self, kv_caches):
            """Register KV caches - they are already on CPU (from worker hook)."""
            # The original register_kv_caches builds raw_views from tensor storage.
            # With head_dim=1 and device=cpu, this works as-is since the tensors
            # are already CPU tensors. Just call original.
            original_register(self, kv_caches)
            logger.info(
                "[V6D Hijack] register_kv_caches: registered %d tensors (CPU)",
                len(kv_caches),
            )

        target.register_kv_caches = override_register_kv_caches

        logger.info("[V6D Hijack] V6dObjectConnectorWorker hook installed")
