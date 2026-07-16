"""
DEPRECATED — real V6D IPC component hook, NOT part of current functional scope.

Hooks V6dObjectConnectorWorker, a REAL vLLM class that only gets instantiated
if a real V6dObjectConnector is created. Since
kv_connector.C_KVConnectorFactoryHook unconditionally returns
MockHybridConnector, this class is never instantiated in the current CPU
simulation path, and this hook is NOT registered in startup.py's init_hook()
(see startup.py comment). Current scope only requires scheduling-behavior
parity via MockHybridConnector; real V6D daemon / vineyard IPC connection is
not required. Safe to delete entirely unless a future task explicitly
requires validating the real V6D/vineyard daemon path.

Original docstring below is kept for reference only.
---
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
            """Skip worker-side V6D RPC/SRPC data channel initialization."""
            self._device = torch.device("cpu")
            self.client = None
            self._load_handler = None
            self._store_handler = None
            logger.info(
                "[V6D Hijack] _start_async_v6d_init: skipped worker "
                "V6D RPC/SRPC data channel initialization (CPU mode)"
            )

        target._start_async_v6d_init = override_start_async_v6d_init

        # Override register_kv_caches to allocate on CPU instead of GPU
        original_register = target.register_kv_caches

        def override_register_kv_caches(self, kv_caches):
            """Register KV cache metadata without starting SRPC data channels."""
            self.kv_caches = kv_caches
            self._start_async_v6d_init()
            logger.info(
                "[V6D Hijack] register_kv_caches: registered %d tensors "
                "and skipped worker-side V6D/SRPC transport (CPU)",
                len(kv_caches),
            )

        target.register_kv_caches = override_register_kv_caches

        def override_start_load_kv(self, metadata):
            return None

        def override_start_store_kv(self, metadata):
            return None

        async def _completed_v6d_task():
            return None

        def override_async_start_load_kv(self, metadata):
            import asyncio
            return {
                req_id: asyncio.create_task(_completed_v6d_task())
                for req_id in metadata.reqs_to_load
            }

        def override_async_start_store_kv(self, metadata):
            import asyncio
            return {
                req_id: asyncio.create_task(_completed_v6d_task())
                for req_id in metadata.reqs_to_store
            }

        def override_get_finished(self, finished_req_ids):
            return set(), set()

        target.start_load_kv = override_start_load_kv
        target.start_store_kv = override_start_store_kv
        target.async_start_load_kv = override_async_start_load_kv
        target.async_start_store_kv = override_async_start_store_kv
        target.get_finished = override_get_finished

        logger.info("[V6D Hijack] V6dObjectConnectorWorker hook installed")
