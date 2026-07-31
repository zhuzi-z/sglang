"""
V6D SwapHandler Hook - Replaces CUDA operations with CPU equivalents.

Preserves:
- V6D client.get() real calls (cross-node communication)
- obj.resolver() to get mmap tensor reference
- Data copy logic (replaced with CPU memcpy)

Mocks:
- torch.cuda.Stream() -> DummyStream
- torch.cuda.Event() -> DummyEvent
- ops.v6d_swap_blocks() -> CPU memcpy (via monkey-patch)
- torch.cuda.stream() context -> No-op
- torch.cuda.current_stream() -> DummyStream
"""

import asyncio
from collections import deque
from typing import Any

import torch

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger()

_OPS_PATCHED = False

from sglang_simulator.simulation.vllm.cpu_stubs import DummyEvent, DummyStream

class DummyStream:
    """Replace torch.cuda.Stream - all operations are no-ops."""

    def record_event(self, event=None):
        return DummyEvent()

    def wait_event(self, event):
        pass

    def wait_stream(self, stream):
        pass

    def synchronize(self):
        pass

    def query(self):
        return True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

def _patch_v6d_ops():
    """Monkey-patch vllm._custom_ops to replace CUDA V6D kernels.

    Called lazily on first V6dSwapHandler instantiation (after vllm is imported).
    """
    global _OPS_PATCHED
    if _OPS_PATCHED:
        return
    try:
        from vllm import _custom_ops as ops

        def mock_v6d_swap_blocks(
            layer_gpu_tensor_ptrs, cpu_block_tensor_ptrs,
            page_size, gpu_block_ids, swap_in
        ):
            """No-op replacement for the CUDA DMA kernel.

            KV tensors are MINIMAL (1 page, 4096 bytes) while page_size
            is the real model value, so pointer arithmetic
            (base + block_id * page_size) points outside the allocation
            — an actual memmove would corrupt memory.  Simulation never
            reads KV data contents, so no copy is needed.
            """
            return None

        ops.v6d_swap_blocks = mock_v6d_swap_blocks
        ops.v6d_register_host_memory = lambda base_addr, size: None
        ops.v6d_unregister_host_memory = lambda base_addr: None

        _OPS_PATCHED = True
        logger.info("[V6D Hijack] Patched vllm._custom_ops V6D functions")
    except ImportError:
        logger.warning(
            "[V6D Hijack] Could not import vllm._custom_ops for patching"
        )

class C_V6dSwapHandlerHook(BaseHook):
    """Hook V6dSwapHandler to replace CUDA operations with CPU equivalents.

    The V6dSwapHandler is responsible for GPU<->V6D data transfer.
    In simulation:
    - CUDA Stream/Event -> DummyStream/DummyEvent
    - ops.v6d_swap_blocks() -> mock CPU memcpy (via module patch)
    - torch.cuda.stream()/current_stream() -> bypassed
    - V6D client calls (get/create/seal) are PRESERVED (real communication)
    """

    HOOK_CLASS_NAME = "V6dSwapHandler"
    HOOK_MODULE_NAME = (
        "vllm.distributed.kv_transfer.kv_connector.v1.v6d_object_connector"
    )

    @classmethod
    def hook(cls, target):

        def override_init(
            self, rank_id, rank_size, client, swap_in, total_num_kv_heads=0
        ):
            """Replace CUDA Stream/Event with dummies, keep V6D client."""
            # Patch ops on first instantiation
            _patch_v6d_ops()

            self._rank_id = rank_id
            self._rank_size = rank_size
            self._client = client
            self._swap_in = swap_in
            # Replace CUDA primitives
            self._stream = DummyStream()
            self._event_pool: list = []
            self._event_jobs: deque = deque()
            # KV cache references (set later)
            self._cross_layers_kv_cache = None
            self._per_layer_raw_views = None
            self._layer_names = None
            self._buckets = []
            self._gpu_device = torch.device("cpu")
            self._logged_page_size_check = {}
            self._job_objs = {}

            # KV head dedup logic (preserved from original)
            if total_num_kv_heads > 0 and rank_size > total_num_kv_heads:
                num_replicas = rank_size // total_num_kv_heads
                self._dedup_rank_index = rank_id // num_replicas
                self._dedup_is_representative = (rank_id % num_replicas) == 0
            else:
                self._dedup_rank_index = rank_id
                self._dedup_is_representative = True

            logger.info(
                "[V6D Hijack] V6dSwapHandler.__init__: "
                "rank=%d, swap_in=%s, device=cpu",
                rank_id,
                swap_in,
            )

        target.__init__ = override_init

        # Override swap() to remove torch.cuda.stream() usage
        original_validate_swap = target._validate_swap
        original_process_swap_batch = target._process_swap_batch

        def override_swap(
            self, job_id, keys, gpu_block_ids, request_id,
            group_ids=None
        ):
            """Swap without CUDA stream context - CPU direct execution."""
            failed_gpu_block_ids: list = []
            if not self._validate_swap(request_id, job_id, group_ids):
                return failed_gpu_block_ids

            # No CUDA stream synchronization needed on CPU
            # Directly execute V6D operations
            try:
                if self._swap_in:
                    batch_objs = self._client.get(
                        keys, request_id=request_id)
                else:
                    batch_objs = self._client.get(
                        keys, peer="local", unsafe=True,
                        wait_timeout=0, request_id=request_id)
            except Exception as e:
                logger.exception(
                    f"[{request_id=}, {job_id=}] Failed to batch "
                    f"{'load' if self._swap_in else 'store'} "
                    f"{len(keys)} keys for groups {group_ids}: {e}")
                if self._swap_in:
                    failed_gpu_block_ids = list(gpu_block_ids)
                return failed_gpu_block_ids

            try:
                got_objs = self._process_swap_batch(
                    batch_objs, keys, gpu_block_ids, job_id,
                    request_id, group_ids)
            except Exception:
                if self._swap_in:
                    failed_gpu_block_ids = list(gpu_block_ids)
                return failed_gpu_block_ids

            self._finalize_swap(job_id, got_objs)
            return failed_gpu_block_ids

        target.swap = override_swap

        # Override async_swap to remove torch.cuda.stream() and Event usage
        async def override_async_swap(
            self, job_id, keys, gpu_block_ids, request_id,
            group_ids=None
        ):
            """Async swap without CUDA - direct execution."""
            if not self._validate_swap(request_id, job_id, group_ids):
                raise RuntimeError(
                    f"[{request_id=}, {job_id=}] "
                    f"_validate_swap failed for groups {group_ids}")

            try:
                if self._swap_in:
                    batch_objs = await self._client.async_get(
                        keys, request_id=request_id)
                else:
                    batch_objs = await self._client.async_get(
                        keys, peer="local", unsafe=True,
                        wait_timeout=0, request_id=request_id)
            except Exception as e:
                logger.exception(
                    f"[{request_id=}, {job_id=}] Failed to async batch "
                    f"{'load' if self._swap_in else 'store'} "
                    f"{len(keys)} keys for groups {group_ids}: {e}")
                raise

            # Process directly on CPU (no CUDA stream needed)
            got_objs = self._process_swap_batch(
                batch_objs, keys, gpu_block_ids, job_id,
                request_id, group_ids)
            # No event needed - CPU execution is synchronous
            del got_objs
            logger.debug(f"Job {job_id} finished.")

        target.async_swap = override_async_swap

        # Override _finalize_swap to use DummyEvent
        def override_finalize_swap(self, job_id, got_objs):
            """Use DummyEvent instead of torch.cuda.Event."""
            event = DummyEvent()
            self._event_jobs.append((event, job_id))
            self._job_objs[job_id] = got_objs

        target._finalize_swap = override_finalize_swap

        # Override get_finished
        def override_get_finished(self):
            """All events immediately done with DummyEvent."""
            finished = []
            while self._event_jobs:
                event, job_id = self._event_jobs[0]
                if event.query():
                    self._event_jobs.popleft()
                    finished.append(job_id)
                    self._job_objs.pop(job_id, None)
                else:
                    break
            return finished

        target.get_finished = override_get_finished

        logger.info("[V6D Hijack] V6dSwapHandler hook installed")
