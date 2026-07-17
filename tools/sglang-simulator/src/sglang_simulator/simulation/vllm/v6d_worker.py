"""
Worker-side hooks for native V6D control-plane simulation on CPU.

The real V6dObjectConnectorWorker class is preserved so scheduler/worker
metadata flow remains native, while CUDA/SRPC data-plane operations are
converted to deterministic no-op completions.  This lets CPU-only dual-node
validation exercise the real control plane without transferring full KV data.
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
        def override_async_v6d_init_and_notify(self) -> None:
            """Notify scheduler that V6D is ready without CUDA/SRPC init."""
            self._device = torch.device("cpu")
            self.client = None
            try:
                import asyncio
                import time
                from vllm.v1.hybrid_connector import IoDoneReqs, hybridworker
                from vllm.v1.hybrid_connector.engine_proxy import get_hybrid_worker_loop
                from vllm.v1.hybrid_connector.utils import kill_me_if_exception
                from vllm.v1.hybrid_connector.v6d_object_backend import (
                    _V6D_READY_REQ,
                    _V6D_READY_RESP,
                )

                @kill_me_if_exception
                async def _send_v6d_ready():
                    last_log_time = 0.0
                    while True:
                        try:
                            await hybridworker().io_done_rpc(
                                IoDoneReqs(worker_tprank=self._rank_id, reqids=[]),
                                _V6D_READY_REQ,
                                _V6D_READY_RESP,
                            )
                            logger.info(
                                "[V6D Hijack] sent CPU v6d ready notification "
                                "for tprank=%s",
                                self._rank_id,
                            )
                            return
                        except Exception as e:
                            if isinstance(e, AssertionError):
                                raise
                            if time.time() - last_log_time > 10:
                                logger.warning(
                                    "[V6D Hijack] send CPU v6d ready failed "
                                    "for tprank=%s, retrying: %s",
                                    self._rank_id,
                                    e,
                                )
                                last_log_time = time.time()
                            await asyncio.sleep(0.5)

                asyncio.run_coroutine_threadsafe(
                    _send_v6d_ready(), get_hybrid_worker_loop())
            except Exception:
                logger.exception(
                    "[V6D Hijack] failed to schedule CPU v6d ready notification")

        def override_start_async_v6d_init(self):
            """Skip worker-side V6D RPC/SRPC data channel initialization."""
            self._load_handler = None
            self._store_handler = None
            override_async_v6d_init_and_notify(self)
            logger.info(
                "[V6D Hijack] _start_async_v6d_init: skipped worker "
                "V6D RPC/SRPC data channel initialization (CPU mode)"
            )

        target._async_v6d_init_and_notify = override_async_v6d_init_and_notify
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
            self._sim_finished_load_reqs = set(metadata.reqs_to_load)
            logger.info(
                "[V6D Hijack] start_load_kv: completed CPU no-op loads %s",
                sorted(self._sim_finished_load_reqs),
            )
            return None

        def override_start_store_kv(self, metadata):
            self._sim_finished_store_reqs = set(metadata.reqs_to_store)
            logger.info(
                "[V6D Hijack] start_store_kv: completed CPU no-op stores %s",
                sorted(self._sim_finished_store_reqs),
            )
            return None

        async def _completed_v6d_task():
            return None

        def override_async_start_load_kv(self, metadata):
            import asyncio
            req_ids = set(metadata.reqs_to_load)
            self._sim_finished_load_reqs = (
                getattr(self, "_sim_finished_load_reqs", set()) | req_ids
            )
            logger.info(
                "[V6D Hijack] async_start_load_kv: completed CPU no-op loads %s",
                sorted(req_ids),
            )
            return {
                req_id: asyncio.create_task(_completed_v6d_task())
                for req_id in req_ids
            }

        def override_async_start_store_kv(self, metadata):
            import asyncio
            req_ids = set(metadata.reqs_to_store)
            self._sim_finished_store_reqs = (
                getattr(self, "_sim_finished_store_reqs", set()) | req_ids
            )
            logger.info(
                "[V6D Hijack] async_start_store_kv: completed CPU no-op stores %s",
                sorted(req_ids),
            )
            return {
                req_id: asyncio.create_task(_completed_v6d_task())
                for req_id in req_ids
            }

        def override_get_finished(self, finished_req_ids):
            load_reqs = set(getattr(self, "_sim_finished_load_reqs", set()))
            store_reqs = set(getattr(self, "_sim_finished_store_reqs", set()))
            self._sim_finished_load_reqs = set()
            self._sim_finished_store_reqs = set()
            if load_reqs or store_reqs:
                logger.info(
                    "[V6D Hijack] get_finished: store=%s load=%s "
                    "finished_req_ids=%s",
                    sorted(store_reqs),
                    sorted(load_reqs),
                    sorted(finished_req_ids or []),
                )
            return store_reqs, load_reqs

        target.start_load_kv = override_start_load_kv
        target.start_store_kv = override_start_store_kv
        target.async_start_load_kv = override_async_start_load_kv
        target.async_start_store_kv = override_async_start_store_kv
        target.get_finished = override_get_finished

        logger.info("[V6D Hijack] V6dObjectConnectorWorker hook installed")
