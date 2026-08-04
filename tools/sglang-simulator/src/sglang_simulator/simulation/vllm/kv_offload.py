"""
vLLM KV Offload Worker Hooks - Bypass CUDA requirements for the
worker-side of vLLM's native CPU offloading connectors.

Since the simulator has no real model runner, actual GPU<->CPU data transfer
is unnecessary. These hooks make workers immediately report all store/load
events as complete, allowing the scheduler-side to populate its CPU block pool
hash index and detect L2 (host) cache hits normally.

Covers both:
- SimpleCPUOffloadWorker (VLLM_USE_SIMPLE_KV_OFFLOAD=1)
- OffloadingConnectorWorker (default native path)
"""

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger()


class C_VLLMSimpleCPUOffloadWorkerHook(BaseHook):
    """Hook SimpleCPUOffloadWorker to skip CUDA and auto-complete transfers."""

    HOOK_CLASS_NAME = "SimpleCPUOffloadWorker"
    HOOK_MODULE_NAME = "vllm.v1.simple_kv_offload.worker"

    @classmethod
    def hook(cls, target):
        def override_init(self, vllm_config, kv_cache_config, cpu_capacity_bytes):
            """Skip pinned memory allocation and CUDA stream creation."""
            self._captured_metadata = None
            self._completed_store_events: dict[int, int] = {}
            logger.info(
                "[vLLM Hijack] SimpleCPUOffloadWorker.__init__: "
                "skipped CUDA allocation (simulation mode)"
            )

        def override_register_kv_caches(self, kv_caches):
            """No-op: no actual KV cache tensors to register."""
            pass

        def override_bind_connector_metadata(self, metadata):
            """Capture metadata so we can report events as completed."""
            self._captured_metadata = metadata

        def override_clear_connector_metadata(self):
            """Clear captured metadata."""
            self._captured_metadata = None

        def override_handle_preemptions(self, kv_connector_metadata):
            """No-op: no in-flight transfers to flush."""
            pass

        def override_get_finished(self, finished_req_ids):
            """Immediately report all load requests as finished.

            Since there is no actual data transfer, loads are instant.
            Returns (finished_sending, finished_recving).
            """
            meta = self._captured_metadata
            finished_recving = set()

            if meta is not None and meta.load_event >= 0:
                for reqs in meta.load_event_to_reqs.values():
                    finished_recving.update(reqs)

            # Report store events as completed immediately
            if meta is not None and meta.store_event >= 0:
                self._completed_store_events[meta.store_event] = 1

            return None, finished_recving or None

        def override_build_connector_worker_meta(self):
            """Report all pending store events as completed."""
            from vllm.v1.simple_kv_offload.metadata import (
                SimpleCPUOffloadWorkerMetadata,
            )

            if not self._completed_store_events:
                return None
            meta = SimpleCPUOffloadWorkerMetadata(
                completed_store_events=self._completed_store_events,
            )
            self._completed_store_events = {}
            return meta

        target.__init__ = override_init
        target.register_kv_caches = override_register_kv_caches
        target.bind_connector_metadata = override_bind_connector_metadata
        target.clear_connector_metadata = override_clear_connector_metadata
        target.handle_preemptions = override_handle_preemptions
        target.get_finished = override_get_finished
        target.build_connector_worker_meta = override_build_connector_worker_meta


class C_VLLMOffloadingConnectorWorkerHook(BaseHook):
    """Hook OffloadingConnectorWorker to skip CUDA and auto-complete transfers."""

    HOOK_CLASS_NAME = "OffloadingConnectorWorker"
    HOOK_MODULE_NAME = "vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker"

    @classmethod
    def hook(cls, target):
        def override_init(self, spec):
            """Skip OffloadingWorker (CUDA streams) creation."""
            from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
                OffloadingWorkerMetadata,
            )
            from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
                OffloadingConnectorStats,
            )

            self.spec = spec
            self.worker = None  # no real worker
            self.kv_connector_stats = OffloadingConnectorStats()
            self._load_jobs: dict[int, str] = {}
            self._unsubmitted_store_jobs: list = []
            self._connector_worker_meta = OffloadingWorkerMetadata()
            logger.info(
                "[vLLM Hijack] OffloadingConnectorWorker.__init__: "
                "skipped CUDA allocation (simulation mode)"
            )

        def override_register_kv_caches(self, kv_caches):
            """No-op: no actual KV cache tensors to register."""
            pass

        def override_register_cross_layers_kv_cache(self, kv_cache, attn_backend):
            """No-op: no cross-layer tensors needed."""
            pass

        def override_handle_preemptions(self, kv_connector_metadata):
            """No-op: no in-flight transfers to flush."""
            self._unsubmitted_store_jobs.clear()

        def override_start_kv_transfers(self, metadata):
            """Immediately mark all load jobs as complete."""
            self._unsubmitted_store_jobs.clear()
            for job_id, entry in metadata.load_jobs.items():
                self._load_jobs[job_id] = entry.req_id

        def override_prepare_store_kv(self, metadata):
            """Track store jobs for immediate completion reporting."""
            for job_id in metadata.store_jobs:
                self._connector_worker_meta.mark_completed(job_id)

        def override_get_finished(self, finished_req_ids):
            """Report all loads as immediately finished."""
            finished_recving: set[str] = set()
            for job_id, req_id in self._load_jobs.items():
                self._connector_worker_meta.mark_completed(job_id)
                finished_recving.add(req_id)
            self._load_jobs.clear()
            return set(), finished_recving

        def override_build_connector_worker_meta(self):
            """Return completed job IDs since last call."""
            from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
                OffloadingWorkerMetadata,
            )

            if not self._connector_worker_meta.completed_jobs:
                return None
            meta = self._connector_worker_meta
            self._connector_worker_meta = OffloadingWorkerMetadata()
            return meta

        def override_shutdown(self):
            """No-op: nothing to clean up."""
            self._unsubmitted_store_jobs.clear()
            self._load_jobs.clear()

        target.__init__ = override_init
        target.register_kv_caches = override_register_kv_caches
        target.register_cross_layers_kv_cache = override_register_cross_layers_kv_cache
        target.handle_preemptions = override_handle_preemptions
        target.start_kv_transfers = override_start_kv_transfers
        target.prepare_store_kv = override_prepare_store_kv
        target.get_finished = override_get_finished
        target.build_connector_worker_meta = override_build_connector_worker_meta
        target.shutdown = override_shutdown
