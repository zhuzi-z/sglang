"""TPU connector CPU data-plane adapters.

Scheduler-side lookup and connector metadata remain native. Worker-side data
movement is replaced by explicit load/store completion sets so replay can run
without CUDA/TPU memory, SRPC, RDMA, or a real decode peer.
"""

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger


logger = get_logger()


def _metadata_with(args, kwargs, field):
    for value in (*args, *kwargs.values()):
        if hasattr(value, field):
            return value
    return None


class _TPUConnectorWorkerCPUHook(BaseHook):
    """Shared lifecycle implementation for TPU v6d and Raiden workers."""

    @classmethod
    def _ensure_state(cls, worker):
        if not hasattr(worker, "_sim_tpu_load_done"):
            worker._sim_tpu_load_done = set()
        if not hasattr(worker, "_sim_tpu_store_done"):
            worker._sim_tpu_store_done = set()

    @classmethod
    def hook(cls, target):
        def start_load(self, *args, **kwargs):
            cls._ensure_state(self)
            metadata = _metadata_with(args, kwargs, "reqs_to_load")
            for req_id, groups in (getattr(metadata, "reqs_to_load", {}) or {}).items():
                # Empty key sets do not represent an external load.
                if any(len(keys) > 0 for keys, _ids in groups.values()):
                    self._sim_tpu_load_done.add(req_id)
            return None

        def start_store(self, *args, **kwargs):
            cls._ensure_state(self)
            metadata = _metadata_with(args, kwargs, "reqs_to_store")
            for req_id in (getattr(metadata, "reqs_to_store", {}) or {}):
                self._sim_tpu_store_done.add(req_id)
            return None

        def get_finished(self, *args, **kwargs):
            cls._ensure_state(self)
            loaded = set(self._sim_tpu_load_done)
            stored = set(self._sim_tpu_store_done)
            self._sim_tpu_load_done.clear()
            self._sim_tpu_store_done.clear()
            # vLLM/TPU contract: (finished_sending, finished_recving).
            return stored, loaded

        def has_pending_push_work(self):
            cls._ensure_state(self)
            return bool(self._sim_tpu_load_done or self._sim_tpu_store_done)

        def get_finished_count(self):
            cls._ensure_state(self)
            return len(self._sim_tpu_load_done) + len(self._sim_tpu_store_done)

        def register_kv_caches(self, kv_caches, *args, **kwargs):
            cls._ensure_state(self)
            self.kv_caches = kv_caches
            # Do not call native registration on fake 4-KiB KV tensors: the
            # native contract requires full storage views and byte assertions.
            return None

        def register_cross_layers_kv_cache(self, kv_caches, *args, **kwargs):
            cls._ensure_state(self)
            self.kv_caches = kv_caches
            return None

        def noop(self, *args, **kwargs):
            return None

        def empty_errors(self, *args, **kwargs):
            return []

        replacements = {
            "start_load_kv": start_load,
            "start_store_kv": start_store,
            "get_finished": get_finished,
            "has_pending_push_work": has_pending_push_work,
            "get_finished_count": get_finished_count,
            "register_kv_caches": register_kv_caches,
            "register_cross_layers_kv_cache": register_cross_layers_kv_cache,
            "wait_for_jobs": noop,
            "save_kv_layer": noop,
            "handle_preemptions": noop,
            "get_block_ids_with_load_errors": empty_errors,
            "get_handshake_metadata": noop,
            "set_xfer_handshake_metadata": noop,
            "set_xfer_handshake_metadata_pp_aware": noop,
            "register_runner": noop,
            "shutdown": noop,
            "_register_v6d_host_memory": noop,
            "_async_v6d_init_and_notify": noop,
        }
        for name, replacement in replacements.items():
            if getattr(target, name, None) is not None:
                setattr(target, name, replacement)

        if getattr(target, "_start_async_v6d_init", None) is not None:
            def start_async_v6d_init(self):
                cls._ensure_state(self)
                self._load_handler = None
                self._store_handler = None
                return None

            target._start_async_v6d_init = start_async_v6d_init

        logger.info("[V6D Hijack] installed TPU CPU lifecycle hooks on %s", target)


class C_TPUV6dObjectConnectorWorkerHook(_TPUConnectorWorkerCPUHook):
    HOOK_MODULE_NAME = "kvs_connector.v6d_object_connector"
    HOOK_CLASS_NAME = "V6dObjectConnectorWorker"


class C_TPURaidenConnectorWorkerHook(_TPUConnectorWorkerCPUHook):
    HOOK_MODULE_NAME = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
    HOOK_CLASS_NAME = "TPURaidenConnectorWorker"

    @classmethod
    def hook(cls, target):
        super().hook(target)

        def process_send_load(
            self, metadata, wait_for_completion=False, report_completion=True
        ):
            cls._ensure_state(self)
            if self.is_producer:
                for req_id in (getattr(metadata, "reqs_to_send", {}) or {}):
                    self._sim_tpu_store_done.add(req_id)
                return None
            if not report_completion:
                return None
            for req_id, req_meta in (getattr(metadata, "reqs_to_load", {}) or {}).items():
                if getattr(req_meta, "report_completion", True):
                    self._sim_tpu_load_done.add(req_id)
            return None

        target.process_send_load = process_send_load
