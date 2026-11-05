"""TPU connector CPU data-plane adapters.

Scheduler-side lookup and connector metadata remain native. Worker-side data
movement is replaced by explicit load/store completion sets so replay can run
without CUDA/TPU memory, SRPC, RDMA, or a real decode peer.

多组 Raiden producer 是上述原生 scheduler 策略的例外：单 CPU、无真实
D 端时使用本地完成契约；V6D sibling 的查询、保存和延迟释放仍由原生管理。
"""

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger


logger = get_logger()


def _metadata_with(args, kwargs, field):
    for value in (*args, *kwargs.values()):
        if hasattr(value, field):
            return value
    return None


class C_TPURaidenConnectorHook(BaseHook):
    """单 CPU 多组 producer 的本地完成契约，不模拟真实 P→D 交接。"""

    HOOK_MODULE_NAME = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
    HOOK_CLASS_NAME = "TPURaidenConnector"

    @classmethod
    def hook(cls, target):
        if target.__dict__.get("_sim_local_raiden_hook", False):
            return
        original_init = target.__init__
        original_admit = target.on_new_request
        original_match = target.get_num_new_matched_tokens
        original_alloc = target.update_state_after_alloc
        original_meta = target.build_connector_meta
        original_finish = target.request_finished
        original_finish_groups = target.request_finished_all_groups

        def init(self, vllm_config, role, kv_cache_config):
            groups = kv_cache_config.kv_cache_groups
            transfer = vllm_config.kv_transfer_config
            local = transfer.kv_role == "kv_producer" and len(groups) > 1
            if local:
                parallel = vllm_config.parallel_config
                dimensions = (
                    "tensor_parallel_size", "pipeline_parallel_size",
                    "prefill_context_parallel_size", "decode_context_parallel_size",
                    "data_parallel_size", "nnodes",
                )
                if any(getattr(parallel, name, 1) != 1 for name in dimensions):
                    raise ValueError("Local Raiden simulation requires single-worker topology")
            original_init(self, vllm_config, role, kv_cache_config)
            self._sim_local_raiden = local
            self._sim_raiden_num_groups = len(groups)
            if local and self.connector_scheduler is not None:
                logger.info(
                    "[Raiden CPU] local-only producer enabled: kv_groups=%d; "
                    "no remote handoff or Raiden block pinning; V6D lifecycle unchanged",
                    len(groups),
                )

        def admit(self, request):
            if not self._sim_local_raiden:
                return original_admit(self, request)
            params = getattr(request, "kv_transfer_params", None)
            if params is not None and not isinstance(params, dict):
                raise ValueError("Local Raiden simulation requires dict kv_transfer_params")
            remote_fields = (
                "do_remote_decode", "do_remote_prefill", "ali_llumnix_disagg",
                "remote_host", "remote_port", "remote_engine_id",
                "remote_bootstrap_addr", "remote_block_ids", "remote_side_channel_port",
                "src_controller_address", "src_engine_id", "src_job_name", "uuid",
                "transfer_id",
            )
            if any((params or {}).get(key) for key in remote_fields):
                raise ValueError(
                    "Local Raiden simulation does not support remote PD handoff; "
                    "disable DashLLM remote decode routing and omit remote transfer params"
                )
            # 没有 D 端重算最后一个 token，不能执行原生 stage3 的 prompt 截断。

        def matched(self, request, num_computed_tokens):
            if not self._sim_local_raiden:
                return original_match(self, request, num_computed_tokens)
            # Raiden 不提供远端命中；MultiConnector 仍继续查询 V6D。
            return 0, False

        def allocated(self, request, blocks, num_external_tokens):
            if not self._sim_local_raiden:
                return original_alloc(self, request, blocks, num_external_tokens)
            if num_external_tokens != 0:
                raise ValueError("Local Raiden simulation cannot load remote PD tokens")
            # 不登记原生 stage3 register-ahead 或真实 send/load 任务。

        def metadata(self, scheduler_output):
            if not self._sim_local_raiden:
                return original_meta(self, scheduler_output)
            # 延迟导入：hook 安装时目标模块尚未完成类定义。
            from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
                TPUConnectorMetadata,
            )
            return TPUConnectorMetadata()

        def finished(self, request, block_ids):
            if not self._sim_local_raiden:
                return original_finish(self, request, block_ids)
            return False, None

        def finished_groups(self, request, block_ids):
            if not self._sim_local_raiden:
                return original_finish_groups(self, request, block_ids)
            if len(block_ids) != self._sim_raiden_num_groups:
                raise ValueError(
                    "Raiden simulation KV group count mismatch: "
                    f"expected {self._sim_raiden_num_groups}, got {len(block_ids)}"
                )
            # 不压平、不取第一组，也不直接释放任何 block。此 Raiden 子连接器
            # 没有未完成传输；原生 MultiConnector 仍聚合 V6D 的 delay_free，
            # 最终由 Scheduler 统一释放全部组，避免虚构完成事件或抢先释放。
            return False, None

        target.__init__ = init
        target.on_new_request = admit
        target.get_num_new_matched_tokens = matched
        target.update_state_after_alloc = allocated
        target.build_connector_meta = metadata
        target.request_finished = finished
        target.request_finished_all_groups = finished_groups
        target._sim_local_raiden_hook = True


def install_tpu_v6d_completion_guard(scheduler):
    """在已结算的 scheduler 回调中区分保存作业确认和请求级释放许可。"""
    if getattr(scheduler, "_sim_v6d_completion_guard", False):
        return
    connector = getattr(scheduler, "connector", None)
    if connector is None:
        return
    children = getattr(connector, "_connectors", (connector,))
    v6d_count = 0
    for child in children:
        child_type = type(child)
        if (
            child_type.__module__ == "kvs_connector.v6d_object_connector"
            and child_type.__name__ == "TPUV6dObjectConnector"
        ):
            v6d_count += 1
        elif (
            child_type.__module__ == "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
            and child_type.__name__ == "TPURaidenConnector"
            and getattr(child, "_sim_local_raiden", False)
        ):
            continue
        else:
            # 其他发送者有独立的 MultiConnector 完成计数，不能混用本适配。
            return
    if v6d_count != 1:
        return

    original_finished = scheduler._connector_finished
    original_update = connector.update_connector_output
    pending_release = set()

    def request_finished(request):
        # 以原生返回值为准；仅请求终结且确实延迟释放才形成发送义务。
        result = original_finished(request)
        if result[0]:
            pending_release.add(request.request_id)
        else:
            pending_release.discard(request.request_id)
        return result

    def update_connector_output(output):
        incoming = set(output.finished_sending or ())
        live = incoming.intersection(scheduler.requests)
        # 已经释放的请求不能再次进入原生发布/清理，也不能再次释放。
        pending_release.difference_update(incoming - live)
        output.finished_sending = live or None
        try:
            # 先确认作业：中间 chunk 的缓存发布、内部状态清理仍归原生 V6D。
            original_update(output)
        except Exception:
            output.finished_sending = incoming or None
            raise
        # 此时已执行 update_from_output 的 token/终态更新，PP 早投递不会
        # 把 dispatch 时的状态误当成结算状态。不能用当步 finished_req_ids
        # 代替此处跨步保存的发送义务（它只是增量集合）。
        completed = {
            rid for rid in (output.finished_sending or ())
            if rid in pending_release
            and rid in scheduler.requests
            and scheduler.requests[rid].is_finished()
        }
        pending_release.difference_update(completed)
        output.finished_sending = completed or None
        # 不改变 finished_recving、worker metadata 或统计，不自行 free blocks。

    scheduler._connector_finished = request_finished
    connector.update_connector_output = update_connector_output
    scheduler._sim_v6d_completion_guard = True
    logger.info(
        "[V6D CPU] scheduler completion guard enabled: intermediate store acknowledgements "
        "do not release requests; final releases follow native delay_free"
    )


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
