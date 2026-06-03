from queue import Empty
from typing import Optional, Tuple

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import ConfigManager, StateManager
from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager


# Pool name strings used by the v2 storage API (mirrors
# sglang.srt.mem_cache.hicache_storage.PoolName).
_MAMBA_POOL_NAMES = ("MAMBA", "PoolName.MAMBA", "mamba")


class C_HiCacheController(BaseHook):
    HOOK_CLASS_NAME = "HiCacheController"
    HOOK_MODULE_NAME = "sglang.srt.managers.cache_controller"

    KV_CACHE_BYTES: Optional[int] = None
    DISK_READ_BANDWIDTH_BYTES: Optional[float] = None
    DISK_WRITE_BANDWIDTH_BYTES: Optional[float] = None
    # Bytes for one full mamba state slot, across all linear/mamba layers.
    # Lazily resolved from the registered MambaPoolHost; -1 means "not resolved
    # yet"; 0 means "no mamba pool present" (KV-only model — skip mamba math).
    MAMBA_BYTES_PER_SLOT: Optional[int] = None

    # ------------------------------------------------------------------
    # Pure helpers — exposed as staticmethods so unit tests can call them
    # without instantiating a real HiCacheController.
    # ------------------------------------------------------------------

    @staticmethod
    def calc_prefetch_pages(
        required_pages: int, page_size_byte: int, max_dur: float, bandwidth: float
    ) -> tuple[float, float]:
        _prefetch_dur = required_pages * page_size_byte / bandwidth
        if _prefetch_dur > max_dur:
            _completed_pages = max(max_dur * bandwidth / page_size_byte, 1)
            return _completed_pages, max_dur
        else:
            return required_pages, _prefetch_dur

    @staticmethod
    def resolve_mamba_bytes_per_slot(controller) -> int:
        """Probe the storage backend's v2 pool registry for MambaPoolHost,
        compute total bytes for one slot (one request's mamba state across
        every linear layer). Result is cached on the class.

        Returns 0 if no mamba pool was registered (e.g. KV-only model)."""
        if C_HiCacheController.MAMBA_BYTES_PER_SLOT is not None:
            return C_HiCacheController.MAMBA_BYTES_PER_SLOT
        bytes_ = 0
        backend = getattr(controller, "storage_backend", None)
        reg = getattr(backend, "registered_pools", {}) if backend is not None else {}
        mp = None
        for name in _MAMBA_POOL_NAMES:
            if name in reg:
                mp = reg[name]
                break
        if mp is not None:
            try:
                per_layer = (
                    int(mp.temporal_state_elem_size)
                    * int(mp.temporal_dtype.itemsize)
                    + int(sum(mp.conv_state_elem_sizes))
                    * int(mp.conv_dtype.itemsize)
                )
                bytes_ = per_layer * int(mp.num_mamba_layers)
            except Exception:
                bytes_ = 0
        C_HiCacheController.MAMBA_BYTES_PER_SLOT = bytes_
        return bytes_

    @staticmethod
    def operation_has_mamba_hit(operation) -> bool:
        """True iff `operation` requested a mamba pool transfer AND the storage
        hit query reported at least one mamba page hit (extra_pool_hit_pages)."""
        if not getattr(operation, "pool_transfers", None):
            return False
        has_mamba_transfer = False
        for t in operation.pool_transfers:
            name = getattr(t, "name", "")
            name_str = str(getattr(name, "value", name))
            if name_str in _MAMBA_POOL_NAMES or name_str == "MAMBA":
                has_mamba_transfer = True
                break
        if not has_mamba_transfer:
            return False
        ppr = getattr(operation, "pool_storage_result", None)
        if ppr is None:
            return False
        extra = getattr(ppr, "extra_pool_hit_pages", {}) or {}
        for name in _MAMBA_POOL_NAMES:
            if extra.get(name, 0) > 0:
                return True
        return False

    @staticmethod
    def drain_operation(
        operation,
        storage_hit_count: int,
        remain_dur: float,
        kv_bytes_per_token: int,
        disk_read_bw: float,
    ) -> Tuple[float, bool]:
        """Drain one prefetch operation against the given inference budget.

        Order: KV pages first, then mamba state slot (per real sglang's prefetch
        order — see `_page_transfer` calls super after batch_get_v2). When KV
        pages don't all fit this iter, the op stays chunked with KV still pending.
        When KV is done but mamba doesn't fit, the op stays chunked with
        `_mamba_pending_bytes > 0` so the next iter drains the leftover.

        Returns:
            (new_remain_dur, fully_done) — `fully_done=False` means the caller
            should keep the op in `chunked_prefetch_operation`; `True` means
            mark_terminate + release host indices.
        """
        # --- KV phase ---
        kv_remaining = storage_hit_count - int(getattr(operation, "completed_tokens", 0))
        if kv_remaining > 0:
            completed_tokens, prefetch_dur = C_HiCacheController.calc_prefetch_pages(
                kv_remaining, kv_bytes_per_token, remain_dur, disk_read_bw
            )
            if completed_tokens < kv_remaining:
                # KV not finished — stay chunked, no mamba progress this iter
                operation.completed_tokens = (
                    int(getattr(operation, "completed_tokens", 0)) + int(completed_tokens)
                )
                return 0.0, False
            operation.completed_tokens = int(storage_hit_count)
            remain_dur -= prefetch_dur

        # --- Mamba phase ---
        pending = float(getattr(operation, "_mamba_pending_bytes", 0))
        if pending <= 0:
            return remain_dur, True

        needed_time = pending / disk_read_bw
        if needed_time <= remain_dur:
            # Mamba completes this iter
            operation._mamba_pending_bytes = 0
            return remain_dur - needed_time, True

        # Mamba partial — drain what we can, stay chunked
        if remain_dur > 0:
            consumed_bytes = remain_dur * disk_read_bw
            operation._mamba_pending_bytes = pending - consumed_bytes
        return 0.0, False

    @classmethod
    def hook(cls, target):

        original_terminate_prefetch = target.terminate_prefetch
        original_storage_hit_query = target._storage_hit_query

        def override_backup_thread_func(self, *args, **kwargs):
            # Async thread: perform no action
            # The action will be performed by `handle_backup_operation`
            pass

        def override_prefetch_thread_func(self, *args, **kwargs):
            # Async thread: perform no action
            # The action will be performed by `handle_prefetch_operation`
            pass

        def handle_backup_operation(self):
            if not self.enable_storage:
                return
            while True:
                try:
                    operation = self.backup_queue.get(block=False)
                    if operation is None:
                        return

                    if not self.backup_skip:
                        self._page_backup(operation)
                    # NOTE: backup time intentionally not tracked — assumes the
                    # L2->L3 (host->disk) write is fully hidden behind subsequent
                    # inference iterations.
                    self.ack_backup_queue.put(operation)

                except Empty:
                    return

        def handle_prefetch_operation(self):
            if not self.enable_storage:
                return

            if C_HiCacheController.KV_CACHE_BYTES is None:
                C_HiCacheController.KV_CACHE_BYTES = ConfigManager.get_kv_cache_bytes()
            if C_HiCacheController.DISK_READ_BANDWIDTH_BYTES is None:
                C_HiCacheController.DISK_READ_BANDWIDTH_BYTES = (
                    ConfigManager.get_platform_config().disk_read_bandwidth
                )

            kv_bytes = C_HiCacheController.KV_CACHE_BYTES
            disk_bw = C_HiCacheController.DISK_READ_BANDWIDTH_BYTES

            remain_dur = StateManager.get_current_inference_dur()

            # --- Section 1: resume chunked op from previous iter ---
            chunked_prefetch_operation = getattr(
                self, "chunked_prefetch_operation", None
            )
            if chunked_prefetch_operation is not None:
                operation = chunked_prefetch_operation["operation"]
                storage_hit_count = chunked_prefetch_operation["storage_hit_count"]
                if operation._terminated_flag:
                    setattr(self, "chunked_prefetch_operation", None)
                    self.append_host_mem_release(
                        operation.host_indices[int(operation.completed_tokens) :]
                    )
                else:
                    remain_dur, done = C_HiCacheController.drain_operation(
                        operation, storage_hit_count, remain_dur, kv_bytes, disk_bw
                    )
                    if done:
                        operation.mark_terminate()
                        setattr(self, "chunked_prefetch_operation", None)
                        self.append_host_mem_release(
                            operation.host_indices[storage_hit_count:]
                        )

            # --- Section 2: drain queue ---
            while remain_dur > 0:
                try:
                    operation = self.prefetch_queue.get(block=False)
                    if operation is None:
                        return

                    if operation._terminated_flag:
                        self.append_host_mem_release(
                            operation.host_indices[int(operation.completed_tokens) :]
                        )
                        continue

                    hash_value, storage_hit_count = self._storage_hit_query(operation)
                    if (
                        self.prefetch_threshold is not None
                        and storage_hit_count < self.prefetch_threshold
                    ):
                        operation.mark_terminate()
                        self.append_host_mem_release(operation.host_indices)
                        continue

                    operation.hash_value = hash_value[
                        : (storage_hit_count // self.page_size)
                    ]
                    storage_hit_count = (
                        storage_hit_count // self.page_size * self.page_size
                    )

                    # Initialize this op's mamba pending bytes based on the
                    # hit query result (only counts when mamba was actually hit).
                    operation.completed_tokens = 0
                    if not hasattr(operation, "_mamba_pending_bytes"):
                        operation._mamba_pending_bytes = (
                            C_HiCacheController.resolve_mamba_bytes_per_slot(self)
                            if C_HiCacheController.operation_has_mamba_hit(operation)
                            else 0
                        )

                    remain_dur, done = C_HiCacheController.drain_operation(
                        operation, storage_hit_count, remain_dur, kv_bytes, disk_bw
                    )
                    if done:
                        operation.mark_terminate()
                        self.append_host_mem_release(
                            operation.host_indices[operation.completed_tokens :]
                        )
                    else:
                        # Stay chunked — request remains in waiting_queue, the
                        # scheduler's 5ms idle-tick (wrapped_get_new_batch_prefill)
                        # will advance global_clock until prefetch completes.
                        setattr(
                            self,
                            "chunked_prefetch_operation",
                            {
                                "operation": operation,
                                "storage_hit_count": storage_hit_count,
                            },
                        )
                        # remain_dur is 0 from drain_operation
                        break

                except Empty:
                    return

        def override_generic_page_set(
            self, hash_values, host_indices, extra_info=None
        ) -> bool:
            # Always pass extra_info to storage_backend.
            data = [
                self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
                for i in range(len(hash_values))
            ]
            return self.storage_backend.batch_set(hash_values, data, extra_info)

        def wrapped_terminate_prefetch(self, operator):
            result = original_terminate_prefetch(self, operator)
            req_stats = request_stats_manager.get_req_stats(operator.request_id)
            req_stats.final_storage_hit_len = result[0]
            return result

        def wrapped_storage_hit_query(self, operator):
            result = original_storage_hit_query(self, operator)
            req_stats = request_stats_manager.get_req_stats(operator.request_id)
            req_stats.recv_storage_hit_len = result[1]
            return result

        target.prefetch_thread_func = override_prefetch_thread_func
        target.backup_thread_func = override_backup_thread_func
        target.handle_backup_operation = handle_backup_operation
        target.handle_prefetch_operation = handle_prefetch_operation
        target._generic_page_set = override_generic_page_set
        target.terminate_prefetch = wrapped_terminate_prefetch
        target.storage_hit_query = wrapped_storage_hit_query
