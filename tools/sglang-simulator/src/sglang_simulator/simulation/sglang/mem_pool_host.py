from enum import Enum
import torch
import numpy as np
from functools import lru_cache

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import ConfigManager, StateManager
from sglang_simulator.utils import get_logger

logger = get_logger()


class TansportCat(Enum):
    H2D = "H2D"
    D2H = "D2H"


class HicacheTransportEstimator:
    def __init__(
        self, memory_read_bandwidth_bytes: float, memory_write_bandwidth_bytes: float
    ):
        self.memory_read_bandwidth_bytes = memory_read_bandwidth_bytes
        self.memory_write_bandwidth_bytes = memory_write_bandwidth_bytes

    def est_bandwidth_batch(
        self, size_bytes_arr: np.ndarray, cat: TansportCat
    ) -> np.ndarray:
        pass


class HicacheTransportOverheadEstimator(HicacheTransportEstimator):
    """
    Estimate hicache transport bandwidth with overhead model, which has a static overhead time.
    """
    def __init__(self, memory_read_bandwidth_bytes, memory_write_bandwidth_bytes):
        super().__init__(memory_read_bandwidth_bytes, memory_write_bandwidth_bytes)

    def est_bandwidth_batch(
        self, size_bytes_arr: np.ndarray, cat: TansportCat
    ) -> np.ndarray:
        if cat == TansportCat.H2D:
            eff = 0.85
            t0 = 6.67e-6
            bw = self.memory_read_bandwidth_bytes * eff
        else:
            eff = 0.85
            t0 = 4e-6
            bw = self.memory_write_bandwidth_bytes * eff
        return size_bytes_arr * bw / (t0 * bw + size_bytes_arr)


def compute_contiguous_index_lengths(
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
) -> np.ndarray:
    assert len(host_indices) == len(device_indices)

    num_indices = len(host_indices)

    host = np.asarray(host_indices.cpu(), dtype=np.int64)
    dev = np.asarray(device_indices.cpu(), dtype=np.int64)
    cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
    cut = np.flatnonzero(~cont) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, num_indices]
    seg_len = (ends - starts).astype(np.float64)

    return seg_len


def alloc_with_pin_memory(
    dims,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag, 
    and the device will be "meta" to reduce memory usage.
    """
    buffer = torch.empty(dims, dtype=dtype, device="meta", pin_memory=False)
    return buffer


@lru_cache(maxsize=256)
def get_refined_cache_size_per_token(
    host_pool
) -> float:
    scheduler_config = ConfigManager.get_scheduler_config()

    pool_name = host_pool.__class__.__name__
    if hasattr(host_pool, "pool_name"):
        pool_name += f"[{getattr(host_pool, "pool_name")}]"
    inter_dtype = host_pool.dtype
    inter_size_per_token = host_pool.get_size_per_token()

    if scheduler_config is None or scheduler_config.kv_cache_data_type is None:

        logger.warning(
            f"Failed to get the user-customized scheduler configuration. "
            f"The dtype of host pool[{pool_name}] will use internal dtype [{inter_dtype}]. "
            f"Please make sure this value is correct, because initialization depends on the actual platform."
        )
        return inter_size_per_token

    dtype_factor = scheduler_config.kv_cache_data_type.bytes / inter_dtype.itemsize
    size_per_token = inter_size_per_token * dtype_factor
    logger.info(f"Size per token for {pool_name}: {size_per_token}")

    return size_per_token


class C_HostKVCacheHook(BaseHook):
    HOOK_CLASS_NAME = r".*?"
    HOOK_MODULE_NAME = r"sglang\.srt\.mem_cache\.memory_pool_host$"
    REGEX = True

    hicache_transorport_estimator = None

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        # The subclass of HostKVCache
        if len(target.__mro__) < 3 or target.__mro__[-3].__name__ != "HostKVCache":
            # object => ABC => HostKVCache
            return

        def wrapped_init(self, *args, **kwargs):

            from collections import defaultdict
            from sglang.srt.mem_cache import memory_pool_host
            memory_pool_host.ALLOC_MEMORY_FUNCS = defaultdict(lambda: alloc_with_pin_memory, {})

            C_HostKVCacheHook.hicache_transorport_estimator = HicacheTransportOverheadEstimator(
                memory_read_bandwidth_bytes=ConfigManager.get_platform_config().memory_read_bandwidth,
                memory_write_bandwidth_bytes=ConfigManager.get_platform_config().memory_write_bandwidth,
            )

            import psutil
            psutil_virtual_memory = psutil.virtual_memory
            # SGLang preserves at least 10 GB for other usage, which might cause HiCache initialization to fail.
            psutil.virtual_memory = lambda: psutil_virtual_memory()._replace(
                available=1024 * (1024**3)
            )
            ret = original_init(self, *args, **kwargs)
            psutil.virtual_memory = psutil_virtual_memory
            return ret

        def load_to_device_per_layer(
            self, device_pool, host_indices, device_indices, layer_id, io_backend
        ) -> None:
            if hasattr(self, "_to_page_indices"):
                host_indices = self._to_page_indices(host_indices)
                device_indices = self._to_page_indices(device_indices)

            # Merge cache indices
            # https://github.com/sgl-project/sglang/blob/v0.5.8/sgl-kernel/csrc/kvcacheio/transfer.cu#L713
            seg_len = compute_contiguous_index_lengths(host_indices, device_indices)

            size_per_token = get_refined_cache_size_per_token(self)
            size_bytes_arr = seg_len * size_per_token / (self.layer_num if hasattr(self, "layer_num") else 1)
            bandwidth_arr = C_HostKVCacheHook.hicache_transorport_estimator.est_bandwidth_batch(
                size_bytes_arr, cat=TansportCat.H2D
            )
            total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))

            StateManager.inc_hicache_l2_load_dur(total_time_cost)

        def backup_from_device_all_layer(
            self, device_pool, host_indices, device_indices, io_backend
        ) -> None:
            """
            Backup KV data from the device memory pool to the host memory pool for all layers.
            """
            if hasattr(self, "_to_page_indices"):
                host_indices = self._to_page_indices(host_indices)
                device_indices = self._to_page_indices(device_indices)

            seg_len = compute_contiguous_index_lengths(host_indices, device_indices)
            
            size_per_token = get_refined_cache_size_per_token(self)
            size_bytes_arr = seg_len * size_per_token
            bandwidth_arr = C_HostKVCacheHook.hicache_transorport_estimator.est_bandwidth_batch(
                size_bytes_arr, cat=TansportCat.D2H
            )
            total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))

            StateManager.inc_hicache_l2_backup_dur(total_time_cost)

        def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
            """
            Get a flat data page from the host memory pool.
            """
            return torch.ones(size=(1, 1)) * index

        def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
            """
            Set a flat data page to the host memory pool.
            """
            pass

        target.__init__ = wrapped_init
        target.load_to_device_per_layer = load_to_device_per_layer
        target.backup_from_device_all_layer = backup_from_device_all_layer
        target.get_data_page = get_data_page
        target.set_from_flat_data_page = set_from_flat_data_page
