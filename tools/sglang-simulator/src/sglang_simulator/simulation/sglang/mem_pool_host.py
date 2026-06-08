from enum import Enum
import torch
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
        self, size_bytes_arr: torch.Tensor, cat: TansportCat
    ) -> torch.Tensor:
        pass


class HicacheTransportOverheadEstimator(HicacheTransportEstimator):
    """
    Estimate hicache transport bandwidth with overhead model, which has a static overhead time.
    """
    def __init__(self, memory_read_bandwidth_bytes, memory_write_bandwidth_bytes):
        super().__init__(memory_read_bandwidth_bytes, memory_write_bandwidth_bytes)

    def est_bandwidth_batch(
        self, size_bytes_arr: torch.tensor, cat: TansportCat
    ) -> torch.tensor:
        x = size_bytes_arr.to(torch.float64)
        if cat == TansportCat.H2D:
            eff = 0.85
            t0 = 6.67e-6
            bw = self.memory_read_bandwidth_bytes * eff
        else:
            eff = 0.85
            t0 = 4e-6
            bw = self.memory_write_bandwidth_bytes * eff
        return x * bw / (t0 * bw + x)


def compute_contiguous_index_lengths(
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
) -> torch.Tensor:
    assert len(host_indices) == len(device_indices)

    host_indices = host_indices.cpu()
    device_indices = device_indices.cpu()

    n = host_indices.numel()
    if n == 0:
        return torch.empty(0, dtype=torch.float64, device="cpu")

    breaks = (
        (host_indices[1:] != host_indices[:-1] + 1)
        | (device_indices[1:] != device_indices[:-1] + 1)
    ).nonzero(as_tuple=False).flatten() + 1

    boundaries = torch.cat(
        [
            torch.tensor([0], dtype=torch.int64),
            breaks.to(torch.int64),
            torch.tensor([n], dtype=torch.int64),
        ]
    )

    return boundaries[1:] - boundaries[:-1]


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

            return original_init(self, *args, **kwargs)
        

        def load_to_device_per_layer(
            self, device_pool, host_indices, device_indices, layer_id, io_backend
        ) -> None:
            if hasattr(self, "_to_page_indices"):
                host_indices = self._to_page_indices(host_indices)
                device_indices = self._to_page_indices(device_indices)

            # Merge cache indices
            # https://github.com/sgl-project/sglang/blob/v0.5.8/sgl-kernel/csrc/kvcacheio/transfer.cu#L713
            seg_len = compute_contiguous_index_lengths(host_indices, device_indices)

            size_per_token = self.get_size_per_token()
            size_bytes_arr = seg_len * size_per_token  # FIXME: per layer
            bandwidth_arr = C_HostKVCacheHook.hicache_transorport_estimator.est_bandwidth_batch(
                size_bytes_arr, cat=TansportCat.H2D
            )
            total_time_cost = float(torch.sum(size_bytes_arr / bandwidth_arr))

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

            size_per_token = self.get_size_per_token()
            size_bytes_arr = seg_len * size_per_token
            bandwidth_arr = C_HostKVCacheHook.hicache_transorport_estimator.est_bandwidth_batch(
                size_bytes_arr, cat=TansportCat.D2H
            )
            total_time_cost = float(torch.sum(size_bytes_arr / bandwidth_arr))

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
