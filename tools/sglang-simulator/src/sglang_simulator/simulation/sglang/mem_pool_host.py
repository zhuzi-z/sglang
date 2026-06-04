from enum import Enum
import inspect
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


hicache_transorport_estimator = HicacheTransportOverheadEstimator(
    memory_read_bandwidth_bytes=ConfigManager.get_platform_config().memory_read_bandwidth,
    memory_write_bandwidth_bytes=ConfigManager.get_platform_config().memory_write_bandwidth,
)


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


class C_HostKVCacheHook(BaseHook):
    HOOK_CLASS_NAME = r"MHATokenToKVPoolHost|MLATokenToKVPoolHost|DeepSeekV4PagedHostPool|DeepSeekV4StateHostPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool_host"
    REGEX = True

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_get_size_per_token = target.get_size_per_token

        def wrapped_init(self, *args, **kwargs):
            # Disable pin memory, which might fail on CPU platforms.
            sig = inspect.signature(original_init)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()

            if "pin_memory" in bound.arguments:
                bound.arguments["pin_memory"] = False
            
            return original_init(*bound.args, **bound.kwargs)
        

        def load_to_device_per_layer(
            self, device_pool, host_indices, device_indices, layer_id, io_backend
        ) -> None:
            if hasattr(self, "_to_page_indices"):
                host_indices = self._to_page_indices(host_indices)
                device_indices = self._to_page_indices(device_indices)

            # Merge cache indices
            # https://github.com/sgl-project/sglang/blob/v0.5.8/sgl-kernel/csrc/kvcacheio/transfer.cu#L713
            seg_len = compute_contiguous_index_lengths(host_indices, device_indices)

            size_bytes_arr = seg_len * self.get_size_per_token()  # FIXME: per layer
            bandwidth_arr = hicache_transorport_estimator.est_bandwidth_batch(
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

            size_bytes_arr = seg_len * self.get_size_per_token()
            bandwidth_arr = hicache_transorport_estimator.est_bandwidth_batch(
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

        def get_size_per_token(self):
            if self.__class__.__name__ == "DeepSeekV4PagedHostPool":
                if self.pool_name in ["swa"]:
                    return self.size_per_token * 130
                elif self.pool_name in ["deepseek_v4_c4"]:
                    return self.size_per_token * 65
                elif self.pool_name in ["deepseek_v4_c4_indexer"]:
                    return self.size_per_token * 132
                elif self.pool_name in ["deepseek_v4_c128"]:
                    return self.size_per_token * 3
                else:
                    # return self.size_per_token
                    raise ValueError(f"[DeepSeekV4PagedHostPool] unsupport pool name: {self.pool_name}")
            elif self.__class__.__name__ == "DeepSeekV4StateHostPool":
                if self.pool_name in ["deepseek_v4_c4_state", "deepseek_v4_c128_state"]:
                    return self.size_per_token * 256
                elif self.pool_name in [
                    "deepseek_v4_indexer_state",
                    "deepseek_v4_c4_indexer_state",
                ]:
                    return self.size_per_token * 128
                else:
                    # return self.size_per_token
                    raise ValueError(f"[DeepSeekV4StateHostPool] unsupport pool name: {self.pool_name}")
            else:
                return original_get_size_per_token(self)

        target.__init__ = wrapped_init
        target.load_to_device_per_layer = load_to_device_per_layer
        target.backup_from_device_all_layer = backup_from_device_all_layer
        target.get_data_page = get_data_page
        target.set_from_flat_data_page = set_from_flat_data_page
        target.get_size_per_token = get_size_per_token
