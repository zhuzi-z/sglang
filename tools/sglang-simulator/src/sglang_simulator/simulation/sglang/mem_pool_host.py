from typing import Optional

import numpy as np
import torch
from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import ConfigManager, StateManager
from sglang_simulator.utils import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# HiCache transfer model — calibrated from hicache_h2d_profile.py + d2h_profile.py
# on DSV4-Flash-FP8, TP=8, H20, hicache_ratio=2.0, page_size=256.
#
# Aggregate per-request behavior (matches measurement, R²≥0.998):
#   total_h2d_time = total_h2d_bytes / 27.4e9    (peak BW from time-vs-bytes fit)
#   total_d2h_time = total_d2h_bytes / 33.89e9
# Each pool's per-layer call contributes its own bytes; with a single per-
# direction BW the per-pool sum collapses to total_bytes / BW. So the simulator
# can keep its per-pool serial-accumulation accounting and still match measured
# aggregate latency.
# ---------------------------------------------------------------------------
HICACHE_H2D_PEAK_BW_BYTES = 27.4e9   # GB/s effective, includes per-call overhead
HICACHE_D2H_PEAK_BW_BYTES = 33.89e9  # write direction is faster (asymmetry)
HICACHE_PER_CALL_OVERHEAD_S = 0.0    # overhead is absorbed into the BW above

# Per-pool per-layer bytes per token. Total across all 7 active pools sums to
# ~76,734 bytes per loaded token (regression R²=1.0000 on first-chunk data).
# Within {compute group} and {state group}, individual per-pool bytes are not
# separately identifiable from the measurement (loaded_n is identical group-wide),
# so we split the group total evenly by layer count.
# NB: 'swa' lives in PagedHostPool but its H2D loaded_n caps like a state pool;
#     the per-token byte size we still treat as state-class (~424).
_BYTES_PER_TOKEN_PER_LAYER_PAGED = {
    "deepseek_v4_c128": 65,         # 1300 (group total) / 20 layers
    "deepseek_v4_c4": 62,           # 1300 / 21
    "deepseek_v4_c4_indexer": 62,   # 1300 / 21
    "swa": 424,                     # 18209 / 43
}
_BYTES_PER_TOKEN_PER_LAYER_STATE = {
    "deepseek_v4_c128_state": 911,         # 18209 / 20
    "deepseek_v4_c4_state": 867,           # 18209 / 21
    "deepseek_v4_c4_indexer_state": 867,   # 18209 / 21
}


def _hicache_transfer_time(size_bytes_arr: np.ndarray, cat: str) -> np.ndarray:
    """Compute per-segment transfer time in seconds.

    Model: time = per_call_overhead + bytes / peak_bw
    Returned as an effective bandwidth array (so callers can keep using
    time = size / bw): bw(x) = x * peak_bw / (t0 * peak_bw + x).
    """
    x = size_bytes_arr.astype(np.float64)
    bw = HICACHE_H2D_PEAK_BW_BYTES if cat == "H2D" else HICACHE_D2H_PEAK_BW_BYTES
    t0 = HICACHE_PER_CALL_OVERHEAD_S
    if t0 == 0:
        # Avoid divide-by-zero and just return the constant peak bandwidth
        return np.full_like(x, bw)
    return x * bw / (t0 * bw + x)


class C_MHATokenToKVPoolHostHook(BaseHook):
    HOOK_CLASS_NAME = "MHATokenToKVPoolHost"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool_host"

    KV_CACHE_BYTES: Optional[int] = None
    KV_CACHE_BYTES_PER_LAYER: Optional[int] = None
    MEMORY_READ_BANDWIDTH_BYTES: Optional[float] = None
    MEMORY_WRITE_BANDWIDTH_BYTES: Optional[float] = None

    @classmethod
    def hook(cls, target):

        def est_bandwidth_batch(size_bytes_arr: np.ndarray, cat: str):
            if cls.MEMORY_READ_BANDWIDTH_BYTES is None:
                cls.MEMORY_READ_BANDWIDTH_BYTES = (
                    ConfigManager.get_platform_config().memory_read_bandwidth
                )
            if cls.MEMORY_WRITE_BANDWIDTH_BYTES is None:
                cls.MEMORY_WRITE_BANDWIDTH_BYTES = (
                    ConfigManager.get_platform_config().memory_write_bandwidth
                )
            x = size_bytes_arr.astype(np.float64)
            if cat == "H2D":
                eff = 0.85
                t0 = 6.67e-6
                bw = cls.MEMORY_READ_BANDWIDTH_BYTES * eff
            else:
                eff = 0.85
                t0 = 4e-6
                bw = cls.MEMORY_WRITE_BANDWIDTH_BYTES * eff
            return x * bw / (t0 * bw + x)

        def load_to_device_per_layer(
            self, device_pool, host_indices, device_indices, layer_id, io_backend
        ) -> None:
            # update global clock
            # Merge cache indices
            # https://github.com/sgl-project/sglang/blob/v0.5.8/sgl-kernel/csrc/kvcacheio/transfer.cu#L713
            assert len(host_indices) == len(device_indices)
            num_indices = len(host_indices)

            host = np.asarray(host_indices.cpu(), dtype=np.int64)
            dev = np.asarray(device_indices.cpu(), dtype=np.int64)
            cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
            cut = np.flatnonzero(~cont) + 1
            starts = np.r_[0, cut]
            ends = np.r_[cut, num_indices]
            seg_len = (ends - starts).astype(np.float64)

            if cls.KV_CACHE_BYTES_PER_LAYER is None:
                cls.KV_CACHE_BYTES_PER_LAYER = (
                    ConfigManager.get_kv_cache_bytes_per_layer()
                )

            size_bytes_arr = seg_len * float(cls.KV_CACHE_BYTES_PER_LAYER)
            bandwidth_arr = est_bandwidth_batch(size_bytes_arr, cat="H2D")
            total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
            # total_time_cost += 3.3e-6 * len(size_bytes_arr)  # CPU Overhead
            StateManager.inc_hicache_l2_load_dur(total_time_cost)

        def backup_from_device_all_layer(
            self, device_pool, host_indices, device_indices, io_backend
        ) -> None:
            """
            Backup KV data from the device memory pool to the host memory pool for all layers.
            """
            # update global clock
            num_indices = len(host_indices)

            host = np.asarray(host_indices.cpu(), dtype=np.int64)
            dev = np.asarray(device_indices.cpu(), dtype=np.int64)
            cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
            cut = np.flatnonzero(~cont) + 1
            starts = np.r_[0, cut]
            ends = np.r_[cut, num_indices]
            seg_len = (ends - starts).astype(np.float64)

            if cls.KV_CACHE_BYTES is None:
                cls.KV_CACHE_BYTES = ConfigManager.get_kv_cache_bytes()

            size_bytes_arr = seg_len * float(cls.KV_CACHE_BYTES)
            bandwidth_arr = est_bandwidth_batch(size_bytes_arr, cat="D2H")
            total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
            # total_time_cost += 3.3e-6 * len(size_bytes_arr)  # CPU Overhead

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

        target.load_to_device_per_layer = load_to_device_per_layer
        target.backup_from_device_all_layer = backup_from_device_all_layer
        target.get_data_page = get_data_page
        target.set_from_flat_data_page = set_from_flat_data_page


class C_HostKVCacheHook(BaseHook):
    HOOK_CLASS_NAME = "HostKVCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool_host"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def wrapped_init(self, *args, **kwargs):
            # Disable pip memory, which might fail on CPU platforms.
            if "pin_memory" in kwargs:
                kwargs["pin_memory"] = False
            elif len(args) > 5:
                args = list(args)
                args[5] = False
            else:
                logger.warning(
                    "Failed to disable pip memory while initializing the hoot memory pool."
                )
            return original_init(self, *args, **kwargs)

        target.__init__ = wrapped_init


class C_DeepSeekV4SingleKVPoolHook(BaseHook):
    HOOK_CLASS_NAME = "DeepSeekV4SingleKVPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.deepseek_v4_memory_pool"

    @classmethod
    def hook(cls, target):
        def ceil_div(x: int, y: int) -> int:
            return (x + y - 1) // y

        def override_create_buffer(self, *, num_pages: int):
            bytes_per_token = self.get_bytes_per_token()
            self.kv_cache_total_dim = bytes_per_token
            bytes_per_page_non_padded = self.page_size * bytes_per_token
            self.bytes_per_page_padded = ceil_div(bytes_per_page_non_padded, 576) * 576

            assert self.store_dtype == torch.uint8

            return torch.zeros(
                num_pages,
                self.bytes_per_page_padded,
                dtype=self.store_dtype,
                device=self.device,
            )

        target.create_buffer = override_create_buffer


class C_DeepSeekV4PagedHostPoolHook(BaseHook):
    HOOK_CLASS_NAME = "DeepSeekV4PagedHostPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool_host"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def wrapped_init(self, *args, **kwargs):
            # Disable pin memory, which might fail on CPU platforms.
            if "pin_memory" in kwargs:
                kwargs["pin_memory"] = False
            elif len(args) > 6:
                args = list(args)
                args[6] = False
            elif "pin_memory" not in kwargs:
                kwargs["pin_memory"] = False
            else:
                logger.warning(
                    "Failed to disable pin memory while initializing the DeepSeekV4PagedHostPool."
                )
            return original_init(self, *args, **kwargs)

        def _segment_lens(host_indices, device_indices):
            num_indices = len(host_indices)
            host = np.asarray(host_indices.cpu(), dtype=np.int64)
            dev = np.asarray(device_indices.cpu(), dtype=np.int64)
            cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
            cut = np.flatnonzero(~cont) + 1
            starts = np.r_[0, cut]
            ends = np.r_[cut, num_indices]
            return (ends - starts).astype(np.float64)

        def backup_from_device_all_layer(
            self, device_pool, host_indices, device_indices, io_backend
        ):
            # D2H: write_through has been measured to NOT cap at chunked_prefill_size
            # — backup_n equals full prefill length for all pools (including swa).
            # The simulator just trusts the indices sglang hands us.
            if host_indices is None or device_indices is None:
                return
            self._check_io_backend(io_backend)
            host_indices = self._to_page_indices(host_indices)
            device_indices = self._to_page_indices(device_indices)

            seg_len = _segment_lens(host_indices, device_indices)
            size_bytes_arr = seg_len * self.get_size_per_token()
            bandwidth_arr = _hicache_transfer_time(size_bytes_arr, cat="D2H")
            total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
            StateManager.inc_hicache_l2_backup_dur(total_time_cost)

        def load_to_device_per_layer(
            self, device_pool, host_indices, device_indices, layer_id, io_backend
        ) -> None:
            assert len(host_indices) == len(device_indices)
            host_indices = self._to_page_indices(host_indices)
            device_indices = self._to_page_indices(device_indices)

            seg_len = _segment_lens(host_indices, device_indices)
            size_bytes_arr = seg_len * self.get_size_per_token()
            bandwidth_arr = _hicache_transfer_time(size_bytes_arr, cat="H2D")
            total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
            StateManager.inc_hicache_l2_load_dur(total_time_cost)

        def get_size_per_token(self):
            # Per-layer bytes/token × tokens-per-page (seg_len is in pages after
            # _to_page_indices). Calibrated from hicache_h2d_profile.py.
            try:
                bytes_per_token_per_layer = _BYTES_PER_TOKEN_PER_LAYER_PAGED[self.pool_name]
            except KeyError:
                raise ValueError(
                    f"[DeepSeekV4PagedHostPool] unsupported pool name: {self.pool_name}"
                )
            return bytes_per_token_per_layer * self.slot_page_size

        target.__init__ = wrapped_init
        target.backup_from_device_all_layer = backup_from_device_all_layer
        target.load_to_device_per_layer = load_to_device_per_layer
        target.get_size_per_token = get_size_per_token


class C_DeepSeekV4StateHostPoolHook(BaseHook):
    HOOK_CLASS_NAME = "DeepSeekV4StateHostPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool_host"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def wrapped_init(self, *args, **kwargs):
            # Disable pin memory, which might fail on CPU platforms.
            if "pin_memory" in kwargs:
                kwargs["pin_memory"] = False
            elif len(args) > 5:
                args = list(args)
                args[5] = False
            elif "pin_memory" not in kwargs:
                kwargs["pin_memory"] = False
            else:
                logger.warning(
                    "Failed to disable pin memory while initializing the DeepSeekV4StateHostPool."
                )
            return original_init(self, *args, **kwargs)

        def _segment_lens(host_indices, device_indices):
            num_indices = len(host_indices)
            host = np.asarray(host_indices.cpu(), dtype=np.int64)
            dev = np.asarray(device_indices.cpu(), dtype=np.int64)
            cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
            cut = np.flatnonzero(~cont) + 1
            starts = np.r_[0, cut]
            ends = np.r_[cut, num_indices]
            return (ends - starts).astype(np.float64)

        def backup_from_device_all_layer(
            self, device_pool, host_indices, device_indices, io_backend
        ):
            # D2H for state pools also does NOT cap — measurement confirms
            # state backup_n equals full N even when N > chunked_prefill_size.
            if host_indices is None or device_indices is None:
                return
            self._check_io_backend(io_backend)
            host_indices = self._to_page_indices(host_indices)
            device_indices = self._to_page_indices(device_indices)

            seg_len = _segment_lens(host_indices, device_indices)
            size_bytes_arr = seg_len * self.get_size_per_token()
            bandwidth_arr = _hicache_transfer_time(size_bytes_arr, cat="D2H")
            total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
            StateManager.inc_hicache_l2_backup_dur(total_time_cost)

        def load_to_device_per_layer(
            self, device_pool, host_indices, device_indices, layer_id, io_backend
        ) -> None:
            # H2D for state pools DOES cap: real sglang code passes only
            # ((N-1)//page_size*page_size) mod chunked_prefill_size indices.
            # We trust the indices and just compute bytes.
            assert len(host_indices) == len(device_indices)
            host_indices = self._to_page_indices(host_indices)
            device_indices = self._to_page_indices(device_indices)

            seg_len = _segment_lens(host_indices, device_indices)
            size_bytes_arr = seg_len * self.get_size_per_token()
            bandwidth_arr = _hicache_transfer_time(size_bytes_arr, cat="H2D")
            total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
            StateManager.inc_hicache_l2_load_dur(total_time_cost)

        def get_size_per_token(self):
            # Per-layer bytes/token × tokens-per-page (seg_len is in SWA pages).
            try:
                bytes_per_token_per_layer = _BYTES_PER_TOKEN_PER_LAYER_STATE[self.pool_name]
            except KeyError:
                raise ValueError(
                    f"[DeepSeekV4StateHostPool] unsupported pool name: {self.pool_name}"
                )
            return bytes_per_token_per_layer * self.swa_page_size

        target.__init__ = wrapped_init
        target.backup_from_device_all_layer = backup_from_device_all_layer
        target.load_to_device_per_layer = load_to_device_per_layer
        target.get_size_per_token = get_size_per_token