"""
vLLM Platform Hook - Hijacks the Platform class to inject a mock CUDA platform.

On CPU-only environments (no NVIDIA driver), vLLM's platform detection returns
UnspecifiedPlatform with empty device_type, causing DeviceConfig to crash.
This hook intercepts Platform class construction and injects a lightweight mock
that reports device_type='cuda' without importing CUDA-dependent modules.

The hook fires when `Platform` is being defined in `vllm.platforms.interface`.
At that point `vllm.platforms` is already in sys.modules (parent package loads
first), so we can set `current_platform` immediately.
"""

import sys

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger()


class C_VLLMPlatformHook(BaseHook):
    """Hook Platform class to inject _MockCudaPlatform as current_platform."""

    HOOK_CLASS_NAME = "Platform"
    HOOK_MODULE_NAME = "vllm.platforms.interface"

    @classmethod
    def hook(cls, target_class):
        import torch

        iface = sys.modules["vllm.platforms.interface"]
        PlatformEnum = iface.PlatformEnum

        class _MockCudaPlatform(target_class):
            _enum = PlatformEnum.CUDA
            device_type = "cuda"
            device_name = "cuda"
            dispatch_key = "CUDA"
            device_control_env_var = "CUDA_VISIBLE_DEVICES"

            def is_cuda(self) -> bool:
                # Return False to skip module-level CUDA ops checks
                # (cutlass_scaled_mm_supports_fp8, etc.) that need libcuda.so.
                return False

            def is_cuda_alike(self) -> bool:
                return False

            @classmethod
            def get_device_capability(cls, device_id=0):
                return None

            @classmethod
            def check_and_update_config(cls, vllm_config) -> None:
                parallel_config = vllm_config.parallel_config
                if parallel_config.worker_cls == "auto":
                    parallel_config.worker_cls = (
                        "vllm.v1.worker.gpu_worker.Worker"
                    )

            @classmethod
            def support_hybrid_kv_cache(cls) -> bool:
                return True

            @classmethod
            def manual_seed_all(cls, seed: int) -> None:
                pass

            @classmethod
            def set_device(cls, device: torch.device) -> None:
                """No-op: skip torch.cuda.set_device on CPU simulation."""
                pass

            @classmethod
            def get_device_name(cls, device_id: int = 0) -> str:
                return "MockCUDA"

            @classmethod
            def get_device_total_memory(cls, device_id: int = 0) -> int:
                """Return 80 GiB (A100 equivalent) for profiling calculations."""
                return 80 * (1 << 30)

        # Use sys.modules directly: at hook time the parent module's attribute
        # binding is not yet complete, so `import vllm.platforms` would fail.
        platforms_module = sys.modules["vllm.platforms"]
        platforms_module.current_platform = _MockCudaPlatform()
