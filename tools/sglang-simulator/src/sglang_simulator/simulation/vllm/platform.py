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

            # ---- Attributes accessed directly (not via method call) ----
            empty_cache = None  # checked as `if empty_cache is not None`

            def is_cuda(self) -> bool:
                # Return False to skip module-level CUDA ops checks
                # (cutlass_scaled_mm_supports_fp8, etc.) that need libcuda.so.
                return False

            def is_cuda_alike(self) -> bool:
                return False

            @property
            def supported_dtypes(self):
                return [torch.bfloat16, torch.float16, torch.float32]

            @classmethod
            def get_device_capability(cls, device_id=0):
                from vllm.platforms.interface import DeviceCapability
                return DeviceCapability(8, 0)  # A100 equivalent

            @classmethod
            def has_device_capability(
                cls, capability: int, device_id: int = 0
            ) -> bool:
                return capability <= 80

            @classmethod
            def check_and_update_config(cls, vllm_config) -> None:
                parallel_config = vllm_config.parallel_config
                if parallel_config.worker_cls == "auto":
                    parallel_config.worker_cls = (
                        "vllm.v1.worker.gpu_worker.Worker"
                    )

            @classmethod
            def pre_register_and_update(cls, parser=None) -> None:
                """No-op: nothing to pre-register in simulation."""
                pass

            @classmethod
            def support_hybrid_kv_cache(cls) -> bool:
                return True

            @classmethod
            def manual_seed_all(cls, seed: int) -> None:
                pass

            @classmethod
            def set_device(cls, device) -> None:
                """No-op: skip torch.cuda.set_device on CPU simulation."""
                pass

            @classmethod
            def get_device_name(cls, device_id: int = 0) -> str:
                return "NVIDIA A100-SXM4-80GB"

            @classmethod
            def get_device_total_memory(cls, device_id: int = 0) -> int:
                """Return 80 GiB (A100 equivalent) for profiling calculations."""
                return 80 * (1 << 30)

            @classmethod
            def get_attn_backend_cls(
                cls, selected_backend, head_size, dtype,
                kv_cache_dtype, block_size, use_v1, use_mla=False,
            ):
                return None

            @classmethod
            def verify_quantization(cls, quant: str) -> None:
                pass

            @classmethod
            def get_current_memory_usage(cls, device=None) -> float:
                return 0.0

            def __getattr__(self, name):
                """Fallback for any unimplemented platform method.
                Returns a no-op callable to avoid AttributeError crashes."""
                def _noop(*args, **kwargs):
                    return None
                return _noop

        # Use sys.modules directly: at hook time the parent module's attribute
        # binding is not yet complete, so `import vllm.platforms` would fail.
        platforms_module = sys.modules["vllm.platforms"]
        platforms_module.current_platform = _MockCudaPlatform()
