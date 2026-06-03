from sglang_simulator.hook import BaseHook


class M_SGLangKernelLoadUtilHook(BaseHook):
    HOOK_CLASS_NAME = ""
    HOOK_MODULE_NAME = "sgl_kernel.load_utils"

    @classmethod
    def hook(cls, target):
        def override_load_architecture_specific_ops(*args, **kwargs):
            """
            ImportError:
            [sgl_kernel] CRITICAL: Could not load any common_ops library!
            """
            pass

        target._load_architecture_specific_ops = override_load_architecture_specific_ops

class M_GDNTritonKernelHook(BaseHook):
    """Prevent gdn_triton.py from crashing on CPU when sgl_kernel native ops
    are not loaded (the module hook for sgl_kernel.load_utils suppresses the
    native library load, so torch.ops.sgl_kernel.* are unavailable)."""
    HOOK_CLASS_NAME = ""
    HOOK_MODULE_NAME = "sglang.srt.layers.attention.linear.kernels.gdn_triton"

    @classmethod
    def hook(cls, target):
        # Provide stub implementations so the module imports cleanly.
        # In simulation the attention backend is set to None — these stubs
        # are never called at runtime.
        def _stub(*args, **kwargs):
            raise RuntimeError("GDN kernel stubs should not be called in simulation")

        if not hasattr(target, "chunk_gated_delta_rule"):
            target.chunk_gated_delta_rule = _stub
        if not hasattr(target, "fused_sigmoid_gating_delta_rule_update"):
            target.fused_sigmoid_gating_delta_rule_update = _stub
