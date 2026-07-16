"""
vLLM Simulation Startup - Installs all hooks needed to hijack the vLLM
inference framework for simulation purposes.

IMPORTANT: init_hook() must be called BEFORE any `import vllm.*` statement.
"""

import os


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_enabled(*names: str) -> bool:
    return any(os.environ.get(name, "").strip().lower() in _TRUE_VALUES for name in names)

# Native V6D control-plane mode keeps the real vLLM connector stack
# (HybridConnector -> V6dObjectKVTBackend -> V6dObjectBackend/PBackend ->
# V6dObjectConnectorScheduler/V6dObjectManager) and installs only runtime
# hijack hooks for CPU-only execution.  The default CPU simulation mode still
# uses MockHybridConnector + V6DCacheStorage(etcd) for scheduling parity.
#
# This file must not patch DashServing/vLLM source files on disk.  All behavior
# changes are installed through class hooks, sitecustomize, and monkey patches
# gated by explicit environment variables.


def _patch_triton_for_cpu():
    """Ensure triton.language.extra.cuda exists for CPU-only environments.

    Some vLLM versions use a TritonLanguagePlaceholder when no GPU driver is
    detected. That placeholder lacks `extra.cuda`, causing AttributeError in
    modules like vllm.lora.ops.triton_ops.utils. Setting CUDA_VISIBLE_DEVICES
    to empty string triggers vLLM's 'distributed environment' heuristic which
    keeps HAS_TRITON=True and uses the real triton module.

    As a safety net, we also pre-import triton.language.extra.cuda so it's
    resolved in sys.modules regardless of placeholder usage.
    """
    # Trick vLLM's triton driver check into lenient mode (distributed env)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    # Safety net: ensure triton.language.extra.cuda is importable
    try:
        import triton.language.extra.cuda  # noqa: F401
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass


def _patch_torch_cuda_for_cpu():
    """Patch torch.cuda functions for CPU-only environments.

    Some modified vLLM versions call torch.cuda.get_device_properties() at
    module level (e.g. in expert_parallel.py). This patches those calls to
    return fake device properties so the imports succeed without a GPU.
    """
    import torch

    if torch.cuda.is_available():
        return  # Real GPU available, no patching needed

    class _FakeDeviceProperties:
        """Fake CUDA device properties for simulation."""
        major = 9
        minor = 0
        name = "NVIDIA Simulated GPU"
        total_memory = 80 * (1 << 30)  # 80 GiB
        multi_processor_count = 132

    _original_get_device_properties = torch.cuda.get_device_properties

    def _mock_get_device_properties(device=None):
        try:
            return _original_get_device_properties(device)
        except (RuntimeError, AssertionError):
            return _FakeDeviceProperties()

    torch.cuda.get_device_properties = _mock_get_device_properties

    # Also patch _lazy_init to be a no-op if CUDA is unavailable
    _original_lazy_init = torch.cuda._lazy_init

    def _mock_lazy_init():
        try:
            _original_lazy_init()
        except RuntimeError:
            pass

    torch.cuda._lazy_init = _mock_lazy_init


def init_hook(force: bool = False):
    """Install all vLLM hooks. Must be called before importing vllm.

    The startup entry lives in the project working directory as sitecustomize.py.
    Python imports it only when that directory is added to PYTHONPATH, and hooks
    are installed only when explicitly enabled by environment variable. This
    avoids installing global .pth files or affecting normal vLLM users sharing
    the same Python environment.
    """
    if not force and not _env_enabled(
        "SGLANG_SIMULATOR_ENABLE_HOOK",
        "SGLANG_SIMULATOR_ENABLE_VLLM_HOOK",
    ):
        return False

    import logging

    import sglang_simulator.hook as sglang_simulator_hook
    from sglang_simulator.simulation.vllm import (
        engine_args,
        kv_connector,
        kv_offload,
        platform,
        scheduler,
        v6d_backend,
        v6d_manager,
        v6d_swap,
        v6d_worker,
        worker,
    )

    # Keep v6d_manager imported: in native V6D mode its hooks tag managers with
    # the active worker id and publish/lookup cross-node ownership; in default
    # MockHybridConnector mode the env helpers remain useful for worker identity.
    _ = v6d_manager

    # Ensure deterministic block hashes across nodes for cross-node cache sharing.
    # vLLM uses PYTHONHASHSEED to initialize NONE_HASH; without it, each process
    # gets os.urandom(32) making block hashes non-reproducible across machines.
    if 'PYTHONHASHSEED' not in os.environ:
        os.environ['PYTHONHASHSEED'] = '42'
        logging.getLogger('sglang_simulator').warning(
            '[init_hook] PYTHONHASHSEED not set, defaulting to 42 for cross-node hash consistency')

    _patch_triton_for_cpu()
    _patch_torch_cuda_for_cpu()

    native_v6d_control_plane = _env_enabled(
        "SGLANG_SIMULATOR_NATIVE_V6D_CONTROL_PLANE"
    )

    hooks = [
        # Platform hook — intercepts Platform class definition to inject
        # _MockCudaPlatform before any platform detection runs.
        platform.C_VLLMPlatformHook,
        # EngineArgs hook — forces parallelism to 1 for simulation
        engine_args.C_VLLMEngineArgsHook,
        # Worker hook (handles everything — no model_runner hooks needed)
        worker.C_VLLMWorkerHook,
        # Scheduler hook for time prediction
        scheduler.C_VLLMSchedulerHook,
        # Native SimpleCPUOffloadWorker hook (bypasses CUDA for native offload)
        kv_offload.C_VLLMSimpleCPUOffloadWorkerHook,
        # Native OffloadingConnectorWorker hook (default native path)
        kv_offload.C_VLLMOffloadingConnectorWorkerHook,
    ]

    if native_v6d_control_plane:
        logging.getLogger('sglang_simulator').info(
            '[init_hook] Native V6D control-plane mode enabled: '
            'keeping real KVConnectorFactory and installing V6D CUDA-bypass hooks')
        hooks.extend([
            v6d_swap.C_V6dSwapHandlerHook,
            v6d_worker.C_V6dObjectConnectorWorkerHook,
            v6d_backend.C_V6dObjectBackendHook,
            v6d_backend.C_KVTPBackendHook,
            v6d_manager.C_V6dObjectConnectorSchedulerHook,
            v6d_manager.C_V6dObjectManagerHook,
        ])
    else:
        # Default CPU simulation path: replace HybridConnector with
        # MockHybridConnector + V6DCacheStorage(etcd) for scheduling parity.
        hooks.append(kv_connector.C_KVConnectorFactoryHook)

    sglang_simulator_hook.install_class_hooks(hooks)
    return True
