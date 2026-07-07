"""
vLLM Simulation Startup - Installs all hooks needed to hijack the vLLM
inference framework for simulation purposes.

IMPORTANT: init_hook() must be called BEFORE any `import vllm.*` statement.
"""

import os

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


def init_hook():
    """Install all vLLM hooks. Must be called before importing vllm."""
    # Ensure deterministic block hashes across nodes for cross-node cache sharing.
    # vLLM uses PYTHONHASHSEED to initialize NONE_HASH; without it, each process
    # gets os.urandom(32) making block hashes non-reproducible across machines.
    import os
    if 'PYTHONHASHSEED' not in os.environ:
        os.environ['PYTHONHASHSEED'] = '42'
        import logging
        logging.getLogger('sglang_simulator').warning(
            '[init_hook] PYTHONHASHSEED not set, defaulting to 42 for cross-node hash consistency')

    _patch_triton_for_cpu()
    _patch_torch_cuda_for_cpu()

    sglang_simulator_hook.install_class_hooks(
        [
            # Platform hook — intercepts Platform class definition to inject
            # _MockCudaPlatform before any platform detection runs.
            platform.C_VLLMPlatformHook,
            # EngineArgs hook — forces parallelism to 1 for simulation
            engine_args.C_VLLMEngineArgsHook,
            # Worker hook (handles everything — no model_runner hooks needed)
            worker.C_VLLMWorkerHook,
            # Scheduler hook for time prediction
            scheduler.C_VLLMSchedulerHook,
            # KV connector factory hook (returns MockOffloadConnector)
            kv_connector.C_KVConnectorFactoryHook,
            # Native SimpleCPUOffloadWorker hook (bypasses CUDA for native offload)
            kv_offload.C_VLLMSimpleCPUOffloadWorkerHook,
            # Native OffloadingConnectorWorker hook (default native path)
            kv_offload.C_VLLMOffloadingConnectorWorkerHook,
            # V6D SwapHandler hook — replaces CUDA Stream/Event/DMA with CPU
            v6d_swap.C_V6dSwapHandlerHook,
            # V6D ObjectConnectorWorker hook — skips CUDA host registration
            v6d_worker.C_V6dObjectConnectorWorkerHook,
            # V6D ObjectBackend hook — replaces CUDA Event pool
            v6d_backend.C_V6dObjectBackendHook,
            # V6D ObjectManager hook — RPC bypass + ownership tracking
            v6d_manager.C_V6dObjectManagerHook,
        ]
    )
