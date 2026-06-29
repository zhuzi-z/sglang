"""
vLLM Simulation Startup - Installs all hooks needed to hijack the vLLM
inference framework for simulation purposes.

IMPORTANT: init_hook() must be called BEFORE any `import vllm.*` statement.
"""

import sglang_simulator.hook as sglang_simulator_hook
from sglang_simulator.simulation.vllm import (
    kv_connector,
    kv_offload,
    scheduler,
    worker,
)


def init_hook():
    """Install all vLLM hooks. Must be called before importing vllm."""
    sglang_simulator_hook.install_class_hooks(
        [
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
        ]
    )
