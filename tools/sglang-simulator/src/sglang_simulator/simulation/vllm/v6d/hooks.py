"""V6D hook registration for native V6D control-plane mode."""

import logging

logger = logging.getLogger("sglang_simulator")


def register_v6d_hooks(hooks: list) -> None:
    """Append V6D native control-plane hooks to the given hook list.

    Called unconditionally from startup.init_hook(): these are class hooks
    on vLLM's v6d connector classes and only take effect when a
    HybridConnector/v6d backend is actually configured.
    """
    from sglang_simulator.simulation.vllm.v6d.v6d_swap import C_V6dSwapHandlerHook
    from sglang_simulator.simulation.vllm.v6d.v6d_backend import (
        C_HybridBackendHook,
        C_HybridConnectorHook,
        C_V6dObjectBackendHook,
        C_KVTPBackendHook,
    )
    from sglang_simulator.simulation.vllm.v6d.v6d_worker import C_V6dObjectConnectorWorkerHook
    from sglang_simulator.simulation.vllm.v6d.v6d_manager import (
        C_V6dObjectConnectorSchedulerHook,
        C_V6dObjectManagerHook,
        C_HybridSchedulerHook,
    )

    hooks.extend([
        C_V6dSwapHandlerHook,
        C_HybridBackendHook,
        C_HybridConnectorHook,
        C_V6dObjectConnectorWorkerHook,
        C_V6dObjectBackendHook,
        C_KVTPBackendHook,
        C_V6dObjectConnectorSchedulerHook,
        C_V6dObjectManagerHook,
        C_HybridSchedulerHook,
    ])
    logger.info("[init_hook] Registered %d V6D native control-plane hooks", 9)
