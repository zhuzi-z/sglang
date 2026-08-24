"""V6D hook registration for native V6D control-plane mode."""

from sglang_simulator.utils import get_logger

logger = get_logger()


def register_v6d_hooks(hooks: list) -> None:
    """Append V6D native control-plane hooks to the given hook list.

    Called unconditionally from startup.init_hook(): these are class hooks
    on vLLM's v6d connector classes and only take effect when a
    HybridConnector/v6d backend is actually configured.

    Hook-module imports stay inside this function: their load order is
    orchestrated exclusively by init_hook() (after the v6d ipc env setup).
    """
    from sglang_simulator.simulation.vllm.v6d.v6d_backend import (
        C_HybridBackendHook,
        C_HybridConnectorHook,
        C_V6dObjectBackendHook,
        C_KVTPBackendHook,
    )
    from sglang_simulator.simulation.vllm.v6d.v6d_manager import (
        C_V6dObjectConnectorSchedulerHook,
        C_V6dObjectFetchHelperHook,
        C_V6dObjectManagerHook,
    )
    from sglang_simulator.simulation.vllm.v6d.v6d_swap import C_V6dSwapHandlerHook
    from sglang_simulator.simulation.vllm.v6d.v6d_worker import (
        C_V6dObjectConnectorWorkerHook,
    )

    v6d_hooks = [
        C_V6dSwapHandlerHook,
        C_HybridBackendHook,
        C_HybridConnectorHook,
        C_V6dObjectConnectorWorkerHook,
        C_V6dObjectBackendHook,
        C_KVTPBackendHook,
        C_V6dObjectConnectorSchedulerHook,
        C_V6dObjectFetchHelperHook,
        C_V6dObjectManagerHook,
    ]
    hooks.extend(v6d_hooks)
    logger.info("[init_hook] Registered %d V6D native control-plane hooks",
                len(v6d_hooks))
