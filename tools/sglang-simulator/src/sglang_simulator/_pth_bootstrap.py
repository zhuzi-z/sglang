
import os
from sglang_simulator.hook import install_class_hooks, install_module_hooks

if os.environ.get("SIM_COLLECTOR_ENABLE", "").lower() in ("1", "true", "yes", "on"):
    from sglang_simulator.collector.vllm_hook.worker_hook import (
        C_VLLMEngineArgsHook,
        C_WorkerHook,
        C_SchedulerHook,
        C_EngineCoreHook,
    )

    from sglang_simulator.collector.dashserving import (
        M_DashservingEntrypointHook,
    )

    install_module_hooks(
        [
            M_DashservingEntrypointHook,
        ]
    )

    install_class_hooks(
        [
            C_VLLMEngineArgsHook,
            C_WorkerHook,
            C_SchedulerHook,
            C_EngineCoreHook,
        ]
    )

if os.environ.get("SIM_DEBUG_ENABLE", "").lower() in ("1", "true", "yes", "on"):
    # Standalone observability hooks for real deployments. Unlike the
    # collector these are read-only (logs + GET /server_info on the dashserving
    # control port), so they are safe to enable in a real serving process.
    from sglang_simulator.debug import (
        C_HitRateHook,
        C_TieredVineyardPeerHook,
        C_HitRateStatsHook,
        C_DashservingControlHandlerHook,
    )

    install_class_hooks(
        [
            C_HitRateHook,
            C_TieredVineyardPeerHook,
            C_HitRateStatsHook,
            C_DashservingControlHandlerHook,
        ]
    )