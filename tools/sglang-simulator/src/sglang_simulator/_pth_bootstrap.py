
import os
from sglang_simulator.hook import install_class_hooks, install_module_hooks

if os.environ.get("SIM_COLLECTOR_ENABLE", "").lower() in ("1", "true", "yes", "on"):
    from sglang_simulator.collector.vllm_hook.worker_hook import (
        C_VLLMEngineArgsHook,
        C_WorkerHook,
        C_TpuWorkerHook,
        C_MultiprocExecutorProfileHook,
        C_SchedulerHook,
        # torchtpu subclass overrides Scheduler.schedule; re-apply on it.
        C_TpuSchedulerHook,
        C_EngineCoreHook,
        C_V6dWorkerHook,
        C_V6dSwapHook,
        # v6d connector ships as top-level kvs_connector/ on this image.
        C_KvsV6dWorkerHook,
        C_KvsV6dSwapHook,
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
            # torchtpu fork: same collection body, different (module, class).
            C_TpuWorkerHook,
            # Multi-node PP followers keep a bare MultiprocExecutor as their
            # backend engine; dashllm calls .start_profile() on it directly.
            C_MultiprocExecutorProfileHook,
            C_SchedulerHook,
            # torchtpu subclass overrides Scheduler.schedule; re-apply on it.
            C_TpuSchedulerHook,
            C_EngineCoreHook,
            C_V6dWorkerHook,
            C_V6dSwapHook,
            # v6d connector ships as top-level kvs_connector/ on this image.
            C_KvsV6dWorkerHook,
            C_KvsV6dSwapHook,
        ]
    )