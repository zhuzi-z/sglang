
import os
from sglang_simulator.hook import install_class_hooks

if os.environ.get("SGLANG_SIMULATOR_COLLECTOR_ENABLE", "").lower() in ("1", "true", "yes", "on"):
    from sglang_simulator.collector.vllm_hook.worker_hook import (
        C_WorkerWrapperBaseHook,
        C_WorkerHook,
        C_SchedulerHook,
        C_EngineCoreHook,
    )

    install_class_hooks(
        [C_WorkerWrapperBaseHook, C_WorkerHook, C_SchedulerHook, C_EngineCoreHook]
    )