from sglang_simulator.hook import install_class_hooks, install_module_hooks
from sglang_simulator.collector.vllm_hook.worker_hook import (
    C_VLLMEngineArgsHook,
    C_WorkerHook,
    C_SchedulerHook,
    C_EngineCoreHook,
    C_ModelRunnerBatchHook,
    M_DPBatchCoordinationHook,
)

install_module_hooks([M_DPBatchCoordinationHook])
install_class_hooks(
    [
        C_VLLMEngineArgsHook,
        C_WorkerHook,
        C_SchedulerHook,
        C_EngineCoreHook,
        C_ModelRunnerBatchHook,
    ]
)

# -*- coding: utf-8 -*-
import sys
from vllm.entrypoints.cli.main import main
if __name__ == "__main__":
    if sys.argv[0].endswith("-script.pyw"):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith(".exe"):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())