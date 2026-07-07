from sglang_simulator.hook import install_class_hooks
from worker_hook import C_WorkerWrapperBaseHook, C_WorkerHook

install_class_hooks([C_WorkerWrapperBaseHook, C_WorkerHook])

# -*- coding: utf-8 -*-
import sys
from vllm.entrypoints.cli.main import main
if __name__ == "__main__":
    if sys.argv[0].endswith("-script.pyw"):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith(".exe"):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())