"""Loaded by sglang_simulator.pth at every Python interpreter startup.

Installs the simulation hooks only when SGLANG_SIMULATOR_ENABLE is truthy
(1/true/yes/on); otherwise this is a no-op.
"""
import json
import os
import sys

if os.environ.get("SGLANG_SIMULATOR_ENABLE", "").lower() in ("1", "true", "yes", "on"):
    print("[HISIM_BOOTSTRAP] " + json.dumps({
        "pid": os.getpid(), "ppid": os.getppid(), "stage": "pth_entry",
        "executable": sys.executable, "bootstrap_origin": __file__,
        "isolated": sys.flags.isolated, "no_site": sys.flags.no_site,
        "ignore_environment": sys.flags.ignore_environment,
        "vllm_present": "vllm" in sys.modules,
    }), file=sys.stderr, flush=True)
    from sglang_simulator.simulation.vllm.startup import init_hook as _vllm_init_hook
    from sglang_simulator.simulation.sglang.startup import init_hook as _sglang_init_hook

    from sglang_simulator.simulation.vllm.v6d.v6d_capacity import install_v6d_runtime_hooks

    _vllm_init_hook()
    _sglang_init_hook()
    install_v6d_runtime_hooks()
