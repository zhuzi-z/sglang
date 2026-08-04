"""Loaded by sglang_simulator.pth at every Python interpreter startup.

Installs the simulation hooks only when SGLANG_SIMULATOR_ENABLE is truthy
(1/true/yes/on); otherwise this is a no-op.
"""
import os

if os.environ.get("SGLANG_SIMULATOR_ENABLE", "").lower() in ("1", "true", "yes", "on"):
    from sglang_simulator.simulation.vllm.startup import init_hook as _vllm_init_hook
    from sglang_simulator.simulation.sglang.startup import init_hook as _sglang_init_hook

    from sglang_simulator.simulation.vllm.v6d.v6d_capacity import install_v6d_runtime_hooks

    _vllm_init_hook()
    _sglang_init_hook()
    install_v6d_runtime_hooks()
