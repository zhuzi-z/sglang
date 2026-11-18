"""Regression for transfer modules imported before the simulation bootstrap."""

from types import ModuleType
import sys
from unittest.mock import Mock

from sglang_simulator.simulation.vllm.v6d import v6d_capacity
from sglang_simulator.simulation.vllm.v6d.tair_kvcm import (
    C_TairKVCMTransferRuntimeHook,
)


def test_bootstrap_patches_preimported_transfer_runtime(monkeypatch):
    module = ModuleType(C_TairKVCMTransferRuntimeHook.HOOK_MODULE_NAME)

    class Runtime:
        def _create_transfer_client(self):
            raise AssertionError("Real payload client must be replaced in simulation")

    module.TairKVCMTransferRuntime = Runtime
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(v6d_capacity, "_RUNTIME_HOOKS_INSTALLED", False)
    register = Mock()
    monkeypatch.setattr(v6d_capacity, "install_class_hooks", register)

    v6d_capacity.install_v6d_runtime_hooks()
    client = Runtime()._create_transfer_client()
    client.close()
    assert Runtime._sim_tair_kvcm_transfer_hook
    first = Runtime._create_transfer_client

    v6d_capacity.install_v6d_runtime_hooks()
    C_TairKVCMTransferRuntimeHook.hook(Runtime)
    assert Runtime._create_transfer_client is first
    register.assert_called_once()


def test_bootstrap_keeps_future_import_hook(monkeypatch):
    monkeypatch.delitem(
        sys.modules, C_TairKVCMTransferRuntimeHook.HOOK_MODULE_NAME, raising=False
    )
    monkeypatch.setattr(v6d_capacity, "_RUNTIME_HOOKS_INSTALLED", False)
    register = Mock()
    monkeypatch.setattr(v6d_capacity, "install_class_hooks", register)

    v6d_capacity.install_v6d_runtime_hooks()
    assert C_TairKVCMTransferRuntimeHook in register.call_args.args[0]
