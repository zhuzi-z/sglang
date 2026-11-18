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


def test_inprocess_multi_connector_metrics_match_serialized_stats():
    from dataclasses import dataclass
    from sglang_simulator.simulation.vllm.engine_core_pipeline import (
        C_VLLMMultiConnectorPromMetricsHook,
    )

    @dataclass
    class Stats:
        data: dict

    class Metrics:
        def observe(self, stats, engine_idx=0):
            return {key: value["data"] for key, value in stats.items()}, engine_idx

    C_VLLMMultiConnectorPromMetricsHook.hook(Metrics)
    before = Metrics.observe
    C_VLLMMultiConnectorPromMetricsHook.hook(Metrics)
    assert Metrics.observe is before
    nested = Stats({"loaded": [1, 2]})
    serialized = {"data": {"stored": [3]}}
    result, index = Metrics().observe({"v6d": nested, "other": serialized}, 2)
    assert result == {"v6d": {"loaded": [1, 2]}, "other": {"stored": [3]}}
    assert index == 2
    assert nested.data == {"loaded": [1, 2]}
    assert serialized == {"data": {"stored": [3]}}
    assert Metrics().observe({}) == ({}, 0)
