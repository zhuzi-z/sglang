def test_cpu_startup_sanitizes_tpu_platform_but_keeps_connector_plugin():
    import os
    from sglang_simulator.simulation.vllm.startup import (
        _sanitize_tpu_environment_for_cpu,
    )

    keys = (
        "VLLM_TARGET_DEVICE", "VLLM_PLUGINS", "PJRT_DEVICE",
        "VLLM_PP_LAYER_PARTITION", "TPU_VISIBLE_CHIPS",
        "VLLM_NNODES", "RANK_SIZE", "SGLANG_SIMULATOR_PP_LAYER_PARTITION",
    )
    old = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.pop("SGLANG_SIMULATOR_PP_LAYER_PARTITION", None)
        os.environ["VLLM_TARGET_DEVICE"] = "tpu"
        os.environ["VLLM_PLUGINS"] = "torchtpu,torchtpu_layers,kvs_connector"
        os.environ["PJRT_DEVICE"] = "TPU"
        os.environ["VLLM_PP_LAYER_PARTITION"] = "24,24,24,20"
        os.environ["TPU_VISIBLE_CHIPS"] = "0,1,2,3"
        os.environ["VLLM_NNODES"] = "4"
        os.environ["RANK_SIZE"] = "4"
        _sanitize_tpu_environment_for_cpu()
        assert "VLLM_TARGET_DEVICE" not in os.environ
        assert os.environ["VLLM_PLUGINS"] == "kvs_connector"
        assert "PJRT_DEVICE" not in os.environ
        assert "VLLM_PP_LAYER_PARTITION" not in os.environ
        assert os.environ["SGLANG_SIMULATOR_PP_LAYER_PARTITION"] == "24,24,24,20"
        assert os.environ["TPU_VISIBLE_CHIPS"] == ""
        assert os.environ["VLLM_NNODES"] == "1"
        assert os.environ["RANK_SIZE"] == "1"
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_cpu_startup_preserves_explicit_cpu_target_in_subprocess(monkeypatch):
    import os
    import subprocess
    import sys
    from sglang_simulator.simulation.vllm.startup import (
        _sanitize_tpu_environment_for_cpu,
    )

    monkeypatch.setenv("VLLM_TARGET_DEVICE", "cpu")
    _sanitize_tpu_environment_for_cpu()
    assert os.environ["VLLM_TARGET_DEVICE"] == "cpu"
    result = subprocess.run(
        [sys.executable, "-S", "-c",
         'import os; print(os.environ.get("VLLM_TARGET_DEVICE"))'],
        check=True, capture_output=True, text=True, timeout=15,
    )
    assert result.stdout.strip() == "cpu"


def test_cpu_startup_empty_cache_noop_without_accelerator(monkeypatch):
    import torch
    from unittest.mock import Mock
    from sglang_simulator.simulation.vllm.startup import _patch_torch_cuda_for_cpu

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", torch.cuda.get_device_properties
    )
    monkeypatch.setattr(torch.cuda, "_lazy_init", torch.cuda._lazy_init)

    empty_cache = Mock(side_effect=RuntimeError(
        "Cannot access accelerator device when none is available"
    ))
    empty_host_cache = Mock(side_effect=RuntimeError(
        "Cannot access accelerator device when none is available"
    ))
    monkeypatch.setattr(torch.accelerator, "empty_cache", empty_cache)
    # Torch 2.11 lacks this API; inject the target Torch 2.13 API without skipping.
    monkeypatch.setattr(
        torch.accelerator, "empty_host_cache", empty_host_cache, raising=False
    )
    _patch_torch_cuda_for_cpu()
    assert torch.accelerator.empty_cache() is None
    assert torch.accelerator.empty_host_cache() is None
    empty_cache.assert_not_called()
    empty_host_cache.assert_not_called()


def test_cpu_startup_preserves_empty_cache_with_available_accelerator(monkeypatch):
    import torch
    from unittest.mock import Mock
    from sglang_simulator.simulation.vllm.startup import _patch_torch_cuda_for_cpu

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", torch.cuda.get_device_properties
    )
    monkeypatch.setattr(torch.cuda, "_lazy_init", torch.cuda._lazy_init)
    empty_cache = Mock(name="accelerator_empty_cache")
    empty_host_cache = Mock(name="accelerator_empty_host_cache")
    monkeypatch.setattr(torch.accelerator, "empty_cache", empty_cache)
    # Torch 2.11 lacks this API; inject the target Torch 2.13 API without skipping.
    monkeypatch.setattr(
        torch.accelerator, "empty_host_cache", empty_host_cache, raising=False
    )
    _patch_torch_cuda_for_cpu()
    assert torch.accelerator.empty_cache is empty_cache
    assert torch.accelerator.empty_host_cache is empty_host_cache
    torch.accelerator.empty_cache()
    torch.accelerator.empty_host_cache()
    empty_cache.assert_called_once_with()
    empty_host_cache.assert_called_once_with()
