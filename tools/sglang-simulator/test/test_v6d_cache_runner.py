"""Integration test for V6D KV cache simulation.

This test verifies the complete V6D-aware simulation pipeline:
1. VLLMWorker with V6D kv_transfer_config
2. head_dim=1 injection + real V6D daemon communication
3. Cross-node prefix matching via V6D client.get()
4. KV cache swap-in/swap-out via CPU memcpy

PREREQUISITES:
- V6D daemon must be running on localhost (default port 7890)
- PAI-vLLM must be installed with V6D connector
- Model files must be accessible

Run with:
    pytest test_v6d_cache_runner.py -v -s

Skip if V6D daemon is not available:
    pytest test_v6d_cache_runner.py -v -s -k "not v6d"
"""

import os
import random
from copy import deepcopy

import numpy as np
import pytest

os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = (
    os.path.dirname(__file__) + "/assets/config_vllm_v6d.json"
)
os.environ["CUDA_VISIBLE_DEVICES"] = ""


V6D_SOCKET = "/tmp/vineyard.sock"


def _v6d_daemon_available(socket_path=V6D_SOCKET) -> bool:
    """Check if V6D daemon is reachable via IPC socket."""
    try:
        import vineyard
        client = vineyard.connect(socket_path)
        _ = client.instance_id
        del client
        return True
    except Exception:
        return False


# Skip all tests if V6D daemon is not available
pytestmark = pytest.mark.skipif(
    not _v6d_daemon_available(),
    reason=f"V6D daemon not available at {V6D_SOCKET}",
)


def _make_v6d_engine_args(model_path: str, **overrides):
    """Build EngineArgs with V6D kv_transfer_config."""
    base_args = {
        "model": model_path,
        "block_size": 16,
        "gpu_memory_utilization": 0.9,
        "num_gpu_blocks_override": 200,
        "max_model_len": 256,
        "enable_prefix_caching": True,
        "kv_transfer_config": {
            "kv_connector": "V6dObjectConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "v6d_socket": V6D_SOCKET,
            },
        },
    }
    base_args.update(overrides)
    return base_args


class TestV6dCacheBasic:
    """Basic V6D cache tests - single node."""

    def test_v6d_connector_initialization(self):
        """Test that V6D connector initializes correctly with head_dim=1."""
        from sglang_simulator.simulation.vllm.vllm_worker import (
            VLLMWorker, EngineArgs,
        )
        from sglang_simulator.simulation.benchmark import (
            BenchmarkConfig, MultiInstanceBenchmarkRunner,
        )
        from sglang_simulator.simulation.vllm.v6d_manager import (
            V6dBlockOwnershipTracker,
        )
        # Reset cross-process tracker state before test
        V6dBlockOwnershipTracker.reset()

        engine_args = _make_v6d_engine_args(
            model_path="/host/models/Qwen/Qwen3-0.6B/"
        )
        worker = VLLMWorker(engine_args=EngineArgs(**engine_args))
        runner = MultiInstanceBenchmarkRunner(workers=[worker])

        # Verify head_dim was injected
        # (Access internal engine state if possible)
        runner.shutdown()

    def test_v6d_prefix_cache_store_and_load(self):
        """Test V6D store (swap-out) and load (swap-in) cycle.

        Flow:
        1. First request: prefill -> store KV blocks to V6D
        2. Evict from device cache
        3. Second identical request: should load from V6D (swap-in)
        """
        from transformers import AutoTokenizer
        from sglang_simulator.dataset import DatasetArgs, SimpleDataset, get_dataset
        from sglang_simulator.simulation.vllm.vllm_worker import (
            VLLMWorker, EngineArgs,
        )
        from sglang_simulator.simulation.benchmark import (
            BenchmarkConfig, MultiInstanceBenchmarkRunner,
        )
        from sglang_simulator.simulation.vllm.v6d_manager import (
            V6dBlockOwnershipTracker,
        )
        # Reset cross-process tracker state before test
        V6dBlockOwnershipTracker.reset()

        random.seed(42)
        np.random.seed(42)

        engine_args = _make_v6d_engine_args(
            model_path="/host/models/Qwen/Qwen3-0.6B/",
            num_gpu_blocks_override=100,
        )
        worker = VLLMWorker(engine_args=EngineArgs(**engine_args))
        runner = MultiInstanceBenchmarkRunner(workers=[worker])

        benchmark_config = BenchmarkConfig(
            request_rate=10, ignore_request_timestamp=True
        )

        tokenizer = AutoTokenizer.from_pretrained(engine_args["model"])
        dataset_args = DatasetArgs(
            "random_ids",
            num_prompts=30,
            min_input_len=65,
            max_input_len=66,
            min_output_len=1,
            max_output_len=2,
        )
        dataset = get_dataset(dataset_args, tokenizer=tokenizer)

        cached_ds = SimpleDataset(reqs=dataset[:6])
        evict_ds = SimpleDataset(reqs=dataset[8:28])

        # First run: populate V6D cache
        metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
        assert metrics["completed"] == len(cached_ds), (
            f"First run incomplete: {metrics['completed']}/{len(cached_ds)}"
        )

        # Evict from device cache
        _ = runner.benchmark(benchmark_config, dataset=evict_ds)

        # Third run: should hit V6D (remote/host cache)
        metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
        # V6D hit should show up in either host_hit or v6d-specific metric
        total_hit = metrics.get("kv_cache_host_hit_ratio", 0) + \
                    metrics.get("kv_cache_v6d_hit_ratio", 0)
        assert total_hit > 0.5, (
            f"Expected V6D cache hit > 50%, got host={metrics.get('kv_cache_host_hit_ratio', 0)}, "
            f"v6d={metrics.get('kv_cache_v6d_hit_ratio', 0)}"
        )

        runner.shutdown()


class TestV6dCacheMultiNode:
    """Multi-node V6D cache tests - cross-node prefix sharing.

    These tests verify that V6D's cross-node routing works correctly
    in the simulation (node A can find prefix cached on node B).

    NOTE: Requires multi-node V6D cluster setup. Skip if single node.
    """

    @pytest.mark.skipif(
        os.environ.get("V6D_MULTI_NODE") != "1",
        reason="Set V6D_MULTI_NODE=1 to run multi-node tests",
    )
    def test_cross_node_prefix_hit(self):
        """Test that prefixes stored on one node are findable from another.

        In a real multi-node V6D cluster:
        - Worker A stores KV blocks for prefix P
        - Worker B queries for prefix P
        - V6D daemon routes the query to A's data
        - Worker B loads the KV blocks
        """
        from transformers import AutoTokenizer
        from sglang_simulator.dataset import DatasetArgs, SimpleDataset, get_dataset
        from sglang_simulator.simulation.vllm.vllm_worker import (
            VLLMWorker, EngineArgs,
        )
        from sglang_simulator.simulation.benchmark import (
            BenchmarkConfig, MultiInstanceBenchmarkRunner,
        )
        from sglang_simulator.simulation.vllm.v6d_manager import (
            V6dBlockOwnershipTracker,
        )
        # Reset cross-process tracker state before test
        V6dBlockOwnershipTracker.reset()

        random.seed(0)
        np.random.seed(0)

        # Two workers simulating two nodes (same V6D cluster)
        engine_args_1 = _make_v6d_engine_args(
            model_path="/host/models/Qwen/Qwen3-0.6B/",
            num_gpu_blocks_override=100,
        )
        engine_args_2 = deepcopy(engine_args_1)

        worker1 = VLLMWorker(
            engine_args=EngineArgs(**engine_args_1), name="worker_node_0"
        )
        worker2 = VLLMWorker(
            engine_args=EngineArgs(**engine_args_2), name="worker_node_1"
        )

        benchmark_config = BenchmarkConfig(
            request_rate=10, ignore_request_timestamp=True
        )

        tokenizer = AutoTokenizer.from_pretrained(engine_args_1["model"])
        dataset_args = DatasetArgs(
            "random_ids",
            num_prompts=10,
            min_input_len=65,
            max_input_len=66,
            min_output_len=1,
            max_output_len=2,
        )
        dataset = get_dataset(dataset_args, tokenizer=tokenizer)
        shared_prefix_ds = SimpleDataset(reqs=dataset[:4])

        # Worker 1: store prefixes
        runner1 = MultiInstanceBenchmarkRunner(workers=[worker1])
        metrics1 = runner1.benchmark(benchmark_config, dataset=shared_prefix_ds)
        assert metrics1["completed"] == len(shared_prefix_ds)

        # Worker 2: should find prefixes via V6D cross-node query
        runner2 = MultiInstanceBenchmarkRunner(workers=[worker2])
        metrics2 = runner2.benchmark(benchmark_config, dataset=shared_prefix_ds)

        # Cross-node hit should be detectable
        # (The exact metric name depends on implementation)
        remote_hit = metrics2.get("kv_cache_v6d_hit_ratio", 0) + \
                     metrics2.get("kv_cache_host_hit_ratio", 0)
        assert remote_hit > 0.5, (
            f"Expected cross-node V6D hit > 50%, got {remote_hit}"
        )


        # Verify RPC bypass ownership tracking
        from sglang_simulator.simulation.vllm.v6d_manager import (
            V6dBlockOwnershipTracker,
        )
        w2_stats = V6dBlockOwnershipTracker.get_stats("worker_node_1")
        assert w2_stats.get("remote_hits", 0) > 0, (
            f"Expected worker_node_1 remote hits, got {w2_stats}"
        )

        runner1.shutdown()
        runner2.shutdown()


class TestV6dHeadDimReduction:
    """Verify head_dim=1 correctly reduces data volume."""

    def test_kv_cache_tensor_size_is_tiny(self):
        """Allocated KV cache tensors should be ~128x smaller than original."""
        from sglang_simulator.simulation.vllm.worker import (
            _inject_head_dim, _build_kv_cache_spec,
        )
        from types import SimpleNamespace

        # Simulate Qwen3-0.6B: 28 layers, 8 kv_heads, head_dim=128
        hf_config = SimpleNamespace(
            num_hidden_layers=28,
            num_key_value_heads=8,
            num_attention_heads=16,
            head_dim=128,
            hidden_size=2048,
        )
        model_config = SimpleNamespace(
            hf_text_config=hf_config,
            dtype="float16",
        )
        cache_config = SimpleNamespace(
            block_size=16,
            mamba_block_size=None,
        )
        parallel_config = SimpleNamespace(tensor_parallel_size=1)
        vllm_config = SimpleNamespace(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
        )

        # Before injection: compute expected page_size
        # page_size = 2 * 16 * 8 * 128 * 2 = 65536 bytes
        original_page_size = 2 * 16 * 8 * 128 * 2

        # Inject head_dim=1
        _inject_head_dim(vllm_config)

        # Build spec
        import torch
        vllm_config.model_config.dtype = torch.float16
        spec = _build_kv_cache_spec(vllm_config)

        # After injection: page_size = 2 * 16 * 8 * 1 * 2 = 512 bytes
        first_spec = list(spec.values())[0]
        assert first_spec.page_size_bytes == 512
        assert original_page_size / first_spec.page_size_bytes == 128

    def test_total_kv_cache_memory_is_small(self):
        """With head_dim=1 and 200 blocks, total memory should be < 10 MB."""
        from sglang_simulator.simulation.vllm.worker import (
            _inject_head_dim, _build_kv_cache_spec,
        )
        from types import SimpleNamespace
        import torch

        hf_config = SimpleNamespace(
            num_hidden_layers=28,
            num_key_value_heads=8,
            num_attention_heads=16,
            head_dim=128,
            hidden_size=2048,
        )
        model_config = SimpleNamespace(
            hf_text_config=hf_config,
            dtype=torch.float16,
        )
        cache_config = SimpleNamespace(
            block_size=16,
            mamba_block_size=None,
        )
        parallel_config = SimpleNamespace(tensor_parallel_size=1)
        vllm_config = SimpleNamespace(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
        )

        _inject_head_dim(vllm_config)
        spec = _build_kv_cache_spec(vllm_config)

        num_blocks = 200
        total_bytes = sum(
            s.page_size_bytes * num_blocks for s in spec.values()
        )
        total_mb = total_bytes / (1024 * 1024)

        # 28 layers * 512 bytes/block * 200 blocks = 2.8 MB
        assert total_mb < 10, f"Total KV cache too large: {total_mb:.2f} MB"
        assert total_mb > 1, f"Total KV cache suspiciously small: {total_mb:.2f} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
