#!/usr/bin/env python3
"""GLM-5.1 bl_trace L3 IO 端到端仿真示例脚本

使用说明：
    cd /path/to/sglang/tools/sglang-simulator/examples/glm51_bl_trace_l3
    python run_sim.py [--max-requests N] [--predictor fixed|gbr] [--trace PATH]

默认行为：
    - 使用当前目录下的 sample_trace_20.jsonl (20 条请求)
    - 使用 fixed 预测器（无需外部训练数据）
    - 输出产物到 ./output/

如需使用 GBR 预测器（更接近真实延迟）：
    python run_sim.py --predictor gbr \
        --gbr-database /nfs_3820/projects/kunlun_insight/data/serving/bailian/B300-GLM-5-NVFP4-TP8/260522.top_pods_hisim_glm-5.run_batch

如需使用完整 trace (1835 条)：
    python run_sim.py --trace /nfs_3820/users/maruiyan.mry/bl_trace/multi_node_trace_combine_glm-5/260531/multi_node_trace_combine_glm-5/hisim-num-node-1-glm-5-blksz-256-bucket-85-128-cnt-1835-time-60min.jsonl
"""

# ======================================================================
# 关键：以下代码必须在所有其他 import 之前执行
# ======================================================================
import os
import multiprocessing as mp

# 1) 禁用 CUDA — CPU-only 仿真模式
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 2) 强制 fork 模式 — 防止 sglang 内部的 mp.set_start_method("spawn")
#    导致子进程丢失 builtins.__build_class__ hook
mp.set_start_method("fork", force=True)
mp.set_start_method = lambda *a, **kw: None  # 屏蔽后续覆盖

# 3) 安全 fallback for torch.cuda.get_device_capability (CPU 模式无 GPU)
import torch
_orig_gdc = torch.cuda.get_device_capability
def _safe_get_device_capability(device=None):
    try:
        return _orig_gdc(device)
    except Exception:
        return (9, 0)  # 伪装 SM90 以通过硬件检查
torch.cuda.get_device_capability = _safe_get_device_capability

# 4) 必需的环境变量
os.environ["SGLANG_NUMA_BIND_V2"] = "0"
os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
os.environ["SGLANG_ENABLE_UNIFIED_RADIX_TREE"] = "0"  # 必须=0，HiRadixCache 才支持 L3
os.environ["HISIM_LOG_LEVEL"] = "INFO"

# ======================================================================

import argparse
import glob
import json
import shutil
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="GLM-5.1 L3 IO Simulation")
    parser.add_argument("--max-requests", type=int, default=None,
                        help="Max requests to simulate (default: all in trace)")
    parser.add_argument("--predictor", choices=["fixed", "gbr"], default="fixed",
                        help="Time predictor type (default: fixed)")
    parser.add_argument("--trace", type=str, default=None,
                        help="Path to trace JSONL (default: sample_trace_20.jsonl)")
    parser.add_argument("--gbr-database", type=str, default=None,
                        help="Root dir for GBR training data (schedule_batch.jsonl)")
    parser.add_argument("--model-path", type=str,
                        default="/nfs/models/ZhipuAI/GLM-5.1-FP8",
                        help="Model path for config parsing")
    parser.add_argument("--max-total-tokens", type=int, default=300_000,
                        help="Max total tokens in KV pool")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ./output/)")
    return parser.parse_args()


def load_trace(filepath: str, max_requests=None):
    """加载 hisim-collection 格式的 trace JSONL"""
    from sglang_simulator.dataset import SimpleDataset, GenericRequest

    dataset = SimpleDataset()
    count = 0
    min_ts = float("inf")

    with open(filepath) as f:
        for line in f:
            if max_requests and count >= max_requests:
                break
            req = json.loads(line)
            ts = req.get("timestamp") or req.get("created_time")
            min_ts = min(min_ts, ts)
            dataset.add_request(
                GenericRequest(
                    token_ids=req["input_ids"],
                    input_length=req["input_length"],
                    output_length=req["output_length"],
                    custom_params={"created_time": ts},
                )
            )
            count += 1

    # 时间归零
    for req in dataset.data:
        req.custom_params["created_time"] -= min_ts

    return dataset


def build_sim_config(args) -> dict:
    """构建 simulator 配置"""
    if args.predictor == "gbr":
        if not args.gbr_database:
            raise ValueError("--gbr-database is required when using gbr predictor")
        root = Path(args.gbr_database)
        gbr_paths = sorted(str(p) for p in root.glob("*/data/*/no_cache.schedule_batch.jsonl"))
        if not gbr_paths:
            raise RuntimeError(f"No schedule_batch.jsonl found under {root}")
        predictor_cfg = {
            "name": "gbr",
            "database_path": ",".join(gbr_paths),
        }
    else:
        predictor_cfg = {
            "name": "fixed",
            "prefill_ms_per_token": 0.05,
            "prefill_overhead_ms": 5.0,
            "decode_ms_per_request": 20.0,
            "decode_overhead_ms": 0.0,
        }

    return {
        "platform": {
            "accelerator": {"name": "b300_sxm"},
            "disk_read_bandwidth_gb": 4,
            "disk_write_bandwidth_gb": 4,
            "memory_read_bandwidth_gb": 64,
            "memory_write_bandwidth_gb": 64,
            "num_device_per_node": 8,
        },
        "predictor": predictor_cfg,
        "scheduler": {
            "tp_size": 8,
            "ep_size": 1,
            "dp_size": 1,
            "data_type": "FP8",
            "kv_cache_data_type": "FP8",
            "backend_name": "sglang",
            "backend_version": "0.5.11",
        },
    }


def main():
    args = parse_args()

    # 路径
    script_dir = Path(__file__).parent
    trace_path = args.trace or str(script_dir / "sample_trace_20.jsonl")
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    hicache_dir = "/tmp/hicache_glm51_bl_trace"
    os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = hicache_dir

    # 构建并写入 sim config
    sim_config = build_sim_config(args)
    cfg_path = output_dir / "hisim_config.json"
    with open(cfg_path, "w") as f:
        json.dump(sim_config, f, indent=4)
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(cfg_path)

    # 清理旧数据
    if os.path.exists(hicache_dir):
        shutil.rmtree(hicache_dir)
    os.makedirs(hicache_dir, exist_ok=True)
    for fp in glob.glob("/tmp/sglang_simulator/hicache_storage_keys_*.txt"):
        os.remove(fp)
    sim_output = "/tmp/sglang_simulator/output"
    if os.path.exists(sim_output):
        shutil.rmtree(sim_output)

    # 加载数据
    print(f"[sim] Loading trace: {trace_path}")
    dataset = load_trace(trace_path, max_requests=args.max_requests)
    print(f"[sim] Loaded {len(dataset.data)} requests")
    print(f"[sim] Predictor: {args.predictor}")
    print(f"[sim] Model: {args.model_path}")
    print(f"[sim] max_total_tokens: {args.max_total_tokens}")

    # 启动仿真
    from sglang.srt.server_args import ServerArgs
    from sglang_simulator.simulation.sglang.bench_runner import SGLangBenchmarkRunner
    from sglang_simulator.simulation.benchmark import BenchmarkConfig

    server_config = {
        "model_path": args.model_path,
        "device": "cpu",
        "load_format": "dummy",
        "trust_remote_code": True,
        "max_total_tokens": args.max_total_tokens,
        "page_size": 256,
        "chunked_prefill_size": 16384,
        "max_prefill_tokens": 16384,
        "enable_hierarchical_cache": True,
        "disable_overlap_schedule": True,
        "hicache_storage_backend": "file",
        "hicache_storage_prefetch_policy": "wait_complete",
        "hicache_io_backend": "direct",
        "hicache_storage_backend_extra_config": json.dumps(
            {"hicache_storage_pass_prefix_keys": True}
        ),
        "hicache_ratio": 2.0,
        "disable_cuda_graph": True,
    }

    runner = SGLangBenchmarkRunner(ServerArgs(**server_config))

    print(f"\n[sim] Running simulation...")
    t0 = time.time()
    metrics = runner.benchmark(BenchmarkConfig(request_rate=1), dataset)
    dur = time.time() - t0

    # 保存产物
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if hasattr(runner, "get_iteration_stats"):
        with open(output_dir / "iteration.jsonl", "w") as f:
            for item in runner.get_iteration_stats():
                f.write(json.dumps(item) + "\n")
    if hasattr(runner, "get_request_stats"):
        with open(output_dir / "request.jsonl", "w") as f:
            for item in runner.get_request_stats():
                f.write(json.dumps(item) + "\n")

    # L3 IO trace
    l3_io_src = "/tmp/sglang_simulator/output/l3_io.jsonl"
    if os.path.exists(l3_io_src):
        shutil.copy2(l3_io_src, output_dir / "l3_io.jsonl")

    # 统计
    l3_io_path = output_dir / "l3_io.jsonl"
    l3_ops = sum(1 for _ in open(l3_io_path)) if l3_io_path.exists() else 0
    l3_keys_path = "/tmp/sglang_simulator/hicache_storage_keys_kv.txt"
    l3_keys = sum(1 for _ in open(l3_keys_path)) if os.path.exists(l3_keys_path) else 0

    print(f"\n{'='*60}")
    print(f"  Simulation Complete ({dur:.1f}s)")
    print(f"{'='*60}")
    print(f"  Requests:        {metrics.get('num_requests', 0)}")
    print(f"  mean TTFT:       {metrics.get('mean_ttft_ms', 0):.1f} ms")
    print(f"  mean TPOT:       {metrics.get('mean_tpot_ms', 0):.2f} ms")
    print(f"  storage_hit:     {metrics.get('kv_cache_storage_hit_ratio', 0):.2%}")
    print(f"  host_hit:        {metrics.get('kv_cache_host_hit_ratio', 0):.2%}")
    print(f"  L3 IO ops:       {l3_ops}")
    print(f"  L3 keys (cum):   {l3_keys}")
    print(f"{'='*60}")
    print(f"  Output dir:      {output_dir}")
    print(f"  L3 IO trace:     {output_dir / 'l3_io.jsonl'}")

    runner.shutdown()


if __name__ == "__main__":
    main()
