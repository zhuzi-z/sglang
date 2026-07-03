# HiSim Prefill 节点部署指南

## 概述

在 CPU 仿真环境中部署 sglang_simulator + PAI-LLM (dashllm) serving，模拟 B300 SXM 8TB HBM 硬件上的 prefill 节点行为。

**核心原理：** 通过 PTH hook 自动劫持 vLLM Worker 生命周期（init_device → load_model → determine_available_memory → initialize_from_config），在纯 CPU 环境下完成 KV Cache 调度仿真，不依赖任何 GPU/CUDA 硬件。

---

## 前置条件

| 项目 | 要求 |
|------|------|
| 目标机器 | pai_vllm_v6d_1 (SSH 可达) |
| Python | 3.12 |
| dashllm | 已预装 (含 dashservingd 守护进程) |
| 模型目录 | `/root/workspace/models/Qwen/Qwen3___5-0___8B` |
| etcd 服务 | `http://11.226.24.110:2379` (V6D Cache Storage 依赖) |
| 网络 | 可访问 gh-proxy.com (pip install 需要) |

---

## 第1步：环境变量配置

### 1.1 模型 & V6D 连接（必须根据实际环境修改）

```bash
export model_path="/root/workspace/models/Qwen/Qwen3___5-0___8B"
export V6D_ETCD_ENDPOINT="http://11.226.24.110:2379"
```

> **注意：** `model_path` 必须显式设置。dashllm 默认读取 `/dev/shm/model`，该路径在仿真环境中不存在会导致 `FileNotFoundError`。

### 1.2 DashLLM 引擎配置

```bash
export DASHGEN_DEPLOYMENT_ROLE="prefill"
export DS_GPU_NUM="0"
export DS_LLM_ENABLE_JEMALLOC="0"
export DS_LLM_ENABLE_PROCESS_REQUEST="1"
export DS_LLM_DECODE_TPOT_TIME_MS="18"
export DS_LLM_GRACEFUL_SHUTDOWN_WAIT_SECONDS="1200"
export DS_LLM_IGNORE_WARMUP="1"
export DS_LLM_MAX_THINK_TOKENS="81920"
export DS_LLM_MULTI_ENGINE_NUM="1"
export DS_LLM_PD_PREFILL_TIMEOUT_TIME="595"
export DS_LLM_SERVER_MAX_CONCURRENCY="256"
export DS_MODEL_PRELOAD_TO_SHM="0"
```

### 1.3 V6D 配置

```bash
export DS_LLM_LAUNCH_V6D="1"
export DS_LLM_SHARE_V6D="0"
export V6D_ARGS="--peer=tiered_vineyard --vineyard-size=500G --memory-usage-max=0.95 --tracker-ttl=86400 --memory-usage-min=0.85 --memory-usage-emergency-min=0.6"
export V6D_ENABLE_TRACKER="1"
export LAZY_INITIALIZE_KV_TRANSFER_OUTSIDE_VLLM="1"
export VLLM_V6D_ASYNC_REGISTER="1"
```

### 1.4 vLLM 配置

```bash
export VLLM_USE_V1="1"
export VLLM_DISABLE_COMPILE_CACHE="1"
export VLLM_DRAFT_USE_DYNAMIC_GLOBAL_SCALE_FOR_FP4="1"
export VLLM_ENABLE_MODEL_RUNNER_WARMUP="0"
export VLLM_FLASHINFER_MOE_BACKEND="latency"
export VLLM_FLASH_ATTN_FP8_ATTENTION="1"
export VLLM_FLASH_ATTN_VERSION="3"
export VLLM_GDN_USE_FLASHINFER="0"
export VLLM_PD_TRY_CONNECT_TIMEOUT_SECONDS="120"
export VLLM_QUANTIZE_ROUTED_EXPERTS_ONLY="1"
export VLLM_RESPONSE_TIMEOUT="590"
export VLLM_SKIP_KV_CACHE_ZEROING="1"
export VLLM_TIMEOUT_TRACEBACK="1"
export VLLM_USE_FLASHINFER_MOE_FP4="1"
export VLLM_USE_FP8_ATTN_PROJ_FOR_NVFP4="1"
```

### 1.5 NCCL / GPU 兼容（CPU仿真仍需设置以防 import 报错）

```bash
export NCCL_CUMEM_ENABLE="0"
export NCCL_SOCKET_IFNAME="lo"
export NVIDIA_IMEX_CHANNELS="0"
export NVSHMEM_DISABLE_IBGDA_MONITOR="1"
```

### 1.6 Aquila IPC & KV Transfer

```bash
export AQUILA_HEALTHY_PROCESS_TIME_AVG_THRESHOLD="7200000"
export AQUILA_SERVING_IPC_INIT_TIMEOUT="72000000"
export BLLM_KVTRANS_RDMA_SP="2"
export BLLM_KVTRANS_TXSTUB_CAP="3200"
export ACCL_MAX_USER_MR_GB="14"
```

### 1.7 杂项

```bash
export CUDA_CACHE_MAXSIZE="4294967296"
export OMP_NUM_THREADS="4"
export PYTHONHASHSEED="0"
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export SGLANG_SIMULATOR_MAX_DECODE_STEPS="1"
```

### 1.8 DS_LLM_ENGINE_CONFIG（vLLM EngineArgs JSON）

```bash
export DS_LLM_ENGINE_CONFIG='{"async_scheduling":false,"block_size":256,"compilation_config":{"cudagraph_mode":"NONE"},"disable_cascade_attn":true,"distributed_executor_backend":"mp","dtype":"bfloat16","enable_chunked_prefill":true,"enable_prefix_caching":true,"enable_think":1,"enforce_eager":false,"gpu_memory_utilization":0.93,"hf_overrides":{"text_config":{"max_position_embeddings":524288,"rope_parameters":{"dynamic":true,"factor":2,"original_max_position_embeddings":262144,"partial_rotary_factor":0.25,"rope_theta":10000000,"rope_type":"yarn","semi_dynamic":false,"type":"yarn"}}},"kv_transfer_config":{"kv_connector":"HybridConnector","kv_connector_extra_config":{"backend":"v6d_object+kvt"},"kv_role":"kv_both"},"limit_mm_per_prompt":{"image":0,"video":0},"mamba_cache_mode":"light","max_model_len":524288,"max_num_batched_tokens":8192,"mm_processor_cache_gb":0,"skip_mm_profiling":true,"speculative_config":{"hf_overrides":{"text_config":{"max_position_embeddings":524288,"rope_parameters":{"dynamic":true,"factor":2,"original_max_position_embeddings":262144,"partial_rotary_factor":0.25,"rope_theta":10000000,"rope_type":"yarn","semi_dynamic":false,"type":"yarn"}}},"max_model_len":524288,"method":"qwen3_5_mtp","num_speculative_tokens":3},"tensor_parallel_size":4,"think_mode":"auto"}'
```

---

## 第2步：安装依赖

```bash
# 安装 sglang_simulator（从 feat/vllm-v6d-dev-lsy 分支）
/usr/local/bin/python3 -m pip install \
    "git+https://gh-proxy.com/https://github.com/zhuzi-z/sglang.git@feat/vllm-v6d-dev-lsy#subdirectory=tools/sglang-simulator" \
    --no-deps

# 安装预测器依赖
/usr/local/bin/python3 -m pip install \
    aiconfigurator joblib scikit-learn==1.8.0 \
    --no-deps -i https://mirrors.cloud.tencent.com/pypi/simple
```

---

## 第3步：写入配置文件

### 3.1 HiSim 平台配置 (`/home/admin/hisim/config.json`)

```bash
mkdir -p /home/admin/hisim
cat > /home/admin/hisim/config.json <<'EOF'
{
  "platform": {
    "accelerator": {
      "name": "b300_sxm",
      "hbm_capacity_gb": 8000
    },
    "disk_read_bandwidth_gb": 64,
    "disk_write_bandwidth_gb": 64,
    "memory_read_bandwidth_gb": 64,
    "memory_write_bandwidth_gb": 64,
    "num_device_per_node": 8
  },
  "predictor": {
    "name": "aiconfigurator",
    "database_mode": "SOL"
  },
  "scheduler": {
    "tp_size": 4,
    "ep_size": 1,
    "dp_size": 1,
    "data_type": "BF16",
    "kv_cache_data_type": "FP16",
    "backend_name": "vllm",
    "backend_version": "0.19.0"
  }
}
EOF
```

### 3.2 PTH Hook 自动加载文件

```bash
cat > /usr/local/lib/python3.12/dist-packages/sglang_simulator.pth <<'EOF'
import sglang_simulator.simulation.vllm.startup as vllm_startup; vllm_startup.init_hook()
EOF
```

> **原理：** Python 启动时自动执行 `.pth` 文件中的代码，使所有子进程（EngineCore、Worker）都自动加载劫持 hook，无需修改 dashllm/vLLM 源码。

---

## 第4步：启动服务

```bash
# Serving 阶段额外覆盖
export DS_MODEL_PRELOAD_TO_SHM=0
export SGLANG_SIMULATOR_OUTPUT_MODE=BLOCKING
export SGLANG_SIMULATOR_CONFIG_PATH="/home/admin/hisim/config.json"
export VLLM_FLASH_ATTN_FP8_ATTENTION=0

# 启动
/usr/local/bin/dashllm_cmd serving
```

---

## 第5步：验证服务就绪

```bash
# 检查健康状态
curl -s http://localhost:8601/status

# 预期输出：
# {"readiness":true,"liveness":true,"workers":[{"healthy":true,...}]}
```

服务启动后约 55 秒 Worker 完成初始化，日志中应看到以下关键行：

```
[vLLM Hijack] Worker.determine_available_memory: capping simulated ... → ... to fit physical RAM
[vLLM Hijack] Allocated N MINIMAL CPU KV cache tensors (num_blocks=..., actual_bytes=minimal)
[vLLM Hijack] Worker.initialize_from_config: V6D-aware init complete
Worker is ready
All workers ready
Server started successfully!
```

---

## 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `FileNotFoundError: /dev/shm/model/config.json` | `model_path` 未设置 | 设置 `export model_path=<实际模型路径>` |
| Worker 进程 OOM Kill (zombie) | 仿真 HBM 过大导致 tensor 分配超出物理 RAM | 确认 sglang_simulator 版本 >= `6cac120` (含内置 memory cap) |
| EngineCore OOM Kill (exit -9) | block 元数据超出物理 RAM | 同上，确认版本含 `determine_available_memory` RAM 上限逻辑 |
| `OSError: port 8001 already in use` | 残留进程未清理 | `pkill -9 -f dashservingd` |
| V6D etcd 连接失败 | `V6D_ETCD_ENDPOINT` 未设置或不可达 | 检查 etcd 服务是否存活 |

---

## 附录：CPU 仿真内存安全机制

sglang_simulator 的 `worker.py` 已内置两层保护（无需外部 patch）：

1. **`determine_available_memory`**：将模拟的 HBM 可用字节数 cap 到物理 RAM 的 25%。例如模拟 8TB HBM 返回 7438 GiB，在 16GB RAM 机器上会被限制为 3.78 GiB，使 num_blocks 降至 ~150K（元数据 ~30MB）。

2. **`initialize_from_config`**：KV cache tensor 仅分配 minimal 大小（每个 4096 bytes），保留 num_blocks 计数用于调度和前缀匹配逻辑，但不实际分配 TB 级物理存储。
