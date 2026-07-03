#!/bin/bash
# ==============================================================================
# HiSim Prefill 节点部署脚本
# ==============================================================================
# 用途: 在 CPU 仿真环境中部署 sglang_simulator + dashllm serving
# 角色: prefill (通过 DASHGEN_DEPLOYMENT_ROLE 控制)
#
# 使用方式 (在目标机器上执行):
#   bash scripts/deploy_hisim_prefill.sh
#
# 前置条件:
#   - 机器上已有模型目录 (通过 model_path 环境变量指定)
#   - etcd 服务可达 (通过 V6D_ETCD_ENDPOINT 环境变量指定)
#   - Python 3.12 + dashllm 已安装
#
# 部署流程:
#   1. 清理残留进程
#   2. 设置环境变量
#   3. 安装 sglang_simulator (从源码 pip install)
#   4. 写入 HiSim 配置和 PTH hook
#   5. 启动 dashllm serving
#
# 关于 CPU 仿真内存管理 (无需外部 patch):
#   sglang_simulator 的 worker.py 已内置以下机制:
#   - determine_available_memory: 将模拟的 HBM 容量 cap 到物理 RAM 的 25%
#     以避免 EngineCore block 元数据 OOM (prefix tree, block tables 等)
#   - initialize_from_config: KV cache tensor 仅分配 minimal 大小 (4096 bytes/tensor)
#     保留 num_blocks 计数用于调度逻辑, 但不实际分配 TB 级存储
# ==============================================================================
set -euo pipefail

echo "=========================================="
echo "[HiSim] Prefill 节点部署开始"
echo "[HiSim] $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# ==============================================================================
# 第1步: 清理残留进程
# ==============================================================================
echo "[1/5] 清理残留进程..."
pkill -f "dashllm_cmd serving" 2>/dev/null || true
pkill -f "dashservingd" 2>/dev/null || true
pkill -f "vllm.engine" 2>/dev/null || true
sleep 2
if pgrep -f "dashllm_cmd|dashservingd" > /dev/null 2>&1; then
    echo "[WARN] 仍有残留进程, 强制 kill..."
    pkill -9 -f "dashllm_cmd|dashservingd" 2>/dev/null || true
    sleep 1
fi
echo "[1/5] 完成"

# ==============================================================================
# 第2步: 设置环境变量
# ==============================================================================
echo "[2/5] 设置环境变量..."

# ---- 模型 & V6D 连接 (必须根据实际环境修改) ----
export model_path="${model_path:-/root/workspace/models/Qwen/Qwen3___5-0___8B}"
export V6D_ETCD_ENDPOINT="${V6D_ETCD_ENDPOINT:-http://11.226.24.110:2379}"

# ---- DashLLM 引擎配置 ----
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

# ---- V6D ----
export DS_LLM_LAUNCH_V6D="1"
export DS_LLM_SHARE_V6D="0"
export V6D_ARGS="--peer=tiered_vineyard --vineyard-size=500G --memory-usage-max=0.95 --tracker-ttl=86400 --memory-usage-min=0.85 --memory-usage-emergency-min=0.6"
export V6D_ENABLE_TRACKER="1"
export LAZY_INITIALIZE_KV_TRANSFER_OUTSIDE_VLLM="1"
export VLLM_V6D_ASYNC_REGISTER="1"

# ---- vLLM ----
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

# ---- NCCL / GPU (CPU仿真仍需设置以防import报错) ----
export NCCL_CUMEM_ENABLE="0"
export NCCL_SOCKET_IFNAME="lo"
export NVIDIA_IMEX_CHANNELS="0"
export NVSHMEM_DISABLE_IBGDA_MONITOR="1"

# ---- Aquila IPC ----
export AQUILA_HEALTHY_PROCESS_TIME_AVG_THRESHOLD="7200000"
export AQUILA_SERVING_IPC_INIT_TIMEOUT="72000000"

# ---- KV Transfer ----
export BLLM_KVTRANS_RDMA_SP="2"
export BLLM_KVTRANS_TXSTUB_CAP="3200"
export ACCL_MAX_USER_MR_GB="14"

# ---- 杂项 ----
export CUDA_CACHE_MAXSIZE="4294967296"
export OMP_NUM_THREADS="4"
export PYTHONHASHSEED="0"
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export SGLANG_SIMULATOR_MAX_DECODE_STEPS="1"

# ---- DS_LLM_ENGINE_CONFIG (vLLM EngineArgs JSON) ----
export DS_LLM_ENGINE_CONFIG='{
  "async_scheduling": false,
  "block_size": 256,
  "compilation_config": {"cudagraph_mode": "NONE"},
  "disable_cascade_attn": true,
  "distributed_executor_backend": "mp",
  "dtype": "bfloat16",
  "enable_chunked_prefill": true,
  "enable_prefix_caching": true,
  "enable_think": 1,
  "enforce_eager": false,
  "gpu_memory_utilization": 0.93,
  "hf_overrides": {
    "text_config": {
      "max_position_embeddings": 524288,
      "rope_parameters": {
        "dynamic": true,
        "factor": 2,
        "original_max_position_embeddings": 262144,
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "yarn",
        "semi_dynamic": false,
        "type": "yarn"
      }
    }
  },
  "kv_transfer_config": {
    "kv_connector": "HybridConnector",
    "kv_connector_extra_config": {"backend": "v6d_object+kvt"},
    "kv_role": "kv_both"
  },
  "limit_mm_per_prompt": {"image": 0, "video": 0},
  "mamba_cache_mode": "light",
  "max_model_len": 524288,
  "max_num_batched_tokens": 8192,
  "mm_processor_cache_gb": 0,
  "skip_mm_profiling": true,
  "speculative_config": {
    "hf_overrides": {
      "text_config": {
        "max_position_embeddings": 524288,
        "rope_parameters": {
          "dynamic": true,
          "factor": 2,
          "original_max_position_embeddings": 262144,
          "partial_rotary_factor": 0.25,
          "rope_theta": 10000000,
          "rope_type": "yarn",
          "semi_dynamic": false,
          "type": "yarn"
        }
      }
    },
    "max_model_len": 524288,
    "method": "qwen3_5_mtp",
    "num_speculative_tokens": 3
  },
  "tensor_parallel_size": 4,
  "think_mode": "auto"
}'

echo "[2/5] 完成 (model_path=$model_path)"

# ==============================================================================
# 第3步: 安装 sglang_simulator
# ==============================================================================
echo "[3/5] 安装 sglang_simulator..."

URL_SGLANG_SIMULATOR_WHL="git+https://gh-proxy.com/https://github.com/zhuzi-z/sglang.git@feat/vllm-v6d-dev-lsy#subdirectory=tools/sglang-simulator"

/usr/local/bin/python3 -m pip install "$URL_SGLANG_SIMULATOR_WHL" --no-deps 2>&1 | tail -3
/usr/local/bin/python3 -m pip install aiconfigurator joblib scikit-learn==1.8.0 --no-deps \
    -i https://mirrors.cloud.tencent.com/pypi/simple 2>&1 | tail -3

echo "[3/5] 完成"

# ==============================================================================
# 第4步: 写入配置文件
# ==============================================================================
echo "[4/5] 写入 HiSim 配置..."

CONFIG_PATH="/home/admin/hisim/config.json"
PTH_PATH="/usr/local/lib/python3.12/dist-packages/sglang_simulator.pth"
mkdir -p "$(dirname "$CONFIG_PATH")"

cat > "$CONFIG_PATH" <<'EOF'
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

cat > "$PTH_PATH" <<'EOF'
import sglang_simulator.simulation.vllm.startup as vllm_startup; vllm_startup.init_hook()
EOF

echo "[4/5] 完成"

# ==============================================================================
# 第5步: 启动服务
# ==============================================================================
echo "[5/5] 启动 dashllm serving..."

# Serving 阶段覆盖
export SGLANG_SIMULATOR_OUTPUT_MODE=BLOCKING
export SGLANG_SIMULATOR_CONFIG_PATH="$CONFIG_PATH"
export VLLM_FLASH_ATTN_FP8_ATTENTION=0

echo "=========================================="
echo "[HiSim] model_path=$model_path"
echo "[HiSim] V6D_ETCD_ENDPOINT=$V6D_ETCD_ENDPOINT"
echo "[HiSim] CONFIG_PATH=$CONFIG_PATH"
echo "=========================================="

exec /usr/local/bin/dashllm_cmd serving
