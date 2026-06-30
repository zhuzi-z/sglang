# GLM-5.1 L3 IO 端到端仿真 — 完整还原指南

> **目的**：在 sglang-simulator 上以真实 trace 驱动 GLM-5.1 的多级 KV cache 仿真，
> 导出 L3 IO 操作流 (`l3_io.jsonl`)，用于后续存储带宽建模和容量规划。

---

## 一、整体架构概述

### 1.1 仿真器工作原理

sglang-simulator 通过 **Python class hook 机制** 在不加载真实模型权重的情况下，
复用 sglang 的完整调度链路（Scheduler → ModelRunner → RadixCache → HiCache），
仅将 GPU 计算替换为 **时延预测器**（predictor）。

核心流程：
```
┌─────────────────────────────────────────────────────┐
│  用户脚本 (run_sim.py)                              │
│    ↓                                                │
│  SGLangBenchmarkRunner                              │
│    ├─ install_class_hooks() ← 注入 mock            │
│    ├─ Engine(server_args) ← sglang 原生引擎        │
│    │    └─ Scheduler (子进程, fork 继承 hook)       │
│    │         ├─ ModelRunner.forward() → predictor   │
│    │         ├─ HiRadixCache → L3 IO 日志           │
│    │         └─ HiCacheStorage → file backend mock  │
│    └─ benchmark(config, dataset) → metrics          │
└─────────────────────────────────────────────────────┘
```

### 1.2 关键 Hook 机制

仿真器通过替换 `builtins.__build_class__` 来拦截类定义：

- **ModelRunner hook**: 将 `initialize()` 替换为 mock 版本（跳过权重加载），
  将 `forward_batch_generation()` 替换为调用 predictor 的 sleep
- **MemPoolHost hook**: mock 内存分配（CPU 模式无真实 HBM）
- **HiCacheStorage hook**: 拦截存储操作，记录到 `l3_io.jsonl`

**关键约束**：sglang 的 Engine 在子进程启动时调用 `mp.set_start_method("spawn")`，
spawn 模式会创建全新 Python 进程导致 hook 丢失。**必须在脚本开头强制 fork 模式**。

### 1.3 L3 IO 日志

L3 IO 日志记录所有对三级缓存（disk/file storage）的操作：

| 字段 | 含义 |
|---|---|
| `ts` | 仿真全局时钟 (s) |
| `wall_ts` | 真实墙钟 |
| `iteration` | 调度器 iteration 编号 |
| `op` | 操作类型: `exists` / `set` / `get` |
| `api` | API 版本: `v1` (HiRadixCache 路径) |
| `pool` | 缓存池: `kv` |
| `page_size` | 每个 page 的 token 数 |
| `n_keys` | 本次操作涉及的 page 数量 |
| `keys` | page hash 列表 |
| `hits` | 命中/写入结果 |
| `prefix_keys` | 前缀 page hash（用于拼接完整 token 指纹） |

---

## 二、环境准备（从零开始）

### 2.1 基础环境要求

| 项 | 版本 |
|---|---|
| Python | 3.12.x |
| PyTorch | >= 2.11 (CPU-only 即可) |
| transformers | >= 5.12.x (GLM-5.1 模型需要此版本) |
| scikit-learn | >= 1.4 (GBR predictor 依赖) |
| numpy | >= 1.26 |
| OS | Linux (需要 fork 支持, macOS 不保证) |

### 2.2 克隆代码

```bash
# 克隆 sglang 仓库并切换到本分支
git clone https://github.com/zhuzi-z/sglang.git
cd sglang
git checkout linsiyuan/glm-5.1-L3-io-dev
```

### 2.3 安装 sglang (editable mode)

```bash
# 安装 sglang 主包（跳过 CUDA 编译）
pip install -e "python/.[all]" --no-build-isolation 2>/dev/null || \
pip install -e "python/" --no-build-isolation

# 安装 sglang-simulator
pip install -e "tools/sglang-simulator/"
```

### 2.4 安装额外依赖

```bash
pip install scikit-learn>=1.4  # GBR predictor
pip install transformers>=5.12  # GLM-5.1 模型 config 解析
```

### 2.5 模型文件

仿真模式只读取模型的 config.json / tokenizer 文件（`load_format="dummy"` 跳过权重），
需要模型目录存在：

```
/nfs/models/ZhipuAI/GLM-5.1-FP8/
├── config.json
├── tokenizer.json / tokenizer_config.json
├── generation_config.json
└── ... (权重文件可以不下载)
```

如果环境中没有该模型路径，可通过 `--model-path` 指定任何已有的模型路径，
但模型架构会影响 sglang 的 page_size 自动检测逻辑。

---

## 三、运行仿真

### 3.1 快速验证（20 条 trace + fixed 预测器）

```bash
cd tools/sglang-simulator/examples/glm51_bl_trace_l3
python run_sim.py
```

预期输出（约 10s）：
```
  Simulation Complete (10.1s)
  Requests:        20
  mean TTFT:       ~30000-46000 ms
  storage_hit:     0.00%
  host_hit:        ~15%
  L3 IO ops:       ~220
```

### 3.2 使用 GBR 预测器（更真实的时延）

```bash
python run_sim.py --predictor gbr \
    --gbr-database /nfs_3820/projects/kunlun_insight/data/serving/bailian/B300-GLM-5-NVFP4-TP8/260522.top_pods_hisim_glm-5.run_batch
```

GBR 预测器在首次启动时从 9 个 pod 的 `schedule_batch.jsonl` 训练（约 27s），
后续可通过 `model_path` 参数缓存训练好的模型。

### 3.3 完整仿真（1835 条 trace）

```bash
python run_sim.py --predictor gbr \
    --gbr-database /nfs_3820/projects/kunlun_insight/data/serving/bailian/B300-GLM-5-NVFP4-TP8/260522.top_pods_hisim_glm-5.run_batch \
    --trace /nfs_3820/users/maruiyan.mry/bl_trace/multi_node_trace_combine_glm-5/260531/multi_node_trace_combine_glm-5/hisim-num-node-1-glm-5-blksz-256-bucket-85-128-cnt-1835-time-60min.jsonl \
    --max-total-tokens 2944960
```

预期耗时约 4 分钟，产出：
- ~3000 条 L3 IO ops
- ~200K L3 keys

---

## 四、产物说明

| 文件 | 内容 |
|---|---|
| `output/metrics.json` | 仿真指标（TTFT/TPOT/hit ratio 等） |
| `output/iteration.jsonl` | 每个调度 iteration 的统计 |
| `output/request.jsonl` | 每个请求的统计 |
| `output/l3_io.jsonl` | **L3 IO 操作流**（核心产物） |
| `output/hisim_config.json` | 本次仿真使用的配置 |

---

## 五、代码结构与关键模块

### 5.1 目录结构

```
sglang/
├── python/sglang/srt/          # sglang 推理引擎源码
│   ├── entrypoints/engine.py   # Engine 入口（mp.set_start_method 在此）
│   ├── managers/scheduler.py   # Scheduler 调度逻辑
│   └── mem_cache/
│       ├── hiradix_cache.py    # HiRadixCache (L1-L2-L3 分级缓存)
│       └── hicache_storage.py  # HiCacheStorage (L3 file backend)
│
└── tools/sglang-simulator/
    └── src/sglang_simulator/
        ├── hook/
        │   ├── class_hook_entry.py    # builtins.__build_class__ hook 核心
        │   └── base_hook.py           # hook 基类
        ├── simulation/
        │   ├── sglang/
        │   │   ├── bench_runner.py    # SGLangBenchmarkRunner 入口
        │   │   ├── model_runner.py    # ModelRunner mock (注入 predictor)
        │   │   ├── scheduler.py       # Scheduler hook (注入仿真时钟)
        │   │   ├── hicache_storage.py # HiCacheStorage mock (L3 IO 日志)
        │   │   └── hiradix_cache.py   # HiRadixCache hook
        │   ├── manager/
        │   │   ├── config.py          # ConfigManager (读 predictor 配置)
        │   │   ├── state.py           # 全局仿真状态
        │   │   └── l3_io_log.py       # L3IOLog 写入器
        │   └── benchmark/             # BenchmarkConfig
        ├── time_predictor/
        │   ├── base.py               # InferTimePredictor 基类
        │   ├── fixed.py              # FixedTimePredictor (常数时延)
        │   ├── gbr.py                # GBRTimePredictor (GradientBoosting)
        │   └── aiconfigurator.py     # AIConfiguratorTimePredictor
        └── dataset/                  # SimpleDataset / GenericRequest
```

### 5.2 核心工作流详解

#### (a) Hook 安装 (`class_hook_entry.py`)

```python
builtins.__build_class__ = _custom_build_class_
# 当 Python 定义任何 class 时，先检查是否命中 hook 列表
# 命中则调用 hook.hook(target_class) 注入 mock 方法
```

#### (b) ModelRunner Mock (`simulation/sglang/model_runner.py`)

- `override_initialize()`: 跳过权重加载，只初始化 token 容量
- `override_forward_batch_generation()`: 不做计算，调用 predictor 获取时延，
  用 `asyncio.sleep(predicted_time)` 模拟耗时
- 返回随机 token_ids 作为 next_token

#### (c) HiCacheStorage Mock (`simulation/sglang/hicache_storage.py`)

- 拦截 `exists()` / `set()` / `get()` 调用
- 记录操作到 `L3IOLog`，输出到 `/tmp/sglang_simulator/output/l3_io.jsonl`
- 维护内存中的 key 集合模拟存储命中

#### (d) GBR Predictor (`time_predictor/gbr.py`)

- 训练数据：线上服务化的 `schedule_batch.jsonl`（每条记录含 batch 组成 + 实测延迟）
- 特征：11 维聚合特征（batch_size / total_extend / max_prefix / ...）
- Prefill: GBR 回归预测
- Decode: 固定 0.02s/iter（GBR 不学 decode）

### 5.3 关键环境变量

| 变量 | 作用 |
|---|---|
| `CUDA_VISIBLE_DEVICES=""` | 禁用 GPU |
| `SGLANG_USE_CPU_ENGINE=1` | CPU 模式 |
| `SGLANG_ENABLE_UNIFIED_RADIX_TREE=0` | 关闭 UnifiedRadixCache，启用 HiRadixCache |
| `SGLANG_SIMULATOR_CONFIG_PATH` | 仿真配置 JSON 路径 |
| `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` | L3 file storage 目录 |
| `HISIM_LOG_LEVEL` | 仿真器日志级别 |

### 5.4 关键约束

1. **必须 fork 不能 spawn**：spawn 创建新进程丢失 hook
2. **必须关闭 UnifiedRadixCache**：GLM-5.1 默认走 Unified，但它不支持 L3
3. **page_size 可能被自动覆盖**：GLM-5.1 config 中的 NSA/DSA 检测可能将 page_size 覆盖为 64
4. **GBR 训练数据必须与目标硬件匹配**：B300 数据不能预测 H20

---

## 六、Trace 数据格式

输入 JSONL 每行一个请求：

```json
{
  "created_time": 1778497200.031,    // UNIX timestamp (arrival time)
  "input_length": 92672,              // prompt token 数
  "output_length": 1,                 // 生成 token 数 (纯 prefill = 1)
  "model_name": "glm-5",
  "request_id": "uuid-...",
  "input_ids": [1, 2, 3, ...]        // 完整 token id 列表 (用于 prefix cache hash)
}
```

- `input_ids` 是必须字段 — prefix cache 通过 token 序列计算 page hash
- `created_time` 用于 OFFLINE 模式按真实到达时序排序请求

---

## 七、常见问题

### Q1: 报错 `No module named 'vllm'`
**原因**：hook 未生效，ModelRunner 尝试加载真实模型。
**解决**：确保脚本最开头有 `mp.set_start_method("fork", force=True)`。

### Q2: L3 IO ops 为 0
**原因**：`SGLANG_ENABLE_UNIFIED_RADIX_TREE` 未设为 0，走了 UnifiedRadixCache 路径。
**解决**：`os.environ["SGLANG_ENABLE_UNIFIED_RADIX_TREE"] = "0"`

### Q3: GBR 训练报错 "No Prefill samples found"
**原因**：schedule_batch.jsonl 路径不对或文件为空。
**解决**：检查 `--gbr-database` 路径下是否有 `*/data/*/no_cache.schedule_batch.jsonl`。

### Q4: page_size 与预期不符
**原因**：GLM-5.1 的 NSA/DSA 配置触发 sglang 自动检测将 page_size 覆盖为 64。
**说明**：不影响 L3 IO 功能，仅影响 page 粒度和 key 数量统计。

---

## 八、参考数据路径（NFS）

| 资源 | 路径 |
|---|---|
| GLM-5.1-FP8 模型 | `/nfs/models/ZhipuAI/GLM-5.1-FP8/` |
| 完整 trace (1835 条) | `/nfs_3820/users/maruiyan.mry/bl_trace/multi_node_trace_combine_glm-5/260531/multi_node_trace_combine_glm-5/hisim-num-node-1-glm-5-blksz-256-bucket-85-128-cnt-1835-time-60min.jsonl` |
| GBR 训练数据 (9 pod) | `/nfs_3820/projects/kunlun_insight/data/serving/bailian/B300-GLM-5-NVFP4-TP8/260522.top_pods_hisim_glm-5.run_batch/` |
| Backup 原始脚本 | `/nfs_3820/users/linsiyuan.lsy/backup_20260603/insight_benchmark_hisim/` |

---

## 九、版本信息

| 组件 | 版本 | 备注 |
|---|---|---|
| sglang | 0.5.12.post1 | editable install from this branch |
| sglang-simulator | 0.1.0 | editable install |
| Python | 3.12.3 | |
| PyTorch | 2.11.0+cu130 | CPU-only mode |
| transformers | 5.12.1 | GLM-5.1 支持需要 >= 5.12 |
| scikit-learn | 1.9.0 | GBR predictor |
| numpy | 1.26.4 | |
| Git branch | `linsiyuan/glm-5.1-L3-io-dev` | base: `linsiyuan/qwen35-hybrid-ssm` |
| Git remote | `origin` = `https://github.com/zhuzi-z/sglang.git` | |
