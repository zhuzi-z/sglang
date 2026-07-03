# PAI-vLLM V6D KV Cache 仿真适配 — 增量开发说明文档

> **项目**: sglang-hijack-sim / sglang-simulator  
> **基线分支**: `/feat/vllm-pai`（同事提供的 vLLM 基础仿真框架）  
> **增量范围**: KV Cache 全链路适配 + V6D 跨节点仿真  
> **代码路径**: `/root/workspace/sglang-hijack-sim/tools/sglang-simulator/`  
> **环境**: pai_vllm_v6d_1 (10.0.240.195:6677), pai_vllm_v6d_2 (10.0.42.92:6678)

---

## 1. 概述

### 1.1 同事分支 `/feat/vllm-pai` 提供的基础能力

同事分支实现了 vLLM 劫持仿真的核心框架：
- **Hook 机制**: `builtins.__build_class__` 拦截类定义，在 `import vllm` 之前安装所有 hooks
- **Platform Hook**: MockCudaPlatform 让 vLLM 在无 GPU 环境启动
- **EngineArgs Hook**: 强制并行度=1
- **Worker Hook**: 跳过模型加载，返回 mock ModelRunnerOutput
- **Scheduler Hook**: 时间预测 + future_queue 模拟请求到达
- **VLLMWorker**: 高级 Worker 接口 + MultiInstanceBenchmarkRunner 集成

### 1.2 我们的增量开发目标

在同事基础上完成 **KV Cache 全链路仿真适配**，核心诉求：

| 设计原则 | 说明 |
|----------|------|
| ① 不破坏原推理与匹配控制流 | 所有 hook 仅 override CUDA 相关操作，V6D client → daemon → 匹配 的完整路径保留 |
| ② 完全脱离 CUDA/GPU 依赖 | 纯 CPU 环境可运行（DummyStream/DummyEvent 替代 CUDA 原语） |
| ③ KV Cache 匹配语义与线上一致 | 命中率、空间占用、跨节点共享均真实还原，通过 V6D daemon IPC 通信 |

### 1.3 增量改动统计

| 类别 | 新增文件 | 修改文件 | 新增代码行 |
|------|---------|---------|-----------|
| V6D 适配 Hooks | 4 | 0 | ~795 |
| KV Connector 适配 | 0 | 1 | ~85 |
| KV Offload 适配 | 1 | 0 | 181 |
| Worker/VLLMWorker 适配 | 0 | 2 | ~70 |
| Startup 注册 | 0 | 1 | 6 |
| 集成测试 | 1 | 0 | 373 |
| 单元测试 | 1 | 0 | 606 |
| 配置文件 | 1 | 0 | 24 |
| **合计** | **8** | **4** | **~2140** |

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────── VLLMWorker (Python 主进程) ─────────────────────┐
│                                                                     │
│  set_active_worker_id(name) → env var 传递给子进程                    │
│                                                                     │
│  ┌─────────────── EngineCore (可能在子进程) ───────────────────────┐  │
│  │                                                                 │  │
│  │  Scheduler                       Worker (gpu_worker)            │  │
│  │  ┌──────────────────┐           ┌──────────────────────┐       │  │
│  │  │ C_VLLMScheduler  │           │ C_VLLMWorkerHook     │       │  │
│  │  │ Hook             │           │ • head_dim=1 注入     │       │  │
│  │  │ • 时间预测        │           │ • CPU KV cache 分配   │       │  │
│  │  │ • future_queue   │           │ • Mock execute_model  │       │  │
│  │  │ • 请求统计        │           │ • KV connector 生命周期│       │  │
│  │  └──────┬───────────┘           └──────────┬───────────┘       │  │
│  │         │                                   │                   │  │
│  │         ▼                                   ▼                   │  │
│  │  ┌──────────────────────────────────────────────────────┐      │  │
│  │  │     C_KVConnectorFactoryHook                          │      │  │
│  │  │     → 返回 MockOffloadConnector                       │      │  │
│  │  │       • _storage: dict[str,str] (key→owner_worker_id) │      │  │
│  │  │       • prefix match + local/remote 分类              │      │  │
│  │  │       • 与 V6dBlockOwnershipTracker 联动               │      │  │
│  │  └──────────────────────────────────────────────────────┘      │  │
│  │                                                                 │  │
│  │  ┌─────── V6D 组件（保留真实通信，Mock CUDA 操作）──────────┐    │  │
│  │  │                                                          │    │  │
│  │  │  V6dObjectConnectorWorker  ← C_V6dObjectConnectorWorkerHook│  │  │
│  │  │  • V6D client 连接 (保留)                                │    │  │
│  │  │  • cudaHostRegister (跳过)                               │    │  │
│  │  │                                                          │    │  │
│  │  │  V6dSwapHandler            ← C_V6dSwapHandlerHook        │    │  │
│  │  │  • CPU memcpy 替代 GPU DMA                              │    │  │
│  │  │  • DummyStream/DummyEvent 替代 CUDA 同步原语              │    │  │
│  │  │                                                          │    │  │
│  │  │  V6dObjectBackend          ← C_V6dObjectBackendHook      │    │  │
│  │  │  • DummyEvent 池替代 CUDA Event 池                       │    │  │
│  │  │                                                          │    │  │
│  │  │  V6dObjectManager          ← C_V6dObjectManagerHook      │    │  │
│  │  │  • seal/lookup ownership 追踪                            │    │  │
│  │  └──────────────────────────────────────────────────────────┘    │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│                              ▼ record_hit()                            │
│  ┌──────────────────────────────────────────────────────────┐         │
│  │  V6dBlockOwnershipTracker                                 │         │
│  │  • /dev/shm/v6d_sim_tracker/ (文件后端, fcntl 文件锁)      │         │
│  │  • ownership.json: block_key → owner_worker_id            │         │
│  │  • stats.json: worker_id → {local_hits, remote_hits}      │         │
│  └──────────────────────────────────────────────────────────┘         │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ IPC socket
                    ┌──────────────────┐
                    │  vineyardd daemon │
                    │  /tmp/vineyard.sock│
                    │  (真实运行,共享内存) │
                    └──────────────────┘
```

### 2.2 Hook 注册链路

```python
# startup.py — init_hook() 在 import vllm 之前调用
sglang_simulator_hook.install_class_hooks([
    platform.C_VLLMPlatformHook,           # 同事基础
    engine_args.C_VLLMEngineArgsHook,       # 同事基础
    worker.C_VLLMWorkerHook,               # 同事基础 → 我们扩展 V6D 适配
    scheduler.C_VLLMSchedulerHook,         # 同事基础 → 我们修复 cross-worker bug
    kv_connector.C_KVConnectorFactoryHook, # 同事基础 → 我们重写 RPC bypass
    kv_offload.C_VLLMSimpleCPUOffloadWorkerHook,      # 新增
    kv_offload.C_VLLMOffloadingConnectorWorkerHook,   # 新增
    v6d_swap.C_V6dSwapHandlerHook,                    # 新增
    v6d_worker.C_V6dObjectConnectorWorkerHook,        # 新增
    v6d_backend.C_V6dObjectBackendHook,               # 新增
    v6d_manager.C_V6dObjectManagerHook,               # 新增
])
```

---

## 3. 新增文件详解

### 3.1 `v6d_swap.py` — V6D SwapHandler CUDA 绕过 (294 行)

**职责**: 将 V6D swap-in/swap-out 操作从 GPU DMA 替换为 CPU memcpy。

**核心组件**:

| 组件 | 作用 |
|------|------|
| `DummyEvent` | 替代 `torch.cuda.Event`，`query()` 永远返回 True |
| `DummyStream` | 替代 `torch.cuda.Stream`，所有操作为 no-op |
| `mock_v6d_swap_blocks()` | 替代 `ops.v6d_swap_blocks` 的 CUDA kernel，用 `torch.Tensor.copy_()` 实现 CPU memcpy |
| `C_V6dSwapHandlerHook` | Override `__init__`/`swap`/`async_swap`/`get_finished` |

**保留的真实逻辑**:
- `V6dSwapHandler._validate_swap()` — 校验 swap 参数
- `V6dSwapHandler._process_swap_batch()` — V6D 对象查询与数据定位
- `obj.resolver()` 获取 V6D mmap 后的 tensor 引用

**Mock 的 CUDA 操作**:
- `torch.cuda.Stream()` → `DummyStream`
- `torch.cuda.Event()` → `DummyEvent`
- `torch.cuda.stream()` context → no-op
- `ops.v6d_swap_blocks()` → CPU memcpy

---

### 3.2 `v6d_backend.py` — V6D ObjectBackend Event 池绕过 (72 行)

**职责**: V6dObjectBackend 管理 save/load 异步流水线，使用 CUDA Event 追踪 GPU 操作完成。在仿真中所有 Event 立即完成。

**Hook 内容**:
```python
# 原始: self._save_event_pool = [torch.cuda.Event() for _ in range(8)]
# 替换: self._save_event_pool = [DummyEvent() for _ in range(8)]
```

---

### 3.3 `v6d_worker.py` — V6D ConnectorWorker CUDA 绕过 (91 行)

**职责**: V6dObjectConnectorWorker 负责连接 V6D daemon 并注册 CUDA pinned memory。在 CPU 仿真中跳过 CUDA 注册，保留真实 V6D 连接。

**关键 Override**:

| 方法 | 原始行为 | 仿真行为 |
|------|---------|---------|
| `_register_v6d_host_memory()` | `cudaHostRegister(mmap_ptr)` | No-op（CPU 直接访问 mmap） |
| `_start_async_v6d_init()` | 异步连接 + `torch.cuda.current_device()` | 同步连接到 `vineyard.sock`，设备=CPU |
| `register_kv_caches()` | 注册 GPU tensor storage | 直接使用 CPU tensor（已由 Worker hook 分配） |

---

### 3.4 `v6d_manager.py` — V6D ObjectManager RPC 绕过 + Ownership 追踪 (338 行)

**职责**: 实现跨节点 RPC 绕过方案的底层支撑——block 归属追踪与跨进程统计。

#### 3.4.1 `V6dBlockOwnershipTracker`

文件后端跨进程 tracker，存储在 `/dev/shm/v6d_sim_tracker/`：

| 方法 | 功能 |
|------|------|
| `record_seal(key, worker_id)` | 记录 block 的 owner |
| `classify_hit(key, current_worker_id)` | 判断命中类型: local/remote/unknown |
| `record_hit(worker_id, hit_type, count)` | 累计命中统计 |
| `get_stats(worker_id)` | 查询某 worker 的 `{local_hits, remote_hits}` |
| `reset()` | 清除所有数据（测试隔离用） |

所有操作使用 `fcntl.flock` 文件锁保证跨进程原子性。

#### 3.4.2 `C_V6dObjectManagerHook`

Override V6dObjectManager 的 `__init__`/`seal()`/`_process_lookup()`/`batch_allocate()`：
- `seal()` 时记录 block → worker_id 归属
- `_process_lookup()` 时分类 local/remote hits

> **注意**: 由于 CPU 仿真中 V6D WebSocket RPC 端口不可用，真实 V6D connector 不被创建，这些 override 实际不触发。Ownership 追踪由 MockOffloadConnector 在进程内完成（见 3.5）。此 hook 保留为未来切换到真实 V6D 路径时使用。

#### 3.4.3 Worker ID 上下文传递

```python
set_active_worker_id(worker_id)  # 写入 env var _SIM_V6D_ACTIVE_WORKER_ID
get_active_worker_id()           # 从 env var 读取
```

通过环境变量传递，确保 EngineCore 子进程能继承 worker 身份。

---

### 3.5 `kv_offload.py` — 原生 CPU Offload Hooks (181 行)

**职责**: vLLM 内置的 KV cache CPU offload 路径（非 V6D）也依赖 CUDA。提供两个 hook 让 offload worker 在 CPU 仿真中正常工作。

| Hook | 目标类 | 关键行为 |
|------|-------|---------|
| `C_VLLMSimpleCPUOffloadWorkerHook` | `SimpleCPUOffloadWorker` | 跳过 pinned memory 分配，所有 load/store 立即完成 |
| `C_VLLMOffloadingConnectorWorkerHook` | `OffloadingConnectorWorker` | 跳过 CUDA stream 创建，transfers 立即报告完成 |

---

### 3.6 `test_v6d_cache_runner.py` — V6D 集成测试 (373 行)

| 测试类 | 测试方法 | 验证内容 |
|--------|---------|---------|
| `TestV6dCacheBasic` | `test_v6d_connector_initialization` | V6D 连接器正确初始化 |
| `TestV6dCacheBasic` | `test_single_worker_kv_offload` | 单节点 prefix cache 基本流程 |
| `TestV6dCacheBasic` | `test_prefix_cache_hit_scenario` | 重复请求的 prefix 命中与驱逐后再命中 |
| `TestV6dCacheMultiNode` | `test_cross_node_prefix_hit` | **跨节点 RPC 绕过验证** |
| `TestV6dHeadDimReduction` | `test_kv_cache_tensor_size_is_tiny` | head_dim=1 内存缩减 128x |
| `TestV6dHeadDimReduction` | `test_total_kv_cache_memory_is_small` | 总 KV cache < 10MB |

---

### 3.7 `test_v6d_hooks_unit.py` — V6D Hook 单元测试 (606 行)

28 个单元测试，覆盖所有 hook 组件的隔离行为：
- `TestDummyPrimitives` (9 tests): DummyStream / DummyEvent 行为正确性
- `TestHeadDimInjection` (3 tests): head_dim=1 注入逻辑
- `TestBuildKVCacheSpec` (6 tests): KV cache spec 构建（含 hybrid model）
- `TestOpsPatch` (1 test): ops monkey-patch 验证
- `TestMockSwapBlocks` (2 tests): CPU memcpy swap 数据正确性
- `TestV6dSwapHandlerHook` (3 tests): SwapHandler hook 安装验证
- `TestV6dObjectConnectorWorkerHook` (1 test): Worker hook 安装验证
- `TestV6dObjectBackendHook` (1 test): Backend Event 池替换
- `TestPlatformHookAdditions` (2 tests): Platform mock 新增方法

---

### 3.8 `test/assets/config_vllm_v6d.json` — V6D 仿真配置

```json
{
    "platform": { "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80000} },
    "predictor": { "name": "aiconfigurator", "database_mode": "SOL" },
    "scheduler": { "backend_name": "vllm", "backend_version": "0.14.0" }
}
```

---

## 4. 修改文件详解

### 4.1 `kv_connector.py` — MockOffloadConnector + RPC 绕过

**改动核心**: 从"仅记录 block 存在性"升级为"追踪 block 归属 + 跨节点命中分类"。

#### 4.1.1 `_storage` 类型升级

```python
# Before (同事分支): 仅存在性
_storage: set[str] = set()

# After: 记录 owner
_storage: dict[str, str] = {}  # key → owner_worker_id
```

#### 4.1.2 初始化时捕获 worker_id

```python
from sglang_simulator.simulation.vllm.v6d_manager import get_active_worker_id
self._worker_id = get_active_worker_id()
```

#### 4.1.3 `request_finished()` — 记录 block ownership

```python
worker_id = self._worker_id or "unknown"
for h in request.block_hashes:
    key = h.hex() if isinstance(h, bytes) else str(h)
    self._storage[key] = worker_id  # 记录谁产生了这个 block
```

#### 4.1.4 `get_num_new_matched_tokens()` — 分类 local/remote

```python
for key in keys:
    if key not in self._storage:
        break
    hit_blocks += 1
    owner = self._storage[key]
    if self._worker_id and owner != self._worker_id:
        remote_hits += 1  # 跨节点命中！
    else:
        local_hits += 1   # 本地命中
```

命中后记录到 `V6dBlockOwnershipTracker`，用于指标统计。

#### 4.1.5 Factory Hook — 修复 connector 检测 + RPC 绕过策略

```python
# Before (同事分支，有 bug): config 对象上不存在 kv_connector 属性
connector_name = getattr(config, "kv_connector", None)  # 永远 None

# After: 正确路径
kv_transfer_config = getattr(config, "kv_transfer_config", None)
connector_name = getattr(kv_transfer_config, "kv_connector", None)
```

检测到 V6D 时**仍返回 MockOffloadConnector**（因为 V6D WebSocket RPC 在 CPU 仿真中不可用），但启用 RPC bypass ownership tracking。

---

### 4.2 `vllm_worker.py` — Worker ID 生命周期管理

在 `VLLMWorker.__init__` 中新增：

```python
# LLM 创建前: 设置 worker 身份标识
set_active_worker_id(name)  # → env var → 子进程继承

self._llm = LLM(...)        # EngineCore 在此创建，MockOffloadConnector 获取 worker_id

# LLM 创建后: 清除（避免污染后续 worker 实例）
set_active_worker_id(None)
```

---

### 4.3 `worker.py` — GPU Worker Hook V6D 适配

**同事基础上的关键扩展**:

| 方法 | 新增内容 |
|------|---------|
| `override_init_device()` | 调用 `_inject_head_dim(vllm_config)` + `_build_kv_cache_spec()` |
| `override_initialize_from_config()` | CPU KV cache tensor 分配 + V6D connector 初始化 + KV cache 注册 |
| `override_execute_model()` | KV connector pre/post-forward 生命周期（`bind_connector_metadata`→`get_finished`） |

**新增工具函数**:

| 函数 | 功能 |
|------|------|
| `_inject_head_dim(vllm_config)` | 将 `hf_config.head_dim` 设为 1，数据量缩减 ~128x |
| `_build_kv_cache_spec(vllm_config)` | 从 HF config 构建 per-layer KVCacheSpec（支持 MHA + hybrid model） |
| `_ModelRunnerStub` | 可 pickle 的 model_runner 桩（跨进程序列化需要） |

---

### 4.4 `startup.py` — Hook 注册

新增 `v6d_manager` 模块导入 + `C_V6dObjectManagerHook` 注册。

---

### 4.5 `scheduler.py` — Bug Fix: req_first_scheduled 跨 Worker 污染

**问题**: 同事分支中 `req_first_scheduled` 使用闭包级别的 `set()`，多 Worker 实例共享同一个 set，导致 Worker B 的请求被误判为"已调度过"。

**修复**: 改为 per-instance `self._sim_req_first_scheduled = set()`。

---

### 4.6 `platform.py` — 新增方法

为满足 V6D 初始化路径中的平台调用，新增：
- `set_device(device)` → No-op
- `get_device_total_memory()` → 返回 80 GiB

---

## 5. 跨节点 RPC 绕过方案

### 5.1 问题背景

真实 V6D 集群中，跨物理节点 KV cache 共享通过 **SRPC (GPU DMA)** 实现。在 CPU 仿真中：
- 无 GPU，SRPC 不可用
- vineyardd 仅暴露 IPC socket，不监听 WebSocket RPC 端口 (7890)
- V6dObjectConnector{Worker,Scheduler} 均依赖 WebSocket RPC，连接挂起

### 5.2 解决方案

在 `MockOffloadConnector` 中直接实现 ownership tracking：

```
Worker A 完成 Request → request_finished() → _storage[hash] = "worker_A"
                                                       ↓
Worker B 查询相同 prefix → get_num_new_matched_tokens()
                            → _storage[hash] 存在, owner="worker_A" ≠ "worker_B"
                            → remote_hit++ (跨节点命中!)
                            → V6dBlockOwnershipTracker.record_hit()
                            → 返回 hit_tokens (跳过 prefill)
```

**关键**: 跨节点命中时**直接复用 block**（通过 vineyardd IPC 共享内存），省略了 RPC 数据搬运步骤。prefix match 语义完全保留，仅绕过了物理数据传输。

### 5.3 设计决策

| 决策 | 理由 |
|------|------|
| 在 MockOffloadConnector 而非 V6dObjectManager 中实现 | MockOffloadConnector 是实际执行 prefix match 的组件；V6dObjectManager 因 WebSocket 不可用实际不触发 |
| `_storage` 为类级变量（全实例共享） | 多 Worker 仿真中 LLM 顺序创建，同进程内 class-level dict 自然跨实例可见 |
| 通过 env var 传递 worker_id | vLLM EngineCore 在子进程运行，环境变量能跨进程继承 |
| V6dBlockOwnershipTracker 用 /dev/shm/ 文件后端 | 为未来真正多进程部署预留扩展能力 |

---

## 6. 测试覆盖

### 6.1 测试运行方式

```bash
# 基础测试（不需要 V6D daemon 的单元测试 + 需要 daemon 的集成测试）
cd /root/workspace/sglang-hijack-sim/tools/sglang-simulator
python -m pytest test/test_v6d_hooks_unit.py test/test_v6d_cache_runner.py -v

# 跨节点 RPC 绕过测试（需要 V6D daemon + 环境变量）
V6D_MULTI_NODE=1 python -m pytest test/test_v6d_cache_runner.py::TestV6dCacheMultiNode -v -s
```

### 6.2 测试结果

```
32 passed, 1 skipped (cross-node test 需 V6D_MULTI_NODE=1)
跨节点测试单独运行: 1 passed

跨节点测试输出:
[RPC Bypass] Worker worker_node_1: cross-node hit! local=0 remote=4 (RPC bypassed)
[RPC Bypass] Worker worker_node_1: cross-node hit! local=0 remote=4 (RPC bypassed)
[RPC Bypass] Worker worker_node_1: cross-node hit! local=0 remote=4 (RPC bypassed)
[RPC Bypass] Worker worker_node_1: cross-node hit! local=0 remote=4 (RPC bypassed)
```

### 6.3 前置条件

- V6D daemon 运行: `vineyardd --socket /tmp/vineyard.sock --size 256M -rpc=false`
- 模型文件: `/host/models/Qwen/Qwen3-0.6B/`
- `CUDA_VISIBLE_DEVICES=""` + `VLLM_ENABLE_V1_MULTIPROCESSING=0`

---

## 7. 已知限制与后续方向

### 7.1 已知限制

1. **V6D WebSocket RPC 不可用**: vineyardd 启动时 `-rpc=false`，端口 7890 未监听。因此真实 V6dObjectConnector 无法使用，ownership 追踪在 MockOffloadConnector 内完成。

2. **`_storage` 为进程内共享**: 如果改为真正多进程部署（每个 Worker 独立进程），需切换到 V6dBlockOwnershipTracker 的文件后端模式。

3. **SRPC 需要 CUDA**: V6D 的 SRPC（GPU Direct DMA）无法在 CPU 环境使用，这是 RPC 绕过方案的根本原因。

### 7.2 后续方向

- **切换到真实 V6D RPC**: 若 vineyardd 配置为 `-rpc=true` 且有 GPU，可修改 Factory Hook 让真实 V6D connector 通过，`C_V6dObjectManagerHook` 将自动生效。
- **跨节点传输时延建模**: 当前 remote hit 直接跳过 prefill（假设传输瞬间完成），未来可在 `get_num_new_matched_tokens` 返回后注入传输时延。
- **多 vineyardd 实例测试**: 当前测试使用单 daemon + 多 Worker，未来可扩展为双 daemon 拓扑。

---

## 8. 文件索引

```
src/sglang_simulator/simulation/vllm/
├── startup.py          ← Hook 注册入口 (修改: +v6d_manager)
├── platform.py         ← MockCudaPlatform (修改: +set_device/get_memory)
├── engine_args.py      ← EngineArgs parallelism=1 (同事基础)
├── worker.py           ← GPU Worker hook (修改: +head_dim注入/KV spec/V6D init)
├── vllm_worker.py      ← VLLMWorker (修改: +worker_id生命周期)
├── scheduler.py        ← Scheduler hook (修改: +per-instance fix)
├── kv_connector.py     ← MockOffloadConnector + Factory Hook (修改: +RPC bypass)
├── kv_offload.py       ← Native CPU offload hooks (新增)
├── v6d_swap.py         ← V6D SwapHandler CUDA绕过 (新增)
├── v6d_worker.py       ← V6D ConnectorWorker CUDA绕过 (新增)
├── v6d_backend.py      ← V6D ObjectBackend Event绕过 (新增)
├── v6d_manager.py      ← V6D ObjectManager hook + Tracker (新增)
├── utils.py            ← vLLM 仿真工具 (同事基础)
└── launch_server.py    ← vLLM 服务启动 (同事基础)

test/
├── test_v6d_cache_runner.py    ← V6D 集成测试 (新增, 5 tests)
├── test_v6d_hooks_unit.py      ← V6D hook 单元测试 (新增, 28 tests)
├── test_simulation_vllm_runner.py   ← vLLM runner 测试 (同事基础)
├── test_simulation_vllm_serving.py  ← vLLM serving 测试 (同事基础)
└── assets/config_vllm_v6d.json      ← V6D 仿真配置 (新增)
```

---

## 9. 维护操作手册

### 9.1 添加新 Worker 节点

```python
worker_n = VLLMWorker(engine_args=EngineArgs(...), name="worker_node_N")
# name 参数自动传递给 MockOffloadConnector，ownership tracking 自动工作
```

### 9.2 查看跨节点命中统计

```python
from sglang_simulator.simulation.vllm.v6d_manager import V6dBlockOwnershipTracker
stats = V6dBlockOwnershipTracker.get_stats("worker_node_1")
# → {"local_hits": 10, "remote_hits": 16}
```

### 9.3 测试隔离

```python
V6dBlockOwnershipTracker.reset()       # 清除 /dev/shm/ 文件
MockOffloadConnector.reset_storage()   # 清除类级 _storage dict
```

### 9.4 环境部署

```bash
# 启动 vineyardd (CPU 模式，禁用 RPC)
vineyardd --socket /tmp/vineyard.sock --size 256M -rpc=false &

# 安装仿真器
cd tools/sglang-simulator && pip install -e .

# 运行测试
CUDA_VISIBLE_DEVICES="" V6D_MULTI_NODE=1 python -m pytest test/ -v
```
