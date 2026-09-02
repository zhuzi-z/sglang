# V6D 传输时延建模（CPU 仿真）

CPU 仿真把 v6d 的真实数据面（GPU↔host DMA、跨节点 SRPC）替换为 no-op，导致搬运"零耗时"、跨节点 block 过早可见。本模块补回这些时延，**全部 env 门控，默认关闭 = 原行为不变**。

## 建模的时延

| 段 | 内容 | 应用位置 | 口径 |
|---|---|---|---|
| **seg2 load** | host→GPU DMA (L2→L1)，forward 前 | `v6d_backend` load deadline | DMA 带宽 |
| **seg2' store** | GPU→host DMA (L1→L2)，forward 后 | `v6d_backend` store deadline | DMA 带宽 |
| **save_completion** | seal/announce 控制面完成（**非 DMA**），门控跨节点可见 | store deadline | 固定+每块 |
| **seg1** | 跨节点 fetch peer→local（SRPC/RDMA） | load deadline（远端命中） | 占位，待校准 |
| **cold_start** | 引擎首 iter 一次性预热（CUDA graph/JIT） | `scheduler` 首个非空 iter | 固定 |

## 文件

- `.../simulation/vllm/v6d/bandwidth.py` — 模型（读 profile、算时延）
- `.../simulation/vllm/v6d/v6d_backend.py` — 应用 seg2/seg2'/save_completion/seg1（deadline 机制，非阻塞）
- `.../simulation/vllm/scheduler.py` — 应用 cold_start
- `tools/bw_calib/collect_bandwidth.py` — DMA 段采集器（双后端：v6d_kernel / torch）
- `tools/bw_calib/README.md` — 采集器详细说明（后端探测、诊断轴、profile 字段布局）
- `tools/bw_calib/bandwidth_profile.example.json` — 示例 profile（**schema 1 旧格式，仅示意**；当前产物为 schema 3，字段以 README §5 为准）

## 采集（换环境时）

在目标 GPU 节点运行（完整参数见 `tools/bw_calib/README.md`）：
```bash
python tools/bw_calib/collect_bandwidth.py \
    --gpu 0 --page-size 2146304 --num-layers 12 \
    --blocks 1,2,4,8,16 --out bandwidth_profile.json
```
- **双后端**：`--backend auto`（默认）优先用生产 kernel `ops.v6d_swap_blocks`（PAI 版 vLLM）；镜像里没有定制 kernel 时自动回退纯 torch（裸 `cudaMemcpyAsync`，**只能当上界**）。每次运行都打印探测表说明选了哪个、另一个为何不可用；显式 `--backend v6d_kernel` 但不可用时直接 `exit 2`，不会深到测量里才报错。
- `--page-size` / `--num-layers`：模型相关（TP 分片后的单 rank page size、层数）。Qwen3.5-122B = 2146304 / 12；**沿用内置默认值时会打 `[note]` 提醒**，换模型/换 TP 必须显式传。
- 采集 seg2 load/store DMA + 并发模式（dual_dir 验全双工、dual_gpu 验 TP rank 争用；这两项需 kernel 后端，torch 后端用 `--streams N` 代替）。
- **seg1 不在此采集**：需要两个 peer + 可用的 SRPC/RDMA 数据面（`collect_remote.py`，真实硬件）。
- save_completion 默认（70+6/blk）来自真实运行日志分析；可用 `--save-floor-ms/--save-per-blk-ms` 覆盖。

### ⚠️ 生效带宽由 samples 末两点决定，不是 summary 字段

`SegmentModel.from_samples()` 把 `samples` 按 `bytes` 排序后**用最大的两个点重算边际带宽**；`bandwidth_gib_per_s` / `fixed_overhead_s` 只是采集器的记录字段，**运行时不参与计算**（本环境 profile 记 60.88 GiB/s，实际生效 48.73）。因此：

- 想改变生效带宽必须改 `samples` 数组，手改 summary 字段无效；
- 曲线若「先升后降」（GB300 实测 32/64 block 降速区：末两点算出 111 GiB/s，平坦区实为 197，低估 43%），采集器会**告警并默认只把单调前缀写入 `segments.*.samples`**（`--no-trim` 可关闭），全量 sweep 始终保留在 `diagnostics.*`；
- 每次运行都会打印真正生效的模型，直接看这行确认：
  ```text
  EFFECTIVE AT RUNTIME (from_samples(last two samples)): BW=48.73 GiB/s, floor=2.205 ms, t0=12.4 us
  ```

## 配置（跑仿真时）

```bash
export SGLANG_SIMULATOR_BW_PROFILE=/path/to/bandwidth_profile.json  # DMA 带宽 + 控制面
export SGLANG_SIMULATOR_COLD_START_S=1.43                           # 首 iter 冷启动（秒）
```
可选 env 覆盖（优先级 **env > profile > 0**）：
```bash
SGLANG_SIMULATOR_SAVE_CTRL_FLOOR_MS / _PER_BLK_MS   # 覆盖 save_completion
SGLANG_SIMULATOR_SEG1_FLOOR_MS      / _PER_BLK_MS   # 覆盖 seg1（校准前为占位 0）
```
未设 `BW_PROFILE` 且未设任何 env → 全部时延为 0，回到建模前行为。

## 本环境参数与来源（RTX PRO 6000 Blackwell，Qwen3.5-122B-A10B-FP8）

| 参数 | 值 | 来源 |
|---|---|---|
| local_store | t0=9.8µs, BW=47.68 GiB/s（干净仿射） | collect_bandwidth.py 实测 |
| local_load | `max(2.2ms, bytes/48.7GiB/s)` **非仿射** | collect_bandwidth.py 实测 |
| save_completion | 70ms + 6ms/block | 真实日志 `mark_saved − 末store`，190 样本，Pearson r=0.27 |
| cold_start | 1.43 s | 真实 iter0（1.789/1.408s − 预测值）|
| seg1 | 0（**占位**） | 数据面 stub，需真实硬件采集 |

## 换模型/GPU 重新校准

1. 用新的 `--page-size/--num-layers` 重跑 `collect_bandwidth.py` → 新 profile（seg2、layout）。
2. save_completion 与 cold_start：从一次真实运行的日志重新推导（`mark_saved−末store`；`iter0−预测`），或先沿用作初值。
3. seg1：有可用数据面时用 `collect_remote.py` 采集填入。

## seg1（跨节点）如何后续填值

seg1 是唯一跨网络的段。CPU 仿真里 SRPC/RDMA 数据面被 stub，**本环境无法测**。在有可用数据面的真实硬件上，测 peer→local fetch 带宽，填入 profile 的 `control_plane.seg1_cross_node`（或 SEG1 env）。模型形式：`max(floor, bytes/BW)`，与 DMA 段一致。注入点：远端命中（`external_tokens>0`）的 load deadline，`= seg1(远端块) + seg2(全部块)`。

## 机制说明（为什么不用 sleep）

时延通过 **deadline + 每步非阻塞轮询** 实现（`_sim_pending_store/_sim_pending_load` + `override_get_finished`），复用 BLOCKING 模式已有的 `time.sleep(predicted_latency)` 推进墙钟。**不新增任何 sleep** —— 早期在数据面协程里 `await asyncio.sleep()` 会因 BLOCKING 主流程独占线程而饿死协程、导致仿真卡死。

## 验证

sf=0.2 / N=100 ABAB 跨节点：前缀命中逐请求 **200/200 对齐实测**（连续 3 次）。默认（env 全未设）复现建模前行为。

## save_completion 参数说明（重要）

### 物理含义

save_completion（70ms + 6ms/blk）**不是 DMA 拷贝耗时，也不是控制面 RPC 耗时**。

在真实系统中，store 的完整链路是：
    bind_connector_metadata → forward（GPU计算）→ clear_backend_metadata
    → _async_do_store（事件循环）→ async_get → DMA(~11ms) → seal → 跨节点可见

其中 DMA 只有 ~11ms（实测打点），隔离 create+seal RPC 只有 ~2ms。但从
build_connector_meta（bind 时刻）到 seal 可见，实测中位 315ms（主要是等 forward
完成 + 事件循环调度延迟）。

仿真 BLOCKING 模式里 store 的 deadline 从 **bind 时刻**（forward 前）起算，
get_finished 在 **forward 后** 检查。这意味着：
- forward > save_completion 时：deadline 在 forward 期间到期 → 当步释放 seal
- forward < save_completion 时：deadline 未到期 → 延后到后续步

save_completion 的数值（70+6/blk）标定的是：**在仿真单线程 BLOCKING 架构下，
store→seal 可见需要额外延后多少才对齐真实系统中 forward 后的异步 store 链路时延
（约 39ms 中位：async_get + DMA + 10ms 轮询 + rank 同步）**。它是时序补偿参数，
不是独立的物理测量值。

### 采集器的控制面测量（参考）

采集器 --v6d-endpoint 测量的是**隔离的 create+seal RPC**（无并发负载）：
    python collect_bandwidth.py --gpu 0 --v6d-endpoint http://localhost:7890 --out profile.json
结果记为 profile 的 control_plane.save_completion_measured（约 1.3ms + 0.08ms/blk）。
这是控制面 RPC 的下界参考，**不用于仿真建模**。

### 配参指引

- DMA 两段（seg2 load/store）：用 collect_bandwidth.py 直接实测，物理量忠实。
- save_completion：保持日志推导值（从 build_connector_meta 到 mark_saved 的中位，
  190 样本回归 70+6/blk）。已由 9 组测试验证命中率偏差 <=0.68pp。
- 换环境重标定 save_completion：从该环境真实运行日志回归 mark_saved 减
  build_connector_meta(末store) 的中位/分桶拟合。不要用隔离 RPC 值替换。

## L1<->L2 搬运排队建模（TODO，当前未实现）

**现状**：`v6d_backend.py` 的 store/load deadline 为 `max(pending.get(rid,0), now + lat)` ——
每个请求的完成时刻是各自独立的 `now + lat`，请求之间**不串行累加**。即多个
并发的 store/load 在仿真中被视为**并行完成**，未建模"DMA 通道被占用、后来的
搬运需排队等待"的行为。

**真实系统**：async_swap 协程都跑在单一事件循环线程上（v6d_object_backend
_async_do_store），存在真实排队；高负载下 mark_saved 尾巴可达数百 ms 部分源于此。

**何时需要建模排队**：
- 当前 loopback + 大 prefill（每步 forward 数百 ms）场景不需要——排队被 step 粒度吸收。
- 超长请求（数千 block）+ 高并发场景需要——多个大 store 挤占同一通道。

**TODO 实现思路**（共享通道串行累加）：
```
channel_free_time = max(channel_free_time, now) + DMA(nblk)
deadline = channel_free_time + poll_granularity + rank_sync
```
即用一个 `channel_free_time` 游标表示 DMA 通道下次空闲时刻，新搬运排在其后。
load / store 可各自一个通道，或共享（取决于 PCIe 是否全双工——实测双向基本全双工，
可各自独立通道）。
