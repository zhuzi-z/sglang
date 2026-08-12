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
- `tools/bw_calib/collect_bandwidth.py` — DMA 段采集器
- `tools/bw_calib/bandwidth_profile.example.json` — 示例 profile

## 采集（换环境时）

在目标 GPU 节点运行：
```bash
python tools/bw_calib/collect_bandwidth.py \
    --gpu 0 --page-size 2146304 --num-layers 12 \
    --out bandwidth_profile.json
```
- `--page-size` / `--num-layers`：模型相关（TP 分片后的单 rank page size、层数）。Qwen3.5-122B = 2146304 / 12。
- 采集 seg2 load/store DMA + 并发模式（dual_dir 验全双工、dual_gpu 验 TP rank 争用）。
- **seg1 不在此采集**：需要两个 peer + 可用的 SRPC/RDMA 数据面（`collect_remote.py`，真实硬件）。
- save_completion 默认（70+6/blk）来自真实运行日志分析；可用 `--save-floor-ms/--save-per-blk-ms` 覆盖。

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

## save_completion 校准说明与"控制面 gap"（重要）

save_completion（sim 使用，70ms+6ms/blk）**不是** v6d 控制面 RPC 的耗时，而是"store 到跨节点可见"的**整体尾巴**（worker 侧异步 save 流水线 + 负载争用 + confirm）。

采集器新增 --v6d-endpoint 真实测量**隔离的 create+seal 控制面 RPC**（镜像连接器 batch_allocate/seal），记为 control_plane.save_completion_measured：

    python collect_bandwidth.py --gpu 0 --v6d-endpoint http://localhost:7890 --out profile.json

本环境实测（RTX PRO 6000）：create+seal = 1.31ms + 0.083ms/blk（约 2ms）。

**gap 结论**：隔离 RPC（约 2ms）与日志尾巴（70+6/blk，实测个别 158-3099ms）相差 50-100 倍。差值来自 worker 侧异步 save 排队与争用（引擎忙于其他请求 forward 时 save 排队），采集器在无负载下测不到这部分。

**配参指引**：
- DMA 两段（seg2 load/store）：用 collect_bandwidth.py 直接实测，忠实可靠。
- save_completion：保持日志推导值（它才能复现真实可见时序，已由命中率验证）；save_completion_measured（约 2ms）仅作控制面下界/健全性参考，不要用它替换 save_completion。
- 换环境重标定 save_completion：从该环境真实运行日志回归 mark_saved 减 V6D_swap_region，而非用隔离 RPC。
