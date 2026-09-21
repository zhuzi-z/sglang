---
name: step-time-predictor-train
description: 基于压测采集的 rank0.schedule_batch.jsonl 训练 18 特征 GBR 步时延预测器（标签 iter_latency）并自动裁决标定 MTP sample_tokens 补偿参数（机理式优先、门控回退 hinge），产出 pkl 模型与 hisim_config 配置片段。Use when 用户提到"训练时间预测器"、"step time predictor 训练"、"压测数据训练预测器"、"标定 MTP/sample_tokens 补偿"、"iter_latency 模型"、"hisim 预测器训练"或给出 schedule_batch jsonl 数据集要求训练。
---

# step-time-predictor-train 管线

固化管线：预处理压测数据 → LODO 训练步时延预测器 → MTP 补偿自动裁决标定 → 产物落盘。
脚本逻辑不变；换数据集只需一份 yaml（用户最低输入只有两样，见下）。

## 数据契约（输入数据必须满足）

每行一个 step 的 jsonl（通常名为 `rank0.schedule_batch.jsonl`，hook 采集）：
- 必填：`forward_mode`、`request_infos[{extend_input_len, prefix_indices_len, rid}]` 或新格式 `requests[[rid, extend_input_len, prefix_indices_len, ...]]`（两种字段名/两种条目格式均兼容）、`iter_latency`（秒，缺失回退 `full_step_latency`）
- 仅 MTP 标定需要：`sample_tokens_latency`、`total_tokens`；缺失时该集自动退出 MTP 标定，训练不受影响
- `request_infos` 的 list/dict 两种格式均兼容；目录结构、路径（含中文）无要求
- 可选配套：同目录 `rank0.requests.jsonl`（分桶请求级过滤与命中审计用；schedule rid 与 requests rid 按末段 8-hex 匹配，也兼容全串精确匹配）

## 用户最低输入（只收集这两样）

1. **一句话任务说明**：模型/环境/数据背景，用于生成 task 名候选与报告头
2. **数据集清单**：每项 `{path, slowdown_factor}`
   - `path` 直指 jsonl 文件（必填）
   - `slowdown_factor` 为压测时配置的 mooncake-slowdown-factor 原值：**>1 越慢（负载越低）、=1 原速、<1 越快（负载越高）**；不知道填 `null`（会弱化速率分档检验并提示）
   - 注意：**不要用 sf 推断 GPU 繁忙/饱和**；sf 仅作分档检验的元数据，分档决策由逐集 base 离散度数据驱动
   - 可选 `bucket: [lo, hi]`（请求长度分桶，input_length 闭区间）：**用户指定分桶范围时必填**——管线会先按桶筛选再做后续分析（见下节）；也可顶层统一设置 `bucket`，被数据集级覆盖
   - 可选 `stages: [train]` / `[mtp]`，默认两者都参与（历史上有 MTP-only 集的场景才用）

## 分桶预筛选（指定长度桶时的第一动作，2026-09-20 固化）

**名义桶不可信**：实测过名义 0-32k 的压测数据实际含 9.27% >32k 请求（其中最长 934k），不筛直接训会让 24.4% 的步来自域外请求。因此用户指定分桶范围（如 "0-32k 分桶"）时，**先从压测数据筛出符合范围的请求，再做后续一切分析**：

1. **请求级过滤（主口径）**：步内任一请求 `input_length` 越出 [lo, hi] → 整步剔除。需要同目录配套 `rank0.requests.jsonl` 做 rid join（自动定位）；rid 匹配不上（如 warmup）按保留处理。
2. **步级回退**：缺 requests.jsonl 时降级为 `max_past ≤ hi` 过滤并出警告（多保留长请求的早期 chunk——它们与桶内 chunk 物理同分布，是可接受的近似，但必须如实告知）。
3. 过滤后各桶占比反而更均衡（超长请求的满 16k chunk 被剔出）；被筛掉的步**勿删原数据**——是全长/更大桶预测器与外推测试的资产。
4. 性能预期：配套 requests.jsonl 的行首正则解析约 60MB/s（单个 3.8GB 文件约 60s），stage 0 随大文件数量线性变慢，属正常现象不要中断。

## 压测审计判断（训练前必看，2026-09-20 固化）

管线 stage 0 自动执行并写进警告，agent 必须如实转达：

- **缓存命中工况审计（J2）**：按 token 加权统计 L1(local_kv_hit_len)/L2(ext_kv_hit_len) 命中率（该 hook 无独立远端字段，ext 即 v6d 层）。两者均 <0.1% → nocache 工况：可训练（已实证 hit-origin 时延不变性 ≤±3%、工作区 ≤0.6%——v6d 加载不在步关键路径）；但 **e2e 命中路径验证需命中富集资产另验**，本数据不承担。
- **覆盖度自检（J6）**：sum_extend 7 桶 `[0,1k)[1k,2k)[2k,4k)[4k,8k)[8k,12k)[12k,16k)[16k,∞)` 任一桶 <5% → 欠采警告；batch_size 5-8 <1% → 高并发组批欠采警告（验收须单独报 bs 5-8 桶 MAPE，必要时增广）；训练有效总量 <5 万 → 警告。
- **（小 extend, 大 past) 角落（J3）**：命中流量的命中步形状（如 e<4k & p>16k）在零命中数据中只占 ~0.1-0.2%（生产 hit 场景占 15-29%）。nocache 目标场景该角落与生产同构（尾块产生），不构成问题；hit 目标场景必须单独检查角落覆盖，必要时用命中富集数据增广。
- **同负载副本识别（J4，agent 侧判断）**：多实例若到达率/排队时间线完全一致且 rid 互斥 → 同一负载分布 × N pods，不是 N 个独立压力场景；此时 LODO 只验证 pod 泛化，应警告并建议补第二负载形态（不同 sf/trace）做最终 held-out。
- **sf 服务端推断（J5，bench 信息缺失时）**：warmup 识别 `regression_` 前缀或极小 input（如 ==3）的请求（可多轮，每轮 bench 启动各一轮）；主回放跨度 ÷ 数据集时间基线（hisim 惯例 60min）→ sf 点估计（例：484.3min/60min ≈ sf 8）；两轮 bench 间长静默（可达 1h）是 bench 端数据集加载，勿误判。推断值按候选与用户提供值核对后填入。

不要问用户训练/测试集怎么分——管线统一 LODO（每集轮流整集留出），无需任何划分先验。

## 输入不完整时的收集协议

原则：**能自动推导的不问；不能推导的列候选让用户确认，不擅自猜；允许降级的先跑并在警告中提示补全**。脚本本身非交互，所有收集动作由 agent 在执行前完成。

| 缺失项 | 收集方式 |
|---|---|
| 任务说明 | 反问一句即可；用户不愿给的，按数据集路径特征自动命名 |
| 数据集路径不完整/模糊 | **禁止猜测**。用 Glob 按 `**/rank0.schedule_batch.jsonl`（或用户给的目录片段）发现候选 → 列出候选清单（完整路径 + 行数）请用户确认/增删后再写 yaml |
| `tag` | 不问，自动取 path 父目录名；多集父目录同名（如都叫 engine0）时自动追加 `#2/#3` 后缀去重（LODO 折防互相覆盖），如需语义化命名可显式指定 |
| `slowdown_factor` | 允许 `null` 先跑；SUMMARY 警告会提示补填后可获得速率分档检验；补填后重跑即可（配置变化会自动落 `_rN` 新目录，不污染旧产物）。bench 信息缺失时可按上节 J5 服务端推断法给候选 |
| `bucket` | 用户在任务说明中指定了分桶范围（如 0-32k/32-128k）→ 必须填入；未指定不问。缺 requests.jsonl 会自动降级步级过滤并警告 |
| filter / gbr / mtp / eval / out | 不问，定稿默认值生效；仅用户主动提出才覆盖 |
| yaml 校验失败 | 脚本以字段级错误退出 → 将错误原文转达用户，只问缺失的那一个字段 |
| 数据缺 `sample_tokens_latency` | 不询问（属数据能力而非配置缺失）：训练照常，该集退出 MTP 标定并警告；全部缺失时提示“hisim_config 不得启用补偿” |
| task 名 | **唯一强制确认点**：生成候选 + 冲突检查结果（new/rerun/_rN）给用户确认后才执行 |

## 执行流程

1. 按上节收集协议补齐输入（路径模糊先 Glob 发现候选让用户确认）
2. 写 yaml（参照 `configs/` 两个模板），只需 `task` + `datasets`；其余字段全部有定稿默认值
3. 生成 task 名候选：`<说明关键词>-0-<量程>k-<yyyymmdd>`（slug 化为 `[a-z0-9-]`）；检查 `results/` 下冲突：依次比对 `slug`、`slug_r2`、`slug_r3`…，配置一致→原地重跑（复用最早匹配目录），全不一致→追加下一个 `_rN`。**先给用户确认 task 名再执行**
4. 运行：
   ```bash
   python3 .qoder/skills/step-time-predictor-train/scripts/run_pipeline.py --config <yaml>
   ```
5. 读 `results/<task>/report/SUMMARY.md`，向用户汇报：数据审计表（各集有效步数/V3 剔除/桶剔除/负载分位数）→ 缓存命中审计 → 覆盖度自检 → LODO 各折与全量指标 → MTP 裁决路径与参数 → hisim_config 片段 → 警告清单
6. 警告必须如实转达（V3 剔除率 >3%（历史正常范围 0~2%）、nocache 工况、覆盖欠采、bs 5-8 欠采、无原速锚点、缺 st 字段等），不要静默放过

## 定稿默认值（高级用户才需在 yaml 覆盖）

| 字段 | 默认值 | 含义 |
|---|---|---|
| `filter.forward_mode` | `[1]` | 只用 prefill/extend step |
| `filter.lat_max_s` | `30` | iter_latency 上界 |
| `filter.v3` | `{iter_gt_s: 1.0, sum_extend_lt: 4096}` | V3 污染步剔除（计算量小却耗时长）；健康数据剔除≈0；置 `null` 关闭 |
| `filter.st_max_ms` | `150` | sample_tokens_latency 上界 |
| `bucket` | `null` | 请求长度分桶 [lo, hi]（input_length 闭区间）；顶层或数据集级；指定后先做分桶预筛选 |
| `gbr` | `500树 depth6 lr0.05 sub0.8 rs42` | 项目定稿超参，勿轻改 |
| `eval.tt_buckets` | `[0,4096,8192,16384,32768,65536]` | 报告分 total_tokens 桶边界（只影响报告，不影响训练） |
| `mtp.form` | `auto` | `auto`（机理式优先门控回退 hinge）/ `mechanistic` / `hinge` |

## MTP 自动裁决逻辑（form=auto）

先机理式 nnls 拟合 `st_ms = base + a·Σext/1e3 + b·Σ(extᵢ·pastᵢ)/1e9`（b 项必须逐请求求和），再跑门控：
- G1 b 项中位贡献 < 0.5ms → FAIL（低于分辨率）
- G2 corr(Σext, Σ(ext·past)) > 0.9 → FAIL（共线不适定）
- G3 逐集自标定 b 变异系数 > 0.5 或出现 b=0 → FAIL（乱跳）
- G4 机理式 MAPE 不优于 hinge ≥0.5pp → FAIL
- G5 a 或 b 拟合为 0 → FAIL（退化）

任一 FAIL → 回退 hinge（断点网格扫描 + lstsq：`st = base + lo·tt + (hi-lo)·relu(tt-bp)`）。
裁决路径与逐门结果写入 `mtp/mtp_fit.json`，报告必须显式呈现。

**速率分档检验**：逐集自标定 base 的最大/最小 ≥1.15 时，按 sf 分组（≤1 原速组 / >1 慢放组）分别拟合，e2e 推荐原速档；若全是慢放集，统一参数并警告"无原速锚点，base 可能含慢放采集伪影"。

## 产物（`results/<task>/`）

```
config.resolved.yaml      # 生效配置快照（默认值填充后）
model/  18feat_<task>.pkl + .features.txt + train_metrics.json
mtp/    mtp_fit.json + mtp_fit.png + hisim_config.snippet.json
report/ train_log.txt + mtp_fit.log + SUMMARY.md
data_audit/ filter_report.json
```

模型为 `{"model": <GBR>, "features": <18名>}`，MLTimePredictor 可直接加载（`predictor.name='ml'`，`database_path` 指向 pkl）。
**模型标签是 iter_latency，hisim_config 必须配套 `hisim_config.snippet.json` 的补偿参数，否则 step 时间系统性偏低。**

## 回归对账基线（改动脚本后必须重跑两个模板验证）

```bash
python3 scripts/run_pipeline.py --config configs/qwen37max_l2_32k_l2new2.yaml
python3 scripts/run_pipeline.py --config configs/l20b_l1_0-32k_bench5.yaml
```

| 模板 | 训练基线 | MTP 基线 |
|---|---|---|
| qwen37max_l2_32k_l2new2 | 全量 R²=0.9970 MAPE=1.63%；LODO held-out 1.81~2.12% | auto→机理式 base=5.043 a=0.9813 b=6.6792（n=49258 MAPE 5.61%） |
| l20b_l1_0-32k_bench5 | LODO 各折 MAPE 个位数 | auto→hinge（G1/G3/G4 FAIL）：base≈14.6 lo≈0.08~0.09 hi≈1.01~1.03 bp∈平底盆地[8448,11264]含 9728（n=62613 MAPE≈6.64%）；断点盆地内 lo/hi 存在协变，逐参数与历史值（14.690/0.0775/1.0093/9728）的小幅差异属正常 |

依赖：numpy / scikit-learn / scipy / pyyaml（matplotlib 可选，缺失时仅跳过 png）。
