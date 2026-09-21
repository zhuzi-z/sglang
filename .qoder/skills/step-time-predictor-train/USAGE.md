# step-time-predictor-train 调用规范（用户版）

> 用途：基于压测采集的 `rank0.schedule_batch.jsonl`，一键训练 18 特征 GBR 步时延预测器（标签 `iter_latency`）并自动裁决标定 MTP `sample_tokens` 补偿参数，产出 hisim 可直接加载的 pkl 与 hisim_config 片段。
> 最近更新：2026-09-21（分桶预筛选 / 命中审计 / 覆盖自检 / tag 自动去重 / _rN 配置匹配复用）

---

## 1. 如何发起调用

自然语言说明任务即可，触发词示例："训练时间预测器"、"step time predictor 训练"、"标定 MTP/sample_tokens 补偿"、"hisim 预测器训练"，或直接 `/step-time-predictor-train`。

**你只需要提供两样东西**：

1. **一句话任务说明**：模型/环境/数据背景（如 "qwen3.7-max L2 0-32k分桶 新一轮压测数据"）。
2. **数据集清单**：每个数据集一个 `rank0.schedule_batch.jsonl` 路径（目录路径模糊时 agent 会先 Glob 发现候选让你确认，不会猜）。

其余全部由 agent 按协议补齐并在执行前向你确认 **task 名**（唯一强制确认点）。

## 2. 数据集字段（yaml，全部有默认值，除 path 外均可省）

| 字段 | 必填 | 说明 |
|---|---|---|
| `path` | ✓ | 直指 jsonl 文件 |
| `tag` |  | 缺省取父目录名；多集同名自动追加 `#2/#3` 去重 |
| `slowdown_factor` |  | 压测时 mooncake-slowdown-factor 原值（>1 慢/=1 原速/<1 快）；不知道填 `null`（弱化速率分档检验，给警告）；**仅作元数据，不用来推断 GPU 繁忙** |
| `bucket` | 条件必填 | `[lo, hi]` 请求长度分桶（input_length 闭区间）。**你在任务说明里指定了分桶范围（如 0-32k / 32k++）时必填**——管线先按桶筛选再做后续一切分析。可顶层统一设置 |
| `stages` |  | 缺省 `[train, mtp]`；仅参与 MTP 标定的集写 `[mtp]` |

示例（32k++ 四实例）：

```yaml
task: qwen37max-l2-32kplus-l2new4-20260921
description: qwen3.7-max L2 32k++分桶 新一轮压测（四实例 命中富集）训练 + MTP 标定
bucket: [32768, 1048576]
datasets:
  - path: /workspace/l20b/l2_new4/inst1/engine0/rank0.schedule_batch.jsonl
    tag: inst1
    slowdown_factor: null
  - path: /workspace/l20b/l2_new4/inst2/engine0/rank0.schedule_batch.jsonl
    tag: inst2
    slowdown_factor: null
  # ... inst3 / inst4 同
```

## 3. 管线自动做的事（你需要知道的行为）

1. **分桶预筛选**（指定 bucket 时）：请求级过滤（步内任一请求 input_length 越界 → 整步剔除，需同目录 `rank0.requests.jsonl` 做 rid join；缺失时降级步级 `max_past ≤ hi` 并警告）。**名义桶不可信**：实测过名义 0-32k 数据实际含 9.27% 超长请求。
2. **数据审计**：命中工况（L1/L2 token 加权命中率；≈0 报 nocache 工况）、覆盖度自检（sum_extend 7 桶 ≥5%、bs 5-8 ≥1%、总量 ≥5 万，不足给警告）、V3 死锁步剔除（>3% 给警告）。
3. **训练**：18 特征 GBR（500 树/深 6/lr 0.05/sub 0.8），统一 **LODO**（每集轮流整集留出，无需你划分训练/测试集）+ 全量终训。
4. **MTP 裁决**（form=auto）：机理式优先，G1-G5 任一门 FAIL 回退 hinge；经验规律：0-32k/L1 → hinge，32k+/L2 → 机理式。速率分档由逐集 base 离散度 ≥1.15 驱动（sf 仅作分组元数据）。
5. **task 名冲突处理**：依次比对 `slug`/`slug_r2`/`slug_r3`…的配置快照，一致→原地重跑（复用最早匹配目录），全不一致→追加下一个 `_rN`，不污染旧产物。

**警告必须如实转达**（nocache 工况、覆盖欠采、bs 5-8 欠采、sf 缺失、V3 高剔除、无原速锚点等），不是错误而是采集决策依据。

## 4. 产物（`results/<task>/`）

```
config.resolved.yaml      # 生效配置快照
model/  18feat_<task>.pkl + .features.txt + train_metrics.json
mtp/    mtp_fit.json + mtp_fit.png + hisim_config.snippet.json
report/ train_log.txt + mtp_fit.log + SUMMARY.md
data_audit/ filter_report.json
```

接线（**模型与补偿必须配套**，模型标签是 iter_latency，不配补偿 step 时间系统性偏低）：

```json
"predictor": {"name": "ml", "database_path": "<pkl 路径>", "latency_scale": 1.0},
"extra_config": { ...hisim_config.snippet.json 内容... }
```

## 5. 验收口径速查

| 指标 | 优秀 | 可用 |
|---|---|---|
| 整体 held-out step MAPE | ≤3% | ≤5% |
| 各 token 桶 MAPE | ≤5% | ≤8%（小 token 桶有噪声地板，单独看） |
| bias（总量差） | ≤1% | ≤2% |
| 逐 case MAPE 极差 | ≤2pp | ≤4pp（极差大 = 存在 off-distribution pod） |

## 6. 注意事项（已知行为）

- **同负载副本**：多实例同 trace 并行打流只算 1 个负载分布 × N pods；LODO 验证 pod 泛化，最终 held-out 需补第二负载形态（不同 sf/trace）。
- **nocache 数据可训练**（hit-origin 时延不变性 ≤±3% 已实证），但 e2e 命中路径验证需命中富集资产另验。
- 配套 requests.jsonl 大文件解析约 60MB/s（3.8GB ≈ 60s/个），stage 0 变慢属正常。
- **改脚本后必须重跑两个回归模板对账**（`configs/qwen37max_l2_32k_l2new2.yaml` 全量 R²=0.9970 MAPE=1.63% + 机理式 5.043/0.9813/6.6792；`configs/l20b_l1_0-32k_bench5.yaml` auto→hinge bp∈[8448,11264] MAPE≈6.65%）。
- 依赖：numpy / scikit-learn / scipy / pyyaml（matplotlib 可选，缺失仅跳过 png）。

## 7. 参考实现（本工作区已验证的两次真实调用）

- `configs/qwen37max_l2_0-32k_l2new3.yaml` → results/qwen37max-l2-0-32k-l2new3-20260920_r2（0-32k nocache：全量 R²=0.9298 MAPE=5.13%，hinge 5.8327/0.8897/1.1165/8704）
- `configs/qwen37max_l2_32kplus_l2new4.yaml` → results/qwen37max-l2-32kplus-l2new4-20260921（32k++ 命中富集：全量 R²=0.9651 MAPE=1.59%，机理式 5.7481/0.9637/5.5706）
