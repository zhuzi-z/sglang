#!/usr/bin/env python3
"""step-time-predictor-train 共享库：数据契约 / 特征 / 过滤 / 指标 / 双拟合器 / 裁决门。

数据契约（每行一个 step 的 jsonl）：
    forward_mode, request_infos[{extend_input_len, prefix_indices_len, rid}],
    iter_latency (s); 可选 sample_tokens_latency (s), total_tokens, full_step_latency。
特征：18 维，与 sglang_simulator MLTimePredictor.FEATURE_NAMES 严格一致。
标签：iter_latency（RPC-1 主模型前向）；sample_tokens（RPC-2）由补偿函数独立叠加。
"""
import json
import math
import os
import re

import numpy as np
from scipy.optimize import nnls
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

FEATURE_NAMES = [
    "batch_size", "sum_extend", "max_extend", "min_extend",
    "sum_past", "max_past", "min_past",
    "sum_extend_x_past", "sum_extend_squared", "sum_past_squared",
    "sum_attn_flops", "sum_extend_x_max_past",
    "log1p_sum_past", "log1p_sum_attn_flops",
    "batch_size_x_sum_extend", "max_past_minus_min_past",
    "is_decode", "is_prefill",
]

# 定稿默认值（l2_new2 / bench5 两条历史线沉淀；用户 yaml 仅需 datasets）
DEFAULTS = {
    "filter": {
        "forward_mode": [1],
        "lat_max_s": 30.0,
        # V3 污染步口径：iter 大但计算量小的死锁/停顿步；健康数据剔除≈0。置 null 关闭。
        "v3": {"iter_gt_s": 1.0, "sum_extend_lt": 4096},
        "st_max_ms": 150.0,
    },
    "gbr": {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "random_state": 42},
    "eval": {"tt_buckets": [0, 4096, 8192, 16384, 32768, 65536]},
    "mtp": {
        "form": "auto",            # auto | mechanistic | hinge
        "gate_b_contrib_ms": 0.5,    # G1: b 项中位贡献低于该值 → 低于分辨率
        "gate_collinearity": 0.9,    # G2: corr(se, ep) 高于该值 → 不适定
        "gate_b_cv": 0.5,            # G3: 逐集 b 变异系数高于该值 → 不稳定
        "gate_mape_delta_pp": 0.5,   # G4: 机理式须比 hinge 优至少该幅度
        "tier_base_ratio": 1.15,     # 逐集 base 最大/最小超过该值 → 检查速率分档
        "hinge_bp_grid": [512, 16384, 256],  # 断点扫描 start/stop/step
    },
}


def deep_merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def iter_req_infos(ri):
    if isinstance(ri, dict):
        return list(ri.values())
    return list(ri or [])


def parse_req_entry(r):
    """返回 (extend, past, rid)；兼容两种 hook 格式：

    - dict: {"extend_input_len", "prefix_indices_len", "rid"}（旧 request_infos）
    - list: [rid, extend_input_len, prefix_indices_len, ...]（新 requests 数组格式）
    """
    if isinstance(r, dict):
        return int(r["extend_input_len"]), int(r["prefix_indices_len"]), r.get("rid")
    return int(r[1]), int(r[2]), (r[0] if len(r) > 0 and isinstance(r[0], str) else None)


def get_req_infos(b):
    """兼容 request_infos（旧）与 requests（新）两种字段名。"""
    ri = b.get("request_infos")
    if ri is None:
        ri = b.get("requests")
    return iter_req_infos(ri)


# ---------------------------------------------------------------- 分桶预筛选与命中审计

_RX_REQ = {
    "rid": re.compile(rb'"rid": "([^"]+)"'),
    "input_length": re.compile(rb'"input_length": (\d+)'),
    "local": re.compile(rb'"local_kv_hit_len": (\d+)'),
    "ext": re.compile(rb'"ext_kv_hit_len": (\d+)'),
}


def find_requests_file(schedule_path):
    """定位与 schedule_batch 配套的 requests.jsonl（请求级桶过滤/命中审计用）。"""
    d = os.path.dirname(schedule_path)
    base = os.path.basename(schedule_path)
    cands = [os.path.join(d, "rank0.requests.jsonl"),
             os.path.join(d, "TP0.requests.jsonl"),
             os.path.join(d, base.replace("schedule_batch", "requests"))]
    for c in cands:
        if os.path.isfile(c) and os.path.getsize(c) > 0:
            return c
    return None


def load_request_index(req_path, head_bytes=700):
    """流式解析 requests.jsonl（只扫行首元数据），返回 (rid->input_length 索引, 命中审计)。

    索引同时注册全串 rid 与末段（schedule_batch 的 rid 常为 requests rid 的末段，
    如 '502-d78425ed' ↔ '3842695b-...-ed5e9ee3a502-d78425ed'）；8-hex 末段在
    10^5 量级请求下碰撞期望 <1，可接受。
    """
    il = {}
    n = sum_il = sum_l = sum_e = n_l = n_e = 0
    with open(req_path, "rb") as f:
        for raw in f:
            h = raw[:head_bytes]
            m = _RX_REQ["rid"].search(h)
            if not m:
                continue
            rid = m.group(1).decode("utf-8", "ignore")
            mi = _RX_REQ["input_length"].search(h)
            if not mi:
                continue
            v = int(mi.group(1))
            il.setdefault(rid, v)
            suf = rid.split("-")[-1]
            if len(suf) >= 6:  # 仅注册类 hex 末段，避免 'cmpl-...-0' 短尾误配
                il.setdefault(suf, v)
            n += 1
            sum_il += v
            ml = _RX_REQ["local"].search(h)
            me = _RX_REQ["ext"].search(h)
            lv = int(ml.group(1)) if ml else 0
            ev = int(me.group(1)) if me else 0
            sum_l += lv
            sum_e += ev
            n_l += lv > 0
            n_e += ev > 0
    hit = {"n_requests": n,
           "local_tok_ratio": sum_l / max(sum_il, 1),
           "ext_tok_ratio": sum_e / max(sum_il, 1),
           "local_req_ratio": n_l / max(n, 1),
           "ext_req_ratio": n_e / max(n, 1)}
    return il, hit


SE_BUCKET_EDGES = [0, 1024, 2048, 4096, 8192, 12288, 16384]
SE_BUCKET_LABELS = ["[0,1k)", "[1k,2k)", "[2k,4k)", "[4k,8k)",
                    "[8k,12k)", "[12k,16k)", "[16k,inf)"]


def coverage_report(se, bs):
    """覆盖度自检（数据需求手册 §6 口径）：sum_extend 7 桶 + batch_size 4 桶占比。"""
    n = max(len(se), 1)
    rows = []
    edges = list(SE_BUCKET_EDGES) + [np.inf]
    for lo, hi, lab in zip(edges[:-1], edges[1:], SE_BUCKET_LABELS):
        m = (se >= lo) & (se < hi)
        rows.append({"bucket": lab, "n": int(m.sum()), "pct": 100.0 * m.sum() / n})
    bs_rows = []
    for lab, m in [("bs==1", bs == 1), ("bs 2-4", (bs >= 2) & (bs <= 4)),
                   ("bs 5-8", (bs >= 5) & (bs <= 8)), ("bs>8", bs > 8)]:
        bs_rows.append({"bucket": lab, "n": int(m.sum()), "pct": 100.0 * m.sum() / n})
    return {"sum_extend": rows, "batch_size": bs_rows}


def extract_features(extends, pasts):
    """18 维特征，与 MLTimePredictor._extract_features() 逐行对齐。"""
    bs = len(extends)
    sum_e = sum(extends)
    sum_p = sum(pasts)
    sum_ep = sum(e * p for e, p in zip(extends, pasts))
    sum_e2 = sum(e * e for e in extends)
    sum_p2 = sum(p * p for p in pasts)
    sum_attn = sum(e * (p + e / 2) for e, p in zip(extends, pasts))
    return [
        bs, sum_e, max(extends), min(extends),
        sum_p, max(pasts), min(pasts),
        sum_ep, sum_e2, sum_p2, sum_attn,
        sum_e * max(pasts), math.log1p(sum_p), math.log1p(sum_attn),
        bs * sum_e, max(pasts) - min(pasts),
        int(all(e == 1 for e in extends)),
        int(any(e > 1 for e in extends)),
    ]


def load_step_file(path, filt, bucket=None, req_index=None):
    """加载单个 rank0.schedule_batch.jsonl，返回训练/MTP 数组与过滤统计。

    bucket: 可选 (lo, hi) 请求长度分桶（input_length 闭区间）。指定后先做分桶
    预筛选再进入常规过滤：
      - req_index 非空（请求级，推荐）：步内任一请求 input_length 越界 → 整步剔除；
        查不到 input_length 的请求（如 warmup）按保留处理。
      - req_index 为空（步级回退）：max_past > hi → 整步剔除。
    """
    fm_set = set(filt["forward_mode"])
    lat_max = float(filt["lat_max_s"])
    st_max = float(filt["st_max_ms"])
    v3 = filt.get("v3")
    st = dict(n_total=0, skip_mode=0, skip_empty=0, skip_bad=0,
              skip_lat=0, skip_v3=0, skip_bucket=0, v3_examples=[])
    X, y, tt, se, sp, ep, stm = [], [], [], [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            st["n_total"] += 1
            b = json.loads(line)
            if b.get("forward_mode") not in fm_set:
                st["skip_mode"] += 1
                continue
            ri = get_req_infos(b)
            if not ri:
                st["skip_empty"] += 1
                continue
            li = b.get("iter_latency")
            if li is None:
                li = b.get("full_step_latency")
            if li is None or li <= 0 or li > lat_max:
                st["skip_lat"] += 1
                continue
            try:
                parsed = [parse_req_entry(r) for r in ri]
            except (KeyError, TypeError, ValueError, IndexError):
                st["skip_bad"] += 1
                continue
            extends = [p[0] for p in parsed]
            pasts = [p[1] for p in parsed]
            if bucket is not None:
                lo, hi = bucket
                if req_index is not None:
                    keep = True
                    for r in parsed:
                        rid = r[2]
                        il = req_index.get(rid) if rid else None
                        if il is None and rid:  # 末段回退匹配
                            il = req_index.get(rid.split("-")[-1])
                        if il is not None and not (lo <= il <= hi):
                            keep = False
                            break
                else:  # 步级回退：max_past 上界
                    keep = max(pasts) <= hi
                if not keep:
                    st["skip_bucket"] += 1
                    continue
            s_e = sum(extends)
            if v3 and li > float(v3["iter_gt_s"]) and s_e < float(v3["sum_extend_lt"]):
                st["skip_v3"] += 1
                if len(st["v3_examples"]) < 20:
                    st["v3_examples"].append(
                        {"iter_latency_s": round(li, 3), "sum_extend": s_e})
                continue
            s_p = sum(pasts)
            s_ep = sum(e * p for e, p in zip(extends, pasts))
            # tt = 本步调度的总 token 数（= Σext；MTP 补偿运行时自变量语义）。
            # 新格式数据无 total_tokens 字段时必须回退 s_e，绝不能含 past——
            # 否则 hinge 断点/斜率会标定在错误的 x 轴上（2026-09-21 l2_new3 实证）。
            t = b.get("total_tokens") or s_e
            sv = b.get("sample_tokens_latency")
            sv_ms = sv * 1e3 if sv else float("nan")
            if not (0 < sv_ms <= st_max):
                sv_ms = float("nan")
            X.append(extract_features(extends, pasts))
            y.append(li)
            tt.append(int(t))
            se.append(s_e)
            sp.append(s_p)
            ep.append(s_ep)
            stm.append(sv_ms)
    d = dict(
        X=np.array(X, dtype=np.float64), y=np.array(y, dtype=np.float64),
        tt=np.array(tt, dtype=np.float64), se=np.array(se, dtype=np.float64),
        sp=np.array(sp, dtype=np.float64), ep=np.array(ep, dtype=np.float64),
        st=np.array(stm, dtype=np.float64), stats=st,
    )
    d["mtp_mask"] = np.isfinite(d["st"]) & (d["tt"] > 0)
    return d


def metrics(y_true, y_pred, mae_scale=1000.0):
    """mae_scale: 训练标签为秒(×1000→ms)；MTP 的 st 本身已是 ms(×1)。"""
    pe = (y_pred - y_true) / np.maximum(y_true, 1e-6)
    return {"n": int(len(y_true)),
            "R2": float(r2_score(y_true, y_pred)),
            "MAE_ms": float(mean_absolute_error(y_true, y_pred) * mae_scale),
            "MAPE": float(np.abs(pe).mean() * 100),
            "MPE": float(pe.mean() * 100),
            "sum_diff_pct": float((y_pred.sum() / max(y_true.sum(), 1e-9) - 1) * 100)}


def fmt_metrics(tag, m):
    return (f"  {tag:<24} n={m['n']:<7} R2={m['R2']:7.4f} MAE={m['MAE_ms']:7.2f}ms "
            f"MAPE={m['MAPE']:5.2f}% MPE={m['MPE']:+5.2f}% 总量差={m['sum_diff_pct']:+5.2f}%")


def gbr(params):
    return GradientBoostingRegressor(**params)


def bucket_rows(y_true, y_pred, tt, edges):
    rows = []
    for lo, hi in zip(edges[:-1], list(edges[1:]) + [np.inf]):
        m = (tt >= lo) & (tt < hi)
        if m.sum() == 0:
            continue
        pe = (y_pred[m] - y_true[m]) / y_true[m]
        rows.append({"lo": int(lo), "hi": None if np.isinf(hi) else int(hi),
                     "n": int(m.sum()),
                     "MAPE": float(np.abs(pe).mean() * 100),
                     "MPE": float(pe.mean() * 100)})
    return rows


# ---------------------------------------------------------------- MTP 拟合

def mechanistic_fit(st_ms, se, ep):
    """机理式三参数 nnls：st_ms = base + a*se/1e3 + b*ep/1e9（系数全非负）。"""
    X3 = np.stack([np.ones(len(st_ms)), se / 1e3, ep / 1e9], axis=1)
    coef, _ = nnls(X3, st_ms)
    yhat = X3 @ coef
    return {"base": float(coef[0]), "a": float(coef[1]), "b": float(coef[2]),
            "yhat": yhat, "metrics": metrics(st_ms, yhat, mae_scale=1.0)}


def hinge_fit(tt, st_ms, bp_grid):
    """hinge 四参数：st_ms = base + lo*tt + (hi-lo)*relu(tt-bp)；断点网格扫描 + lstsq。

    lo/hi 内部以 ms/token 拟合，返回时换算为 us/token。
    """
    start, stop, step = bp_grid
    best = None
    scan = []
    tt_s = tt / 1e3  # ms/token = us/token 数值相同；tt/1e3 使 lo 数值即 us/tok 的 ms 表达
    for bp in range(int(start), int(stop) + 1, int(step)):
        relu = np.maximum(0.0, tt - bp) / 1e3
        Xh = np.stack([np.ones(len(tt)), tt_s, relu], axis=1)
        coef, *_ = np.linalg.lstsq(Xh, st_ms, rcond=None)
        yhat = Xh @ coef
        sse = float(((st_ms - yhat) ** 2).sum())
        m = metrics(st_ms, yhat, mae_scale=1.0)
        scan.append({"bp": int(bp), "sse": sse, "MAPE": m["MAPE"]})
        if best is None or sse < best["sse"]:
            best = {"bp": int(bp), "base": float(coef[0]),
                    "lo": float(coef[1]), "hi": float(coef[1] + coef[2]),
                    "sse": sse, "yhat": yhat, "metrics": m}
    min_sse = best["sse"]
    plateau = [s["bp"] for s in scan if s["sse"] <= min_sse * 1.005]
    best["plateau"] = [int(min(plateau)), int(max(plateau))]
    best["scan_top"] = sorted(scan, key=lambda s: s["sse"])[:5]
    return best


def mtp_gates(mtp_sets, mech, hinge, cfg):
    """机理式合理性门 G1-G5（全部 PASS 才允许 auto 选机理式）。"""
    st = np.concatenate([s["st"][s["mtp_mask"]] for s in mtp_sets])
    se = np.concatenate([s["se"][s["mtp_mask"]] for s in mtp_sets])
    ep = np.concatenate([s["ep"][s["mtp_mask"]] for s in mtp_sets])
    gates = []
    contrib = float(np.median(mech["b"] * ep / 1e9)) if len(ep) else 0.0
    gates.append({"gate": "G1_b项量级", "pass": contrib >= cfg["gate_b_contrib_ms"],
                  "detail": f"b·Σ(ext·past) 中位贡献 {contrib:.3f}ms "
                            f"(阈值 {cfg['gate_b_contrib_ms']}ms)"})
    corr = float(np.corrcoef(se, ep)[0, 1]) if len(se) > 2 else 1.0
    gates.append({"gate": "G2_共线性", "pass": corr <= cfg["gate_collinearity"],
                  "detail": f"corr(Σext, Σ(ext·past)) = {corr:.4f} "
                            f"(阈值 {cfg['gate_collinearity']})"})
    bs = []
    for s in mtp_sets:
        mk = s["mtp_mask"]
        if mk.sum() < 100:
            continue
        r = mechanistic_fit(s["st"][mk], s["se"][mk], s["ep"][mk])
        bs.append(r["b"])
    if bs:
        cv = float(np.std(bs) / np.mean(bs)) if np.mean(bs) > 0 else float("inf")
        stable = all(b > 0 for b in bs) and cv <= cfg["gate_b_cv"]
        detail = f"逐集 b={['%.3f' % b for b in bs]} cv={cv:.3f} (阈值 {cfg['gate_b_cv']})"
    else:
        stable, detail = False, "有效数据集不足，无法检验逐集 b 稳定性"
    gates.append({"gate": "G3_b稳定性", "pass": stable, "detail": detail})
    delta = hinge["metrics"]["MAPE"] - mech["metrics"]["MAPE"]
    gates.append({"gate": "G4_精度对比", "pass": delta >= cfg["gate_mape_delta_pp"],
                  "detail": f"机理式 MAPE {mech['metrics']['MAPE']:.2f}% vs "
                            f"hinge {hinge['metrics']['MAPE']:.2f}% "
                            f"(需优 ≥{cfg['gate_mape_delta_pp']}pp)"})
    gates.append({"gate": "G5_系数退化", "pass": mech["a"] > 0 and mech["b"] > 0,
                  "detail": f"a={mech['a']:.4f} b={mech['b']:.4f} (均需 >0)"})
    return gates


def decide_form(gates, requested):
    if requested in ("mechanistic", "hinge"):
        return requested, f"用户显式指定 form={requested}（门结果仅供参考）"
    failed = [g["gate"] for g in gates if not g["pass"]]
    if not failed:
        return "mechanistic", "G1-G5 全部 PASS，选用机理式"
    return "hinge", f"以下门未通过：{', '.join(failed)} → 回退 hinge"


def _fit_form(form, st, se, ep, tt, cfg):
    if form == "mechanistic":
        r = mechanistic_fit(st, se, ep)
        return {"base": r["base"], "a": r["a"], "b": r["b"], "metrics": r["metrics"]}
    r = hinge_fit(tt, st, cfg["hinge_bp_grid"])
    return {"base": r["base"], "lo": r["lo"], "hi": r["hi"], "bp": r["bp"],
            "metrics": r["metrics"]}


def tiering_check(mtp_sets, tags, sfs, form, cfg):
    """逐集自标定 base → 离散度小则统一参数；出现系统性电平差则按 sf 分档。

    注意：sf 仅作分组元数据，不用来推断 GPU 繁忙度；e2e 推荐档锚定最低 sf 组。
    """
    bases, per_set = [], {}
    for s, tg in zip(mtp_sets, tags):
        mk = s["mtp_mask"]
        if mk.sum() < 100:
            continue
        r = _fit_form(form, s["st"][mk], s["se"][mk], s["ep"][mk], s["tt"][mk], cfg)
        per_set[tg] = {k: v for k, v in r.items() if k != "metrics"}
        per_set[tg]["metrics"] = r["metrics"]
        bases.append((tg, r["base"]))
    out = {"per_dataset": per_set}
    if len(bases) < 2:
        out.update(mode="unified", note="有效数据集不足，直接统一拟合")
        return out
    vals = np.array([b for _, b in bases])
    ratio = float(vals.max() / max(vals.min(), 1e-9))
    out["base_spread"] = {"min": float(vals.min()), "max": float(vals.max()),
                          "ratio": ratio, "threshold": cfg["tier_base_ratio"]}
    if ratio < cfg["tier_base_ratio"]:
        out.update(mode="unified", note=f"逐集 base 离散度 {ratio:.3f} < 阈值，统一参数")
        return out
    native = [i for i, sf in enumerate(sfs) if sf is not None and sf <= 1.0]
    slow = [i for i, sf in enumerate(sfs) if sf is not None and sf > 1.0]
    if native and slow:
        out.update(mode="tiered",
                   note="base 电平差超阈值且原速/慢放集齐全，按速率分档；e2e 用原速档")
        return out
    if slow:
        out.update(mode="unified", warning="no_native_anchor",
                   note="base 电平差超阈值但全部为慢放(sf>1)集：统一参数；e2e 使用需知悉 "
                        "base 可能含慢放采集伪影")
    elif native:
        out.update(mode="unified",
                   note="base 电平差超阈值，但全部数据集均为原速(sf<=1)语义：按统一参数输出")
    else:
        out.update(mode="unified", warning="sf_metadata_missing",
                   note="base 离散度超阈值但 slowdown_factor 元数据缺失，无法分组检验；"
                        "统一参数。若关注速率分档请补填 slowdown_factor")
    return out
