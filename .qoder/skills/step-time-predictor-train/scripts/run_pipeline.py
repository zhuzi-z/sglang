#!/usr/bin/env python3
"""step-time-predictor-train 主编排：数据审计 → LODO 训练 → MTP 裁决标定 → 产物落盘。

用法：
    python3 run_pipeline.py --config <config.yaml> [--task <name>] [--results-root <dir>]

产物目录（默认 <skill>/results/<task_slug>/）：
    config.resolved.yaml / model/ / mtp/ / report/ / data_audit/
"""
import argparse
import datetime
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_pipeline as L

SKILL_ROOT = Path(__file__).resolve().parents[1]


class StageLog:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg)
        self.f.write(str(msg) + "\n")

    def close(self):
        self.f.close()


def slugify(s):
    s = re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")
    return s or "task"


def validate_config(raw):
    if not isinstance(raw, dict):
        sys.exit("config 必须是 yaml mapping")
    ds = raw.get("datasets")
    if not isinstance(ds, list) or len(ds) < 1:
        sys.exit("datasets 必填且至少 1 集")
    meta = []
    for i, d in enumerate(ds):
        if not isinstance(d, dict) or not d.get("path"):
            sys.exit(f"datasets[{i}] 缺少必填字段 path")
        p = Path(d["path"])
        if not p.is_file():
            sys.exit(f"datasets[{i}] 文件不存在: {p}")
        tag = d.get("tag") or p.parent.name
        sf = d.get("slowdown_factor", None)
        if sf is not None:
            sf = float(sf)
        stages = d.get("stages") or ["train", "mtp"]
        bad = set(stages) - {"train", "mtp"}
        if bad:
            sys.exit(f"datasets[{i}] stages 含非法值: {bad}（允许 train/mtp）")
        bucket = d.get("bucket", raw.get("bucket"))
        if bucket is not None:
            if not (isinstance(bucket, (list, tuple)) and len(bucket) == 2):
                sys.exit(f"datasets[{i}] bucket 必须是 [lo, hi] 两元素（input_length 闭区间）")
            bucket = [float(bucket[0]), float(bucket[1])]
            if bucket[0] < 0 or bucket[1] <= bucket[0]:
                sys.exit(f"datasets[{i}] bucket 区间非法: {bucket}")
        meta.append({"tag": str(tag), "path": str(p), "slowdown_factor": sf,
                     "stages": list(stages), "bucket": bucket})
    cfg = L.deep_merge(L.DEFAULTS, {k: v for k, v in raw.items()
                                    if k in ("filter", "gbr", "eval", "mtp")})
    if cfg["mtp"]["form"] not in ("auto", "mechanistic", "hinge"):
        sys.exit("mtp.form 仅允许 auto|mechanistic|hinge")
    n_train = sum(1 for m in meta if "train" in m["stages"])
    if n_train < 2:
        sys.exit("参与 train 的数据集至少 2 集（LODO 需要）")
    # tag 撞名自动去重：缺省 tag 取 path 父目录名，多集同父目录（如都叫 engine0）
    # 时 LODO 折/报告会互相覆盖；首个保留原名，后续追加 #2/#3 后缀。
    seen = {}
    for m in meta:
        t = m["tag"]
        if t in seen:
            seen[t] += 1
            m["tag"] = f"{t}#{seen[t]}"
            print(f"[tag-dedup] 数据集 tag 撞名: '{t}' → '{m['tag']}'"
                  f"（{m['path']}；如需语义化命名请在 datasets[].tag 显式指定）")
        else:
            seen[t] = 1
    return cfg, meta, raw.get("task"), raw.get("description", "")


def _cfg_compare_key(datasets, cfg):
    return json.dumps({"datasets": datasets, "filter": cfg["filter"],
                       "gbr": cfg["gbr"], "mtp": cfg["mtp"], "eval": cfg["eval"]},
                      sort_keys=True, ensure_ascii=False, default=str)


def resolve_task(raw_task, desc, sets, meta, cfg, results_root):
    """task 名：显式 > 自动生成；冲突时配置一致则复用、否则追加 _rN。"""
    explicit = raw_task
    if explicit:
        slug = slugify(explicit)
    else:
        words = re.findall(r"[A-Za-z][A-Za-z0-9]*", desc)[:4]
        prefix = "-".join(w.lower() for w in words) or "task"
        tt_all = np.concatenate([s["tt"] for s in sets]) if sets else np.array([0])
        hi = max(32, int(np.ceil(np.percentile(tt_all, 99) / 32768) * 32))
        slug = slugify(f"{prefix}-0-{hi}k-{datetime.date.now():%Y%m%d}")
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    # 依次检查 slug, slug_r2, slug_r3, ...：配置一致→原地重跑；全不一致→下一个 _rN
    n = 1
    while True:
        cand = slug if n == 1 else f"{slug}_r{n}"
        cdir = root / cand
        if not cdir.is_dir():
            return cand, ("new" if n == 1 else f"conflict_rename_from:{slug}")
        old_cfg = cdir / "config.resolved.yaml"
        same = False
        if old_cfg.is_file():
            try:
                old = yaml.safe_load(open(old_cfg, encoding="utf-8")) or {}
                same = _cfg_compare_key(old.get("datasets"), old) == \
                    _cfg_compare_key(meta, cfg)
            except Exception:
                same = False
        if same:
            return cand, ("rerun_same_config" if n == 1
                          else f"rerun_same_config:{cand}")
        n += 1


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default=None, help="覆盖 yaml 中的 task 名")
    ap.add_argument("--results-root", default=str(SKILL_ROOT / "results"))
    args = ap.parse_args()

    raw = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.task:
        raw = dict(raw or {}, task=args.task)
    cfg, meta, raw_task, desc = validate_config(raw)

    # ---------------- 0 数据装载与审计 ----------------
    print("=" * 100)
    print("阶段 0  数据装载与审计")
    print("=" * 100)
    sets = []
    audit, warnings = [], []
    v3_on = bool(cfg["filter"].get("v3"))
    for m in meta:
        # ---- 分桶预筛选与命中审计（先于常规过滤；判断 J1/J2 固化） ----
        req_index, hit_info, bucket_mode = None, None, None
        req_file = L.find_requests_file(m["path"])
        if m["bucket"] is not None:
            if req_file:
                req_index, hit_info = L.load_request_index(req_file)
                bucket_mode = "request"
            else:
                bucket_mode = "max_past"
                warnings.append(
                    f"[{m['tag']}] 指定 bucket={m['bucket']} 但未找到配套 requests.jsonl："
                    f"降级为步级 max_past<={int(m['bucket'][1])} 过滤"
                    f"（多保留长请求早期 chunk，物理上与桶内 chunk 同分布，可接受但需知悉）")
        elif req_file:
            _, hit_info = L.load_request_index(req_file)  # 无桶过滤也做命中审计
        d = L.load_step_file(m["path"], cfg["filter"],
                             bucket=m["bucket"], req_index=req_index)
        sets.append(d)
        st = d["stats"]
        n_tr = len(d["y"])
        n_mtp = int(d["mtp_mask"].sum())
        v3_ratio = st["skip_v3"] / max(st["n_total"], 1) * 100
        audit.append({
            "tag": m["tag"], "path": m["path"],
            "slowdown_factor": m["slowdown_factor"], "stages": m["stages"],
            "bucket": m["bucket"], "bucket_mode": bucket_mode,
            "n_total": st["n_total"], "n_train": n_tr, "n_mtp": n_mtp,
            "skip_mode": st["skip_mode"], "skip_empty": st["skip_empty"],
            "skip_bad": st["skip_bad"], "skip_lat": st["skip_lat"],
            "skip_v3": st["skip_v3"], "skip_bucket": st["skip_bucket"],
            "v3_drop_pct": round(v3_ratio, 4),
            "v3_examples": st["v3_examples"], "hit": hit_info,
            "tt_p50": pct(d["tt"], 50), "tt_p90": pct(d["tt"], 90),
            "se_p50": pct(d["se"], 50), "se_p90": pct(d["se"], 90),
            "bs_p50": pct(d["X"][:, 0], 50), "bs_p90": pct(d["X"][:, 0], 90),
        })
        print(f"  [{m['tag']}] 总 {st['n_total']} 行 → 训练有效 {n_tr} / MTP 有效 {n_mtp} "
              f"(剔除 mode={st['skip_mode']} empty={st['skip_empty']} "
              f"lat={st['skip_lat']} v3={st['skip_v3']} bucket={st['skip_bucket']}) "
              f"| tt p50/p90={pct(d['tt'],50):.0f}/{pct(d['tt'],90):.0f} "
              f"bs p50/p90={pct(d['X'][:,0],50):.0f}/{pct(d['X'][:,0],90):.0f}")
        if m["bucket"] is not None:
            print(f"      bucket={m['bucket']} mode={bucket_mode} 剔除 {st['skip_bucket']} 步"
                  + (f"（配套 requests: {req_file}）" if req_file else ""))
        if hit_info and hit_info["n_requests"] > 0:
            lr, er = hit_info["local_tok_ratio"], hit_info["ext_tok_ratio"]
            print(f"      命中审计: L1(local) tok命中率={lr*100:.3f}% "
                  f"L2(ext) tok命中率={er*100:.3f}% "
                  f"请求级 L1>0={hit_info['local_req_ratio']*100:.2f}% "
                  f"L2>0={hit_info['ext_req_ratio']*100:.2f}%")
            if lr < 0.001 and er < 0.001:
                warnings.append(
                    f"[{m['tag']}] 缓存命中≈0（L1 {lr*100:.3f}% / L2 {er*100:.3f}% token 加权）"
                    f"→ nocache 工况：可训练（hit-origin 时延不变性 ≤±3% 已实证）；"
                    f"e2e 命中路径验证需命中富集资产另验，本数据不承担")
        if v3_on and v3_ratio > 3.0:
            warnings.append(f"[{m['tag']}] V3 剔除比例 {v3_ratio:.2f}% 超过 3%（历史正常范围 "
                            f"0~2%），请确认剔除步画像（data_audit/filter_report.json）后再采信")
        if n_mtp == 0 and "mtp" in m["stages"]:
            warnings.append(f"[{m['tag']}] 无有效 sample_tokens 行，不参与 MTP 标定")
    if v3_on and sum(a["skip_v3"] for a in audit) == 0:
        warnings.append("V3 口径开启但全量数据零剔除：数据健康，V3 未实际生效")
    if all(m["slowdown_factor"] is None for m in meta):
        warnings.append("所有数据集 slowdown_factor 未提供：速率分档检验退化为逐集 base 离散度检查")

    # ---- 覆盖度自检（判断 J6 固化；对 train 集，过滤后口径） ----
    print("\n  覆盖度自检（sum_extend 7 桶 ≥5% / bs 5-8 ≥1%，数据需求 §6）:")
    for i, m in enumerate(meta):
        if "train" not in m["stages"]:
            continue
        d = sets[i]
        cov = L.coverage_report(d["se"], d["X"][:, 0])
        audit[i]["coverage"] = cov
        print(f"    [{m['tag']}] se: " + " ".join(
            f"{r['bucket']}{r['pct']:.1f}%" for r in cov["sum_extend"]))
        print(f"    [{m['tag']}] bs: " + " ".join(
            f"{r['bucket']}{r['pct']:.1f}%" for r in cov["batch_size"]))
        for row in cov["sum_extend"]:
            if 0 < row["pct"] < 5.0 or row["n"] == 0:
                warnings.append(
                    f"[{m['tag']}] sum_extend 桶 {row['bucket']} 占比 {row['pct']:.1f}%"
                    f"（n={row['n']}）低于 5% 阈值 → 欠采，该桶精度不可信")
        bs58 = next(r for r in cov["batch_size"] if r["bucket"] == "bs 5-8")
        if bs58["pct"] < 1.0:
            warnings.append(
                f"[{m['tag']}] batch_size 5-8 占比 {bs58['pct']:.2f}% < 1%：高并发组批工况欠采；"
                f"验收须单独报 bs 5-8 桶 MAPE，必要时用同域样本增广")
    n_tr_total = sum(len(sets[i]["y"]) for i, m in enumerate(meta) if "train" in m["stages"])
    if n_tr_total < 50000:
        warnings.append(f"训练有效总量 {n_tr_total} < 5 万（数据需求 §6 下限），精度风险")

    # ---------------- task 名与产物目录 ----------------
    slug, name_note = resolve_task(raw_task, desc,
                                   [s for s, m in zip(sets, meta) if "train" in m["stages"]],
                                   meta, cfg, args.results_root)
    out = Path(args.results_root) / slug
    for sub in ("model", "mtp", "report", "data_audit"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    print(f"\ntask = {slug}  ({name_note})\n产物目录 = {out}")
    with open(out / "data_audit" / "filter_report.json", "w", encoding="utf-8") as f:
        json.dump({"datasets": audit, "warnings": warnings}, f,
                  ensure_ascii=False, indent=1)

    # ---------------- 1 训练（LODO + 全量终训） ----------------
    log = StageLog(out / "report" / "train_log.txt")
    log("=" * 100)
    log("阶段 1  步时延预测器训练（18feat GBR, 标签 iter_latency）")
    log("=" * 100)
    tr_idx = [i for i, m in enumerate(meta) if "train" in m["stages"]]
    tr_sets = [sets[i] for i in tr_idx]
    tr_tags = [meta[i]["tag"] for i in tr_idx]
    edges = list(cfg["eval"]["tt_buckets"])
    lodo = {}
    for k, (held, tag) in enumerate(zip(tr_sets, tr_tags)):
        others = [s for j, s in enumerate(tr_sets) if j != k]
        X_tr = np.vstack([s["X"] for s in others])
        y_tr = np.concatenate([s["y"] for s in others])
        mdl = L.gbr(cfg["gbr"]).fit(X_tr, y_tr)
        m_tr = L.metrics(y_tr, mdl.predict(X_tr))
        m_te = L.metrics(held["y"], mdl.predict(held["X"]))
        lodo[tag] = {"train": m_tr, "heldout": m_te,
                     "buckets": L.bucket_rows(held["y"], mdl.predict(held["X"]),
                                              held["tt"], edges)}
        log(f"\nheld-out = {tag}")
        log(L.fmt_metrics("train(其余集)", m_tr))
        log(L.fmt_metrics(f"heldout[{tag}]", m_te))
        for b in lodo[tag]["buckets"]:
            log(f"    tt [{b['lo']:>6},{b['hi'] or 'inf':>6}) n={b['n']:<7} "
                f"MAPE={b['MAPE']:5.2f}% MPE={b['MPE']:+5.2f}%")
    X_all = np.vstack([s["X"] for s in tr_sets])
    y_all = np.concatenate([s["y"] for s in tr_sets])
    final = L.gbr(cfg["gbr"]).fit(X_all, y_all)
    m_full = L.metrics(y_all, final.predict(X_all))
    log("\n全量终训:")
    log(L.fmt_metrics("train(全量)", m_full))
    replay = {}
    for s, tag in zip(tr_sets, tr_tags):
        replay[tag] = L.metrics(s["y"], final.predict(s["X"]))
        log(L.fmt_metrics(f"  回放 {tag}", replay[tag]))
    imp = dict(sorted(zip(L.FEATURE_NAMES, final.feature_importances_),
                      key=lambda x: -x[1]))
    log("\n特征重要性 top10:")
    for name, w in list(imp.items())[:10]:
        log(f"  {name:>26}: {w:.4f}")
    pkl_path = out / "model" / f"18feat_{slug}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"model": final, "features": L.FEATURE_NAMES}, f)
    with open(out / "model" / f"18feat_{slug}.features.txt", "w",
              encoding="utf-8") as f:
        f.write("\n".join(L.FEATURE_NAMES) + "\n")
    train_metrics = {"lodo": lodo, "full_train": m_full, "per_set_replay": replay,
                     "feature_importance": imp, "gbr_params": cfg["gbr"],
                     "filter": cfg["filter"]}
    with open(out / "model" / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, ensure_ascii=False, indent=1)
    log(f"\n模型已保存: {pkl_path}")
    log.close()

    # ---------------- 2 MTP 补偿标定 ----------------
    mtp_result = None
    mt_idx = [i for i, m in enumerate(meta)
              if "mtp" in m["stages"] and sets[i]["mtp_mask"].sum() > 0]
    if not mt_idx:
        warnings.append("全部数据集缺 sample_tokens_latency：跳过 MTP 标定；"
                        "hisim_config 不得启用 sample_tokens 补偿，否则 step 时间系统性偏差")
    else:
        log = StageLog(out / "report" / "mtp_fit.log")
        log("=" * 100)
        log("阶段 2  MTP sample_tokens 补偿标定（form=auto: 机理式优先，门控回退 hinge）")
        log("=" * 100)
        mtp_sets = [sets[i] for i in mt_idx]
        mt_tags = [meta[i]["tag"] for i in mt_idx]
        mt_sfs = [meta[i]["slowdown_factor"] for i in mt_idx]

        def cat(key):
            return np.concatenate([s[key][s["mtp_mask"]] for s in mtp_sets])

        st, se, ep, tt = cat("st"), cat("se"), cat("ep"), cat("tt")
        log(f"MTP 有效样本 n={len(st)}（{len(mt_idx)} 集）")
        mech = L.mechanistic_fit(st, se, ep)
        log(f"\n机理式(nnls): base={mech['base']:.3f} a={mech['a']:.4f} b={mech['b']:.4f}  "
            + L.fmt_metrics("mechanistic", mech["metrics"]).strip())
        hng = L.hinge_fit(tt, st, cfg["mtp"]["hinge_bp_grid"])
        log(f"hinge(scan): base={hng['base']:.3f} lo={hng['lo']:.4f}us/tok "
            f"hi={hng['hi']:.4f}us/tok bp={hng['bp']} "
            f"(平底盆地 {hng['plateau']})  " + L.fmt_metrics("hinge", hng["metrics"]).strip())
        gates = L.mtp_gates(mtp_sets, mech, hng, cfg["mtp"])
        log("\n机理式合理性门:")
        for g in gates:
            log(f"  [{'PASS' if g['pass'] else 'FAIL'}] {g['gate']}: {g['detail']}")
        form, reason = L.decide_form(gates, cfg["mtp"]["form"])
        log(f"\n裁决: form={form}  ({reason})")
        tier = L.tiering_check(mtp_sets, mt_tags, mt_sfs, form, cfg["mtp"])
        log(f"速率分档检验: {tier.get('mode')}  ({tier.get('note')})")
        if "base_spread" in tier:
            bs = tier["base_spread"]
            log(f"  逐集 base: min={bs['min']:.3f} max={bs['max']:.3f} "
                f"ratio={bs['ratio']:.3f} (阈值 {bs['threshold']})")
        if tier.get("warning"):
            warnings.append(f"MTP 分档检验: {tier['note']}")

        def fit_group(idxs):
            gs = [mtp_sets[i] for i in idxs]
            _st = np.concatenate([s["st"][s["mtp_mask"]] for s in gs])
            _se = np.concatenate([s["se"][s["mtp_mask"]] for s in gs])
            _ep = np.concatenate([s["ep"][s["mtp_mask"]] for s in gs])
            _tt = np.concatenate([s["tt"][s["mtp_mask"]] for s in gs])
            return L._fit_form(form, _st, _se, _ep, _tt, cfg["mtp"]), len(_st)

        params, recommended = {}, "unified"
        if tier.get("mode") == "tiered":
            native = [i for i, sf in enumerate(mt_sfs) if sf is not None and sf <= 1.0]
            slow = [i for i, sf in enumerate(mt_sfs) if sf is not None and sf > 1.0]
            for label, idxs in (("native", native), ("slow", slow)):
                p, n = fit_group(idxs)
                params[label] = {k: v for k, v in p.items() if k != "metrics"}
                params[label]["n"] = n
                params[label]["metrics"] = p["metrics"]
                log(f"  {label} 档 (n={n}): "
                    + " ".join(f"{k}={v}" for k, v in params[label].items()
                               if k not in ("metrics", "n")))
            recommended = "native"
        else:
            p, n = fit_group(range(len(mtp_sets)))
            params["unified"] = {k: v for k, v in p.items() if k != "metrics"}
            params["unified"]["n"] = n
            params["unified"]["metrics"] = p["metrics"]
        mtp_result = {
            "form": form,
            "decision": {"requested": cfg["mtp"]["form"], "reason": reason,
                         "gates": gates},
            "candidates": {
                "mechanistic": {k: mech[k] for k in ("base", "a", "b")},
                "hinge": {k: hng[k] for k in ("base", "lo", "hi", "bp", "plateau")},
            },
            "params": params, "recommended": recommended, "tier_check": tier,
        }
        with open(out / "mtp" / "mtp_fit.json", "w", encoding="utf-8") as f:
            json.dump(mtp_result, f, ensure_ascii=False, indent=1, default=str)
        rec = params[recommended]
        if form == "mechanistic":
            snippet = {"sample_tokens_base_ms": round(rec["base"], 4),
                       "sample_tokens_a_ms_per_1k_ext": round(rec["a"], 4),
                       "sample_tokens_b_ms_per_1g_ext_past": round(rec["b"], 4)}
        else:
            snippet = {"sample_tokens_base_ms": round(rec["base"], 4),
                       "sample_tokens_lo_us_per_token": round(rec["lo"], 4),
                       "sample_tokens_hi_us_per_token": round(rec["hi"], 4),
                       "sample_tokens_breakpoint_tokens": int(rec["bp"])}
        snippet["_comment"] = (f"form={form}, recommended={recommended}; "
                               f"模型标签为 iter_latency，必须配套本补偿参数使用")
        with open(out / "mtp" / "hisim_config.snippet.json", "w",
                  encoding="utf-8") as f:
            json.dump(snippet, f, ensure_ascii=False, indent=1)
        _plot_mtp(out / "mtp" / "mtp_fit.png", mtp_sets, mt_tags, st, se, tt,
                  form, rec, mech, hng)
        log(f"\n参数已存: {out / 'mtp' / 'mtp_fit.json'}")
        log(f"配置片段: {out / 'mtp' / 'hisim_config.snippet.json'}")
        log.close()

    # ---------------- 3 SUMMARY 与配置快照 ----------------
    _write_summary(out / "report" / "SUMMARY.md", slug, meta, audit,
                   train_metrics, mtp_result, warnings)
    resolved = {"task": slug, "description": desc, "datasets": meta,
                "filter": cfg["filter"], "gbr": cfg["gbr"], "eval": cfg["eval"],
                "mtp": cfg["mtp"], "out": str(out),
                "generated_at": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(out / "config.resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, allow_unicode=True, sort_keys=False)

    print("\n" + "=" * 100)
    print(f"完成: {out}")
    print(f"  模型    model/18feat_{slug}.pkl  (全量 R2={m_full['R2']:.4f} "
          f"MAPE={m_full['MAPE']:.2f}%)")
    if mtp_result:
        rec = mtp_result["params"][mtp_result["recommended"]]
        print(f"  MTP     form={mtp_result['form']} params={rec}")
    for w in warnings:
        print(f"  [WARN] {w}")
    print(f"  摘要    report/SUMMARY.md")


def _plot_mtp(path, mtp_sets, tags, st, se, tt, form, rec, mech, hng):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [WARN] matplotlib 不可用，跳过 png: {e}")
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    for s, tg in zip(mtp_sets, tags):
        mk = s["mtp_mask"]
        ax.scatter(s["se"][mk] / 1e3, s["st"][mk], s=2, alpha=0.3, label=tg)
    if form == "mechanistic":
        xs = np.linspace(0, se.max() / 1e3, 50)
        ax.plot(xs, rec["base"] + rec["a"] * xs, "k-", lw=2,
                label=f"mech (b={rec['b']:.2f})")
    ax.set_xlabel("sum_extend /1k")
    ax.set_ylabel("sample_tokens ms")
    ax.legend(fontsize=7, markerscale=4)
    ax.set_title("fit")
    ax = axes[0, 1]
    yhat = (rec["base"] + rec["a"] * se / 1e3 + rec["b"] *
            np.concatenate([s["ep"][s["mtp_mask"]] for s in mtp_sets]) / 1e9
            if form == "mechanistic"
            else rec["base"] + rec["lo"] * tt / 1e3 +
            (rec["hi"] - rec["lo"]) * np.maximum(0, tt - rec["bp"]) / 1e3)
    ax.scatter(tt / 1e3, yhat - st, s=2, alpha=0.3)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("total_tokens /1k")
    ax.set_ylabel("residual ms")
    ax.set_title("residual vs tt")
    ax = axes[1, 0]
    pe = (yhat - st) / st * 100
    ax.hist(pe, bins=80, range=(-30, 30))
    ax.set_xlabel("APE %")
    ax.set_title(f"APE dist (MAPE={np.abs(pe).mean():.2f}%)")
    ax = axes[1, 1]
    names = ["mechanistic", "hinge"]
    vals = [mech["metrics"]["MAPE"], hng["metrics"]["MAPE"]]
    ax.bar(names, vals, color=["tab:blue" if form == "mechanistic" else "tab:gray",
                               "tab:orange" if form == "hinge" else "tab:gray"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
    ax.set_ylabel("MAPE %")
    ax.set_title(f"form decision -> {form}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_summary(path, slug, meta, audit, tm, mtp, warnings):
    L_ = []
    L_.append(f"# {slug} 训练与标定摘要\n")
    L_.append("## 数据集")
    L_.append("| tag | sf | bucket | stages | 总行 | 训练有效 | MTP有效 | V3剔除 | 桶剔除 | tt p50/p90 |")
    L_.append("|---|---|---|---|---|---|---|---|---|---|")
    for m, a in zip(meta, audit):
        sf = m["slowdown_factor"]
        bk = m.get("bucket")
        bk_s = f"[{int(bk[0])},{int(bk[1])}]/{a.get('bucket_mode')}" if bk else "-"
        L_.append(f"| {m['tag']} | {sf if sf is not None else '?'} | {bk_s} | "
                  f"{'/'.join(m['stages'])} | {a['n_total']} | {a['n_train']} | "
                  f"{a['n_mtp']} | {a['skip_v3']} ({a['v3_drop_pct']}%) | "
                  f"{a.get('skip_bucket', 0)} | {a['tt_p50']:.0f}/{a['tt_p90']:.0f} |")
    hit_rows = [(m["tag"], a["hit"]) for m, a in zip(meta, audit)
                if a.get("hit") and a["hit"].get("n_requests")]
    if hit_rows:
        L_.append("\n## 缓存命中审计（token 加权 / 请求级）")
        L_.append("| tag | L1 tok% | L2 tok% | L1 请求级% | L2 请求级% |")
        L_.append("|---|---|---|---|---|")
        for tag, h in hit_rows:
            L_.append(f"| {tag} | {h['local_tok_ratio']*100:.3f} | "
                      f"{h['ext_tok_ratio']*100:.3f} | {h['local_req_ratio']*100:.2f} | "
                      f"{h['ext_req_ratio']*100:.2f} |")
    cov_rows = [(m["tag"], a["coverage"]) for m, a in zip(meta, audit)
                if a.get("coverage")]
    if cov_rows:
        L_.append("\n## 覆盖度自检（sum_extend / batch_size，阈值 5%）")
        for tag, cov in cov_rows:
            L_.append(f"- **{tag}** se: " + " ".join(
                f"{r['bucket']} {r['pct']:.1f}%" for r in cov["sum_extend"]))
            L_.append(f"  {tag} bs: " + " ".join(
                f"{r['bucket']} {r['pct']:.1f}%" for r in cov["batch_size"]))
    L_.append("\n## 步时延预测器（18feat GBR, 标签 iter_latency）")
    L_.append("| 折 | heldout R2 | heldout MAPE% | heldout MPE% | 总量差% |")
    L_.append("|---|---|---|---|---|")
    for tag, r in tm["lodo"].items():
        h = r["heldout"]
        L_.append(f"| {tag} | {h['R2']:.4f} | {h['MAPE']:.2f} | {h['MPE']:+.2f} | "
                  f"{h['sum_diff_pct']:+.2f} |")
    f_ = tm["full_train"]
    L_.append(f"\n全量终训: R2={f_['R2']:.4f} MAPE={f_['MAPE']:.2f}% "
              f"MAE={f_['MAE_ms']:.2f}ms")
    L_.append("\n## MTP sample_tokens 补偿")
    if mtp:
        L_.append(f"- 裁决 form = **{mtp['form']}**（{mtp['decision']['reason']}）")
        for g in mtp["decision"]["gates"]:
            L_.append(f"  - [{'PASS' if g['pass'] else 'FAIL'}] {g['gate']}: {g['detail']}")
        rec = mtp["params"][mtp["recommended"]]
        L_.append(f"- 推荐参数（{mtp['recommended']}）: "
                  + ", ".join(f"{k}={v}" for k, v in rec.items()
                              if k not in ("metrics", "n")))
        L_.append(f"- 指标: {rec['metrics']}")
        L_.append("- 运行时配置片段见 `mtp/hisim_config.snippet.json`；"
                  "模型标签为 iter_latency，**必须**配套该补偿参数")
    else:
        L_.append("- 未标定（数据缺 sample_tokens_latency）")
    if warnings:
        L_.append("\n## 警告")
        for w in warnings:
            L_.append(f"- {w}")
    L_.append("\n## 文件清单")
    L_.append("- `model/18feat_<task>.pkl` + `.features.txt` + `train_metrics.json`")
    L_.append("- `mtp/mtp_fit.json` + `mtp_fit.png` + `hisim_config.snippet.json`")
    L_.append("- `report/train_log.txt` + `mtp_fit.log`；`data_audit/filter_report.json`")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L_) + "\n")


if __name__ == "__main__":
    main()
