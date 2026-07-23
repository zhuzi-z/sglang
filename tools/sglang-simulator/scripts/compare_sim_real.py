"""Aggregate sim-vs-real comparison for schedule_batch / requests dumps.

Compares OVERALL statistics only (not per-iteration / per-request):
  - forward latency: iteration counts and iter_latency aggregates, split by
    prefill (forward_mode==1) and decode (forward_mode==2);
  - prefix-cache hit rate: device / local / external hit tokens over total input.

Both sides are produced by the shared collector, so file names/schema match:
  <dir>/TP{rank}.schedule_batch.jsonl   {"forward_mode", "iter_latency", "request_infos":[{"extend_input_len","prefix_indices_len"}]}
  <dir>/TP{rank}.requests.jsonl         {"input_length","final_device_hit_len","local_kv_hit_len","ext_kv_hit_len", ...}

Usage:
  python compare_sim_real.py --sim-dir <sim_out> --real-dir <real_out> [--rank 0]
  python compare_sim_real.py --sim-dir <sim_out>          # summarize one side only
"""

import argparse
import json
import os
from statistics import mean, median


def _percentile(values, q):
    """Nearest-rank percentile (q in [0,100]); pure stdlib, no numpy."""
    if not values:
        return 0.0
    s = sorted(values)
    if q <= 0:
        return s[0]
    if q >= 100:
        return s[-1]
    k = max(0, min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1)))))
    return s[k]


def _read_jsonl(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize_latency(batches):
    """Aggregate iter_latency stats, split by forward_mode."""
    out = {}
    for label, fmode in (("prefill", 1), ("decode", 2), ("all", None)):
        lats = [
            float(b["iter_latency"])
            for b in batches
            if fmode is None or b.get("forward_mode") == fmode
        ]
        out[label] = {
            "iters": len(lats),
            "latency_sum": sum(lats),
            "latency_mean": mean(lats) if lats else 0.0,
            "latency_p50": median(lats) if lats else 0.0,
            "latency_p90": _percentile(lats, 90),
            "latency_p99": _percentile(lats, 99),
        }
    return out


def summarize_hit(requests):
    """Aggregate prefix-cache hit rate over all requests."""
    tot_in = sum(int(r.get("input_length", 0)) for r in requests)
    tot_dev = sum(int(r.get("final_device_hit_len", 0)) for r in requests)
    tot_loc = sum(int(r.get("local_kv_hit_len", 0)) for r in requests)
    tot_ext = sum(int(r.get("ext_kv_hit_len", 0)) for r in requests)
    denom = tot_in or 1
    return {
        "requests": len(requests),
        "input_tokens": tot_in,
        "device_hit_tokens": tot_dev,
        "local_hit_tokens": tot_loc,
        "ext_hit_tokens": tot_ext,
        "device_hit_rate": tot_dev / denom,
        "local_hit_rate": tot_loc / denom,
        "ext_hit_rate": tot_ext / denom,
    }


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _rel_err(sim, real):
    if real == 0:
        return "n/a" if sim == 0 else "inf"
    return f"{(sim - real) / real * 100:+.2f}%"


def _print_block(title, sim, real):
    print(f"\n=== {title} ===")
    keys = list(sim.keys())
    if real is None:
        print(f"{'metric':<20}{'sim':>16}")
        for k in keys:
            print(f"{k:<20}{_fmt(sim[k]):>16}")
        return
    print(f"{'metric':<20}{'sim':>16}{'real':>16}{'rel_err':>12}")
    for k in keys:
        print(
            f"{k:<20}{_fmt(sim[k]):>16}{_fmt(real[k]):>16}"
            f"{_rel_err(sim[k], real[k]):>12}"
        )


def load_side(dirpath, rank):
    sb = _read_jsonl(os.path.join(dirpath, f"TP{rank}.schedule_batch.jsonl"))
    rq = _read_jsonl(os.path.join(dirpath, f"TP{rank}.requests.jsonl"))
    if sb is None:
        raise FileNotFoundError(
            f"missing TP{rank}.schedule_batch.jsonl under {dirpath}"
        )
    return sb, (rq or [])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--real-dir", default=None)
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args()

    sim_sb, sim_rq = load_side(args.sim_dir, args.rank)
    sim_lat = summarize_latency(sim_sb)
    sim_hit = summarize_hit(sim_rq)

    real_lat = real_hit = None
    if args.real_dir:
        real_sb, real_rq = load_side(args.real_dir, args.rank)
        real_lat = summarize_latency(real_sb)
        real_hit = summarize_hit(real_rq)

    for label in ("prefill", "decode", "all"):
        _print_block(
            f"latency [{label}]",
            sim_lat[label],
            real_lat[label] if real_lat else None,
        )
    _print_block("prefix-cache hit", sim_hit, real_hit)


if __name__ == "__main__":
    main()
