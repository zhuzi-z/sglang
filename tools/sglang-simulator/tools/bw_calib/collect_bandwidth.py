#!/usr/bin/env python3
"""Standalone v6d KV-transfer bandwidth collector.

Measures the transfer segments the simulator needs to model, without needing
vLLM or (for the local segments) a v6d server:

  seg2   local load    pinned host -> GPU HBM   (ops.v6d_swap_blocks swap_in=True)
  seg2'  local store   GPU HBM -> pinned host   (ops.v6d_swap_blocks swap_in=False)
  seg1   remote fetch  peer v6d -> local v6d    (SRPC; needs two peers -- see
                                                 collect_remote.py, not here)

Faithful to production in three ways: the host buffer is an anonymous mmap
registered through ``ops.v6d_register_host_memory`` (exactly what the connector
does to the v6d mmap), the copy goes through the same CUDA kernel, and the
(blocks x layers x page_size) decomposition matches the connector's buckets.

Concurrency modes exist to settle whether the two TP ranks' DMAs contend:
  single_dir  one GPU, one direction            -> per-link one-way bandwidth
  dual_dir    one GPU, load+store together      -> is PCIe really full duplex?
  dual_gpu    two GPUs, same direction together -> do the ranks slow each other?

Cross-node / RDMA is out of scope here by design: only seg1 crosses the network,
and its collector takes an explicit peer endpoint so the same code path serves
loopback, TCP and RDMA. The profile records ``peer_topology`` so a loopback
calibration is never mistaken for a cross-machine one.
"""
import argparse, ctypes, json, mmap, os, statistics, sys, time

import torch
from vllm import _custom_ops as ops

# Production layout for Qwen3.5-122B-A10B-FP8 @ block_size=2096, tp=2.
# page_size_bytes is the per-rank value; one worker moves num_layers * page_size
# per block, which is what this kernel call does.
DEF_PAGE_SIZE = 2_146_304
DEF_NUM_LAYERS = 12


class HostArena:
    """Anonymous mmap registered with CUDA, mirroring the v6d mmap region."""

    def __init__(self, nbytes: int):
        self._mm = mmap.mmap(-1, nbytes)
        # Fault every page in before registering. A fresh anonymous mapping is
        # backed by the shared zero page, which makes the first host->device
        # read pay for page population inside the timed region and flattens the
        # small-transfer end of the curve.
        self._mm.write(b"\xa5" * nbytes)
        self._mm.seek(0)
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self._mm))
        self.size = nbytes
        ops.v6d_register_host_memory(self.addr, nbytes)

    def close(self):
        try:
            ops.v6d_unregister_host_memory(self.addr)
        except Exception:
            pass
        self._mm.close()


def build_case(device: int, blocks: int, layers: int, page_size: int):
    """Allocate one side's buffers and the pointer tensors the kernel wants."""
    torch.cuda.set_device(device)
    dev = torch.device(f"cuda:{device}")
    gpu = [torch.empty(blocks * page_size, dtype=torch.int8, device=dev)
           for _ in range(layers)]
    for t in gpu:            # touch, so the timed region is pure transfer
        t.fill_(0)
    torch.cuda.synchronize(dev)
    # The kernel requires the layer pointer table on the device; the block
    # pointer table and the block-id list stay on the host (matching the
    # connector's cpu_ptrs_cpu / gpu_ids_cpu naming).
    layer_ptrs = torch.tensor([t.data_ptr() for t in gpu], dtype=torch.int64,
                              device=dev)
    block_ids = torch.arange(blocks, dtype=torch.int64)
    arena = HostArena(blocks * layers * page_size)
    cpu_ptrs = torch.tensor(
        [arena.addr + b * layers * page_size for b in range(blocks)],
        dtype=torch.int64, pin_memory=True)
    return dict(gpu=gpu, layer_ptrs=layer_ptrs, block_ids=block_ids,
                arena=arena, cpu_ptrs=cpu_ptrs, device=device,
                nbytes=blocks * layers * page_size, page_size=page_size)


def time_once(case, swap_in: bool, stream) -> float:
    """Wall time of one kernel, measured with CUDA events on *stream*."""
    torch.cuda.set_device(case["device"])
    beg, end = torch.cuda.Event(True), torch.cuda.Event(True)
    with torch.cuda.stream(stream):
        beg.record(stream)
        ops.v6d_swap_blocks(case["layer_ptrs"], case["cpu_ptrs"],
                            case["page_size"], case["block_ids"], swap_in)
        end.record(stream)
    end.synchronize()
    return beg.elapsed_time(end) / 1000.0


def measure(case, swap_in, stream, iters, warmup):
    for _ in range(warmup):
        time_once(case, swap_in, stream)
    return [time_once(case, swap_in, stream) for _ in range(iters)]


def fit_affine(points):
    """Least squares on t = t0 + bytes / bw. Returns (t0_s, bw_bytes_per_s)."""
    xs = [b for b, _ in points]
    ys = [t for _, t in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var else 0.0
    t0 = my - slope * mx
    return t0, (1.0 / slope if slope > 0 else float("inf"))


def gib(x):
    return x / (1 << 30)


def measure_save_completion(v6d_url, blocks, seg2_bytes, iters, warmup):
    """Real control-plane latency: time client.create + obj.seal against a
    running v6d daemon, mirroring the connector's batch_allocate/seal path.
    This is the isolated seal/announce RPC cost (NOT the worker-side async-save
    pipeline tail that the log-derived save_completion captures)."""
    import v6d
    from v6d.data.tensor import Tensor
    try:
        import torch as _t
        dt = _t.int8
    except Exception:
        dt = "int8"
    client = v6d.connect(v6d_url)
    otype = Tensor.type(shape=[seg2_bytes], dtype=dt)
    rows = []
    for nb in blocks:
        ts = []
        for it in range(iters + warmup):
            keys = [f"bwcal_{nb}_{it}_{i}" for i in range(nb)]
            t0 = time.perf_counter()
            objs = client.create(keys=keys, sizes=[seg2_bytes] * nb,
                                 object_types=[otype] * nb, ignore_existing=True)
            for o in objs:
                if o is not None:
                    o.seal()
            if it >= warmup:
                ts.append(time.perf_counter() - t0)
        med = statistics.median(ts)
        rows.append({"blocks": nb, "median_s": med, "median_ms": med * 1e3})
        print(f"  save_ctrl  {nb:2d} blk  create+seal {med * 1e3:.3f} ms")
    b0, b1 = rows[0]["blocks"], rows[-1]["blocks"]
    m0, m1 = rows[0]["median_s"], rows[-1]["median_s"]
    per_blk_ms = (m1 - m0) / (b1 - b0) * 1e3 if b1 > b0 else 0.0
    floor_ms = m0 * 1e3 - per_blk_ms * b0
    return {"floor_ms": round(max(0.0, floor_ms), 3),
            "per_block_ms": round(per_blk_ms, 4), "samples": rows}


def run(args):
    blocks = [int(b) for b in args.blocks.split(",")]
    out = {
        "schema": 1,
        "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": os.uname().nodename,
        "gpu_name": torch.cuda.get_device_name(args.gpu),
        "gpu_count": torch.cuda.device_count(),
        "layout": {"page_size_bytes": args.page_size,
                   "num_layers": args.num_layers,
                   "seg2_bytes_per_block": args.num_layers * args.page_size},
        # seg1 is filled in by the remote collector; recorded here so consumers
        # can never mistake a loopback number for a cross-machine one.
        "peer_topology": args.peer_topology,
        "peer_endpoint": args.peer_endpoint,
        # Control-plane completion latencies (NOT DMA). save_completion
        # is calibrated from real-run logs; seg1 is a placeholder until a
        # working SRPC/RDMA data path allows a real cross-node measurement.
        "control_plane": {
            "save_completion": {
                "floor_ms": args.save_floor_ms,
                "per_block_ms": args.save_per_blk_ms,
                "calibrated": args.save_floor_ms > 0 or args.save_per_blk_ms > 0,
                "source": args.save_ctrl_source,
            },
            "seg1_cross_node": {
                "floor_ms": args.seg1_floor_ms,
                "per_block_ms": args.seg1_per_blk_ms,
                "calibrated": args.seg1_floor_ms > 0 or args.seg1_per_blk_ms > 0,
                "source": ("measured" if args.seg1_floor_ms > 0
                           else "PLACEHOLDER: data path stubbed; use collect_remote.py on real HW"),
            },
        },
        "segments": {},
        "concurrency": {},
    }
    print(f"GPU: {out['gpu_name']} x{out['gpu_count']}")
    print(f"layout: {args.num_layers} layers x {args.page_size} B "
          f"= {args.num_layers * args.page_size} B/block "
          f"({gib(args.num_layers * args.page_size) * 1024:.2f} MiB)")
    print()

    stream = torch.cuda.Stream(device=args.gpu)

    # ---- single_dir: the baseline the model needs ----
    for name, swap_in in (("local_load", True), ("local_store", False)):
        pts, rows = [], []
        for nb in blocks:
            case = build_case(args.gpu, nb, args.num_layers, args.page_size)
            try:
                ts = measure(case, swap_in, stream, args.iters, args.warmup)
            finally:
                case["arena"].close()
            med = statistics.median(ts)
            pts.append((case["nbytes"], med))
            rows.append({"blocks": nb, "bytes": case["nbytes"],
                         "median_s": med,
                         "p10_s": min(ts), "p90_s": max(ts),
                         "gib_per_s": gib(case["nbytes"]) / med})
            print(f"  {name:12s} {nb:3d} blk  {gib(case['nbytes']) * 1024:8.1f} MiB  "
                  f"{med * 1e3:8.3f} ms  {gib(case['nbytes']) / med:7.2f} GiB/s")
        t0, bw = fit_affine(pts)
        out["segments"][name] = {"fixed_overhead_s": t0,
                                 "bandwidth_bytes_per_s": bw,
                                 "bandwidth_gib_per_s": gib(bw),
                                 "samples": rows}
        print(f"  -> {name}: t0={t0 * 1e6:.1f} us, BW={gib(bw):.2f} GiB/s")
        print()

    if args.concurrency:
        nb = blocks[-1]
        # dual_dir: load and store on one GPU at the same time. PCIe is full
        # duplex, so if the two directions do not slow each other down the
        # model may keep separate load/store budgets.
        a = build_case(args.gpu, nb, args.num_layers, args.page_size)
        b = build_case(args.gpu, nb, args.num_layers, args.page_size)
        s1 = torch.cuda.Stream(device=args.gpu)
        s2 = torch.cuda.Stream(device=args.gpu)
        try:
            measure(a, True, s1, 2, 1); measure(b, False, s2, 2, 1)
            t = time.perf_counter()
            for _ in range(args.iters):
                with torch.cuda.stream(s1):
                    ops.v6d_swap_blocks(a["layer_ptrs"], a["cpu_ptrs"],
                                        a["page_size"], a["block_ids"], True)
                with torch.cuda.stream(s2):
                    ops.v6d_swap_blocks(b["layer_ptrs"], b["cpu_ptrs"],
                                        b["page_size"], b["block_ids"], False)
                s1.synchronize(); s2.synchronize()
            per = (time.perf_counter() - t) / args.iters
        finally:
            a["arena"].close(); b["arena"].close()
        out["concurrency"]["dual_dir"] = {
            "blocks": nb, "bytes_each": a["nbytes"], "wall_s": per,
            "aggregate_gib_per_s": gib(2 * a["nbytes"]) / per}
        print(f"  dual_dir   {nb} blk each dir  wall {per * 1e3:.3f} ms  "
              f"aggregate {gib(2 * a['nbytes']) / per:.2f} GiB/s")

        # dual_gpu: two GPUs, same direction. This is the tp=2 question -- if
        # they do not slow each other, seg2 should be modelled per rank.
        if torch.cuda.device_count() > 1:
            g2 = (args.gpu + 1) % torch.cuda.device_count()
            a = build_case(args.gpu, nb, args.num_layers, args.page_size)
            b = build_case(g2, nb, args.num_layers, args.page_size)
            s1 = torch.cuda.Stream(device=args.gpu)
            s2 = torch.cuda.Stream(device=g2)
            try:
                measure(a, True, s1, 2, 1); measure(b, True, s2, 2, 1)
                t = time.perf_counter()
                for _ in range(args.iters):
                    torch.cuda.set_device(args.gpu)
                    with torch.cuda.stream(s1):
                        ops.v6d_swap_blocks(a["layer_ptrs"], a["cpu_ptrs"],
                                            a["page_size"], a["block_ids"], True)
                    torch.cuda.set_device(g2)
                    with torch.cuda.stream(s2):
                        ops.v6d_swap_blocks(b["layer_ptrs"], b["cpu_ptrs"],
                                            b["page_size"], b["block_ids"], True)
                    s1.synchronize(); s2.synchronize()
                per = (time.perf_counter() - t) / args.iters
            finally:
                a["arena"].close(); b["arena"].close()
            solo = out["segments"]["local_load"]["samples"][-1]["median_s"]
            out["concurrency"]["dual_gpu"] = {
                "gpus": [args.gpu, g2], "blocks": nb, "bytes_each": a["nbytes"],
                "wall_s": per, "solo_s": solo, "slowdown_vs_solo": per / solo,
                "aggregate_gib_per_s": gib(2 * a["nbytes"]) / per}
            print(f"  dual_gpu   GPU{args.gpu}+GPU{g2} load  wall {per * 1e3:.3f} ms  "
                  f"solo {solo * 1e3:.3f} ms  slowdown {per / solo:.3f}x  "
                  f"aggregate {gib(2 * a['nbytes']) / per:.2f} GiB/s")

    # ---- optional: real control-plane (create+seal) measurement ----
    if getattr(args, "v6d_endpoint", None):
        print(f"\nmeasuring save_completion control-plane on {args.v6d_endpoint} ...")
        try:
            seg2 = args.num_layers * args.page_size
            meas = measure_save_completion(args.v6d_endpoint, blocks, seg2,
                                           args.iters, args.warmup)
            meas["source"] = f"measured: v6d create+seal RPC on {args.v6d_endpoint}"
            out["control_plane"]["save_completion_measured"] = meas
            lg = out["control_plane"]["save_completion"]
            print(f"  GAP: measured create+seal floor={meas['floor_ms']:.2f}ms "
                  f"per_blk={meas['per_block_ms']:.3f}ms  VS  log-derived "
                  f"save_completion floor={lg['floor_ms']:.1f}ms "
                  f"per_blk={lg['per_block_ms']:.1f}ms")
        except Exception as e:
            print(f"  save_completion measurement FAILED: {e}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--page-size", type=int, default=DEF_PAGE_SIZE)
    p.add_argument("--num-layers", type=int, default=DEF_NUM_LAYERS)
    p.add_argument("--blocks", default="1,2,4,8,16")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--no-concurrency", dest="concurrency", action="store_false")
    p.add_argument("--peer-topology", default="same_host_loopback",
                   choices=["same_host_loopback", "cross_host_tcp",
                            "cross_host_rdma"],
                   help="recorded in the profile; seg1 numbers are only valid "
                        "for the topology they were collected on")
    p.add_argument("--peer-endpoint", default=None,
                   help="reserved for the seg1 collector (host:port)")
    # Control-plane completion latencies folded into the same profile.
    p.add_argument("--save-floor-ms", type=float, default=70.0,
                   help="save/seal completion fixed latency (from log analysis)")
    p.add_argument("--save-per-blk-ms", type=float, default=6.0,
                   help="save/seal completion per-block latency")
    p.add_argument("--save-ctrl-source", default="real-run log analysis",
                   help="provenance string recorded in the profile")
    p.add_argument("--seg1-floor-ms", type=float, default=0.0,
                   help="cross-node fetch fixed latency (0=placeholder, needs real HW)")
    p.add_argument("--seg1-per-blk-ms", type=float, default=0.0,
                   help="cross-node fetch per-block latency (0=placeholder)")
    p.add_argument("--v6d-endpoint", default=None,
                   help="v6d daemon URL (e.g. http://localhost:7890); if set, "
                        "measures the real create+seal control-plane latency "
                        "and records it as save_completion_measured for gap "
                        "analysis (does NOT overwrite the log-derived value)")
    p.add_argument("--out", default="/root/workspace/server/bandwidth_profile.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
