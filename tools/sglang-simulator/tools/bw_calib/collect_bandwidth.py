#!/usr/bin/env python3
"""Standalone L1<->L2 (GPU HBM <-> host memory) KV-transfer bandwidth collector.

Produces the `bandwidth_profile.json` that the v6d CPU simulator consumes to
model transfer latency.  Measures the two local segments:

  seg2   local load    host -> GPU HBM   (L2 -> L1)
  seg2'  local store   GPU HBM -> host   (L1 -> L2)

seg1 (cross-node peer fetch over SRPC/RDMA) is out of scope by design: it needs
two peers and a working data path, and is collected separately.

Two interchangeable backends
----------------------------
  v6d_kernel  Calls the production kernel `ops.v6d_swap_blocks` on an anonymous
              mmap registered through `ops.v6d_register_host_memory` -- exactly
              what the connector does.  Requires the PAI build of vLLM.
              This is the faithful measurement; prefer it whenever available.
  torch       Plain `Tensor.copy_` between pinned host memory and the device.
              Runs on any stock torch+CUDA image (e.g. a GB300 container with
              no custom kernel).  Measures raw cudaMemcpyAsync, so it is an
              upper bound on what the connector can reach.

`--backend auto` (default) probes both and picks v6d_kernel, falling back to
torch with a loud warning.  Every run prints the probe result, the selected
backend and the reason the other one was rejected, so a profile is never
collected silently on the wrong path.

Diagnostics (torch backend)
---------------------------
GB300 sweeps showed bandwidth peaking around 16 blocks and then dropping at
32/64 blocks.  To tell a real C2C characteristic from a measurement artifact:
  --mode loop    num_layers separate copies (mirrors the per-layer store shape)
  --mode contig  one contiguous copy (clean upper bound)
  --streams N    spread copies over N CUDA streams (is one stream the limit?)
  --reverse      sweep large->small (detect thermal / ordering effects)

Runtime-effective model
-----------------------
The simulator's `SegmentModel.from_samples()` does NOT read the summary
`bandwidth_gib_per_s` field: it recomputes the marginal bandwidth from the
LAST TWO samples of the sweep.  A sweep whose tail sits in a slow-down region
therefore silently under-reports (measured on GB300: 111 GiB/s from the tail vs
197 GiB/s in the flat region, a 43% error).  This collector prints the model
that will actually take effect and, by default, keeps only the monotonic prefix
of the sweep in `segments.*.samples` (`--no-trim` to disable; the full sweep is
always kept under `diagnostics`).

Usage
-----
    # PAI image (custom kernel present)
    python3 collect_bandwidth.py --gpu 0 \\
        --page-size 2146304 --num-layers 12 \\
        --blocks 1,2,4,8,16 --out bandwidth_profile.json

    # stock image (GB300), diagnostic sweep
    python3 collect_bandwidth.py --gpu 0 --backend torch \\
        --page-size 2146304 --num-layers 15 \\
        --blocks 1,2,4,8,16,24,32,48,64 --mode both --streams 4 \\
        --out bw_gb300.json
"""
import argparse
import ctypes
import json
import mmap
import os
import statistics
import sys
import time

# torch is imported lazily so that --help works outside the container.
torch = None


def ensure_torch():
    """Import torch with an actionable message instead of a bare traceback."""
    global torch
    if torch is not None:
        return
    try:
        import torch as _torch
    except ImportError as e:
        sys.stderr.write(
            f"collect_bandwidth: cannot import torch ({e}).\n"
            "  Both backends need torch with CUDA support. Run this collector\n"
            "  inside the inference container (the one that serves the model),\n"
            "  not on a login node or a laptop.\n")
        raise SystemExit(2)
    torch = _torch

# Production layout for Qwen3.5-122B-A10B-FP8 @ block_size=2096, tp=2.
# page_size_bytes is the per-rank value; one worker moves num_layers * page_size
# per block, which is what a single kernel call does.
DEF_PAGE_SIZE = 2_146_304
DEF_NUM_LAYERS = 12

# Drop from peak to the last swept point above which the tail is considered
# unusable for the consumer's two-point marginal-bandwidth fit.
TAIL_DROP_WARN_PCT = 10.0


def gib(x):
    return x / (1 << 30)


# ---------------------------------------------------------------------------
# backend probing -- always logged, never silent
# ---------------------------------------------------------------------------
def probe_v6d_kernel():
    """(available, reason). Needs the PAI vLLM build with the v6d swap kernel."""
    try:
        from vllm import _custom_ops as ops  # noqa: F401
    except Exception as e:
        return False, f"cannot import vllm._custom_ops ({type(e).__name__}: {e})"
    from vllm import _custom_ops as ops
    needed = ("v6d_swap_blocks", "v6d_register_host_memory",
              "v6d_unregister_host_memory")
    missing = [n for n in needed if not hasattr(ops, n)]
    if missing:
        return False, ("vllm._custom_ops imported but PAI custom op(s) missing: "
                       + ", ".join(missing) + " (stock vLLM build?)")
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() == False"
    return True, "ops.v6d_swap_blocks + host-memory registration available"


def probe_torch():
    """(available, reason). Needs CUDA and the ability to pin host memory."""
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() == False"
    try:
        t = torch.empty(1 << 20, dtype=torch.int8, pin_memory=True)
        del t
    except Exception as e:
        return False, (f"cannot pin host memory ({type(e).__name__}: {e}); "
                       "start the container with --ulimit memlock=-1")
    return True, f"CUDA ok ({torch.cuda.get_device_name(0)}), host pinning ok"


def select_backend(requested):
    """Print the probe table, return the backend name or exit with a message."""
    probes = {"v6d_kernel": probe_v6d_kernel(), "torch": probe_torch()}
    print("=== backend probe ===")
    for name in ("v6d_kernel", "torch"):
        ok, why = probes[name]
        print(f"  {name:<11s} {'OK         ' if ok else 'UNAVAILABLE'}  {why}")

    if requested != "auto":
        ok, why = probes[requested]
        if not ok:
            print(f"\n!! backend '{requested}' was requested explicitly but is "
                  f"not usable here:\n!!   {why}")
            other = "torch" if requested == "v6d_kernel" else "v6d_kernel"
            if probes[other][0]:
                print(f"!! '{other}' IS usable -- rerun with --backend {other} "
                      f"(or --backend auto).")
            else:
                print("!! neither backend is usable on this machine; nothing "
                      "was measured.")
            sys.exit(2)
        print(f"=== selected backend: {requested} (explicit) ===\n")
        return requested

    if probes["v6d_kernel"][0]:
        print("=== selected backend: v6d_kernel (auto, production-faithful) ===\n")
        return "v6d_kernel"
    if probes["torch"][0]:
        print("=== selected backend: torch (auto FALLBACK) ===")
        print("!! WARNING: the production kernel is unavailable here, so this "
              "run measures raw")
        print("!!   cudaMemcpyAsync instead of ops.v6d_swap_blocks. Treat the "
              "numbers as an UPPER")
        print("!!   BOUND, and do not mix them with kernel-collected profiles "
              "in one comparison.")
        print(f"!!   reason v6d_kernel rejected: {probes['v6d_kernel'][1]}\n")
        return "torch"
    print("\n!! no usable backend: CUDA is required for both paths.")
    print(f"!!   v6d_kernel: {probes['v6d_kernel'][1]}")
    print(f"!!   torch     : {probes['torch'][1]}")
    sys.exit(2)


# ---------------------------------------------------------------------------
# backend: v6d_kernel (production path)
# ---------------------------------------------------------------------------
class HostArena:
    """Anonymous mmap registered with CUDA, mirroring the v6d mmap region."""

    def __init__(self, nbytes: int):
        from vllm import _custom_ops as ops
        self._ops = ops
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
            self._ops.v6d_unregister_host_memory(self.addr)
        except Exception as e:
            # Not fatal for the measurement, but never swallow it silently:
            # a leaked registration skews later points of the same sweep.
            print(f"  [warn] v6d_unregister_host_memory failed: "
                  f"{type(e).__name__}: {e}")
        self._mm.close()


def kernel_build_case(device, blocks, layers, page_size):
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


def kernel_time_once(case, swap_in, stream):
    """Wall time of one kernel, measured with CUDA events on *stream*."""
    from vllm import _custom_ops as ops
    torch.cuda.set_device(case["device"])
    beg, end = torch.cuda.Event(True), torch.cuda.Event(True)
    with torch.cuda.stream(stream):
        beg.record(stream)
        ops.v6d_swap_blocks(case["layer_ptrs"], case["cpu_ptrs"],
                            case["page_size"], case["block_ids"], swap_in)
        end.record(stream)
    end.synchronize()
    return beg.elapsed_time(end) / 1000.0


def kernel_measure(case, swap_in, stream, iters, warmup):
    for _ in range(warmup):
        kernel_time_once(case, swap_in, stream)
    return [kernel_time_once(case, swap_in, stream) for _ in range(iters)]


# ---------------------------------------------------------------------------
# backend: torch (stock image)
# ---------------------------------------------------------------------------
def torch_build_loop(dev, blocks, layers, page_size):
    """num_layers separate GPU + pinned-host buffers (per-layer store shape)."""
    gpu = [torch.empty(blocks * page_size, dtype=torch.int8, device=dev)
           for _ in range(layers)]
    for t in gpu:
        t.fill_(0)
    host = [torch.empty(blocks * page_size, dtype=torch.int8, pin_memory=True)
            for _ in range(layers)]
    for t in host:
        t.fill_(0x5A)  # touch every page so paging is not inside the timed region
    torch.cuda.synchronize(dev)
    return gpu, host


def torch_build_contig(dev, blocks, layers, page_size):
    """One contiguous GPU + pinned-host buffer (bandwidth upper bound)."""
    total = blocks * layers * page_size
    gpu = torch.empty(total, dtype=torch.int8, device=dev)
    gpu.fill_(0)
    host = torch.empty(total, dtype=torch.int8, pin_memory=True)
    host.fill_(0x5A)
    torch.cuda.synchronize(dev)
    return gpu, host


def torch_launch_loop(gpu, host, swap_in):
    def _go(streams):
        n = len(streams)
        for i, (g, h) in enumerate(zip(gpu, host)):
            with torch.cuda.stream(streams[i % n]):
                if swap_in:
                    g.copy_(h, non_blocking=True)
                else:
                    h.copy_(g, non_blocking=True)
        done = []
        for s in streams:
            ev = torch.cuda.Event()
            ev.record(s)
            done.append(ev)
        return done
    return _go


def torch_launch_contig(gpu, host, swap_in):
    total = gpu.numel()

    def _go(streams):
        n = len(streams)
        chunk = (total + n - 1) // n
        done = []
        for i, s in enumerate(streams):
            lo, hi = i * chunk, min(total, (i + 1) * chunk)
            if lo >= hi:
                continue
            with torch.cuda.stream(s):
                if swap_in:
                    gpu[lo:hi].copy_(host[lo:hi], non_blocking=True)
                else:
                    host[lo:hi].copy_(gpu[lo:hi], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(s)
            done.append(ev)
        return done
    return _go


def torch_timed(launch, streams):
    """Seconds for one launch, correct for N streams via begin/wait_event/end."""
    torch.cuda.synchronize()
    lead = streams[0]
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    begin.record(lead)
    for s in streams[1:]:
        s.wait_event(begin)
    for ev in launch(streams):
        lead.wait_event(ev)
    end.record(lead)
    end.synchronize()
    return begin.elapsed_time(end) / 1000.0


def torch_measure(launch, streams, iters, warmup):
    for _ in range(warmup):
        torch_timed(launch, streams)
    return [torch_timed(launch, streams) for _ in range(iters)]


# ---------------------------------------------------------------------------
# fitting and the runtime-effective model
# ---------------------------------------------------------------------------
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


def monotonic_prefix(rows):
    """Longest leading run with both bytes and median_s strictly increasing.

    This is the region where the consumer's two-point marginal fit is valid.
    """
    keep = [rows[0]] if rows else []
    for r in rows[1:]:
        if r["bytes"] > keep[-1]["bytes"] and r["median_s"] > keep[-1]["median_s"]:
            keep.append(r)
        else:
            break
    return keep


def effective_runtime_model(rows, fallback_bw):
    """Reproduce SegmentModel.from_samples() so the operator sees what will
    actually be used at simulation time (summary fields are ignored there)."""
    if len(rows) >= 2:
        b0, t0s = rows[-2]["bytes"], rows[-2]["median_s"]
        b1, t1s = rows[-1]["bytes"], rows[-1]["median_s"]
        if b1 > b0 and t1s > t0s:
            bw = (b1 - b0) / (t1s - t0s)
            floor = min(r["median_s"] for r in rows)
            t0 = max(0.0, t1s - b1 / bw)
            return {"source": "from_samples(last two samples)",
                    "bandwidth_gib_per_s": gib(bw), "floor_s": floor,
                    "fixed_overhead_s": t0}
    return {"source": "fallback: summary bandwidth_bytes_per_s, floor=0",
            "bandwidth_gib_per_s": gib(fallback_bw), "floor_s": 0.0,
            "fixed_overhead_s": 0.0}


def sweep(backend, args, direction, mode, streams, blocks, warnings):
    """One (direction x mode) sweep. Returns (rows, t0, bw, summary)."""
    swap_in = direction == "load"
    label = f"{mode}_{direction}" if mode else direction
    rows, pts = [], []
    print(f"=== {label} ===")
    print(f"{'blk':>4} {'MiB':>9} {'min ms':>9} {'med ms':>9} {'max ms':>9} "
          f"{'GiB/s':>8} {'spread%':>8}")
    for nb in blocks:
        if backend == "v6d_kernel":
            case = kernel_build_case(args.gpu, nb, args.num_layers,
                                     args.page_size)
            try:
                ts = kernel_measure(case, swap_in, streams[0], args.iters,
                                    args.warmup)
            finally:
                case["arena"].close()
            nbytes = case["nbytes"]
        else:
            dev = torch.device(f"cuda:{args.gpu}")
            build = torch_build_loop if mode == "loop" else torch_build_contig
            launcher = torch_launch_loop if mode == "loop" else torch_launch_contig
            gpu, host = build(dev, nb, args.num_layers, args.page_size)
            try:
                ts = torch_measure(launcher(gpu, host, swap_in), streams,
                                   args.iters, args.warmup)
            finally:
                del gpu, host
                torch.cuda.empty_cache()
            nbytes = nb * args.num_layers * args.page_size
        lo, med, hi = min(ts), statistics.median(ts), max(ts)
        rows.append({"blocks": nb, "bytes": nbytes, "min_s": lo,
                     "median_s": med, "max_s": hi,
                     "gib_per_s": gib(nbytes) / med,
                     "spread_pct": 100.0 * (hi - lo) / med})
        pts.append((nbytes, med))
        print(f"{nb:>4} {gib(nbytes) * 1024:>9.1f} {lo * 1e3:>9.3f} "
              f"{med * 1e3:>9.3f} {hi * 1e3:>9.3f} "
              f"{gib(nbytes) / med:>8.2f} {100.0 * (hi - lo) / med:>8.1f}")
    t0, bw = fit_affine(pts)
    peak = max(rows, key=lambda r: r["gib_per_s"])
    tail = max(rows, key=lambda r: r["bytes"])
    drop = 100.0 * (1 - tail["gib_per_s"] / peak["gib_per_s"])
    summary = {"peak_gib_per_s": peak["gib_per_s"], "peak_blocks": peak["blocks"],
               "tail_gib_per_s": tail["gib_per_s"], "tail_blocks": tail["blocks"],
               "drop_pct_peak_to_tail": drop}
    print(f"  affine fit: t0={t0 * 1e6:.1f} us, BW={gib(bw):.2f} GiB/s")
    print(f"  peak {peak['gib_per_s']:.2f} GiB/s @ {peak['blocks']} blk -> "
          f"tail {tail['gib_per_s']:.2f} GiB/s @ {tail['blocks']} blk "
          f"(drop {drop:.1f}%)")
    if drop > TAIL_DROP_WARN_PCT:
        msg = (f"{label}: sweep tail is {drop:.1f}% below peak "
               f"({tail['gib_per_s']:.1f} vs {peak['gib_per_s']:.1f} GiB/s). "
               f"The consumer fits the LAST TWO samples, so an untrimmed "
               f"profile under-reports bandwidth here.")
        print(f"  !! {msg}")
        warnings.append(msg)
    print()
    return rows, t0, bw, summary


# ---------------------------------------------------------------------------
# optional: real control-plane (create+seal) measurement
# ---------------------------------------------------------------------------
def measure_save_completion(v6d_url, blocks, seg2_bytes, iters, warmup):
    """Isolated seal/announce RPC cost against a running v6d daemon, mirroring
    the connector's batch_allocate/seal path. NOT the worker-side async-save
    pipeline tail that the log-derived save_completion captures."""
    import v6d
    from v6d.data.tensor import Tensor
    try:
        dt = torch.int8
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
                                 object_types=[otype] * nb,
                                 ignore_existing=True)
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


# ---------------------------------------------------------------------------
def run(args):
    ensure_torch()
    backend = select_backend(args.backend)
    blocks = [int(b) for b in args.blocks.split(",")]
    if args.reverse:
        blocks = sorted(blocks, reverse=True)
    warnings = []

    if backend == "v6d_kernel":
        modes = [None]                     # the kernel loops over layers itself
        if args.mode != "loop":
            warnings.append(f"--mode {args.mode} ignored: the v6d kernel "
                            f"decomposes layers internally")
            print(f"  [note] --mode {args.mode} does not apply to the "
                  f"v6d_kernel backend; ignored.")
        if args.streams != 1:
            warnings.append(f"--streams {args.streams} ignored on v6d_kernel; "
                            f"use --concurrency for contention tests")
            print(f"  [note] --streams {args.streams} does not apply to the "
                  f"v6d_kernel backend; use the concurrency modes instead.")
        streams = [torch.cuda.Stream(device=args.gpu)]
    else:
        modes = ["loop", "contig"] if args.mode == "both" else [args.mode]
        streams = [torch.cuda.Stream(device=args.gpu)
                   for _ in range(args.streams)]

    seg2 = args.num_layers * args.page_size
    out = {
        "schema": 3,
        "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": os.uname().nodename,
        "gpu_name": torch.cuda.get_device_name(args.gpu),
        "gpu_count": torch.cuda.device_count(),
        "collector": {
            "backend": backend,
            "faithful_to_production": backend == "v6d_kernel",
            "mode": args.mode if backend == "torch" else "kernel_internal_loop",
            "streams": args.streams if backend == "torch" else 1,
            "iters": args.iters, "warmup": args.warmup,
            "reverse_order": args.reverse,
            "trim_to_monotonic": not args.no_trim,
        },
        "layout": {"page_size_bytes": args.page_size,
                   "num_layers": args.num_layers,
                   "seg2_bytes_per_block": seg2},
        # seg1 is filled in by the remote collector; recorded here so consumers
        # can never mistake a loopback number for a cross-machine one.
        "peer_topology": args.peer_topology,
        "peer_endpoint": args.peer_endpoint,
        # Control-plane completion latencies (NOT DMA). save_completion is
        # calibrated from real-run logs; seg1 is a placeholder until a working
        # SRPC/RDMA data path allows a real cross-node measurement.
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
                "calibrated": args.seg1_floor_ms > 0,
                "source": ("measured" if args.seg1_floor_ms > 0 else
                           "PLACEHOLDER: needs two peers with a working data path"),
            },
            "store_completion": {
                "poll_granularity_ms": args.store_poll_ms,
                "rank_sync_ms": args.store_rank_sync_ms,
                "calibrated": True,
                "source": ("poll_granularity=code const asyncio.sleep(0.01); "
                           "rank_sync from real-run SIMPROBE copy_done->mark_saved"),
            },
        },
        "segments": {},
        "diagnostics": {},
        "concurrency": {},
        "warnings": warnings,
    }
    print(f"GPU: {out['gpu_name']} x{out['gpu_count']}")
    print(f"layout: {args.num_layers} layers x {args.page_size} B = {seg2} B/block "
          f"({gib(seg2) * 1024:.2f} MiB)")
    print(f"blocks: {blocks}  iters={args.iters} warmup={args.warmup}\n")
    if args.page_size == DEF_PAGE_SIZE and args.num_layers == DEF_NUM_LAYERS:
        print("  [note] using the built-in Qwen3.5-122B-A10B-FP8 @ tp=2 layout. "
              "Pass --page-size/--num-layers\n         for any other model or TP "
              "degree, otherwise the profile describes the wrong transfer "
              "size.\n")

    # ---- the sweeps ----
    primary_mode = None if backend == "v6d_kernel" else (
        "loop" if "loop" in modes else modes[0])
    for mode in modes:
        for direction in ("load", "store"):
            rows, t0, bw, summary = sweep(backend, args, direction, mode,
                                          streams, blocks, warnings)
            key = f"{mode}_{direction}" if mode else direction
            out["diagnostics"][key] = {"samples": rows,
                                       "affine_fixed_overhead_s": t0,
                                       "affine_bandwidth_gib_per_s": gib(bw),
                                       "summary": summary}
            if mode == primary_mode:
                # Mirror the consumer, which sorts samples by bytes before
                # fitting -- otherwise a --reverse sweep would be trimmed to a
                # single point and silently fall back to the summary fit.
                ordered = sorted(rows, key=lambda r: r["bytes"])
                kept = ordered if args.no_trim else monotonic_prefix(ordered)
                if len(kept) < len(ordered):
                    dropped = [r["blocks"] for r in ordered[len(kept):]]
                    msg = (f"local_{direction}: trimmed non-monotonic tail "
                           f"blocks={dropped} out of segments.samples "
                           f"(full sweep kept under diagnostics.{key})")
                    print(f"  [trim] {msg}")
                    warnings.append(msg)
                eff = effective_runtime_model(kept, bw)
                out["segments"][f"local_{direction}"] = {
                    "fixed_overhead_s": t0,
                    "bandwidth_bytes_per_s": bw,
                    "bandwidth_gib_per_s": gib(bw),
                    "effective_at_runtime": eff,
                    "samples": kept,
                }
                print(f"  EFFECTIVE AT RUNTIME ({eff['source']}): "
                      f"BW={eff['bandwidth_gib_per_s']:.2f} GiB/s, "
                      f"floor={eff['floor_s'] * 1e3:.3f} ms, "
                      f"t0={eff['fixed_overhead_s'] * 1e6:.1f} us\n")

    # ---- loop vs contig verdict: is an early peak a real link limit? ----
    if backend == "torch" and args.mode == "both":
        for direction in ("load", "store"):
            lp = out["diagnostics"].get(f"loop_{direction}", {}).get("summary")
            cg = out["diagnostics"].get(f"contig_{direction}", {}).get("summary")
            if not (lp and cg):
                continue
            ld, cd = lp["drop_pct_peak_to_tail"], cg["drop_pct_peak_to_tail"]
            if max(ld, cd) <= TAIL_DROP_WARN_PCT:
                verdict = "no significant drop in either pattern"
            elif ld - cd > TAIL_DROP_WARN_PCT:
                verdict = ("per-layer-loop artifact (contig stays flat) -- retry "
                           "with --streams N")
            else:
                verdict = "both patterns drop -- likely a real link characteristic"
            print(f"  [{direction}] loop drop={ld:.1f}%  contig drop={cd:.1f}%  "
                  f"-> {verdict}")
        print()

    # ---- concurrency (v6d_kernel only; torch uses --streams) ----
    if args.concurrency and backend == "v6d_kernel":
        from vllm import _custom_ops as ops
        nb = blocks[-1]
        # dual_dir: load and store on one GPU at the same time. PCIe is full
        # duplex, so if the two directions do not slow each other down the
        # model may keep separate load/store budgets.
        a = kernel_build_case(args.gpu, nb, args.num_layers, args.page_size)
        b = kernel_build_case(args.gpu, nb, args.num_layers, args.page_size)
        s1 = torch.cuda.Stream(device=args.gpu)
        s2 = torch.cuda.Stream(device=args.gpu)
        try:
            kernel_measure(a, True, s1, 2, 1)
            kernel_measure(b, False, s2, 2, 1)
            t = time.perf_counter()
            for _ in range(args.iters):
                with torch.cuda.stream(s1):
                    ops.v6d_swap_blocks(a["layer_ptrs"], a["cpu_ptrs"],
                                        a["page_size"], a["block_ids"], True)
                with torch.cuda.stream(s2):
                    ops.v6d_swap_blocks(b["layer_ptrs"], b["cpu_ptrs"],
                                        b["page_size"], b["block_ids"], False)
                s1.synchronize()
                s2.synchronize()
            per = (time.perf_counter() - t) / args.iters
        finally:
            a["arena"].close()
            b["arena"].close()
        out["concurrency"]["dual_dir"] = {
            "blocks": nb, "bytes_each": a["nbytes"], "wall_s": per,
            "aggregate_gib_per_s": gib(2 * a["nbytes"]) / per}
        print(f"  dual_dir   {nb} blk each dir  wall {per * 1e3:.3f} ms  "
              f"aggregate {gib(2 * a['nbytes']) / per:.2f} GiB/s")

        # dual_gpu: two GPUs, same direction -- the TP question. If they do not
        # slow each other down, seg2 should be modelled per rank.
        if torch.cuda.device_count() > 1:
            g2 = (args.gpu + 1) % torch.cuda.device_count()
            a = kernel_build_case(args.gpu, nb, args.num_layers, args.page_size)
            b = kernel_build_case(g2, nb, args.num_layers, args.page_size)
            s1 = torch.cuda.Stream(device=args.gpu)
            s2 = torch.cuda.Stream(device=g2)
            try:
                kernel_measure(a, True, s1, 2, 1)
                kernel_measure(b, True, s2, 2, 1)
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
                    s1.synchronize()
                    s2.synchronize()
                per = (time.perf_counter() - t) / args.iters
            finally:
                a["arena"].close()
                b["arena"].close()
            solo = out["segments"]["local_load"]["samples"][-1]["median_s"]
            out["concurrency"]["dual_gpu"] = {
                "gpus": [args.gpu, g2], "blocks": nb, "bytes_each": a["nbytes"],
                "wall_s": per, "solo_s": solo, "slowdown_vs_solo": per / solo,
                "aggregate_gib_per_s": gib(2 * a["nbytes"]) / per}
            print(f"  dual_gpu   GPU{args.gpu}+GPU{g2} load  wall {per * 1e3:.3f} ms  "
                  f"solo {solo * 1e3:.3f} ms  slowdown {per / solo:.3f}x  "
                  f"aggregate {gib(2 * a['nbytes']) / per:.2f} GiB/s")
    elif args.concurrency and backend == "torch":
        print("  [note] dual_dir / dual_gpu need the v6d kernel; on the torch "
              "backend use --streams N instead.")

    # ---- optional control-plane measurement ----
    if args.v6d_endpoint:
        print(f"\nmeasuring save_completion control-plane on {args.v6d_endpoint} ...")
        try:
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
            msg = (f"save_completion measurement FAILED on "
                   f"{args.v6d_endpoint}: {type(e).__name__}: {e}")
            print(f"  !! {msg}")
            print("  !! (needs the v6d python package and a reachable daemon; "
                  "the DMA segments above are unaffected)")
            warnings.append(msg)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    if warnings:
        print(f"\n=== {len(warnings)} warning(s) recorded in the profile ===")
        for w in warnings:
            print(f"  - {w}")
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["auto", "v6d_kernel", "torch"],
                   default="auto",
                   help="auto probes both and prefers the production kernel")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--page-size", type=int, default=DEF_PAGE_SIZE,
                   help="bytes per layer per block (post-TP-shard, per rank)")
    p.add_argument("--num-layers", type=int, default=DEF_NUM_LAYERS)
    p.add_argument("--blocks", default="1,2,4,8,16")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--mode", choices=["loop", "contig", "both"], default="loop",
                   help="torch backend only: per-layer copies (loop, matches "
                        "the connector), one contiguous copy (contig, upper "
                        "bound), or both")
    p.add_argument("--streams", type=int, default=1,
                   help="torch backend only: spread copies over N CUDA streams")
    p.add_argument("--reverse", action="store_true",
                   help="sweep blocks large->small (detect thermal/ordering)")
    p.add_argument("--no-trim", action="store_true",
                   help="keep the full sweep in segments.samples even if its "
                        "tail is non-monotonic (default: trim, see module doc)")
    p.add_argument("--no-concurrency", dest="concurrency", action="store_false")
    p.add_argument("--peer-topology", default="same_host_loopback",
                   choices=["same_host_loopback", "cross_host_tcp",
                            "cross_host_rdma"],
                   help="recorded in the profile; seg1 numbers are only valid "
                        "for the topology they were collected on")
    p.add_argument("--peer-endpoint", default=None,
                   help="reserved for the seg1 collector (host:port)")
    p.add_argument("--save-floor-ms", type=float, default=70.0,
                   help="save/seal completion fixed latency (from log analysis)")
    p.add_argument("--save-per-blk-ms", type=float, default=6.0,
                   help="save/seal completion per-block latency")
    p.add_argument("--save-ctrl-source", default="real-run log analysis",
                   help="provenance string recorded in the profile")
    p.add_argument("--seg1-floor-ms", type=float, default=0.0,
                   help="cross-node fetch fixed latency (0=placeholder)")
    p.add_argument("--seg1-per-blk-ms", type=float, default=0.0,
                   help="cross-node fetch per-block latency (0=placeholder)")
    p.add_argument("--store-poll-ms", type=float, default=10.0,
                   help="store completion poll granularity (code const 10ms)")
    p.add_argument("--store-rank-sync-ms", type=float, default=1.0,
                   help="store completion all-rank save_done sync (~1ms)")
    p.add_argument("--v6d-endpoint", default=None,
                   help="v6d daemon URL (e.g. http://localhost:7890); if set, "
                        "measures the real create+seal control-plane latency "
                        "and records it as save_completion_measured for gap "
                        "analysis (does NOT overwrite the log-derived value)")
    p.add_argument("--out", default="bandwidth_profile.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
