# L1 <-> L2 KV-Transfer Bandwidth Collector

`collect_bandwidth.py` measures the host <-> GPU transfer segments that the v6d
CPU simulator needs in order to model KV-cache movement latency, and writes them
into a `bandwidth_profile.json`.

| Segment | Direction | Meaning |
|---|---|---|
| `local_load` (seg2) | host -> GPU HBM | L2 -> L1, executed before forward |
| `local_store` (seg2') | GPU HBM -> host | L1 -> L2, executed after forward |

Cross-node fetch (seg1, peer -> local over SRPC/RDMA) is **out of scope**: it
needs two peers and a working data path, so it is collected separately and only
recorded here as a placeholder together with the `peer_topology` it belongs to.

## 1. Backends

The collector has two interchangeable measurement backends and always reports
which one it selected and why.

| Backend | Requires | Measures | Use when |
|---|---|---|---|
| `v6d_kernel` | PAI build of vLLM exposing `ops.v6d_swap_blocks` and `ops.v6d_register_host_memory` | the production kernel, over an anonymous `mmap` registered with CUDA (what the connector does) | preferred whenever available |
| `torch` | stock torch + CUDA, host pinning allowed | plain `Tensor.copy_` between pinned host memory and the device | stock images with no custom kernel (e.g. GB300 containers) |

`--backend auto` (default) probes both, prefers `v6d_kernel`, and falls back to
`torch` with an explicit warning. Every run starts with the probe table:

```text
=== backend probe ===
  v6d_kernel  UNAVAILABLE  vllm._custom_ops imported but PAI custom op(s) missing: v6d_swap_blocks (stock vLLM build?)
  torch       OK           CUDA ok (NVIDIA GB300), host pinning ok
=== selected backend: torch (auto FALLBACK) ===
!! WARNING: the production kernel is unavailable here, so this run measures raw
!!   cudaMemcpyAsync instead of ops.v6d_swap_blocks. Treat the numbers as an UPPER
!!   BOUND, and do not mix them with kernel-collected profiles in one comparison.
```

If a backend is requested explicitly but unusable, the collector prints the
reason, names the backend that *would* work, and exits with status 2 instead of
failing deep inside a measurement.

### Notes

- `torch` numbers are an upper bound: the production kernel adds per-layer
  pointer indirection that a single `copy_` does not pay.
- The profile records `collector.backend` and
  `collector.faithful_to_production`, so a fallback profile can never be
  mistaken for a kernel-collected one.

## 2. How the measurement works

1. **Buffers.** For each sweep point the collector allocates
   `blocks x num_layers x page_size` bytes on both sides. `--page-size` is the
   **per-rank, post-TP-shard** bytes per layer per block; one worker moves
   `num_layers * page_size` per block, which is exactly one kernel call.
   Host pages are written once before timing (a fresh anonymous mapping is
   backed by the shared zero page, and paying page population inside the timed
   region flattens the small-transfer end of the curve). GPU buffers are filled
   and synchronised for the same reason.
2. **Transfer.** `v6d_kernel` calls `ops.v6d_swap_blocks(..., swap_in)` with
   `swap_in=True` for load and `False` for store. `torch` either issues
   `num_layers` separate copies (`--mode loop`, the connector's shape) or one
   contiguous copy (`--mode contig`, a clean upper bound).
3. **Timing.** CUDA events around the enqueue, `end.synchronize()` before
   reading the elapsed time. With `--streams N` the begin event is recorded on
   the lead stream, the others wait on it, and the lead waits for every stream's
   completion event, so the interval covers all streams.
4. **Statistics.** `--warmup` iterations are discarded, then `--iters`
   iterations are kept; each point reports min / median / max and the spread in
   percent. The median is used for fitting.
5. **Fit.** Least squares on `t = t0 + bytes / bandwidth` across the sweep.

## 3. Usage

```sh
# PAI image (production kernel available)
python3 collect_bandwidth.py --gpu 0 \
    --page-size 2146304 --num-layers 12 \
    --blocks 1,2,4,8,16 --iters 20 --warmup 5 \
    --out bandwidth_profile.json

# stock image, e.g. GB300 container (no custom kernel)
python3 collect_bandwidth.py --gpu 0 --backend torch \
    --page-size 2146304 --num-layers 15 \
    --blocks 1,2,4,8,16,24,32,48,64 --mode both --streams 4 \
    --out bw_gb300.json
```

Key options:

| Option | Effect |
|---|---|
| `--backend auto\|v6d_kernel\|torch` | backend selection; `auto` prefers the kernel |
| `--blocks 1,2,4,8,16` | sweep points, in blocks |
| `--mode loop\|contig\|both` | torch only: per-layer copies vs one contiguous copy |
| `--streams N` | torch only: spread copies over N CUDA streams |
| `--reverse` | sweep large -> small, to expose thermal / ordering effects |
| `--no-trim` | keep a non-monotonic tail in `segments.*.samples` (see section 4) |
| `--no-concurrency` | skip the `dual_dir` / `dual_gpu` contention tests |
| `--v6d-endpoint URL` | additionally time the isolated `create + seal` RPC |

### Notes

- The container must allow page pinning (`--ulimit memlock=-1`), otherwise the
  torch backend fails the probe with an explicit message.
- `dual_dir` (load and store together on one GPU) answers whether the link is
  really full duplex; `dual_gpu` (two GPUs, same direction) answers whether TP
  ranks slow each other down. Both need the `v6d_kernel` backend; on `torch`
  use `--streams` instead.

## 4. The tail of the sweep decides the runtime bandwidth

The simulator does **not** read the summary `bandwidth_gib_per_s` field. Its
`SegmentModel.from_samples()` recomputes the marginal bandwidth from the **last
two samples** of `segments.*.samples`, taking the floor from the smallest
median. A sweep whose tail sits in a slow-down region therefore silently
under-reports: on GB300 the tail gave 111 GiB/s while the flat region was
197 GiB/s, a 43% error, even though the summary field looked right.

The collector therefore:

- prints the model that will actually take effect, per direction:

  ```text
  EFFECTIVE AT RUNTIME (from_samples(last two samples)): BW=197.31 GiB/s, floor=0.412 ms, t0=11.8 us
  ```

- warns when the tail is more than 10% below the peak, and records the warning
  in the profile's `warnings` array;
- keeps, by default, only the leading run of points where both bytes and time
  increase strictly (`segments.*.samples`), while the full sweep always stays
  under `diagnostics.<mode>_<direction>.samples`. `--no-trim` disables this.

### Notes

- Editing `bandwidth_gib_per_s` by hand has no effect at simulation time; change
  the `samples` array instead.
- If a curve peaks early and then drops, first check whether it is an artifact:
  run `--mode contig` and `--streams 4`. If `contig` does not drop but `loop`
  does, the drop comes from the per-layer loop, not from the link.

## 5. Profile layout

```jsonc
{
  "schema": 3,
  "collector": { "backend": "v6d_kernel", "faithful_to_production": true,
                 "mode": "kernel_internal_loop", "streams": 1,
                 "trim_to_monotonic": true },
  "layout": { "page_size_bytes": 2146304, "num_layers": 12,
              "seg2_bytes_per_block": 25755648 },
  "segments": {
    "local_load":  { "fixed_overhead_s": 1.2e-05,
                     "bandwidth_bytes_per_s": 5.2e+10,
                     "effective_at_runtime": { "source": "from_samples(last two samples)",
                                               "bandwidth_gib_per_s": 48.73,
                                               "floor_s": 0.0022 },
                     "samples": [ { "blocks": 1, "bytes": 25755648,
                                    "min_s": 0.0, "median_s": 0.0, "max_s": 0.0,
                                    "gib_per_s": 0.0, "spread_pct": 0.0 } ] },
    "local_store": { "...": "same shape" }
  },
  "diagnostics": { "loop_load": { "samples": [] }, "contig_load": { "samples": [] } },
  "concurrency": { "dual_dir": {}, "dual_gpu": {} },
  "control_plane": {
    "save_completion":  { "floor_ms": 70.0, "per_block_ms": 6.0,
                          "source": "real-run log analysis" },
    "store_completion": { "poll_granularity_ms": 10.0, "rank_sync_ms": 1.0 },
    "seg1_cross_node":  { "floor_ms": 0.0, "per_block_ms": 0.0,
                          "source": "PLACEHOLDER: needs two peers with a working data path" }
  },
  "warnings": []
}
```

`control_plane` values are **not** bandwidth measurements:

- `save_completion` (70 ms + 6 ms/block) is a timing-compensation parameter
  regressed from real-run logs (`mark_saved - last store`, 190 samples). It
  covers the asynchronous save pipeline tail, not a DMA copy.
- `store_completion` holds structural constants: the 10 ms polling granularity
  of the async swap (`asyncio.sleep(0.01)`) and the ~1 ms all-rank `save_done`
  aggregation.
- `--v6d-endpoint` measures the isolated `create + seal` RPC (about
  1.3 ms + 0.08 ms/block in the reference environment) and stores it under
  `save_completion_measured` **for gap analysis only**; it never overwrites the
  log-derived value.

## 6. Recalibrating for a new model or GPU

1. Recompute `--page-size` and `--num-layers` for the new model and TP degree,
   then rerun the sweep and keep the new profile next to the run it belongs to.
2. Check the printed `EFFECTIVE AT RUNTIME` line, not the summary field.
3. Re-derive `save_completion` from that environment's real-run logs; do not
   substitute the isolated RPC measurement for it.
4. Fill in `seg1_cross_node` only from a two-peer measurement on the topology
   recorded in `peer_topology`.
