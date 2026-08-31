"""Transfer-latency model for the simulated v6d data plane.

The simulation keeps the v6d control plane real (lookup/create/seal go to the
daemon) but the DMA between the v6d mmap and GPU HBM never runs (the worker's
swap handlers are left None), so it costs nothing. That single omission is
what this module restores.

Where the latency is applied matters, and it follows the production code:

  seg2  load   ``start_load_kv`` enqueues on its own stream and returns. The
               request only enters ``running`` once ``get_finished()`` reports
               it, which the scheduler turns into ``finished_recving``. So the
               latency belongs on the load event's ready time.
  seg2' store  ``wait_for_save()`` (which despite the name does not wait) also
               only enqueues. The scheduler frees the request's GPU blocks on
               ``finished_sending``, so delaying the store event's ready time
               delays block release -- the real back-pressure path.

Neither consumes engine time, which is why nothing here sleeps in the worker's
main flow.

Calibration comes from ``tools/bw_calib/collect_bandwidth.py``. Point
``SGLANG_SIMULATOR_BW_PROFILE`` at its output to enable the model; with the
variable unset the model stays disabled and behaviour is unchanged.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_ENV_PROFILE = "SGLANG_SIMULATOR_BW_PROFILE"


class SegmentModel:
    """Latency of one transfer segment: ``max(floor, t0 + bytes / bw)``.

    The floor is not cosmetic. Measured host->device transfers sit at a
    constant ~2.2 ms for anything up to four blocks and only become
    bandwidth-bound above that, so a plain affine fit understates small
    transfers by 4x and overstates the asymptotic bandwidth.
    """

    def __init__(self, name: str, floor_s: float, t0_s: float, bw_bytes_per_s: float):
        self.name = name
        self.floor_s = max(0.0, floor_s)
        self.t0_s = max(0.0, t0_s)
        self.bw = bw_bytes_per_s if bw_bytes_per_s > 0 else float("inf")

    def latency(self, nbytes: int) -> float:
        if nbytes <= 0:
            return 0.0
        return max(self.floor_s, self.t0_s + nbytes / self.bw)

    @classmethod
    def from_samples(cls, name: str, seg: dict) -> "SegmentModel":
        """Derive floor and marginal bandwidth from the raw size sweep.

        Marginal bandwidth comes from the two largest samples so it is not
        polluted by the fixed cost; the floor is the smallest observed time.
        Falls back to the profile's affine fit when there are too few samples.
        """
        rows = sorted(seg.get("samples", []), key=lambda r: r["bytes"])
        if len(rows) >= 2:
            (b0, t0), (b1, t1) = ((rows[-2]["bytes"], rows[-2]["median_s"]),
                                  (rows[-1]["bytes"], rows[-1]["median_s"]))
            if t1 > t0 and b1 > b0:
                bw = (b1 - b0) / (t1 - t0)
                floor = min(r["median_s"] for r in rows)
                # Anchor the affine part on the largest sample so the model
                # reproduces it exactly.
                intercept = max(0.0, t1 - b1 / bw)
                return cls(name, floor, intercept, bw)
        return cls(name, 0.0, seg.get("fixed_overhead_s", 0.0),
                   seg.get("bandwidth_bytes_per_s", 0.0))

    def __repr__(self):
        return (f"SegmentModel({self.name}: floor={self.floor_s * 1e3:.3f}ms, "
                f"t0={self.t0_s * 1e6:.1f}us, bw={self.bw / (1 << 30):.2f}GiB/s)")


class BandwidthModel:
    """Singleton holding the calibrated segments and the byte-count layout."""

    _instance: "BandwidthModel | None" = None

    def __init__(self, profile: dict | None):
        self.enabled = False
        self.local_load: SegmentModel | None = None
        self.local_store: SegmentModel | None = None
        self.page_size_bytes = 0
        self.num_layers = 0
        self.peer_topology = None
        self._seg1_floor_ms = 0.0
        self._seg1_per_blk_ms = 0.0
        if not profile:
            return
        layout = profile.get("layout", {})
        self.page_size_bytes = int(layout.get("page_size_bytes", 0))
        self.num_layers = int(layout.get("num_layers", 0))
        self.peer_topology = profile.get("peer_topology")
        cp = profile.get("control_plane", {})
        _s1 = cp.get("seg1_cross_node", {})
        self._seg1_floor_ms = float(_s1.get("floor_ms", 0) or 0)
        self._seg1_per_blk_ms = float(_s1.get("per_block_ms", 0) or 0)
        # Store completion structure (see store_completion_latency):
        #   poll_granularity_ms: the async_swap loop polls the CUDA event with
        #     `await asyncio.sleep(0.01)` (v6d_object_connector async_swap), so
        #     completion is detected at 10 ms granularity regardless of DMA size.
        #   rank_sync_ms: extra delay from all-TP-rank save_done aggregation
        #     (_do_save_done try_advance) + cross-process notify.
        _st = cp.get("store_completion", {})
        self._poll_granularity_ms = float(_st.get("poll_granularity_ms", 10.0))
        self._rank_sync_ms = float(_st.get("rank_sync_ms", 1.0))
        segs = profile.get("segments", {})
        if "local_load" in segs and "local_store" in segs:
            self.local_load = SegmentModel.from_samples("local_load", segs["local_load"])
            self.local_store = SegmentModel.from_samples("local_store", segs["local_store"])
        if self.page_size_bytes > 0 and self.num_layers > 0 and self.local_load:
            self.enabled = True

    # -- byte counts --------------------------------------------------------

    def seg2_bytes(self, num_blocks: int) -> int:
        """Bytes one worker moves over PCIe for *num_blocks* blocks.

        Per rank, not per object. ``page_size_bytes`` is already the
        post-TP-shard value and each worker only touches its own shard; two
        ranks were measured to run their DMAs concurrently at 1.004x of the
        solo time, so the wall-clock cost is the per-rank figure.
        """
        return max(0, num_blocks) * self.num_layers * self.page_size_bytes

    def latency_for(self, num_blocks: int, swap_in: bool) -> float:
        if not self.enabled or num_blocks <= 0:
            return 0.0
        seg = self.local_load if swap_in else self.local_store
        return seg.latency(self.seg2_bytes(num_blocks)) if seg else 0.0

    def store_completion_latency(self, num_blocks: int) -> float:
        """Wall-clock from store submit (bind) to it being harvestable by
        get_finished, matching production's async save pipeline:

            max(DMA(nblk), poll_granularity) + rank_sync

        - DMA(nblk): the actual GPU->host copy (seg2 store bandwidth).
        - poll_granularity (~10 ms): production polls the CUDA completion event
          with `await asyncio.sleep(0.01)`, so even a 3 ms copy is only seen at
          the next 10 ms tick.  For very large stores DMA dominates and the
          max() switches over automatically.
        - rank_sync (~1 ms): all-TP-rank save_done aggregation + notify.

        Measured (SIMPROBE, 6 blk): copy_submitted->copy_done ~11 ms (=max(3,10)+1),
        constant across 3-17 blk -> poll-granularity bound, as this formula
        reproduces.  env overrides profile for the two structural constants."""
        if not self.enabled or num_blocks <= 0:
            return 0.0
        dma_s = self.latency_for(num_blocks, False)
        _p = os.environ.get("SGLANG_SIMULATOR_STORE_POLL_MS")
        _r = os.environ.get("SGLANG_SIMULATOR_STORE_RANK_SYNC_MS")
        poll_s = (float(_p) if _p is not None else self._poll_granularity_ms) / 1000.0
        rank_s = (float(_r) if _r is not None else self._rank_sync_ms) / 1000.0
        return max(dma_s, poll_s) + rank_s

    # -- control-plane latencies (env overrides profile) --------------------

    @staticmethod
    def _cp_latency(nblocks, env_floor, env_per, prof_floor, prof_per):
        f = os.environ.get(env_floor)
        p = os.environ.get(env_per)
        if f is not None or p is not None:
            floor_ms, per_ms = float(f or 0), float(p or 0)
        else:
            floor_ms, per_ms = prof_floor, prof_per
        if floor_ms <= 0 and per_ms <= 0:
            return 0.0
        return (floor_ms + per_ms * max(0, nblocks)) / 1000.0

    def seg1_latency(self, nblocks: int) -> float:
        """Cross-node fetch (peer v6d -> local v6d), the seg1 transfer.

        PLACEHOLDER: the CPU sim stubs the SRPC/RDMA data path, so this
        cannot be calibrated here; it stays 0 unless the profile/env
        supplies a figure measured on real hardware (collect_remote.py).
        env overrides profile."""
        return self._cp_latency(
            nblocks, "SGLANG_SIMULATOR_SEG1_FLOOR_MS",
            "SGLANG_SIMULATOR_SEG1_PER_BLK_MS",
            self._seg1_floor_ms, self._seg1_per_blk_ms)

    # -- singleton ----------------------------------------------------------

    @classmethod
    def get(cls) -> "BandwidthModel":
        if cls._instance is None:
            cls._instance = cls._load()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    @classmethod
    def _load(cls) -> "BandwidthModel":
        path = os.environ.get(_ENV_PROFILE, "").strip()
        if not path:
            logger.info("[v6d-bw] %s unset: transfer-latency modelling disabled",
                        _ENV_PROFILE)
            return cls(None)
        try:
            with open(path) as f:
                profile = json.load(f)
        except Exception as exc:
            logger.warning("[v6d-bw] cannot read profile %s (%s): "
                           "transfer-latency modelling disabled", path, exc)
            return cls(None)
        model = cls(profile)
        if not model.enabled:
            logger.warning("[v6d-bw] profile %s incomplete: modelling disabled", path)
            return model
        logger.info("[v6d-bw] enabled from %s | layout %d layers x %d B "
                    "(= %d B/block per rank) | topology=%s | %s | %s",
                    path, model.num_layers, model.page_size_bytes,
                    model.seg2_bytes(1), model.peer_topology,
                    model.local_load, model.local_store)
        return model
