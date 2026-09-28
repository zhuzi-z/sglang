"""
Per-request [V6D HitSource] logger for the v6d daemon (debug module).

Complements the vLLM-side [HitRate] hook (hitrate_hook.py): that one logs how
many prompt tokens hit the V6D external cache as a whole; this one runs inside
the v6d daemon and splits every read batch by hit source:

    [V6D HitSource] request_id=<vllm_req_id> queried=<n> local=<n> p2p=<n> \
        sharedfs=<n> tair_kvcm=<n> miss=<n>

  local      - object already sealed in this node's vineyard tier
  p2p        - object fetched from a remote v6d peer (tracker + SRPC)
  sharedfs   - object fetched from the SharedFS/localfs tier
  tair_kvcm  - object fetched from the Tair KVCM store
  miss       - requested keys not served by any tier in this batch
               (prefix-stop semantics: the un-queried tail counts as miss)

Two classes are hooked, sharing one asyncio-task-local counter bag:
  * TieredVineyardPeer._acquire_tiered_read is wrapped to push a per-call
    counter bag into a contextvar and log it when the call finishes. Only
    Scope.READ acquires reach this method, so store-side gets stay silent.
  * HitRateStats.record_*_hit are wrapped to increment the current bag.
    Contextvars are asyncio-task-local, so per-request attribution stays
    exact even when the daemon interleaves many concurrent acquires.

Note: one vLLM request normally produces two [V6D HitSource] lines — one for
the scheduler lookup (get_num_new_matched_tokens) and one for the worker
KV-load (V6dSwapHandler.swap) — both carry the vLLM request_id. Keys already
staged in the connector's _cached_objs never reach the daemon and are not
counted here; use the vLLM-side [HitRate] line for the full picture.

Disable at runtime with V6D_HITSOURCE_LOG=0.
"""

from __future__ import annotations

import contextvars
import os

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger("debug.hitsource")

_COUNTERS = ("local", "p2p", "sharedfs", "tair_kvcm")

# None -> outside a tracked read; dict -> current read's counter bag.
_hit_bag: contextvars.ContextVar = contextvars.ContextVar(
    "v6d_hitsource_bag", default=None
)


def _enabled() -> bool:
    return os.environ.get("V6D_HITSOURCE_LOG", "1") not in ("0", "false", "False")


class C_TieredVineyardPeerHook(BaseHook):
    """Wrap ``TieredVineyardPeer._acquire_tiered_read`` to log hit sources."""

    HOOK_CLASS_NAME = "TieredVineyardPeer"
    HOOK_MODULE_NAME = "v6d.server.peers.tiered_vineyard.peer"

    @classmethod
    def hook(cls, target) -> None:
        original = target._acquire_tiered_read

        async def wrapped_acquire_tiered_read(self, *args, **kwargs):
            if not _enabled():
                return await original(self, *args, **kwargs)
            object_keys = args[0] if args else kwargs.get("object_keys") or ()
            request_id = kwargs.get("request_id")
            bag = dict.fromkeys(_COUNTERS, 0)
            token = _hit_bag.set(bag)
            error = None
            try:
                return await original(self, *args, **kwargs)
            except Exception as exc:
                error = type(exc).__name__
                raise
            finally:
                _hit_bag.reset(token)
                queried = len(object_keys)
                if queried:
                    hits = sum(bag.values())
                    logger.info(
                        "[V6D HitSource] request_id=%s queried=%d local=%d "
                        "p2p=%d sharedfs=%d tair_kvcm=%d miss=%d%s",
                        request_id,
                        queried,
                        bag["local"],
                        bag["p2p"],
                        bag["sharedfs"],
                        bag["tair_kvcm"],
                        max(queried - hits, 0),
                        f" error={error}" if error else "",
                    )

        target._acquire_tiered_read = wrapped_acquire_tiered_read
        logger.info(
            "TieredVineyardPeer._acquire_tiered_read patched for "
            "[V6D HitSource] logging"
        )


class C_HitRateStatsHook(BaseHook):
    """Wrap ``HitRateStats.record_*_hit`` to attribute hits per read batch."""

    HOOK_CLASS_NAME = "HitRateStats"
    HOOK_MODULE_NAME = "v6d.server.peers.tiered_vineyard.stats"

    @classmethod
    def hook(cls, target) -> None:
        def wrap(name, counter):
            original = getattr(target, name, None)
            if original is None:
                return

            def wrapped(self, size: int = 0):
                bag = _hit_bag.get()
                if bag is not None:
                    bag[counter] += 1
                return original(self, size)

            setattr(target, name, wrapped)

        wrap("record_local_vineyard_hit", "local")
        wrap("record_remote_vineyard_hit", "p2p")
        wrap("record_sharedfs_hit", "sharedfs")
        wrap("record_tair_kvcm_hit", "tair_kvcm")
        logger.info("HitRateStats record_*_hit patched for [V6D HitSource] logging")
