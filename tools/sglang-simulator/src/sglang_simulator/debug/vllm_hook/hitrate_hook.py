"""
Real-workload [HitRate] logger for vLLM v1 (debug module).

Logs one line per newly-scheduled request, mirroring the simulator's
scheduler.py:

    [HitRate] rid=<req_id> input_len=<prompt_tokens> local_hit=<local_radix> ext_hit=<external>

Where the prompt tokens of each request are split into:
  * local_hit - tokens served from this node's local radix cache
  * ext_hit   - tokens served from the external (V6D) cache

Unlike the collector this hook only prints; it collects nothing and keeps no
state, so it is safe to enable in a real deployment for ad-hoc observability.
Disable at runtime with V6D_HITRATE_LOG=0.
"""

from __future__ import annotations

import os

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger("debug.hitrate")


def _enabled() -> bool:
    return os.environ.get("V6D_HITRATE_LOG", "1") not in ("0", "false", "False")


class C_HitRateHook(BaseHook):
    """Wrap vLLM v1 ``Scheduler.schedule`` to log per-request hit rates."""

    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "vllm.v1.core.sched.scheduler"

    @classmethod
    def hook(cls, target) -> None:
        original_schedule = target.schedule

        def wrapped_schedule(self):
            scheduler_output = original_schedule(self)
            if not _enabled():
                return scheduler_output
            for new_req in getattr(
                scheduler_output, "scheduled_new_reqs", None
            ) or ():
                rid = getattr(new_req, "req_id", None)
                request = self.requests.get(rid) if rid is not None else None
                if request is None:
                    continue
                input_len = int(getattr(request, "num_prompt_tokens", 0) or 0)
                cached = max(int(getattr(request, "num_cached_tokens", 0) or 0), 0)
                ext = int(getattr(request, "num_external_computed_tokens", 0) or 0)
                logger.info(
                    "[HitRate] rid=%s input_len=%d local_hit=%d ext_hit=%d",
                    rid,
                    input_len,
                    cached - ext,
                    ext,
                )
            return scheduler_output

        target.schedule = wrapped_schedule
        logger.info("vLLM v1 Scheduler.schedule patched for [HitRate] logging")
