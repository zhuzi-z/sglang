"""KV-cache observability debug hooks for real deployments.

This module mirrors the ``collector`` package layout but is a standalone,
read-only observability capability: instead of collecting/exporting data it
only logs per-request KV-cache hit information, so it is safe to enable in a
real (non-simulation) deployment.

Three independent hook groups are provided:
  * ``vllm_hook`` — [HitRate] per request, splitting prompt tokens into local
    radix hits vs external (V6D) hits (patches vLLM v1 ``Scheduler``).
  * ``v6d_hook``  — [V6D HitSource] per read batch inside the v6d daemon,
    splitting V6D hits into local / p2p / sharedfs / tair_kvcm / miss
    (patches ``TieredVineyardPeer`` and ``HitRateStats``).
  * ``dashserving`` — ``GET /server_info?config_format=text|json`` on the
    dashserving worker control port, exporting the resolved ``VllmConfig``
    (patches the worker ``ControlHandler``).

Enable at interpreter startup with the ``SIM_DEBUG_ENABLE`` environment
variable (see ``sglang_simulator._pth_bootstrap``). Silence individual groups
at runtime with ``V6D_HITRATE_LOG=0`` / ``V6D_HITSOURCE_LOG=0``.
"""

from sglang_simulator.debug.dashserving import C_DashservingControlHandlerHook
from sglang_simulator.debug.v6d_hook import (
    C_HitRateStatsHook,
    C_TieredVineyardPeerHook,
)
from sglang_simulator.debug.vllm_hook import C_HitRateHook


__all__ = [
    "C_HitRateHook",
    "C_TieredVineyardPeerHook",
    "C_HitRateStatsHook",
    "C_DashservingControlHandlerHook",
]
