"""Shared emit path for the ``[lat-probe]`` latency-probe log lines.

Two probes feed the same log prefix from two different processes:

- ``simulation/vllm/dashllm/latency_probe.py`` runs in the ``dashserving.worker``
  process and times the segments upstream of EngineCore.
- ``simulation/vllm/engine_core_pipeline.py`` runs in the EngineCore process and
  emits one line per request at first-token time.

They share the env gate and the dashlog fallback so the two sides can never
drift apart (e.g. one enabled while the other is not), and so a single grep for
``lat-probe`` collects the whole picture.

Lines are joined offline on the request id. vLLM's request id is turbo's id plus
a ``-<suffix>`` segment, so ``rid.rsplit("-", 1)[0]`` recovers the turbo id.
"""

import os

from sglang_simulator.utils import get_logger

logger = get_logger()

ENV_DISABLE = "SGLANG_SIMULATOR_LATENCY_PROBE"

PREFIX = "[lat-probe]"

# Sentinel for "this timestamp was not available", kept as a float so the
# ``:.3f`` formatting used by callers stays valid. Never silently coerced to 0,
# which would read as "no latency" instead of "not measured".
MISSING = -1.0


def enabled() -> bool:
    """On by default; set ``SGLANG_SIMULATOR_LATENCY_PROBE=0`` to silence.

    Defaulting to on follows ``startup.py``'s stated convention that reaching
    the hooks already implies simulation mode, so no opt-in is warranted. It
    also removes the worst failure mode: an opt-in flag forgotten at deploy
    time yields no data at all, and re-deploying costs a full rolling restart
    (this deployment's graceful-shutdown window is 20 minutes per instance)
    before anyone notices the probe never ran. The kill switch stays for the
    case where the per-request lines become too noisy.
    """
    return os.environ.get(ENV_DISABLE, "").strip().lower() not in ("0", "false", "no")


def emit(line: str) -> None:
    """Log through dashlog when present, else the simulator logger.

    dashlog is the native channel inside dashllm-launched processes (already
    initialised there), which lands these lines in the same stream as dashllm's
    own per-request logs and makes them directly correlatable. The simulator
    logger is the fallback for standalone vllm-serve runs.
    """
    try:
        import dashlog
        dashlog.info(line)
    except Exception:
        logger.info(line)
