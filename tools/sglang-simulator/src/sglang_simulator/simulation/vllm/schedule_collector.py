"""Shared schedule-batch / request collector for the vLLM real-GPU and CPU-sim paths.

Both deployment modes populate the SAME in-memory buffers and dump identical-schema
files on /stop_profile, so simulation output can be diffed directly against real-GPU
output:

- TP{rank}.schedule_batch.jsonl (one line per scheduler iteration):
    {"forward_mode": 1|2,
     "iter_latency": float,
     "request_infos": [{"extend_input_len": int, "prefix_indices_len": int}, ...]}
- TP{rank}.requests.jsonl (one line per request): request-level stats used for
    prefix-hit comparison.

`iter_latency` semantics differ by mode (real = measured GPU time, sim = predicted).
The file is the comparison artifact; the caller must know which side is ground-truth.

Design constraints that shaped this module:
- `install_class_hooks` applies only ONE hook per class name (first match wins), so the
  dump hooks are split across EngineCore / Worker to avoid clashing with the existing
  sim Worker/Scheduler hooks.
- In real-GPU multiprocessing, the schedule_batch buffer lives in each worker process
  while the request buffer lives in the engine (scheduler) process. Hence schedule_batch
  is dumped from Worker.profile (rank-0 gated) and requests from EngineCore.profile.
- In CPU sim (multiprocessing off, TP=1) both buffers live in one process; the single
  EngineCore.profile hook dumps both.
"""

import json
import os

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager.env import Envs
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")

# Process-local buffers. Each real-GPU worker/engine process owns its own copy; the
# rank-0 gate on the schedule_batch dump keeps the output to a single file.
SCHEDULE_INFOS: list[dict] = []
REQUEST_INFOS: dict[str, dict] = {}

# "sim" or "real"; set by startup.init_hook so the shared EngineCore dump hook knows
# whether the schedule_batch buffer also lives in this (engine) process.
_MODE: str = "sim"


def set_mode(mode: str) -> None:
    global _MODE
    _MODE = (mode or "sim").strip().lower()


def get_mode() -> str:
    return _MODE


def record_iteration(reqs, forward_mode: int, iter_latency: float) -> None:
    """Append one scheduler iteration.

    reqs: iterable of (extend_input_len, prefix_indices_len) integer pairs.
    """
    SCHEDULE_INFOS.append(
        {
            "forward_mode": int(forward_mode),
            "iter_latency": float(iter_latency),
            "request_infos": [
                {"extend_input_len": int(e), "prefix_indices_len": int(p)}
                for e, p in reqs
            ],
        }
    )


def record_request(rid: str, **fields) -> None:
    """Merge request-level fields into REQUEST_INFOS[rid]."""
    info = REQUEST_INFOS.setdefault(rid, {"rid": rid})
    info.update(fields)


def _rank_suffix(rank) -> str:
    return f"TP{rank}"


def dump_schedule_batch(rank=0, output_dir=None) -> None:
    output_dir = output_dir or Envs.output_dir()
    path = os.path.join(output_dir, f"{_rank_suffix(rank)}.schedule_batch.jsonl")
    with open(path, "w") as f:
        for item in SCHEDULE_INFOS:
            f.write(json.dumps(item, default=str) + "\n")
    logger.info(
        "[collector] dumped %d iterations -> %s", len(SCHEDULE_INFOS), path
    )
    SCHEDULE_INFOS.clear()


def dump_requests(rank=0, output_dir=None) -> None:
    output_dir = output_dir or Envs.output_dir()
    path = os.path.join(output_dir, f"{_rank_suffix(rank)}.requests.jsonl")
    with open(path, "w") as f:
        for item in REQUEST_INFOS.values():
            f.write(json.dumps(item, default=str) + "\n")
    logger.info(
        "[collector] dumped %d requests -> %s", len(REQUEST_INFOS), path
    )
    REQUEST_INFOS.clear()


class C_EngineProfileDumpHook(BaseHook):
    """Dump on EngineCore.profile(is_start=False).

    - Always dumps requests (buffer lives in the engine/scheduler process).
    - Also dumps schedule_batch when running in sim mode (single process, TP=1, buffer
      lives in this same process). In real mode the schedule_batch buffer lives in the
      worker processes and is dumped by C_WorkerProfileDumpHook instead, so we skip it
      here to avoid writing an empty file / racing on the same path.
    """

    HOOK_CLASS_NAME = "EngineCore"
    HOOK_MODULE_NAME = "vllm.v1.engine.core"

    @classmethod
    def hook(cls, target):
        original_profile = target.profile

        def wrapped_profile(self, *args, **kwargs):
            # EngineCore.profile(self, is_start=True, profile_prefix=None); callers pass
            # profile_prefix positionally, so accept/forward everything.
            is_start = args[0] if args else kwargs.get("is_start", True)
            # Preserve native profiler behavior (and forward to workers).
            try:
                ret = original_profile(self, *args, **kwargs)
            except Exception:
                logger.debug(
                    "[collector] original EngineCore.profile failed; ignoring",
                    exc_info=True,
                )
                ret = None
            if not is_start:
                try:
                    if _MODE == "sim":
                        dump_schedule_batch(rank=0)
                    dump_requests(rank=0)
                except Exception:
                    logger.warning(
                        "[collector] EngineCore dump failed", exc_info=True
                    )
            return ret

        target.profile = wrapped_profile


class C_WorkerProfileDumpHook(BaseHook):
    """Real-GPU only: dump schedule_batch from the worker process on Worker.profile.

    Rank-0 gated so multi-TP runs produce a single TP0.schedule_batch.jsonl (all TP
    ranks receive the same broadcast SchedulerOutput, so their batch composition is
    identical; only rank 0 needs to be written).
    """

    HOOK_CLASS_NAME = "Worker"
    HOOK_MODULE_NAME = r"vllm\.v1\.worker\.(gpu_worker|worker)"
    REGEX = True

    @classmethod
    def hook(cls, target):
        original_profile = getattr(target, "profile", None)

        def wrapped_profile(self, *args, **kwargs):
            is_start = args[0] if args else kwargs.get("is_start", True)
            ret = None
            if original_profile is not None:
                try:
                    ret = original_profile(self, *args, **kwargs)
                except Exception:
                    logger.debug(
                        "[collector] original Worker.profile failed; ignoring",
                        exc_info=True,
                    )
            if not is_start and getattr(self, "rank", 0) == 0:
                try:
                    dump_schedule_batch(rank=getattr(self, "rank", 0))
                except Exception:
                    logger.warning(
                        "[collector] Worker dump failed", exc_info=True
                    )
            return ret

        target.profile = wrapped_profile
