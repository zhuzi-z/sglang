import os
import shlex
import subprocess
import sys

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


def _is_llm_engine_serving_process() -> bool:
    """True only for `dashserving run dashllm.worker.llm:worker ...`.

    In the dashllm process tree this matches exactly ONE process: the
    supervisor spawned by dashllm_cmd. The per-engine workers are
    `python -m dashserving.worker` children of dashservingd and never import
    this CLI module; the DASHSERVING_WORKER_PATH marker (set by the daemon
    for those children, see dashllm/core/llm.py) is kept as the more precise
    signal in case that ever changes. Frontend/proxy CLI processes must NOT
    run GPU sweeps.
    """
    if os.getenv("DASHSERVING_WORKER_PATH") == "dashllm.worker.llm:worker":
        return True
    argv = sys.argv[1:]
    return "run" in argv and any("dashllm.worker.llm:worker" in a for a in argv)


def _acquire_sweep_lock(base_dir: str) -> bool:
    """Cross-process guard so the sweep runs at most once per output dir.
    The winner writes its pid; a loser skips. A lock whose pid is dead
    (crashed mid-sweep) is reclaimed. A finished profile also skips the
    sweep, so restarts reuse it -- delete the profile to recalibrate."""
    profile_path = os.path.join(base_dir, "bandwidth_profile.json")
    if os.path.exists(profile_path):
        logger.info(f"[sim-collector] {profile_path} exists, skipping bandwidth calibration")
        return False
    lock_path = profile_path + ".lock"
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                with open(lock_path) as f:
                    pid = int(f.read().strip() or "0")
                if pid <= 0:
                    raise ValueError(f"invalid lock owner pid {pid}")
                os.kill(pid, 0)  # raises ProcessLookupError if the owner died
                logger.info(
                    f"[sim-collector] bandwidth calibration already running in pid {pid}, skipping")
                return False
            except (ValueError, ProcessLookupError):
                try:
                    os.remove(lock_path)  # stale lock: reclaim and retry once
                except OSError:
                    return False
            except OSError:
                return False
    return False


def _run_bw_calibration() -> None:
    """Best-effort bandwidth sweep, run as a child process before the
    supervisor starts dashservingd (GPUs are still free at that point).

    A subprocess is used so the long-lived supervisor never holds a CUDA
    context, and so the collector's argparse can never see this process's
    dashserving CLI args. Never raises: a calibration failure must not take
    down serving.
    """
    if not _is_llm_engine_serving_process():
        return
    try:
        # Bandwidth is a node-level hardware property, so the profile lives
        # at the top of the output dir, not in a per-engine subdir.
        base = os.getenv("SIM_COLLECTOR_OUTPUT_DIR", os.getcwd())
        os.makedirs(base, exist_ok=True)
        if not _acquire_sweep_lock(base):
            return

        out_path = os.path.join(base, "bandwidth_profile.json")
        # Extra sweep options (e.g. --page-size, --num-layers, --blocks) can
        # be injected via SIM_BW_CALIB_ARGS; a later --out there overrides.
        cmd = [sys.executable, "-m",
               "sglang_simulator.collector.bw_calib.collect_bandwidth",
               "--out", out_path]
        cmd += shlex.split(os.getenv("SIM_BW_CALIB_ARGS", ""))

        logger.info(f"[sim-collector] bandwidth calibration -> {out_path} (cmd: {cmd})")
        try:
            subprocess.run(cmd, check=False)
        finally:
            # Only the lock owner reaches here; release so a later restart
            # after a failed sweep can retry (a written profile still skips).
            try:
                os.remove(out_path + ".lock")
            except OSError:
                pass
    except Exception:
        logger.exception("[sim-collector] bandwidth calibration failed; serving continues")


class M_DashservingEntrypointHook(BaseHook):
    HOOK_CLASS_NAME = ""
    HOOK_MODULE_NAME = "dashserving.entrypoints.cli.main"

    @classmethod
    def hook(cls, target):

        original_main = target.main

        def wrapped_main(*args, **kwargs):
            _run_bw_calibration()
            original_main(*args, **kwargs)

        target.main = wrapped_main
