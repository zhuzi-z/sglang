import json
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


def _collect_gpu_models(profile_path: str | None = None) -> list[str]:
    """Read GPU models without creating a CUDA context in this process."""
    if profile_path and os.path.isfile(profile_path):
        try:
            with open(profile_path) as f:
                profile = json.load(f)
            gpu_name = profile.get("gpu_name")
            gpu_count = int(profile.get("gpu_count", 0))
            if gpu_name and gpu_count > 0:
                return [str(gpu_name)] * gpu_count
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                f"[sim-collector] failed to read GPU metadata from profile: {exc}"
            )

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        logger.warning(f"[sim-collector] failed to query GPU models: {exc}")
        return []
    if result.returncode != 0:
        logger.warning(
            f"[sim-collector] nvidia-smi GPU query failed: {result.stderr.strip()}"
        )
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _write_env_metadata(base_dir: str, profile_path: str | None = None) -> str:
    """Write sanitized process environment and GPU model data to env.json."""
    environment = {
        key: value
        for key, value in sorted(os.environ.items())
        if not key.upper().startswith("SIM_COLLECTOR")
    }
    gpu_models = _collect_gpu_models(profile_path)
    path = os.path.join(base_dir, "env.json")
    with open(path, "w") as f:
        json.dump(
            {
                "env": environment,
                "gpu": {
                    "count": len(gpu_models),
                    "models": gpu_models,
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )
    os.chmod(path, 0o600)
    logger.info(f"[sim-collector] environment metadata saved to {path}")
    return path


def _upload_collector_file(path: str, description: str) -> None:
    """Best-effort upload of one completed node-level collector file."""
    if os.getenv("SIM_COLLECTOR_UPLOAD", "1").lower() in (
        "0", "false", "no", "off"
    ):
        return
    if not os.path.isfile(path):
        return
    try:
        from sglang_simulator.collector.uploader.upload import upload

        upload(path, root_dir=os.path.dirname(path))
    except Exception:
        logger.exception(
            f"[sim-collector] {description} upload failed; serving continues"
        )


def _collect_common_info() -> None:
    """Collect node-level environment, GPU, and bandwidth information.

    The bandwidth sweep runs as a child process before the supervisor starts
    dashservingd (GPUs are still free at that point). A subprocess is used so
    the long-lived supervisor never holds a CUDA context, and so the collector's
    argparse can never see this process's dashserving CLI args. Never raises:
    collection failures must not take down serving.
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

        # argparse keeps the last --out value, so resolve the effective path for
        # the post-process upload when SIM_BW_CALIB_ARGS overrides it.
        upload_path = out_path
        for index, arg in enumerate(cmd):
            if arg == "--out" and index + 1 < len(cmd):
                upload_path = cmd[index + 1]
            elif arg.startswith("--out="):
                upload_path = arg.split("=", 1)[1]

        logger.info(f"[sim-collector] bandwidth calibration -> {out_path} (cmd: {cmd})")
        try:
            result = subprocess.run(cmd, check=False)

            # Capture the same process environment that launches serving. The
            # OSS config and all other SIM_COLLECTOR* values are excluded before
            # writing, so credentials cannot enter env.json. Prefer GPU metadata
            # from the completed profile; nvidia-smi is the fallback.
            profile_path = upload_path if result.returncode == 0 else None
            try:
                env_path = _write_env_metadata(base, profile_path)
                _upload_collector_file(env_path, "environment metadata")
            except Exception:
                logger.exception(
                    "[sim-collector] environment metadata collection failed; "
                    "serving continues"
                )

            if result.returncode == 0:
                # The subprocess has exited, so the profile is closed and
                # complete before the single-file upload starts.
                _upload_collector_file(upload_path, "bandwidth profile")
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
            _collect_common_info()
            original_main(*args, **kwargs)

        target.main = wrapped_main
