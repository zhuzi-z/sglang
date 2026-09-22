#!/usr/bin/env python3
"""Upload the local collector output to OSS.

Function API (no CLI). Equivalent to the manual one-liner::

    ossutil cp -r /hisim/collection/ <path>/<OUT_DIR> \\
        -i <ak> -k <sk> -e <endpoint>

OSS config is passed as a JSON string in the ``SIM_COLLECTOR_OSS_CONFIG``
environment variable::

    {"ak": "...", "sk": "...", "endpoint": "oss-cn-hangzhou.aliyuncs.com",
     "path": "oss://kunlun-hisim/hisim_collection"}

Because many instances upload into the same ``path``, the machine hostname is
appended as a per-instance sub-directory so one instance never overwrites
another's data.
"""
import json
import os
import shlex
import socket
import subprocess

DEF_SRC_DIR = "/hisim/collection/"
DEF_OSSUTIL = "ossutil"
OSS_CONFIG_ENV = "SIM_COLLECTOR_OSS_CONFIG"

_REQUIRED_KEYS = ("ak", "sk", "endpoint", "path")


class UploadError(RuntimeError):
    """Raised for config or ossutil execution problems."""


def default_out_dir():
    """Per-instance OSS sub-directory: hostname keeps instances from colliding."""
    return socket.gethostname()


def load_config(env=None):
    """Parse and validate the OSS config JSON from ``SIM_COLLECTOR_OSS_CONFIG``."""
    raw = (env if env is not None else os.environ).get(OSS_CONFIG_ENV)
    if not raw:
        raise UploadError(
            "missing {} env (JSON string with ak/sk/endpoint/path)".format(
                OSS_CONFIG_ENV))
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UploadError(
            "invalid {} JSON: {}".format(OSS_CONFIG_ENV, exc)) from exc
    if not isinstance(cfg, dict):
        raise UploadError("{} must be a JSON object".format(OSS_CONFIG_ENV))
    missing = [k for k in _REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        raise UploadError(
            "{} missing keys: {}".format(OSS_CONFIG_ENV, ", ".join(missing)))
    return cfg


def build_command(cfg, src, out_dir, dest_name=None, ossutil=DEF_OSSUTIL):
    """Build the ossutil argv and the resolved destination URI."""
    base = "{}/{}".format(str(cfg["path"]).rstrip("/"), out_dir)
    creds = ["-i", cfg["ak"], "-k", cfg["sk"], "-e", cfg["endpoint"]]
    if os.path.isdir(src):
        return [ossutil, "cp", "-r", src, base] + creds, base
    dst = "{}/{}".format(base, dest_name or os.path.basename(src))
    return [ossutil, "cp", src, dst] + creds, dst


def _relative_dest_name(src, root_dir):
    """Return an OSS-safe path for ``src`` relative to the collection root."""
    src_path = os.path.abspath(src)
    root_path = os.path.abspath(root_dir)
    try:
        relative = os.path.relpath(src_path, root_path)
    except ValueError as exc:
        raise UploadError(
            "source and collection root are on different filesystems"
        ) from exc
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        raise UploadError(
            "source {} is outside collection root {}".format(src, root_dir)
        )
    return relative.replace(os.sep, "/")


def upload(src=DEF_SRC_DIR, out_dir=None, dest_name=None, root_dir=None,
           config=None, ossutil=DEF_OSSUTIL):
    """Mirror ``src`` (a file or directory) to OSS.

    Config comes from the JSON env var by default. For a file, its path relative
    to ``root_dir`` is preserved below ``<path>/<out_dir>``. ``dest_name`` can
    explicitly override that relative object key. Raises ``UploadError`` on
    config or execution problems.
    """
    cfg = config if config is not None else load_config()
    if out_dir is None:
        out_dir = default_out_dir()
    if not os.path.exists(src):
        raise UploadError("source not found: {}".format(src))
    if os.path.isfile(src) and dest_name is None:
        if root_dir is None:
            root_dir = os.getenv("SIM_COLLECTOR_OUTPUT_DIR", DEF_SRC_DIR)
        dest_name = _relative_dest_name(src, root_dir)

    cmd, dst = build_command(cfg, src, out_dir, dest_name, ossutil)

    # Redact both access-key fields when echoing the command for logs.
    printable = list(cmd)
    for flag in ("-i", "-k"):
        secret_index = printable.index(flag)
        printable[secret_index + 1] = "***"
    print("uploading {} -> {}".format(src, dst))
    print("+ " + " ".join(shlex.quote(c) for c in printable))

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise UploadError("ossutil exited with code {}".format(proc.returncode))
    print("upload done: {}".format(dst))
    return dst
