"""Optional startup entry for sglang-simulator hooks.

This module is intentionally placed in the project working directory instead of
site-packages. Python imports ``sitecustomize`` during interpreter startup when
this directory is present in PYTHONPATH, which lets spawned worker processes load
hooks without installing a global .pth file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _ensure_src_on_path() -> None:
    src_dir = Path(__file__).resolve().parent / "src"
    src_dir_text = str(src_dir)
    if src_dir.exists() and src_dir_text not in sys.path:
        sys.path.insert(0, src_dir_text)


def _install_v6d_ipc_hook() -> None:
    """Patch v6d startup so CPU-only environments can expose vineyard IPC."""
    try:
        import v6d.common.transfer as transfer
        from v6d.server.peers.vineyard.peer import VineyardPeer
    except Exception:
        return

    def _skip_srpc_transfer(base_addr: int, size: int) -> None:
        print(
            "[sglang-simulator] skip v6d init_srpc_transfer "
            f"base_addr={base_addr} size={size}",
            flush=True,
        )

    transfer.init_srpc_transfer = _skip_srpc_transfer

    if getattr(VineyardPeer, "_sglang_simulator_cpu_ipc_hook", False):
        return

    original_init = VineyardPeer.__init__

    def _patched_init(
        self,
        argc=0,
        argv=None,
        tracker_url=None,
        tracker_key_prefix=None,
        lazy_load=True,
    ):
        patched_argv = list(argv) if argv is not None else None
        if patched_argv is not None:
            has_rpc_flag = any(
                arg.startswith("-rpc=") or arg.startswith("--rpc=")
                for arg in patched_argv
            )
            if not has_rpc_flag:
                patched_argv.append("-rpc=false")
                argc = len(patched_argv)
        return original_init(
            self,
            argc,
            patched_argv,
            tracker_url,
            tracker_key_prefix,
            lazy_load,
        )

    VineyardPeer.__init__ = _patched_init
    VineyardPeer._sglang_simulator_cpu_ipc_hook = True


def _install_hooks() -> None:
    enable_all = _enabled("SGLANG_SIMULATOR_ENABLE_HOOK")
    enable_vllm = enable_all or _enabled("SGLANG_SIMULATOR_ENABLE_VLLM_HOOK")
    enable_sglang = enable_all or _enabled("SGLANG_SIMULATOR_ENABLE_SGLANG_HOOK")
    enable_v6d_ipc = enable_all or _enabled("SGLANG_SIMULATOR_ENABLE_V6D_IPC_HOOK")

    if not enable_vllm and not enable_sglang and not enable_v6d_ipc:
        return

    _ensure_src_on_path()

    if enable_v6d_ipc:
        _install_v6d_ipc_hook()

    if enable_vllm:
        from sglang_simulator.simulation.vllm.startup import init_hook as init_vllm_hook

        init_vllm_hook(force=True)

    if enable_sglang:
        from sglang_simulator.simulation.sglang.startup import init_hook as init_sglang_hook

        init_sglang_hook(force=True)


_install_hooks()
