"""Optional startup entry for sglang-simulator hooks.

This module is intentionally placed in the project working directory instead of
site-packages. Python imports ``sitecustomize`` during interpreter startup when
this directory is present in PYTHONPATH, which lets spawned worker processes load
hooks without installing a global .pth file.
"""

from __future__ import annotations

import inspect
import json
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

    def _skip_srpc_init(base_addr: int, size: int, *args, **kwargs) -> None:
        print(
            "[sglang-simulator] skip v6d SRPC init "
            f"base_addr={base_addr} size={size}",
            flush=True,
        )

    transfer.init_srpc_transfer = _skip_srpc_init
    if hasattr(transfer, "init_srpc"):
        transfer.init_srpc = _skip_srpc_init
    if hasattr(transfer, "init_srpc_"):
        transfer.init_srpc_ = _skip_srpc_init

    if hasattr(transfer, "init_mmap") and hasattr(
        transfer, "init_transfer_engine_client"
    ):
        def _init_transfer_engine_client_without_srpc(fd: int, size: int) -> int:
            addr = transfer.init_mmap(fd, size)
            print(
                "[sglang-simulator] init transfer engine client mmap only "
                f"fd={fd} size={size} addr={addr}",
                flush=True,
            )
            return addr

        transfer.init_transfer_engine_client = _init_transfer_engine_client_without_srpc

    try:
        import v6d.lite.common.transfer_engine as transfer_engine
    except Exception:
        transfer_engine = None

    if transfer_engine is not None:
        if hasattr(transfer_engine, "init_srpc"):
            transfer_engine.init_srpc = _skip_srpc_init
        if hasattr(transfer_engine, "init_srpc_"):
            transfer_engine.init_srpc_ = _skip_srpc_init
        if hasattr(transfer_engine, "init_srpc_transfer"):
            transfer_engine.init_srpc_transfer = _skip_srpc_init

    try:
        from v6d.client.peers.vineyard import mmap_manager
    except Exception:
        mmap_manager = None

    if mmap_manager is not None and not getattr(
        mmap_manager.ClientV6dMmapManager,
        "_sglang_simulator_cpu_mmap_hook",
        False,
    ):
        def _create_mmap_without_srpc(self, socket_path: str, is_lazy_strategy: bool):
            from v6d.common.transfer import _vineyard_connect
            from v6d.lite.common.transfer_engine import init_mmap

            fd, map_size, offset, sock = _vineyard_connect(socket_path)
            base_addr = init_mmap(fd, map_size)
            os.close(fd)
            print(
                "[sglang-simulator] create client v6d mmap without SRPC "
                f"socket={socket_path} base_addr={base_addr} size={map_size} "
                f"lazy={is_lazy_strategy}",
                flush=True,
            )
            return mmap_manager.MmapInfo(
                socket_path=socket_path,
                fd=fd,
                base_addr=base_addr,
                map_size=map_size,
                refcount=1,
                socket=sock,
            )

        mmap_manager.ClientV6dMmapManager._create_mmap = _create_mmap_without_srpc
        mmap_manager.ClientV6dMmapManager._sglang_simulator_cpu_mmap_hook = True

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

    try:
        from dashllm.utils import vineyard as dashllm_vineyard
    except Exception:
        return

    if getattr(dashllm_vineyard, "_sglang_simulator_launch_v6d_hook", False):
        return

    original_launch_v6d = dashllm_vineyard.launch_v6d

    def _patched_launch_v6d(*args, **kwargs):
        envs_to_update = dict(kwargs.get("envs_to_update") or {})
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        if current_pythonpath:
            existing_pythonpath = envs_to_update.get("PYTHONPATH", "")
            envs_to_update["PYTHONPATH"] = (
                current_pythonpath
                if not existing_pythonpath
                else f"{current_pythonpath}:{existing_pythonpath}"
            )
        envs_to_update["SGLANG_SIMULATOR_ENABLE_V6D_IPC_HOOK"] = "1"
        kwargs["envs_to_update"] = envs_to_update
        print(
            "[sglang-simulator] patch dashllm launch_v6d envs: "
            f"{sorted(envs_to_update)}",
            flush=True,
        )
        return original_launch_v6d(*args, **kwargs)

    dashllm_vineyard.launch_v6d = _patched_launch_v6d
    dashllm_vineyard._sglang_simulator_launch_v6d_hook = True

    try:
        import dashllm.core.backend._backend_vllm as backend_vllm
        if getattr(backend_vllm, "launch_v6d", None) is original_launch_v6d:
            backend_vllm.launch_v6d = _patched_launch_v6d
            print(
                "[sglang-simulator] patch dashllm backend_vllm.launch_v6d",
                flush=True,
            )

        backend_cls = getattr(backend_vllm, "_LLMBackend4vLLM", None)
        if backend_cls is not None and not getattr(
            backend_cls,
            "_sglang_simulator_kv_transfer_params_hook",
            False,
        ):
            original_generate = backend_cls.generate

            def _decode_kv_transfer_params(value):
                if value is None:
                    return {}
                if isinstance(value, dict):
                    return dict(value)
                if isinstance(value, str):
                    try:
                        return _decode_kv_transfer_params(json.loads(value))
                    except Exception:
                        return {}
                if isinstance(value, (list, tuple)):
                    merged = {}
                    for item in value:
                        merged.update(_decode_kv_transfer_params(item))
                    return merged
                return {}

            def _patched_generate(self, model, **kwargs):
                kv_params = _decode_kv_transfer_params(
                    kwargs.get("kv_transfer_params")
                )
                for key in (
                    "do_remote_decode",
                    "do_remote_prefill",
                    "ali_llumnix_disagg",
                    "remote_host",
                    "remote_port",
                ):
                    if key in kwargs and key not in kv_params:
                        kv_params[key] = kwargs[key]
                if kv_params:
                    encoder_extra_args = dict(kwargs.get("encoder_extra_args") or {})
                    merged = _decode_kv_transfer_params(
                        encoder_extra_args.get("kv_transfer_params")
                    )
                    merged.update(kv_params)
                    encoder_extra_args["kv_transfer_params"] = merged
                    kwargs["encoder_extra_args"] = encoder_extra_args
                    print(
                        "[sglang-simulator] inject kv_transfer_params into "
                        f"sampling extra_args: {sorted(merged)}",
                        flush=True,
                    )
                return original_generate(self, model, **kwargs)

            backend_cls.generate = _patched_generate
            backend_cls._sglang_simulator_kv_transfer_params_hook = True
            print(
                "[sglang-simulator] patch dashllm _LLMBackend4vLLM.generate "
                "kv_transfer_params passthrough",
                flush=True,
            )

        try:
            import dashllm.core.backend.engine._vllm_v1 as vllm_v1
        except Exception:
            vllm_v1 = None

        engine_cls = getattr(vllm_v1, "vLLMEngine", None) if vllm_v1 else None
        if engine_cls is not None and not getattr(
            engine_cls,
            "_sglang_simulator_vllm_engine_kv_params_hook",
            False,
        ):
            original_engine_generate = engine_cls.generate
            original_engine_generate_impl = getattr(engine_cls, "_generate_impl", None)

            def _force_kv_params(kwargs):
                request_context = kwargs.get("request_context")
                request_id = getattr(request_context, "request_id", "")
                force_ids = ("native-v6d", "rdecode", "pprefill")
                if any(token in request_id for token in force_ids):
                    sampling_params = dict(kwargs.get("sampling_params") or {})
                    extra_args = dict(sampling_params.get("extra_args") or {})
                    kv_params = dict(extra_args.get("kv_transfer_params") or {})
                    kv_params.setdefault("do_remote_decode", True)
                    extra_args["kv_transfer_params"] = kv_params
                    sampling_params["extra_args"] = extra_args
                    kwargs["sampling_params"] = sampling_params
                    print(
                        "[sglang-simulator] force vLLMEngine kv_transfer_params "
                        f"for req={request_id}: {sorted(kv_params)}",
                        flush=True,
                    )
                return kwargs

            def _patched_engine_generate(self, *args, **kwargs):
                kwargs = _force_kv_params(kwargs)
                yield from original_engine_generate(self, *args, **kwargs)

            engine_cls.generate = _patched_engine_generate

            if original_engine_generate_impl is not None:
                def _patched_engine_generate_impl(self, *args, **kwargs):
                    kwargs = _force_kv_params(kwargs)
                    yield from original_engine_generate_impl(self, *args, **kwargs)

                engine_cls._generate_impl = _patched_engine_generate_impl

            engine_cls._sglang_simulator_vllm_engine_kv_params_hook = True
            print(
                "[sglang-simulator] patch dashllm vLLMEngine.generate "
                "kv_transfer_params passthrough",
                flush=True,
            )
    except Exception:
        pass


def _call_init_hook(init_hook) -> None:
    signature = inspect.signature(init_hook)
    if "force" in signature.parameters:
        init_hook(force=True)
    else:
        init_hook()


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

        _call_init_hook(init_vllm_hook)

    if enable_sglang:
        from sglang_simulator.simulation.sglang.startup import init_hook as init_sglang_hook

        _call_init_hook(init_sglang_hook)


_install_hooks()
