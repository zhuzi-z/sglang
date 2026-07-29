"""V6D IPC hook: bypass SRPC hardware dependency in vLLM process.

Extracted from startup.py during V6D decoupling.
Patches v6d client modules to skip SRPC BAREX init (no RDMA on CPU).
"""
import logging
import os


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_enabled(*names: str) -> bool:
    return any(os.environ.get(name, "").strip().lower() in _TRUE_VALUES for name in names)


def _env_default_enabled(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in _FALSE_VALUES


def _install_v6d_ipc_hook() -> None:
    """Patch v6d startup so CPU-only package/PTH deployments can expose IPC."""
    import logging
    logger = logging.getLogger("sglang_simulator")
    logger.info("[v6d_ipc_hook] begin")

    try:
        import v6d.common.transfer as transfer
    except Exception as exc:
        logger.exception("[v6d_ipc_hook] import v6d.common.transfer failed: %r", exc)
        return

    def _skip_srpc_init(base_addr: int, size: int, *args, **kwargs) -> None:
        logger.info(
            "[v6d_ipc_hook] skip SRPC init base_addr=%s size=%s",
            base_addr,
            size,
        )
        return None

    transfer.init_srpc_transfer = _skip_srpc_init
    if hasattr(transfer, "init_srpc"):
        transfer.init_srpc = _skip_srpc_init
    if hasattr(transfer, "init_srpc_"):
        transfer.init_srpc_ = _skip_srpc_init
    logger.info("[v6d_ipc_hook] patched v6d.common.transfer SRPC entrypoints")

    if hasattr(transfer, "init_mmap") and hasattr(
        transfer, "init_transfer_engine_client"
    ):
        def _init_transfer_engine_client_without_srpc(fd: int, size: int) -> int:
            logger.info(
                "[v6d_ipc_hook] init transfer engine client mmap only fd=%s size=%s",
                fd,
                size,
            )
            return transfer.init_mmap(fd, size)

        transfer.init_transfer_engine_client = _init_transfer_engine_client_without_srpc
        logger.info("[v6d_ipc_hook] patched transfer engine client to mmap only")

    try:
        import v6d.lite.common.transfer_engine as transfer_engine
    except Exception as exc:
        logger.info("[v6d_ipc_hook] import lite transfer_engine skipped: %r", exc)
        transfer_engine = None

    if transfer_engine is not None:
        if hasattr(transfer_engine, "init_srpc"):
            transfer_engine.init_srpc = _skip_srpc_init
        if hasattr(transfer_engine, "init_srpc_"):
            transfer_engine.init_srpc_ = _skip_srpc_init
        if hasattr(transfer_engine, "init_srpc_transfer"):
            transfer_engine.init_srpc_transfer = _skip_srpc_init
        logger.info("[v6d_ipc_hook] patched lite transfer_engine SRPC entrypoints")

    if _env_enabled("SGLANG_SIMULATOR_V6D_IPC_PATCH_MMAP_MANAGER"):
        try:
            from v6d.client.peers.vineyard import mmap_manager
        except Exception as exc:
            logger.info("[v6d_ipc_hook] import mmap_manager skipped: %r", exc)
            mmap_manager = None

        if mmap_manager is not None and not getattr(
            mmap_manager.ClientV6dMmapManager,
            "_sglang_simulator_cpu_mmap_hook",
            False,
        ):
            def _create_mmap_without_srpc(
                self,
                socket_path: str,
                is_lazy_strategy: bool,
            ):
                from v6d.common.transfer import _vineyard_connect
                from v6d.lite.common.transfer_engine import init_mmap

                fd, map_size, offset, sock = _vineyard_connect(socket_path)
                base_addr = init_mmap(fd, map_size)
                os.close(fd)
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
            logger.info("[v6d_ipc_hook] patched ClientV6dMmapManager._create_mmap")
    else:
        logger.info("[v6d_ipc_hook] mmap_manager patch disabled; enable SGLANG_SIMULATOR_V6D_IPC_PATCH_MMAP_MANAGER=1 to test it")

    if _env_default_enabled("SGLANG_SIMULATOR_V6D_IPC_PATCH_VINEYARD_PEER"):
        try:
            from v6d.server.peers.vineyard.peer import VineyardPeer
        except Exception as exc:
            logger.info("[v6d_ipc_hook] import VineyardPeer skipped: %r", exc)
            VineyardPeer = None

        if VineyardPeer is not None and not getattr(
            VineyardPeer,
            "_sglang_simulator_cpu_ipc_hook",
            False,
        ):
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
            logger.info("[v6d_ipc_hook] patched VineyardPeer.__init__")
    else:
        logger.info("[v6d_ipc_hook] VineyardPeer patch disabled by env")

    if _env_enabled("SGLANG_SIMULATOR_V6D_IPC_PATCH_DASHLLM_LAUNCH"):
        try:
            from dashllm.utils import vineyard as dashllm_vineyard
        except Exception as exc:
            logger.info("[v6d_ipc_hook] import dashllm.utils.vineyard skipped: %r", exc)
            dashllm_vineyard = None

        if dashllm_vineyard is not None and not getattr(
            dashllm_vineyard,
            "_sglang_simulator_launch_v6d_hook",
            False,
        ):
            original_launch_v6d = dashllm_vineyard.launch_v6d

            def _patched_launch_v6d(*args, **kwargs):
                envs_to_update = dict(kwargs.get("envs_to_update") or {})
                envs_to_update["SGLANG_SIMULATOR_ENABLE_V6D_IPC_HOOK"] = "1"
                kwargs["envs_to_update"] = envs_to_update
                return original_launch_v6d(*args, **kwargs)

            dashllm_vineyard.launch_v6d = _patched_launch_v6d
            dashllm_vineyard._sglang_simulator_launch_v6d_hook = True
            _install_dashllm_kv_transfer_hook(original_launch_v6d)
            logger.info("[v6d_ipc_hook] patched dashllm.utils.vineyard.launch_v6d")
    else:
        logger.info("[v6d_ipc_hook] dashllm launch_v6d patch disabled; enable SGLANG_SIMULATOR_V6D_IPC_PATCH_DASHLLM_LAUNCH=1 to test it")

    logger.info("[v6d_ipc_hook] completed")

# Native V6D control-plane mode keeps the real vLLM connector stack
# (HybridConnector -> V6dObjectKVTBackend -> V6dObjectBackend/PBackend ->
# V6dObjectConnectorScheduler/V6dObjectManager) and installs only runtime
# hijack hooks for CPU-only execution.  The default CPU simulation mode still
# uses MockHybridConnector + V6DCacheStorage(etcd) for scheduling parity.
#
# This file must not patch DashServing/vLLM source files on disk.  All behavior
# changes are installed through class hooks, sitecustomize, and monkey patches
# gated by explicit environment variables.

