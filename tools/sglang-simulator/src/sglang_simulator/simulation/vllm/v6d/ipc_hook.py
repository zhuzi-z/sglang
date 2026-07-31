"""V6D IPC hook: CPU-only v6d client support in the vLLM process.

Simplified: the v6d SRPC engine (``libsrpc_stream_engine.so``) natively
supports ``SRPC_STREAM_DISABLE_RDMA=1``, which skips loading
``libsrpc_barex_bridge.so`` (the component that pulls in ``libcuda.so.1``,
``libibverbs`` and ``libmlx5``) and runs SRPC in TCP-only mode.

Setting this env var replaces most of the former Python-level monkey
patches (blanket ``init_srpc*`` no-ops, ``init_transfer_engine_client``
mmap-only rewrite, ``ClientV6dMmapManager._create_mmap`` patch,
``VineyardPeer -rpc=false`` injection, ``launch_v6d`` env propagation):
with RDMA disabled, the real v6d client stack works on CPU-only hosts,
and child processes (EngineCore workers, dashllm-launched v6d daemons)
inherit the env var automatically.

ONE patch remains — and it is about alignment, not RDMA: the simulator's
capacity-mode daemon runs with ``-2M_alignment=false`` (see
``v6d_capacity.py``), which makes the shared mmap size a non-2MB
multiple.  SRPC memory registration requires 2MB-aligned chunks, so the
client-side ``init_srpc`` call in ``ClientV6dMmapManager._create_mmap``
raises ``SRPC_STREAM_ERROR_MEM_ALIGN_ERROR`` against such a daemon.  The
sim never transfers KV data over SRPC, so registration is skipped: the
client keeps the plain mmap (IPC reads still work).
"""
import logging
import os


def _install_v6d_ipc_hook() -> None:
    """Enable CPU-only v6d IPC: disable RDMA + skip SRPC registration."""
    logger = logging.getLogger("sglang_simulator")

    # RDMA/libcuda avoidance is handled natively by the SRPC engine.
    # setdefault so child processes inherit it via the environment.
    os.environ.setdefault("SRPC_STREAM_DISABLE_RDMA", "1")
    logger.info(
        "[v6d_ipc_hook] SRPC_STREAM_DISABLE_RDMA=%s (TCP-only SRPC, "
        "no libcuda/barex dependency)",
        os.environ["SRPC_STREAM_DISABLE_RDMA"],
    )

    # Skip client-side SRPC memory registration: the capacity-mode daemon
    # mmap is 4K-aligned (non-2MB-multiple size) and cannot be registered.
    def _skip_srpc_register(base_addr: int, size: int, *args, **kwargs) -> None:
        logger.info(
            "[v6d_ipc_hook] skip SRPC memory registration base_addr=%s "
            "size=%s (4K-aligned sim daemon mmap)",
            base_addr, size,
        )
        return None

    try:
        import v6d.lite.common.transfer_engine as transfer_engine
        # init_srpc_ is the C binding; init_srpc and
        # init_transfer_engine_client resolve it from module globals, and
        # ClientV6dMmapManager._create_mmap imports init_srpc at call time,
        # so patching these two names covers every registration path.
        transfer_engine.init_srpc_ = _skip_srpc_register
        transfer_engine.init_srpc = _skip_srpc_register
        logger.info("[v6d_ipc_hook] patched lite transfer_engine init_srpc")
    except Exception as exc:
        logger.exception(
            "[v6d_ipc_hook] import v6d.lite.common.transfer_engine failed: %r",
            exc,
        )
        return

    try:
        import v6d.common.transfer as transfer
        # v6d.common.transfer star-imports the names above; refresh its copies.
        transfer.init_srpc_ = _skip_srpc_register
        transfer.init_srpc = _skip_srpc_register
        logger.info("[v6d_ipc_hook] patched v6d.common.transfer init_srpc")
    except Exception as exc:
        logger.info("[v6d_ipc_hook] import v6d.common.transfer skipped: %r", exc)

    logger.info("[v6d_ipc_hook] completed")

# Native V6D control-plane mode keeps the real vLLM connector stack
# (HybridConnector -> V6dObjectKVTBackend -> V6dObjectBackend/PBackend ->
# V6dObjectConnectorScheduler/V6dObjectManager) and installs only runtime
# hijack hooks for CPU-only execution.  The default CPU simulation mode still
# uses MockHybridConnector + V6DCacheStorage(etcd) for scheduling parity.
#
# This file must not patch DashServing/vLLM source files on disk.  All behavior
# changes are installed through environment variables and monkey patches
# gated by explicit environment variables.
