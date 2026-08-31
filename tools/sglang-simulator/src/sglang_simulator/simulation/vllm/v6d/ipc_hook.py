"""V6D IPC hook: CPU-only v6d client support in the vLLM process.

Simplified: the v6d SRPC engine (``libsrpc_stream_engine.so``) natively
supports ``SRPC_STREAM_DISABLE_RDMA=1``, which skips loading
``libsrpc_barex_bridge.so`` (the component that pulls in ``libcuda.so.1``,
``libibverbs`` and ``libmlx5``) and runs SRPC in TCP-only mode.

Setting this env var replaces the former Python-level monkey patches
(blanket ``init_srpc*`` no-ops, ``init_transfer_engine_client`` mmap-only
rewrite, ``ClientV6dMmapManager._create_mmap`` patch, ``VineyardPeer
-rpc=false`` injection, ``launch_v6d`` env propagation): with RDMA
disabled, the real v6d client stack works on CPU-only hosts, and child
processes (EngineCore workers, dashllm-launched v6d daemons) inherit the
env var automatically.

SRPC memory registration is NOT skipped anymore: the daemon runs a
2M-aligned sparse mmap (``reserve_memory=false``), so registration
succeeds and the real SRPC meta probe works end to end.  (Client-side
bulkstore mmap itself is skipped by ``C_V6dMmapManagerHook`` — the C++
``init_mmap`` would populate the whole arena; see v6d_manager.py.)
"""
import logging
import os


def _install_v6d_ipc_hook() -> None:
    """Enable CPU-only v6d IPC: disable RDMA (TCP-only SRPC)."""
    logger = logging.getLogger("sglang_simulator")

    # RDMA/libcuda avoidance is handled natively by the SRPC engine.
    # setdefault so child processes inherit it via the environment.
    os.environ.setdefault("SRPC_STREAM_DISABLE_RDMA", "1")
    logger.info(
        "[v6d_ipc_hook] SRPC_STREAM_DISABLE_RDMA=%s (TCP-only SRPC, "
        "no libcuda/barex dependency)",
        os.environ["SRPC_STREAM_DISABLE_RDMA"],
    )

    logger.info("[v6d_ipc_hook] completed")

# Native V6D control-plane mode keeps the real vLLM connector stack
# (HybridConnector -> V6dObjectKVTBackend -> V6dObjectBackend/PBackend ->
# V6dObjectConnectorScheduler/V6dObjectManager) and installs only runtime
# hijack hooks for CPU-only execution.  It is the only supported mode
# (the legacy MockHybridConnector path has been removed).
#
# This file must not patch DashServing/vLLM source files on disk.  All behavior
# changes are installed through environment variables and monkey patches
# gated by explicit environment variables.
