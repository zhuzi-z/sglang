# Set once init_hook() has run; makes repeated installs (e.g. an explicit
# simulator entrypoint plus the .pth auto-start trigger in the same process)
# no-ops instead of double-registering every class hook.
_HOOKS_INSTALLED = False


def init_hook():
    """Install all SGLang hooks.

    Hooks install unconditionally: this entry is only imported by the
    simulator's own entrypoints (launch_server / worker) or by the .pth
    auto-start bootstrap (which itself is gated on SGLANG_SIMULATOR_ENABLE),
    so reaching it already means simulation mode — no opt-in env var gating
    here.

    Idempotent: calling it more than once in the same process is a no-op.
    """
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return True

    import torch

    import sglang_simulator.hook as sglang_simulator_hook
    from sglang_simulator.simulation.sglang import (
        cache_controller,
        disaggregation,
        hicache_storage,
        hiradix_cache,
        mem_cache_allocator,
        mem_pool,
        mem_pool_host,
        model_runner,
        scheduler,
        server_args,
        sgl_kernel_hook,
    )

    # hook the sglang implementation
    if not torch.cuda.is_available():
        # CPU Platform
        sglang_simulator_hook.install_module_hooks(
            [sgl_kernel_hook.M_SGLangKernelLoadUtilHook]
        )
    # COMMENTED-OUT (M_SGLangCommonHook removed in rebase, replaced by M_SGLangKernelLoadUtilHook above):
    # sglang_simulator_hook.install_module_hooks(
    #     [sgl_kernel_hook.M_SGLangCommonHook]
    # )
    sglang_simulator_hook.install_class_hooks(
        [
            server_args.C_ServerArgsHook,
            scheduler.C_SchedulerHook,
            scheduler.C_SglangPrefillAdderHook,
            scheduler.C_SchedulerRequestReceiver,
            model_runner.C_ModelRunnerHook,
            hicache_storage.C_StorageBackendFactory,
            cache_controller.C_HiCacheController,
            hiradix_cache.C_HiRadixCacheHook,
            mem_pool.C_MambaPoolHook,
            mem_pool.C_DeepSeekV4SingleKVPoolHook,
            mem_cache_allocator.C_PagedTokenToKVPoolAllocatorHook,
            mem_pool_host.C_HostKVCacheHook,
            disaggregation.C_DecodePreallocQueueHook,
        ]
    )
    _HOOKS_INSTALLED = True
    return True
