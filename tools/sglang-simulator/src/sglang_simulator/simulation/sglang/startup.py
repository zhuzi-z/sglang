import os


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_enabled(*names: str) -> bool:
    return any(os.environ.get(name, "").strip().lower() in _TRUE_VALUES for name in names)


def init_hook(force: bool = False):
    """Install all SGLang hooks only when explicitly enabled."""
    if not force and not _env_enabled(
        "SGLANG_SIMULATOR_ENABLE_HOOK",
        "SGLANG_SIMULATOR_ENABLE_SGLANG_HOOK",
    ):
        return False

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
    return True
