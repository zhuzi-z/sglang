import sglang_simulator.hook as sglang_simulator_hook
import torch
from sglang_simulator.simulation.sglang import (
    cache_controller,
    hicache_storage,
    hiradix_cache,
    mem_cache_allocator,
    mem_pool,
    mem_pool_host,
    model_runner,
    scheduler,
    sgl_kernel_hook,
    server_args,
    disaggregation,
)


def init_hook():
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
            mem_pool.C_DSATokenToKVPoolHook,
            mem_cache_allocator.C_PagedTokenToKVPoolAllocatorHook,
            mem_pool_host.C_HostKVCacheHook,
            disaggregation.C_DecodePreallocQueueHook,
        ]
    )
