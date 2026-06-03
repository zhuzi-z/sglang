from sglang_simulator.hook import BaseHook


class C_HiRadixCacheHook(BaseHook):
    HOOK_CLASS_NAME = "HiRadixCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.hiradix_cache"

    @classmethod
    def hook(cls, target):
        original_check_hicache_events = target.check_hicache_events

        def wrapped_check_hicache_events(self, *args, **kwargs):
            # The async thread for prefetching and backup in `HiCacheController` has been deprecated.
            # So we have to handle the backup or prefetch operation manually.
            self.cache_controller.handle_backup_operation()
            self.cache_controller.handle_prefetch_operation()
            return original_check_hicache_events(self, *args, **kwargs)

        target.check_hicache_events = wrapped_check_hicache_events


class C_HiMambaRadixCacheHook(BaseHook):
    """Same treatment as HiRadixCache, but for hybrid-SSM models (Qwen3.5, Qwen3-Next, ...)
    which route to HiMambaRadixCache instead of HiRadixCache when hicache is enabled."""

    HOOK_CLASS_NAME = "HiMambaRadixCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.hi_mamba_radix_cache"

    @classmethod
    def hook(cls, target):
        original_check_hicache_events = target.check_hicache_events

        def wrapped_check_hicache_events(self, *args, **kwargs):
            # The async prefetch/backup threads on HiCacheController are stubbed
            # out in simulation mode; advance the cache state machine in lock-step
            # with the virtual clock.
            self.cache_controller.handle_backup_operation()
            self.cache_controller.handle_prefetch_operation()
            return original_check_hicache_events(self, *args, **kwargs)

        target.check_hicache_events = wrapped_check_hicache_events
