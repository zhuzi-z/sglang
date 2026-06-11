import inspect
from dataclasses import replace
from sglang_simulator.hook import BaseHook



class C_MambaPoolHook(BaseHook):
    HOOK_CLASS_NAME = "MambaPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool"
    
    @classmethod
    def hook(cls, target):

        original_init = target.__init__

        original_init = target.__init__
        sig = inspect.signature(original_init)

        def wrapped_init(self, *args, **kwargs):
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()

            cache_params = bound.arguments.get("cache_params")
            if cache_params is not None:
                bound.arguments["cache_params"] = cls.modify_cache_params(cache_params)

            return original_init(**bound.arguments)

        target.__init__ = wrapped_init

    @staticmethod
    def modify_cache_params(cache_params):
        shape = cache_params.shape

        new_conv = tuple(
            tuple(1 for _ in conv_item)
            for conv_item in shape.conv
        )
        new_temporal = tuple(1 for _ in shape.temporal)

        # Current dataclass is frozen
        new_shape = replace(
            shape,
            conv=new_conv,
            temporal=new_temporal,
        )

        return replace(cache_params, shape=new_shape)