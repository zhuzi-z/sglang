import torch
import inspect
from dataclasses import replace
from typing import Optional
from sglang_simulator.hook import BaseHook


class C_DSATokenToKVPoolHook(BaseHook):

    HOOK_CLASS_NAME = "DSATokenToKVPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool"

    @classmethod
    def hook(cls, target):

        def override_init(
            self,
            size: int,
            page_size: int,
            kv_lora_rank: int,
            dtype: torch.dtype,
            qk_rope_head_dim: int,
            layer_num: int,
            device: str,
            index_head_dim: int,
            enable_memory_saver: bool,
            kv_cache_dim: int,
            start_layer: Optional[int] = None,
            end_layer: Optional[int] = None,
            index_buf_size: Optional[int] = None,
        ):
            override_dim = (
                kv_cache_dim if kv_cache_dim != kv_lora_rank + qk_rope_head_dim else None
            )

            super(target, self).__init__(
                size,
                page_size,
                dtype,
                kv_lora_rank,
                qk_rope_head_dim,
                layer_num,
                device,
                enable_memory_saver,
                start_layer,
                end_layer,
                use_dsa=True,
                override_kv_cache_dim=override_dim,
            )
            # self.index_k_dtype = torch.float8_e4m3fn
            # self.index_k_scale_dtype = torch.float32
            self.index_head_dim = index_head_dim
            if index_buf_size is None:
                index_buf_size = size
            # num head == 1 and head dim == 128 for index_k in DSA
            assert index_head_dim == 128

            # override the index head dim to 1 during allocation
            index_head_dim = 1

            self.index_k_with_scale_buffer = [
                torch.zeros(
                    # Layout:
                    #     ref: test_attention.py :: kv_cache_cast_to_fp8
                    #     shape: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4)
                    #     data: for page i,
                    #         * buf[i, :page_size * head_dim] for fp8 data
                    #         * buf[i, page_size * head_dim:].view(float32) for scale
                    (
                        (index_buf_size + page_size + 1) // self.page_size,
                        self.page_size
                        * (
                            index_head_dim + index_head_dim // self.quant_block_size * 4
                        ),
                    ),
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=device,
                )
                for _ in range(layer_num)
            ]
            self._finalize_allocation_log(size)

        target.__init__ = override_init




class C_DeepSeekV4SingleKVPoolHook(BaseHook):
    HOOK_CLASS_NAME = "DeepSeekV4SingleKVPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.deepseek_v4_memory_pool"

    @classmethod
    def hook(cls, target):
        def ceil_div(x: int, y: int) -> int:
            return (x + y - 1) // y

        def override_create_buffer(self, *, num_pages: int):
            bytes_per_token = self.get_bytes_per_token()
            self.kv_cache_total_dim = bytes_per_token
            bytes_per_page_non_padded = self.page_size * bytes_per_token
            self.bytes_per_page_padded = ceil_div(bytes_per_page_non_padded, 576) * 576

            # Remove the following assertion.
            # assert bytes_per_token == 448 + 64 * 2 + 8, (
            #     "DSV4 KV layout: qk_nope_head_dim FP8 (448) + qk_rope_head_dim BF16 "
            #     "(64*2) + nope FP8 scales + scale_pad = 584 bytes/token"
            # )
            # assert self.store_dtype == torch.uint8

            return torch.zeros(
                num_pages,
                1, # self.bytes_per_page_padded,
                dtype=self.store_dtype,
                device=self.device,
            )

        target.create_buffer = override_create_buffer


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