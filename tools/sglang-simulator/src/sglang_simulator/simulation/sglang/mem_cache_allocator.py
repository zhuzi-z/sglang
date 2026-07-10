import torch
import types
from sglang_simulator.hook import BaseHook


class C_PagedTokenToKVPoolAllocatorHook(BaseHook):
    HOOK_CLASS_NAME = "PagedTokenToKVPoolAllocator"
    HOOK_MODULE_NAME = r"^sglang\.srt\.mem_cache\.allocator(?:\.paged)?$"
    REGEX = True

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        # The implementation of `alloc_extend` and `alloc_decode` is derived from `NPUPagedTokenToKVPoolAllocator`.
        def alloc_extend(
            self,
            prefix_lens: torch.Tensor,
            prefix_lens_cpu: torch.Tensor,
            seq_lens: torch.Tensor,
            seq_lens_cpu: torch.Tensor,
            last_loc: torch.Tensor,
            extend_num_tokens: int,
            num_new_pages: int = None,
        ):

            try:
                from sglang.srt.mem_cache.allocator import paged as module
            except ImportError:
                from sglang.srt.mem_cache import allocator as module

            if num_new_pages is None:
                num_new_pages_tensor = (
                    (seq_lens + self.roundup) // self.page_size
                    - (prefix_lens + self.roundup) // self.page_size
                ).sum()
                num_new_pages_item = num_new_pages_tensor.item()
            else:
                num_new_pages_item = num_new_pages
            if self.need_sort and num_new_pages_item > len(self.free_pages):
                self.merge_and_sort_free()

            if num_new_pages_item > len(self.free_pages):
                return None

            out_indices = torch.empty(
                (extend_num_tokens,),
                dtype=self.free_pages.dtype,
                device=self.device,
            )
            module.alloc_extend_naive(
                prefix_lens,
                seq_lens,
                last_loc,
                self.free_pages,
                out_indices,
                self.page_size,
                self.device,
            )

            self.free_pages = self.free_pages[num_new_pages_item:]
            return out_indices

        def alloc_decode(
            self,
            seq_lens: torch.Tensor,
            seq_lens_cpu: torch.Tensor,
            last_loc: torch.Tensor,
        ):

            from sglang.srt.utils import get_num_new_pages

            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                decode=True,
            )

            if num_new_pages > len(self.free_pages):
                self.merge_and_sort_free()

            if num_new_pages > len(self.free_pages):
                return None

            need_new_pages = (seq_lens % self.page_size == 1).int()
            end_new_pages = torch.cumsum(need_new_pages, 0)
            start_new_pages = end_new_pages - need_new_pages
            if num_new_pages == 0:
                out_indices = last_loc + 1
            else:
                out_indices = (last_loc + 1) * (1 - need_new_pages) + self.free_pages[
                    start_new_pages
                ] * self.page_size * need_new_pages

            self.free_pages = self.free_pages[num_new_pages:]
            return out_indices.to(self.free_pages.dtype)

        def wrapped_init(self, *args, **kwargs):

            original_init(self, *args, **kwargs)

            if self.device == "cpu":
                self.alloc_extend = types.MethodType(alloc_extend, self)
                self.alloc_decode = types.MethodType(alloc_decode, self)
                self.roundup = self.page_size - 1

        target.__init__ = wrapped_init
