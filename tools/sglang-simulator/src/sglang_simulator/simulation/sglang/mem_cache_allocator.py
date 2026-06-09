import torch
from sglang_simulator.hook import BaseHook


class IndexableWrapper:
    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, _):
        return self._fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def ceil_div(x: torch.Tensor | int, y: int):
    return (x + y - 1) // y


def alloc_extend_cpu(
    pre_lens_ptr: torch.Tensor,
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,
    bs_upper: int,
    page_size: int,
    extend_num_tokens=None,
):
    # Ensure all tensors are on CPU and contiguous
    pre_lens = pre_lens_ptr.to(device="cpu", dtype=torch.long).contiguous()
    seq_lens = seq_lens_ptr.to(device="cpu", dtype=torch.long).contiguous()
    last_loc = last_loc_ptr.to(device="cpu", dtype=torch.long).contiguous()
    free_pages = free_page_ptr.to(device="cpu", dtype=torch.long).contiguous()
    out_indices = out_indices.to(device="cpu", dtype=torch.long)

    batch_size = pre_lens.numel()

    extend_lens = seq_lens - pre_lens
    pages_before = (pre_lens + page_size - 1) // page_size
    pages_after = (seq_lens + page_size - 1) // page_size
    num_new_pages_per_seq = pages_after - pages_before

    output_offset = 0
    free_page_offset = 0

    # Reusable page template: [0, 1, 2, ..., page_size - 1]
    page_template = torch.arange(page_size, dtype=out_indices.dtype, device=out_indices.device)

    for pid in range(batch_size):
        pre_len = int(pre_lens[pid].item())
        seq_len = int(seq_lens[pid].item())
        extend_len = int(extend_lens[pid].item())
        last_token_pos = int(last_loc[pid].item())
        new_pages = int(num_new_pages_per_seq[pid].item())

        if extend_len <= 0:
            free_page_offset += new_pages
            continue

        # Part 1: fill remaining slots in current incomplete page
        current_page_end = ((pre_len + page_size - 1) // page_size) * page_size
        part1_end = min(seq_len, current_page_end)
        num_part1 = part1_end - pre_len

        if num_part1 > 0:
            out_indices[output_offset: output_offset + num_part1] = torch.arange(
                last_token_pos + 1,
                last_token_pos + 1 + num_part1,
                dtype=out_indices.dtype,
                device=out_indices.device,
            )
            output_offset += num_part1

        if pre_len + num_part1 == seq_len:
            free_page_offset += new_pages
            continue

        # Part 2: fill complete new pages
        full_pages_start = current_page_end
        full_pages_end = (seq_len // page_size) * page_size
        num_part2 = full_pages_end - full_pages_start

        if num_part2 > 0:
            num_full_pages = num_part2 // page_size
            page_ids = free_pages[free_page_offset: free_page_offset + num_full_pages]

            # Build [page_id * page_size + 0..page_size-1] for each full page
            values = page_ids.unsqueeze(1) * page_size + page_template.unsqueeze(0)
            out_indices[output_offset: output_offset + num_part2] = values.reshape(-1)

            output_offset += num_part2

        if pre_len + num_part1 + num_part2 == seq_len:
            free_page_offset += new_pages
            continue

        # Part 3: fill the last incomplete new page
        num_part3 = seq_len - full_pages_end
        if num_part3 > 0:
            last_page_idx = free_page_offset + new_pages - 1
            last_page_id = int(free_pages[last_page_idx].item())
            out_indices[output_offset: output_offset + num_part3] = (
                last_page_id * page_size + page_template[:num_part3]
            )
            output_offset += num_part3

        free_page_offset += new_pages


def alloc_decode_cpu(
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,
    bs_upper: int,
    page_size: int,
):
    # Ensure CPU tensors
    seq_lens = seq_lens_ptr.to(device="cpu", dtype=torch.long).contiguous()
    last_loc = last_loc_ptr.to(device="cpu", dtype=torch.long).contiguous()
    free_pages = free_page_ptr.to(device="cpu", dtype=torch.long).contiguous()

    if torch.any(seq_lens <= 0):
        raise ValueError("seq_lens must be positive for decode allocation")

    pre_lens = seq_lens - 1

    # A new page is needed when the previous length is exactly at a page boundary
    need_new_page = (pre_lens % page_size) == 0

    # Default case: reuse current page
    out_indices[: seq_lens.numel()] = last_loc + 1

    if need_new_page.any():
        # Compute per-sequence offset in free_pages using exclusive prefix sum
        need_new_page_long = need_new_page.to(torch.long)
        new_page_offsets = torch.cumsum(need_new_page_long, dim=0) - need_new_page_long

        selected_page_ids = free_pages[new_page_offsets[need_new_page]]
        out_indices[: seq_lens.numel()][need_new_page] = selected_page_ids * page_size


class C_PagedTokenToKVPoolAllocatorHook(BaseHook):
    HOOK_CLASS_NAME = "PagedTokenToKVPoolAllocator"
    HOOK_MODULE_NAME = r"^sglang\.srt\.mem_cache\.allocator(?:\.paged)?$"
    REGEX = True

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def wrapped_init(self, *args, **kwargs):
            
            try:
                from sglang.srt.mem_cache.allocator import paged as module
            except ImportError:
                from sglang.srt.mem_cache import allocator as module

            # triton kernels are not compatible with the CPU allocator, so we use python implementation instead.
            module.alloc_extend_kernel = IndexableWrapper(alloc_extend_cpu)
            module.alloc_decode_kernel = IndexableWrapper(alloc_decode_cpu)

            original_init(self, *args, **kwargs)

        target.__init__ = wrapped_init
