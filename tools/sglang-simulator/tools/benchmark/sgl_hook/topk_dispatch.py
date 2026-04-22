import os
import random
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl
from sglang_simulator.hook import BaseHook

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import StandardTopKOutput


LOG_TOPK_STATS = os.getenv("SGL_LOG_TOPK_STATS", "0") == "1"


_EP_PERM_CACHE = {}


def get_ep_balanced_perm(
    M: int, ep_world_size: int, device: torch.device
) -> torch.Tensor:
    """
    Build a deterministic expert permutation for EP-balanced dispatch.

    The permutation interleaves experts from different EP ranks, while each
    rank-local expert list is shuffled independently. This gives:
    1. approximately uniform expert usage,
    2. deterministic behavior for debugging,
    3. some randomness in expert selection order.
    """
    if (M, ep_world_size, device) not in _EP_PERM_CACHE:
        assert (
            M % ep_world_size == 0
        ), f"Experts {M} must be divisible by EP size {ep_world_size}"
        E_local = M // ep_world_size

        # Split experts into EP-local groups.
        groups = [
            list(range(r * E_local, (r + 1) * E_local)) for r in range(ep_world_size)
        ]

        # Use fixed seeds so that all processes generate the same shuffle result.
        base_seed = 12345
        for i, g in enumerate(groups):
            rng = random.Random(base_seed + i)
            rng.shuffle(g)

        # Interleave shuffled experts from each EP rank.
        perm = []
        for i in range(E_local):
            for r in range(ep_world_size):
                perm.append(groups[r][i])

        _EP_PERM_CACHE[(M, ep_world_size, device)] = torch.tensor(
            perm, dtype=torch.int32, device=device
        )

    return _EP_PERM_CACHE[(M, ep_world_size, device)]


@triton.jit
def balanced_topk_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    perm_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_k = tl.arange(0, K)

    # Global slot index for flattened [N, K].
    idx = offs_n[:, None] * K + offs_k[None, :]

    # Position inside one expert cycle, and cycle number.
    idx_mod_M = idx % M
    idx_div_M = idx // M

    mask_2d = mask_n[:, None] & (offs_k[None, :] < K)

    # Lookup from permutation table, then shift by cycle id.
    # This keeps dispatch balanced while avoiding a fully fixed pattern.
    expert_id_raw = tl.load(perm_ptr + idx_mod_M, mask=mask_2d, other=0)
    expert_id = (expert_id_raw + idx_div_M) % M

    # Write balanced expert ids.
    tl.store(
        topk_ids_ptr + offs_n[:, None] * K + offs_k[None, :],
        expert_id,
        mask=mask_2d,
    )

    # Assign uniform top-k weights for debug-friendly dispatch.
    tl.store(
        topk_weights_ptr + offs_n[:, None] * K + offs_k[None, :],
        1.0 / K,
        mask=mask_2d,
    )


def transform_select_experts_indexes(
    topk_output: "StandardTopKOutput",
) -> "StandardTopKOutput":
    """
    Override router top-k results with a balanced expert assignment.

    This is mainly used in EP deployment for debugging:
    - tokens are distributed evenly across experts,
    - expert choices still preserve controlled randomness.
    """
    num_tokens, top_k = topk_output.topk_ids.shape
    _, num_experts = topk_output.router_logits.shape

    from sglang.srt.server_args import get_global_server_args

    perm_tensor = get_ep_balanced_perm(
        num_experts,
        get_global_server_args().ep_size,
        topk_output.topk_ids.device,
    )

    BLOCK_N = 128
    grid = (triton.cdiv(num_tokens, BLOCK_N),)

    balanced_topk_kernel[grid](
        topk_output.topk_ids,
        topk_output.topk_weights,
        perm_tensor,
        N=num_tokens,
        M=num_experts,
        K=top_k,
        BLOCK_N=BLOCK_N,
    )

    return topk_output


def expert_load_stats(topk_ids: torch.Tensor, num_experts: int):
    """
    Compute expert load distribution statistics.
    """
    loads = torch.bincount(topk_ids.reshape(-1), minlength=num_experts).cpu().numpy()
    mean = loads.mean()
    std = loads.std()
    cv = std / (mean + 1e-12)
    return loads, cv


class C_TopKBalancedDispatchHook(BaseHook):
    HOOK_CLASS_NAME = "TopK"
    HOOK_MODULE_NAME = "sglang.srt.layers.moe.topk"

    @classmethod
    def hook(cls, target):
        original_forward = target.forward

        def wrapped_forward(self, *args, **kwargs):
            from sglang.srt.server_args import get_global_server_args

            topk_output = original_forward(self, *args, **kwargs)

            if LOG_TOPK_STATS:
                _, num_experts = topk_output.router_logits.shape
                load1, cv1 = expert_load_stats(topk_output.topk_ids, num_experts)
                print(f"TopK: load1={load1}, cv1={cv1}")

            if get_global_server_args().ep_dispatch_algorithm == "balanced":
                topk_output = transform_select_experts_indexes(topk_output)

                if LOG_TOPK_STATS:
                    load2, cv2 = expert_load_stats(topk_output.topk_ids, num_experts)
                    print(f"TopK: load2={load2}, cv2={cv2}")

            return topk_output

        target.forward = wrapped_forward
