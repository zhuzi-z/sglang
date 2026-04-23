from dataclasses import asdict, dataclass

import numpy as np
import torch
from sgl_annotation_hook import (
    C_SglangModelForwardAnnotationHook,
    C_SglangSchedulerRunBatchAnnotationHook,
)
from sglang_simulator.hook import install_class_hooks

install_class_hooks(
    [C_SglangModelForwardAnnotationHook, C_SglangSchedulerRunBatchAnnotationHook]
)


from sglang.srt.entrypoints.engine import Engine  # noqa
from sglang.srt.server_args import ServerArgs  # noqa


@dataclass
class ScheduleBatchRequest:
    extend_len: int = 0
    past_kv_len: int = 0
    input_ids: list[int] | None = None
    output_ids: list[int] | None = None

    def full_tokens(self) -> list[int]:
        total_len = self.extend_len + self.past_kv_len
        if self.input_ids is None:
            self.input_ids = np.random.randint(1000, 100000, size=total_len).tolist()
            self.output_ids = []
        return (self.input_ids + self.output_ids)[:total_len]

    def prefix_tokens(self) -> list[int]:
        if self.past_kv_len > 0:
            tokens = self.full_tokens()
            return tokens[: self.past_kv_len]
        return []


@dataclass
class ScheduleBatch:
    reqs: list[ScheduleBatchRequest]

    def __repr__(self) -> str:
        result = []
        for req in self.reqs:
            result.append((req.extend_len, req.past_kv_len))
        return f"{result}"

    def prefix_reqs(self) -> list[list[int]]:
        tokens = []
        for req in self.reqs:
            prefix_tokens = req.prefix_tokens()
            if len(prefix_tokens):
                tokens.append(prefix_tokens)
        return tokens

    def full_reqs(self) -> list[list[int]]:
        tokens = []
        for req in self.reqs:
            tokens.append(req.full_tokens())
        return tokens

    def total_tokens(self) -> int:
        return sum([req.extend_len + req.past_kv_len for req in self.reqs])


def run(
    server_args: ServerArgs,
    batch_list: list[ScheduleBatch],
    output_dir: str,
    profiler: str = "torch",
    num_replay: int = 3,
    max_new_tokens: int = 1,
    skip_out_of_tokens: bool = True,
    flush_cache: bool = True,  # When profiling decode cases, keeping the radix cache avoids the prefill stage.
):

    print(f"Replaying a total of {len(batch_list)} batches.")
    llm = Engine(**asdict(server_args))

    llm.generate(
        prompt="warmup!",
    )
    # `get_server_info` hangs when called immediately after engine initialization
    server_info = llm.get_server_info()
    max_total_num_tokens = server_info["max_total_num_tokens"]

    for idx, batch in enumerate(batch_list):
        print(f"Profiling: {batch}")
        if skip_out_of_tokens and batch.total_tokens() > max_total_num_tokens:
            print(
                f"The current batch requires {batch.total_tokens()} tokens, "
                f"which exceeds the maximum total token limit({max_total_num_tokens})."
            )
            continue
        sampling_params = {
            "temperature": 0,
            "top_p": 1,
            "max_new_tokens": max_new_tokens,
        }

        # Start profiling
        if profiler == "torch":
            llm.start_profile(
                activities=["GPU", "CPU"],
                with_stack=False,
                output_dir=output_dir + f"/{idx}",
            )
        else:
            torch.cuda.cudart().cudaProfilerStart()

        # LLM Inference
        for _ in range(num_replay):
            prefix_reqs = batch.prefix_reqs()
            if len(prefix_reqs) > 0:
                llm.generate(input_ids=prefix_reqs, sampling_params=sampling_params)
            llm.generate(input_ids=batch.full_reqs(), sampling_params=sampling_params)
            # clear cache
            if flush_cache:
                llm.flush_cache()

        # Stop profiling
        if profiler == "torch":
            llm.stop_profile()
        else:
            torch.cuda.cudart().cudaProfilerStop()

        if flush_cache:
            llm.flush_cache()

    llm.shutdown()
