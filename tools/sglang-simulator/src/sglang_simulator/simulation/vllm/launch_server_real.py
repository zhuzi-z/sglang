"""vLLM Real-GPU Server Launcher with schedule_batch / requests collection.

Starts vLLM's OpenAI-compatible API server on real GPUs and installs ONLY the
observer collection hooks (no CPU-sim mocks), so a /start_profile.../stop_profile
benchmark run produces TP0.schedule_batch.jsonl + TP0.requests.jsonl with the exact
same schema as the CPU-sim launcher — enabling a direct sim-vs-real diff.

Usage:
    SGLANG_SIMULATOR_ENABLE_VLLM_HOOK=1 \
    python -m sglang_simulator.simulation.vllm.launch_server_real --model <path> ...

Environment variables:
    SGLANG_SIMULATOR_OUTPUT_DIR: where TP{rank}.*.jsonl are written.
"""

import os

# Force real-GPU collection mode. Unlike the sim launcher we do NOT disable V1
# multiprocessing — real TP deployments run one worker process per rank; the rank-0
# gate in the dump hook keeps schedule_batch to a single file.
os.environ["SGLANG_SIMULATOR_HOOK_MODE"] = "real"

from sglang_simulator.simulation.vllm.startup import init_hook
from sglang_simulator.utils import get_logger

init_hook(mode="real")

logger = get_logger("sgl_simulator")


if __name__ == "__main__":
    import uvloop
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="vLLM Real-GPU Server (OpenAI-compatible) with collection hooks"
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    logger.info("Starting vLLM real-GPU server with schedule_batch collection.")
    uvloop.run(run_server(args))
