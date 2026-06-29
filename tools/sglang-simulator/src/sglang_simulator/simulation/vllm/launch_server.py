"""
vLLM Simulation Server Launcher - Starts vLLM's OpenAI-compatible API server
with simulation hooks installed.

Usage:
    python -m sglang_simulator.simulation.vllm.launch_server --model <model_path> [vLLM args...]

Environment variables:
    SGLANG_SIMULATOR_CONFIG_PATH: Path to simulation config JSON
    SGLANG_SIMULATOR_OUTPUT_MODE: "BLOCKING" or "OFFLINE" (default: BLOCKING for server)
"""

import os

# Default to BLOCKING mode for server scenario
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")

from sglang_simulator.simulation.vllm.startup import init_hook
from sglang_simulator.utils import get_logger

init_hook()

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
        description="vLLM Simulation Server (OpenAI-compatible)"
    )
    parser = make_arg_parser(parser)
    parser.add_argument(
        "--sim-config-path",
        type=str,
        default=None,
        help="Path to simulation JSON config (same as SGLANG_SIMULATOR_CONFIG_PATH).",
    )

    args = parser.parse_args()
    validate_parsed_serve_args(args)

    # Set config path from CLI arg if env var not set
    config_path = os.getenv("SGLANG_SIMULATOR_CONFIG_PATH")
    if config_path and os.path.exists(config_path):
        logger.info(f"Using config from env: {config_path}")
    elif args.sim_config_path:
        os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = args.sim_config_path
        logger.info(f"Using config from arg: {args.sim_config_path}")

    logger.info(
        "Starting vLLM simulation server in %s mode.",
        os.environ.get("SGLANG_SIMULATOR_OUTPUT_MODE", "BLOCKING"),
    )

    uvloop.run(run_server(args))
