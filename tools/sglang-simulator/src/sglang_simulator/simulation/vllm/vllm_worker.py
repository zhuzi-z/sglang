"""
vLLM Worker - High-level worker class that wraps vLLM's AsyncLLM engine
for use in the simulation benchmark framework.

Similar to SGLangWorker, this class:
1. Installs hooks before importing vLLM
2. Creates an AsyncLLM instance with hijacked backend
3. Provides generate()/async_generate() interface compatible with BaseWorker
4. Supports MultiInstanceBenchmarkRunner via per-request async streaming

AsyncLLM runs the EngineCore in a background process (EngineCoreProc), so
connectors that need an engine-side runtime (e.g. HybridConnector's
engine_proxy.core_init) initialize natively there.  Per-request / iteration
stats are recorded by the child-process hooks, exported to
SGLANG_SIMULATOR_OUTPUT_DIR by the profile hook on each round boundary
(trigger_simulation), and loaded back from disk by get_*_stats().
"""

import asyncio
import json
import os
import uuid

from sglang_simulator.dataset import GenericRequest
from sglang_simulator.simulation.benchmark import BaseWorker
from sglang_simulator.utils import get_logger
from sglang_simulator.simulation.vllm.startup import init_hook

# Environment must be set before vllm import
os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")

init_hook()

from vllm import SamplingParams  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402

# AsyncLLM.from_engine_args requires AsyncEngineArgs (adds enable_log_requests
# etc.); it is an EngineArgs subclass accepting the same fields, so re-export
# it under the old name to keep callers unchanged.
EngineArgs = AsyncEngineArgs

logger = get_logger("sglang_simulator")


# Simulation-fixed defaults applied to EngineArgs
_SIMULATION_DEFAULTS = {
    "enforce_eager": True,
    "load_format": "dummy",
    "async_scheduling": False,
}


def _resolve_prompt(req: GenericRequest):
    """Resolve prompt from GenericRequest (text or token_ids)."""
    from vllm import TokensPrompt

    if req.prompt is not None:
        return req.prompt
    if req.token_ids is not None:
        return TokensPrompt(prompt_token_ids=req.token_ids)
    raise ValueError("Request must have either prompt or token_ids")


class VLLMWorker(BaseWorker):
    """High-level vLLM worker for simulation benchmarks.

    Accepts a vLLM EngineArgs directly. Simulation-fixed defaults are applied
    automatically (enforce_eager, load_format, async_scheduling) but can
    be overridden in the EngineArgs if needed.
    """

    def __init__(
        self,
        engine_args: EngineArgs,
        name: str = "vllm_worker0",
    ):
        super().__init__(name)

        # The engine-core child inherits this at spawn time; its profile hook
        # dumps request/iteration stats here.  setdefault so an explicit
        # SGLANG_SIMULATOR_OUTPUT_DIR from the environment wins.
        os.environ.setdefault(
            "SGLANG_SIMULATOR_OUTPUT_DIR", f"/tmp/sglang_simulator/{name}"
        )
        self.output_dir = os.path.realpath(os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"])

        # Apply simulation defaults for fields still at their EngineArgs default
        for field_name, sim_default in _SIMULATION_DEFAULTS.items():
            current = getattr(engine_args, field_name)
            ea_default = getattr(AsyncEngineArgs, field_name, None)
            if current == ea_default:
                setattr(engine_args, field_name, sim_default)

        # Construction is safe outside a running event loop: AsyncLLM defers
        # its output handler to the first generate() call.
        self._llm = AsyncLLM.from_engine_args(engine_args)
        logger.info("[VLLMWorker] Initialized with model=%s", engine_args.model)

        # trigger_simulation alternates start/stop profile (round boundaries)
        self._profile_is_start = True

    # ------------------------------------------------------------------
    # Async interface (for MultiInstanceBenchmarkRunner)
    # ------------------------------------------------------------------

    async def async_generate(self, req: GenericRequest):
        """Stream a single request through the engine; returns the final output."""
        # Pass simulation metadata (created_time) via extra_args
        extra_args = None
        if req.custom_params:
            sim_meta = {}
            if "created_time" in req.custom_params:
                sim_meta["created_time"] = req.custom_params["created_time"]
            if "total_request" in req.custom_params:
                sim_meta["total_request"] = req.custom_params["total_request"]
            if sim_meta:
                extra_args = {"simulation": sim_meta}

        sp = SamplingParams(
            max_tokens=req.output_length, ignore_eos=True, extra_args=extra_args
        )
        prompt = _resolve_prompt(req)

        final_output = None
        async for output in self._llm.generate(
            prompt, sp, request_id=uuid.uuid4().hex
        ):
            final_output = output
        return final_output

    async def trigger_simulation(self, output_dir: str | None = None):
        """Round separator: profile() in the engine-core child dumps
        request/iteration stats to SGLANG_SIMULATOR_OUTPUT_DIR and resets
        them (is_start=True additionally resets the local prefix cache)."""
        if self._profile_is_start:
            await self._llm.start_profile()
        else:
            await self._llm.stop_profile()
        self._profile_is_start = not self._profile_is_start

    async def pause_generation(self):
        pass

    async def continue_generation(self):
        pass

    # ------------------------------------------------------------------
    # Stats interface (loaded back from the child-process dump)
    # ------------------------------------------------------------------

    def _load_jsonl(self, filename: str) -> list[dict]:
        data = []
        file_path = os.path.join(self.output_dir, filename)
        if os.path.exists(file_path):
            with open(file_path) as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            logger.error(f"The statistics data({file_path}) does not exist.")
        return data

    def get_request_stats(self) -> list[dict]:
        return self._load_jsonl("request.jsonl")

    def get_iteration_stats(self) -> list[dict]:
        return self._load_jsonl("iteration.jsonl")

    def reset_stats(self):
        """No-op: stats live in the engine-core child and are reset by the
        profile hook at each round boundary (trigger_simulation)."""
        pass

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------

    def generate(self, req: GenericRequest):
        """Generate output for a single request (sync).

        Only safe as the first generate entrypoint on this worker: AsyncLLM
        binds its output handler to the loop of the first generate() call.
        """
        return asyncio.run(self.async_generate(req))

    def flush_cache(self):
        """Not yet supported for vLLM simulation."""
        pass

    def shutdown(self):
        """Shutdown the vLLM engine and its background process."""
        self._llm.shutdown()
        logger.info("[VLLMWorker] Shutting down.")
