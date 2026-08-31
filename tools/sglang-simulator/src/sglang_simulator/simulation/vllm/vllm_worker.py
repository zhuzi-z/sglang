"""
vLLM Worker - High-level worker class that wraps vLLM's LLM engine
for use in the simulation benchmark framework.

Similar to SGLangWorker, this class:
1. Installs hooks before importing vLLM
2. Creates an LLM instance with hijacked backend
3. Provides generate()/async_generate() interface compatible with BaseWorker
4. Supports MultiInstanceBenchmarkRunner via native enqueue/wait_for_completion API
"""

import asyncio
import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor

from sglang_simulator.dataset import GenericRequest
from sglang_simulator.simulation.benchmark import BaseWorker
from sglang_simulator.simulation.manager import StateManager
from sglang_simulator.simulation.req_stats_manager import request_stats_manager
from sglang_simulator.utils import get_logger
from sglang_simulator.simulation.vllm.startup import init_hook

# Environment must be set before vllm import
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")

init_hook()

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402

# Hook modules are imported only after init_hook(): by then they are already
# loaded by the hook installation sequence, and any future transitive
# dependency can never precede hook installation.
from sglang_simulator.simulation.vllm.scheduler import C_VLLMSchedulerHook  # noqa: E402

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
    automatically (enforce_eager, load_format, enable_prefix_caching) but can
    be overridden in the EngineArgs if needed.
    """

    def __init__(
        self,
        engine_args: EngineArgs,
        name: str = "vllm_worker0",
    ):
        super().__init__(name)

        # Apply simulation defaults for fields still at their EngineArgs default
        for field_name, sim_default in _SIMULATION_DEFAULTS.items():
            current = getattr(engine_args, field_name)
            ea_default = getattr(EngineArgs, field_name, None)
            if current == ea_default:
                setattr(engine_args, field_name, sim_default)

        self._llm = LLM(
            **{
                f.name: getattr(engine_args, f.name)
                for f in dataclasses.fields(engine_args)
            }
        )
        logger.info("[VLLMWorker] Initialized with model=%s", engine_args.model)

        # Detect API availability: newer vLLM has enqueue/wait_for_completion
        self._has_enqueue_api = hasattr(self._llm, "enqueue")

        # Async state
        self._completed_reqs: list[tuple[GenericRequest, object]] = []
        self._executor = ThreadPoolExecutor(max_workers=1)
        # Batch coordination for async_generate
        self._enqueue_count = 0
        self._batch_outputs: list = []
        self._batch_processed = False
        self._batch_lock = asyncio.Lock()
        # For generate()-based fallback: collect prompts/params
        self._batch_prompts: list = []
        self._batch_sampling_params: list = []

    # ------------------------------------------------------------------
    # Async interface (for MultiInstanceBenchmarkRunner)
    # Supports both enqueue/wait_for_completion (v0.23+) and generate() fallback
    # ------------------------------------------------------------------

    async def trigger_simulation(self, output_dir: str | None = None):
        """Reset batch coordination state between benchmark rounds.

        Request and iteration statistics are collected directly by the
        scheduler hook.  Do not call ``LLM.start_profile()`` here: this method
        is invoked at both boundaries of a benchmark round, and the offline
        dummy engine intentionally has no CUDA profiler configured.
        """
        self._enqueue_count = 0
        self._batch_outputs = []
        self._batch_processed = False
        self._batch_prompts = []
        self._batch_sampling_params = []

    async def pause_generation(self):
        """Called at the start of each benchmark round; clear previous stats."""
        self._completed_reqs = []

    async def async_generate(self, req: GenericRequest):
        """Enqueue a request and coordinate batch processing."""
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

        if self._has_enqueue_api:
            # Newer vLLM: enqueue() is sync & fast, adds to engine queue
            self._llm.enqueue(prompt, sp)
        else:
            # Older vLLM: collect for batch generate()
            self._batch_prompts.append(prompt)
            self._batch_sampling_params.append(sp)

        my_index = self._enqueue_count
        self._enqueue_count += 1

        # Yield to let all other concurrent tasks enqueue first
        await asyncio.sleep(0)

        # First task to acquire lock triggers batch processing
        async with self._batch_lock:
            if not self._batch_processed:
                loop = asyncio.get_running_loop()
                if self._has_enqueue_api:
                    self._batch_outputs = await loop.run_in_executor(
                        self._executor,
                        lambda: self._llm.wait_for_completion(use_tqdm=True),
                    )
                else:
                    # Fallback: use generate() with collected batch
                    prompts = self._batch_prompts
                    params = self._batch_sampling_params
                    self._batch_outputs = await loop.run_in_executor(
                        self._executor,
                        lambda: self._llm.generate(prompts, params, use_tqdm=True),
                    )
                self._batch_processed = True

        output = self._batch_outputs[my_index]
        self._completed_reqs.append((req, output))
        return output

    async def continue_generation(self):
        pass

    # ------------------------------------------------------------------
    # Stats interface
    # ------------------------------------------------------------------

    def get_request_stats(self) -> list[dict]:
        """Build per-request stats from scheduler-tracked data."""
        stats = []
        for req, output in self._completed_reqs:
            input_len = len(req.token_ids) if req.token_ids else 0
            output_len = len(output.outputs[0].token_ids) if output.outputs else 0

            # Look up per-request stats recorded by the scheduler hook
            req_id = output.request_id
            tracked = request_stats_manager.stats.get(req_id)

            if tracked:
                gen_token_latencies = tracked.gen_token_latencies
                created_time = tracked.created_time
                queue_start = tracked.queue_start
                queue_end = tracked.queue_end
                last_event_time = tracked.last_event_time
                final_device_hit_len = tracked.final_device_hit_len
                local_kv_hit_len = tracked.local_kv_hit_len
                ext_kv_hit_len = tracked.ext_kv_hit_len
                final_host_hit_len = tracked.final_host_hit_len
            else:
                # Fallback for non-simulation requests
                gen_token_latencies = [0.001] * max(1, output_len)
                created_time = 0.0
                queue_start = 0.0
                queue_end = 0.0
                last_event_time = sum(gen_token_latencies)
                final_device_hit_len = 0
                local_kv_hit_len = 0
                ext_kv_hit_len = 0
                final_host_hit_len = 0

            stats.append(
                {
                    "gen_token_latencies": gen_token_latencies,
                    "created_time": created_time,
                    "queue_start": queue_start,
                    "queue_end": queue_end,
                    "last_event_time": last_event_time,
                    "input_length": input_len,
                    "output_length": output_len,
                    "final_device_hit_len": final_device_hit_len,
                    "local_kv_hit_len": local_kv_hit_len,
                    "ext_kv_hit_len": ext_kv_hit_len,
                    "final_host_hit_len": final_host_hit_len,
                    "final_storage_hit_len": 0,
                }
            )
        return stats

    def reset_stats(self):
        """Reset per-request stats for the next benchmark round."""
        request_stats_manager.reset()
        C_VLLMSchedulerHook.ITERATION_STATS.clear()
        # Reset global simulation state (global_clock, iteration counter, etc.)
        # to prevent state leakage across consecutive benchmark runs.
        StateManager.reset()

    def get_iteration_stats(self) -> list[dict]:
        """Return per-iteration stats collected by the scheduler hook."""
        return list(C_VLLMSchedulerHook.ITERATION_STATS)

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------

    def generate(self, req: GenericRequest):
        """Generate output for a single request (sync)."""
        sp = SamplingParams(max_tokens=req.output_length, ignore_eos=True)
        outputs = self._llm.generate([_resolve_prompt(req)], [sp])
        return outputs[0]

    def flush_cache(self):
        """Not yet supported for vLLM simulation."""
        pass

    def shutdown(self):
        """Shutdown the vLLM engine."""
        self._executor.shutdown(wait=False)
        logger.info("[VLLMWorker] Shutting down.")
