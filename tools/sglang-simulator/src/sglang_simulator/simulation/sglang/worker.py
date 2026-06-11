import json
import os

import sglang_simulator.hook as sglang_simulator_hook
import torch
from sglang_simulator.dataset import (
    GenericRequest,
)
from sglang_simulator.simulation.benchmark import (
    BaseWorker,
)
from sglang_simulator.simulation.sglang import (
    cache_controller,
    hicache_storage,
    hiradix_cache,
    mem_cache_allocator,
    mem_pool_host,
    model_runner,
    scheduler,
    sgl_kernel_hook,
    disaggregation
)
from sglang_simulator.utils.logger import get_logger

# hook the sglang implementation
if not torch.cuda.is_available():
    # CPU Platform
    sglang_simulator_hook.install_module_hooks(
        [sgl_kernel_hook.M_SGLangKernelLoadUtilHook]
    )
sglang_simulator_hook.install_class_hooks(
    [
        scheduler.C_SchedulerHook,
        scheduler.C_SglangPrefillAdderHook,
        scheduler.C_SchedulerRequestReceiver,
        model_runner.C_ModelRunnerHook,
        hicache_storage.C_StorageBackendFactory,
        cache_controller.C_HiCacheController,
        hiradix_cache.C_HiRadixCacheHook,
        mem_cache_allocator.C_PagedTokenToKVPoolAllocatorHook,
        mem_pool_host.C_HostKVCacheHook,
        disaggregation.C_DecodePreallocQueueHook,
    ]
)


if os.getenv("HISIM_SIMULATION_MODE") is None:
    os.environ["HISIM_SIMULATION_MODE"] = "OFFLINE"

# The sglang must be imported after the hook installer
from sglang.srt.entrypoints.engine import Engine  # noqa
from sglang.srt.server_args import ServerArgs  # noqa

logger = get_logger("sglang_simulator")


class SGLangWorker(BaseWorker):
    def __init__(self, server_args: ServerArgs, name="worker0"):
        super().__init__(name)
        # disable some features which is not necessary for simulation.

        os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"] = f"/tmp/sglang_simulator/{name}"
        os.environ["SGLANG_SIMULATOR_HICACHE_STORAGE_KEYS_PATH"] = (
            f"/tmp/sglang_simulator/{name}/hicache_storage_keys.txt"
        )

        server_args.disable_cuda_graph = True
        self._engine = Engine(server_args=server_args)
        self.output_dir: str = None

    def flush_cache(self):
        self._engine.flush_cache()

    def clear_hicache_storage(self):
        self._engine.loop.run_until_complete(
            self.engine.tokenizer_manager.clear_hicache_storage()
        )

    async def async_generate(self, req: GenericRequest):
        simulation_params = {}
        simulation_params["created_time"] = req.custom_params.get("created_time", 0)
        if "total_request" in req.custom_params:
            simulation_params["total_request"] = req.custom_params.get("total_request")
        return await self._engine.async_generate(
            prompt=req.prompt,
            input_ids=req.token_ids,
            sampling_params={
                "ignore_eos": True,
                "max_new_tokens": req.output_length,
                "custom_params": {
                    # (tmp) Transfer simulation arguments to the scheduler through the custom_params in sampling_params
                    "simulation": simulation_params
                },
            },
            **req.extra_args
        )

    def generate(self, req: GenericRequest):
        self._engine.loop.run_until_complete(self.async_generate(req))

    async def trigger_simulation(self, output_dir: str | None = None):
        if output_dir is None:
            self.output_dir = f"/tmp/sglang_simulator/{self.name}"
        await self._engine.tokenizer_manager.start_profile(output_dir=output_dir)

    def get_iteration_stats(self) -> list[dict]:
        data = []
        file_path = f"{self.output_dir}/iteration.jsonl"
        if os.path.exists(file_path):
            with open(file_path) as f:
                line = f.readline()
                while line:
                    data.append(json.loads(line))
                    line = f.readline()
        else:
            logger.error(f"The iteration statistics data({file_path}) does not exist.")
        return data

    def get_request_stats(self) -> list[dict]:
        data = []
        file_path = f"{self.output_dir}/request.jsonl"
        if os.path.exists(file_path):
            with open(file_path) as f:
                line = f.readline()
                while line:
                    data.append(json.loads(line))
                    line = f.readline()
        else:
            logger.error(f"The request statistics data({file_path}) does not exist.")
        return data

    async def pause_generation(self):
        from sglang.srt.managers.io_struct import PauseGenerationReqInput

        await self._engine.tokenizer_manager.pause_generation(
            PauseGenerationReqInput(mode="in_place")
        )

    async def continue_generation(self):
        from sglang.srt.managers.io_struct import ContinueGenerationReqInput

        await self._engine.tokenizer_manager.continue_generation(
            ContinueGenerationReqInput()
        )

    def shutdown(self):
        logger.info("Attempting to shut down the SGLang backend engine.")
        return self._engine.shutdown()
