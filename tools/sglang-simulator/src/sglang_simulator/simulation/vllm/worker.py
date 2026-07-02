"""
vLLM Worker Hook - Hijacks the Worker class at the worker level only.

All model_runner interactions are short-circuited here:
- No GPUModelRunner construction (avoids all CUDA dependencies)
- KV cache spec built from HF config directly
- Mock execute_model output with KV connector lifecycle
- Works identically on GPU and CPU (no branching)
"""

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger()


class _ModelRunnerStub:
    """Picklable stub for model_runner (must be module-level for multiprocess)."""

    def __init__(self, kv_spec=None):
        self._kv_spec = kv_spec
        self.model_memory_usage = 0

    def get_kv_cache_spec(self):
        return self._kv_spec

    def __getattr__(self, name):
        # Any attribute not explicitly defined returns a no-op callable
        if name.startswith('_'):
            raise AttributeError(name)
        return _noop


def _noop(*args, **kwargs):
    """Module-level no-op function (picklable)."""
    return None


def _build_kv_cache_spec(vllm_config) -> dict:
    """Build KV cache spec from HF model config (no model construction).

    Handles both pure-MHA models and hybrid models (e.g. Qwen3.5 with
    full_attention + linear_attention layers) by inspecting the
    hf_text_config's `layer_types` field.

    All KV dimensions are set to 1 (minimal intervention strategy): this
    preserves block count accuracy while avoiding memory allocation and
    page-size alignment issues that would normally require CUDA-dependent
    platform logic.
    """
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    # MambaAttentionBackendEnum moved between vLLM versions
    try:
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
        _mamba_gdn_type = MambaAttentionBackendEnum.GDN_ATTN
    except (ImportError, ModuleNotFoundError):
        try:
            from vllm.attention.backends.registry import MambaAttentionBackendEnum
            _mamba_gdn_type = MambaAttentionBackendEnum.GDN_ATTN
        except (ImportError, ModuleNotFoundError):
            _mamba_gdn_type = None

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    # Use hf_text_config to handle nested configs (e.g. multimodal wrappers)
    hf_config = model_config.hf_text_config
    block_size = cache_config.block_size
    dtype = model_config.dtype
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    num_layers = hf_config.num_hidden_layers

    # Minimal-dim FullAttentionSpec: head_num=1, head_dim=1
    # Preserves block structure without real memory allocation.
    full_attn_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=dtype,
    )

    # Detect hybrid model via layer_types
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is None:
        # Pure MHA model
        return {
            f"model.layers.{i}.self_attn.attn": full_attn_spec
            for i in range(num_layers)
        }

    # Hybrid model: build per-layer specs with uniform page size.
    # MambaSpec shapes set to (1,1) to match FullAttentionSpec page size.
    # page_size_padded ensures uniform page size across groups.
    attn_page_size = full_attn_spec.page_size_bytes

    mamba_block_size = cache_config.mamba_block_size
    if mamba_block_size is None:
        mamba_block_size = block_size

    # Build MambaSpec - constructor varies between vLLM versions:
    # - Public vLLM v0.23: MambaSpec(shapes, dtypes, block_size, ..., mamba_type=Enum)
    # - Modified vLLM: MambaSpec(block_size, shapes, dtypes, ..., mamba_type=str)
    import dataclasses
    mamba_kwargs = dict(
        block_size=mamba_block_size,
        shapes=((1, 1), (1, 1, 1)),
        dtypes=(dtype, dtype),
        page_size_padded=attn_page_size,
        mamba_cache_mode=getattr(cache_config, "mamba_cache_mode", "none"),
    )
    # Determine mamba_type based on what the class expects
    mamba_type_field = next(
        (f for f in dataclasses.fields(MambaSpec) if f.name == "mamba_type"), None
    )
    if mamba_type_field is not None:
        if mamba_type_field.type == str or mamba_type_field.type == "str":
            # String-based mamba_type (modified vLLM)
            mamba_kwargs["mamba_type"] = "gdn_attention"
        elif _mamba_gdn_type is not None:
            # Enum-based mamba_type (public vLLM)
            mamba_kwargs["mamba_type"] = _mamba_gdn_type

    mamba_spec = MambaSpec(**mamba_kwargs)

    kv_cache_spec: dict = {}
    for i, layer_type in enumerate(layer_types):
        if layer_type == "full_attention":
            kv_cache_spec[f"model.layers.{i}.self_attn.attn"] = full_attn_spec
        elif layer_type == "linear_attention":
            kv_cache_spec[f"model.layers.{i}.linear_attn"] = mamba_spec
        else:
            # Unknown layer type - default to full attention
            kv_cache_spec[f"model.layers.{i}.self_attn.attn"] = full_attn_spec
    return kv_cache_spec


class C_VLLMWorkerHook(BaseHook):
    """Hook the vLLM Worker class in vllm.v1.worker.gpu_worker.

    Strategy: override all worker methods that would call into model_runner.
    No GPUModelRunner is constructed — a minimal stub provides get_kv_cache_spec.
    This eliminates all CUDA dependencies and works on both GPU and CPU.
    """

    HOOK_CLASS_NAME = "Worker"
    HOOK_MODULE_NAME = "vllm.v1.worker.gpu_worker"

    @classmethod
    def hook(cls, target):
        # Cache imports at hook-install time
        from vllm.v1.outputs import ModelRunnerOutput as _ModelRunnerOutput
        from vllm.v1.outputs import KVConnectorOutput as _KVConnectorOutput
        from vllm.distributed.kv_transfer import (
            has_kv_transfer_group as _has_kv_transfer_group,
            get_kv_transfer_group as _get_kv_transfer_group,
        )

        def override_init_device(self):
            """Minimal init: distributed env + stub model_runner."""
            import torch
            from vllm.v1.worker.gpu_worker import (
                init_worker_distributed_environment,
                init_workspace_manager,
                set_random_seed,
            )

            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                "gloo",
            )
            set_random_seed(self.vllm_config.model_config.seed)
            self.device = torch.device("cpu")

            # Stub model_runner — provides get_kv_cache_spec; all other
            # method calls are no-ops. Uses module-level class for picklability.
            kv_spec = _build_kv_cache_spec(self.vllm_config)
            self.model_runner = _ModelRunnerStub(kv_spec=kv_spec)

            self.init_snapshot = None
            self.requested_memory = 0
            init_workspace_manager(self.device, 1)
            logger.info("[vLLM Hijack] Worker.init_device: stub initialized")

        def override_load_model(self, *args, **kwargs):
            """Skip model loading entirely (no real model needed)."""
            logger.info("[vLLM Hijack] Worker.load_model: skipped")

        def override_determine_available_memory(self) -> int:
            """Estimate available memory for KV cache using simulator profiling.

            Uses model weights + device HBM to compute realistic available bytes.
            If num_gpu_blocks_override is set, vLLM clamps the block count anyway.
            """
            from sglang_simulator.simulation.manager import ConfigManager
            from sglang_simulator.simulation.vllm.utils import (
                resolve_model_info,
                resolve_scheduler_config,
            )
            from sglang_simulator.simulation.utils import (
                profile_device_available_bytes,
            )

            model = resolve_model_info(self.vllm_config.model_config)
            ConfigManager.set_model_info(model)

            hw = ConfigManager.get_accelerator_info()

            sched_config = resolve_scheduler_config(self.vllm_config)
            ConfigManager.set_scheduler_config(sched_config)

            available_bytes = profile_device_available_bytes(model, hw, sched_config)
            logger.info(
                "[vLLM Hijack] Worker.determine_available_memory: "
                "profiled %d bytes (%.2f GiB)",
                available_bytes,
                available_bytes / (1 << 30),
            )
            return available_bytes

        def override_get_kv_cache_spec(self):
            """Delegate to stub (built from HF config)."""
            return self.model_runner.get_kv_cache_spec()

        def override_initialize_from_config(self, kv_cache_config):
            """Init KV connector only — no KV cache tensor allocation."""
            from vllm.distributed.kv_transfer import (
                ensure_kv_transfer_initialized,
            )

            self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
            # Skip worker_init (hybrid_connector) which requires CUDA
            try:
                import vllm.v1.hybrid_connector.engine_proxy as _engine_proxy
                _orig_worker_init = _engine_proxy.worker_init
                _engine_proxy.worker_init = lambda *a, **kw: None
            except (ImportError, AttributeError):
                _orig_worker_init = None
            try:
                ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
            finally:
                if _orig_worker_init is not None:
                    _engine_proxy.worker_init = _orig_worker_init
            logger.info(
                "[vLLM Hijack] Worker.initialize_from_config: "
                "KV connector initialized, num_blocks=%d",
                kv_cache_config.num_blocks,
            )

        def override_compile_or_warm_up_model(self):
            """Skip compilation and warmup entirely."""
            logger.info("[vLLM Hijack] Worker.compile_or_warm_up_model: skipped")
            # Return type varies by vLLM version:
            # - Public v0.23: returns CompilationTimes dataclass
            # - Modified versions: returns None
            try:
                from vllm.v1.worker.worker_base import CompilationTimes
                return CompilationTimes(language_model=0.0, encoder=0.0)
            except ImportError:
                return None

        def override_get_supported_tasks(self):
            """Return generate task (default for causal LM)."""
            return ("generate",)

        def override_execute_model(self, scheduler_output):
            """Return mock ModelRunnerOutput with KV connector lifecycle."""
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens

            # KV connector pre-forward
            kv_connector = None
            try:
                if _has_kv_transfer_group():
                    kv_connector = _get_kv_transfer_group()
                    kv_connector_metadata = scheduler_output.kv_connector_metadata
                    if kv_connector_metadata is not None:
                        kv_connector.handle_preemptions(kv_connector_metadata)
                        kv_connector.bind_connector_metadata(kv_connector_metadata)
            except Exception:
                kv_connector = None

            # Build mock output
            req_ids = list(num_scheduled_tokens.keys()) if num_scheduled_tokens else []
            # Build kwargs for ModelRunnerOutput (cross-version compat)
            mro_kwargs = dict(
                req_ids=req_ids,
                req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
                sampled_token_ids=[[1] for _ in req_ids],
                logprobs=None,
                prompt_logprobs_dict={},
            )
            # Newer vLLM versions require kv_lens and pooler_output
            import dataclasses as _dc
            _mro_fields = {f.name for f in _dc.fields(_ModelRunnerOutput)}
            if "kv_lens" in _mro_fields:
                mro_kwargs["kv_lens"] = [0] * len(req_ids)
            if "pooler_output" in _mro_fields:
                mro_kwargs["pooler_output"] = [None] * len(req_ids)
            output = _ModelRunnerOutput(**mro_kwargs)

            # KV connector post-forward
            if kv_connector is not None:
                kv_output = _KVConnectorOutput()
                finished_req_ids = getattr(scheduler_output, "finished_req_ids", set())
                kv_output.finished_sending, kv_output.finished_recving = (
                    kv_connector.get_finished(finished_req_ids)
                )
                kv_output.kv_connector_worker_meta = (
                    kv_connector.build_connector_worker_meta()
                )
                kv_connector.clear_connector_metadata()
                output.kv_connector_output = kv_output

            # Store for sample_tokens (used in batch_queue/async mode)
            self._last_model_output = output
            return output

        def override_sample_tokens(self, grammar_output):
            """Return the mock output built by execute_model.

            In batch_queue mode, sample_tokens is called separately from
            execute_model. Return the stored output from the last execute_model
            call (both run on the same worker sequentially).
            """
            return self._last_model_output

        def override_sleep(self, level=1):
            pass

        def override_wake_up(self, tags=None):
            pass

        target.init_device = override_init_device
        target.load_model = override_load_model
        target.determine_available_memory = override_determine_available_memory
        target.get_kv_cache_spec = override_get_kv_cache_spec
        target.get_supported_tasks = override_get_supported_tasks
        target.initialize_from_config = override_initialize_from_config
        target.compile_or_warm_up_model = override_compile_or_warm_up_model
        target.execute_model = override_execute_model
        target.sample_tokens = override_sample_tokens
        target.sleep = override_sleep
        target.wake_up = override_wake_up
