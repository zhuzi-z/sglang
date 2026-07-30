"""
vLLM Worker Hook - Hijacks the Worker class at the worker level only.

V6D-aware simulation strategy (merged from feat/vllm-pai + V6D additions):
- Inject head_dim=1 to shrink KV cache data volume (~128x reduction)
- Use real num_kv_heads so V6D page_size matches production structure
- No GPUModelRunner construction (avoids all CUDA dependencies)
- KV cache spec built from HF config with head_dim=1
- Mock execute_model output with KV connector lifecycle
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


def _inject_head_dim(vllm_config):
    """Inject head_dim=1 into HF config.

    This propagates through the entire system:
    - model_config.get_head_size() -> returns 1
    - FullAttentionSpec.page_size_bytes -> shrinks ~128x
    - V6D _compute_group_block_bytes -> uses same small page_size
    - KV cache allocation -> tiny CPU tensors
    """
    model_config = vllm_config.model_config
    hf_config = model_config.hf_text_config

    config_head_dim = getattr(hf_config, "head_dim", None)
    computed_head_dim = (
        hf_config.hidden_size // hf_config.num_attention_heads
    )
    # Use the larger value as effective head_dim for reduction ratio
    original_head_dim = max(config_head_dim or 0, computed_head_dim)

    hf_config.head_dim = 1
    logger.info(
        "[V6D Hijack] Injected head_dim=1 "
        "(config_head_dim=%s, computed_head_dim=%d, "
        "effective_original=%d, reduction=%dx)",
        config_head_dim,
        computed_head_dim,
        original_head_dim,
        original_head_dim,
    )
    return original_head_dim


def _build_kv_cache_spec(vllm_config) -> dict:
    """Build KV cache spec from HF model config with head_dim=1.

    Uses REAL num_kv_heads from the model config so that V6D object layout
    (page_size_bytes) matches between scheduler and worker sides.
    head_size=1 is the injected value from _inject_head_dim().

    Handles both pure-MHA models and hybrid models (e.g. Qwen3.5 with
    full_attention + linear_attention layers).
    """
    import torch
    import dataclasses
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
    hf_config = model_config.hf_text_config

    # Real num_kv_heads (divided by TP for per-shard spec)
    scheduler_config = None
    try:
        from sglang_simulator.simulation.manager import ConfigManager
        scheduler_config = ConfigManager.get_scheduler_config()
    except Exception:
        pass

    tp_size = scheduler_config.tp_size if scheduler_config else 1
    # Support both real ModelConfig (has method) and SimpleNamespace mocks
    if hasattr(model_config, "get_total_num_kv_heads"):
        total_num_kv_heads = model_config.get_total_num_kv_heads()
    else:
        total_num_kv_heads = getattr(hf_config, "num_key_value_heads",
                                     hf_config.num_attention_heads)
    num_kv_heads = max(total_num_kv_heads // tp_size, 1)

    # head_size=1 (injected by _inject_head_dim)
    head_size = 1

    block_size = cache_config.block_size
    dtype = model_config.dtype
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    # Determine layer types
    layer_types = getattr(hf_config, "layer_types", None)
    num_hidden_layers = hf_config.num_hidden_layers

    # Determine KV cache dtype
    cache_dtype = getattr(cache_config, "cache_dtype", "auto")
    if cache_dtype == "auto":
        pass  # keep model dtype
    elif cache_dtype == "fp8":
        dtype = torch.float8_e4m3fn

    # Detect FullAttentionSpec supported fields (varies across vLLM versions)
    _fa_fields = {f.name for f in dataclasses.fields(FullAttentionSpec)}
    _fa_kwargs = dict(num_kv_heads=num_kv_heads, head_size=head_size, dtype=dtype)
    if "use_mla" in _fa_fields:
        _fa_kwargs["use_mla"] = False
    if "block_size" in _fa_fields:
        _fa_kwargs["block_size"] = block_size

    full_attn_spec = FullAttentionSpec(**_fa_kwargs)

    if layer_types is None:
        # Pure MHA model - use layer name format matching vLLM convention
        return {
            f"model.layers.{i}": full_attn_spec
            for i in range(num_hidden_layers)
        }

    # Hybrid model: build per-layer specs with uniform page size.
    attn_page_size = full_attn_spec.page_size_bytes

    mamba_block_size = getattr(cache_config, "mamba_block_size", None)
    if mamba_block_size is None:
        mamba_block_size = block_size

    # Build MambaSpec
    mamba_kwargs = dict(
        block_size=mamba_block_size,
        shapes=((1, 1), (1, 1, 1)),
        dtypes=(dtype, dtype),
        page_size_padded=attn_page_size,
        mamba_cache_mode=getattr(cache_config, "mamba_cache_mode", "none"),
    )
    # Forward num_speculative_blocks (MTP/EAGLE) so light-mode runtime block
    # accounting matches the real deployment: the fork's MambaManager uses
    # _num_runtime_blocks = 1 + num_speculative_blocks per request.  Leaving
    # it at the default 0 makes each request hold 3 fewer blocks than real
    # (MTP k=3), underestimating pool eviction pressure and letting cached
    # mamba state snapshots live too long (task29: prefix-cache hit ratio
    # overestimated by up to +10.3pp on node1_0047).
    _mamba_fields = {f.name for f in dataclasses.fields(MambaSpec)}
    if "num_speculative_blocks" in _mamba_fields:
        spec_cfg = getattr(vllm_config, "speculative_config", None)
        num_spec_tokens = (
            getattr(spec_cfg, "num_speculative_tokens", 0) or 0
        ) if spec_cfg is not None else 0
        mamba_kwargs["num_speculative_blocks"] = num_spec_tokens
    mamba_type_field = next(
        (f for f in dataclasses.fields(MambaSpec) if f.name == "mamba_type"), None
    )
    if mamba_type_field is not None:
        if mamba_type_field.type == str or mamba_type_field.type == "str":
            mamba_kwargs["mamba_type"] = "gdn_attention"
        elif _mamba_gdn_type is not None:
            mamba_kwargs["mamba_type"] = _mamba_gdn_type

    mamba_spec = MambaSpec(**mamba_kwargs)

    kv_cache_spec: dict = {}
    for i, layer_type in enumerate(layer_types):
        layer_name = f"model.layers.{i}"
        if layer_type == "full_attention":
            kv_cache_spec[layer_name] = full_attn_spec
        elif layer_type == "linear_attention":
            kv_cache_spec[layer_name] = mamba_spec
        else:
            kv_cache_spec[layer_name] = full_attn_spec

    logger.info(
        "[V6D Hijack] Built KV cache spec: %d layers, "
        "num_kv_heads=%d (total=%d, tp=%d), head_size=%d, block_size=%d, "
        "page_size=%d bytes",
        len(kv_cache_spec), num_kv_heads, total_num_kv_heads, tp_size,
        head_size, block_size, attn_page_size,
    )
    return kv_cache_spec


class C_VLLMWorkerHook(BaseHook):
    """Hook Worker to run on CPU without model/CUDA dependencies.

    V6D-aware: preserves real V6D daemon connectivity while removing
    all CUDA operations.
    """

    HOOK_CLASS_NAME = "Worker"
    HOOK_MODULE_NAME = r"vllm\.v1\.worker\.(gpu_worker|worker)"
    REGEX = True

    @classmethod
    def hook(cls, target):
        # Cache imports at hook-install time
        from vllm.v1.outputs import ModelRunnerOutput as _ModelRunnerOutput

        _KVConnectorOutput = None
        try:
            from vllm.v1.outputs import KVConnectorOutput as _KVConnectorOutput
        except ImportError:
            pass

        _has_kv_transfer_group = None
        _get_kv_transfer_group = None
        try:
            from vllm.distributed.kv_transfer import (
                has_kv_transfer_group as _has_kv_transfer_group,
                get_kv_transfer_group as _get_kv_transfer_group,
            )
        except ImportError:
            pass

        def override_init_device(self):
            """Minimal init: distributed env + head_dim injection + stub."""
            import torch
            import sys
            try:
                from vllm.v1.worker.gpu_worker import (
                    init_worker_distributed_environment,
                    init_workspace_manager,
                    set_random_seed,
                )
            except ImportError:
                from vllm.v1.worker.worker import (
                    init_worker_distributed_environment,
                    init_workspace_manager,
                    set_random_seed,
                )

            # PAI-vLLM's init_distributed_environment needs vllm_config context
            try:
                from vllm.config.vllm import set_current_vllm_config
                with set_current_vllm_config(self.vllm_config):
                    init_worker_distributed_environment(
                        self.vllm_config,
                        self.rank,
                        self.distributed_init_method,
                        self.local_rank,
                        "gloo",
                    )
            except (ImportError, TypeError):
                # Fallback: older vLLM without set_current_vllm_config
                init_worker_distributed_environment(
                    self.vllm_config,
                    self.rank,
                    self.distributed_init_method,
                    self.local_rank,
                    "gloo",
                )
            set_random_seed(self.vllm_config.model_config.seed)

            # Inject head_dim=1 for V6D simulation
            _inject_head_dim(self.vllm_config)

            self.device = torch.device("cpu")

            # Stub model_runner
            kv_spec = _build_kv_cache_spec(self.vllm_config)
            self.model_runner = _ModelRunnerStub(kv_spec=kv_spec)

            self.init_snapshot = None
            self.requested_memory = 0
            init_workspace_manager(self.device, 1)
            logger.info("[vLLM Hijack] Worker.init_device: stub initialized")

        def override_load_model(self, *args, **kwargs):
            """Skip model loading entirely."""
            import sys
            logger.info("[vLLM Hijack] Worker.load_model: skipped")

        def override_determine_available_memory(self):
            """Return fake GPU memory to satisfy block calculation."""
            try:
                from sglang_simulator.simulation.manager import ConfigManager
                from sglang_simulator.simulation.vllm.utils import (
                    resolve_model_info, resolve_scheduler_config,
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

                # In CPU simulation, cap reported memory to avoid OOM from block
                # metadata structures (prefix tree, block tables, free queues).
                # Each block needs ~100-200 bytes of metadata in EngineCore.
                # Physical RAM limit: use min(simulated, 4 GiB) to keep block
                # count under ~1M, which needs ~200 MB metadata.
                import psutil
                phys_ram = psutil.virtual_memory().total
                # Allow at most 25% of physical RAM for KV cache simulation
                max_safe_bytes = int(phys_ram * 0.25)
                if available_bytes > max_safe_bytes:
                    logger.info(
                        "[vLLM Hijack] Worker.determine_available_memory: "
                        "capping simulated %d bytes (%.2f GiB) → %d bytes "
                        "(%.2f GiB) to fit physical RAM (%d bytes)",
                        available_bytes, available_bytes / (1 << 30),
                        max_safe_bytes, max_safe_bytes / (1 << 30), phys_ram,
                    )
                    available_bytes = max_safe_bytes
                else:
                    logger.info(
                        "[vLLM Hijack] Worker.determine_available_memory: "
                        "profiled %d bytes (%.2f GiB)",
                        available_bytes, available_bytes / (1 << 30),
                    )
                return available_bytes
            except Exception:
                return 80 * (1 << 30)  # 80 GiB fallback

        def override_get_kv_cache_spec(self):
            """Return pre-built KV cache spec."""
            return self.model_runner.get_kv_cache_spec()

        def override_initialize_from_config(self, kv_cache_config):
            """Init KV connector and allocate CPU KV cache tensors."""
            import torch
            from vllm.distributed.kv_transfer import (
                ensure_kv_transfer_initialized,
                has_kv_transfer_group,
                get_kv_transfer_group,
            )

            # PAI-vLLM passes kv_cache_config as a list (one per TP shard)
            if isinstance(kv_cache_config, (list, tuple)):
                kv_cache_config = kv_cache_config[0]

            self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
            num_blocks = kv_cache_config.num_blocks

            # Initialize the hybrid worker loop without touching source files.
            # Native HybridConnector expects engine_proxy._g_worker_loop to exist
            # during connector construction.  In CPU simulation we avoid the full
            # worker_init path because it validates GPU-oriented HybridWorker group
            # layouts before our no-op data-plane hooks can take over.
            try:
                import asyncio
                import threading
                import vllm.v1.hybrid_connector.engine_proxy as _engine_proxy
                _orig_worker_init = _engine_proxy.worker_init

                def _ensure_cpu_worker_loop():
                    if getattr(_engine_proxy, "_g_worker_loop", None) is not None:
                        return
                    loop = asyncio.new_event_loop()

                    def _run_loop():
                        asyncio.set_event_loop(loop)
                        loop.run_forever()

                    thread = threading.Thread(
                        target=_run_loop,
                        name="hybridworker-cpu-loop",
                        daemon=True,
                    )
                    thread.start()
                    _engine_proxy._g_worker_loop = loop
                    logger.info(
                        "[vLLM Hijack] installed CPU hybrid worker loop"
                    )

                def _cpu_worker_init(vllm_config, local_rank):
                    _ensure_cpu_worker_loop()
                    return None

                _ensure_cpu_worker_loop()
                _engine_proxy.worker_init = _cpu_worker_init
            except (ImportError, AttributeError):
                _orig_worker_init = None

            # Initialize KV connector (V6D or Mock)
            try:
                ensure_kv_transfer_initialized(
                    self.vllm_config, kv_cache_config, self.local_rank
                )
            except TypeError:
                # Public vLLM version: no local_rank arg
                ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
            finally:
                if _orig_worker_init is not None:
                    _engine_proxy.worker_init = _orig_worker_init

            # Allocate CPU KV cache tensors for V6D mmap
            kv_caches: dict = {}
            # CPU simulation: allocate MINIMAL tensors (1 page each) to avoid OOM.
            # The full num_blocks count is preserved for scheduling/prefix logic,
            # but we don't need actual KV data storage in simulation mode.
            if hasattr(kv_cache_config, "kv_cache_tensors") and \
                    kv_cache_config.kv_cache_tensors:
                for kv_tensor in kv_cache_config.kv_cache_tensors:
                    # Allocate only 1 page instead of full size
                    minimal_size = min(kv_tensor.size, 4096)
                    tensor = torch.zeros(
                        minimal_size, dtype=torch.int8, device="cpu"
                    )
                    for layer_name in kv_tensor.shared_by:
                        kv_caches[layer_name] = tensor
                logger.info(
                    "[vLLM Hijack] Allocated %d MINIMAL CPU KV cache tensors "
                    "(num_blocks=%d, simulated_bytes=%d, actual_bytes=minimal)",
                    len(kv_cache_config.kv_cache_tensors), num_blocks,
                    sum(t.size for t in kv_cache_config.kv_cache_tensors),
                )
            else:
                kv_spec = self.model_runner.get_kv_cache_spec()
                for layer_name, spec in kv_spec.items():
                    # Allocate only 1 page instead of num_blocks * page_size
                    minimal_size = min(spec.page_size_bytes, 4096)
                    tensor = torch.zeros(
                        minimal_size, dtype=torch.int8, device="cpu"
                    )
                    kv_caches[layer_name] = tensor
                logger.info(
                    "[vLLM Hijack] Allocated %d MINIMAL CPU KV cache tensors "
                    "(num_blocks=%d, page_size=%d, actual_alloc=minimal)",
                    len(kv_caches), num_blocks,
                    spec.page_size_bytes if kv_spec else 0,
                )

            # Register with KV connector
            if has_kv_transfer_group():
                kv_transfer_group = get_kv_transfer_group()
                kv_transfer_group.register_kv_caches(kv_caches)
                logger.info("[vLLM Hijack] KV caches registered with connector")

            logger.info(
                "[vLLM Hijack] Worker.initialize_from_config: "
                "V6D-aware init complete, num_blocks=%d", num_blocks,
            )

        def override_compile_or_warm_up_model(self):
            """Skip compilation and warmup entirely."""
            logger.info("[vLLM Hijack] Worker.compile_or_warm_up_model: skipped")
            try:
                from vllm.v1.worker.worker_base import CompilationTimes
                return CompilationTimes(language_model=0.0, encoder=0.0)
            except (ImportError, ModuleNotFoundError, TypeError):
                return None

        def override_get_supported_tasks(self):
            """Return generate task (default for causal LM)."""
            return ("generate",)

        def override_execute_model(self, scheduler_output):
            """Return mock ModelRunnerOutput with KV connector lifecycle."""
            _has_kvt = _has_kv_transfer_group
            _kvt_result = _has_kvt() if _has_kvt else None
            _meta = getattr(scheduler_output, "kv_connector_metadata", None)
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens

            # KV connector pre-forward
            kv_connector = None
            try:
                if _has_kv_transfer_group and _has_kv_transfer_group():
                    kv_connector = _get_kv_transfer_group()
                    kv_connector_metadata = scheduler_output.kv_connector_metadata
                    logger.debug(
                        "[V6D Hijack] execute_model: kv_connector=%s metadata=%s",
                        type(kv_connector).__name__,
                        type(kv_connector_metadata).__name__
                        if kv_connector_metadata is not None else None,
                    )
                    if kv_connector_metadata is not None:
                        if hasattr(kv_connector, "handle_preemptions"):
                            kv_connector.handle_preemptions(kv_connector_metadata)
                        kv_connector.bind_connector_metadata(kv_connector_metadata)
                        kv_connector.start_load_kv(None)
                else:
                    logger.debug(
                        "[V6D Hijack] execute_model: no kv_transfer_group"
                    )
            except Exception:
                logger.exception(
                    "[V6D Hijack] execute_model: KV pre-forward lifecycle failed"
                )
                kv_connector = None

            # Build mock output
            req_ids = list(num_scheduled_tokens.keys()) if num_scheduled_tokens else []
            # A chunked-prefill forward only produces a sampled token when the
            # request has reached the end of its prompt.  Returning a token for
            # every partial chunk makes max_tokens=1 requests finish after the
            # first 8192-token chunk and silently skips the remaining prompt.
            try:
                from sglang_simulator.simulation.vllm.kv_connector import (
                    get_scheduler_ref,
                )

                scheduler = get_scheduler_ref()
            except Exception:
                scheduler = None

            if scheduler is None:
                raise RuntimeError(
                    "vLLM simulator scheduler reference is unavailable"
                )

            sampled_token_ids = []
            for req_id in req_ids:
                request = scheduler.requests.get(req_id)
                if request is None:
                    raise RuntimeError(
                        f"scheduled request {req_id!r} is missing from scheduler"
                    )
                prompt_tokens = getattr(request, "num_prompt_tokens", None)
                computed_tokens = getattr(request, "num_computed_tokens", 0)
                prompt_complete = (
                    prompt_tokens is None or computed_tokens >= prompt_tokens
                )
                sampled_token_ids.append([1] if prompt_complete else [])

            import dataclasses as _dc
            mro_kwargs = dict(
                req_ids=req_ids,
                req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
                sampled_token_ids=sampled_token_ids,
                logprobs=None,
                prompt_logprobs_dict={},
            )
            _mro_fields = {f.name for f in _dc.fields(_ModelRunnerOutput)}
            if "kv_lens" in _mro_fields:
                mro_kwargs["kv_lens"] = [0] * len(req_ids)
            if "pooler_output" in _mro_fields:
                mro_kwargs["pooler_output"] = [None] * len(req_ids)
            output = _ModelRunnerOutput(**mro_kwargs)

            # KV connector post-forward
            if kv_connector is not None and _KVConnectorOutput is not None:
                try:
                    kv_connector.wait_for_save()
                    kv_output = _KVConnectorOutput()
                    finished_req_ids = getattr(scheduler_output, "finished_req_ids", set())
                    kv_output.finished_sending, kv_output.finished_recving = (
                        kv_connector.get_finished(finished_req_ids)
                    )
                    if kv_output.finished_sending or kv_output.finished_recving:
                        logger.info(
                            "[V6D Hijack] execute_model: finished_sending=%s "
                            "finished_recving=%s",
                            sorted(kv_output.finished_sending or []),
                            sorted(kv_output.finished_recving or []),
                        )
                    else:
                        logger.debug(
                            "[V6D Hijack] execute_model: finished_sending=[] "
                            "finished_recving=[]"
                        )
                    if hasattr(kv_connector, "build_connector_worker_meta"):
                        kv_output.kv_connector_worker_meta = (
                            kv_connector.build_connector_worker_meta()
                        )
                    kv_connector.clear_connector_metadata()
                    output.kv_connector_output = kv_output
                except Exception:
                    logger.exception(
                        "[V6D Hijack] execute_model: KV post-forward lifecycle failed"
                    )

            self._last_model_output = output
            return output

        def override_sample_tokens(self, grammar_output):
            """Return the mock output built by execute_model."""
            return self._last_model_output

        def override_get_attn_backends_type(self):
            """Return empty list - no real attention backends in simulation."""
            return []

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
        def override_reset_mm_cache(self):
            """No-op: no real model_runner to reset."""
            pass

        def override_initialize_cache(self, num_gpu_blocks, num_cpu_blocks):
            """Store block counts only."""
            self.cache_config.num_gpu_blocks = num_gpu_blocks
            self.cache_config.num_cpu_blocks = num_cpu_blocks

        def override_initialize_kv_transfer(self):
            """No-op in simulation."""
            pass

        target.sleep = override_sleep
        target.wake_up = override_wake_up
        target.get_attn_backends_type = override_get_attn_backends_type
        target.reset_mm_cache = override_reset_mm_cache
        target.initialize_cache = override_initialize_cache
        target.initialize_kv_transfer = override_initialize_kv_transfer
