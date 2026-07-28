"""
vLLM Simulation Startup - Installs all hooks needed to hijack the vLLM
inference framework for simulation purposes.

IMPORTANT: init_hook() must be called BEFORE any `import vllm.*` statement.
"""

import json
import os


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_enabled(*names: str) -> bool:
    return any(os.environ.get(name, "").strip().lower() in _TRUE_VALUES for name in names)


def _env_default_enabled(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in _FALSE_VALUES


def _decode_kv_transfer_params(value):
    """Decode explicitly provided kv_transfer_params from request inputs."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            return _decode_kv_transfer_params(json.loads(value))
        except Exception:
            return {}
    if isinstance(value, (list, tuple)):
        merged = {}
        for item in value:
            merged.update(_decode_kv_transfer_params(item))
        return merged
    return {}


def _extract_explicit_kv_transfer_params(kwargs: dict) -> dict:
    """Collect KV params from explicit fields only; never inspect request_id."""
    if not isinstance(kwargs, dict):
        return {}

    kv_params = _decode_kv_transfer_params(kwargs.get("kv_transfer_params"))
    encoder_extra_args = kwargs.get("encoder_extra_args")
    if isinstance(encoder_extra_args, dict):
        kv_params.update(
            _decode_kv_transfer_params(
                encoder_extra_args.get("kv_transfer_params")
            )
        )

    for key in (
        "do_remote_decode",
        "do_remote_prefill",
        "ali_llumnix_disagg",
        "ali_max_computed_tokens",
        "remote_host",
        "remote_port",
        "transfer_id",
        "remote_engine_id",
        "remote_bootstrap_addr",
    ):
        if key in kwargs and key not in kv_params:
            kv_params[key] = kwargs[key]
    return kv_params


def _merge_kv_params_into_encoder_extra_args(kwargs: dict) -> dict:
    kv_params = _extract_explicit_kv_transfer_params(kwargs)
    if not kv_params:
        return {}
    encoder_extra_args = dict(kwargs.get("encoder_extra_args") or {})
    merged = _decode_kv_transfer_params(
        encoder_extra_args.get("kv_transfer_params")
    )
    merged.update(kv_params)
    encoder_extra_args["kv_transfer_params"] = merged
    kwargs["encoder_extra_args"] = encoder_extra_args
    return merged


def _merge_kv_params_into_sampling_params(kwargs: dict) -> dict:
    kv_params = {}
    kv_params.update(
        _extract_explicit_kv_transfer_params(
            kwargs.get("kwargs_for_epd_transfer") or {}
        )
    )
    kv_params.update(_extract_explicit_kv_transfer_params(kwargs))

    sampling_params = dict(kwargs.get("sampling_params") or {})
    extra_args = dict(sampling_params.get("extra_args") or {})
    existing = _decode_kv_transfer_params(extra_args.get("kv_transfer_params"))
    existing.update(kv_params)
    if not existing:
        return {}

    extra_args["kv_transfer_params"] = existing
    sampling_params["extra_args"] = extra_args
    kwargs["sampling_params"] = sampling_params
    return existing


def _install_dashllm_kv_transfer_hook(original_launch_v6d=None) -> None:
    """Install non-invasive DashServing/vLLM kv_transfer_params passthrough."""
    try:
        import dashllm.core.backend._backend_vllm as backend_vllm
    except Exception:
        return

    if (
        original_launch_v6d is not None
        and getattr(backend_vllm, "launch_v6d", None) is original_launch_v6d
    ):
        try:
            from dashllm.utils import vineyard as dashllm_vineyard
            backend_vllm.launch_v6d = dashllm_vineyard.launch_v6d
        except Exception:
            pass

    backend_cls = getattr(backend_vllm, "_LLMBackend4vLLM", None)
    if backend_cls is not None and not getattr(
        backend_cls,
        "_sglang_simulator_kv_transfer_params_hook",
        False,
    ):
        original_generate = backend_cls.generate

        def _patched_generate(self, model, **kwargs):
            _merge_kv_params_into_encoder_extra_args(kwargs)
            return original_generate(self, model, **kwargs)

        backend_cls.generate = _patched_generate
        backend_cls._sglang_simulator_kv_transfer_params_hook = True

    try:
        import dashllm.core.backend.engine._vllm_v1 as vllm_v1
    except Exception:
        vllm_v1 = None

    engine_cls = getattr(vllm_v1, "vLLMEngine", None) if vllm_v1 else None
    if engine_cls is not None and not getattr(
        engine_cls,
        "_sglang_simulator_vllm_engine_kv_params_hook",
        False,
    ):
        original_engine_generate = engine_cls.generate
        original_engine_generate_impl = getattr(engine_cls, "_generate_impl", None)

        def _patched_engine_generate(self, *args, **kwargs):
            _merge_kv_params_into_sampling_params(kwargs)
            yield from original_engine_generate(self, *args, **kwargs)

        engine_cls.generate = _patched_engine_generate

        if original_engine_generate_impl is not None:
            def _patched_engine_generate_impl(self, *args, **kwargs):
                _merge_kv_params_into_sampling_params(kwargs)
                yield from original_engine_generate_impl(self, *args, **kwargs)

            engine_cls._generate_impl = _patched_engine_generate_impl

        engine_cls._sglang_simulator_vllm_engine_kv_params_hook = True


def _install_v6d_ipc_hook() -> None:
    """Patch v6d startup so CPU-only package/PTH deployments can expose IPC."""
    import logging

    logger = logging.getLogger("sglang_simulator")
    logger.info("[v6d_ipc_hook] begin")

    try:
        import v6d.common.transfer as transfer
    except Exception as exc:
        logger.exception("[v6d_ipc_hook] import v6d.common.transfer failed: %r", exc)
        return

    def _skip_srpc_init(base_addr: int, size: int, *args, **kwargs) -> None:
        logger.info(
            "[v6d_ipc_hook] skip SRPC init base_addr=%s size=%s",
            base_addr,
            size,
        )
        return None

    transfer.init_srpc_transfer = _skip_srpc_init
    if hasattr(transfer, "init_srpc"):
        transfer.init_srpc = _skip_srpc_init
    if hasattr(transfer, "init_srpc_"):
        transfer.init_srpc_ = _skip_srpc_init
    logger.info("[v6d_ipc_hook] patched v6d.common.transfer SRPC entrypoints")

    if hasattr(transfer, "init_mmap") and hasattr(
        transfer, "init_transfer_engine_client"
    ):
        def _init_transfer_engine_client_without_srpc(fd: int, size: int) -> int:
            logger.info(
                "[v6d_ipc_hook] init transfer engine client mmap only fd=%s size=%s",
                fd,
                size,
            )
            return transfer.init_mmap(fd, size)

        transfer.init_transfer_engine_client = _init_transfer_engine_client_without_srpc
        logger.info("[v6d_ipc_hook] patched transfer engine client to mmap only")

    try:
        import v6d.lite.common.transfer_engine as transfer_engine
    except Exception as exc:
        logger.info("[v6d_ipc_hook] import lite transfer_engine skipped: %r", exc)
        transfer_engine = None

    if transfer_engine is not None:
        if hasattr(transfer_engine, "init_srpc"):
            transfer_engine.init_srpc = _skip_srpc_init
        if hasattr(transfer_engine, "init_srpc_"):
            transfer_engine.init_srpc_ = _skip_srpc_init
        if hasattr(transfer_engine, "init_srpc_transfer"):
            transfer_engine.init_srpc_transfer = _skip_srpc_init
        logger.info("[v6d_ipc_hook] patched lite transfer_engine SRPC entrypoints")

    if _env_enabled("SGLANG_SIMULATOR_V6D_IPC_PATCH_MMAP_MANAGER"):
        try:
            from v6d.client.peers.vineyard import mmap_manager
        except Exception as exc:
            logger.info("[v6d_ipc_hook] import mmap_manager skipped: %r", exc)
            mmap_manager = None

        if mmap_manager is not None and not getattr(
            mmap_manager.ClientV6dMmapManager,
            "_sglang_simulator_cpu_mmap_hook",
            False,
        ):
            def _create_mmap_without_srpc(
                self,
                socket_path: str,
                is_lazy_strategy: bool,
            ):
                from v6d.common.transfer import _vineyard_connect
                from v6d.lite.common.transfer_engine import init_mmap

                fd, map_size, offset, sock = _vineyard_connect(socket_path)
                base_addr = init_mmap(fd, map_size)
                os.close(fd)
                return mmap_manager.MmapInfo(
                    socket_path=socket_path,
                    fd=fd,
                    base_addr=base_addr,
                    map_size=map_size,
                    refcount=1,
                    socket=sock,
                )

            mmap_manager.ClientV6dMmapManager._create_mmap = _create_mmap_without_srpc
            mmap_manager.ClientV6dMmapManager._sglang_simulator_cpu_mmap_hook = True
            logger.info("[v6d_ipc_hook] patched ClientV6dMmapManager._create_mmap")
    else:
        logger.info("[v6d_ipc_hook] mmap_manager patch disabled; enable SGLANG_SIMULATOR_V6D_IPC_PATCH_MMAP_MANAGER=1 to test it")

    if _env_default_enabled("SGLANG_SIMULATOR_V6D_IPC_PATCH_VINEYARD_PEER"):
        try:
            from v6d.server.peers.vineyard.peer import VineyardPeer
        except Exception as exc:
            logger.info("[v6d_ipc_hook] import VineyardPeer skipped: %r", exc)
            VineyardPeer = None

        if VineyardPeer is not None and not getattr(
            VineyardPeer,
            "_sglang_simulator_cpu_ipc_hook",
            False,
        ):
            original_init = VineyardPeer.__init__

            def _patched_init(
                self,
                argc=0,
                argv=None,
                tracker_url=None,
                tracker_key_prefix=None,
                lazy_load=True,
            ):
                patched_argv = list(argv) if argv is not None else None
                if patched_argv is not None:
                    has_rpc_flag = any(
                        arg.startswith("-rpc=") or arg.startswith("--rpc=")
                        for arg in patched_argv
                    )
                    if not has_rpc_flag:
                        patched_argv.append("-rpc=false")
                        argc = len(patched_argv)
                return original_init(
                    self,
                    argc,
                    patched_argv,
                    tracker_url,
                    tracker_key_prefix,
                    lazy_load,
                )

            VineyardPeer.__init__ = _patched_init
            VineyardPeer._sglang_simulator_cpu_ipc_hook = True
            logger.info("[v6d_ipc_hook] patched VineyardPeer.__init__")
    else:
        logger.info("[v6d_ipc_hook] VineyardPeer patch disabled by env")

    if _env_enabled("SGLANG_SIMULATOR_V6D_IPC_PATCH_DASHLLM_LAUNCH"):
        try:
            from dashllm.utils import vineyard as dashllm_vineyard
        except Exception as exc:
            logger.info("[v6d_ipc_hook] import dashllm.utils.vineyard skipped: %r", exc)
            dashllm_vineyard = None

        if dashllm_vineyard is not None and not getattr(
            dashllm_vineyard,
            "_sglang_simulator_launch_v6d_hook",
            False,
        ):
            original_launch_v6d = dashllm_vineyard.launch_v6d

            def _patched_launch_v6d(*args, **kwargs):
                envs_to_update = dict(kwargs.get("envs_to_update") or {})
                envs_to_update["SGLANG_SIMULATOR_ENABLE_V6D_IPC_HOOK"] = "1"
                kwargs["envs_to_update"] = envs_to_update
                return original_launch_v6d(*args, **kwargs)

            dashllm_vineyard.launch_v6d = _patched_launch_v6d
            dashllm_vineyard._sglang_simulator_launch_v6d_hook = True
            _install_dashllm_kv_transfer_hook(original_launch_v6d)
            logger.info("[v6d_ipc_hook] patched dashllm.utils.vineyard.launch_v6d")
    else:
        logger.info("[v6d_ipc_hook] dashllm launch_v6d patch disabled; enable SGLANG_SIMULATOR_V6D_IPC_PATCH_DASHLLM_LAUNCH=1 to test it")

    logger.info("[v6d_ipc_hook] completed")

# Native V6D control-plane mode keeps the real vLLM connector stack
# (HybridConnector -> V6dObjectKVTBackend -> V6dObjectBackend/PBackend ->
# V6dObjectConnectorScheduler/V6dObjectManager) and installs only runtime
# hijack hooks for CPU-only execution.  The default CPU simulation mode still
# uses MockHybridConnector + V6DCacheStorage(etcd) for scheduling parity.
#
# This file must not patch DashServing/vLLM source files on disk.  All behavior
# changes are installed through class hooks, sitecustomize, and monkey patches
# gated by explicit environment variables.


def _patch_triton_for_cpu():
    """Ensure triton.language.extra.cuda exists for CPU-only environments.

    Some vLLM versions use a TritonLanguagePlaceholder when no GPU driver is
    detected. That placeholder lacks `extra.cuda`, causing AttributeError in
    modules like vllm.lora.ops.triton_ops.utils. Setting CUDA_VISIBLE_DEVICES
    to empty string triggers vLLM's 'distributed environment' heuristic which
    keeps HAS_TRITON=True and uses the real triton module.

    As a safety net, we also pre-import triton.language.extra.cuda so it's
    resolved in sys.modules regardless of placeholder usage.
    """
    # Trick vLLM's triton driver check into lenient mode (distributed env)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    # Safety net: ensure triton.language.extra.cuda is importable
    try:
        import triton.language.extra.cuda  # noqa: F401
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass


def _patch_torch_cuda_for_cpu():
    """Patch torch.cuda functions for CPU-only environments.

    Some modified vLLM versions call torch.cuda.get_device_properties() at
    module level (e.g. in expert_parallel.py). This patches those calls to
    return fake device properties so the imports succeed without a GPU.
    """
    import torch

    if torch.cuda.is_available():
        return  # Real GPU available, no patching needed

    class _FakeDeviceProperties:
        """Fake CUDA device properties for simulation."""
        major = 9
        minor = 0
        name = "NVIDIA Simulated GPU"
        total_memory = 80 * (1 << 30)  # 80 GiB
        multi_processor_count = 132

    _original_get_device_properties = torch.cuda.get_device_properties

    def _mock_get_device_properties(device=None):
        try:
            return _original_get_device_properties(device)
        except (RuntimeError, AssertionError):
            return _FakeDeviceProperties()

    torch.cuda.get_device_properties = _mock_get_device_properties

    # Also patch _lazy_init to be a no-op if CUDA is unavailable
    _original_lazy_init = torch.cuda._lazy_init

    def _mock_lazy_init():
        try:
            _original_lazy_init()
        except RuntimeError:
            pass

    torch.cuda._lazy_init = _mock_lazy_init


def init_hook(force: bool = False):
    """Install all vLLM hooks. Must be called before importing vllm.

    The startup entry lives in the project working directory as sitecustomize.py.
    Python imports it only when that directory is added to PYTHONPATH, and hooks
    are installed only when explicitly enabled by environment variable. This
    avoids installing global .pth files or affecting normal vLLM users sharing
    the same Python environment.
    """
    enable_vllm = _env_enabled(
        "SGLANG_SIMULATOR_ENABLE_HOOK",
        "SGLANG_SIMULATOR_ENABLE_VLLM_HOOK",
    )
    enable_v6d_ipc = _env_enabled(
        "SGLANG_SIMULATOR_ENABLE_HOOK",
        "SGLANG_SIMULATOR_ENABLE_V6D_IPC_HOOK",
    )
    if not force and not (enable_vllm or enable_v6d_ipc):
        return False

    import logging

    if enable_v6d_ipc:
        _install_v6d_ipc_hook()
    if not force and not enable_vllm:
        return True

    import sglang_simulator.hook as sglang_simulator_hook
    from sglang_simulator.simulation.vllm import (
        engine_args,
        kv_connector,
        kv_offload,
        platform,
        scheduler,
        v6d_backend,
        v6d_manager,
        v6d_swap,
        v6d_worker,
        worker,
    )

    # Keep v6d_manager imported: in native V6D mode its hooks tag managers with
    # the active worker id and publish/lookup cross-node ownership; in default
    # MockHybridConnector mode the env helpers remain useful for worker identity.
    _ = v6d_manager

    # Ensure deterministic block hashes across nodes for cross-node cache sharing.
    # vLLM uses PYTHONHASHSEED to initialize NONE_HASH; without it, each process
    # gets os.urandom(32) making block hashes non-reproducible across machines.
    if 'PYTHONHASHSEED' not in os.environ:
        os.environ['PYTHONHASHSEED'] = '42'
        logging.getLogger('sglang_simulator').warning(
            '[init_hook] PYTHONHASHSEED not set, defaulting to 42 for cross-node hash consistency')

    _patch_triton_for_cpu()
    _patch_torch_cuda_for_cpu()

    native_v6d_control_plane = _env_enabled(
        "SGLANG_SIMULATOR_NATIVE_V6D_CONTROL_PLANE"
    )

    hooks = [
        # Platform hook — intercepts Platform class definition to inject
        # _MockCudaPlatform before any platform detection runs.
        platform.C_VLLMPlatformHook,
        # EngineArgs hook — forces parallelism to 1 for simulation
        engine_args.C_VLLMEngineArgsHook,
        # Worker hook (handles everything — no model_runner hooks needed)
        worker.C_VLLMWorkerHook,
        # Scheduler hook for time prediction
        scheduler.C_VLLMSchedulerHook,
        # Native SimpleCPUOffloadWorker hook (bypasses CUDA for native offload)
        kv_offload.C_VLLMSimpleCPUOffloadWorkerHook,
        # Native OffloadingConnectorWorker hook (default native path)
        kv_offload.C_VLLMOffloadingConnectorWorkerHook,
    ]

    if native_v6d_control_plane:
        logging.getLogger('sglang_simulator').info(
            '[init_hook] Native V6D control-plane mode enabled: '
            'keeping real KVConnectorFactory and installing V6D CUDA-bypass hooks')
        hooks.extend([
            v6d_swap.C_V6dSwapHandlerHook,
            v6d_backend.C_HybridBackendHook,
            v6d_backend.C_HybridConnectorHook,
            v6d_worker.C_V6dObjectConnectorWorkerHook,
            v6d_backend.C_V6dObjectBackendHook,
            v6d_backend.C_KVTPBackendHook,
            v6d_manager.C_V6dObjectConnectorSchedulerHook,
            v6d_manager.C_V6dObjectManagerHook,
            v6d_manager.C_HybridSchedulerHook,
        ])
    else:
        # Default CPU simulation path: replace HybridConnector with
        # MockHybridConnector + V6DCacheStorage(etcd) for scheduling parity.
        # DEPRECATED: This branch is only taken when NATIVE_V6D_CONTROL_PLANE is NOT set.
        # The MockHybridConnector path is legacy and no longer maintained.
        # V6DCacheStorage has been deleted; this code will fail if activated.
        # Use SGLANG_SIMULATOR_NATIVE_V6D_CONTROL_PLANE=1 for the supported P2P path.
        hooks.append(kv_connector.C_KVConnectorFactoryHook)

    sglang_simulator_hook.install_class_hooks(hooks)
    _install_dashllm_kv_transfer_hook()
    return True
