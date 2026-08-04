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


def _install_dashllm_kv_transfer_hook() -> None:
    """Install non-invasive DashServing/vLLM kv_transfer_params passthrough."""
    try:
        import dashllm.core.backend._backend_vllm as backend_vllm
    except Exception:
        return

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