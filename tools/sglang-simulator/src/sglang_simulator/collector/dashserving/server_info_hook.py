"""Expose GET /server_info on DashServing's worker control port.

The control server (dashserving/worker/__main__.py, port
DSV_CONTROL_SERVER_BASE_PORT + DSV_WORKER_GLOBAL_INDEX, default 8100) is the
only HTTP surface of a dashllm worker; vLLM's own /server_info lives in its
OpenAI api_server, which dashserving never starts. Resolved vLLM configs are
read from the dashllm worker registry and always returned as JSON. The
(__main__, ControlHandler) target is guarded by module spec so unrelated
handlers and routes remain untouched.
"""

from __future__ import annotations

import json
import sys

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")

_WORKER_MAIN_SPEC = "dashserving.worker.__main__"
_ROUTE = "/server_info"


def _resolve_vllm_config(model):
    """model._backend._backend_engine is a dashllm vLLMEngine (whose ._engine
    is LLMEngine/AsyncLLM) or, on the executor-only path, a MultiprocExecutor
    that carries vllm_config itself."""
    engine = getattr(getattr(model, "_backend", None), "_backend_engine", None)
    for holder in (getattr(engine, "_engine", None), engine):
        cfg = getattr(holder, "vllm_config", None)
        if cfg is not None:
            return cfg
    return None


def _collect_server_info() -> dict:
    """Return vllm_config for one registry entry, or a models map for several."""
    from dashllm.utils import LLMRegistry

    registry = LLMRegistry.get()
    if registry is None:
        return {"error": "engine not initialized"}
    models = list(registry) if isinstance(registry, (list, tuple)) else [registry]

    import pydantic
    from vllm.config import VllmConfig

    adapter = pydantic.TypeAdapter(VllmConfig)

    def _one(model):
        cfg = _resolve_vllm_config(model)
        if cfg is None:
            return {
                "error": "vllm_config unavailable "
                "(engine not loaded yet, or a non-vLLM backend)"
            }
        # fallback=str covers values pydantic cannot encode (torch.dtype).
        return {"vllm_config": adapter.dump_python(cfg, mode="json", fallback=str)}

    if len(models) == 1:
        return _one(models[0])
    return {
        "models": {
            (getattr(m, "_model_id", None) or f"model_{i}"): _one(m)
            for i, m in enumerate(models)
        }
    }


def _send_json(handler, status: int, payload) -> None:
    body = json.dumps(payload, default=str).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _handle_server_info(handler) -> None:
    try:
        payload = _collect_server_info()
    except Exception as e:
        logger.exception("[server_info] failed to collect vllm_config")
        _send_json(handler, 500, {"error": str(e)})
        return
    _send_json(handler, 200, payload)


def _is_worker_main(target) -> bool:
    """True only when ``target`` was built by dashserving's worker __main__."""
    module = sys.modules.get(getattr(target, "__module__", ""), None)
    spec = getattr(module, "__spec__", None)
    return getattr(spec, "name", None) == _WORKER_MAIN_SPEC


class C_DashservingControlHandlerHook(BaseHook):
    """Wrap the worker control server's ``do_GET`` to serve /server_info."""

    HOOK_CLASS_NAME = "ControlHandler"
    HOOK_MODULE_NAME = "__main__"

    @classmethod
    def hook(cls, target) -> None:
        # (__main__, ControlHandler) is generic; only patch dashserving's.
        if not _is_worker_main(target):
            return
        original_do_GET = getattr(target, "do_GET", None)
        if original_do_GET is None:
            return

        def wrapped_do_GET(self):
            if self.path.split("?", 1)[0] != _ROUTE:
                return original_do_GET(self)
            _handle_server_info(self)

        target.do_GET = wrapped_do_GET
        logger.info("dashserving ControlHandler.do_GET patched for GET /server_info")
