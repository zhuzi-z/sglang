"""
GET /server_info on the dashserving worker control port (debug module).

dashllm_cmd serves over gRPC and never starts vLLM's FastAPI app, so vLLM's
own ``GET /server_info`` (api_server.py, gated on VLLM_SERVER_DEV_MODE) is
never registered. The data it returns is just ``VllmConfig``, which
``LLMEngine`` keeps in the worker main process — the same process that hosts
dashserving's control HTTP server. This hook adds the route there:

    curl http://127.0.0.1:<DSV_CONTROL_SERVER_BASE_PORT + DSV_WORKER_GLOBAL_INDEX>/server_info?config_format=json

``config_format`` is ``text`` (default, ``str(VllmConfig)``) or ``json``
(pydantic dump, mirroring vLLM's show_server_info()); anything else is 400.

Mechanism: the control handler is a class local to
``_start_control_server`` in ``dashserving/worker/__main__.py``. That file is
run via ``python -m dashserving.worker``, so the class is built in module
``__main__``; the hook therefore targets ``(__main__, ControlHandler)`` and
double-checks the running module spec before wrapping ``do_GET``. Every other
path falls through to the original handler untouched.
"""

from __future__ import annotations

import json
import sys
from urllib.parse import parse_qs, urlparse

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger("debug.server_info")

_WORKER_MAIN_SPEC = "dashserving.worker.__main__"
_ROUTE = "/server_info"


def _collect_server_info(config_format: str = "text") -> dict:
    """Return the resolved vLLM VllmConfig for the engine(s) on this worker.

    Walks the dashllm object graph in this process:
        LLMRegistry.get() -> LLM._backend -> ._backend_engine -> ._engine
                          -> LLMEngine.vllm_config

    Response shape matches vLLM's {"vllm_config": ...} for the usual
    single-model deployment. With DS_LLM_MULTI_MODEL_CONFIG the worker hosts
    several engines, so those are returned under {"models": {name: {...}}}.
    """
    from dashllm.utils import LLMRegistry

    registry = LLMRegistry.get()
    if registry is None:
        return {"error": "engine not initialized"}
    models = list(registry) if isinstance(registry, (list, tuple)) else [registry]

    adapter = None
    if config_format == "json":
        import pydantic
        from vllm.config import VllmConfig

        adapter = pydantic.TypeAdapter(VllmConfig)

    def _one(model):
        engine = getattr(getattr(model, "_backend", None), "_backend_engine", None)
        cfg = getattr(getattr(engine, "_engine", None), "vllm_config", None)
        if cfg is None:
            return {
                "error": "vllm_config unavailable "
                "(engine not loaded yet, or a non-vLLM backend)"
            }
        if adapter is not None:
            # fallback=str covers values pydantic cannot encode (torch.dtype).
            return {"vllm_config": adapter.dump_python(cfg, mode="json", fallback=str)}
        return {"vllm_config": str(cfg)}

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
    fmt = (
        parse_qs(urlparse(handler.path).query)
        .get("config_format", ["text"])[0]
        .strip()
        .lower()
    )
    if fmt not in ("text", "json"):
        _send_json(handler, 400, {"error": "config_format must be text or json"})
        return
    try:
        payload = _collect_server_info(fmt)
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
