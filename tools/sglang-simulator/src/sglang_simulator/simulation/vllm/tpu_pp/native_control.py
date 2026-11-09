"""从固化的原生源码快照中按 AST 抽取纯控制逻辑定义。

不执行 vendor 模块的 import，也不触发 TPU/vLLM/torch 初始化；被抽取的类与
函数原样执行，只把硬件相关的名字（张量、PP 通信、计时注解）替换为 CPU 适配。
源码指纹与 SOURCE_MANIFEST.json 不一致时直接拒绝加载。
"""

import ast
import dataclasses
import hashlib
import json
import threading
import time
from collections import deque
from concurrent.futures import Future, InvalidStateError
from contextlib import nullcontext, suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

from sglang_simulator.utils import get_logger

VENDOR = Path(__file__).resolve().parent / "vendor"
MANIFEST = VENDOR / "SOURCE_MANIFEST.json"


def _selected(filename: str, names: list[str], namespace: dict) -> None:
    source = (VENDOR / filename).read_bytes()
    manifest = json.loads(MANIFEST.read_text())
    if hashlib.sha256(source).hexdigest() != manifest[filename]["sha256"]:
        raise RuntimeError(f"vendor fingerprint mismatch: {filename}")
    tree = ast.parse(source)
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in names for t in node.targets):
            # 常量（如 _PUSH_DELAY）直接取原生赋值，不在此处另抄数值
            nodes.append(node)
    found = {n.name if not isinstance(n, ast.Assign) else n.targets[0].id
             for n in nodes}
    if found != set(names):
        raise RuntimeError(f"{filename}: missing {sorted(set(names) - found)}")
    module = ast.Module(body=nodes, type_ignores=[])
    code = "from __future__ import annotations\n" + ast.unparse(module)
    exec(compile(code, f"<vendor:{filename}>", "exec"), namespace)


def load(async_handoff: bool, execute_model_timeout_s: float) -> SimpleNamespace:
    ns = dict(
        dataclasses=dataclasses, time=time, deque=deque, Future=Future,
        Any=Any, Callable=Callable, cast=cast, threading=threading,
        suppress=suppress, InvalidStateError=InvalidStateError,
        logger=get_logger("tpu_pp_native"),
        TraceAnnotation=lambda *a, **k: nullcontext(),
        HIDDEN_STATES="hidden_states", RESIDUAL="residual",
        IntermediateTensors=dict,
        envs=SimpleNamespace(TPU_PP_ASYNC_HANDOFF=bool(async_handoff)),
        vllm_envs=SimpleNamespace(
            VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=execute_model_timeout_s),
        # 描述符替代设备张量：flush 无事可做，wait 等到描述符就绪时刻
        _flush=lambda tensors: None,
        _wait=lambda tensors: [t.wait() for t in tensors],
    )
    _selected("pp_push.py", ["_WaveState", "wave_state"], ns)
    _selected("multiproc_executor.py", ["FutureWrapper"], ns)
    _selected("pp_lazy_step.py", [
        "_PUSH_DELAY", "_MEASURE_WEIGHT", "_MEASURE_BAND", "_STEP_METHODS",
        "_STATELESS_FIELDS", "_carries_state", "_banded", "_reply_ready",
        "_Slot", "_Step", "LazyPushStepper",
    ], ns)
    _selected("pp_wave.py", ["PPWave"], ns)
    return SimpleNamespace(
        PPWave=ns["PPWave"], LazyPushStepper=ns["LazyPushStepper"],
        WaveState=ns["_WaveState"], FutureWrapper=ns["FutureWrapper"],
        STEP_METHODS=ns["_STEP_METHODS"], namespace=ns)
