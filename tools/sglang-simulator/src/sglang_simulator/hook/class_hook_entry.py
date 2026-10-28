import atexit
import builtins
import inspect
import json
import os
from pathlib import Path
import re
import sys
from types import FunctionType
from typing import List, Union

from sglang_simulator.hook.base_hook import BaseHook, _register_hooks
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


CLASS_HOOKS: List[BaseHook] = []

_builtins_build_class_ = builtins.__build_class__

# 诊断只观察，不改变首匹配行为，不按数量或缺失状态拒绝启动。
_HOOK_RECORDS = []
_REGISTERED_LOADED = {}
_REPORT_EXIT_REGISTERED = False


def _hook_id(hook):
    return f"{hook.__module__}.{hook.__qualname__}"


def _matches(hook, module, name=None):
    pattern = hook.HOOK_MODULE_NAME
    if not isinstance(pattern, str) or not pattern:
        return False
    try:
        module_ok = bool(re.search(pattern, module)) if hook.REGEX else pattern == module
        if name is None:
            return module_ok
        pattern = hook.HOOK_CLASS_NAME
        if not isinstance(pattern, str) or not pattern:
            return False
        return module_ok and (bool(re.search(pattern, name)) if hook.REGEX else pattern == name)
    except re.error:
        return False


def _loaded_targets(hook, kind):
    # 只查 sys.modules，不能为了报告导入目标而改变原来的启动顺序。
    found = []
    for module_name, module in list(sys.modules.items()):
        if module is None or not _matches(hook, module_name):
            continue
        if kind == "module":
            found.append(module)
            continue
        for name, obj in list(vars(module).items()):
            if isinstance(obj, type) and obj.__module__ == module_name and _matches(hook, module_name, name):
                found.append(obj)
    return found


def _register_diagnostics(hooks, kind):
    global _REPORT_EXIT_REGISTERED
    for hook in hooks:
        key = (kind, id(hook))
        if key not in _REGISTERED_LOADED:
            _REGISTERED_LOADED[key] = _loaded_targets(hook, kind)
    if not _REPORT_EXIT_REGISTERED:
        atexit.register(emit_hook_report, "process-exit")
        _REPORT_EXIT_REGISTERED = True


def _apply_with_diagnostics(hook, target, kind="class"):
    before = dict(vars(target))
    record = {"hook_object": hook, "target_object": target, "kind": kind,
              "applied": False, "error": None, "delta": {}}
    _HOOK_RECORDS.append(record)
    try:
        hook.hook(target)
        record["applied"] = True
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        after = dict(vars(target))
        record["delta"] = {name: after[name] for name in after
                           if name not in before or after[name] is not before[name]}
        logger.info("[HOOK_APPLY] pid=%s hook=%s target=%s.%s status=%s changed=%s",
                    os.getpid(), _hook_id(hook), getattr(target, "__module__", ""),
                    getattr(target, "__name__", ""),
                    "installed" if record["applied"] else "failed",
                    sorted(record["delta"]))


def _describe(value):
    obj = value.__func__ if isinstance(value, (classmethod, staticmethod)) else value
    return f"{getattr(obj, '__module__', type(obj).__module__)}.{getattr(obj, '__qualname__', type(obj).__name__)}"


def get_hook_report(phase="manual", role=None):
    from sglang_simulator.hook.module_hook_entry import MODULE_HOOKS
    rows = []
    for kind, hooks in (("class", CLASS_HOOKS), ("module", MODULE_HOOKS)):
        for index, hook in enumerate(hooks):
            row = {"kind": kind, "hook": _hook_id(hook),
                   "target_module": hook.HOOK_MODULE_NAME,
                   "target_class": hook.HOOK_CLASS_NAME if kind == "class" else None,
                   "regex": hook.REGEX, "registered_before_target": not bool(
                       _REGISTERED_LOADED.get((kind, id(hook)))), "targets": []}
            if not hook.HOOK_MODULE_NAME or (kind == "class" and not hook.HOOK_CLASS_NAME):
                row["status"] = "invalid_target"
                rows.append(row)
                continue
            records = [r for r in _HOOK_RECORDS if r["hook_object"] is hook and r["kind"] == kind]
            for record in records:
                target = record["target_object"]
                methods = {}
                for name, replacement in record["delta"].items():
                    current = inspect.getattr_static(target, name, None)
                    methods[name] = {"identity_ok": current is replacement,
                                     "installed": _describe(replacement), "current": _describe(current)}
                current_module = sys.modules.get(getattr(target, "__module__", ""))
                binding = (vars(current_module).get(target.__name__) is target
                           if kind == "class" and current_module else True)
                status = ("failed" if not record["applied"] else
                          "replaced_target" if not binding else
                          "overwritten" if any(not m["identity_ok"] for m in methods.values()) else
                          "installed" if methods else "no_patch")
                row["targets"].append({"target": f"{getattr(target, '__module__', '')}.{target.__name__}",
                                       "status": status, "error": record["error"], "methods": methods})
            if row["targets"]:
                row["status"] = row["targets"][-1]["status"]
            else:
                loaded = _loaded_targets(hook, kind)
                shadowed = any(any(r["target_object"] is obj and r["kind"] == kind and
                                   r["hook_object"] is not hook for r in _HOOK_RECORDS) for obj in loaded)
                row["status"] = ("shadowed" if shadowed else "imported_before_registration"
                                 if loaded and not row["registered_before_target"] else
                                 "loaded_unmatched" if loaded else "pending_not_loaded")
            rows.append(row)
    return {"schema_version": 1, "phase": phase, "role": role or "python-process",
            "pid": os.getpid(), "ppid": os.getppid(), "run_id": os.getenv("SIMULATOR_HOOK_RUN_ID"),
            "scope": "all_registered_class_and_module_hooks", "blocking": False,
            "class_interceptor_active": builtins.__build_class__ is _custom_build_class_,
            "hooks": rows}


def emit_hook_report(phase="manual", role=None):
    try:
        report = get_hook_report(phase, role)
        counts = {}
        for row in report["hooks"]:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            logger.info("[HOOK_STATUS] pid=%s role=%s hook=%s target=%s.%s status=%s",
                        report["pid"], report["role"], row["hook"], row["target_module"],
                        row["target_class"], row["status"])
        directory = Path(os.getenv("SIMULATOR_HOOK_REPORT_DIR", "/tmp/simulator_hook_reports"))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{report['pid']}-{phase}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temporary.replace(path)
        logger.info("[HOOK_REPORT] pid=%s role=%s phase=%s summary=%s file=%s blocking=false",
                    report["pid"], report["role"], phase, counts, path)
        return report
    except Exception:
        # 诊断失败要显式可见，但不得把报告器变成启动门禁。
        logger.exception("[HOOK_REPORT] 报告生成失败，服务不因此退出")
        return None


def _custom_build_class_(func, name: str, *bases, **kwargs):
    for hook in CLASS_HOOKS:
        if (
            hook.REGEX
            and hook.HOOK_CLASS_NAME
            and re.search(hook.HOOK_CLASS_NAME, name)
        ) or name == hook.HOOK_CLASS_NAME:
            module_name = None
            if isinstance(func, FunctionType):
                module_name = getattr(func, "__globals__", {}).get("__name__", "")
            if (
                hook.REGEX and re.search(hook.HOOK_MODULE_NAME, module_name)
            ) or module_name == hook.HOOK_MODULE_NAME:
                logger.debug(
                    f"Hooking Class: {hook.__name__} into {module_name}|{name}"
                    + (
                        "(Regex is enabled, which might cause unexpected behavior.)"
                        if hook.REGEX
                        else ""
                    )
                )
                target_class = _builtins_build_class_(func, name, *bases, **kwargs)
                try:
                    _apply_with_diagnostics(hook, target_class)
                except Exception as e:
                    logger.warning(
                        "Failed to hook class [%s]. Error type: %s, message: %r",
                        name,
                        type(e).__name__,
                        e,
                    )
                return target_class

    return _builtins_build_class_(func, name, *bases, **kwargs)


def install_class_hooks(hooks: Union[List[BaseHook], BaseHook]):
    _register_hooks(CLASS_HOOKS, hooks)
    _register_diagnostics(CLASS_HOOKS, "class")
    builtins.__build_class__ = _custom_build_class_


def remove_class_hooks():
    # Clear the registered hooks and reset the build class function.
    # Note: The classes that have been hooked will not be reset.
    CLASS_HOOKS.clear()
    _HOOK_RECORDS[:] = [r for r in _HOOK_RECORDS if r["kind"] != "class"]
    for key in list(_REGISTERED_LOADED):
        if key[0] == "class":
            del _REGISTERED_LOADED[key]
    builtins.__build_class__ = _builtins_build_class_
