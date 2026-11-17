import builtins
import json
import os
import re
import sys
from types import FunctionType, ModuleType
from typing import List, Union

from sglang_simulator.hook.base_hook import BaseHook, _register_hooks
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


CLASS_HOOKS: List[BaseHook] = []

_builtins_build_class_ = builtins.__build_class__

_HOOK_DIAGNOSTICS = {}
_MISSING = object()


def _safe_event_value(value):
    if value is None or type(value) in (bool, int, float):
        return value
    if type(value) is str:
        return "[redacted]" if re.search(r"[a-zA-Z][\w+.-]*://", value) else value
    if type(value) in (list, tuple):
        return [_safe_event_value(item) for item in value]
    if type(value) is dict:
        return {
            key: _safe_event_value(item)
            for key, item in value.items()
            if type(key) is str and (
                key in ("exception_type", "ignore_environment") or not re.search(
                    r"(^|_)(env|environ|environment|url|message|traceback|exception|error)($|_)",
                    key.lower(),
                )
            )
        }
    if isinstance(value, type):
        return value.__module__ + "." + value.__qualname__
    return type(value).__name__


def emit_hook_event(event, **fields):
    try:
        event = re.sub(r"[^A-Z0-9_]", "_", event.upper())
        payload = _safe_event_value(fields)
        payload.update(pid=os.getpid(), ppid=os.getppid())
        print(
            f"[HISIM_{event}] " + json.dumps(payload, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        # 诊断不能让可选 hook 变成启动失败。
        pass


def _class_dict(target):
    if not isinstance(target, type):
        return {}
    return dict(type.__getattribute__(target, "__dict__"))


def _is_method(value):
    return isinstance(value, (FunctionType, staticmethod, classmethod, property)) or (
        callable(value) and not isinstance(value, type)
    )


def _hook_fields(hook, module_name=None, class_name=None):
    return {
        "hook_name": hook.__name__,
        "target_module": hook.HOOK_MODULE_NAME if module_name is None else module_name,
        "target_class": hook.HOOK_CLASS_NAME if class_name is None else class_name,
    }


def _loaded_targets(hook):
    loaded_modules, targets, seen = [], [], set()
    for module_name, module in list(sys.modules.items()):
        if not isinstance(module, ModuleType):
            continue
        if not ((hook.REGEX and re.search(hook.HOOK_MODULE_NAME, module_name))
                or module_name == hook.HOOK_MODULE_NAME):
            continue
        loaded_modules.append(module_name)
        for target in list(ModuleType.__getattribute__(module, "__dict__").values()):
            if not isinstance(target, type) or id(target) in seen:
                continue
            name = type.__getattribute__(target, "__name__")
            if type.__getattribute__(target, "__module__") != module_name:
                continue
            if ((hook.REGEX and hook.HOOK_CLASS_NAME
                 and re.search(hook.HOOK_CLASS_NAME, name))
                    or name == hook.HOOK_CLASS_NAME):
                seen.add(id(target))
                targets.append({
                    **_hook_fields(hook, module_name, name),
                    "status": "early_import", "changed_methods": [],
                    "_target": target, "_methods": {},
                })
    return loaded_modules, targets


def _public_record(record):
    result = {key: value for key, value in record.items() if not key.startswith("_")}
    result["changed_methods"] = list(record["changed_methods"])
    if record["status"] == "applied":
        current = _class_dict(record["_target"])
        if any(current.get(name, _MISSING) is not method
               for name, method in record["_methods"].items()):
            result["status"] = "overwritten"
    return result


def _record_registration(hook):
    try:
        already_registered = hook in _HOOK_DIAGNOSTICS
        loaded_modules, targets = _loaded_targets(hook)
        if not already_registered:
            _HOOK_DIAGNOSTICS[hook] = targets
        emit_hook_event(
            "HOOK_REGISTER", **_hook_fields(hook), status="registered",
            already_registered=already_registered, already_loaded=loaded_modules,
            targets=[_public_record(record) for record in _HOOK_DIAGNOSTICS[hook]],
        )
    except Exception:
        # 无效正则仍由原匹配点报错。
        pass


def _record_application(hook, target, module_name, name, before, exception_type):
    try:
        after = _class_dict(target)
        changed = sorted(
            key for key in before.keys() | after.keys()
            if before.get(key, _MISSING) is not after.get(key, _MISSING)
            and (_is_method(before.get(key, _MISSING))
                 or _is_method(after.get(key, _MISSING)))
        )
        status = "failed" if exception_type else ("applied" if changed else "no_patch")
        record = {
            **_hook_fields(hook, module_name, name),
            "status": status, "changed_methods": changed,
            "_target": target if isinstance(target, type) else None,
            "_methods": {key: after.get(key, _MISSING) for key in changed},
        }
        if exception_type:
            record["exception_type"] = exception_type
        records = _HOOK_DIAGNOSTICS.setdefault(hook, [])
        records[:] = [item for item in records if item["_target"] is not target]
        records.append(record)
        emit_hook_event("HOOK_APPLY", **_public_record(record))
    except Exception:
        pass


def report_class_hooks(stage):
    rows, seen = [], set()
    for hook in CLASS_HOOKS:
        if hook in seen:
            continue
        seen.add(hook)
        records = _HOOK_DIAGNOSTICS.get(hook, [])
        if records:
            rows.extend(_public_record(record) for record in records)
        else:
            rows.append({**_hook_fields(hook), "status": "pending", "changed_methods": []})
    report = {"stage": stage, "hooks": rows}
    emit_hook_event("HOOK_REPORT", **report)
    return report


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
                before = _class_dict(target_class)
                emit_hook_event(
                    "HOOK_APPLY", **_hook_fields(hook, module_name, name),
                    status="matched", changed_methods=[],
                )
                exception_type = None
                try:
                    hook.hook(target_class)
                except Exception as e:
                    exception_type = type(e).__name__
                    logger.warning(
                        "Failed to hook class [%s]. Error type: %s, message: %r",
                        name,
                        type(e).__name__,
                        e,
                    )
                _record_application(
                    hook, target_class, module_name, name, before, exception_type
                )
                return target_class

    return _builtins_build_class_(func, name, *bases, **kwargs)


def install_class_hooks(hooks: Union[List[BaseHook], BaseHook]):
    previous_count = len(CLASS_HOOKS)
    _register_hooks(CLASS_HOOKS, hooks)
    builtins.__build_class__ = _custom_build_class_
    for hook in CLASS_HOOKS[previous_count:]:
        _record_registration(hook)


def remove_class_hooks():
    # Clear the registered hooks and reset the build class function.
    # Note: The classes that have been hooked will not be reset.
    CLASS_HOOKS.clear()
    _HOOK_DIAGNOSTICS.clear()
    builtins.__build_class__ = _builtins_build_class_
