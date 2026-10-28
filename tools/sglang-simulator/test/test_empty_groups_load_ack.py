"""Regression: an empty ``groups`` must still register the load acknowledgement.

``async_load_kv`` is overridden to return without ever yielding, so this
controller is the only possible source of the native ``_do_load_done``. The
invariant is therefore: whenever the native side decides to wait for a load --
``async_update_state_after_alloc`` returning ``None`` with
``num_external_tokens > 0`` -- a deadline must be registered.

``_reqs_to_load`` is wiped at the end of every native scheduling step, so the
``prepare_load`` coroutine can resolve on the connector loop after that wipe and
legitimately observe ``{}``. Guarding ``queue_load`` on a non-empty ``groups``
then strands the request in the native ``_loading`` set forever: nothing
acknowledges the load, ``has_pending()`` reports no work, and EngineCore blocks
on its input queue until the step timeout kills the process.

These tests drive the installed ``prepare_load`` instead of calling
``queue_load`` directly. The guard lived in ``prepare_load``, so a test that
shortcuts to ``queue_load`` cannot observe the regression at all.
"""

import asyncio
import os
from collections import defaultdict
from importlib import util
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# A machine that only carries the repository has neither the simulation hooks
# nor vLLM, so every import that reaches them stays inside a function body
# covered by the skip below: collection must report skips, not an ImportError.
_REQUIRED_MODULES = (
    "sglang_simulator.simulation.vllm.v6d.v6d_backend",
    "vllm",
)


def _module_missing(name):
    """``find_spec`` raises instead of returning None when a parent is absent."""
    try:
        return util.find_spec(name) is None
    except (ImportError, ValueError):
        return True


def _deps_missing():
    return ", ".join(name for name in _REQUIRED_MODULES if _module_missing(name))


_MISSING = _deps_missing()
_SKIP_REASON = "5 项空 groups 回归未运行，缺少仿真源码或依赖: " + _MISSING

if __name__ == "__main__" and _MISSING:
    raise SystemExit(_SKIP_REASON)


def make_controller():
    from vllm.v1.core.block_pool import BlockPool

    from sglang_simulator.simulation.vllm.v6d.v6d_backend import SimulatedKVController

    cache = SimpleNamespace(
        _block_pool=BlockPool(8, True, 4224),
        _swap_protected_blocks=defaultdict(list),
        mamba_group_ids={0},
        _reqs_to_load={},
    )
    cache._release_protected_blocks = lambda rid: cache._block_pool.free_blocks(
        cache._swap_protected_blocks.pop(rid, [])
    )
    hybrid = SimpleNamespace(
        _backend=SimpleNamespace(_scheduler=cache),
        _saving={},
        _saved=[],
        _loaded=[],
        pending_requests=[],
        _step_saved=MagicMock(),
        loop=None,
        _tp_size=lambda: 2,
    )
    return SimulatedKVController(hybrid)


@pytest.fixture
def controller(monkeypatch):
    from sglang_simulator.simulation.manager import StateManager
    from sglang_simulator.simulation.vllm.v6d.bandwidth import BandwidthModel

    # OFFLINE pins now() to the virtual clock, so a non-zero modeled latency
    # leaves the deadline strictly in the future for the whole test, and a zero
    # one lands exactly on now().
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE")
    # The hook refuses to install outside controller timing, which is the only
    # mode where this invariant exists; pin it so an inherited shell value
    # cannot turn these assertions into an install-time error.
    monkeypatch.setenv("SIMULATOR_LOAD_TIMING", "controller")
    monkeypatch.setattr(StateManager, "_global_clock", 0.0)
    monkeypatch.setattr(
        BandwidthModel,
        "get",
        lambda: SimpleNamespace(
            enabled=True,
            latency_for=lambda n, load: 0.5 if n else 0.0,
            store_completion_latency=lambda n: 0.5 if n else 0.0,
            seg1_latency=lambda n, nbytes=0: 0.0,
            seg2_bytes=lambda n: 0,
        ),
    )
    return make_controller()


def make_hooked_backend(controller, native_result=None):
    """Install the backend hook on a stub that mimics V6dObjectBackend.

    Only the surface ``C_V6dObjectBackendHook`` touches is modelled: the
    scheduler holding ``_reqs_to_load``, the controller reference, and the
    native ``async_update_state_after_alloc`` whose verdict selects the branch
    under test. The stub class is per-call, so the hook's closures never leak
    between tests.
    """
    from sglang_simulator.simulation.vllm.v6d.v6d_backend import C_V6dObjectBackendHook

    class StubBackend:
        _save_event_pool = []
        _load_event_pool = []

        def __init__(self, scheduler, sim_controller):
            self._scheduler = scheduler
            self._sim_controller = sim_controller

        async def async_update_state_after_alloc(
            self, request, blocks, num_external_tokens
        ):
            return native_result

    C_V6dObjectBackendHook.hook(StubBackend)
    return StubBackend(controller.cache, controller)


def drive_prepare(backend, request_id, num_external_tokens):
    """Run the hooked preparation exactly as the connector loop would."""
    request = SimpleNamespace(request_id=request_id)
    return asyncio.run(
        backend.async_update_state_after_alloc(request, [], num_external_tokens)
    )


@pytest.mark.skipif(bool(_MISSING), reason=_SKIP_REASON)
def test_empty_groups_still_registers_load_deadline(controller):
    backend = make_hooked_backend(controller)
    # The native scheduling step already wiped the entry, so groups is {}.
    assert controller.cache._reqs_to_load == {}

    assert drive_prepare(backend, "r", 8448) is None

    assert "r" in controller._loads, (
        "no deadline registered for an empty groups: the native _loading entry "
        "would never be acknowledged"
    )
    deadline, num_tokens = controller._loads["r"]
    assert num_tokens == 8448
    # Zero modeled blocks cost nothing, so the next progress() settles it.
    assert deadline <= controller.now()
    assert controller.has_pending() is True


@pytest.mark.skipif(bool(_MISSING), reason=_SKIP_REASON)
def test_empty_groups_load_is_acknowledged_by_the_next_progress(controller):
    backend = make_hooked_backend(controller)
    acknowledged = []
    controller._load_done = lambda rid, n: acknowledged.append((rid, n))

    drive_prepare(backend, "r", 8448)
    controller.progress()

    assert acknowledged == [("r", 8448)]
    assert controller._loads == {}
    assert controller.has_pending() is False


@pytest.mark.skipif(bool(_MISSING), reason=_SKIP_REASON)
def test_non_empty_groups_keeps_its_modeled_latency(controller):
    backend = make_hooked_backend(controller)
    controller.cache._reqs_to_load = {"r": {0: (["k0"], [0])}}

    drive_prepare(backend, "r", 8448)

    # Real blocks must still be paid for, not acknowledged immediately.
    assert controller._loads["r"][0] > controller.now()


@pytest.mark.skipif(bool(_MISSING), reason=_SKIP_REASON)
def test_satisfied_preparation_does_not_block_a_later_load(controller):
    """The no-load branch must not leave a sticky cancellation behind.

    ``_forget_native_request`` marks a request in ``_cancelled_preparations``
    while any preparation is still in flight, and ``queue_load`` silently
    returns for a marked request. Calling it from the branch that merely has
    nothing to load would make the *next* legitimate load for the same request
    skip its deadline -- the same hang by another route.
    """
    backend = make_hooked_backend(controller)
    # A preparation is in flight, as it is whenever the connector drives
    # prepare_load through SimulatedKVController.prepare().
    controller._preparing_requests["r"] += 1

    drive_prepare(backend, "r", 0)
    assert "r" not in controller._cancelled_preparations

    controller.cache._reqs_to_load = {"r": {0: (["k0"], [0])}}
    drive_prepare(backend, "r", 8448)
    assert "r" in controller._loads


@pytest.mark.skipif(bool(_MISSING), reason=_SKIP_REASON)
def test_native_satisfied_result_registers_nothing(controller):
    # A non-None native result means no load is pending, so the controller
    # must stay idle and let EngineCore block on its input queue.
    backend = make_hooked_backend(controller, native_result=object())

    drive_prepare(backend, "r", 8448)

    assert controller._loads == {}
    assert controller.has_pending() is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
