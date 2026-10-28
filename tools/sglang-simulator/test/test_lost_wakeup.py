"""Regression: the engine wake-up criterion must include modeled transfers.

A modeled load/store settles only when ``SimulatedKVController.progress()``
runs, and ``progress()`` runs only while EngineCore keeps stepping. EngineCore
stops stepping as soon as ``Scheduler.has_requests()`` is False, and that asks
the KV connector first. Native ``HybridScheduler.has_requests()`` inspects only
its own queues (``_waiting``/``_loaded``/``_saved``/...), so it cannot know
about transfers whose completion the controller owns. The connector hook must
therefore fold the controller's unsettled work into that verdict; otherwise the
last in-flight request hangs until some unrelated request happens to wake the
engine up.

These tests deliberately never call ``progress()``. They assert the wake-up
criterion itself, which is the part no existing test covers: every other
controller test drives ``progress()`` by hand and so stands in for the very
driver whose absence causes the hang.
"""

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
_SKIP_REASON = "4 项唤醒回归未运行，缺少仿真源码或依赖: " + _MISSING

if __name__ == "__main__" and _MISSING:
    raise SystemExit(_SKIP_REASON)


def make_controller():
    from vllm.v1.core.block_pool import BlockPool

    from sglang_simulator.simulation.vllm.v6d.v6d_backend import SimulatedKVController

    cache = SimpleNamespace(
        _block_pool=BlockPool(8, True, 4224),
        _swap_protected_blocks=defaultdict(list),
        mamba_group_ids={0},
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
    # leaves the deadline strictly in the future for the whole test.
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE")
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


def make_hooked_connector(sim_controller):
    """Install the connector hook on a stub that mimics HybridConnector.

    Native ``HybridConnector.has_requests()`` is a plain forward to
    ``self._sched.has_requests()``, so a stub whose verdict is settable
    reproduces the decision the engine actually makes.
    """
    from sglang_simulator.simulation.vllm.v6d.v6d_backend import C_HybridConnectorHook

    class StubConnector:
        native_verdict = False

        def __init__(self, sched):
            self._sched = sched

        def has_requests(self):
            return self.native_verdict

    C_HybridConnectorHook.hook(StubConnector)
    return StubConnector(SimpleNamespace(_sim_controller=sim_controller))


def queue_one_load(controller, request_id="r"):
    """Register one modeled load through the normal controller entry point."""
    request = SimpleNamespace(request_id=request_id)
    controller.queue_load(request, 8, {0: (["k0"], [0])})


@pytest.mark.skipif(bool(_MISSING), reason=_SKIP_REASON)
def test_unexpired_load_keeps_engine_awake(controller):
    connector = make_hooked_connector(controller)
    assert connector.has_requests() is False

    queue_one_load(controller)

    # The deadline is in the future and nothing has settled it.
    assert controller.next_wakeup() > controller.now()
    assert connector.has_requests() is True
    # Asking the criterion must not settle the transfer behind our back.
    assert "r" in controller._loads


@pytest.mark.skipif(bool(_MISSING), reason=_SKIP_REASON)
def test_staged_store_keeps_engine_awake(controller):
    connector = make_hooked_connector(controller)
    block = controller.cache._block_pool.get_new_blocks(1)[0]
    block.ref_cnt += 1
    controller.cache._swap_protected_blocks["r"].append(block)
    request = SimpleNamespace(request_id="r", is_finished=lambda: True)
    metadata = SimpleNamespace(
        reqs=SimpleNamespace(
            v6d=SimpleNamespace(
                inner=SimpleNamespace(
                    reqs_to_store={"r": ({0: (["k0"], [block.block_id])}, True)}
                )
            )
        )
    )
    controller.hybrid._saving = {"r": SimpleNamespace(_req=request)}

    controller.capture_stores(metadata)

    assert controller.has_pending() is True
    assert connector.has_requests() is True


@pytest.mark.skipif(bool(_MISSING), reason=_SKIP_REASON)
def test_idle_controller_defers_to_native_verdict(controller):
    connector = make_hooked_connector(controller)

    # No modeled work: the hook must not pin the engine awake, otherwise
    # EngineCore never blocks on its input queue and spins forever.
    assert controller.has_pending() is False
    assert connector.has_requests() is False

    connector.native_verdict = True
    assert connector.has_requests() is True


@pytest.mark.skipif(bool(_MISSING), reason=_SKIP_REASON)
def test_without_controller_native_behaviour_is_unchanged():
    connector = make_hooked_connector(None)

    assert connector.has_requests() is False
    connector.native_verdict = True
    assert connector.has_requests() is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
