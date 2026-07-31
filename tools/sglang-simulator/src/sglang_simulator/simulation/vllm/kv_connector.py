"""Scheduler reference sharing between the sim scheduler hook and sim worker.

The legacy MockHybridConnector / C_KVConnectorFactoryHook simulation path
(etcd-backed V6DCacheStorage) has been removed: the native V6D control-plane
mode — real HybridConnector stack plus the runtime hooks in ``v6d/`` — is the
only supported path and is always installed by ``startup.init_hook()``.

What remains here is the module-level scheduler reference: the scheduler hook
publishes the live scheduler instance after ``__init__`` and the sim worker's
``execute_model`` reads ``scheduler.requests`` through it to build mock
outputs.
"""

from __future__ import annotations

from sglang_simulator.utils import get_logger

logger = get_logger()


# Module-level scheduler reference (set by C_VLLMSchedulerHook)
_scheduler_ref = None


def set_scheduler_ref(scheduler):
    """Called by scheduler hook after init to share the reference."""
    global _scheduler_ref
    _scheduler_ref = scheduler
    logger.info("[Scheduler Ref] Scheduler reference acquired")


def get_scheduler_ref():
    return _scheduler_ref
