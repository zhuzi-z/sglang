"""
V6D ObjectBackend Hook - Replaces CUDA Event pool with DummyEvent.

The V6dObjectBackend manages the async save/load pipeline and uses
torch.cuda.Event for synchronization. In simulation, these become no-ops.

Preserves:
- V6D object lifecycle (create, seal, get, delete)
- Backend scheduling logic

Mocks:
- torch.cuda.Event() → DummyEvent
- event.record(torch.cuda.current_stream()) → No-op
"""

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.vllm.v6d_swap import DummyEvent
from sglang_simulator.utils import get_logger

logger = get_logger()


class C_V6dObjectBackendHook(BaseHook):
    """Hook V6dObjectBackend to replace CUDA Event pool.

    The V6dObjectBackend coordinates save/load operations and uses
    CUDA Events to track when GPU operations complete. In CPU simulation,
    all events are immediately complete (DummyEvent.query() → True).
    """

    HOOK_CLASS_NAME = "V6dObjectBackend"
    HOOK_MODULE_NAME = "vllm.v1.hybrid_connector.v6d_object_backend"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def override_init(self, *args, **kwargs):
            """Call original init, then replace CUDA Event pool."""
            original_init(self, *args, **kwargs)
            # Replace Event pools with DummyEvent lists
            if hasattr(self, "_save_event_pool"):
                self._save_event_pool = [DummyEvent() for _ in range(8)]
            if hasattr(self, "_load_event_pool"):
                self._load_event_pool = [DummyEvent() for _ in range(8)]
            logger.info(
                "[V6D Hijack] V6dObjectBackend.__init__: "
                "replaced CUDA Event pools with DummyEvent"
            )

        target.__init__ = override_init

        # Patch _get_event / _new_event if they exist
        def override_new_event(self):
            """Return DummyEvent instead of torch.cuda.Event."""
            return DummyEvent()

        if hasattr(target, "_new_event"):
            target._new_event = override_new_event

        # Override _record_event if exists
        def override_record_event(self, event=None):
            """No-op: DummyEvent.record() does nothing."""
            if event is None:
                event = DummyEvent()
            event.record()
            return event

        if hasattr(target, "_record_event"):
            target._record_event = override_record_event

        logger.info("[V6D Hijack] V6dObjectBackend hook installed")
