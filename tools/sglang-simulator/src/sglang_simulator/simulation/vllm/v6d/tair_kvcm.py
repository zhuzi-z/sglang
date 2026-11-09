"""v6d daemon-side Tair KVCM hook — keep KVCM control plane, stub MemPool I/O.

With ``--tair-kvcm-mode=report-event-and-cache-transfer`` the daemon creates a
``TairKVCMTransferRuntime`` wrapping ``kvcm_py_client.TransferClient``.  The
native client needs a local tair_mempool_server (TENT on CPU hosts); when it
cannot be created v6d silently stops reporting BLOCK_ADD / writing to KVCM.
Here the client is replaced by a no-op and the payload batches succeed
without moving bytes, so every KVCM control-plane call (location allocation,
event report, lookup, reclaim) still runs for real against the manager.
"""

from sglang_simulator.hook import BaseHook
from sglang_simulator.utils import get_logger

logger = get_logger()


class _NoopTransferClient:
    """Stands in for kvcm_py_client.TransferClient; its I/O users are stubbed."""

    def close(self):
        pass


class C_TairKVCMTransferRuntimeHook(BaseHook):
    HOOK_CLASS_NAME = "TairKVCMTransferRuntime"
    HOOK_MODULE_NAME = "v6d.common.tair_kvcm.transfer"

    @classmethod
    def hook(cls, target):
        def create_transfer_client(self):
            return _NoopTransferClient()

        def batch_load_to_memory(self, uris, pointers, sizes):
            if not uris:
                return True
            if not (len(uris) == len(pointers) == len(sizes)):
                raise ValueError("uris, pointers, and sizes length mismatch")
            from v6d.common.tair_kvcm.transfer import _record_transfer_result

            _record_transfer_result("read", True)
            return True

        def batch_save_from_memory(self, uris, pointers, sizes):
            if not uris:
                return []
            if not (len(uris) == len(pointers) == len(sizes)):
                raise ValueError("uris, pointers, and sizes length mismatch")
            from v6d.common.tair_kvcm.transfer import _record_transfer_result

            _record_transfer_result("write", True)
            return [True] * len(uris)

        target._create_transfer_client = create_transfer_client
        target.batch_load_to_memory = batch_load_to_memory
        target.batch_save_from_memory = batch_save_from_memory
        logger.info(
            "[v6d-sim] C_TairKVCMTransferRuntimeHook installed: native KVCM "
            "transfer client and MemPool payload I/O stubbed; control plane real"
        )
