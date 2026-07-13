from typing import Optional

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import ConfigManager, StateManager
from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager


class BaseKVCacheTransferEstimator:
    def __init__(self, config: dict):
        pass

    def est_transfer_dur(self, bytes: int) -> float:
        # Return the transfer duration in seconds.
        pass


class ConstBandwidthKVCacheTransferEstimator(BaseKVCacheTransferEstimator):
    def __init__(self, config: dict):
        self.bandwidth = config.get("bandwidth", 32e9)  # in bytes per second

    def est_transfer_dur(self, bytes: int) -> float:
        # Return the transfer duration in seconds.
        return bytes / self.bandwidth


def create_transfer_estimator(config: dict) -> BaseKVCacheTransferEstimator:
    """Factory: create a KV cache transfer estimator from config."""
    name = config.get("name", "const_bandwidth")
    if name == "const_bandwidth":
        return ConstBandwidthKVCacheTransferEstimator(config)
    return ConstBandwidthKVCacheTransferEstimator(config)


class C_DecodePreallocQueueHook(BaseHook):

    HOOK_CLASS_NAME = "DecodeTransferQueue"
    HOOK_MODULE_NAME = "sglang.srt.disaggregation.decode"

    TRANSFER_ESTIMATOR: Optional[BaseKVCacheTransferEstimator] = None
    KV_CACHE_BYTES: Optional[int] = None
    # Tracks when the link becomes free (end of the last scheduled transfer).
    LAST_TRANSFER_COMPLETION_TIME: float = -1

    @classmethod
    def _ensure_initialized(cls):
        if cls.TRANSFER_ESTIMATOR is None:
            estimator_config = dict(ConfigManager.get_kv_transfer_config())
            # Default bandwidth from accelerator inter-node bandwidth.
            if "bandwidth" not in estimator_config:
                hw = ConfigManager.get_accelerator_info()
                if hw.inter_node_bandwidth_gb is not None:
                    estimator_config["bandwidth"] = (
                        hw.inter_node_bandwidth_gb * 1e9
                    )
            cls.TRANSFER_ESTIMATOR = create_transfer_estimator(estimator_config)
        if cls.KV_CACHE_BYTES is None:
            cls.KV_CACHE_BYTES = ConfigManager.get_kv_cache_bytes()

    @classmethod
    def hook(cls, target):

        original_extend = target.extend
        original_commit_transfer_to_req = target._commit_transfer_to_req
        original_poll_with_metadata_gate = target._poll_with_metadata_gate
        original_poll_with_staging = target._poll_with_staging

        def _downgrade_incomplete_polls(polls, decode_reqs):
            from sglang.srt.disaggregation.base.conn import KVPoll

            for i, poll_val in enumerate(polls):
                if poll_val == int(KVPoll.Success):
                    decode_req = decode_reqs[i]
                    completion_time = getattr(
                        decode_req, "_sim_transfer_completion_time", None
                    )
                    if (
                        completion_time is not None
                        and StateManager.get_global_clock()
                        < completion_time
                    ):
                        polls[i] = int(KVPoll.Transferring)
            return polls

        def wrapped_extend(self, decode_reqs):
            # Record the queue entry time and estimate the simulated transfer
            # completion time when requests enter the transfer queue.
            cls._ensure_initialized()
            current_clock = StateManager.get_global_clock()
            for decode_req in decode_reqs:
                req_stats = request_stats_manager.get_req_stats(
                    decode_req.req.rid
                )
                req_stats.kv_cache_transfer_queue_start_time = current_clock

                # Estimate transfer bytes and duration.
                transfer_tokens = decode_req.req.seqlen - len(
                    decode_req.req.prefix_indices
                )
                transfer_bytes = max(transfer_tokens, 0) * cls.KV_CACHE_BYTES
                transfer_dur = cls.TRANSFER_ESTIMATOR.est_transfer_dur(
                    transfer_bytes
                )
                req_stats.kv_cache_transfer_duration = transfer_dur

                # Transfers are serialized on a single link: the actual
                # start is the later of queue entry, current clock, and the
                # previous transfer's completion time.
                transfer_start_time = max(
                    current_clock,
                    cls.LAST_TRANSFER_COMPLETION_TIME,
                )
                req_stats.kv_cache_transfer_start_time = transfer_start_time
                completion_time = transfer_start_time + transfer_dur
                cls.LAST_TRANSFER_COMPLETION_TIME = completion_time
                setattr(
                    decode_req,
                    "_sim_transfer_completion_time",
                    completion_time,
                )
            return original_extend(self, decode_reqs)

        def wrapped_poll_with_metadata_gate(self):
            polls = original_poll_with_metadata_gate(self)
            return _downgrade_incomplete_polls(polls, self.queue)

        def wrapped_poll_with_staging(self):
            polls = original_poll_with_staging(self)
            return _downgrade_incomplete_polls(polls, self.queue)

        def wrapped_commit_transfer_to_req(self, decode_req):
            # The poll gate ensures this is only called when the simulated
            # transfer is complete, so we can always delegate to the original.
            return original_commit_transfer_to_req(self, decode_req)

        target.extend = wrapped_extend
        target._commit_transfer_to_req = wrapped_commit_transfer_to_req
        target._poll_with_metadata_gate = wrapped_poll_with_metadata_gate
        target._poll_with_staging = wrapped_poll_with_staging
