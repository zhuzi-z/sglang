import torch
from sglang_simulator.hook import BaseHook


class C_ServerArgsHook(BaseHook):
    HOOK_CLASS_NAME = "ServerArgs"
    HOOK_MODULE_NAME = "sglang.srt.server_args"

    @classmethod
    def hook(cls, target):
        original_post_init = target.__post_init__
        original_handle_model_specific_adjustments = target._handle_model_specific_adjustments

        def wrapped_post_init(self):
            parallel_attrs = [
                "tp_size",
                "ep_size",
                "pp_size",
                "dp_size",
                "attn_cp_size",
                "moe_dp_size",
            ]
            for attr in parallel_attrs:
                setattr(self, attr, 1)
            result = original_post_init(self)
            if self.disaggregation_mode == "decode":
                # No real KV transfer in simulation; forcing fake makes
                # _is_fake_transfer True at all decode gates (add() fast path,
                # metadata gate, room validation; see C_DecodeReceiverHook).
                # Prefill is excluded: native asserts forbid fake+prefill and
                # no fake bootstrap server exists, so the P side selects
                # FakeKVSender via C_PrefillBootstrapQueueHook instead.
                self.disaggregation_transfer_backend = "fake"
            return result
        
        def wrapped_handle_model_specific_adjustments(self):
            if not torch.cuda.is_available():
                # The function will call torch.cuda.get_device_capability(), 
                # which may break if no CUDA device is available.
                return
            else:
                return original_handle_model_specific_adjustments(self)

        target.__post_init__ = wrapped_post_init
        target._handle_model_specific_adjustments = wrapped_handle_model_specific_adjustments
