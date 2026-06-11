from sglang_simulator.hook import BaseHook



class C_DecodePreallocQueueHook(BaseHook):

    HOOK_CLASS_NAME = "DecodeTransferQueue"
    HOOK_MODULE_NAME = "sglang.srt.disaggregation.decode"

    @classmethod
    def hook(cls, target):
    
        original_commit_transfer_to_req = target._commit_transfer_to_req

        def wrapped_commit_transfer_to_req(self, req):
            transfer_finished = True
            # TODO: The fake reciever will return success instantly,
            # so during simulation we need to wait for the transfer to complete according to the global clock
            if transfer_finished:
                return original_commit_transfer_to_req(self, req)
            else:
                return False
            
        
        target._commit_transfer_to_req = wrapped_commit_transfer_to_req
