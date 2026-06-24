import torch
from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import ConfigManager
from sglang_simulator.simulation.sglang.utils import (
    resolve_model_info,
    resolve_scheduler_config,
)
from sglang_simulator.simulation.utils import estimate_kv_cache_pool_capacity
from sglang_simulator.utils import get_logger

logger = get_logger()


class C_ModelRunnerHook(BaseHook):
    HOOK_CLASS_NAME = "ModelRunner"
    HOOK_MODULE_NAME = "sglang.srt.model_executor.model_runner"

    @classmethod
    def hook(cls, target):

        def override_initialize(self, *args, **kwargs):
            class MockModel:
                def forward(self):
                    pass

            self.model = MockModel()

            self.dtype = self.model_config.dtype
            self.configure_kv_cache_dtype()

            if ConfigManager.get_model_info() is None:
                model = resolve_model_info(self.model_config)
                ConfigManager.set_model_info(model)

            if self.server_args.max_total_tokens is not None:
                self.max_total_num_tokens = self.server_args.max_total_tokens
            else:
                model = ConfigManager.get_model_info()
                hw = ConfigManager.get_accelerator_info()
                config = resolve_scheduler_config(
                    server_args=self.server_args,
                )

                assert model is not None and hw is not None and config is not None
                self.max_total_num_tokens = estimate_kv_cache_pool_capacity(
                    model, hw, config
                )

            if hasattr(self, "page_size") and self.page_size > 1:
                self.max_total_num_tokens = (
                    self.max_total_num_tokens // self.page_size * self.page_size
                )

            if self.is_hybrid_swa:
                self.sliding_window_size = self.model_config.sliding_window_size
                # if self.model_config.is_swa_with_compressed_attention:
                #     self.set_num_tokens_hybrid_swa_compress()
                # else:
                #     self.set_num_tokens_hybrid_swa()

                # The method names for setting parameters such as num_token vary significantly 
                # across different versions. Here, we directly assign values to these parameters. 
                # In the future, we may attempt to decouple or adapt to different versions 
                # to ensure consistent behavior.
                self.memory_pool_config = self._resolve_memory_pool_config(
                    10
                )
                self.max_total_num_tokens = self.memory_pool_config.max_total_num_tokens
                self.max_running_requests = self.memory_pool_config.max_running_requests
                self.full_max_total_num_tokens = self.memory_pool_config.full_max_total_num_tokens
                self.swa_max_total_num_tokens = self.memory_pool_config.swa_max_total_num_tokens

                self.c4_max_total_num_tokens = self.memory_pool_config.c4_max_total_num_tokens
                self.c128_max_total_num_tokens = self.memory_pool_config.c128_max_total_num_tokens
                self.c4_state_pool_size = self.memory_pool_config.c4_state_pool_size
                self.c128_state_pool_size = self.memory_pool_config.c128_state_pool_size
                self.state_dtype = torch.float32

            max_num_reqs = min(
                max(
                    int(
                        self.max_total_num_tokens / self.model_config.context_len * 512
                    ),
                    2048,
                ),
                4096,
            )
            logger.info(
                f"Model runner initialized with {self.max_total_num_tokens} tokens. Maximum number of requests: {max_num_reqs}"
            )

            model_has_mtp_layers = (
                self.model_config.num_nextn_predict_layers is not None
            )
            model_num_layers = (
                self.model_config.num_nextn_predict_layers
                if self.is_draft_worker and model_has_mtp_layers
                else max(
                    self.model_config.num_hidden_layers,
                    self.model_config.num_attention_layers,
                )
            )
            self.start_layer = getattr(self.model, "start_layer", 0)
            self.end_layer = getattr(self.model, "end_layer", model_num_layers)
            self.num_effective_layers = self.end_layer - self.start_layer

            from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

            self.req_to_token_pool = ReqToTokenPool(
                size=max_num_reqs,
                max_context_len=self.model_config.context_len,
                device=self.device,
                enable_memory_saver=False,
            )

            # During simulation, the actual data in kv cache pool is not important since the MHA computation is skipped,
            # so the head_num and head_dim can be set to 1 to reduce the memory usage.
            # And the scheduler only matters about whether the token_to_kv_pool can be allocated enough space for the requests,
            # so the pool's implementation is not important and can be replaced with `MHATokenToKVPool` that only simulates the allocation logic.
            if self.is_hybrid_swa:
                assert (
                    self.page_size == 256
                ), "In paged swa mode, page_size must be 256."
                from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
                    DeepSeekV4TokenToKVPool,
                )

                print(
                    f"[override_initialize] {self.model_config.qk_nope_head_dim=}, {self.model_config.qk_rope_head_dim=}, {self.model_config.index_head_dim=}, {self.device=}, {self.swa_max_total_num_tokens=}"
                )
                if self.is_draft_worker:
                    from sglang.srt.models.deepseek_v4_nextn import (
                        COMPRESS_RATIO_NEXTN_LAYER,
                    )
                    compression_ratios = [
                        COMPRESS_RATIO_NEXTN_LAYER
                    ] * self.num_effective_layers
                else:
                    compression_ratios = self.model_config.compress_ratios
                self.token_to_kv_pool = DeepSeekV4TokenToKVPool(
                    max_num_reqs=self.server_args.max_running_requests,
                    swa_size=self.swa_max_total_num_tokens,
                    c4_size=self.c4_max_total_num_tokens,
                    c128_size=self.c128_max_total_num_tokens,
                    c4_state_pool_size=self.c4_state_pool_size,
                    c128_state_pool_size=self.c128_state_pool_size,
                    page_size=self.page_size,
                    swa_page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    state_dtype=self.state_dtype,
                    qk_nope_head_dim=1,  # Overwrite dim size
                    qk_rope_head_dim=1,
                    indexer_head_dim=1,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    compression_ratios=compression_ratios,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    enable_hisparse=self.enable_hisparse,
                )
            else:
                from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

                self.token_to_kv_pool = MHATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=1,  # Overwrite head_num and head_dim to 1.
                    head_dim=1,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                    enable_alt_stream=False,
                )

            if self.is_hybrid_swa:
                from sglang.srt.mem_cache.swa_memory_pool import (
                    SWATokenToKVPoolAllocator,
                )

                self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                    self.full_max_total_num_tokens,
                    self.swa_max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=False,
                )
            elif self.page_size == 1:
                from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

                self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=False,
                )
            else:
                from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

                self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=False,
                )

            self.attn_backend = None
            self.graph_mem_usage = 0
            self.weight_load_mem_usage = 10

            self.max_running_requests = min(
                (
                    self.max_total_num_tokens // 2
                    if self.server_args.max_running_requests is None
                    else self.server_args.max_running_requests
                    // (
                        self.server_args.dp_size
                        if self.server_args.enable_dp_attention
                        else 1
                    )
                ),
                self.req_to_token_pool.size,
            )

            self.use_ngram_embedding = False

            return

        def wrapped_forward(self, *args, **kwargs):
            batch = args[0]
            from sglang.srt.layers.logits_processor import LogitsProcessorOutput

            output = LogitsProcessorOutput(
                next_token_logits=torch.empty(
                    size=(batch.batch_size, self.model_config.vocab_size),
                    device=self.device,
                )
            )
            from sglang.srt.model_executor.model_runner import ModelRunnerOutput

            return ModelRunnerOutput(
                logits_output=output,
                can_run_graph=False,
                expert_distribution_metrics=None,
            )

        def wrapped_sample(self, *args, **kwargs):
            logits = args[0]
            forward_batch = args[1]
            batch_size = logits.next_token_logits.shape[0]
            ids = torch.ones(
                size=(batch_size,),
                device=self.device,
                dtype=torch.int64,
            )
            # Populate dummy logprobs when return_logprob is requested,
            # otherwise process_batch_result_decode will crash on None.
            if getattr(forward_batch, "return_logprob", False):
                logits.next_token_logprobs = torch.zeros(
                    size=(batch_size,),
                    device=self.device,
                    dtype=torch.float32,
                )
            return ids

        def wrapped_compute_logprobs_only(*args, **kwargs):
            return None

        target.initialize = override_initialize
        target.forward = wrapped_forward
        target.sample = wrapped_sample
        target.compute_logprobs_only = wrapped_compute_logprobs_only
