"""CPU 仿真的 attention 元数据适配层，不提供真实计算内核。

本模块只能在 vLLM 解析 Platform 返回的类路径时加载；不能在安装
Platform Hook 时导入，否则会提前触发 vLLM 的设备检测。
"""

try:
    from vllm.v1.attention.backend import AttentionBackend
except ModuleNotFoundError as exc:
    # 旧版 vLLM 的抽象接口位于 attention.backends.abstract。
    # 只回退模块迁移，不能吞掉新接口自身的依赖缺失错误。
    if exc.name not in ("vllm.v1.attention", "vllm.v1.attention.backend"):
        raise
    from vllm.attention.backends.abstract import AttentionBackend


class SimulatedAttentionBackend(AttentionBackend):
    """仅供 KV 元数据和 layout 协商使用，由 Worker Hook 接管计算。"""

    @staticmethod
    def get_name() -> str:
        return "HISIM_METADATA"

    @classmethod
    def supported_kv_cache_layouts(cls):
        # 没有真实内核，故不额外限制 layout。仍由原生 resolver 联合
        # cache config、KVCacheSpec 和 connector 的约束选择布局。
        return None

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError(
            "HiSim metadata backend has no attention kernel; "
            "C_VLLMWorkerHook must handle model execution"
        )

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError(
            "HiSim metadata backend has no attention metadata builder; "
            "C_VLLMWorkerHook must handle model execution"
        )

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size,
        cache_dtype_str="auto", **kwargs,
    ):
        # 旧版 API 的逻辑 NHD shape；只返回尺寸，不分配 KV 张量。
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def is_mla(cls) -> bool:
        return False

    @classmethod
    def is_ssm(cls) -> bool:
        return False


class SimulatedMLAAttentionBackend(SimulatedAttentionBackend):
    """保留 MLA 身份及压缩 KV shape，不把 MLA 冒充普通 attention。"""

    @staticmethod
    def get_name() -> str:
        return "HISIM_MLA_METADATA"

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size,
        cache_dtype_str="auto", **kwargs,
    ):
        return (num_blocks, block_size, head_size)
