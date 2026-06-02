class StateManager:
    _iteration: int = 0
    _global_clock: float = 0
    _last_inference_dur: float = 0
    _current_inference_dur: float = 0
    _hicache_l2_load_dur: float = 0
    _hicache_l2_backup_dur: float = 0
    _last_real_time_ts: float = 0
    _last_flush_time_ts: float = 0
    # Per-D2H-call context set by C_HiCacheController.start_writing wrapper
    # so mem_pool_host's backup_from_device_all_layer can attach op-level
    # metadata (op_ids, node_ids) to its d2h.jsonl record.
    _current_backup_ctx: dict | None = None

    @classmethod
    def reset(cls):
        cls._iteration = 0
        cls._global_clock = 0
        cls._last_inference_dur = 0
        cls._current_inference_dur = 0
        cls._hicache_l2_backup_dur = 0
        cls._hicache_l2_load_dur = 0
        cls._last_real_time_ts = 0
        cls._current_backup_ctx = None

    @classmethod
    def inc_iteration(cls) -> None:
        cls._iteration += 1

    @classmethod
    def get_iteration(cls) -> int:
        return cls._iteration

    @classmethod
    def inc_hicache_l2_load_dur(cls, dur: float) -> None:
        cls._hicache_l2_load_dur += dur

    @classmethod
    def inc_hicache_l2_backup_dur(cls, dur: float) -> None:
        cls._hicache_l2_backup_dur += dur

    @classmethod
    def pop_hicache_l2_load_dur(cls) -> float:
        dur = cls._hicache_l2_load_dur
        cls._hicache_l2_load_dur = 0
        return dur

    @classmethod
    def pop_hicache_l2_backup_dur(cls) -> float:
        dur = cls._hicache_l2_backup_dur
        cls._hicache_l2_backup_dur = 0
        return dur

    @classmethod
    def get_global_clock(cls) -> float:
        return cls._global_clock

    @classmethod
    def step_global_clock(cls, dur: float) -> None:
        cls._global_clock += dur

    @classmethod
    def set_global_clock(cls, clock: float) -> None:
        cls._global_clock = clock

    @classmethod
    def set_current_inference_dur(cls, dur: float) -> None:
        cls._last_inference_dur = cls._current_inference_dur
        cls._current_inference_dur = dur

    @classmethod
    def get_last_inference_dur(cls) -> float:
        return cls._last_inference_dur

    @classmethod
    def get_current_inference_dur(cls) -> float:
        return cls._current_inference_dur

    @classmethod
    def set_last_real_time_ts(cls, ts):
        cls._last_real_time_ts = ts

    @classmethod
    def get_last_real_time_ts(cls):
        return cls._last_real_time_ts

    @classmethod
    def set_last_flush_time_ts(cls, ts: float):
        cls._last_flush_time_ts = ts

    @classmethod
    def get_last_flush_time_ts(cls) -> float:
        return cls._last_flush_time_ts

    @classmethod
    def set_current_backup_ctx(cls, ctx: dict | None) -> None:
        cls._current_backup_ctx = ctx

    @classmethod
    def get_current_backup_ctx(cls) -> dict | None:
        return cls._current_backup_ctx

    @classmethod
    def clear_current_backup_ctx(cls) -> None:
        cls._current_backup_ctx = None
