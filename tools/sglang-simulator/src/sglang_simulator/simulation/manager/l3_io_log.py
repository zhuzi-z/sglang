"""Append-only in-memory log of L3 (HiCache storage) IO operations.

Each scheduler step appends one record per L3 storage call (per pool for v2);
the log is drained and written to ``{output_dir}/l3_io.jsonl`` by
``C_SchedulerHook.wrapped_profile`` together with the existing
metrics.json / iteration.jsonl / request.jsonl outputs.

Multi-TP: each TP rank has its own copy (class attribute is per-process /
per-thread); only rank 0 actually writes the log to disk. Non-rank-0 ranks
just reset their own log on profile boundaries.
"""
from typing import Any


class L3IOLog:
    _records: list[dict] = []

    @classmethod
    def append(cls, record: dict) -> None:
        cls._records.append(record)

    @classmethod
    def drain(cls) -> list[dict]:
        out = cls._records
        cls._records = []
        return out

    @classmethod
    def reset(cls) -> None:
        cls._records = []

    @classmethod
    def size(cls) -> int:
        return len(cls._records)
