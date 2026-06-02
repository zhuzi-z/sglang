"""Append-only in-memory log of D2H (L1→L2 backup) transfer events.

One record per ``backup_from_device_all_layer`` call. The log is drained and
written to ``{output_dir}/d2h.jsonl`` by ``C_SchedulerHook.wrapped_profile``
together with the existing metrics.json / iteration.jsonl / request.jsonl
outputs.

Multi-TP: each TP rank has its own copy (class attribute is per-process);
only rank 0 writes the log to disk. Non-rank-0 ranks just reset.
"""
from typing import Any


class D2HLog:
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
