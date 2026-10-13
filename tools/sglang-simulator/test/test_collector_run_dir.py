"""Tests for the per-round export subdirectory in collector vllm_hook.

worker_hook.get_output_dir(is_start) mints a fresh run subdirectory
({proc_start_ts}_run{NNN}) on every start_profile and reuses it on
stop_profile, so successive profile rounds no longer overwrite each
other. EngineCore and Worker processes keep independent call counters
that stay aligned 1:1 via collective_rpc, pairing the same round's
rank0.requests.jsonl and rank{N}.schedule_batch.jsonl by "run{NNN}".

torch is stubbed: worker_hook only touches torch.cuda.Event at runtime,
never at import or get_output_dir time.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("torch", MagicMock())

from sglang_simulator.collector.vllm_hook import worker_hook as wh


@pytest.fixture()
def export_env(tmp_path, monkeypatch):
    """Fresh process state + SIM_COLLECTOR_OUTPUT_DIR/DS_LLM_PROC_RANK."""
    monkeypatch.setattr(wh, "_PROFILE_ROUND", 0)
    monkeypatch.setattr(wh, "_CURRENT_RUN_SUBDIR", None)
    monkeypatch.setenv("SIM_COLLECTOR_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("DS_LLM_PROC_RANK", "3")
    return tmp_path


def test_start_mints_run_dir_under_engine_rank(export_env):
    out = wh.get_output_dir(is_start=True)
    assert out.endswith(os.path.join("engine3", f"{wh._PROCESS_START_TS}_run001"))
    assert os.path.isdir(out)


def test_stop_reuses_round_minted_by_start(export_env):
    start_dir = wh.get_output_dir(is_start=True)
    assert wh.get_output_dir(is_start=False) == start_dir
    # repeated stops keep the same round dir until the next start
    assert wh.get_output_dir(is_start=False) == start_dir


def test_successive_rounds_do_not_overwrite(export_env):
    round1 = wh.get_output_dir(is_start=True)
    wh.get_output_dir(is_start=False)
    round2 = wh.get_output_dir(is_start=True)
    assert round1.endswith("_run001")
    assert round2.endswith("_run002")
    assert round1 != round2


def test_worker_process_counter_aligns_by_run_number(export_env):
    # A second process (e.g. a TP worker) keeps its own counter; the same
    # profile round pairs across processes purely by the run number.
    engine_round1 = wh.get_output_dir(is_start=True)
    wh._PROFILE_ROUND = 0
    wh._CURRENT_RUN_SUBDIR = None  # simulate the worker process
    worker_round1 = wh.get_output_dir(is_start=True)
    assert os.path.basename(engine_round1) == os.path.basename(worker_round1)
    assert os.path.basename(os.path.dirname(worker_round1)) == "engine3"


def test_stop_before_any_start_lazily_mints(export_env):
    assert wh.get_output_dir(is_start=False).endswith("_run001")


def test_pid_fallback_without_engine_rank(export_env, monkeypatch):
    monkeypatch.delenv("DS_LLM_PROC_RANK")
    out = wh.get_output_dir(is_start=True)
    assert os.path.basename(os.path.dirname(out)) == f"pid{os.getpid()}"


def test_restart_with_new_process_ts_does_not_collide(export_env, monkeypatch):
    round_before = wh.get_output_dir(is_start=True)
    monkeypatch.setattr(wh, "_PROFILE_ROUND", 0)
    monkeypatch.setattr(wh, "_CURRENT_RUN_SUBDIR", None)
    monkeypatch.setattr(wh, "_PROCESS_START_TS", "20990101_000000")
    round_after = wh.get_output_dir(is_start=True)
    assert "20990101_000000_run001" in round_after
    assert round_after != round_before


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
