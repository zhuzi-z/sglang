"""Tests for the per-call export directory in collector vllm_hook.

The collector records continuously; every profile call -- /start_profile and
/stop_profile alike -- dumps what accumulated since the previous call into
its own directory:

    $SIM_COLLECTOR_OUTPUT_DIR/engine{RANK}/run{NNN}/

One call = one directory shared by the EngineCore and every worker of the
node (the rank in each file name keeps them apart). The EngineCore allocates
the number and records it in the per-node .current_run marker; the workers
join it, so a call never splits its files across run001/run002 and never
overwrites the previous call's files.

torch is stubbed: worker_hook only touches torch.cuda.Event at runtime,
never at import or get_output_dir time.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("torch", MagicMock())

from sglang_simulator.collector.vllm_hook import worker_hook as wh

SUFFIX = ".gz" if wh.EXPORT_GZIP else ""
ENGINE = "engine3"


@pytest.fixture()
def export_env(tmp_path, monkeypatch):
    """Fresh node state + SIM_COLLECTOR_OUTPUT_DIR/DS_LLM_PROC_RANK."""
    monkeypatch.setenv("SIM_COLLECTOR_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("DS_LLM_PROC_RANK", "3")
    return tmp_path


def _drain_exports():
    """Join the export thread started by a start-side (async) export."""
    while wh._LAST_EXPORT_THREAD is not None:
        wh._LAST_EXPORT_THREAD.join()
        wh._LAST_EXPORT_THREAD = None


def _runs(root):
    """runNNN directories only (the .current_run marker is a file)."""
    return sorted(
        n
        for n in os.listdir(os.path.join(root, ENGINE))
        if os.path.isdir(os.path.join(root, ENGINE, n))
    )


def _names(root, run):
    return sorted(os.listdir(os.path.join(root, ENGINE, run)))


def test_every_call_mints_plain_run_dir(export_env):
    assert wh.get_output_dir().endswith(os.path.join(ENGINE, "run001"))
    # the next call gets its own directory -- a stop dumps its own slice too
    assert wh.get_output_dir().endswith(os.path.join(ENGINE, "run002"))
    assert wh.get_output_dir().endswith(os.path.join(ENGINE, "run003"))


def test_enginecall_puts_all_exports_in_one_dir(export_env):
    class FakeEngineCore:
        def profile(self, is_start=True, profile_prefix=None):
            return "orig"

    wh.C_EngineCoreHook.hook(FakeEngineCore)
    core = FakeEngineCore()
    core.profile(True)
    _drain_exports()
    core.profile(False)

    assert _runs(export_env) == ["run001", "run002"]
    for run in ("run001", "run002"):
        assert _names(export_env, run) == [
            f"rank0.engine_steps.jsonl{SUFFIX}",
            f"rank0.requests.jsonl{SUFFIX}",
        ]


def test_workers_join_the_engine_dir_of_the_same_call(export_env):
    """Fan-out order: EngineCore allocates, then every worker joins."""

    class FakeEngineCore:
        def profile(self, is_start=True, profile_prefix=None):
            return "orig"

    class FakeWorker:
        def __init__(self, rank):
            self.rank = rank

        def execute_model(self, scheduler_output, *args, **kwargs):
            return None

        def profile(self, is_start=True, profile_prefix=None):
            return None

    wh.C_EngineCoreHook.hook(FakeEngineCore)
    wh.C_WorkerHook.hook(FakeWorker)
    FakeEngineCore().profile(True)
    _drain_exports()
    for rank in range(2):
        FakeWorker(rank).profile(True)
        _drain_exports()

    assert _runs(export_env) == ["run001"]
    assert _names(export_env, "run001") == [
        f"rank0.engine_steps.jsonl{SUFFIX}",
        f"rank0.requests.jsonl{SUFFIX}",
        f"rank0.schedule_batch.jsonl{SUFFIX}",
        f"rank0.transfer.jsonl{SUFFIX}",
        f"rank1.schedule_batch.jsonl{SUFFIX}",
        f"rank1.transfer.jsonl{SUFFIX}",
    ]


def test_worker_allocates_when_node_has_no_engine(export_env):
    """No marker (no EngineCore on this node): first worker allocates, the
    rest join it instead of minting a directory each."""
    first = wh.get_output_dir(join=True)
    second = wh.get_output_dir(join=True)
    assert first.endswith(os.path.join(ENGINE, "run001"))
    assert second == first


def test_export_publishes_only_after_close(export_env):
    """A file carries its final name only once it is complete."""
    run_dir = os.path.join(export_env, ENGINE, "run001")
    os.makedirs(run_dir)
    path = os.path.join(run_dir, f"rank0.requests.jsonl{SUFFIX}")

    with wh._open_export(path) as f:
        f.write("{\"rid\": \"r1\"}\n")
        # still being written: only the .profiling name exists on disk
        assert os.path.exists(path + wh._INPROGRESS_SUFFIX)
        assert not os.path.exists(path)

    assert os.path.exists(path)
    assert not os.path.exists(path + wh._INPROGRESS_SUFFIX)


def test_failed_export_leaves_profiling_file(export_env):
    """A write that never completes keeps the .profiling name, so a reader
    never mistakes a truncated file for a finished one."""
    run_dir = os.path.join(export_env, ENGINE, "run001")
    os.makedirs(run_dir)
    path = os.path.join(run_dir, f"rank0.engine_steps.jsonl{SUFFIX}")

    with pytest.raises(RuntimeError):
        with wh._open_export(path) as f:
            f.write("{\"seq\": 1}\n")
            raise RuntimeError("disk full")

    assert os.path.exists(path + wh._INPROGRESS_SUFFIX)
    assert not os.path.exists(path)


def test_previous_round_is_never_overwritten(export_env):
    first = wh.get_output_dir()
    with open(os.path.join(first, "keep.me"), "w") as f:
        f.write("x")
    second = wh.get_output_dir()
    worker_dir = wh.get_output_dir(join=True)
    assert first.endswith("run001")
    assert second.endswith("run002")
    assert worker_dir == second
    assert os.path.exists(os.path.join(first, "keep.me"))


def test_collision_archives_the_existing_dir(export_env):
    """A taken runNNN is moved aside; the caller keeps the plain name."""
    root = os.path.join(export_env, ENGINE)
    path = os.path.join(root, "run001")
    os.makedirs(path)
    with open(os.path.join(path, "stale.txt"), "w") as f:
        f.write("x")

    wh._archive_run_dir(path)

    assert not os.path.exists(path)
    archived = [n for n in os.listdir(root) if n.startswith("run001_old_")]
    assert len(archived) == 1
    assert os.listdir(os.path.join(root, archived[0])) == ["stale.txt"]
    # archived dirs are not counted as run dirs
    assert wh._run_number(archived[0]) == 0


def test_numbering_continues_after_restart(export_env):
    """The allocator reads the node state, so a restarted process continues
    the sequence instead of reusing run001."""
    assert wh.get_output_dir().endswith("run001")
    assert wh.get_output_dir().endswith("run002")


def test_pid_fallback_without_engine_rank(export_env, monkeypatch):
    monkeypatch.delenv("DS_LLM_PROC_RANK")
    out = wh.get_output_dir()
    assert os.path.basename(os.path.dirname(out)) == f"pid{os.getpid()}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
