"""Isolated lifecycle recorder tests; no simulator/engine package import."""

import builtins
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import uuid


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "sglang_simulator" / "lifecycle_recorder.py"
SPEC = importlib.util.spec_from_file_location("_lifecycle_recorder_unit", MODULE_PATH)
lifecycle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lifecycle)
LifecycleRecorder = lifecycle.LifecycleRecorder


class _ControlledFile:
    """Gate writer I/O deterministically, or inject a sensitive error message."""

    def __init__(self, raw, pause=False, failure=None):
        self.raw = raw
        self.pause = pause
        self.failure = failure
        self.started = threading.Event()
        self.release = threading.Event()
        self.io_threads = []

    def write(self, data):
        self.io_threads.append(threading.get_ident())
        if self.pause:
            self.started.set()
            if not self.release.wait(10):
                raise TimeoutError("test writer gate was not released")
        if self.failure == "write":
            raise OSError("sensitive-path-and-token-must-not-be-recorded")
        if self.failure == "short_write":
            return len(data) - 1
        return self.raw.write(data)

    def flush(self):
        self.io_threads.append(threading.get_ident())
        if self.failure == "flush":
            raise OSError("sensitive-path-and-token-must-not-be-recorded")
        return self.raw.flush()

    def close(self):
        self.io_threads.append(threading.get_ident())
        self.raw.close()
        if self.failure == "close":
            raise OSError("sensitive-path-and-token-must-not-be-recorded")


class TestLifecycleRecorder(unittest.TestCase):
    def setUp(self):
        # Keep even temporary test artifacts within the authorized copy.
        self.tempdir = tempfile.TemporaryDirectory(
            prefix=".lifecycle-test-", dir=str(ROOT),
        )
        self.addCleanup(self.tempdir.cleanup)
        self.directory = Path(self.tempdir.name)

    def recorder(self, **kwargs):
        recorder = LifecycleRecorder(self.directory, "engine", **kwargs)
        self.addCleanup(recorder.close)
        return recorder

    def controlled_recorder(self, pause=False, failure=None, **kwargs):
        files = []

        def opener(path, mode):
            self.assertEqual(mode, "xb")
            controlled = _ControlledFile(
                builtins.open(path, mode), pause=pause, failure=failure,
            )
            files.append(controlled)
            return controlled

        with patch.object(lifecycle, "open", opener, create=True):
            recorder = self.recorder(**kwargs)
        controlled = files[0]
        # LIFO cleanup releases a stalled writer before close/tempdir cleanup.
        self.addCleanup(controlled.release.set)
        return recorder, controlled

    def records(self, recorder):
        return [
            json.loads(line)
            for line in Path(recorder.path).read_bytes().splitlines()
        ]

    def test_direct_import_has_no_heavy_dependencies(self):
        code = (
            "import importlib.util, sys; "
            "spec = importlib.util.spec_from_file_location('recorder', sys.argv[1]); "
            "module = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module); "
            "assert not any(name.split('.')[0] in "
            "{'sglang_simulator', 'sglang', 'vllm', 'torch', 'numpy'} "
            "for name in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code, str(MODULE_PATH)],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_from_env_disabled_by_default_and_empty_directory(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(LifecycleRecorder.from_env("engine"))
            self.assertIsNone(lifecycle.from_env("engine"))
            os.environ["SGLANG_SIMULATOR_LIFECYCLE_DIR"] = ""
            self.assertIsNone(LifecycleRecorder.from_env("engine"))
            self.assertIsNone(lifecycle.from_env("engine"))
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_from_env_only_reads_configured_directory(self):
        directory = self.directory / "user-configured" / "nested"
        reads = []

        class Environment(dict):
            def get(self, key, default=None):
                reads.append(key)
                return super().get(key, default)

        env = Environment({
            "SGLANG_SIMULATOR_LIFECYCLE_DIR": str(directory),
            "SGLANG_SIMULATOR_LIFECYCLE_RUN_ID": "not-a-uuid-secret-token",
            "TOKEN": "must-not-be-read",
        })
        with patch.object(lifecycle.os, "environ", env):
            recorder = LifecycleRecorder.from_env("stage")
        self.addCleanup(recorder.close)
        self.assertEqual(reads, ["SGLANG_SIMULATOR_LIFECYCLE_DIR"])
        self.assertEqual(Path(recorder.path).parent, directory)
        self.assertEqual(str(uuid.UUID(recorder.run_id)), recorder.run_id)
        self.assertNotEqual(recorder.run_id, env["SGLANG_SIMULATOR_LIFECYCLE_RUN_ID"])

    def test_unique_uuid_files_preserve_existing_files(self):
        existing = self.directory / "existing.jsonl"
        existing.write_bytes(b"existing-content\n")
        first = self.recorder()
        second = self.recorder()
        self.assertNotEqual(first.path, second.path)
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual(first.process_id, second.process_id)
        for recorder in (first, second):
            name = Path(recorder.path).name
            uuid.UUID(name.removeprefix("lifecycle-").removesuffix(".jsonl"))
            self.assertEqual(Path(recorder.path).read_bytes(), b"")
        self.assertEqual(existing.read_bytes(), b"existing-content\n")

    def test_fork_reset_changes_process_identity_without_closing_parent(self):
        parent = self.recorder()
        with patch.object(lifecycle, "_PROCESS_ID", parent.process_id):
            lifecycle._after_fork()
            child = self.recorder()
            self.assertNotEqual(child.process_id, parent.process_id)
            self.assertTrue(parent.emit("parent.after_fork"))
            self.assertTrue(child.emit("child.after_fork"))
            self.assertTrue(parent.flush() and child.flush())
            self.assertEqual(self.records(parent)[0]["process_id"], parent.process_id)
            self.assertEqual(self.records(child)[0]["process_id"], child.process_id)

    def test_uuid_collision_retries_exclusive_without_overwriting(self):
        run, session, collision, unique = [uuid.uuid4() for _ in range(4)]
        existing = self.directory / ("lifecycle-" + str(collision) + ".jsonl")
        existing.write_bytes(b"keep-this\n")
        with patch.object(
            lifecycle.uuid, "uuid4", side_effect=[run, session, collision, unique],
        ):
            recorder = self.recorder()
        self.assertEqual(Path(recorder.path).name, "lifecycle-" + str(unique) + ".jsonl")
        self.assertEqual(existing.read_bytes(), b"keep-this\n")

    def test_persistent_uuid_collision_is_bounded_and_preserves_existing_file(self):
        collision = uuid.uuid4()
        existing = self.directory / ("lifecycle-" + str(collision) + ".jsonl")
        existing.write_bytes(b"keep-this\n")
        with patch.object(lifecycle.uuid, "uuid4", return_value=collision) as generate:
            recorder = self.recorder()
        self.assertEqual(generate.call_count, 18)
        self.assertEqual(recorder.status()["error"], "FileExistsError")
        self.assertFalse(recorder.emit("rejected"))
        self.assertFalse(recorder.close())
        self.assertEqual(existing.read_bytes(), b"keep-this\n")

    def test_supplied_run_uuid_and_role_cannot_affect_filename(self):
        run_id = uuid.uuid4()
        recorder = LifecycleRecorder(self.directory, "../../outside", run_id=run_id)
        self.addCleanup(recorder.close)
        self.assertEqual(recorder.run_id, str(run_id))
        self.assertEqual(Path(recorder.path).parent, self.directory)
        self.assertNotIn(str(run_id), Path(recorder.path).name)
        self.assertNotIn("outside", Path(recorder.path).name)

    def test_invalid_configuration_creates_no_file(self):
        for kwargs in (
            {"queue_capacity": 0}, {"queue_capacity": -1},
            {"queue_capacity": 1.5}, {"max_event_bytes": 0},
            {"max_event_bytes": -1}, {"max_event_bytes": 1.5},
            {"run_id": "arbitrary-token"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.recorder(**kwargs)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_standard_identity_event_and_timestamps_cannot_be_overridden(self):
        recorder = self.recorder()
        payload = {"nested": {"number": 1}}
        with patch.object(lifecycle.time, "time_ns", return_value=101), patch.object(
            lifecycle.time, "monotonic_ns", return_value=202,
        ):
            self.assertTrue(recorder.emit(
                "actual-event", **{
                    "event": "spoofed", "run_id": "spoofed",
                    "session_id": "spoofed", "process_id": "spoofed", "instance_id": "spoofed",
                    "pid": -1, "role": "spoofed", "schema_version": -1,
                    "seq": -1, "time_ns": -1, "monotonic_ns": -1,
                    "payload": payload,
                },
            ))
        self.assertTrue(recorder.flush())
        record = self.records(recorder)[0]
        self.assertEqual(record["event"], "actual-event")
        self.assertEqual(record["run_id"], recorder.run_id)
        self.assertEqual(record["session_id"], recorder.session_id)
        self.assertEqual(record["process_id"], recorder.process_id)
        self.assertEqual(record["instance_id"], socket.gethostname())
        self.assertEqual(record["pid"], os.getpid())
        self.assertEqual(record["role"], "engine")
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["seq"], 1)
        self.assertEqual(record["time_ns"], 101)
        self.assertEqual(record["monotonic_ns"], 202)
        self.assertEqual(payload, {"nested": {"number": 1}})

    def test_concurrent_emit_has_monotonic_file_order(self):
        recorder = self.recorder(queue_capacity=2048)
        start = threading.Barrier(9)
        results = [None] * 8

        def producer(index):
            start.wait()
            results[index] = [
                recorder.emit("concurrent", producer=index, index=value)
                for value in range(100)
            ]

        threads = [threading.Thread(target=producer, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        start.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertTrue(all(all(result) for result in results))
        self.assertTrue(recorder.flush())
        records = self.records(recorder)
        self.assertEqual([record["seq"] for record in records], list(range(1, 801)))
        monotonic = [record["monotonic_ns"] for record in records]
        self.assertEqual(monotonic, sorted(monotonic))
        self.assertEqual(len({(r["producer"], r["index"]) for r in records}), 800)
        self.assertEqual(recorder.status()["written"], 800)
        self.assertEqual(recorder.status()["dropped"], 0)

    def test_full_queue_never_waits_or_performs_io_on_emitter(self):
        recorder, controlled = self.controlled_recorder(pause=True, queue_capacity=1)
        self.assertTrue(recorder._thread.daemon)
        self.assertTrue(recorder.emit("active"))
        self.assertTrue(controlled.started.wait(5))
        self.assertTrue(recorder.emit("queued"))
        begin = time.monotonic()
        results = [recorder.emit("dropped", attempt=i) for i in range(1000)]
        self.assertLess(time.monotonic() - begin, 0.5)
        self.assertFalse(any(results))
        self.assertFalse(recorder.flush(timeout=5))  # Full returns immediately.
        status = recorder.status()
        self.assertEqual(status["queue_size"], 1)
        self.assertEqual(status["seq"], 1002)
        self.assertEqual(status["dropped"], 1000)
        controlled.release.set()
        self.assertTrue(recorder.close())
        self.assertEqual([r["seq"] for r in self.records(recorder)], [1, 2])
        self.assertTrue(controlled.io_threads)
        self.assertNotIn(threading.get_ident(), controlled.io_threads)

    def test_flush_control_occupies_the_same_bounded_queue(self):
        recorder, controlled = self.controlled_recorder(pause=True, queue_capacity=1)
        self.assertTrue(recorder.emit("active"))
        self.assertTrue(controlled.started.wait(5))
        admitted = threading.Event()
        original_put = recorder._queue.put_nowait

        def put(item):
            original_put(item)
            if isinstance(item, lifecycle._Barrier):
                admitted.set()

        result = []
        with patch.object(recorder._queue, "put_nowait", side_effect=put):
            thread = threading.Thread(target=lambda: result.append(recorder.flush()))
            thread.start()
            self.assertTrue(admitted.wait(5))
            self.assertEqual(recorder.status()["queue_size"], 1)
            self.assertFalse(recorder.emit("cannot-bypass-barrier"))
            self.assertFalse(recorder.flush(timeout=0))
            self.assertEqual(recorder.status()["dropped"], 1)
            controlled.release.set()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(result, [True])
        self.assertEqual(recorder.status()["seq"], 2)
        self.assertEqual(recorder.status()["written"], 1)

    def test_flush_timeout_does_not_reset_or_close_recorder(self):
        recorder, controlled = self.controlled_recorder(pause=True, queue_capacity=2)
        self.assertTrue(recorder.emit("before-timeout"))
        self.assertTrue(controlled.started.wait(5))
        begin = time.monotonic()
        self.assertFalse(recorder.flush(timeout=0.02))
        self.assertLess(time.monotonic() - begin, 0.5)
        self.assertEqual(recorder.status()["seq"], 1)
        self.assertEqual(recorder.status()["dropped"], 0)
        self.assertTrue(recorder.emit("after-timeout"))
        controlled.release.set()
        self.assertTrue(recorder.close())
        self.assertEqual(recorder.status()["written"], 2)

    def test_flush_preserves_counters_and_accepts_later_events(self):
        recorder = self.recorder()
        self.assertTrue(recorder.emit("valid"))
        self.assertTrue(recorder.emit("invalid", value=float("nan")))
        self.assertTrue(recorder.flush())
        before = recorder.status()
        self.assertEqual((before["seq"], before["written"], before["dropped"]), (2, 1, 1))
        self.assertTrue(recorder.flush())
        self.assertEqual(recorder.status(), before)
        self.assertTrue(recorder.emit("later"))
        self.assertTrue(recorder.flush())
        self.assertEqual([r["seq"] for r in self.records(recorder)], [1, 3])
        self.assertEqual(recorder.status()["dropped"], 1)

    def test_byte_cap_drops_ascii_and_utf8_oversize_without_poisoning_writer(self):
        recorder = self.recorder(max_event_bytes=1024)
        self.assertTrue(recorder.emit("oversize", value="x" * 2048))
        self.assertTrue(recorder.emit("oversize", value="界" * 300))
        self.assertTrue(recorder.emit("valid", value="small"))
        self.assertTrue(recorder.flush())
        self.assertEqual([r["seq"] for r in self.records(recorder)], [3])
        self.assertLessEqual(len(Path(recorder.path).read_bytes()), 1024)
        self.assertEqual(recorder.status()["dropped"], 2)
        self.assertIsNone(recorder.status()["error"])

    def test_byte_cap_exact_boundary_includes_newline(self):
        with patch.object(lifecycle.time, "time_ns", return_value=1), patch.object(
            lifecycle.time, "monotonic_ns", return_value=2,
        ):
            probe = self.recorder()
            self.assertTrue(probe.emit("boundary", value="é"))
            self.assertTrue(probe.flush())
            size = len(Path(probe.path).read_bytes())
            exact = self.recorder(max_event_bytes=size)
            too_small = self.recorder(max_event_bytes=size - 1)
            self.assertTrue(exact.emit("boundary", value="é"))
            self.assertTrue(too_small.emit("boundary", value="é"))
            self.assertTrue(exact.flush())
            self.assertTrue(too_small.flush())
        self.assertEqual(len(Path(exact.path).read_bytes()), size)
        self.assertEqual(too_small.status()["dropped"], 1)
        self.assertEqual(Path(too_small.path).read_bytes(), b"")

    def test_nan_infinity_nonserializable_and_cycle_are_dropped(self):
        recorder = self.recorder()
        cycle = []
        cycle.append(cycle)
        values = [float("nan"), float("inf"), -float("inf"), {1}, object(), cycle]
        for value in values:
            self.assertTrue(recorder.emit("invalid", value=value))
        self.assertTrue(recorder.emit("valid", value=1))
        self.assertTrue(recorder.flush())
        self.assertEqual(recorder.status()["dropped"], len(values))
        self.assertEqual(recorder.status()["written"], 1)
        self.assertIsNone(recorder.status()["error"])
        self.assertTrue(recorder._thread.is_alive())

    def test_write_failure_drops_pending_events_and_rejects_future_emits(self):
        recorder, controlled = self.controlled_recorder(pause=True, failure="write")
        self.assertTrue(recorder.emit("active"))
        self.assertTrue(controlled.started.wait(5))
        for _ in range(3):
            self.assertTrue(recorder.emit("pending"))
        controlled.release.set()
        self.assertFalse(recorder.flush())
        self.assertFalse(recorder.emit("after-error"))
        self.assertFalse(recorder.close())
        status = recorder.status()
        self.assertEqual(status["error"], "OSError")
        self.assertEqual((status["seq"], status["written"], status["dropped"]), (5, 0, 5))
        self.assertEqual(status["queue_size"], 0)
        self.assertTrue(controlled.raw.closed)
        self.assertFalse(recorder._thread.is_alive())

    def test_short_write_and_flush_failure_disable_recording(self):
        for failure in ("short_write", "flush"):
            with self.subTest(failure=failure):
                recorder, controlled = self.controlled_recorder(failure=failure)
                self.assertTrue(recorder.emit("event"))
                self.assertFalse(recorder.flush())
                self.assertEqual(recorder.status()["error"], "OSError")
                self.assertFalse(recorder.emit("rejected"))
                self.assertFalse(recorder.close())
                self.assertTrue(controlled.raw.closed)

    def test_initial_open_failure_is_nonthrowing_and_sanitized(self):
        with patch.object(
            lifecycle, "open", create=True,
            side_effect=PermissionError("sensitive-path-and-token-must-not-be-recorded"),
        ):
            recorder = self.recorder()
        self.assertEqual(recorder.status()["error"], "PermissionError")
        self.assertFalse(recorder.emit("rejected"))
        self.assertFalse(recorder.flush())
        self.assertFalse(recorder.close())
        self.assertEqual(recorder.status()["dropped"], 1)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_initial_directory_failure_is_nonthrowing_and_sanitized(self):
        not_a_directory = self.directory / "regular-file"
        not_a_directory.write_bytes(b"preserve")
        recorder = LifecycleRecorder(not_a_directory / "child", "engine")
        self.addCleanup(recorder.close)
        self.assertEqual(recorder.status()["error"], "NotADirectoryError")
        self.assertFalse(recorder.emit("rejected"))
        self.assertFalse(recorder.flush())
        self.assertEqual(not_a_directory.read_bytes(), b"preserve")

    def test_close_drains_flushes_is_idempotent_and_rejects_events(self):
        recorder = self.recorder()
        for index in range(10):
            self.assertTrue(recorder.emit("event", index=index))
        self.assertTrue(recorder.close())
        self.assertTrue(recorder.close())
        self.assertEqual(len(self.records(recorder)), 10)
        self.assertTrue(recorder._file.closed)
        self.assertFalse(recorder._thread.is_alive())
        self.assertFalse(recorder.flush())
        self.assertFalse(recorder.emit("after-close"))
        status = recorder.status()
        self.assertEqual((status["seq"], status["written"], status["dropped"]), (11, 10, 1))
        self.assertEqual(status["queue_size"], 0)

    def test_close_timeout_with_full_queue_still_eventually_closes(self):
        recorder, controlled = self.controlled_recorder(pause=True, queue_capacity=1)
        self.assertTrue(recorder.emit("active"))
        self.assertTrue(controlled.started.wait(5))
        self.assertTrue(recorder.emit("pending"))
        self.assertFalse(recorder.close(timeout=0.02))
        self.assertFalse(recorder.emit("after-close-started"))
        controlled.release.set()
        self.assertTrue(recorder._stopped.wait(5))  # No second close is required.
        self.assertTrue(controlled.raw.closed)
        self.assertTrue(recorder.close())
        self.assertEqual([r["seq"] for r in self.records(recorder)], [1, 2])

    def test_repeated_close_queues_only_one_bounded_stop_control(self):
        recorder, controlled = self.controlled_recorder(pause=True, queue_capacity=2)
        self.assertTrue(recorder.emit("active"))
        self.assertTrue(controlled.started.wait(5))
        for _ in range(20):
            self.assertFalse(recorder.close(timeout=0))
            self.assertEqual(recorder.status()["queue_size"], 1)
        self.assertFalse(recorder.flush(timeout=0))
        self.assertEqual(recorder.status()["seq"], 1)
        self.assertEqual(recorder.status()["dropped"], 0)
        controlled.release.set()
        self.assertTrue(recorder.close())
        self.assertEqual(recorder.status()["queue_size"], 0)

    def test_close_io_failure_is_sanitized(self):
        recorder, controlled = self.controlled_recorder(failure="close")
        self.assertTrue(recorder.emit("event"))
        self.assertFalse(recorder.close())
        self.assertEqual(recorder.status()["error"], "OSError")
        self.assertTrue(controlled.raw.closed)
        self.assertFalse(recorder._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
