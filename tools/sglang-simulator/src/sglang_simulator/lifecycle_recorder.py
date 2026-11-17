"""Payloads must remain immutable after enqueue; only explicit flush/close wait."""

import json
import os
import queue
import socket
import threading
import time
import uuid
from pathlib import Path


class _Barrier:
    __slots__ = ("done", "success")

    def __init__(self):
        self.done = threading.Event()
        self.success = False


_STOP = object()
_PROCESS_ID = str(uuid.uuid4())


def _after_fork():
    global _PROCESS_ID
    _PROCESS_ID = str(uuid.uuid4())


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


class LifecycleRecorder:
    """Sequence counts attempts, dropped counts losses, and written requires flush confirmation."""

    SCHEMA_VERSION = 1

    def __init__(
        self, output_dir, role, queue_capacity=8192,
        max_event_bytes=262144, run_id=None,
    ):
        if not isinstance(queue_capacity, int) or queue_capacity <= 0:
            raise ValueError("queue_capacity must be a positive integer")
        if not isinstance(max_event_bytes, int) or max_event_bytes <= 0:
            raise ValueError("max_event_bytes must be a positive integer")

        self.run_id = str(uuid.uuid4() if run_id is None else uuid.UUID(str(run_id)))
        self.session_id = str(uuid.uuid4())
        self.process_id = _PROCESS_ID
        self.instance_id = socket.gethostname()
        self.pid = os.getpid()
        self.role = role
        self._identity = {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "process_id": self.process_id,
            "instance_id": self.instance_id,
            "pid": self.pid,
            "role": self.role,
            "schema_version": self.SCHEMA_VERSION,
        }
        self._max_event_bytes = max_event_bytes
        self._queue = queue.Queue(maxsize=queue_capacity)
        self._lock = threading.Lock()
        self._seq = 0
        self._written = 0
        self._dropped = 0
        self._error = None
        self._closing = False
        self._stopped = threading.Event()
        self._file = None
        self._thread = None
        directory = Path(output_dir)
        self.path = str(directory / ("lifecycle-" + str(uuid.uuid4()) + ".jsonl"))

        try:
            directory.mkdir(parents=True, exist_ok=True)
            # Never truncate an existing file, even in the event of a collision.
            for attempt in range(16):
                try:
                    self._file = open(self.path, "xb")
                    break
                except FileExistsError:
                    if attempt == 15:
                        raise
                    self.path = str(
                        directory / ("lifecycle-" + str(uuid.uuid4()) + ".jsonl")
                    )
            self._thread = threading.Thread(
                target=self._writer, name="lifecycle-recorder", daemon=True,
            )
            self._thread.start()
        except Exception as exc:
            self._record_error(exc)
            if self._file is not None:
                try:
                    self._file.close()
                except Exception as close_exc:
                    self._record_error(close_exc)
            self._stopped.set()

    @classmethod
    def from_env(cls, role):
        """Create a recorder only when the user explicitly configures a directory."""
        output_dir = os.environ.get("SGLANG_SIMULATOR_LIFECYCLE_DIR")
        return cls(output_dir, role) if output_dir else None

    def emit(self, event, /, **data):
        """Standard fields override payload fields; nested values are not copied."""
        with self._lock:
            self._seq += 1
            if self._closing or self._error is not None or self._stopped.is_set():
                self._dropped += 1
                return False
            data.update(self._identity)
            data.update(
                event=event, time_ns=time.time_ns(),
                monotonic_ns=time.monotonic_ns(), seq=self._seq,
            )
            try:
                self._queue.put_nowait(data)
            except queue.Full:
                self._dropped += 1
                return False
            return True

    def status(self):
        """Return a snapshot without flushing, clearing counters, or engine access."""
        with self._lock:
            return {
                "path": self.path,
                "seq": self._seq,
                "written": self._written,
                "dropped": self._dropped,
                "error": self._error,
                "queue_size": self._queue.qsize(),
            }

    def flush(self, timeout=5):
        """Wait for a queued flush barrier; a full queue or timeout returns False."""
        barrier = _Barrier()
        with self._lock:
            if self._closing or self._error is not None or self._stopped.is_set():
                return False
            try:
                self._queue.put_nowait(barrier)
            except queue.Full:
                return False
        return barrier.done.wait(timeout=max(0, timeout)) and barrier.success

    def close(self, timeout=5):
        """A timeout leaves the writer draining; repeated calls wait for the same shutdown."""
        deadline = time.monotonic() + max(0, timeout)
        with self._lock:
            if not self._closing:
                self._closing = True
                if not self._stopped.is_set():
                    try:
                        self._queue.put_nowait(_STOP)
                    except queue.Full:
                        # The writer also checks closing after every queue item.
                        pass
        if not self._stopped.wait(timeout=max(0, deadline - time.monotonic())):
            return False
        # The completion event precedes the thread's return by a few bytecodes.
        # Join within the same deadline so True also guarantees thread exit.
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(0, deadline - time.monotonic()))
            if self._thread.is_alive():
                return False
        with self._lock:
            return self._error is None

    def _record_error(self, exc):
        with self._lock:
            if self._error is None:
                self._error = type(exc).__name__

    def _drop(self):
        with self._lock:
            self._dropped += 1

    def _write_event(self, event):
        with self._lock:
            failed = self._error is not None
        if failed:
            self._drop()
            return
        try:
            # Accumulate at most the configured byte cap, not an unbounded batch.
            encoder = json.JSONEncoder(
                ensure_ascii=False, allow_nan=False, separators=(",", ":"),
            )
            encoded = bytearray()
            for chunk in encoder.iterencode(event):
                # UTF-8 uses at least one byte per character. Reject obviously
                # oversized chunks before allocating another encoded copy.
                if len(encoded) + len(chunk) + 1 > self._max_event_bytes:
                    self._drop()
                    return
                chunk = chunk.encode("utf-8")
                if len(encoded) + len(chunk) + 1 > self._max_event_bytes:
                    self._drop()
                    return
                encoded.extend(chunk)
            encoded.append(10)
        except Exception:
            # Invalid JSON payloads must not terminate the writer or leak data.
            self._drop()
            return
        try:
            if self._file.write(encoded) != len(encoded):
                raise OSError("short write")
        except Exception as exc:
            self._record_error(exc)
            self._drop()
            return
        with self._lock:
            self._written += 1

    def _flush_file(self):
        with self._lock:
            if self._error is not None:
                return False
        try:
            self._file.flush()
        except Exception as exc:
            self._record_error(exc)
            return False
        return True

    def _writer(self):
        try:
            while True:
                with self._lock:
                    if self._closing and self._queue.empty():
                        break
                try:
                    item = self._queue.get(timeout=0.25)
                except queue.Empty:
                    self._flush_file()
                    continue
                try:
                    if item is _STOP:
                        break
                    if isinstance(item, _Barrier):
                        item.success = self._flush_file()
                        item.done.set()
                    else:
                        self._write_event(item)
                finally:
                    self._queue.task_done()
        except Exception as exc:
            self._record_error(exc)
        finally:
            # Release pending barriers as failures if an unexpected writer error
            # occurs. This drain is bounded by the same queue capacity.
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    if isinstance(item, _Barrier):
                        item.done.set()
                    elif item is not _STOP:
                        self._drop()
                finally:
                    self._queue.task_done()
            self._flush_file()
            try:
                self._file.close()
            except Exception as exc:
                self._record_error(exc)
            self._stopped.set()


def from_env(role):
    """Module-level convenience factory with the same opt-in behavior."""
    return LifecycleRecorder.from_env(role)
