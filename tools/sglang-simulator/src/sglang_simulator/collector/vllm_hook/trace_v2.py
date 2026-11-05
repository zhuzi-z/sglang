"""Bounded host-event recording. No torch imports or device synchronization."""
import atexit
from collections import deque
from contextvars import ContextVar
import json
import os
from pathlib import Path
import socket
import threading
import time
import uuid
import warnings

ACTIVE = ContextVar('sim_trace_v2_scope', default=None)
RPC_CONTEXT = '_sim_collector_v2_context'
_RECORDER = None
_INIT_LOCK = threading.Lock()
RANK_FIELDS = ('node', 'global', 'local', 'pp', 'pcp', 'tp', 'dp')


class Scope:
    __slots__ = ('context', 'chunk')

    def __init__(self, context):
        self.context = context
        self.chunk = -1


class Recorder:
    def __init__(self, root, capacity=8192, interval=.25, max_bytes=268435456,
                 namespace='trace_v2'):
        if capacity < 1 or interval <= 0 or max_bytes < 1:
            raise ValueError('positive trace limits required')
        self.capacity, self.interval, self.max_bytes = capacity, interval, max_bytes
        self.focus_pp0 = os.getenv("SIM_TRACE_FOCUS") == "pp0_step"
        self.pid = os.getpid()
        self.identity = (-1,) * len(RANK_FIELDS)
        self.process_id = uuid.uuid4().hex
        self.collection_id = os.getenv('SIM_TRACE_COLLECTION_ID', '')[:128]
        self.build_id = os.getenv('SIM_TRACE_BUILD_ID', '')[:128]
        self.directory = Path(root) / namespace / socket.gethostname() / f'{self.pid}-{self.process_id}'
        self._events = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.accepted = self.written = self.dropped_full = self.dropped_busy = 0
        self.rejected = self.bytes_written = self.observer_errors = 0
        self.failure = None
        self.closed = False
        self.wall_anchor_ns = time.time_ns()
        self.mono_anchor_ns = time.monotonic_ns()
        self._thread = threading.Thread(target=self._write, name='sim-trace-writer', daemon=True)
        self._thread.start()

    def emit(self, kind, context=None, chunk=-1, data=None):
        """Best effort: no waiting, serialization, I/O, logging or wakeups."""
        if self.failure is not None or self._stop.is_set():
            self.rejected += 1
            return False
        if self.focus_pp0:
            from .pp0_steps import keep_event
            if not keep_event(kind, self.identity):
                return False
        stamp = time.monotonic_ns()
        if not self._lock.acquire(False):
            self.dropped_busy += 1
            return False
        try:
            if len(self._events) >= self.capacity:
                self.dropped_full += 1
                return False
            self.accepted += 1
            self._events.append((stamp, self.accepted, kind, context, chunk, self.identity, data))
            return True
        finally:
            self._lock.release()

    def status(self):
        return {key: getattr(self, key) for key in (
            'accepted', 'written', 'dropped_full', 'dropped_busy', 'rejected',
            'bytes_written', 'failure', 'closed', 'observer_errors')}

    def _metadata(self):
        return {'schema': 2, 'measurement_kind': 'host_observation',
                'pid': os.getpid(), 'process_id': self.process_id,
                'collection_id': self.collection_id, 'build_id': self.build_id, 'trace_focus': 'pp0_step' if self.focus_pp0 else 'full',
                'clock_domain': socket.gethostname(),
                'wall_anchor_ns': self.wall_anchor_ns, 'mono_anchor_ns': self.mono_anchor_ns,
                'rank_fields': RANK_FIELDS, **self.status()}

    def _save_metadata(self):
        path = self.directory / 'process.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(self._metadata()) + '\n')
        os.replace(temp, path)

    def _write(self):
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            self._save_metadata()
            with (self.directory / 'events.jsonl').open('w') as stream:
                while True:
                    self._stop.wait(self.interval)
                    with self._lock:
                        batch, self._events = self._events, deque()
                    for stamp, seq, kind, context, chunk, identity, data in batch:
                        row = {'schema': 2, 'mono_ns': stamp, 'seq': seq, 'event': kind,
                               'context': context, 'chunk': chunk,
                               'rank': dict(zip(RANK_FIELDS, identity)), 'data': data}
                        line = json.dumps(row, separators=(',', ':')) + '\n'
                        size = len(line.encode('utf-8'))
                        if self.bytes_written + size > self.max_bytes:
                            raise RuntimeError('trace byte limit reached')
                        stream.write(line)
                        self.bytes_written += size
                        self.written += 1
                    stream.flush()
                    self._save_metadata()
                    if self._stop.is_set():
                        break
            self.closed = True
            self._save_metadata()
        except Exception as exc:
            # Never report a writer error through a serving method.
            self.failure = 'byte_limit' if str(exc) == 'trace byte limit reached' else type(exc).__name__
            try:
                self._save_metadata()
            except Exception:
                pass

    def close(self):
        if os.getpid() != self.pid:
            return
        self._stop.set()
        self._thread.join(timeout=2)


def initialize():
    global _RECORDER
    with _INIT_LOCK:
        if _RECORDER is None:
            try:
                recorder_type = Recorder
                if os.getenv('SIM_COLLECTOR_MODE') == 'pp_v3':
                    from .trace_collection import CollectionRecorder
                    recorder_type = CollectionRecorder
                _RECORDER = recorder_type(
                    os.getenv('SIM_COLLECTOR_OUTPUT_DIR', os.getcwd()),
                    capacity=int(os.getenv('SIM_TRACE_CAPACITY', '8192')),
                    interval=float(os.getenv('SIM_TRACE_FLUSH_SECONDS', '.25')),
                    max_bytes=int(os.getenv('SIM_TRACE_MAX_BYTES', '268435456')))
                atexit.register(_RECORDER.close)
            except Exception as exc:
                warnings.warn('SIM trace initialization disabled: ' + type(exc).__name__, RuntimeWarning)
                return None
    return _RECORDER


def _after_fork():
    global _RECORDER, _INIT_LOCK
    # A forked process cannot reuse a parent's writer thread or locks.
    _RECORDER = None
    _INIT_LOCK = threading.Lock()
    ACTIVE.set(None)


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)


def emit(kind, scope=None, chunk=None, data=None):
    recorder = _RECORDER
    if recorder is None:
        return
    try:
        recorder.emit(kind, scope.context if scope else None,
                      scope.chunk if chunk is None and scope else (chunk if chunk is not None else -1), data)
    except Exception:
        # Observer failures never replace a native result or exception.
        recorder.observer_errors += 1
