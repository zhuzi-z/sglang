"""Continuous collection with profile-triggered cuts, not hardware profiling."""
from collections import deque
import json
import os
from pathlib import Path
import threading
import time
import uuid

from .trace_v2 import Recorder, RANK_FIELDS


class CollectionRecorder(Recorder):
    def __init__(self, *args, **kwargs):
        self._cuts = deque()
        self._wake = threading.Event()
        self._cut_lock = threading.Lock()
        self._cut_jobs = {}
        self._window = None
        self._segment = 0
        self._segment_bytes = 0
        self._stream = None
        self._open_spans = {}
        self.exported = 0
        self.host_clock = {'source': 'time.monotonic_ns', 'boot_id': None,
                           'time_namespace': None}
        try:
            self.host_clock['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            self.host_clock['time_namespace'] = os.readlink('/proc/self/ns/time')
        except OSError:
            pass
        super().__init__(*args, namespace='trace_v3', **kwargs)

    def _metadata(self):
        return {**super()._metadata(), 'collection_contract': 3,
                'hardware_profiler': False, 'exported_windows': self.exported,
                'byte_limit_scope': 'per_segment',
                'host_clock': self.host_clock,
                'device_time_available': False}

    def export_cut(self, export_id=None, is_start=True, timeout=30):
        """Cut by process sequence; start is asynchronous, stop waits for disk.

        This never pauses collection or waits for a device. In-flight spans may
        cross cuts, and retain context/span references rather than being split.
        """
        export_id = export_id or uuid.uuid4().hex
        if not isinstance(export_id, str) or not export_id.isalnum() or len(export_id) > 64:
            raise ValueError('invalid export ID')
        done = threading.Event()
        with self._cut_lock:
            job = self._cut_jobs.get(export_id)
            if job is None:
                if self.failure is not None or self._stop.is_set():
                    return {'export_id': export_id, 'complete': False,
                            'failure': self.failure or 'recorder_closed'}
                if len(self._cut_jobs) >= 128:
                    finished = next((key for key, value in self._cut_jobs.items()
                                     if value['done'].is_set()), None)
                    if finished is None:
                        return {'export_id': export_id, 'complete': False,
                                'failure': 'export_queue_full'}
                    del self._cut_jobs[finished]
                with self._lock:
                    job = {'id': export_id, 'trigger': 'start_profile' if is_start else 'stop_profile',
                           'cut_seq': self.accepted, 'cut_mono_ns': time.monotonic_ns(),
                           'done': done, 'result': None}
                    self._cuts.append(job)
                    self._cut_jobs[export_id] = job
            else:
                done = job['done']
        self._wake.set()
        if not is_start:
            done.wait(timeout)
        return job['result'] or {'export_id': export_id, 'cut_seq': job['cut_seq'],
                                 'complete': False, 'pending': not done.is_set(),
                                 'failure': self.failure,
                                 'process_id': self.process_id}

    def _new_window(self):
        self._window = self.directory / 'pending' / uuid.uuid4().hex
        self._window.mkdir(parents=True)
        self._segment = 0
        self._segment_bytes = 0
        self._stream = (self._window / 'events.000000.jsonl').open('w')

    def _append(self, line):
        size = len(line.encode('utf-8'))
        if self._segment_bytes and self._segment_bytes + size > self.max_bytes:
            self._stream.close()
            self._segment += 1
            self._stream = (self._window / f'events.{self._segment:06d}.jsonl').open('w')
            self._segment_bytes = 0
        self._stream.write(line)
        self._segment_bytes += size
        self.bytes_written += size
        self.written += 1

    def _tables(self, window):
        """Streaming export; no full-run accumulation in serving memory."""
        names = ('requests', 'step_inputs', 'chunk_inputs', 'layouts',
                 'schedule_batch', 'engine_steps', 'step_timings')
        streams = {name: (window / (name + '.jsonl')).open('w') for name in names}
        counts = dict.fromkeys(names, 0)
        def write(name, row):
            streams[name].write(json.dumps(row, separators=(',', ':')) + '\n')
            counts[name] += 1
        try:
            for source in sorted(window.glob('events.*.jsonl')):
                with source.open() as events:
                    for line in events:
                        row = json.loads(line)
                        kind = row['event']
                        base = {k: row[k] for k in ('context', 'chunk', 'rank', 'seq', 'mono_ns')}
                        if kind.startswith('request.'):
                            write('requests', {**base, 'kind': kind, **row['data']})
                        elif kind == 'step.inputs':
                            write('step_inputs', {**base, **row['data']})
                        elif kind == 'chunk.inputs':
                            write('chunk_inputs', {**base, **row['data']})
                        elif kind == 'chunk.layout':
                            write('layouts', {**base, **row['data']})
                        elif kind.startswith('engine.'):
                            write('engine_steps', {**base, 'kind': kind, 'data': row['data']})
                        elif kind.startswith('step.output_'):
                            write('step_timings', {**base, **row['data']})
                        elif kind in ('runner.main_forward.enter', 'runner.main_forward.exit'):
                            key = (tuple(row['context'] or ()), row['chunk'])
                            if kind.endswith('.enter'):
                                if len(self._open_spans) < self.capacity:
                                    self._open_spans[key] = (row['mono_ns'], row['seq'])
                            else:
                                start = self._open_spans.pop(key, None)
                                native_ok = (row.get('data') or {}).get('native_ok')
                                write('schedule_batch', {**base,
                                    'start_mono_ns': start[0] if start else None,
                                    'start_event_seq': start[1] if start else None,
                                    'end_mono_ns': row['mono_ns'],
                                    'iter_latency': (row['mono_ns'] - start[0]) / 1e9 if start else None,
                                    'time_kind': 'host_main_forward',
                                    'device_start_ns': None, 'device_end_ns': None,
                                    'native_ok': native_ok,
                                    'valid': start is not None and native_ok is not False})
        finally:
            for stream in streams.values():
                stream.close()
        return counts

    def _publish(self, job):
        self._stream.flush()
        self._stream.close()
        cut = {'schema': 3, 'export_id': job['id'], 'trigger': job['trigger'],
               'process_id': self.process_id, 'rank': dict(zip(RANK_FIELDS, self.identity)),
               'clock_domain': self._metadata()['clock_domain'],
               'host_clock': self.host_clock,
               'cut_seq': job['cut_seq'], 'cut_mono_ns': job['cut_mono_ns'],
               'time_kind': 'host_observation', 'device_time_available': False,
               'cross_process_cut_is_atomic': False, 'health': self.status()}
        cut['table_rows'] = self._tables(self._window)
        destination = self.directory / 'exports' / job['id']
        destination.parent.mkdir(exist_ok=True)
        (self._window / 'export.json').write_text(json.dumps(cut) + '\n')
        # A complete export directory appears atomically; never overwrite a cut.
        os.rename(self._window, destination)
        self.exported += 1
        job['result'] = {'export_id': job['id'], 'cut_seq': job['cut_seq'],
                         'complete': True, 'path': str(destination),
                         'process_id': self.process_id}
        job['done'].set()
        self._new_window()

    def _write(self):
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            self._new_window()
            self._save_metadata()
            while True:
                self._wake.wait(self.interval)
                self._wake.clear()
                with self._lock:
                    batch, self._events = self._events, deque()
                    cuts, self._cuts = self._cuts, deque()
                for stamp, seq, kind, context, chunk, identity, data in batch:
                    while cuts and seq > cuts[0]['cut_seq']:
                        self._publish(cuts.popleft())
                    row = {'schema': 2, 'mono_ns': stamp, 'seq': seq, 'event': kind,
                           'context': context, 'chunk': chunk,
                           'rank': dict(zip(RANK_FIELDS, identity)), 'data': data}
                    self._append(json.dumps(row, separators=(',', ':')) + '\n')
                while cuts:
                    self._publish(cuts.popleft())
                self._stream.flush()
                self._save_metadata()
                if self._stop.is_set():
                    break
            self._stream.close()
            self.closed = True
            self._save_metadata()
        except Exception as exc:
            self.failure = type(exc).__name__
            try:
                self._save_metadata()
            except Exception:
                pass
        finally:
            if self._stream is not None:
                self._stream.close()
            if self.failure is not None:
                with self._cut_lock:
                    for job in self._cut_jobs.values():
                        if not job['done'].is_set():
                            job['result'] = {'export_id': job['id'], 'complete': False,
                                             'failure': self.failure}
                            job['done'].set()

    def close(self):
        if os.getpid() != self.pid:
            return
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2)
