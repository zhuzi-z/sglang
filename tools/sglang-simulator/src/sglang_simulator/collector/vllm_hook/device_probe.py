"""Opt-in ONE inference-window diagnostic; never part of continuous labels.

Starts native torch.profiler before first real execute, stops only after the
native PP settle returns. No per-chunk sync or additional model execution.
Profile APIs retain their existing collection-export semantics.
"""
from contextlib import nullcontext, contextmanager
from functools import wraps
import hashlib
import gzip
import base64
import json
import os
from pathlib import Path
import time
from . import trace_v2 as trace

_PROBE = None


class DeviceProbe:
    def __init__(self, worker):
        self.worker = worker
        self.state = 'armed'
        self.profiler = None
        self.torch = None
        self.start_event = None
        self.directory = None
        self.report = {'contract': 1, 'diagnostic_only': True,
                       'device_timing_verified': False, 'errors': []}

    def save(self):
        if self.directory is None:
            return
        try:
            tmp = self.directory / 'probe.tmp'
            tmp.write_text(json.dumps({**self.report, 'state': self.state}, indent=2)+'\n')
            tmp.replace(self.directory / 'probe.json')
        except OSError:
            pass  # Diagnostic I/O must not replace serving results/exceptions.

    def start(self, scope):
        if self.state != 'armed':
            return
        self.state = 'starting'
        try:
            import torch
            from torch_tpu._internal.profiler import TpuProfilerConfig
            self.torch = torch
            rec = trace._RECORDER
            if rec is None:
                raise RuntimeError('collector_not_initialized')
            self.directory = rec.directory / 'device_probe'
            self.directory.mkdir(exist_ok=False)
            self.report.update(rank=dict(zip(trace.RANK_FIELDS, rec.identity)),
                               context=scope.context, process_id=rec.process_id,
                               build_id=rec.build_id, started_mono_ns=time.monotonic_ns(),
                               started_wall_ns=time.time_ns(),
                               torch_version=str(torch.__version__),
                               measurement='native_hardware_profiler_window')
            # Probe public Event API but do not insert an extra synchronization.
            try:
                self.start_event = torch.tpu.Event(enable_timing=True)
                self.start_event.record()
                self.report['event_api'] = {'type': type(self.start_event).__name__,
                                             'record_start_ok': True}
            except Exception as exc:
                self.report['event_api'] = {'error_type': type(exc).__name__, 'error': str(exc)[:500]}
            config = TpuProfilerConfig(run_dir=str(self.directory),
                         worker_rank=str(rec.identity[1]), host_tracer_level=2,
                         device_tracer_level=1, python_tracer_level=0)
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.PrivateUse1],
                experimental_config=config,
                on_trace_ready=lambda prof: prof.export_chrome_trace(str(self.directory/'trace.json')))
            self.profiler.__enter__()
            self.state = 'recording'
            self.report['profile_config'] = {'host_tracer_level':2,'device_tracer_level':1,'python_tracer_level':0}
            self.save()
        except Exception as exc:
            self.report['errors'].append({'at':'start','type':type(exc).__name__,'message':str(exc)[:1000]})
            self.state = 'failed'
            self.save()

    @contextmanager
    def annotation(self, name, scope):
        if self.state != 'recording' or self.torch is None:
            yield
            return
        c = scope.context
        label = 'sim_probe|step='+str(c[2])+'|chunk='+str(scope.chunk)+'|stage='+name
        context = None
        try:
            context = self.torch.profiler.record_function(label)
            context.__enter__()
        except Exception as exc:
            self.report['errors'].append({'at':'annotation_enter','type':type(exc).__name__})
            context = None
        try:
            yield
        finally:
            if context is not None:
                try:
                    context.__exit__(None,None,None)
                except Exception as exc:
                    self.report['errors'].append({'at':'annotation_exit','type':type(exc).__name__})

    def finish(self):
        if self.state != 'recording':
            return
        self.state = 'stopping'
        self.report['native_settle_return_mono_ns'] = time.monotonic_ns()
        try:
            if self.start_event is not None:
                try:
                    end = self.torch.tpu.Event(enable_timing=True)
                    end.record()
                    elapsed = self.start_event.elapsed_time(end)
                    self.report['event_api']['elapsed_ms'] = elapsed
                    self.report['event_api']['semantics_verified_as_device_time'] = False
                except Exception as exc:
                    self.report['event_api'].update(elapsed_error_type=type(exc).__name__,
                                                    elapsed_error=str(exc)[:500])
            self.profiler.__exit__(None,None,None)
            self.state = 'complete'
            self.report['finished_mono_ns'] = time.monotonic_ns()
            self.report['artifacts'] = []
            for p in sorted(self.directory.rglob('*')):
                if p.is_file() and p.name not in ('probe.json','probe.tmp'):
                    content = p.read_bytes()
                    artifact = {'path':str(p.relative_to(self.directory)),
                      'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
                    # Text sidecars permit ordinary allowlisted file reads.
                    # Never execute an interpreter through the ASI gateway.
                    if len(content) <= 64*1024*1024:
                        encoded = base64.b64encode(gzip.compress(content,compresslevel=1))
                        parts = []
                        if len(encoded) <= 32*1024*1024:
                            for offset in range(0,len(encoded),65536):
                                part = self.directory / (p.name+'.gz.b64.'+str(offset//65536).zfill(4))
                                part.write_bytes(encoded[offset:offset+65536]+b'\n')
                                parts.append({'path':part.name,'bytes':part.stat().st_size})
                            artifact['transport'] = {'encoding':'gzip+base64','parts':parts}
                    self.report['artifacts'].append(artifact)
        except Exception as exc:
            self.report['errors'].append({'at':'stop','type':type(exc).__name__,'message':str(exc)[:1000]})
            self.state = 'failed'
        finally:
            self.save()


def annotation(name, scope):
    if _PROBE is None or scope is None:
        return nullcontext()
    return _PROBE.annotation(name, scope)


def install(target):
    # Disabled means no wrappers, imports of torch, threads, or file writes.
    if os.getenv('SIM_TPU_DEVICE_PROBE') != '1':
        return
    native_execute = target.execute_model
    @wraps(native_execute)
    def execute(self, *args, **kwargs):
        global _PROBE
        scope = trace.ACTIVE.get()
        if scope is not None and scope.context[3] == 'real':
            if _PROBE is None:
                _PROBE = DeviceProbe(self)
            _PROBE.start(scope)
        return native_execute(self,*args,**kwargs)
    target.execute_model = execute
    native_settle = target.pp_settle
    @wraps(native_settle)
    def settle(self,*args,**kwargs):
        result = native_settle(self,*args,**kwargs)
        if _PROBE is not None:
            _PROBE.finish()
        return result
    target.pp_settle = settle
