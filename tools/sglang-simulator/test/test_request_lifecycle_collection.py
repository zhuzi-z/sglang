"""CPU-only lifecycle contracts: isolated importlib stubs, no vLLM/torch/pytest."""
import builtins
from contextvars import ContextVar
from contextlib import contextmanager
import json
from enum import IntEnum, Enum
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch


HOOK_DIR = Path(__file__).resolve().parents[1] / 'src/sglang_simulator/collector/vllm_hook'


class Reason(IntEnum):
    STOP = 0
    LENGTH = 1
    ABORT = 2


class Status(Enum):
    RUNNING = 1
    FINISHED_STOPPED = 2
    FINISHED_ABORTED = 3


class CountOnly:
    """Only length is legal; truth tests and reading token values are forbidden."""
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __bool__(self):
        raise AssertionError('raw token truth test')

    def __iter__(self):
        raise AssertionError('raw token iteration')

    def __getitem__(self, index):
        raise AssertionError('raw token value read')


def request(rid, prompt=24, emitted=0, external=None):
    return NS(request_id=rid, num_prompt_tokens=prompt, num_output_tokens=emitted,
              num_computed_tokens=9999, num_external_computed_tokens=external,
              max_tokens=8, sampling_params=None, arrival_time=1.0,
              status=Status.RUNNING, finish_reason=None)


def scheduler_output(order, new=(), cached=None):
    return NS(scheduled_new_reqs=list(new), scheduled_cached_reqs=cached,
              num_scheduled_tokens=dict(order), total_num_scheduled_tokens=sum(q for _, q in order))


def engine_output(rid, count, reason=None, **fields):
    return NS(request_id=rid, new_token_ids=CountOnly(count), finish_reason=reason,
              **fields)


class RequestLifecycleCollectionTest(unittest.TestCase):
    def setUp(self):
        self.package = '_request_lifecycle_collection_stub'
        pkg = ModuleType(self.package)
        pkg.__path__ = [str(HOOK_DIR)]
        root = ModuleType('sglang_simulator')
        root.__path__ = []
        base = ModuleType('sglang_simulator.hook')
        base.BaseHook = type('BaseHook', (), {})
        self.trace = ModuleType(self.package + '.trace_v2')
        self.trace.ACTIVE = ContextVar('lifecycle_scope', default=None)
        self.trace._RECORDER = NS(observer_errors=0, focus_stage=False, focus_pp0=False,
                                  identity=(0, 0, 0, 3, 0, 0, 0))
        self.events = []
        self.observer_failure = None

        def emit(kind, scope=None, chunk=None, data=None):
            if self.observer_failure is not None:
                raise self.observer_failure
            self.events.append({'event': kind, 'data': data,
                                'context': scope.context if scope is not None else None,
                                'observed_mono_ns': time.monotonic_ns()})
        self.trace.emit = emit
        self.trace.Scope = lambda context: NS(context=context, chunk=-1)
        timeline = ModuleType(self.package + '.trace_v2_hooks')
        for name in ('V2ExecutorHook', 'V2WorkerHook', 'V2RunnerTimingHook', 'V2PPWaveHook'):
            setattr(timeline, name, type(name, (base.BaseHook,), {}))
        timeline.HOOKS = [getattr(timeline, name) for name in (
            'V2ExecutorHook', 'V2WorkerHook', 'V2RunnerTimingHook', 'V2PPWaveHook')]
        host = ModuleType(self.package + '.host_breakdown')
        host.ENABLED = False
        modules = {'sglang_simulator': root, 'sglang_simulator.hook': base,
                   self.package: pkg, self.trace.__name__: self.trace,
                   timeline.__name__: timeline, host.__name__: host}
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        original_import = builtins.__import__
        self.forbidden_imports = []

        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'vllm', 'vllm_torchtpu'):
                self.forbidden_imports.append(name)
                raise AssertionError('device/runtime import: ' + name)
            return original_import(name, *args, **kwargs)
        guard = patch.object(builtins, '__import__', guarded_import)
        guard.start()
        self.addCleanup(guard.stop)
        self.hooks = self.load('trace_collection_hooks')
        self.stage = self.load('stage_collection')

    def load(self, name):
        fullname = self.package + '.' + name
        spec = importlib.util.spec_from_file_location(fullname, HOOK_DIR / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        # Keep these temporary modules inside patch.dict's restoration boundary.
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
        return module

    def rows(self, kind):
        return [e['data'] for e in self.events if e['event'] == kind]

    def digest(self, rid):
        return hashlib.sha256(rid.encode()).hexdigest()

    @contextmanager
    def advancing_clock(self):
        clock = NS(now=100)
        def advance(amount=100):
            clock.now += amount
        clock.advance = advance
        original_emit = self.trace.emit
        def emit(*args, **kwargs):
            advance(1000)
            return original_emit(*args, **kwargs)
        with patch.object(self.hooks.time, 'monotonic_ns', lambda: clock.now), \
                patch.object(self.trace, 'emit', emit):
            yield clock

    def advancing_context(self, context, clock):
        def reset(token):
            clock.advance()
            context.reset(token)
        return NS(get=context.get, set=context.set, reset=reset)

    def make_engine_core(self):
        class EngineCore:
            def __init__(self):
                self.native_calls = []
                self.native_result = None
                self.error = None

            def preprocess_add_request(self, request, *args, **kwargs):
                self.native_calls.append((request, args, kwargs))
                if self.error is not None:
                    raise self.error
                # Native conversion drops external_req_id from the returned Request.
                self.native_result = (NS(request_id=request.request_id), request.current_wave)
                return self.native_result

        self.hooks.CollectionEngineCoreIdentityHook.hook(EngineCore)
        return EngineCore()

    def test_identity_alias_target_registration_and_successful_native_conversion(self):
        hook = self.hooks.CollectionEngineCoreIdentityHook
        self.assertEqual(hook.HOOK_MODULE_NAME, 'vllm.v1.engine.core')
        self.assertEqual(hook.HOOK_CLASS_NAME, 'EngineCore')
        self.assertEqual(self.hooks.HOOKS.count(hook), 1)
        core = self.make_engine_core()
        owner = self

        class EngineCoreRequest:
            request_id = 'randomized-internal-id'
            current_wave = 7

            @property
            def external_req_id(self):
                owner.assertIsNotNone(core.native_result)  # observe only after native success
                return 'client-request-id'

        req = EngineCoreRequest()
        marker = object()
        before = time.time_ns()
        result = core.preprocess_add_request(req, marker, option=marker)
        after = time.time_ns()
        self.assertIs(result, core.native_result)
        self.assertEqual(core.native_calls, [(req, (marker,), {'option': marker})])
        self.assertEqual(result[0].request_id, req.request_id)
        self.assertFalse(hasattr(result[0], 'external_req_id'))
        self.assertEqual(result[1], 7)
        row = self.rows('request.identity_alias')[0]
        self.assertEqual(row, {
            'request_id_sha256': self.digest('randomized-internal-id'),
            'external_request_id_sha256': self.digest('client-request-id'),
            'identity_source': 'EngineCoreRequest.external_req_id',
            'start_mono_ns': row['start_mono_ns'], 'end_mono_ns': row['end_mono_ns'],
            'native_ok': True, 'source': 'EngineCore.preprocess_add_request',
            'time_kind': 'host_native_call', 'wall_ns': row['wall_ns']})
        self.assertLessEqual(row['start_mono_ns'], row['end_mono_ns'])
        self.assertLessEqual(before, row['wall_ns'])
        self.assertLessEqual(row['wall_ns'], after)
        self.assertEqual(self.trace._RECORDER.observer_errors, 0)
        self.assertEqual(self.forbidden_imports, [])

    def test_identity_alias_different_internal_ids_share_external_digest(self):
        core = self.make_engine_core()
        for rid in ('internal-a-random', 'internal-b-random'):
            req = NS(request_id=rid, external_req_id='same-client-id', current_wave=3)
            self.assertIs(core.preprocess_add_request(request=req), core.native_result)
            self.assertEqual(req.external_req_id, 'same-client-id')
        aliases = self.rows('request.identity_alias')
        self.assertEqual([r['request_id_sha256'] for r in aliases],
                         [self.digest('internal-a-random'), self.digest('internal-b-random')])
        self.assertEqual([r['external_request_id_sha256'] for r in aliases],
                         [self.digest('same-client-id')] * 2)

    def test_identity_alias_missing_none_skipped_but_empty_external_is_retained(self):
        core = self.make_engine_core()
        for fields in ({}, {'external_req_id': None}):
            req = NS(request_id='internal', current_wave=0, **fields)
            self.assertIs(core.preprocess_add_request(req), core.native_result)
        self.assertEqual(self.rows('request.identity_alias'), [])
        req = NS(request_id='internal', external_req_id='', current_wave=0)
        self.assertIs(core.preprocess_add_request(req), core.native_result)
        self.assertEqual(self.rows('request.identity_alias')[0]['external_request_id_sha256'],
                         self.digest(''))
        self.assertEqual(self.trace._RECORDER.observer_errors, 0)

    def test_identity_alias_native_exceptions_unchanged_without_observation(self):
        class UnobservableRequest:
            @property
            def external_req_id(self):
                raise AssertionError('must not observe failed native preprocessing')

        for error in (RuntimeError('native preprocess'), KeyboardInterrupt('native control flow')):
            with self.subTest(error=type(error).__name__):
                core = self.make_engine_core()
                core.error = error
                req = UnobservableRequest()
                with self.assertRaises(type(error)) as caught:
                    core.preprocess_add_request(req)
                self.assertIs(caught.exception, error)
                self.assertEqual(len(core.native_calls), 1)
                self.assertIsNone(core.native_result)
        self.assertEqual(self.rows('request.identity_alias'), [])
        self.assertEqual(self.trace._RECORDER.observer_errors, 0)

    def test_identity_alias_emit_and_counter_errors_preserve_native_result_and_exception(self):
        class BrokenCounter:
            @property
            def observer_errors(self):
                raise RuntimeError('counter unavailable')

        for broken_counter in (False, True):
            with self.subTest(broken_counter=broken_counter):
                self.trace._RECORDER = BrokenCounter() if broken_counter else NS(observer_errors=0)
                self.observer_failure = SystemExit('alias observer unavailable')
                core = self.make_engine_core()
                req = NS(request_id='internal', external_req_id='external', current_wave=1)
                self.assertIs(core.preprocess_add_request(req), core.native_result)
                self.assertEqual(len(core.native_calls), 1)
                if not broken_counter:
                    self.assertEqual(self.trace._RECORDER.observer_errors, 1)
                error = ValueError('native preprocess failure')
                core.error = error
                with self.assertRaises(ValueError) as caught:
                    core.preprocess_add_request(req)
                self.assertIs(caught.exception, error)
                self.assertEqual(len(core.native_calls), 2)
        self.assertEqual(self.rows('request.identity_alias'), [])

    def test_identity_alias_metadata_hash_and_clock_errors_are_observed(self):
        class BadMetadata:
            request_id = 'internal'
            current_wave = 1

            @property
            def external_req_id(self):
                raise ValueError('external metadata unavailable')

        core = self.make_engine_core()
        self.assertIs(core.preprocess_add_request(BadMetadata()), core.native_result)
        bad_hash = NS(request_id='internal', external_req_id=object(), current_wave=1)
        self.assertIs(core.preprocess_add_request(bad_hash), core.native_result)
        req = NS(request_id='internal', external_req_id='external', current_wave=1)
        with patch.object(self.hooks.time, 'time_ns', side_effect=RuntimeError('clock unavailable')):
            self.assertIs(core.preprocess_add_request(req), core.native_result)
        self.assertEqual(len(core.native_calls), 3)
        self.assertEqual(self.trace._RECORDER.observer_errors, 3)
        self.assertEqual(self.rows('request.identity_alias'), [])

    def test_lifecycle_rows_use_explicit_hashes_and_active_member_lookup(self):
        sched = self.make_scheduler()
        req = request('private-id')
        sched.add_request(req)
        sched.finished_recving_kv_req_ids = [req.request_id]
        sched.next_output = scheduler_output([(req.request_id, 1)], new=[
            NS(req_id=req.request_id, num_computed_tokens=0)])
        output = sched.schedule()
        req.finish_reason = Reason.STOP
        sched.update_free = [req]
        sched.outputs = {0: NS(outputs=[engine_output(req.request_id, 1, Reason.STOP)])}
        sched.update_from_output(output)
        for kind in ('request.add', 'request.scheduled', 'request.remote_kv_signal_observed',
                     'request.output_observed', 'request.free'):
            row = self.rows(kind)[0]
            self.assertEqual(row['request_id_sha256'], self.digest(req.request_id), kind)
            self.assertNotIn('req_id', row, kind)
        members = self.rows('request.step_membership')[0]['members']
        self.assertEqual(members[0]['request_id_sha256'], self.digest(req.request_id))
        self.assertNotIn('req_id', members[0])
        for kind in ('request.output_observed', 'request.free'):
            row = self.rows(kind)[0]
            self.assertEqual(row['request_lifecycle_id'], members[0]['request_lifecycle_id'])
            self.assertEqual(row['request_iteration'], 1)
        snapshot = self.hooks.step_snapshot(output)
        self.assertEqual(snapshot['requests'][0]['req_id'], req.request_id)
        self.assertNotIn('request_id_sha256', snapshot['requests'][0])
        self.assertEqual(snapshot['members'], members)

    def make_scheduler(self, subclass=False):
        owner = self

        class Scheduler:
            def __init__(self):
                self.requests = {}
                self.native_calls = []
                self.errors = {}
                self.add_result = object()
                self.free_result = object()
                self.next_output = scheduler_output([])
                self.outputs = {}
                self.update_free = []
                self.update_action = None
                self.update_arguments = []
                self.native_update_members = []
                self.finished_recving_kv_req_ids = []
                self.native_schedule_bounds = []

            def native(self, method):
                self.native_calls.append((method, getattr(self, '_sim_collection_active_schedule_id', None)))
                if method in self.errors:
                    raise self.errors[method]

            def add_request(self, req, *args, **kwargs):
                self.native('add')
                self.requests[req.request_id] = req
                return self.add_result

            def _free_request(self, req, *args, **kwargs):
                self.native('free')
                req.status = Status.FINISHED_ABORTED if req.finish_reason == Reason.ABORT else Status.FINISHED_STOPPED
                self.requests.pop(req.request_id, None)
                return self.free_result

            def schedule(self, *args, **kwargs):
                entered = time.monotonic_ns()
                self.native('schedule')
                self.native_schedule_bounds.append((entered, time.monotonic_ns()))
                return self.next_output

            def update_from_output(self, scheduler_output, model_runner_output=None, **kwargs):
                self.update_arguments.append((scheduler_output, model_runner_output, kwargs))
                self.native_update_members.append(getattr(self, '_sim_collection_active_members', None))
                if self.update_action is not None:
                    self.update_action(scheduler_output)
                self.native('update')
                # Native accepting output may free requests before it returns.
                for req in self.update_free:
                    self._free_request(req)
                return self.outputs

        owner.hooks.CollectionSchedulerHook.hook(Scheduler)
        if subclass:
            class TpuScheduler(Scheduler):
                def schedule(self, *args, **kwargs):
                    self.subclass_enter = time.monotonic_ns()
                    result = super().schedule(*args, **kwargs)
                    self.subclass_return = time.monotonic_ns()
                    return result
            owner.hooks.CollectionTpuSchedulerHook.hook(TpuScheduler)
            return TpuScheduler()
        return Scheduler()

    def test_output_update_nonempty_without_outputs_snapshots_only_cpu_keys_and_total(self):
        sched = self.make_scheduler()
        owner = self
        reads = []

        class Opaque:
            def __getattribute__(self, name):
                raise AssertionError('payload/device metadata read')
            def __bool__(self):
                raise AssertionError('payload truth test')
            def __len__(self):
                raise AssertionError('payload length')
            def __iter__(self):
                raise AssertionError('payload iteration')
            def __int__(self):
                raise AssertionError('payload scalar conversion')

        with self.advancing_clock() as clock:
            class KeysOnly(dict):
                def keys(self):
                    owner.assertEqual(sched.update_arguments, [])
                    reads.append('keys')
                    clock.advance()
                    return super().keys()
                def items(self):
                    raise AssertionError('scheduled token value read')
                def values(self):
                    raise AssertionError('scheduled token value read')
                def __getitem__(self, key):
                    raise AssertionError('scheduled token value read')

            class Output:
                tokens = KeysOnly({'private-b': Opaque(), 'private-a': Opaque()})
                total = 9
                @property
                def num_scheduled_tokens(self):
                    owner.assertEqual(sched.update_arguments, [])
                    clock.advance()
                    return self.tokens
                @property
                def total_num_scheduled_tokens(self):
                    owner.assertEqual(sched.update_arguments, [])
                    reads.append('total')
                    clock.advance()
                    return self.total

            output, payload = Output(), Opaque()
            setattr(output, self.hooks.SCHEDULE_KEY, 73)
            # Batch IDs must come from actual scheduled keys, not stale members.
            setattr(output, self.hooks.MEMBERS_KEY, [{'request_id_sha256': self.digest('stale')}])
            bounds = []
            def native_action(value):
                self.assertIs(value, output)
                bounds.append(clock.now)
                clock.advance(10)
                bounds.append(clock.now)
                output.tokens.clear()
                output.total = 0
                setattr(output, self.hooks.SCHEDULE_KEY, 999)
            sched.update_action = native_action
            self.assertIs(sched.update_from_output(output, payload, option=payload), sched.outputs)
            self.assertEqual(reads, ['keys', 'total'])
            row = self.rows('scheduler.output_update')[0]
            self.assertEqual((row['start_mono_ns'], row['end_mono_ns']), tuple(bounds))
            self.assertEqual(row['end_mono_ns'] - row['start_mono_ns'], 10)
            self.assertGreater(clock.now, row['end_mono_ns'])
        self.assertEqual(len(self.rows('scheduler.output_update')), 1)
        self.assertEqual(row['request_ids_sha256'], (self.digest('private-b'), self.digest('private-a')))
        self.assertEqual(row['request_count'], 2)
        self.assertEqual(row['total_num_scheduled_tokens'], 9)
        self.assertEqual(row['schedule_id'], 73)
        self.assertEqual(row['scheduled_token_batch_state'], 'nonempty')
        self.assertTrue(row['num_scheduled_tokens_keys_available'])
        self.assertTrue(row['total_num_scheduled_tokens_available'])
        self.assertTrue(row['native_ok'])
        self.assertEqual(row['source'], 'Scheduler.update_from_output')
        self.assertEqual(row['identity_source'], 'SchedulerOutput.num_scheduled_tokens_before_update')
        self.assertEqual(row['time_kind'], 'host_native_call')
        self.assertEqual(row['timing_scope'], 'batch_shared')
        self.assertTrue(row['shared_batch_span'])
        self.assertFalse(row['per_request_apportionment'])
        self.assertFalse(row['device_completion_confirmed'])
        self.assertFalse(row['valid_pure_device_timing'])
        self.assertEqual(sched.update_arguments, [(output, payload, {'option': payload})])
        self.assertEqual(sched.native_calls, [('update', 73)])
        self.assertIsNone(sched._sim_collection_active_schedule_id)
        self.assertIsNone(sched._sim_collection_active_members)
        self.assertEqual(self.rows('request.output_observed'), [])
        self.assertNotIn('private-', repr(self.events))
        self.assertNotIn(self.digest('stale'), row['request_ids_sha256'])
        self.assertEqual(self.trace._RECORDER.observer_errors, 0)
        self.assertEqual(self.forbidden_imports, [])

    def test_output_update_native_failure_emits_immediate_span_and_preserves_error_identity(self):
        for error in (RuntimeError('native update'), KeyboardInterrupt('native update')):
            for empty in (False, True):
                with self.subTest(error=type(error).__name__, empty=empty), self.advancing_clock() as clock:
                    sched = self.make_scheduler()
                    output = scheduler_output([] if empty else [('r', 3)])
                    setattr(output, self.hooks.SCHEDULE_KEY, 91)
                    sched.errors['update'] = error
                    bounds = []
                    def native_action(value):
                        bounds.append(clock.now)
                        clock.advance(10)
                        bounds.append(clock.now)
                    sched.update_action = native_action
                    self.events.clear()
                    with self.assertRaises(type(error)) as caught:
                        sched.update_from_output(scheduler_output=output, model_runner_output=object())
                    self.assertIs(caught.exception, error)
                    self.assertEqual(sched.native_calls, [('update', 91)])
                    self.assertEqual(len(sched.update_arguments), 1)
                    rows = self.rows('scheduler.output_update')
                    self.assertEqual(len(rows), 1)
                    row = rows[0]
                    self.assertEqual((row['start_mono_ns'], row['end_mono_ns']), tuple(bounds))
                    self.assertFalse(row['native_ok'])
                    self.assertEqual(row['schedule_id'], 91)
                    self.assertEqual(row['request_ids_sha256'], () if empty else (self.digest('r'),))
                    self.assertEqual(row['scheduled_token_batch_state'], 'empty' if empty else 'nonempty')
                    self.assertEqual(self.rows('request.output_observed'), [])
                    self.assertGreater(clock.now, row['end_mono_ns'])
                    self.assertIsNone(sched._sim_collection_active_schedule_id)
                    self.assertIsNone(sched._sim_collection_active_members)

    def test_output_update_batch_observer_failure_does_not_retry_native_or_drop_item_cleanup(self):
        sched = self.make_scheduler()
        req = request('r')
        sched.add_request(req)
        sched.next_output = scheduler_output([('r', 1)], new=[NS(req_id='r', num_computed_tokens=0)])
        output = sched.schedule()
        member = getattr(output, self.hooks.MEMBERS_KEY)[0]
        sched.outputs = {0: NS(outputs=[engine_output('r', 0, Reason.STOP)])}
        original_emit = self.trace.emit
        attempts = []
        def emit(kind, *args, **kwargs):
            if kind == 'scheduler.output_update':
                attempts.append(kind)
                self.assertEqual(sched._sim_collection_active_schedule_id, 1)
                self.assertEqual(sched._sim_collection_active_members, {self.digest('r'): member})
                raise SystemExit('batch observer failure')
            return original_emit(kind, *args, **kwargs)
        with patch.object(self.trace, 'emit', emit):
            self.assertIs(sched.update_from_output(output), sched.outputs)
            error = ValueError('native update failure')
            sched.errors['update'] = error
            with self.assertRaises(ValueError) as caught:
                sched.update_from_output(output)
            self.assertIs(caught.exception, error)
        self.assertEqual(attempts, ['scheduler.output_update'] * 2)
        self.assertEqual([c for c in sched.native_calls if c[0] == 'update'], [('update', 1)] * 2)
        self.assertEqual(len(sched.update_arguments), 2)
        self.assertEqual(self.trace._RECORDER.observer_errors, 2)
        row = self.rows('request.output_observed')[0]
        self.assertEqual(row['schedule_id'], 1)
        self.assertEqual(row['request_lifecycle_id'], member['request_lifecycle_id'])
        self.assertEqual(sched._sim_collection_request_iterations, {})
        self.assertIsNone(sched._sim_collection_active_schedule_id)
        self.assertIsNone(sched._sim_collection_active_members)

    def test_output_update_retains_active_members_through_free_and_item_observation(self):
        sched = self.make_scheduler()
        req = request('r', emitted=2)
        sched.add_request(req)
        sched.next_output = scheduler_output([('r', 1)], new=[NS(req_id='r', num_computed_tokens=0)])
        output = sched.schedule()
        member = getattr(output, self.hooks.MEMBERS_KEY)[0]
        req.finish_reason = Reason.STOP
        sched.update_free = [req]
        sched.outputs = {0: NS(outputs=[engine_output('r', 1, Reason.STOP)])}
        original_emit = self.trace.emit
        observed = []
        def emit(kind, *args, **kwargs):
            if kind in ('request.free', 'scheduler.output_update', 'request.output_observed'):
                observed.append(kind)
                self.assertEqual(sched._sim_collection_active_schedule_id, 1)
                self.assertIs(sched._sim_collection_active_members[self.digest('r')], member)
            return original_emit(kind, *args, **kwargs)
        with patch.object(self.trace, 'emit', emit):
            self.assertIs(sched.update_from_output(output), sched.outputs)
        self.assertEqual(observed, ['request.free', 'scheduler.output_update', 'request.output_observed'])
        self.assertIs(sched.native_update_members[0][self.digest('r')], member)
        batch = self.rows('scheduler.output_update')[0]
        item = self.rows('request.output_observed')[0]
        self.assertEqual((batch['start_mono_ns'], batch['end_mono_ns']),
                         (item['start_mono_ns'], item['end_mono_ns']))
        self.assertEqual(item['request_id_sha256'], self.digest('r'))
        self.assertEqual(item['request_iteration'], 1)
        self.assertTrue(item['shared_batch_span'])
        self.assertEqual(item['time_kind'], 'host_output_accepted_observed')
        self.assertEqual(sched.native_calls[-2:], [('update', 1), ('free', 1)])
        self.assertEqual(self.trace._RECORDER.observer_errors, 0)
        self.assertEqual(sched._sim_collection_request_iterations, {})
        self.assertIsNone(sched._sim_collection_active_schedule_id)
        self.assertIsNone(sched._sim_collection_active_members)

    def test_output_update_known_empty_success_suppresses_batch_only(self):
        sched = self.make_scheduler()
        output = scheduler_output([])
        setattr(output, self.hooks.SCHEDULE_KEY, 5)
        # Existing logical completion observations must not change for empty polls.
        sched.outputs = {0: NS(outputs=[engine_output('not-scheduled', 0, Reason.ABORT)])}
        for _ in range(3):
            self.assertIs(sched.update_from_output(output), sched.outputs)
        self.assertEqual(self.rows('scheduler.output_update'), [])
        self.assertEqual(len(self.rows('request.output_observed')), 3)
        self.assertEqual(sched.native_calls, [('update', 5)] * 3)
        self.assertIsNone(sched._sim_collection_active_schedule_id)
        self.assertIsNone(sched._sim_collection_active_members)

    def test_output_update_unknown_or_inconsistent_contract_never_claims_empty(self):
        opaque_reads = []
        class Opaque:
            def __getattribute__(self, name):
                opaque_reads.append(name)
                raise AssertionError('unverified tensor-like metadata read')
            def __int__(self):
                opaque_reads.append('int')
                raise AssertionError('unverified scalar conversion')
        cases = (
            (NS(), None, None),
            (NS(num_scheduled_tokens={}), 0, None),
            (NS(total_num_scheduled_tokens=0), None, 0),
            (NS(num_scheduled_tokens=None, total_num_scheduled_tokens=0), None, 0),
            (NS(num_scheduled_tokens={'r': 1}), 1, None),
            (NS(num_scheduled_tokens={'r': 1}, total_num_scheduled_tokens=0), 1, 0),
            (NS(num_scheduled_tokens={}, total_num_scheduled_tokens=3), 0, 3),
            (NS(num_scheduled_tokens={}, total_num_scheduled_tokens=None), 0, None),
            (NS(num_scheduled_tokens={}, total_num_scheduled_tokens=False), 0, None),
            (NS(num_scheduled_tokens={}, total_num_scheduled_tokens=-1), 0, None),
            (NS(num_scheduled_tokens={}, total_num_scheduled_tokens='0'), 0, None),
            (NS(num_scheduled_tokens=Opaque(), total_num_scheduled_tokens=Opaque()), None, None),
        )
        for output, count, total in cases:
            with self.subTest(fields=vars(output)):
                sched = self.make_scheduler()
                setattr(output, self.hooks.SCHEDULE_KEY, 67)
                self.events.clear()
                self.assertIs(sched.update_from_output(output), sched.outputs)
                rows = self.rows('scheduler.output_update')
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual(row['scheduled_token_batch_state'], 'unknown')
                self.assertEqual(row['request_count'], count)
                self.assertEqual(row['total_num_scheduled_tokens'], total)
                self.assertEqual(row['num_scheduled_tokens_keys_available'], count is not None)
                self.assertEqual(row['total_num_scheduled_tokens_available'], total is not None)
                self.assertEqual(row['request_ids_sha256'],
                                 None if count is None else () if count == 0 else (self.digest('r'),))
                self.assertEqual(row['schedule_id'], 67)
                self.assertTrue(row['native_ok'])
                self.assertEqual(sched.native_calls, [('update', 67)])
                self.assertIsNone(sched._sim_collection_active_schedule_id)
                self.assertIsNone(sched._sim_collection_active_members)
        self.assertEqual(self.trace._RECORDER.observer_errors, 0)
        self.assertEqual(opaque_reads, [])

    def test_output_update_metadata_and_hash_failure_still_emits_unknown_evidence(self):
        class BrokenOutput:
            @property
            def num_scheduled_tokens(self):
                raise SystemExit('CPU metadata unavailable')
            @property
            def total_num_scheduled_tokens(self):
                raise ValueError('CPU total unavailable')
        for output in (BrokenOutput(), NS(num_scheduled_tokens={object(): 1}, total_num_scheduled_tokens=1)):
            sched = self.make_scheduler()
            self.events.clear()
            self.assertIs(sched.update_from_output(output), sched.outputs)
            row = self.rows('scheduler.output_update')[0]
            self.assertIsNone(row['request_ids_sha256'])
            self.assertIsNone(row['schedule_id'])
            self.assertTrue(row['native_ok'])
            self.assertEqual(sched.native_calls, [('update', None)])
            self.assertIsNone(sched._sim_collection_active_schedule_id)
            self.assertIsNone(sched._sim_collection_active_members)
        self.assertEqual(self.trace._RECORDER.observer_errors, 3)

    def test_output_update_end_clock_failure_preserves_result_error_and_single_call(self):
        for error in (None, ValueError('native'), KeyboardInterrupt('native')):
            with self.subTest(error=error):
                sched = self.make_scheduler()
                if error is not None:
                    sched.errors['update'] = error
                reads = []
                def clock():
                    reads.append(True)
                    if len(reads) == 1:
                        return 100
                    raise SystemExit('end clock failure')
                def emit(kind, scope=None, chunk=None, data=None):
                    self.events.append({'event': kind, 'data': data})
                self.events.clear()
                with patch.object(self.hooks.time, 'monotonic_ns', clock), \
                        patch.object(self.trace, 'emit', emit):
                    if error is None:
                        self.assertIs(sched.update_from_output(scheduler_output([('r', 1)])), sched.outputs)
                    else:
                        with self.assertRaises(type(error)) as caught:
                            sched.update_from_output(scheduler_output([('r', 1)]))
                        self.assertIs(caught.exception, error)
                row = self.rows('scheduler.output_update')[0]
                self.assertEqual(row['start_mono_ns'], 100)
                self.assertIsNone(row['end_mono_ns'])
                self.assertEqual(row['native_ok'], error is None)
                self.assertEqual(len(reads), 2)
                self.assertEqual(sched.native_calls, [('update', None)])
                self.assertIsNone(sched._sim_collection_active_schedule_id)
                self.assertIsNone(sched._sim_collection_active_members)

    def test_chunked_new_then_cached_preserves_order_and_pre_step_metadata(self):
        sched = self.make_scheduler(subclass=True)
        a, b = request('a', external=2), request('b', prompt=32, emitted=3)
        self.assertIs(sched.add_request(a), sched.add_result)
        sched.add_request(b)
        first = scheduler_output([('b', 4), ('a', 8)], new=[
            NS(req_id='a', num_computed_tokens=6), NS(req_id='b', num_computed_tokens=11)])
        sched.next_output = first
        self.assertIs(sched.schedule(throttle=True), first)
        members = first._sim_collection_step_members
        self.assertEqual([m['request_id_sha256'] for m in members], [self.digest('b'), self.digest('a')])
        self.assertEqual([m['past_kv_length'] for m in members], [11, 6])
        self.assertEqual([m['query_length'] for m in members], [4, 8])
        self.assertEqual([m['input_length'] for m in members], [32, 24])
        self.assertEqual([m['output_tokens_before'] for m in members], [3, 0])
        self.assertEqual([m['request_iteration'] for m in members], [1, 1])
        self.assertTrue(members[1]['cache_split_valid'])
        self.assertEqual(members[1]['local_kv_hit_len'], 4)
        self.assertIsNone(members[0]['ext_kv_hit_len'])
        self.assertFalse(members[0]['cache_split_valid'])
        self.assertTrue(all(m['new_request'] and not m['resumed'] for m in members))
        cached = NS(req_ids=['b', 'a'], num_computed_tokens=[15, 14], resumed_req_ids={'a'})
        second = scheduler_output([('a', 10), ('b', 1)], cached=cached)
        sched.next_output = second
        sched.schedule()
        next_members = second._sim_collection_step_members
        self.assertEqual([m['request_id_sha256'] for m in next_members], [self.digest('a'), self.digest('b')])
        self.assertEqual([m['past_kv_length'] for m in next_members], [14, 15])
        self.assertEqual([m['request_iteration'] for m in next_members], [2, 2])
        self.assertEqual([m['resumed'] for m in next_members], [True, False])
        self.assertTrue(second._sim_collection_membership_complete)
        self.assertEqual(first._sim_collection_schedule_id, 1)
        self.assertEqual(second._sim_collection_schedule_id, 2)
        self.assertEqual(len(self.rows('request.step_membership')), 2)
        self.assertEqual(len(self.rows('request.scheduled')), 2)  # no base/subclass duplicates
        first.scheduled_new_reqs[0].num_computed_tokens = 888
        cached.num_computed_tokens[0] = 999
        self.assertEqual(members[1]['past_kv_length'], 6)
        self.assertEqual(next_members[1]['past_kv_length'], 15)
        row = self.rows('request.step_membership')[-1]
        self.assertLessEqual(row['start_mono_ns'], sched.subclass_enter)
        self.assertGreaterEqual(row['end_mono_ns'], sched.subclass_return)
        self.assertEqual(row['time_kind'], 'host_scheduler_call')
        self.assertFalse(row['pp_dispatch_start'])
        self.assertNotIn('context', row)

    def test_complete_members_above_64_and_original_step_inputs_unchanged(self):
        sched = self.make_scheduler()
        order = []
        new = []
        for i in range(130):
            rid = 'private-id-' + str(i)
            sched.add_request(request(rid))
            order.append((rid, 1))
            new.append(NS(req_id=rid, num_computed_tokens=i))
        output = scheduler_output(list(reversed(order)), new=new)
        sched.next_output = output
        sched.schedule()
        snapshot = self.hooks.step_snapshot(output)
        self.assertEqual(snapshot['schedule_id'], 1)
        self.assertEqual(len(snapshot['members']), 130)
        self.assertEqual([m['request_id_sha256'] for m in snapshot['members']],
                         [self.digest(rid) for rid, _ in reversed(order)])
        self.assertEqual(snapshot['requests'][0]['req_id'], 'private-id-129')
        self.assertEqual(snapshot['requests'][0]['past_length'], 129)
        self.assertTrue(snapshot['query_sum_valid'])
        self.assertTrue(snapshot['past_complete'])
        self.assertTrue(snapshot['membership_complete'])
        context = ('collection', 'engine', 77, 'real')

        class Executor:
            _sim_v2_last = context
            def collective_rpc(self, method, *args, **kwargs):
                return output
        self.hooks.CollectionExecutorHook.hook(Executor)
        self.assertIs(Executor().collective_rpc('execute_model', args=(output,)), output)
        event = next(e for e in self.events if e['event'] == 'step.inputs')
        self.assertEqual(event['context'], context)
        self.assertEqual(event['data']['members'], snapshot['members'])
        self.assertTrue(event['data']['native_dispatch_ok'])

    def test_unknown_past_cache_and_resume_never_assume_zero(self):
        sched = self.make_scheduler()
        sched.add_request(request('new'))
        sched.add_request(request('cached'))
        output = scheduler_output([('cached', 1), ('new', 3), ('missing', 1)],
                                  new=[NS(req_id='new')],
                                  cached=NS(req_ids=['cached'], resumed_req_ids=set()))
        sched.next_output = output
        sched.schedule()
        members = output._sim_collection_step_members
        self.assertTrue(all(m['past_kv_length'] is None for m in members))
        self.assertIsNone(members[1]['total_kv_hit_len'])
        self.assertIsNone(members[1]['ext_kv_hit_len'])
        self.assertIsNone(members[1]['local_kv_hit_len'])
        self.assertFalse(members[1]['cache_split_valid'])
        self.assertIsNone(members[2]['input_length'])
        self.assertIsNone(members[2]['request_iteration'])
        self.assertIsNone(members[2]['resumed'])
        self.assertFalse(output._sim_collection_membership_complete)
        self.assertFalse(self.hooks.step_snapshot(output)['past_complete'])

    def test_cache_split_rejects_inconsistent_external_metadata(self):
        for total, external in ((2, 3), (2, -1), (None, 0)):
            with self.subTest(total=total, external=external):
                split = self.hooks.cache_split(NS(num_computed_tokens=total),
                                               request('r', external=external))
                self.assertFalse(split['cache_split_valid'])
                self.assertIsNone(split['local_kv_hit_len'])
        split = self.hooks.cache_split(NS(num_computed_tokens=0, num_external_computed_tokens=0), None)
        self.assertTrue(split['cache_split_valid'])
        self.assertEqual(split['local_kv_hit_len'], 0)

    def test_legacy_cached_resume_flags_and_count_only_prompt_metadata(self):
        sched = self.make_scheduler()
        output = scheduler_output([('new', 2), ('cached', 1)],
                                  new=[NS(req_id='new', num_computed_tokens=0,
                                          prompt_token_ids=CountOnly(9), num_output_tokens=0)],
                                  cached=NS(req_ids=['cached'], num_computed_tokens=[4],
                                            num_output_tokens=[3], resumed_from_preemption=[True]))
        sched.next_output = output
        sched.schedule()
        members = output._sim_collection_step_members
        self.assertEqual(members[0]['input_length'], 9)
        self.assertEqual(members[0]['output_tokens_before'], 0)
        self.assertTrue(members[1]['resumed'])
        self.assertEqual(members[1]['output_tokens_before'], 3)
        self.assertIsNone(members[1]['request_iteration'])

    def test_stop_zero_and_zero_token_abort_observed_after_native_success(self):
        sched = self.make_scheduler()
        stop, abort, live = request('stop', emitted=7), request('abort'), request('live')
        for req in (stop, abort, live):
            sched.add_request(req)
        output = scheduler_output([('stop', 1), ('abort', 1), ('live', 1)],
                                  new=[NS(req_id=r.request_id, num_computed_tokens=2)
                                       for r in (stop, abort, live)])
        sched.next_output = output
        sched.schedule()
        stop.finish_reason, abort.finish_reason = Reason.STOP, Reason.ABORT
        sched.update_free = [stop, abort]
        outputs = {0: NS(outputs=[engine_output('stop', 1, Reason.STOP, num_cached_tokens=2,
                                               spec_decode_num_draft_tokens=4,
                                               spec_decode_num_accepted_tokens=0),
                                  engine_output('abort', 0, Reason.ABORT, num_cached_tokens=0),
                                  engine_output('live', 0)])}
        sched.outputs = outputs
        before = time.time_ns()
        self.assertIs(sched.update_from_output(scheduler_output=output, model_runner_output=object()), outputs)
        after = time.time_ns()
        observed = self.rows('request.output_observed')
        self.assertEqual([r['new_token_count'] for r in observed], [1, 0, 0])
        self.assertEqual([r['finish_reason'] for r in observed], ['STOP', 'ABORT', None])
        self.assertEqual([r['finished'] for r in observed], [True, True, False])
        self.assertEqual([r['num_cached_tokens'] for r in observed], [2, 0, None])
        self.assertEqual(observed[0]['spec_decode_num_draft_tokens'], 4)
        self.assertEqual(observed[0]['spec_decode_num_accepted_tokens'], 0)
        self.assertTrue(all(r['schedule_id'] == 1 for r in observed))
        free = self.rows('request.free')
        self.assertEqual([r['status'] for r in free], ['FINISHED_STOPPED', 'FINISHED_ABORTED'])
        self.assertEqual([r['finish_reason'] for r in free], ['STOP', 'ABORT'])
        self.assertEqual([r['num_output_tokens'] for r in free], [7, 0])
        for row in free + observed:
            self.assertLessEqual(before, row['wall_ns'])
            self.assertLessEqual(row['wall_ns'], after)
            self.assertEqual(row['schedule_id'], 1)
            self.assertEqual(row['request_iteration'], 1)
        self.assertTrue(all(r['logical_finished'] and r['kv_release_confirmed'] is False for r in free))
        self.assertEqual(sched.native_calls[-3:], [('update', 1), ('free', 1), ('free', 1)])
        self.assertIsNone(sched._sim_collection_active_schedule_id)
        self.assertIsNone(sched._sim_collection_active_members)
        self.assertEqual(set(sched._sim_collection_request_iterations), {'live'})
        self.assertIs(sched._free_request(live), sched.free_result)
        self.assertIsNone(self.rows('request.free')[-1]['schedule_id'])
        self.assertEqual(sched._sim_collection_request_iterations, {})

    def test_finish_cleans_iteration_without_native_free_and_spec_aliases(self):
        sched = self.make_scheduler()
        sched.add_request(request('r'))
        output = scheduler_output([('r', 1)], new=[NS(req_id='r', num_computed_tokens=0)])
        sched.next_output = output
        sched.schedule()
        sched.outputs = {0: NS(outputs=[engine_output('r', 0, Reason.STOP,
                                                     num_spec_tokens=3, num_accepted_tokens=2)])}
        sched.update_from_output(output)
        self.assertEqual(sched._sim_collection_request_iterations, {})
        self.assertEqual(self.rows('request.output_observed')[0]['spec_decode_num_draft_tokens'], 3)
        self.assertEqual(self.rows('request.output_observed')[0]['spec_decode_num_accepted_tokens'], 2)
        self.assertEqual(self.rows('request.free'), [])

    def test_free_reason_from_native_getter_and_post_native_count(self):
        sched = self.make_scheduler()
        req = request('r')
        req.get_finished_reason = lambda: Reason.STOP
        del req.finish_reason
        sched.add_request(req)
        native = type(sched)._free_request.__wrapped__
        # Simulate native completion changing CPU counts before free returns.
        def changed(self, req, *args, **kwargs):
            req.num_output_tokens = 6
            req.finish_reason = Reason.STOP
            result = native(self, req, *args, **kwargs)
            del req.finish_reason
            return result
        type(sched)._free_request = changed
        self.hooks.CollectionSchedulerHook.hook(type(sched))
        self.assertIs(sched._free_request(req), sched.free_result)
        row = self.rows('request.free')[-1]
        self.assertEqual(row['num_output_tokens'], 6)
        self.assertEqual(row['finish_reason'], 'STOP')
        self.assertEqual(row['status'], 'FINISHED_STOPPED')

    def test_native_errors_unchanged_and_no_false_completion_or_stale_context(self):
        for method, invoke, absent in (
                ('add', lambda s, r, o: s.add_request(r), 'request.step_membership'),
                ('schedule', lambda s, r, o: s.schedule(), 'request.step_membership'),
                ('update', lambda s, r, o: s.update_from_output(o), 'request.output_observed'),
                ('free', lambda s, r, o: s._free_request(r), 'request.free')):
            with self.subTest(method=method):
                sched = self.make_scheduler()
                req = request('r')
                sched.add_request(req)
                output = scheduler_output([('r', 1)], new=[NS(req_id='r', num_computed_tokens=0)])
                output._sim_collection_schedule_id = 73
                error = RuntimeError('native-' + method)
                sched.errors[method] = error
                self.events.clear()
                with self.assertRaises(RuntimeError) as caught:
                    invoke(sched, req, output)
                self.assertIs(caught.exception, error)
                self.assertEqual(self.rows(absent), [])
                self.assertIsNone(getattr(sched, '_sim_collection_active_schedule_id', None))
                self.assertFalse(getattr(sched, '_sim_collection_in_schedule', False))
                self.assertIn('r', sched._sim_collection_request_iterations)
        sched = self.make_scheduler()
        error = KeyboardInterrupt('native control flow')
        sched.errors['update'] = error
        with self.assertRaises(KeyboardInterrupt) as caught:
            sched.update_from_output(NS(_sim_collection_schedule_id=9))
        self.assertIs(caught.exception, error)
        self.assertIsNone(sched._sim_collection_active_schedule_id)

    def test_schedule_ids_are_instance_local_and_success_only(self):
        sched = self.make_scheduler()
        sched.schedule()
        sched.errors['schedule'] = RuntimeError('fail')
        with self.assertRaises(RuntimeError):
            sched.schedule()
        del sched.errors['schedule']
        output = scheduler_output([])
        sched.next_output = output
        sched.schedule()
        self.assertEqual(output._sim_collection_schedule_id, 2)
        other = self.make_scheduler()
        self.assertEqual(other.schedule()._sim_collection_schedule_id, 1)
        self.assertIsNone(getattr(sched, '_sim_collection_active_schedule_id', None))

    def test_observer_failures_including_counter_do_not_change_native_calls(self):
        class BrokenCounter:
            @property
            def observer_errors(self):
                raise RuntimeError('counter unavailable')
        self.trace._RECORDER = BrokenCounter()
        self.observer_failure = RuntimeError('emit unavailable')
        sched = self.make_scheduler()
        req = request('r')
        self.assertIs(sched.add_request(req), sched.add_result)
        output = scheduler_output([('r', 1)], new=[NS(req_id='r', num_computed_tokens=0)])
        sched.next_output = output
        self.assertIs(sched.schedule(), output)
        sched.outputs = {0: NS(outputs=[engine_output('r', 0, Reason.STOP)])}
        self.assertIs(sched.update_from_output(output), sched.outputs)
        self.assertEqual(sched._sim_collection_request_iterations, {})
        self.assertIs(sched._free_request(req), sched.free_result)
        native_error = ValueError('native failure')
        sched.errors['schedule'] = native_error
        with self.assertRaises(ValueError) as caught:
            sched.schedule()
        self.assertIs(caught.exception, native_error)
        self.assertIsNone(self.hooks.observe(lambda: (_ for _ in ()).throw(SystemExit('observer'))))
        self.assertIsNone(sched._sim_collection_active_schedule_id)

    def test_unassignable_scheduler_output_does_not_change_native_result(self):
        class Output:
            __slots__ = ()
            scheduled_new_reqs = ()
            scheduled_cached_reqs = None
            num_scheduled_tokens = {'r': 1}
        sched = self.make_scheduler()
        output = Output()
        sched.next_output = output
        self.assertIs(sched.schedule(), output)
        self.assertGreater(self.trace._RECORDER.observer_errors, 0)
        self.assertEqual(self.rows('request.step_membership')[0]['schedule_id'], 1)
        sched.outputs = {0: NS(outputs=[engine_output('unscheduled', 0, Reason.ABORT)])}
        self.assertIs(sched.update_from_output(output), sched.outputs)
        self.assertIsNone(self.rows('request.output_observed')[0]['schedule_id'])

    def test_one_bad_output_observer_does_not_drop_other_outputs(self):
        class BadOutput:
            @property
            def request_id(self):
                raise ValueError('bad observation')
        sched = self.make_scheduler()
        sched.outputs = {0: NS(outputs=[BadOutput(), engine_output('ok', 0, Reason.STOP)])}
        self.assertIs(sched.update_from_output(scheduler_output([])), sched.outputs)
        self.assertEqual(len(self.rows('request.output_observed')), 1)
        self.assertEqual(self.rows('request.output_observed')[0]['request_id_sha256'], self.digest('ok'))
        self.assertGreater(self.trace._RECORDER.observer_errors, 0)

    def test_request_iteration_state_is_bounded_and_overflow_is_explicit(self):
        sched = self.make_scheduler()
        limit = self.hooks.REQUEST_ITERATION_LIMIT
        self.assertEqual(limit, 16384)
        for i in range(limit + 5):
            sched.add_request(request('r' + str(i)))
        self.assertEqual(len(sched._sim_collection_request_iterations), limit)
        overflow = 'r' + str(limit)
        output = scheduler_output([(overflow, 1)], new=[NS(req_id=overflow, num_computed_tokens=0)])
        sched.next_output = output
        sched.schedule()
        member = output._sim_collection_step_members[0]
        self.assertIsNone(member['request_iteration'])
        self.assertFalse(member['membership_complete'])
        self.assertFalse(output._sim_collection_membership_complete)
        self.assertEqual(len(sched._sim_collection_request_iterations), limit)
        sched._free_request(sched.requests['r0'])
        cached = scheduler_output([(overflow, 1)], cached=NS(
            req_ids=[overflow], num_computed_tokens=[1], resumed_req_ids=set()))
        sched.next_output = cached
        sched.schedule()
        self.assertIsNone(cached._sim_collection_step_members[0]['request_iteration'])
        self.assertEqual(len(sched._sim_collection_request_iterations), limit - 1)
        sched.outputs = {0: NS(outputs=[engine_output(overflow, 0, Reason.ABORT)])}
        sched.update_from_output(cached)
        replacement = request(overflow)
        sched.add_request(replacement)
        sched.next_output = scheduler_output([(overflow, 1)], new=[NS(req_id=overflow, num_computed_tokens=0)])
        fresh = sched.schedule()._sim_collection_step_members[0]
        self.assertEqual(fresh['request_iteration'], 1)
        self.assertEqual(len(sched._sim_collection_request_iterations), limit)

    def test_reused_id_gets_new_lifecycle_and_duplicate_add_is_not_continuation(self):
        sched = self.make_scheduler()
        req = request('same')
        sched.add_request(req)
        sched.next_output = scheduler_output([('same', 1)], new=[NS(req_id='same', num_computed_tokens=0)])
        first = sched.schedule()._sim_collection_step_members[0].copy()
        sched._free_request(req)
        replacement = request('same')
        sched.add_request(replacement)
        sched.next_output = scheduler_output([('same', 1)], new=[NS(req_id='same', num_computed_tokens=0)])
        second = sched.schedule()._sim_collection_step_members[0]
        self.assertEqual(second['request_iteration'], 1)
        self.assertNotEqual(first['request_lifecycle_id'], second['request_lifecycle_id'])
        sched.add_request(replacement)  # ambiguous duplicate admission of same object
        sched.next_output = scheduler_output([('same', 1)], cached=NS(
            req_ids=['same'], num_computed_tokens=[1], resumed_req_ids=set()))
        duplicate = sched.schedule()._sim_collection_step_members[0]
        self.assertTrue(duplicate['duplicate_id'])
        self.assertIsNone(duplicate['request_iteration'])
        self.assertFalse(duplicate['membership_complete'])

    def test_replacement_without_add_never_inherits_cached_iteration(self):
        sched = self.make_scheduler()
        sched.add_request(request('r'))
        sched.next_output = scheduler_output([('r', 1)], new=[NS(req_id='r', num_computed_tokens=0)])
        original = sched.schedule()._sim_collection_step_members[0]
        sched.requests['r'] = request('r')
        sched.next_output = scheduler_output([('r', 1)], cached=NS(
            req_ids=['r'], num_computed_tokens=[10], resumed_req_ids=set()))
        member = sched.schedule()._sim_collection_step_members[0]
        self.assertEqual(original['request_iteration'], 1)
        self.assertIsNone(member['request_iteration'])
        self.assertFalse(member['membership_complete'])

    def test_late_finish_for_reused_id_does_not_clear_new_lifecycle(self):
        sched = self.make_scheduler()
        old = request('r')
        sched.add_request(old)
        sched.next_output = scheduler_output([('r', 1)], new=[NS(req_id='r', num_computed_tokens=0)])
        old_output = sched.schedule()
        old_lifecycle = old_output._sim_collection_step_members[0]['request_lifecycle_id']
        sched.add_request(request('r'))
        new_lifecycle = sched._sim_collection_request_iterations['r']['lifecycle_id']
        self.assertNotEqual(old_lifecycle, new_lifecycle)
        sched.outputs = {0: NS(outputs=[engine_output('r', 0, Reason.STOP)])}
        sched.update_from_output(old_output)
        self.assertEqual(self.rows('request.output_observed')[-1]['request_lifecycle_id'], old_lifecycle)
        self.assertEqual(sched._sim_collection_request_iterations['r']['lifecycle_id'], new_lifecycle)
        # A free outside update for the old object must not label the replacement.
        sched._free_request(old)
        self.assertIsNone(self.rows('request.free')[-1]['request_lifecycle_id'])
        self.assertEqual(sched._sim_collection_request_iterations['r']['lifecycle_id'], new_lifecycle)

    def test_native_free_failure_inside_update_clears_active_schedule(self):
        sched = self.make_scheduler()
        req = request('r')
        sched.add_request(req)
        output = scheduler_output([('r', 1)], new=[NS(req_id='r', num_computed_tokens=0)])
        sched.next_output = output
        sched.schedule()
        sched.update_free = [req]
        error = OSError('native free')
        sched.errors['free'] = error
        with self.assertRaises(OSError) as caught:
            sched.update_from_output(output)
        self.assertIs(caught.exception, error)
        self.assertIsNone(sched._sim_collection_active_schedule_id)
        self.assertIsNone(sched._sim_collection_active_members)
        self.assertEqual(self.rows('request.free'), [])
        self.assertEqual(self.rows('request.output_observed'), [])
        self.assertIn('r', sched._sim_collection_request_iterations)

    def test_explicit_none_cached_past_stays_unknown_in_step_snapshot(self):
        output = scheduler_output([('r', 1)], cached=NS(req_ids=['r'], num_computed_tokens=None))
        snapshot = self.hooks.step_snapshot(output)
        self.assertIsNone(snapshot['requests'][0]['past_length'])
        self.assertFalse(snapshot['past_complete'])
        self.assertIsNone(snapshot['schedule_id'])
        self.assertFalse(snapshot['membership_complete'])

    def test_empty_scheduler_polls_do_not_emit_membership(self):
        sched = self.make_scheduler()
        for _ in range(100):
            self.assertIs(sched.schedule(), sched.next_output)
        self.assertEqual(sched.next_output._sim_collection_schedule_id, 100)
        self.assertEqual(len(sched.native_calls), 100)
        self.assertEqual(self.rows('request.step_membership'), [])

    def test_minimal_legacy_schedule_result_does_not_claim_complete_empty_members(self):
        sched = self.make_scheduler(subclass=True)
        sched.requests['r'] = request('r')
        output = NS(scheduled_new_reqs=[NS(req_id='r', num_computed_tokens=12)])
        sched.next_output = output
        self.assertIs(sched.schedule(True), output)
        self.assertEqual(len(self.rows('request.scheduled')), 1)
        self.assertEqual(self.rows('request.scheduled')[0]['total_kv_hit_len'], 12)
        self.assertEqual(self.rows('request.step_membership'), [])
        self.assertEqual(output._sim_collection_schedule_id, 1)

    def test_step_input_observer_failure_preserves_executor_result_and_exception(self):
        output = scheduler_output([('r', 1)], new=[NS(req_id='r', num_computed_tokens=0)])
        native_result = object()
        error = RuntimeError('native dispatch')
        class Executor:
            _sim_v2_last = ('c', 'e', 4, 'real')
            fail = False
            def collective_rpc(self, method, *args, **kwargs):
                if self.fail:
                    raise error
                return native_result
        self.hooks.CollectionExecutorHook.hook(Executor)
        executor = Executor()
        self.observer_failure = ValueError('step observer')
        self.assertIs(executor.collective_rpc('execute_model', args=(output,)), native_result)
        executor.fail = True
        with self.assertRaises(RuntimeError) as caught:
            executor.collective_rpc('execute_model', args=(output,))
        self.assertIs(caught.exception, error)
        self.assertGreater(self.trace._RECORDER.observer_errors, 0)

    def test_full_training_tick_keep_controller_lifecycle_and_existing_step_inputs(self):
        kinds = ('request.output_observed', 'request.free', 'request.remote_kv_signal_observed',
                 'request.step_membership', 'request.identity_alias', 'step.inputs', 'scheduler.output_update')
        for detail in ('full', 'training', 'tick'):
            for enabled in (False, True):
                with self.subTest(detail=detail, host=enabled):
                    self.stage.TRAINING = detail != 'full'
                    self.stage.TICK = detail == 'tick'
                    self.stage.host.ENABLED = enabled
                    for pcp in (-1, 0, 7):
                        ctrl = (-1, -1, -1, -1, pcp, -1, -1)
                        for kind in kinds:
                            self.assertTrue(self.stage.keep_event(kind, ctrl), (detail, kind))
                    for pp in range(4):
                        for pcp in (0, 7):
                            worker = (0, 0, 0, pp, pcp, 0, 0)
                            for kind in kinds:
                                self.assertEqual(self.stage.keep_event(kind, worker),
                                                 pcp == 0 and kind.startswith('request.'))
                            self.assertEqual(self.stage.keep_event('chunk.inputs', worker), pcp == 0)
                            self.assertTrue(self.stage.keep_event('native.output_ready_observed', worker))
                    self.assertFalse(self.stage.keep_event('request.free', (0, 0, 0, 4, 0, 0, 0)))

    def test_scheduler_update_export_retains_shared_controller_span(self):
        row = {'event': 'scheduler.output_update', 'context': None, 'rank': {'pp': -1},
               'data': {'request_ids_sha256': [self.digest('private-r')],
                        'schedule_id': 4, 'start_mono_ns': 10, 'end_mono_ns': 20,
                        'native_ok': True, 'timing_scope': 'batch_shared',
                        'per_request_apportionment': False}}
        with tempfile.TemporaryDirectory() as directory:
            window = Path(directory)
            source = window / 'events.0001.jsonl'
            source.write_text(json.dumps(row) + '\n')
            before = source.read_bytes()
            counts = self.stage.export_tables(window)
            self.assertEqual(counts['controller_points'], 1)
            self.assertEqual(json.loads((window / 'controller_points.jsonl').read_text()), row)
            self.assertEqual(source.read_bytes(), before)

    def test_remote_ready_hashed_host_observation_and_membership_export_retention(self):
        sched = self.make_scheduler()
        sched.finished_recving_kv_req_ids = ['private-r']
        sched.schedule()
        row = self.rows('request.remote_kv_signal_observed')[0]
        self.assertEqual(row['request_id_sha256'], self.digest('private-r'))
        self.assertNotIn('req_id', row)
        self.assertIn('wall_ns', row)
        self.assertEqual(row['time_kind'], 'host_remote_kv_signal_observed')
        self.assertFalse(row['ready_success_confirmed'])
        self.assertNotIn('kv_release_confirmed', row)

    def make_backend(self):
        owner = self
        marker = object()

        class Timing:
            def __init__(self):
                self.calls = []
                self.suppress = False

            @contextmanager
            def stage(self, stage):
                self.calls.append(('stage.enter', stage))
                try:
                    yield marker
                except ValueError:
                    if not self.suppress:
                        raise
                finally:
                    self.calls.append(('stage.exit', stage))

            @contextmanager
            def host_stage(self, stage):
                self.calls.append(('host.enter', stage))
                yield marker
                self.calls.append(('host.exit', stage))

        self.hooks.CollectionRunnerTimingHook.hook(Timing)

        class Manager:
            def __init__(self, runner):
                self.runner = runner
                self.calls = []
                self.error = None
                self.result = marker
                self.drafts = NS(draft_token_ids=[CountOnly(2), CountOnly(0)])

            def propose_draft_token_ids(self, *args, **kwargs):
                self.calls.append(('propose', args, kwargs))
                if self.error is not None:
                    raise self.error
                return self.result

            def take_draft_token_ids(self, *args, **kwargs):
                self.calls.append(('take', args, kwargs))
                if self.error is not None:
                    raise self.error
                return self.drafts

        self.hooks.CollectionSpecManagerHook.hook(Manager)

        class Runner:
            def __init__(self):
                self.input_batch = NS(req_ids=['secret-a', 'secret-b', 'secret-c', 'inactive'], num_reqs=3)
                self.speculative_config = NS(method='mtp')
                self.scheduler_config = NS(async_scheduling=True)
                self.scheduler = NS(scheduled_spec_decode_tokens={
                    'secret-a': CountOnly(0), 'secret-b': CountOnly(2), 'secret-c': CountOnly(0)})
                self.execute_model_state = NS(scheduler_output=self.scheduler,
                                              spec_decode_metadata_list=[None, NS()])
                self.timing = Timing()
                self.manager = Manager(self)
                self.calls = []
                self.result = marker
                self.error = None
                self.action = None

            def _prepare_inputs(self, *args, **kwargs):
                self.calls.append(('prepare', args, kwargs))
                return marker

            def _sample_spec_verify_chunk(self, *args, **kwargs):
                self.calls.append(('verify', args, kwargs))
                if self.error is not None:
                    raise self.error
                return marker

            def sample_tokens(self, *args, **kwargs):
                self.calls.append(('sample', args, kwargs))
                self.native_context = owner.hooks.SAMPLE_CONTEXT.get()
                self.native_scope = owner.trace.ACTIVE.get()
                self.execute_model_state = None  # verified native entry clears state
                if self.action is not None:
                    self.action()
                if self.error is not None:
                    raise self.error
                return self.result

        self.hooks.CollectionRunnerHook.hook(Runner)
        return Runner(), marker

    def make_output_classes(self, installed=True):
        native = self.load('native_boundaries')

        class Event:
            def __init__(self):
                self.calls = 0
            def synchronize(self):
                self.calls += 1

        class Copy:
            def __init__(self):
                self.completed = False
                self.calls = 0
                self.event = Event()
                self.error = None
                self.result = object()

            def wait(self, *args, **kwargs):
                self.calls += 1
                if self.error is not None:
                    raise self.error
                if not self.completed:
                    self.event.synchronize()
                    self.completed = True
                return self.result

            @property
            def _sampled_token_ids_cpu(self):
                raise AssertionError('private tensor read')

        self.hooks.CollectionAsyncCopyStateHook.hook(Copy)

        class Output:
            def __init__(self, model_runner_output, copy_state, kv_ready_event=None):
                self._model_runner_output = model_runner_output
                self.copy_state = copy_state
                self.kv_ready_event = kv_ready_event
                self.calls = 0
                self.error = None

            def get_output(self, *args, **kwargs):
                self.calls += 1
                if self.kv_ready_event is not None:
                    self.kv_ready_event.synchronize()
                if self.copy_state is not None:
                    self.copy_state.wait()
                if self.error is not None:
                    raise self.error
                if self.copy_state is None:
                    self._model_runner_output.sampled_token_ids = []
                return self._model_runner_output

        with patch.dict('os.environ', {'SIM_TPU_NATIVE_BOUNDARIES': '1' if installed else '0'}):
            self.hooks.CollectionOutputBoundaryHook.hook(Output)
        return Output, Copy, Event, native

    def test_host_span_start_follows_uuid_and_metadata_assembly(self):
        with self.advancing_clock() as clock:
            class Metadata:
                def keys(self):
                    clock.advance()
                    return ['phase']
                def __getitem__(self, key):
                    clock.advance()
                    return 'metadata-ready'
            def uuid4():
                clock.advance()
                return NS(hex='prepared-uuid')
            with patch.object(self.hooks.uuid, 'uuid4', uuid4):
                span = self.hooks.begin_host_span(Metadata(), stage='sample')
            self.assertEqual(span['start_mono_ns'], 400)
            self.assertEqual(span['phase'], 'metadata-ready')
            self.assertEqual(span['stage'], 'sample')
            clock.advance(10)
            ended = clock.now
            self.hooks.end_host_span('request.test', span, None, True, end_ns=ended)
            self.assertEqual(self.rows('request.test')[0]['end_mono_ns'], 410)
            self.assertGreater(clock.now, ended)
            with self.assertRaises(TypeError):
                self.hooks.end_host_span('request.test', span, None, True)

    def test_sample_end_precedes_context_cleanup_even_on_native_exception(self):
        for error in (None, ValueError('native'), KeyboardInterrupt('native')):
            with self.subTest(error=error), self.advancing_clock() as clock:
                runner, marker = self.make_backend()
                runner.action = lambda: clock.advance(10)
                runner.error = error
                sample_context = self.hooks.SAMPLE_CONTEXT
                active = self.trace.ACTIVE
                with patch.object(self.hooks, 'SAMPLE_CONTEXT', self.advancing_context(sample_context, clock)), \
                        patch.object(self.trace, 'ACTIVE', self.advancing_context(active, clock)):
                    if error is None:
                        self.assertIs(runner.sample_tokens(), marker)
                    else:
                        with self.assertRaises(type(error)) as caught:
                            runner.sample_tokens()
                        self.assertIs(caught.exception, error)
                row = self.rows('request.sample_tokens')[-1]
                self.assertEqual(row['end_mono_ns'], row['start_mono_ns'] + 10)
                self.assertGreaterEqual(clock.now - row['end_mono_ns'], 1200)
                self.assertEqual(len(runner.calls), 1)
                self.assertIsNone(sample_context.get())
                self.assertIsNone(active.get())

    def test_take_end_precedes_result_metadata_counts_and_emit(self):
        runner, marker = self.make_backend()
        with self.advancing_clock() as clock:
            class Row(CountOnly):
                def __len__(self):
                    clock.advance()
                    return super().__len__()
            class Drafts:
                @property
                def draft_token_ids(self):
                    clock.advance()
                    return [Row(2), Row(0)]
            runner.manager.drafts = Drafts()
            self.assertIs(runner.manager.take_draft_token_ids(), runner.manager.drafts)
            row = self.rows('mtp.take_draft_token_ids')[-1]
            self.assertEqual(row['end_mono_ns'], row['start_mono_ns'])
            self.assertEqual(row['draft_counts'], (2, 0))
            self.assertEqual(clock.now - row['end_mono_ns'], 1300)
            self.assertEqual(len(runner.manager.calls), 1)

    def test_copy_wait_end_precedes_completed_observation(self):
        with self.advancing_clock() as clock:
            class Copy:
                calls = 0
                @property
                def completed(self):
                    clock.advance()
                    return self.calls > 0
                def wait(self):
                    self.calls += 1
                    clock.advance(10)
                    return marker
            marker = object()
            self.hooks.CollectionAsyncCopyStateHook.hook(Copy)
            copy = Copy()
            self.assertIs(copy.wait(), marker)
            row = self.rows('request.copy_wait')[-1]
            self.assertEqual(row['end_mono_ns'], row['start_mono_ns'] + 10)
            self.assertEqual(clock.now - row['end_mono_ns'], 1100)
            self.assertTrue(row['completed_after'])
            self.assertTrue(row['effective_wait'])
            self.assertEqual(copy.calls, 1)

    def test_stage_exit_events_share_end_after_native_contextmanager_cleanup(self):
        scope = self.trace.Scope(('c', 'e', 1, 'real'))
        token = self.trace.ACTIVE.set(scope)
        self.addCleanup(self.trace.ACTIVE.reset, token)
        with self.advancing_clock() as clock:
            calls, ended = [], []
            marker = object()
            class Timing:
                @contextmanager
                def stage(self, name):
                    calls.append(('enter', name))
                    try:
                        yield marker
                    finally:
                        clock.advance(10)  # Native __exit__ must remain included.
                        ended.append(clock.now)
                        calls.append(('exit', name))
                host_stage = stage
            self.hooks.CollectionRunnerTimingHook.hook(Timing)
            timing = Timing()
            for method in ('stage', 'host_stage'):
                for error in (None, ValueError('body')):
                    with self.subTest(method=method, error=error):
                        def invoke():
                            with getattr(timing, method)('sample') as value:
                                self.assertIs(value, marker)
                                clock.advance(20)
                                if error is not None:
                                    raise error
                        if error is None:
                            invoke()
                        else:
                            with self.assertRaises(ValueError) as caught:
                                invoke()
                            self.assertIs(caught.exception, error)
                        runner_exit = self.rows('runner.sample.exit')[-1]
                        stage_exit = self.rows('mtp.stage')[-1]
                        self.assertEqual(runner_exit['end_mono_ns'], ended[-1])
                        self.assertEqual(stage_exit['end_mono_ns'], ended[-1])
                        self.assertEqual(stage_exit['native_ok'], error is None)
                        self.assertEqual(clock.now - ended[-1], 2000)
            self.assertEqual(calls, [('enter', 'sample'), ('exit', 'sample')] * 4)

    def test_output_end_precedes_existing_observer_counts_and_cleanup_on_all_paths(self):
        native = self.load('native_boundaries')
        for position in ('before', 'after', 'after_error'):
            for error in (None, KeyboardInterrupt('native get')):
                with self.subTest(position=position, error=error), self.advancing_clock() as clock:
                    class Row(CountOnly):
                        def __len__(self):
                            clock.advance()
                            return super().__len__()
                    class Result:
                        req_ids = ['r']
                        @property
                        def sampled_token_ids(self):
                            clock.advance()
                            return [Row(2)]
                    result = Result()
                    class Output:
                        calls = 0
                        def __init__(self):
                            self._model_runner_output = result
                        def get_output(self):
                            self.calls += 1
                            clock.advance(10)
                            self.native_ended = clock.now
                            if error is not None:
                                raise error
                            return result
                    def install(target):
                        original = target.get_output
                        def get(self):
                            if position == 'before':
                                clock.advance()
                                raise SystemExit('observer before native')
                            try:
                                value = original(self)
                            finally:
                                clock.advance()  # Existing post-native observer.
                            if position == 'after_error':
                                raise SystemExit('observer after native')
                            return value
                        target.get_output = get
                    with patch.object(native, 'install_output_class', install):
                        self.hooks.CollectionOutputBoundaryHook.hook(Output)
                    output = Output()
                    context, active = self.hooks.OUTPUT_CONTEXT, self.trace.ACTIVE
                    with patch.object(self.hooks, 'OUTPUT_CONTEXT', self.advancing_context(context, clock)), \
                            patch.object(self.trace, 'ACTIVE', self.advancing_context(active, clock)):
                        if error is None:
                            self.assertIs(output.get_output(), result)
                        else:
                            with self.assertRaises(KeyboardInterrupt) as caught:
                                output.get_output()
                            self.assertIs(caught.exception, error)
                    row = self.rows('request.output_materialize')[-1]
                    self.assertEqual(row['end_mono_ns'], output.native_ended)
                    self.assertEqual(row['native_ok'], error is None)
                    self.assertEqual(row['sampled_counts'], (2,) if error is None else None)
                    self.assertGreater(clock.now, row['end_mono_ns'])
                    self.assertEqual(output.calls, 1)
                    self.assertIsNone(context.get())
                    self.assertIsNone(active.get())

    def test_end_clock_failure_preserves_every_native_result_and_exception_identity(self):
        runner, marker = self.make_backend()
        Output, Copy, Event, native = self.make_output_classes(installed=False)
        result = NS(req_ids=['r'], sampled_token_ids=[])
        output, copy = Output(result, None), Copy()
        class Proposer:
            error = None
            calls = 0
            runner = NS(input_batch=NS(req_ids=['r'], num_reqs=1))
            def prefill_only(self, scheduler_output, device_seed):
                self.calls += 1
                if self.error is not None:
                    raise self.error
                return marker
        self.hooks.CollectionMtpProposerHook.hook(Proposer)
        proposer = Proposer()
        current_error = None
        def stage():
            with runner.timing.stage('sample'):
                if current_error is not None:
                    raise current_error
            return marker
        invocations = (
            ('request.sample_tokens', runner.sample_tokens, marker, lambda: len(runner.calls)),
            ('mtp.verify_accept', lambda: runner._sample_spec_verify_chunk(
                marker, NS(draft_lengths_cpu=[0]), None, runner.scheduler, 0, 1),
                marker, lambda: len(runner.calls)),
            ('mtp.propose_draft_token_ids', lambda: runner.manager.propose_draft_token_ids(None),
                marker, lambda: len(runner.manager.calls)),
            ('mtp.take_draft_token_ids', runner.manager.take_draft_token_ids,
                runner.manager.drafts, lambda: len(runner.manager.calls)),
            ('mtp.stage', stage, marker, lambda: len(runner.timing.calls) // 2),
            ('request.output_materialize', output.get_output, result, lambda: output.calls),
            ('request.copy_wait', copy.wait, copy.result, lambda: copy.calls),
            ('mtp.prefill_only', lambda: proposer.prefill_only(None, marker), marker,
                lambda: proposer.calls),
        )
        def emit(kind, scope=None, chunk=None, data=None):
            self.events.append({'event': kind, 'data': data})
        for kind, invoke, expected, calls in invocations:
            for current_error in (None, ValueError('native'), KeyboardInterrupt('native')):
                with self.subTest(kind=kind, error=current_error):
                    for target in (runner, runner.manager, output, copy, proposer):
                        target.error = current_error
                    reads = []
                    def clock():
                        reads.append(True)
                        if len(reads) == 1:
                            return 100  # Span starts successfully; only the end clock fails.
                        raise SystemExit('end clock observer failure')
                    before = calls()
                    with patch.object(self.hooks.time, 'monotonic_ns', clock), \
                            patch.object(self.trace, 'emit', emit):
                        if current_error is None:
                            self.assertIs(invoke(), expected)
                        else:
                            with self.assertRaises(type(current_error)) as caught:
                                invoke()
                            self.assertIs(caught.exception, current_error)
                    row = self.rows(kind)[-1]
                    self.assertEqual(row['start_mono_ns'], 100)
                    self.assertIsNone(row['end_mono_ns'])
                    self.assertEqual(row['native_ok'], current_error is None)
                    self.assertEqual(calls(), before + 1)
                    self.assertEqual(len(reads), 2)
                    self.assertIsNone(self.hooks.SAMPLE_CONTEXT.get())
                    self.assertIsNone(self.hooks.OUTPUT_CONTEXT.get())
                    self.assertIsNone(self.trace.ACTIVE.get())

    def test_plain_sample_verify_propose_hashes_counts_and_context_reset(self):
        runner, marker = self.make_backend()
        scope = self.trace.Scope(('c', 'e', 17, 'real'))
        token = self.trace.ACTIVE.set(scope)
        self.addCleanup(self.trace.ACTIVE.reset, token)
        md = NS(draft_lengths_cpu=[2, 0])
        verify_args = (marker, md, None, runner.scheduler, 1, 3, True, marker)
        def action():
            with runner.timing.stage('sample') as value:
                self.assertIs(value, marker)
                self.assertIs(runner._sample_spec_verify_chunk(*verify_args), marker)
            with runner.timing.stage('draft_propose'):
                self.assertIs(runner.manager.propose_draft_token_ids(
                    [CountOnly(0), CountOnly(1)], return_device=True, device_seed=marker), marker)
            for name in ('sampled_token_d2h', 'output_assembly'):
                with runner.timing.host_stage(name) as value:
                    self.assertIs(value, marker)
        runner.action = action
        self.assertIs(runner.sample_tokens(grammar_output=None), marker)
        self.assertEqual([c[0] for c in runner.calls], ['sample', 'verify'])
        self.assertEqual(runner.calls[1][1], verify_args)
        self.assertEqual(len(runner.manager.calls), 1)
        sample = self.rows('request.sample_tokens')[0]
        self.assertEqual(sample['request_ids_sha256'],
                         tuple(self.digest(r) for r in ('secret-a', 'secret-b', 'secret-c')))
        self.assertEqual(sample['scheduled_draft_counts'], (0, 2, 0))
        self.assertTrue(sample['has_scheduled_drafts'])
        self.assertTrue(sample['execute_state_present'])
        self.assertTrue(sample['spec_verify_step'])
        self.assertEqual(sample['sample_phase'], 'spec_verify')
        self.assertIsNone(runner.execute_model_state)
        verify = self.rows('mtp.verify_accept')[0]
        self.assertEqual(verify['request_ids_sha256'], sample['request_ids_sha256'][1:3])
        self.assertEqual(verify['draft_count'], 2)  # chunk-local, nonzero global offset
        self.assertEqual(verify['draft_counts'], (2, 0))
        self.assertEqual(verify['time_kind'], 'host_verify_accept_submission')
        self.assertFalse(verify['device_completion_confirmed'])
        self.assertFalse(verify['independent_pp_forward_duration'])
        self.assertTrue(verify['forward_included_in_main_forward'])
        propose = self.rows('mtp.propose_draft_token_ids')[0]
        self.assertEqual(propose['sampled_counts'], (0, 1))
        self.assertEqual(propose['spec_method'], 'mtp')
        self.assertFalse(propose['device_completion_confirmed'])
        self.assertEqual(verify['parent_span_id'], sample['span_id'])
        self.assertEqual(propose['parent_span_id'], sample['span_id'])
        for row in self.rows('mtp.stage'):
            self.assertEqual(row['request_ids_sha256'], sample['request_ids_sha256'])
            self.assertEqual(row['parent_span_id'], sample['span_id'])
            self.assertTrue(row['hostonly'])
            self.assertTrue(row['native_ok'])
            self.assertLessEqual(row['start_mono_ns'], row['end_mono_ns'])
        enter = self.rows('runner.sample.enter')[0]
        exit = self.rows('runner.sample.exit')[0]
        self.assertEqual(enter['span_id'], exit['span_id'])
        self.assertNotIn('secret-', repr(self.events))
        self.assertNotIn('inactive', repr(self.events))
        self.assertIsNone(self.hooks.SAMPLE_CONTEXT.get())
        self.assertIs(self.trace.ACTIVE.get(), scope)
        self.assertEqual(self.forbidden_imports, [])

    def test_plain_dispatcher_targets_registered_and_take_cpu_only(self):
        runner, marker = self.make_backend()
        hook = self.hooks.CollectionSpecManagerHook
        self.assertEqual(hook.HOOK_MODULE_NAME, 'vllm_torchtpu.runner.speculative_decoding_manager')
        self.assertEqual(hook.HOOK_CLASS_NAME, 'SpeculativeDecodingManager')
        self.assertEqual(self.hooks.HOOKS.count(hook), 1)
        copy_hook = self.hooks.CollectionAsyncCopyStateHook
        self.assertEqual(copy_hook.HOOK_CLASS_NAME, 'AsyncTPUCopyState')
        self.assertEqual(self.hooks.HOOKS.count(copy_hook), 1)
        self.assertEqual(runner.manager.calls, [])
        self.assertIs(runner.manager.take_draft_token_ids(), runner.manager.drafts)
        self.assertEqual(self.rows('mtp.take_draft_token_ids')[0]['draft_counts'], (2, 0))
        runner.manager.drafts = None
        self.assertIsNone(runner.manager.take_draft_token_ids())
        self.assertFalse(self.rows('mtp.take_draft_token_ids')[-1]['result_present'])
        self.assertFalse(self.rows('mtp.take_draft_token_ids')[-1]['device_completion_confirmed'])
        self.assertEqual(len(runner.manager.calls), 2)

    def test_dispatcher_exception_identity_and_observer_failures_preserved(self):
        for failing_emit in (False, True):
            runner, marker = self.make_backend()
            self.observer_failure = SystemExit('observer') if failing_emit else None
            for error in (ValueError('native'), KeyboardInterrupt('native')):
                runner.error = error
                for invoke in (lambda: runner.sample_tokens(None), lambda: runner._sample_spec_verify_chunk(
                        marker, NS(draft_lengths_cpu=[0]), None, runner.scheduler, 0, 1, True, marker)):
                    with self.assertRaises(type(error)) as caught:
                        invoke()
                    self.assertIs(caught.exception, error)
                    self.assertIsNone(self.hooks.SAMPLE_CONTEXT.get())
                runner.manager.error = error
                for invoke in (lambda: runner.manager.propose_draft_token_ids(None),
                               runner.manager.take_draft_token_ids):
                    with self.assertRaises(type(error)) as caught:
                        invoke()
                    self.assertIs(caught.exception, error)
            self.assertEqual(len(runner.calls), 4)
            self.assertEqual(len(runner.manager.calls), 4)
            runner.error = runner.manager.error = None
            self.assertIs(runner.sample_tokens(None), marker)
            self.assertIs(runner.manager.propose_draft_token_ids(None), marker)
            self.assertIs(runner.manager.take_draft_token_ids(), runner.manager.drafts)
        self.assertGreater(self.trace._RECORDER.observer_errors, 0)

    def test_metadata_clock_and_counter_failure_never_prevents_native(self):
        runner, marker = self.make_backend()
        class BrokenBatch:
            @property
            def req_ids(self):
                raise SystemExit('metadata')
        runner.input_batch = BrokenBatch()
        self.assertIs(runner.sample_tokens(None), marker)
        self.assertIs(runner.manager.propose_draft_token_ids(None), marker)
        self.assertIsNone(self.rows('request.sample_tokens')[0]['request_ids_sha256'])
        with patch.object(self.hooks.time, 'monotonic_ns', side_effect=ValueError('clock')):
            self.assertIs(runner.sample_tokens(None), marker)
        self.trace._RECORDER = NS()  # missing identity/counter is also an observer fault
        self.observer_failure = SystemExit('emit')
        self.assertIs(runner.sample_tokens(None), marker)
        self.assertIs(runner.manager.propose_draft_token_ids(None), marker)
        self.assertIsNone(self.hooks.SAMPLE_CONTEXT.get())
        self.assertEqual(len(runner.calls), 3)
        self.assertEqual(len(runner.manager.calls), 2)

    def test_stage_contextmanager_native_yield_suppression_and_failure_behavior(self):
        runner, marker = self.make_backend()
        scope = self.trace.Scope(('c', 'e', 1, 'real'))
        token = self.trace.ACTIVE.set(scope)
        self.addCleanup(self.trace.ACTIVE.reset, token)
        error = ValueError('body')
        with self.assertRaises(ValueError) as caught:
            with runner.timing.stage('sample') as value:
                self.assertIs(value, marker)
                raise error
        self.assertIs(caught.exception, error)
        self.assertFalse(self.rows('runner.sample.exit')[-1]['native_ok'])
        runner.timing.suppress = True
        with runner.timing.stage('sample'):
            raise error
        self.assertTrue(self.rows('runner.sample.exit')[-1]['native_ok'])
        self.observer_failure = KeyboardInterrupt('observer emit')
        with runner.timing.host_stage('output_assembly') as value:
            self.assertIs(value, marker)
        runner.timing.suppress = False
        with self.assertRaises(ValueError) as caught:
            with runner.timing.stage('sample'):
                raise error
        self.assertIs(caught.exception, error)
        self.assertEqual(len(runner.timing.calls), 8)

    def test_output_async_snapshot_repeated_wait_and_native_kv_wait_counts(self):
        runner, marker = self.make_backend()
        Output, Copy, Event, native = self.make_output_classes()
        scope = self.trace.Scope(('c', 'e', 8, 'real'))
        token = self.trace.ACTIVE.set(scope)
        self.addCleanup(self.trace.ACTIVE.reset, token)
        copy, kv = Copy(), Event()
        result = NS(req_ids=['output-secret'], sampled_token_ids=[CountOnly(0), CountOnly(3)])
        runner.action = lambda: setattr(runner, 'result', Output(result, copy, kv))
        output = runner.sample_tokens(None)
        self.assertEqual(output.calls, 0)
        self.assertEqual(copy.calls, 0)
        result.req_ids[:] = ['mutated-secret']
        self.trace.ACTIVE.set(None)
        self.assertIs(output.get_output(), result)
        self.assertIs(output.get_output(), result)
        self.assertEqual(output.calls, 2)
        self.assertEqual(copy.calls, 2)
        self.assertEqual(copy.event.calls, 1)
        self.assertEqual(kv.calls, 2)
        waits = self.rows('request.copy_wait')
        self.assertEqual([r['completed_before'] for r in waits], [False, True])
        self.assertEqual([r['effective_wait'] for r in waits], [True, False])
        self.assertTrue(all(r['device_completion_confirmed'] for r in waits))
        outputs = self.rows('request.output_materialize')
        self.assertEqual([r['effective_copy_wait'] for r in outputs], [True, False])
        self.assertEqual([r['sampled_counts'] for r in outputs], [(0, 3), (0, 3)])
        for i, row in enumerate(outputs):
            self.assertEqual(row['request_ids_sha256'], (self.digest('output-secret'),))
            self.assertEqual(waits[i]['parent_span_id'], row['span_id'])
            self.assertEqual(row['parent_span_id'], self.rows('request.sample_tokens')[0]['span_id'])
        self.assertNotIn('output-secret', repr(self.events))
        self.assertNotIn('mutated-secret', repr(self.events))
        self.assertTrue(all(e['context'] == scope.context for e in self.events
                            if e['event'] in ('request.copy_wait', 'request.output_materialize')))
        self.assertIsNone(self.hooks.OUTPUT_CONTEXT.get())
        self.assertIsNone(self.trace.ACTIVE.get())
        self.assertEqual(self.forbidden_imports, [])

    def test_empty_sample_and_nested_context_restore_without_false_phase(self):
        runner, marker = self.make_backend()
        runner.input_batch.num_reqs = 0
        runner.execute_model_state = None
        runner.result = None
        outer = {'span_id': 'outer-context'}
        token = self.hooks.SAMPLE_CONTEXT.set(outer)
        self.addCleanup(self.hooks.SAMPLE_CONTEXT.reset, token)
        self.assertIsNone(runner.sample_tokens(None))
        self.assertIs(self.hooks.SAMPLE_CONTEXT.get(), outer)
        row = self.rows('request.sample_tokens')[0]
        self.assertEqual(row['request_ids_sha256'], ())
        self.assertIsNone(row['scheduled_draft_counts'])
        self.assertIsNone(row['has_scheduled_drafts'])
        self.assertFalse(row['execute_state_present'])
        self.assertTrue(row['native_ok'])
        self.assertEqual(len(runner.calls), 1)

    def test_preexisting_output_constructor_and_get_observer_failure_is_fail_open(self):
        native = self.load('native_boundaries')
        for failure_position in ('before', 'after'):
            calls = []
            result = NS(req_ids=['r'], sampled_token_ids=[])
            class Output:
                init_error = None
                get_error = None
                def __init__(self):
                    calls.append('init')
                    if self.init_error is not None:
                        raise self.init_error
                    self._model_runner_output = result
                def get_output(self):
                    calls.append('get')
                    if self.get_error is not None:
                        raise self.get_error
                    return result
            def broken_installer(target):
                init, get = target.__init__, target.get_output
                def decorate(original):
                    def call(self, *args, **kwargs):
                        if failure_position == 'before':
                            raise SystemExit('boundary observer before native')
                        value = original(self, *args, **kwargs)
                        raise SystemExit('boundary observer after native')
                    return call
                target.__init__ = decorate(init)
                target.get_output = decorate(get)
            with patch.object(native, 'install_output_class', broken_installer):
                self.hooks.CollectionOutputBoundaryHook.hook(Output)
            output = Output()
            self.assertIs(output.get_output(), result)
            self.assertEqual(calls, ['init', 'get'])
            error = ValueError('native get')
            output.get_error = error
            with self.assertRaises(ValueError) as caught:
                output.get_output()
            self.assertIs(caught.exception, error)
            Output.init_error = error
            with self.assertRaises(ValueError) as caught:
                Output()
            self.assertIs(caught.exception, error)
            self.assertEqual(calls, ['init', 'get', 'get', 'init'])
            self.assertIsNone(self.hooks.OUTPUT_CONTEXT.get())
            self.assertIsNone(self.trace.ACTIVE.get())

    def test_output_sync_zero_missing_metadata_and_no_false_ready(self):
        Output, Copy, Event, native = self.make_output_classes(installed=False)
        for ids in ([], None):
            result = NS(sampled_token_ids=[CountOnly(99)])
            if ids is not None:
                result.req_ids = ids
            output = Output(result, None)
            self.assertIs(output.get_output(), result)
            row = self.rows('request.output_materialize')[-1]
            self.assertEqual(row['sampled_counts'], ())
            self.assertEqual(row['request_ids_sha256'], () if ids == [] else None)
            self.assertFalse(row['device_completion_confirmed'])
            self.assertFalse(row['copy_wait_observed'])
            self.assertEqual(output.calls, 1)
        self.assertEqual(self.rows('request.copy_wait'), [])

    def test_output_and_wait_errors_preserve_native_exception_without_ready(self):
        Output, Copy, Event, native = self.make_output_classes()
        for wait_failure in (False, True):
            copy = Copy()
            output = Output(NS(req_ids=['r'], sampled_token_ids=[]), copy)
            error = KeyboardInterrupt('native wait/output')
            if wait_failure:
                copy.error = error
            else:
                output.error = error
            with self.assertRaises(KeyboardInterrupt) as caught:
                output.get_output()
            self.assertIs(caught.exception, error)
            self.assertEqual(output.calls, 1)
            self.assertEqual(copy.calls, 1)
            self.assertFalse(self.rows('request.output_materialize')[-1]['native_ok'])
            self.assertFalse(self.rows('request.output_materialize')[-1]['device_completion_confirmed'])
            self.assertIsNone(self.hooks.OUTPUT_CONTEXT.get())
            self.assertIsNone(self.trace.ACTIVE.get())
        self.assertFalse(self.rows('request.copy_wait')[-1]['native_ok'])
        self.assertFalse(self.rows('request.copy_wait')[-1]['device_completion_confirmed'])

    def test_output_emit_and_preexisting_boundary_observer_failure_preserve_calls(self):
        Output, Copy, Event, native = self.make_output_classes()
        result = NS(req_ids=['r'], sampled_token_ids=[CountOnly(0)])
        copy = Copy()
        output = Output(result, copy)
        self.observer_failure = SystemExit('emit failure')
        with patch.object(native, 'output_ready', side_effect=ValueError('native boundary observer')):
            self.assertIs(output.get_output(), result)
        self.assertEqual(output.calls, 1)
        self.assertEqual(copy.calls, 1)
        self.assertIs(copy.wait(), copy.result)  # actual repeated native call, never skipped
        self.assertEqual(copy.calls, 2)
        self.assertEqual(copy.event.calls, 1)
        output.error = ValueError('native')
        with self.assertRaises(ValueError) as caught:
            output.get_output()
        self.assertIs(caught.exception, output.error)
        self.assertEqual(output.calls, 2)
        self.assertEqual(copy.calls, 3)
        self.assertIsNone(self.hooks.OUTPUT_CONTEXT.get())
        self.assertGreater(self.trace._RECORDER.observer_errors, 0)

    def test_output_observer_does_not_inspect_tensor_like_public_rows(self):
        Output, Copy, Event, native = self.make_output_classes(installed=False)
        class Opaque:
            def __getattribute__(self, name):
                raise AssertionError('device/tensor state read')
            def __iter__(self):
                raise AssertionError('device iteration')
            def __len__(self):
                raise AssertionError('device length')
        result = NS(req_ids=['r'], sampled_token_ids=Opaque())
        output = Output(result, Copy())
        self.assertIs(output.get_output(), result)
        self.assertIsNone(self.rows('request.output_materialize')[-1]['sampled_counts'])
        self.assertEqual(output.calls, 1)

    def test_pcp0_backend_metadata_only_and_all_rank_native_events_unchanged(self):
        runner, marker = self.make_backend()
        self.trace._RECORDER.identity = (0, 0, 0, 3, 7, 0, 0)
        runner.action = lambda: runner.manager.propose_draft_token_ids(None)
        self.assertIs(runner.sample_tokens(None), marker)
        Output, Copy, Event, native = self.make_output_classes()
        output = Output(NS(req_ids=['r'], sampled_token_ids=[]), Copy())
        output.get_output()
        self.assertFalse(any(e['event'].startswith(('request.', 'mtp.')) for e in self.events))
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(len(runner.manager.calls), 1)
        self.assertEqual(output.calls, 1)
        self.assertEqual(output.copy_state.calls, 1)

    def test_all_lifecycle_mtp_filter_retention_full_training_tick_and_export(self):
        kinds = ('request.sample_tokens', 'request.copy_wait', 'request.output_materialize',
                 'request.future_lifecycle', 'mtp.verify_accept', 'mtp.propose_draft_token_ids',
                 'mtp.take_draft_token_ids', 'mtp.stage')
        for training, tick in ((False, False), (True, False), (True, True)):
            self.stage.TRAINING, self.stage.TICK = training, tick
            for kind in kinds:
                self.assertTrue(self.stage.keep_event(kind, (-1,) * 7))
                for pp in range(4):
                    for pcp in (0, 7):
                        rank = (0, 0, 0, pp, pcp, 0, 0)
                        self.assertEqual(self.stage.keep_event(kind, rank), pcp == 0)
                        self.assertTrue(self.stage.keep_event('native.local_drain.ready', rank))
                        self.assertTrue(self.stage.keep_event('native.output_ready_observed', rank))
        # In-memory file stubs: no new export files in the workspace.
        from io import StringIO
        streams = {}
        class Stream(StringIO):
            def close(self):
                self.exported = self.getvalue()
                super().close()
        class File:
            def __init__(self, name):
                self.name = name
            def open(self, mode=None):
                if mode == 'w':
                    streams[self.name] = Stream()
                    return streams[self.name]
                return StringIO(''.join(json.dumps({'event': k, 'data': {}}) + '\n' for k in kinds))
        class Window:
            def __truediv__(self, name):
                return File(name)
            def glob(self, pattern):
                return [File('events.stub.jsonl')]
        counts = self.stage.export_tables(Window())
        self.assertEqual(counts['requests'], 4)
        self.assertEqual(counts['stage_points'], 4)
        retained = streams['requests.jsonl'].exported + streams['stage_points.jsonl'].exported
        for kind in kinds:
            self.assertIn(kind, retained)

    def test_mtp_prefill_only_preserves_device_seed_and_native_output(self):
        native_calls = []
        marker, seed = object(), object()
        error = RuntimeError('native prefill failed')

        class Proposer:
            def __init__(self):
                self.runner = NS(input_batch=NS(req_ids=['r'], num_reqs=1))

            def prefill_only(self, scheduler_output, device_seed,
                             discard_sampled_tokens_req_indices=None):
                native_calls.append((scheduler_output, device_seed, discard_sampled_tokens_req_indices))
                if scheduler_output == 'error':
                    raise error
                return marker

        self.hooks.CollectionMtpProposerHook.hook(Proposer)
        proposer = Proposer()
        self.assertIs(proposer.prefill_only('native', seed, [0]), marker)
        row = self.rows('mtp.prefill_only')[-1]
        self.assertEqual(row['request_ids_sha256'], (self.digest('r'),))
        self.assertEqual(row['phase'], 'mtp_prefill_only')
        self.assertFalse(row['produces_draft_proposals'])
        self.assertFalse(row['device_completion_confirmed'])
        self.assertIs(native_calls[0][1], seed)
        with self.assertRaises(RuntimeError) as raised:
            proposer.prefill_only('error', seed)
        self.assertIs(raised.exception, error)
        self.assertFalse(self.rows('mtp.prefill_only')[-1]['native_ok'])
        self.observer_failure = RuntimeError('observer failed')
        self.assertIs(proposer.prefill_only('native', seed), marker)
        self.assertEqual(len(native_calls), 3)

    def test_kv_native_log_duration_is_distinct_from_copy_wait_without_formatting(self):
        import logging

        handler = self.hooks.KvPublishWaitHandler()
        context = {'scope': None, 'span_id': 'parent'}
        token = self.hooks.OUTPUT_CONTEXT.set(context)
        try:
            record = logging.LogRecord('native', logging.INFO, 'native.py', 1,
                handler.template, (['r'], 123, 2.5), None)
            record.getMessage = lambda: (_ for _ in ()).throw(AssertionError('formatted raw log'))
            handler.emit(record)
            row = self.rows('request.kv_publish_wait')[-1]
            self.assertEqual(row['request_ids_sha256'], (self.digest('r'),))
            self.assertEqual(row['native_wait_duration_ms'], 2.5)
            self.assertEqual(row['parent_span_id'], 'parent')
            self.assertEqual(row['completion_scope'], 'existing_kv_publish_fence_not_token_copy')
            self.assertNotIn('req_ids', row)
            self.assertNotIn('start_mono_ns', row)
            self.assertFalse(row['valid_pure_device_timing'])
            before = len(self.events)
            record.msg = 'unrelated log with prompt'
            handler.emit(record)
            self.assertEqual(len(self.events), before)
            record.msg, record.args = handler.template, (['r'], 123, float('nan'))
            handler.emit(record)
            self.assertEqual(len(self.events), before)
            self.observer_failure = RuntimeError('observer failed')
            record.args = (['r'], 123, 1.0)
            handler.emit(record)
        finally:
            self.hooks.OUTPUT_CONTEXT.reset(token)

    def test_profile_export_controls_have_no_device_or_profiler_dependency(self):
        rec = self.trace._RECORDER
        calls = []
        def export_cut(export_id, start):
            calls.append((export_id, start))
            return {'complete': True, 'export_id': export_id}
        rec.export_cut = export_cut

        class Worker:
            def profile(self, *args, **kwargs):
                raise AssertionError('hardware profiler called')
            def execute_model(self, *args, **kwargs):
                return None
        self.hooks.CollectionWorkerHook.hook(Worker)
        worker = Worker()

        class Executor:
            def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
                return [getattr(worker, method)(*args, **(kwargs or {}))]
        self.hooks.CollectionExecutorHook.hook(Executor)
        executor = Executor()
        self.assertTrue(executor.start_profile()[0]['complete'])
        self.assertTrue(executor.stop_profile()[0]['complete'])
        self.assertEqual([start for _, start in calls], [True, True, False, False])
        self.assertEqual(calls[0][0], calls[1][0])
        self.assertEqual(calls[2][0], calls[3][0])
        self.assertNotEqual(calls[0][0], calls[2][0])
        self.assertIs(self.trace._RECORDER, rec)
        self.assertEqual(self.forbidden_imports, [])
        self.assertEqual(rec.observer_errors, 0)


if __name__ == '__main__':
    unittest.main()
