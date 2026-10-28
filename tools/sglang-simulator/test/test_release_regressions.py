import os, subprocess, sys
from pathlib import Path
# stub 仅在子进程安装，pytest 收集不会加载模拟包或 torch。
if __name__ == "__main__":
    os.environ["_SGLANG_RELEASE_CHILD"] = "1"
if os.environ.get("_SGLANG_RELEASE_CHILD") == "1":
    """发布前回归：默认配置、时延口径及 controller 完成链，无 GPU 依赖。"""
    import asyncio
    import importlib.util
    import json
    import logging
    import os
    from pathlib import Path
    import sys
    import tempfile
    import types
    import unittest
    from unittest.mock import patch

    ROOT = Path(os.environ.get('RELEASE_TEST_SOURCE', Path(__file__).resolve().parents[1] / "src/sglang_simulator/simulation/vllm/v6d"))

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    bw = load('_release_bandwidth', ROOT / 'bandwidth.py')

    def profile():
        return {'layout': {'page_size_bytes': 100, 'num_layers': 2},
                'segments': {'local_load': {'bandwidth_bytes_per_s': 1000},
                             'local_store': {'bandwidth_bytes_per_s': 2000}},
                'control_plane': {'seg1_cross_node': {'floor_ms': 5, 'per_block_ms': 2,
                                                    'bandwidth_bytes_per_s': 1000}}}

    class BandwidthTests(unittest.TestCase):
        def setUp(self):
            self.env = patch.dict(os.environ, {}, clear=True)
            self.env.start()
            bw.BandwidthModel.reset()

        def tearDown(self):
            bw.BandwidthModel.reset()
            self.env.stop()

        def test_no_profile_public_methods_are_safe(self):
            for config in (None, {}):
                model = bw.BandwidthModel(config)
                self.assertFalse(model.enabled)
                for n in (-1, 0, 5):
                    self.assertEqual(model.seg1_latency(n), 0)
                    self.assertEqual(model.latency_for(n, True), 0)
                    self.assertEqual(model.store_completion_latency(n), 0)

        def test_missing_profile_no_crash(self):
            os.environ['SIMULATOR_HICACHE_BW_PROFILE'] = '/not/a/profile.json'
            self.assertEqual(bw.BandwidthModel.get().seg1_latency(5), 0)

        def test_malformed_json_no_crash(self):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'profile.json'
                path.write_text('{invalid')
                os.environ['SIMULATOR_HICACHE_BW_PROFILE'] = str(path)
                self.assertEqual(bw.BandwidthModel.get().seg1_latency(5), 0)

        def test_invalid_profile_schema_no_crash(self):
            for config in (['bad'], {'layout': {'num_layers': 'bad'}}, {'segments': []}):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'profile.json'
                    path.write_text(json.dumps(config))
                    os.environ['SIMULATOR_HICACHE_BW_PROFILE'] = str(path)
                    bw.BandwidthModel.reset()
                    self.assertEqual(bw.BandwidthModel.get().seg1_latency(5), 0)

        def test_byte_model_per_rank(self):
            model = bw.BandwidthModel(profile())
            self.assertTrue(model.enabled)
            self.assertEqual(model.seg2_bytes(5), 1000)
            self.assertAlmostEqual(model.latency_for(5, True), 1)
            self.assertAlmostEqual(model.seg1_latency(5), 1.005)

        def test_zero_and_negative_load_have_no_floor(self):
            model = bw.BandwidthModel(profile())
            self.assertEqual(model.seg1_latency(0), 0)
            self.assertEqual(model.seg1_latency(-1), 0)

        def test_floor_env_overrides_byte_model(self):
            os.environ['SGLANG_SIMULATOR_SEG1_FLOOR_MS'] = '9'
            self.assertAlmostEqual(bw.BandwidthModel(profile()).seg1_latency(5), 1.009)

        def test_per_block_env_overrides_byte_model(self):
            os.environ['SGLANG_SIMULATOR_SEG1_PER_BLK_MS'] = '3'
            self.assertAlmostEqual(bw.BandwidthModel(profile()).seg1_latency(5), .020)

        def test_env_control_plane_without_profile(self):
            os.environ['SGLANG_SIMULATOR_SEG1_FLOOR_MS'] = '4'
            os.environ['SGLANG_SIMULATOR_SEG1_PER_BLK_MS'] = '3'
            self.assertAlmostEqual(bw.BandwidthModel(None).seg1_latency(5), .019)

        def test_legacy_per_block_profile(self):
            config = profile()
            del config['control_plane']['seg1_cross_node']['bandwidth_bytes_per_s']
            self.assertAlmostEqual(bw.BandwidthModel(config).seg1_latency(5), .015)

        def test_store_completion_poll_and_rank(self):
            self.assertAlmostEqual(bw.BandwidthModel(profile()).store_completion_latency(1), .101)

        def test_samples_refit_largest_two_points(self):
            segment = bw.SegmentModel.from_samples('test', {'samples': [
                {'bytes': 100, 'median_s': .1}, {'bytes': 300, 'median_s': .2},
                {'bytes': 200, 'median_s': .15}], 'bandwidth_bytes_per_s': 1})
            self.assertAlmostEqual(segment.bw, 2000)
            self.assertAlmostEqual(segment.latency(300), .2)

        def test_singleton_reset(self):
            first = bw.BandwidthModel.get()
            self.assertIs(first, bw.BandwidthModel.get())
            bw.BandwidthModel.reset()
            self.assertIsNot(first, bw.BandwidthModel.get())

    class ControllerTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            class BaseHook:
                pass
            class Envs:
                mode = 'OFFLINE'
                @classmethod
                def simulation_mode(cls):
                    return cls.mode
            class StateManager:
                clock = 0
                @classmethod
                def get_global_clock(cls):
                    return cls.clock
            cls.envs, cls.state = Envs, StateManager
            stubs = {
                'sglang_simulator': {"__path__": []},
                'sglang_simulator.simulation': {"__path__": []},
                'sglang_simulator.simulation.vllm': {"__path__": []},
                'sglang_simulator.simulation.vllm.v6d': {"__path__": []},

                'sglang_simulator.hook': {'BaseHook': BaseHook, 'install_class_hooks': lambda hooks: None},
                'sglang_simulator.simulation.manager': {'Envs': Envs, 'StateManager': StateManager},
                'sglang_simulator.simulation.vllm.cpu_stubs': {'DummyEvent': type('DummyEvent', (), {})},
                'sglang_simulator.simulation.vllm.v6d.bandwidth': {'BandwidthModel': bw.BandwidthModel, 'validate_load_timing': bw.validate_load_timing},
                'sglang_simulator.utils': {'get_logger': lambda: logging.getLogger('release-test')},
            }
            cls.saved = {name: sys.modules.get(name) for name in stubs}
            for name, attributes in stubs.items():
                module = types.ModuleType(name)
                vars(module).update(attributes)
                sys.modules[name] = module
            cls.backend = load('_release_backend', ROOT / 'v6d_backend.py')
            cls.capacity = load('_release_capacity', ROOT / 'v6d_capacity.py')

        @classmethod
        def tearDownClass(cls):
            for name, module in cls.saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        def setUp(self):
            self.env = patch.dict(os.environ, {}, clear=True)
            self.env.start()
            bw.BandwidthModel.reset()
            self.envs.mode, self.state.clock = 'OFFLINE', 0
            self.freed, self.done = [], []
            cache = types.SimpleNamespace(
                _release_protected_blocks=lambda rid: None,
                _block_pool=types.SimpleNamespace(free_blocks=lambda blocks: self.freed.extend(blocks)),
                mamba_group_ids={0}, _swap_protected_blocks={})
            native = types.SimpleNamespace(_scheduler=cache)
            hybrid = types.SimpleNamespace(_backend=native, _saving={}, _saved=set(),
                                           _loaded={}, pending_requests={}, _step_saved=lambda: None)
            self.controller = self.backend.SimulatedKVController(hybrid)
            self.controller._load_done = lambda rid, n: self.done.append(('load', rid, n))
            self.controller._store_done = lambda req, failed=False: self.done.append(('store', req.request_id, failed))

        def tearDown(self):
            bw.BandwidthModel.reset()
            self.env.stop()

        def test_no_profile_load_ack_exactly_once(self):
            req = types.SimpleNamespace(request_id='load')
            self.controller.queue_load(req, 10, {0: (['remote'], [0])})
            self.assertEqual(self.controller.progress(), 1)
            self.assertEqual(self.controller.progress(), 0)
            self.assertEqual(self.done, [('load', 'load', 10)])

        def test_offline_clock_controls_load_deadline(self):
            bw.BandwidthModel._instance = bw.BandwidthModel(profile())
            req = types.SimpleNamespace(request_id='load')
            self.controller.queue_load(req, 10, {0: (['remote'], [0])})
            self.assertEqual(self.controller.progress(), 0)
            self.state.clock = 1
            self.assertEqual(self.controller.progress(), 1)

        def test_local_load_skips_seg1(self):
            bw.BandwidthModel._instance = bw.BandwidthModel(profile())
            self.controller._local_keys.add('local')
            self.controller.queue_load(types.SimpleNamespace(request_id='local'), 10, {0: (['local'], [0])})
            self.assertAlmostEqual(self.controller._loads['local'][0], .2)

        def test_store_arity_2_3_and_future_fields(self):
            for arity in (2, 3, 4):
                rid = str(arity)
                req = types.SimpleNamespace(request_id=rid)
                block = types.SimpleNamespace(block_id=arity)
                self.controller.hybrid._saving[rid] = types.SimpleNamespace(_req=req)
                self.controller.cache._swap_protected_blocks[rid] = [block]
                entry = ({0: ([rid], [arity])}, True) + tuple(range(arity - 2))
                meta = types.SimpleNamespace(reqs_to_store={rid: entry})
                self.controller.capture_stores(meta)
                self.assertNotIn(rid, self.controller.cache._swap_protected_blocks)
            self.assertEqual(self.controller.progress(), 3)
            self.assertEqual(len(self.freed), 3)
            self.assertEqual(len(self.done), 3)
            self.assertEqual(self.controller.progress(), 0)
            self.assertEqual(len(self.freed), 3)

        def test_reset_refuses_active_requests(self):
            self.controller.hybrid.pending_requests['active'] = object()
            self.assertFalse(self.controller.reset())

        def test_reset_releases_owned_blocks_once(self):
            req = types.SimpleNamespace(request_id='finished', is_finished=lambda: True)
            self.controller.hybrid._saving[req.request_id] = types.SimpleNamespace(_req=req)
            self.controller._staged = [self.controller.Store(req, ['pin'], 1, ['key'], True)]
            self.assertTrue(self.controller.reset())
            self.controller.hybrid._saving.clear()
            self.assertTrue(self.controller.reset())
            self.assertEqual(self.freed, ['pin'])
            self.assertEqual(self.done, [('store', 'finished', True)])

        def test_default_metadata_probe_is_real(self):
            self.assertFalse(self.capacity._stub_meta_probe_enabled())
            os.environ['V6D_SIM_STUB_META_PROBE'] = '1'
            self.assertTrue(self.capacity._stub_meta_probe_enabled())

        def test_unknown_mode_rejected(self):
            os.environ['SIMULATOR_LOAD_TIMING'] = 'typo'
            with self.assertRaisesRegex(ValueError, 'controller'):
                bw.validate_load_timing()

        def test_invalid_mode_rejected_before_target_patch(self):
            class Target:
                def __init__(self):
                    pass
            original = Target.__init__
            os.environ['SIMULATOR_LOAD_TIMING'] = 'data_plane'
            with self.assertRaisesRegex(ValueError, 'controller'):
                self.backend.C_V6dObjectBackendHook.hook(Target)
            self.assertIs(Target.__init__, original)

    if __name__ == '__main__':
        unittest.main(verbosity=2)

else:
    def test_release_regressions_subprocess():
        result = subprocess.run([sys.executable, str(Path(__file__).resolve())], capture_output=True, text=True, timeout=25)
        assert result.returncode == 0, result.stdout + result.stderr
