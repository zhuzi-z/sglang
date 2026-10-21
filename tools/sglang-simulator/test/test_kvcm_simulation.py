"""Regression tests for metadata-only KVCM/MockPace save and load simulation."""

import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sglang_simulator.simulation.vllm.pace import mock_pace_server as mp
from sglang_simulator.simulation.vllm.v6d.tair_kvcm_simulation import (
    C_TairKVCMTransferRuntimeHook,
    SimulatedKVCMTransfer,
    local_mock_url,
)


class KVCMSimulationTests(unittest.TestCase):
    def setUp(self):
        config = mp.parse_config(["--host", "127.0.0.1", "--nodes", "1", "--dram", "1MB",
                                  "--ssd", "4MB", "--stats-interval", "0"], env={})
        self.app = mp.MockPace(config, env={})
        mp.build_nodes(self.app, config)
        self.provider = self.app.meta.provider(1)
        self.server = mp.Server(("127.0.0.1", 0), self.app)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_address[1])
        self.config = SimpleNamespace(storage_configs=json.dumps([
            {"global_unique_name": "pace_local", "storage_type": "ST_TAIRMEMPOOL",
             "tair_mem_pool": {"domain": self.base}},
            {"global_unique_name": "event", "storage_type": "ST_EVENT_REPORT_L2"},
        ]))
        self.transfer = SimulatedKVCMTransfer(self.config)
        self.size = 65536
        self.ga = self.provider.alloc(mp.MEDIA_AUTO, self.size)
        self.uri = f"pace://pace_local/{self.ga}?size={self.size}&node_id=1&media_type=0"

    def tearDown(self):
        self.transfer.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_save_load_tracks_readability_without_payload(self):
        self.assertFalse(self.transfer.load([self.uri], [1], [self.size]))
        self.assertEqual(self.transfer.save([self.uri], [1], [self.size]), [True])
        self.assertTrue(self.transfer.load([self.uri], [1], [self.size]))
        self.assertTrue(self.provider.describe(self.ga)["sim_written"])
        self.assertEqual(self.app.stats_view()["payload_bytes_moved"], 0)
        self.assertEqual(self.provider.stats["sim_save_bytes"], self.size)

    def test_freed_ga_cannot_be_fabricated_as_a_hit(self):
        self.transfer.save([self.uri], [1], [self.size])
        self.provider.free(self.ga)
        self.assertFalse(self.transfer.load([self.uri], [1], [self.size]))
        self.assertEqual(self.transfer.save([self.uri], [1], [self.size]), [False])
        self.assertNotIn(self.ga, self.provider.sim_written)

    def test_unhealthy_provider_and_partial_results(self):
        invalid = "pace://pace_local/1?size=" + str(self.size)
        self.assertEqual(self.transfer.save([self.uri, invalid], [1, 2], [self.size] * 2), [True, False])
        self.provider.healthy = False
        self.assertFalse(self.transfer.load([self.uri], [1], [self.size]))

    def test_sizes_and_uris_are_validated(self):
        self.assertEqual(self.transfer.save([self.uri], [1], [self.size + 1]), [False])
        self.assertEqual(self.transfer.save(["pace://unknown/1"], [1], [self.size]), [False])
        self.assertEqual(self.transfer.save(["file:///tmp/not-a-ga"], [1], [self.size]), [False])
        with self.assertRaises(ValueError):
            self.transfer.save([self.uri], [], [self.size])
        with self.assertRaises(ValueError):
            self.transfer.save([self.uri], [0], [self.size])
        with self.assertRaises(ValueError):
            self.transfer.save([self.uri], [1], [True])
        self.assertFalse(self.provider.describe(self.ga)["sim_written"])

    def test_faulted_save_does_not_mark_a_ga_written(self):
        self.app.faults.set("transfer_save", {"fail_ratio": 1, "fail_mode": "status"})
        self.assertEqual(self.transfer.save([self.uri], [1], [self.size]), [False])
        self.assertFalse(self.provider.describe(self.ga)["sim_written"])
        self.app.faults.clear("transfer_save")
        self.transfer.save([self.uri], [1], [self.size])
        self.app.faults.set("transfer_load", {"fail_ratio": 1, "fail_mode": "http", "http_code": 503})
        self.assertFalse(self.transfer.load([self.uri], [1], [self.size]))

    def test_clear_and_snapshot_restore_written_state(self):
        self.transfer.save([self.uri], [1], [self.size])
        state = self.provider.dump()
        self.provider.clear()
        self.assertFalse(self.provider.sim_written)
        self.provider.restore(state["seq"], state["gas"], state["sim_written"])
        self.assertTrue(self.transfer.load([self.uri], [1], [self.size]))
        self.provider.restore(state["seq"], state["gas"])
        self.assertFalse(self.transfer.load([self.uri], [1], [self.size]))

    def test_only_loopback_mock_services_are_accepted(self):
        for endpoint in ("http://example.com:12348", "https://127.0.0.1:12348",
                         "http://user:password@127.0.0.1:12348", "http://127.0.0.1:12348/path"):
            with self.assertRaises(ValueError):
                local_mock_url(endpoint)
        with patch.object(SimulatedKVCMTransfer, "request", return_value={"status": "success", "version": "real"}):
            with self.assertRaises(ValueError):
                SimulatedKVCMTransfer(self.config)

    def test_runtime_hook_never_enters_native_initializer(self):
        class NativeRuntime:
            def __init__(self, *args, **kwargs):
                raise AssertionError("native SDK must never initialize")

            def batch_save_from_memory(self, *args):
                raise AssertionError("native SDK must never save")

            def batch_load_to_memory(self, *args):
                raise AssertionError("native SDK must never load")

        C_TairKVCMTransferRuntimeHook.hook(NativeRuntime)
        runtime = NativeRuntime(self.config)
        self.assertIsNone(runtime._pybind)
        self.assertEqual(runtime.batch_save_from_memory([self.uri], [1], [self.size]), [True])
        self.assertTrue(runtime.batch_load_to_memory([self.uri], [1], [self.size]))
        with self.assertRaises(ValueError):
            NativeRuntime(self.config, transfer_client=object())
        runtime._transfer_client.close()

    def test_manager_internal_storage_config_shape(self):
        config = SimpleNamespace(storage_configs=json.dumps([
            {"type": "pace", "global_unique_name": "pace_local", "storage_spec": {"domain": self.base}},
        ]))
        transfer = SimulatedKVCMTransfer(config)
        self.assertEqual(transfer.save([self.uri], [1], [self.size]), [True])
        self.assertTrue(transfer.load([self.uri], [1], [self.size]))
        transfer.close()

    def test_malformed_batch_response_cannot_commit_a_success_prefix(self):
        second_ga = self.provider.alloc(mp.MEDIA_AUTO, self.size)
        second_uri = f"pace://pace_local/{second_ga}?size={self.size}"
        response = {"results": [{"ga": self.ga, "success": True}, {"ga": -1, "success": True}]}
        with patch.object(self.transfer, "request", return_value=response):
            self.assertEqual(self.transfer.save([self.uri, second_uri], [1, 2], [self.size] * 2),
                             [False, False])

    def test_empty_batches_do_not_issue_io(self):
        with patch.object(self.transfer, "request", side_effect=AssertionError("unexpected I/O")):
            self.assertEqual(self.transfer.save([], [], []), [])
            self.assertTrue(self.transfer.load([], [], []))


class StandaloneRunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = runpy.run_path(str(ROOT / "scripts/run_kvcm_simulation.py"))

    def test_transfer_hook_is_installed_without_environment_flag(self):
        from sglang_simulator.simulation.vllm.v6d import v6d_capacity

        for env in ({}, {"V6D_SIM_KVCM_TRANSFER": "0"}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True), patch.object(
                    v6d_capacity, "install_class_hooks") as install:
                v6d_capacity.install_v6d_runtime_hooks()
            install.assert_called_once_with([
                v6d_capacity.C_VineyardPeerHook,
                v6d_capacity.C_TieredVineyardPeerHook,
                v6d_capacity.C_HitRateStatsHook,
                C_TairKVCMTransferRuntimeHook,
            ])

    def test_inherited_production_configuration_is_removed(self):
        with patch.dict(os.environ, {"kvcm_registry_storage_uri": "redis://production:6379",
                                    "KVS_KVCM_ENDPOINT": "spectrum://production",
                                    "SGLANG_SIMULATOR_ENABLE": "1"}):
            env = self.runner["child_environment"]()
        self.assertEqual(env["kvcm_registry_storage_uri"], "")
        self.assertNotIn("KVS_KVCM_ENDPOINT", env)
        self.assertEqual(env["SGLANG_SIMULATOR_ENABLE"], "0")
        startup = self.runner["startup_config"]("http://127.0.0.1:22348", "local-test")
        self.assertNotIn("media_type", startup["storage_config"]["storage_spec"])
        self.assertEqual(startup["instance_group"]["name"], "local-test")

    def test_defaults_match_requested_pool_geometry(self):
        captured = {}
        main = self.runner["main"]
        with patch.object(sys, "argv", ["runner", "run"]), patch.dict(
                main.__globals__, {"run": lambda args: captured.update(vars(args)) or 0}):
            self.assertEqual(main(), 0)
        self.assertEqual(mp.parse_size(captured["capacity"]), 500 * (1 << 30))
        self.assertEqual(captured["object_size"], 200 * (1 << 20))
        self.assertEqual(captured["providers"], 2)
        self.assertEqual(mp.parse_size(captured["dram"]) * captured["providers"], 2 * (1 << 40))
        self.assertEqual(mp.parse_size(captured["ssd"]) * captured["providers"], 5 * (1 << 40))
        startup = self.runner["startup_config"]("http://127.0.0.1:22348", "local-test", 7 * (1 << 40))
        quota = startup["instance_group"]["quota"]
        self.assertEqual(quota["capacity"], 7 * (1 << 40))
        self.assertEqual(quota["quota_config"][0]["capacity"], quota["capacity"])

    def test_capacity_target_exceeds_nominal_by_one_object(self):
        target = self.runner["capacity_target_objects"](7 * (1 << 40), 200 * (1 << 20))
        self.assertEqual(target, 36701)
        self.assertGreater(target * 200 * (1 << 20), 7 * (1 << 40))
        self.assertLessEqual((target - 1) * 200 * (1 << 20), 7 * (1 << 40))
        with self.assertRaises(ValueError):
            self.runner["capacity_target_objects"](0, 1)

    def test_daemon_rejects_nonlocal_services_before_starting(self):
        for manager, redis in (("http://production:6382", "redis://127.0.0.1:6379"),
                               ("http://127.0.0.1:6382", "redis://production:6379")):
            with self.assertRaises(ValueError):
                self.runner["daemon"](SimpleNamespace(manager_url=manager, redis_url=redis))

    @unittest.skipUnless((Path("/home/admin/kv_cache_manager/bin/kv_cache_manager_bin").is_file()
                          and shutil.which("redis-server")), "local Manager and Redis are required")
    def test_scaled_capacity_overflow_end_to_end(self):
        command = [sys.executable, str(ROOT / "scripts/run_kvcm_simulation.py"), "run",
                   "--capacity-overflow", "--capacity", "32M", "--object-size", "1048576",
                   "--dram", "4MB", "--ssd", "8MB", "--batch-size", "4", "--progress-every", "100"]
        env = dict(os.environ, SGLANG_SIMULATOR_ENABLE="0")
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report_line = next(line for line in result.stdout.splitlines() if line.startswith("Report: "))
        report = json.loads(Path(report_line.removeprefix("Report: ")).read_text())
        capacity = report["capacity_case"]
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["simulation_configuration"]["mode"], "capacity-overflow")
        self.assertGreater(capacity["attempted_bytes"], capacity["nominal_capacity_bytes"])
        self.assertGreater(capacity["rejected_objects"], 0)
        self.assertTrue(capacity["rollback_verified"] and capacity["false_location_verified"])
        self.assertTrue(all(process["exitcode"] == 0 for process in report["processes"]))


if __name__ == "__main__":
    unittest.main()
