#!/usr/bin/env python3
"""Run a real v6d/Manager + mock-MemPool workflow without vLLM or payload I/O.

Usage: python3 scripts/run_kvcm_simulation.py run
Logs, isolated configuration, and a JSON report are kept under tmp.out/kvcm/simulation.
All services are newly created locally and stopped when the scenario finishes.
The copied /tair_mempool package and existing deployments are not modified.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
import uuid

ROOT = Path(__file__).resolve().parents[1]
OPENER = build_opener(ProxyHandler({}))


def http(base, path, payload=None, *, method=None, auth=False, timeout=10):
    if urlsplit(base).hostname != "127.0.0.1":
        raise ValueError("The standalone runner only contacts numeric loopback endpoints")
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = "Basic " + base64.b64encode(b"admin:123").decode()
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(base + path, data=data, headers=headers, method=method)
    with OPENER.open(request, timeout=timeout) as response:
        body = response.read()
        result = json.loads(body) if body else {}
    code = result.get("header", {}).get("status", {}).get("code")
    if code is not None and code != "OK":
        raise RuntimeError(f"{path}: {result['header']['status']}")
    if result.get("status") == "failed":
        raise RuntimeError(f"{path}: mock service rejected request")
    return result


def api(base, operation, **payload):
    return http(base, "/api/" + operation, {"trace_id": "standalone-simulation", **payload})


def capacity_target_objects(nominal_capacity, object_size):
    """Smallest object count whose requested bytes exceed nominal capacity."""
    if nominal_capacity <= 0 or object_size <= 0:
        raise ValueError("capacities must be positive")
    return nominal_capacity // object_size + 1


def free_port(used):
    while True:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        if port not in used:
            used.add(port)
            return port


def child_environment():
    # Do not inherit production configuration or Python auto-bootstrap hooks.
    env = {key: value for key, value in os.environ.items()
           if not key.lower().startswith("kvcm_")
           and not key.startswith(("V6D_", "KVS_", "TAIR_MEMPOOL_", "PACE_", "SPECTRUM_"))}
    env.update({"PYTHONPATH": str(ROOT / "src"), "SGLANG_SIMULATOR_ENABLE": "0",
                "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "DS_METER_DISABLE": "1", "DS_LOG_ROOT_DIR": "logs",
                "kvcm_registry_storage_uri": "", "kvcm_coordination_uri": "",
                "kvcm_distributed_lock_uri": ""})
    return env


def startup_config(mempool_url, group_name, capacity_bytes=1 << 40):
    """Build the local Manager configuration; no ignored reference checkout is required."""
    return {
        "storage_config": {"type": "pace", "global_unique_name": "pace_local",
                           "storage_spec": {"domain": mempool_url, "timeout": 3}},
        "instance_group": {
            "name": group_name, "storage_candidates": ["pace_local"],
            "global_quota_group_name": group_name + "_quota", "max_instance_count": 100,
            "quota": {"capacity": capacity_bytes,
                      "quota_config": [{"storage_type": "pace", "capacity": capacity_bytes}]},
            "cache_config": {
                "reclaim_strategy": {"storage_unique_name": "pace_local", "reclaim_policy": 1,
                                     "trigger_strategy": {"used_size": 0, "used_percentage": 0.8},
                                     "trigger_period_seconds": 60, "reclaim_step_size": 67108864,
                                     "reclaim_step_percentage": 10, "delay_before_delete_ms": 1000,
                                     "instance_reclaim_budget_policy": 2},
                "cache_prefer_strategy": 5,
                "meta_indexer_config": {"max_key_count": 1000000, "mutex_shard_num": 16,
                                        "batch_key_size": 16,
                                        "meta_storage_backend_config": {"storage_type": "local", "storage_uri": ""},
                                        "meta_cache_policy_config": {"type": "LRU", "capacity": 64,
                                                                     "cache_shard_bits": 0, "high_pri_pool_ratio": 0.0}},
            },
            "user_data": "isolated no-payload simulation", "version": 1,
        },
    }


class Services:
    def __init__(self, directory):
        self.directory = directory
        self.processes = []

    def start(self, name, command, env):
        handle = (self.directory / (name + ".log")).open("w")
        process = subprocess.Popen(command, cwd=self.directory, env=env,
                                   stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        self.processes.append((name, process, handle))
        return process

    def wait(self, name, predicate, timeout=40):
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            for label, process, _ in self.processes:
                if process.poll() is not None:
                    raise RuntimeError(f"{label} exited with {process.returncode}; see {label}.log")
            try:
                if predicate():
                    return
            except (OSError, ValueError, RuntimeError) as exc:
                last = str(exc)
            time.sleep(0.2)
        raise RuntimeError(f"Timed out waiting for {name}: {last}")

    def stop_one(self, name):
        for label, process, _ in reversed(self.processes):
            if label != name or process.poll() is not None:
                continue
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                # The Manager's MemPool health thread sleeps up to ten seconds.
                process.wait(timeout=20 if name == "manager" else 8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)

    def close(self):
        for name, _, _ in reversed(self.processes):
            self.stop_one(name)
        for _, _, handle in self.processes:
            handle.close()


def daemon(args):
    manager = urlsplit(args.manager_url)
    tracker = urlsplit(args.redis_url)
    if (manager.scheme != "http" or manager.hostname != "127.0.0.1"
            or tracker.scheme != "redis" or tracker.hostname != "127.0.0.1"):
        raise ValueError("The standalone daemon requires loopback Manager and Redis endpoints")
    sys.path.insert(0, str(ROOT / "src"))
    os.environ.update({"V6D_LAZY_LOAD": "0", "SRPC_STREAM_DISABLE_RDMA": "1",
                       "V6D_SIM_STUB_META_PROBE": "0"})
    from sglang_simulator.simulation.vllm.v6d.v6d_capacity import install_v6d_runtime_hooks
    install_v6d_runtime_hooks()
    from v6d.common import telemetry
    telemetry._ENABLED = False  # Standalone validation must not start external telemetry.
    sys.argv = ["v6d", "serve", "--peer=tiered_vineyard", "--vineyard-size=" + args.capacity,
                "--vineyard-socket=vineyard.sock", "--port=" + str(args.port),
                "--vineyard-rpc-port=" + str(args.rpc_port), "--peer-id=" + args.instance,
                "--tracker-redis=" + args.redis_url, "--tracker-prefix=" + args.instance,
                "--memory-capacity=1000000", "--memory-usage-max=0.95",
                "--tair-kvcm-mode=report-event-and-cache-transfer", "--tair-kvcm-p2p=false",
                "--tair-kvcm-manager-endpoint=" + args.manager_url,
                "--tair-kvcm-instance-group=" + args.group,
                "--tair-kvcm-instance-id=" + args.instance,
                "--tair-kvcm-query-type=QT_PREFIX_MATCH", "--no-tair-kvcm-auto-discover-leader",
                "--no-shutdown-kvcm-flush-enabled", "--tair-kvcm-request-timeout-seconds=3"]
    from v6d.cli.cli import main
    return main()


def locations(manager, instance, block_keys):
    result = api(manager, "getCacheLocationsByBackend", instance_id=instance,
                 block_keys=block_keys, query_type="QT_BATCH_GET",
                 block_mask={"offset": 0}, location_spec_names=["F0"] * len(block_keys),
                 backend_selectors=[{"backend_type": "ST_TAIRMEMPOOL", "strategy": "LSS_WEIGHTED_RANDOM"}])
    uris = []
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "uri" and isinstance(item, str) and item.startswith("pace:"):
                    uris.append(item)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(result.get("key_locations", []))
    return result, uris


def run(args):
    package = Path(args.manager_package).resolve()
    binary = package / "bin/kv_cache_manager_bin"
    server_config = package / "etc/default_server_config.conf"
    logger_config = package / "etc/default_logger_config.conf"
    for path in (binary, server_config, logger_config):
        if not path.is_file():
            raise RuntimeError(f"Missing Manager artifact: {path}")
    if not shutil.which("redis-server"):
        raise RuntimeError("redis-server is required for the isolated v6d tracker")
    if (args.objects < 2 or args.object_size <= 0 or args.providers < 1
            or args.batch_size < 1 or args.batch_size > 128 or args.progress_every < 1):
        raise ValueError("objects >= 2, providers/object-size positive, batch-size 1..128, progress-every positive")
    run_id = "sim_" + uuid.uuid4().hex[:12]
    directory = ROOT / "tmp.out/kvcm/simulation" / run_id
    directory.mkdir(parents=True, mode=0o700)
    (directory / "logs").mkdir()
    services = Services(directory)
    used = set()
    ports = {name: free_port(used) for name in
             ("redis", "manager", "admin", "manager_rpc", "admin_rpc", "mempool", "v6d", "vineyard_rpc")}
    urls = {name: "http://127.0.0.1:" + str(ports[name]) for name in ("manager", "admin", "mempool", "v6d")}
    report = {"status": "running", "run_id": run_id, "directory": str(directory),
              "ports": ports, "payload_bytes_moved": 0, "checks": [],
              "simulation_configuration": {"mode": "capacity-overflow" if args.capacity_overflow else "smoke",
                                           "v6d_capacity": args.capacity,
                                           "smoke_objects": None if args.capacity_overflow else args.objects,
                                           "object_size_bytes": args.object_size},
              "copied_mempool": {"path": args.mempool_package,
                                  "present": Path(args.mempool_package).is_dir(),
                                  "used_for_payload_io": False}}
    (directory / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Simulation logs: {directory}", flush=True)
    env = child_environment()

    def check(name, condition, detail=None):
        report["checks"].append({"name": name, "passed": bool(condition), "detail": detail})
        print(("PASS " if condition else "FAIL ") + name, flush=True)
        if not condition:
            raise RuntimeError("Validation failed: " + name)

    def acquire(keys, scope="read", metas=None):
        body = {"scope": scope, "object_keys": keys, "term": 30,
                "request_id": "simulation-" + scope}
        if metas is not None:
            body["object_metas"] = metas
        return http(urls["v6d"], "/acquire", body)

    def release(response):
        http(urls["v6d"], "/release", {"lease_id": response["lease"]["lease_id"]})

    def run_smoke_case():
        block_keys = list(range(1, args.objects + 1))
        keys = [f"{key:016x}_0" for key in block_keys]
        created = acquire(keys, "create", [{"size": args.object_size} for _ in keys])
        check("CREATE without payload allocation", len(created.get("objects", [])) == len(keys))
        pinned = http(urls["v6d"], "/objects/evict", {"keys": [keys[0]]})
        check("leased objects cannot be evicted", keys[0] in pinned.get("skipped_leased", []))
        http(urls["v6d"], "/seal", {"lease_id": created["lease"]["lease_id"]})
        release(created)
        evicted = http(urls["v6d"], "/objects/evict", {"keys": keys})
        check("eviction removed all local objects", set(evicted.get("evicted", [])) == set(keys))
        response, uris = locations(urls["manager"], run_id, block_keys)
        report["locations_after_writeback"] = response
        check("real Manager committed cold locations", len(uris) >= len(keys), uris)
        stats = http(urls["mempool"], "/admin/stats")
        provider_stats = stats["counters"]["provider"]
        check("mock Provider recorded simulated saves", provider_stats.get("sim_save_objects", 0) >= len(keys))
        if args.require_demotion:
            services.wait("transparent SSD demotion", lambda: http(urls["mempool"], "/admin/stats")["counters"]["provider"].get("demoted_gas", 0) > 0,
                          timeout=5)
            check("transparent tiering demoted GAs without copying bytes", True)
        cold_read = acquire(keys)
        check("cold READ materialized all objects", len(cold_read.get("objects", [])) == len(keys))
        release(cold_read)
        stats = http(urls["mempool"], "/admin/stats")
        check("eager READ reached simulated SDK loads", stats["counters"]["provider"].get("sim_load_objects", 0) >= len(keys))
        check("no payload bytes moved", stats.get("payload_bytes_moved") == 0)
        # Remove the hot copy, then invalidate only this isolated mock's GAs.
        http(urls["v6d"], "/objects/evict", {"keys": [keys[0]]})
        gas = http(urls["mempool"], "/v1/api/gas", auth=True)
        all_gas = [key for key in gas if key.isdigit()]
        for start in range(0, len(all_gas), 128):
            http(urls["mempool"], "/v1/api/gas?ga=" + ",".join(all_gas[start:start + 128]),
                 method="DELETE", auth=True)
        try:
            acquire([keys[0]])
        except HTTPError as exc:
            check("missing GA fails instead of returning a fabricated hit", exc.code >= 400)
        else:
            check("missing GA fails instead of returning a fabricated hit", False)
        report["final_mempool_stats"] = http(urls["mempool"], "/admin/stats")

    def wait_for_demotion(demoted_before, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = http(urls["mempool"], "/admin/stats")
            if current["counters"]["provider"].get("demoted_gas", 0) > demoted_before:
                return True
            time.sleep(0.05)
        return False

    def run_capacity_case(nominal_capacity):
        target_objects = capacity_target_objects(nominal_capacity, args.object_size)
        attempted = succeeded = rejected = batches = transient_retries = 0
        first_success = first_rejected = None
        rollback_verified = false_location_verified = False
        report["capacity_case"] = {
            "target_objects": target_objects,
            "target_requested_bytes": target_objects * args.object_size,
            "nominal_capacity_bytes": nominal_capacity,
            "batch_size": args.batch_size,
        }
        max_batches = math.ceil(target_objects / args.batch_size) * 4
        while attempted < target_objects:
            if batches >= max_batches:
                raise RuntimeError("capacity workload exceeded its bounded retry budget")
            count = min(args.batch_size, target_objects - attempted)
            block_keys = list(range(attempted + 1, attempted + count + 1))
            keys = [f"{key:016x}_0" for key in block_keys]
            before = http(urls["mempool"], "/admin/stats")
            before_ga = before["ga_count"]
            before_save = before["counters"]["provider"].get("sim_save_objects", 0)
            before_fail = before["counters"]["meta"].get("alloc_failed_requests", 0)

            created = acquire(keys, "create", [{"size": args.object_size} for _ in keys])
            if len(created.get("objects", [])) != count:
                raise RuntimeError("capacity CREATE returned an incomplete object batch")
            http(urls["v6d"], "/seal", {"lease_id": created["lease"]["lease_id"]})
            release(created)
            evicted = http(urls["v6d"], "/objects/evict", {"keys": keys})
            if set(evicted.get("evicted", [])) != set(keys):
                raise RuntimeError("capacity batch was not fully removed from local Vineyard")

            after = http(urls["mempool"], "/admin/stats")
            ga_delta = after["ga_count"] - before_ga
            save_delta = after["counters"]["provider"].get("sim_save_objects", 0) - before_save
            fail_delta = after["counters"]["meta"].get("alloc_failed_requests", 0) - before_fail
            if save_delta == count and ga_delta == count and fail_delta == 0:
                succeeded += count
                if first_success is None:
                    first_success = (keys[0], block_keys[0])
                capacity = after["capacity"]
                tiering = after["tiering"]
                ssd_cap = after["ga_alloc_config"]["ssd_alloc_max_ratio_pct"]
                should_tick = (capacity["dram_total_bytes"] > 0
                               and capacity["dram_used_bytes"] * 100 >= tiering["hi_pct"] * capacity["dram_total_bytes"]
                               and capacity["ssd_total_bytes"] > 0
                               and capacity["ssd_used_bytes"] * 100 < ssd_cap * capacity["ssd_total_bytes"])
                if should_tick:
                    demoted_before = after["counters"]["provider"].get("demoted_gas", 0)
                    services.wait("MemPool watermark tick", lambda: http(urls["mempool"], "/admin/stats")
                                  ["counters"]["provider"].get("demoted_gas", 0) > demoted_before, timeout=3)
            elif save_delta == 0 and ga_delta == 0 and fail_delta > 0:
                rollback_verified = True
                capacity = after["capacity"]
                tiering = after["tiering"]
                ssd_cap = after["ga_alloc_config"]["ssd_alloc_max_ratio_pct"]
                demoted_before = before["counters"]["provider"].get("demoted_gas", 0)
                can_demote = (capacity["dram_total_bytes"] > 0
                              and capacity["dram_used_bytes"] * 100 >= tiering["hi_pct"] * capacity["dram_total_bytes"]
                              and capacity["ssd_total_bytes"] > 0
                              and capacity["ssd_used_bytes"] * 100 < ssd_cap * capacity["ssd_total_bytes"])
                if can_demote and wait_for_demotion(demoted_before):
                    transient_retries += 1
                    batches += 1
                    continue  # Retry the same keys after asynchronous tiering made progress.
                rejected += count
                if first_rejected is None:
                    first_rejected = (list(keys), list(block_keys))
                    _, rejected_uris = locations(urls["manager"], run_id, block_keys)
                    false_location_verified = not rejected_uris
                    if rejected_uris:
                        raise RuntimeError("Manager exposed cold locations for a rolled-back capacity batch")
            else:
                raise RuntimeError(
                    f"non-atomic capacity result: requested={count} ga_delta={ga_delta} "
                    f"save_delta={save_delta} alloc_fail_delta={fail_delta}"
                )
            attempted += count
            batches += 1
            if batches == 1 or batches % args.progress_every == 0 or attempted == target_objects:
                print("CAPACITY progress attempted=%d/%d requested=%.3fTiB stored=%.3fTiB rejected=%d" % (
                    attempted, target_objects, attempted * args.object_size / float(1 << 40),
                    succeeded * args.object_size / float(1 << 40), rejected), flush=True)

        final = http(urls["mempool"], "/admin/stats")
        attempted_bytes = attempted * args.object_size
        stored_bytes = final["counters"]["provider"].get("sim_save_bytes", 0)
        check("capacity workload requested more than nominal MemPool", attempted_bytes > nominal_capacity)
        check("MemPool rejected writes after saturation", rejected > 0
              and final["counters"]["meta"].get("alloc_failed_requests", 0) > 0)
        check("failed allocation batches rolled back atomically", rollback_verified
              and final["ga_count"] == final["counters"]["provider"].get("sim_save_objects", 0))
        check("Manager did not commit rejected capacity batches", false_location_verified)
        check("capacity workload used SSD transparent tiering",
              final["capacity"]["ssd_used_bytes"] > 0
              and final["counters"]["provider"].get("demoted_gas", 0) > 0)
        check("stored logical bytes stayed within nominal capacity", 0 < stored_bytes <= nominal_capacity)
        if first_success is None or first_rejected is None:
            raise RuntimeError("capacity case did not observe both success and rejection")
        cold = acquire([first_success[0]])
        check("old cold object remains readable after MemPool saturation", len(cold.get("objects", [])) == 1)
        release(cold)
        final = http(urls["mempool"], "/admin/stats")
        check("capacity case moved no payload bytes", final.get("payload_bytes_moved") == 0)
        report["capacity_case"].update({
            "attempted_objects": attempted,
            "attempted_bytes": attempted_bytes,
            "committed_objects": succeeded,
            "committed_bytes": stored_bytes,
            "rejected_objects": rejected,
            "batches": batches,
            "transient_tiering_retries": transient_retries,
            "first_committed_key": first_success[0],
            "first_rejected_key": first_rejected[0][0],
            "rollback_verified": rollback_verified,
            "false_location_verified": false_location_verified,
        })
        report["final_mempool_stats"] = final

    try:
        services.start("redis", [shutil.which("redis-server"), "--bind", "127.0.0.1",
                                "--port", str(ports["redis"]), "--save", "", "--appendonly", "no"], env)
        mock = ROOT / "src/sglang_simulator/simulation/vllm/pace/mock_pace_server.py"
        mock_env = dict(env, TAIR_MEMPOOL_BATCH_ALLOC_CNT_LIMIT_PER_NODE="8",
                        TAIR_MEMPOOL_GA_NODE_SELECT_POLICY="round_robin",
                        TAIR_MEMPOOL_GA_CANDIDATE_SHUFFLE="1", TAIR_MEMPOOL_GA_INFLIGHT_ADJUST="1",
                        TAIR_MEMPOOL_GA_SSD_ALLOC_MAX_RATIO_PCT="88",
                        TAIR_MEMPOOL_TIERING_CAPACITY_WATERMARK_PCT="88")
        services.start("mempool", [sys.executable, str(mock), "--host", "127.0.0.1",
                                  "-p", str(ports["mempool"]), "--nodes", str(args.providers),
                                  "--dram", args.dram, "--ssd", args.ssd,
                                  "--tiering-mode", "transparent", "--demote-high-watermark-pct", "50",
                                  "--demote-low-watermark-pct", "35", "--stats-interval", "0"], mock_env)
        services.wait("mock MemPool", lambda: http(urls["mempool"], "/v1/api/healthy").get("status") == "success")
        initial = http(urls["mempool"], "/admin/stats")
        check("snapshot-aligned MemPool placement and tiering", initial["ga_alloc_config"]["node_select_policy"] == "round_robin"
              and initial["ga_alloc_config"]["per_node_alloc_limit"] == 8
              and initial["tiering"]["hi_pct"] == 50 and initial["tiering"]["lo_pct"] == 35)
        total_dram = initial["capacity"]["dram_total_bytes"]
        total_ssd = initial["capacity"]["ssd_total_bytes"]
        report["mempool_configuration"] = {"providers": args.providers, "dram": args.dram, "ssd": args.ssd,
                                           "dram_total_bytes": total_dram, "ssd_total_bytes": total_ssd,
                                           "placement": initial["ga_alloc_config"], "tiering": initial["tiering"]}
        nominal_capacity = total_dram + total_ssd
        # Capacity-overflow tests isolate physical MemPool admission from KVCM's
        # independent 80% ledger reclaimer by giving Manager ample headroom.
        manager_capacity = nominal_capacity * (4 if args.capacity_overflow else 1)
        startup = startup_config(urls["mempool"], run_id, manager_capacity)
        report["mempool_nominal_capacity_bytes"] = nominal_capacity
        report["manager_capacity_bytes"] = manager_capacity
        startup_path = directory / "startup.json"
        startup_path.write_text(json.dumps(startup, indent=2) + "\n")
        manager_command = [str(binary), "-c", str(server_config), "-l", str(logger_config),
                           "-e", "kvcm.startup_config=" + str(startup_path)]
        for name, port_key in (("http_port", "manager"), ("admin_http_port", "admin"),
                               ("rpc_port", "manager_rpc"), ("admin_rpc_port", "admin_rpc")):
            manager_command += ["-e", "kvcm.service." + name + "=" + str(ports[port_key])]
        manager_command += ["-e", "kvcm.service.io_thread_num=2"]
        services.start("manager", manager_command, env)
        services.wait("Manager bootstrap", lambda: api(urls["admin"], "getInstanceGroup", name=run_id).get("instance_group"))
        api(urls["admin"], "addStorage", storage={"global_unique_name": "sim_event_l2",
            "storage_type": "ST_EVENT_REPORT_L2", "event_report": {"heartbeat_timeout_ms": 30000,
            "cleanup_grace_ms": 300000, "liveness_check_interval_ms": 5000},
            "check_storage_available_when_open": False})
        group = api(urls["admin"], "getInstanceGroup", name=run_id)["instance_group"]
        version = int(group["version"])
        group["event_report_storage_candidates"] = ["sim_event_l2"]
        group["version"] = version + 1
        api(urls["admin"], "updateInstanceGroup", instance_group=group, current_version=version)
        storages = api(urls["admin"], "listStorage")["storage"]
        check("real Manager registered pace storage", any(s["global_unique_name"] == "pace_local"
                                                         and s["storage_type"] == "ST_TAIRMEMPOOL" for s in storages))
        child = [sys.executable, str(Path(__file__).resolve()), "daemon", "--capacity", args.capacity,
                 "--port", str(ports["v6d"]), "--rpc-port", str(ports["vineyard_rpc"]),
                 "--redis-url", "redis://127.0.0.1:" + str(ports["redis"]),
                 "--manager-url", urls["manager"], "--group", run_id, "--instance", run_id]
        services.start("v6d", child, env)
        services.wait("v6d", lambda: http(urls["v6d"], "/health"))
        http(urls["v6d"], "/kv-cache/spec", {"block_size": 16, "use_eagle_pop": False,
             "groups": [{"group_id": 0, "spec_type": "FullAttentionSpec",
                          "object_size_bytes": args.object_size, "meta": {"typename": "vineyard::Object"}}]})
        services.wait("KVCM BOTH_OK", lambda: "BOTH_OK" in (directory / "v6d.log").read_text(), timeout=60)
        check("v6d registration and simulated SDK initialization", True)
        if args.capacity_overflow:
            run_capacity_case(nominal_capacity)
        else:
            run_smoke_case()
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        print("Simulation failed: " + str(exc), file=sys.stderr)
    finally:
        services.close()
        report["processes"] = [{"name": name, "pid": process.pid, "exitcode": process.poll()}
                               for name, process, _ in services.processes]
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Report: " + str(directory / "report.json"))
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    test = subparsers.add_parser("run", help="start, validate, and stop the complete isolated workflow")
    test.add_argument("--manager-package", default="/home/admin/kv_cache_manager")
    test.add_argument("--mempool-package", default="/tair_mempool")
    test.add_argument("--capacity", default="500G", help="sparse Vineyard capacity, not resident RAM")
    test.add_argument("--objects", type=int, default=8)
    test.add_argument("--object-size", type=int, default=200 * (1 << 20),
                      help="modeled bytes per object (default: 200 MiB)")
    test.add_argument("--providers", type=int, default=2)
    test.add_argument("--dram", default="1TB", help="virtual DRAM per Provider (default total: 2 TiB across two Providers)")
    test.add_argument("--ssd", default="2.5TB", help="virtual SSD per Provider (default total: 5 TiB across two Providers)")
    test.add_argument("--require-demotion", action="store_true",
                      help="require a demotion during the smoke test; use small --dram capacity")
    test.add_argument("--capacity-overflow", action="store_true",
                      help="stream batches until attempted bytes exceed nominal MemPool capacity")
    test.add_argument("--batch-size", type=int, default=128,
                      help="objects per capacity-test write batch (default: 128)")
    test.add_argument("--progress-every", type=int, default=20,
                      help="capacity-test progress interval in batches (default: 20)")
    child = subparsers.add_parser("daemon", help="internal: launch only the hooked v6d daemon")
    child.add_argument("--capacity", required=True)
    child.add_argument("--port", type=int, required=True)
    child.add_argument("--rpc-port", type=int, required=True)
    child.add_argument("--redis-url", required=True)
    child.add_argument("--manager-url", required=True)
    child.add_argument("--group", required=True)
    child.add_argument("--instance", required=True)
    args = parser.parse_args()
    try:
        return daemon(args) if args.command == "daemon" else run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
