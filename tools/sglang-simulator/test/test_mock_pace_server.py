#!/usr/bin/env python3
"""
test_mock_pace_server.py -- self test of mock_pace_server.py: start-up option parsing, then a server on a
random port checked route by route against the real protocol semantics.

  python3 test_mock_pace_server.py      # prints PASS/FAIL per check, exit code 1 on failure
  python3 -m pytest test_mock_pace_server.py
"""
import base64
import json
import logging
import os
import sys
import threading
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mock_pace_server as mp  # noqa: E402


def run():
    failures = []

    def check(name, cond, detail=""):
        print("%-4s %s%s" % ("PASS" if cond else "FAIL", name, (" -- " + str(detail)[:600]) if not cond else ""))
        if not cond:
            failures.append(name)

    # ---- start-up options
    c = mp.parse_config(["-p", "23456", "-b", "redis", "-c", "host=x;password=y", "-l", "redis", "-P", "1024",
                         "-i", "1", "-t", "5", "-m", "prometheus", "-M", "12349", "-n", "-L", "WARN",
                         "--exclude_network_interface", "ansm0", "--client_io_threads", "64"], env={})
    check("pace meta CLI accepted", c.port == 23456 and c.backend == "redis" and c.log_level == "WARN", vars(c))
    c = mp.parse_config(["--byenv"], env={
        "TAIR_MEMPOOL_META_LOG_PATH": "/tmp/m.log", "TAIR_MEMPOOL_META_STORAGE": "-b redis",
        "TAIR_MEMPOOL_META_HA": "-l redis", "TAIR_MEMPOOL_META_EXACT_ARGS": "-P 1024 -i 1 -t 5 -m prometheus",
        "TAIR_MEMPOOL_META_EXCLUDE_NIC": "ansm0", "TAIR_MEMPOOL_META_BACKEND_ARGS": "host=r-x;prefix=pace:"})
    check("--byenv mirrors start_metaservice_byenv.sh", c.argv[:2] == ["-o", "/tmp/m.log"] and c.backend == "redis"
          and c.leader_election == "redis" and c.thread_num == 1024 and c.backend_args == "host=r-x;prefix=pace:", c.argv)
    c = mp.parse_config(["--nodes", "2"], env={"MOCK_PACE_NODES": "7", "MOCK_PACE_DRAM": "8GB"})
    check("MOCK_PACE_* env, CLI wins", c.nodes == 2 and c.dram == "8GB", vars(c))
    st = mp.GaAllocConfigStore({"TAIR_MEMPOOL_GA_NODE_SELECT_POLICY": "rr",
                                "TAIR_MEMPOOL_GA_SSD_ALLOC_MAX_RATIO_PCT": "88",
                                "TAIR_MEMPOOL_GA_DRAM_GATE": "7"})
    check("GA env baseline (invalid value keeps default)", st.get()["node_select_policy"] == "round_robin"
          and st.get()["ssd_alloc_max_ratio_pct"] == 88 and st.get()["dram_gate"] == 1, st.get())

    # ---- service: provider 1 transparent, provider 2 explicit
    cfg = mp.parse_config([], env={})
    cfg.port, cfg.host, cfg.state_file, cfg.stats_interval = 0, "127.0.0.1", None, 0
    cfg.node_specs = [{"tiering_mode": "transparent", "dram": "1MB", "ssd": "4MB"},
                      {"tiering_mode": "explicit", "dram": "1MB", "ssd": "4MB"}]
    env = {"TAIR_MEMPOOL_GA_CANDIDATE_SHUFFLE": "0"}
    app = mp.MockPace(cfg, env=env)
    mp.build_nodes(app, cfg)
    server = mp.Server((cfg.host, 0), app)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % server.server_address[1]
    auth = "Basic " + base64.b64encode(b"admin:123").decode()

    def call(method, path, payload=None, with_auth=True, raw=False):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(base + path, data=data, method=method)
        if with_auth:
            req.add_header("Authorization", auth)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                text = resp.read()
                return resp.status, text.decode() if raw else json.loads(text or b"{}")
        except urllib.error.HTTPError as exc:
            text = exc.read()
            if raw or exc.headers.get("Content-Type", "").startswith("text/"):
                return exc.code, text.decode()
            return exc.code, json.loads(text or b"{}")

    try:
        kib = 1024
        code, r = call("GET", "/v1/api/healthy", with_auth=False)
        check("healthy", code == 200 and r == {"status": "success"}, r)
        code, r = call("GET", "/v1/api/memnode", with_auth=False)
        check("WithAuth missing header -> 401 text", code == 401 and r == "Missing Authorization header\n", (code, r))
        code, r = call("GET", "/v1/api/memnode")
        check("memnode shape", code == 200 and set(r) == {"memnodes", "mem_usage_stat", "status"}
              and r["memnodes"]["1"]["status"] == mp.NODE_STATUS_HEALTHY
              and r["memnodes"]["2"]["tiering_mode"] == "explicit"
              and set(r["memnodes"]["1"]["segment"][0]) == {"segment_id", "virt_segment_id", "total_bytes",
                                                             "used_bytes", "alloced_ga_num", "is_active", "media_type"}, r)

        code, r = call("POST", "/v1/api/gas/batch", {"size": 128 * kib, "count": 1, "media_type": 0})
        check("media_type=0 -> 400 legacy media_type must be 2 or 5",
              code == 400 and r == {"status": "failed", "data": "legacy media_type must be 2 or 5"}, (code, r))
        code, r = call("POST", "/v1/api/gas/batch", {"size": 128 * kib, "count": 4, "media_policy": "only_dram"})
        dram_gas = [d["offset"] for d in r.get("data", [])]
        check("only_dram served by explicit node only", code == 200 and len(dram_gas) == 4
              and all(d["node_id"] == 2 and d["media_type"] == 2 for d in r["data"]), r)
        code, r = call("POST", "/v1/api/gas/batch", {"size": 512 * kib, "count": 2, "media_type": 5})
        ssd_gas = [d["offset"] for d in r.get("data", [])]
        check("media_type=5 -> only_ssd on explicit node", code == 200
              and all(d["media_type"] == 5 and d["node_id"] == 2 for d in r.get("data", [])), r)
        code, r = call("POST", "/v1/api/gas/batch", {"size": 128 * kib, "count": 1, "media_type": 2})
        check("legacy media_type=2 is AUTO; transparent node answers media_type=0",
              code == 200 and r["data"][0]["node_id"] == 1 and r["data"][0]["media_type"] == 0, r)
        auto_ga = r["data"][0]["offset"]
        code, r = call("POST", "/v1/api/gas", {"size": 64 * kib, "preferred_nodes": ["2"], "prefer_policy": "strict"})
        check("single AUTO on explicit node answers media_type=2", code == 200 and isinstance(r["data"], dict)
              and r["data"]["node_id"] == 2 and r["data"]["media_type"] == 2, r)
        single_ga = r["data"]["offset"]

        n1 = app.meta.provider(1)       # transparent Provider; tiering counters live on it
        mib = 1024 * kib
        dram_used = lambda: n1.tiers[mp.MEDIA_DRAM].used  # noqa: E731
        code, r = call("POST", "/v1/api/gas/batch", {"size": 128 * kib, "count": 6, "preferred_nodes": ["1"],
                                                      "prefer_policy": "strict"})
        check("no demote inside alloc request (Meta never demotes)", code == 200
              and n1.stats["demoted_gas"] == 0 and dram_used() == 7 * 128 * kib, (code, r, dram_used()))
        code, r = call("POST", "/v1/api/gas/batch", {"size": 128 * kib, "count": 1, "preferred_nodes": ["1"],
                                                      "prefer_policy": "strict"})
        check("dram_gate: DRAM above capacity watermark excludes node", code == 500, (code, r))
        app.tiering_tick_all()
        check("WaterMarkLoop: pct>=85 demotes coldest until pct<70",
              n1.stats["demoted_gas"] == 2 and dram_used() == 5 * 128 * kib
              and n1.tiers[mp.MEDIA_SSD].used == 2 * 128 * kib, (dict(n1.stats), dram_used()))
        code, r = call("GET", "/admin/ga?ga=%d" % auto_ga, with_auth=False)
        check("coldest (oldest) GA demoted first, address unchanged",
              r["gas"][str(auto_ga)]["resident_media_type"] == 5
              and r["gas"][str(auto_ga)]["segment_media_type"] == 2, r)
        code, mem = call("GET", "/v1/api/memnode")
        check("memnode after demote: used/alloced_ga_num count DRAM only, ssd_used grows",
              mem["memnodes"]["1"]["used_bytes"] == 5 * 128 * kib and mem["memnodes"]["1"]["alloced_ga_num"] == 5
              and mem["memnodes"]["1"]["ssd_used_bytes"] == 2 * 128 * kib, mem["memnodes"]["1"])
        app.tiering_tick_all()
        check("below high watermark: no further demote", n1.stats["demoted_gas"] == 2, dict(n1.stats))

        call("POST", "/v1/api/admin/ga_alloc_config", {"dram_gate": 0})
        code, r = call("POST", "/v1/api/gas/batch", {"size": 128 * kib, "count": 4, "preferred_nodes": ["1"],
                                                      "prefer_policy": "strict"})
        check("dram_gate=0: AdmitByWatermark spills to SSD, SyncDemoteForAlloc frees DRAM",
              code == 200 and n1.stats["sync_demote_calls"] >= 1 and n1.stats["demoted_gas"] == 10
              and dram_used() == 128 * kib, (code, r, dict(n1.stats), dram_used()))
        call("POST", "/v1/api/admin/ga_alloc_config", {"dram_gate": 1, "ssd_alloc_max_ratio_pct": 100})

        call("POST", "/admin/node", {"action": "update", "pace_node_id": 1, "ssd": n1.tiers[mp.MEDIA_SSD].used},
             with_auth=False)
        code, r = call("POST", "/v1/api/gas/batch", {"size": 128 * kib, "count": 6, "preferred_nodes": ["1"],
                                                      "prefer_policy": "strict"})
        fails_before = n1.stats["demote_failed"]
        for _ in range(11):
            app.tiering_tick_all()
        check("SSD full: demote write fails, 10 consecutive fail ticks -> backoff 300",
              code == 200 and n1.stats["demote_failed"] > fails_before and n1.demote_backoff_skips == 300
              and n1.stats["ssd_full_backoff"] == 1, (code, r, dict(n1.stats), n1.demote_backoff_skips))
        app.tiering_tick_all()
        check("backoff tick skips demote", n1.demote_backoff_skips == 299, n1.demote_backoff_skips)
        rc = n1.sync_demote_for_alloc(128 * kib, fail_fast=True)
        check("SyncDemoteForAlloc fail_fast during backoff -> ERR_NO_MEM", rc == "ERR_NO_MEM", rc)
        code, r = call("DELETE", "/v1/api/gas?ga=%d" % auto_ga)
        check("freeing an SSD GA -> NotifySsdFreed clears backoff", code == 200 and n1.demote_backoff_skips == 0,
              n1.demote_backoff_skips)
        call("POST", "/v1/api/admin/ga_alloc_config/reset")

        call("POST", "/admin/node", {"action": "update", "pace_node_id": 1, "ssd": n1.tiers[mp.MEDIA_SSD].used},
             with_auth=False)
        code, r = call("POST", "/v1/api/gas/batch", {"size": 64 * kib, "count": 1, "preferred_nodes": ["1"],
                                                      "prefer_policy": "strict"})
        check("ssd_alloc_max_ratio blocks AUTO on transparent node", code == 500 and r["status"] == "failed", r)
        code, r = call("POST", "/v1/api/gas/batch", {"size": 64 * kib, "count": 1, "preferred_nodes": ["1"]})
        check("best_effort falls back to explicit node", code == 200 and r["data"][0]["node_id"] == 2, r)
        call("POST", "/admin/node", {"action": "update", "pace_node_id": 1, "ssd": 4 * mib}, with_auth=False)
        call("DELETE", "/v1/api/gas?id=1")

        before = app.meta.ga_count()
        code, r = call("POST", "/v1/api/gas/batch", {"size": 1024 * kib, "count": 4, "media_type": 5})
        check("insufficient batch -> 500 'Create GA failed' and rollback",
              code == 500 and r == {"status": "failed", "data": "Create GA failed"} and app.meta.ga_count() == before,
              (code, r, before, app.meta.ga_count()))
        code, r = call("POST", "/v1/api/gas/batch", {"size": 1024 * kib, "count": 4, "media_type": 5,
                                                      "preferred_nodes": ["2"], "prefer_policy": "strict"})
        check("strict oversubscribe message", code == 500 and r["data"].startswith("oversubscribe:"), r)

        code, r = call("POST", "/v1/api/copy_ga/batch_sync", {"copies": [
            {"src_ga": dram_gas[0], "dst_ga": ssd_gas[0], "size": 128 * kib, "src_media_type": 2,
             "dst_media_type": 5},
            {"src_ga": dram_gas[1], "dst_ga": ssd_gas[1] + 99, "size": 128 * kib}]})
        tasks = r.get("data", {}).get("tasks", [])
        check("copy batch_sync shape", code == 200 and r["status"] == "success" and len(tasks) == 2
              and tasks[0]["state"] == "success" and tasks[1]["state"] == "failed"
              and tasks[1]["provider_http_status"] == 500 and r["data"]["summary"]["failed"] == 1
              and r["data"]["outcome"] == "failed" and "safe_to_reuse_dst" in tasks[0], r)
        code, r = call("POST", "/v1/api/copy_ga/batch_sync", {"copies": [{"src_ga": 12345, "dst_ga": 1, "size": 1}]})
        check("copy unknown segment -> 400 node not found", code == 400 and r["data"].startswith("node not found"), r)
        code, r = call("POST", "/v1/api/copy_ga/sync", {"src_ga": dram_gas[0], "dst_ga": ssd_gas[0],
                                                         "size": 128 * kib, "dst_media_type": 2})
        check("copy sync media mismatch -> provider 400", code == 400 and r["status"] == "failed"
              and "media_type mismatch" in r["data"]["error"], (code, r))

        code, r = call("GET", "/v1/api/gas?ga=%d,%d" % (dram_gas[0], ssd_gas[0]))
        check("get gas shape", code == 200 and r[str(dram_gas[0])] == {"provider_id": 2, "status": "NORMAL",
                                                                        "addr": dram_gas[0]}, r)
        code, r = call("DELETE", "/v1/api/gas?ga=" + ",".join(str(g) for g in dram_gas + [single_ga, (1 << 40) + 7]))
        check("delete list -> {status: success}", code == 200 and r == {"status": "success"}
              and all(app.meta.describe_ga(g) is None for g in dram_gas), r)
        code, r = call("DELETE", "/v1/api/gas?id=999")
        check("delete by unknown node -> 400", code == 400, (code, r))
        code, r = call("DELETE", "/v1/api/gas")
        code2, mem = call("GET", "/v1/api/memnode")
        check("delete all", code == 200 and r["status"] == "success" and len(r["nodes"]) == 2
              and mem["mem_usage_stat"]["alloced_ga_num"] == 0, (r, mem))

        call("POST", "/admin/node", {"action": "update", "pace_node_id": 1, "healthy": False}, with_auth=False)
        code, r = call("POST", "/v1/api/gas/batch", {"size": 4 * kib, "count": 3})
        check("unhealthy node skipped", code == 200 and all(d["node_id"] == 2 for d in r["data"]), r)
        code, mem = call("GET", "/v1/api/memnode")
        check("unhealthy node status=CONNECT_FAILED", mem["memnodes"]["1"]["healthy"] is False
              and mem["memnodes"]["1"]["status"] == mp.NODE_STATUS_CONNECT_FAILED, mem["memnodes"]["1"])
        call("POST", "/admin/node", {"action": "update", "pace_node_id": 1, "healthy": True}, with_auth=False)

        code, r = call("GET", "/v1/api/admin/ga_alloc_config")
        check("ga_alloc_config GET view", code == 200 and r["config"]["candidate_shuffle"] == 0
              and r["sources"]["candidate_shuffle"] == "env" and r["limits"]["dram_gate"] ==
              {"min": 0, "max": 1, "env": "TAIR_MEMPOOL_GA_DRAM_GATE"} and r["persistence_supported"] is False, r)
        code, r = call("POST", "/v1/api/admin/ga_alloc_config",
                       {"patch": {"ssd_alloc_max_ratio_pct": 88}, "comment": "t"})
        check("ga_alloc_config POST patch", code == 200 and r["config"]["ssd_alloc_max_ratio_pct"] == 88
              and r["sources"]["ssd_alloc_max_ratio_pct"] == "runtime"
              and r["applied"] == {"ssd_alloc_max_ratio_pct": 88}
              and r["comment"] == "t" and r["persisted"] is False, r)
        code, r = call("POST", "/v1/api/admin/ga_alloc_config", {"ssd_alloc_max_ratio_pct": 101})
        check("ga_alloc_config range error", code == 400 and
              r["data"] == "invalid ssd_alloc_max_ratio_pct: 101 out of range [1, 100]", r)
        code, r = call("POST", "/v1/api/admin/ga_alloc_config", {"bogus": 1})
        check("ga_alloc_config unknown key", code == 400 and r["data"] == "unknown config key 'bogus'", r)
        code, r = call("POST", "/v1/api/admin/ga_alloc_config", {"inflight_adjust": False})
        check("ga_alloc_config multi-active guard", code == 400 and r["data"].startswith("inflight_adjust cannot be"), r)
        code, r = call("POST", "/v1/api/admin/ga_alloc_config/reset")
        check("ga_alloc_config reset", code == 200 and r["runtime_patch"] == {}
              and r["config"]["ssd_alloc_max_ratio_pct"] == 90, r)

        call("POST", "/admin/fault", {"endpoint": "gas_batch", "fail_ratio": 1.0, "fail_mode": "http"},
             with_auth=False)
        code, r = call("POST", "/v1/api/gas/batch", {"size": 4 * kib, "count": 1})
        check("fault injection http 500", code == 500, (code, r))
        call("DELETE", "/admin/fault", with_auth=False)
        code, r = call("POST", "/v1/api/nodes", {"x": 1})
        code2, stats = call("GET", "/admin/stats", with_auth=False)
        check("unimplemented route recorded", code == 404 and stats["unknown_requests"][-1]["path"] == "/v1/api/nodes",
              stats.get("unknown_requests"))
        long_list = ",".join(str((1 << 40) + i) for i in range(5000))
        code, r = call("DELETE", "/v1/api/gas?ga=" + long_list)
        check("long DELETE query accepted", code == 200, (code, r))
    finally:
        server.shutdown()
        server.server_close()

    print("\n%d check(s) failed" % len(failures) if failures else "\nall checks passed")
    return failures


def test_mock_pace_server():
    assert not run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    sys.exit(1 if run() else 0)
