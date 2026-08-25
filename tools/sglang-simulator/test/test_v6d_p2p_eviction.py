#!/usr/bin/env python3
"""V6D P2P eviction test (memory_usage path).

Starts Redis + two v6d servers with a REAL arena and --memory-usage-max to
trigger eviction via the memory_usage path, under two arena configs:
  - 512M / 2MB blobs   : fast mechanism check (eviction at ~12 blobs)
  - 500G / 132MB blobs : production capacity config (sparse memfd, pages
                         never touched; 132MB is the production page size,
                         eviction at ~193 blobs)
Verifies evicted objects are removed from both local store and tracker.

Run:
  python test/test_v6d_p2p_eviction.py
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V6D_BIN = shutil.which("v6d") or "/usr/local/bin/v6d"

# Dedicated Redis so FLUSHDB cannot disturb a running benchmark stack.
REDIS_PORT = 6390
REDIS_URL = f"redis://127.0.0.1:{REDIS_PORT}"

USAGE_MAX = 0.96
USAGE_MIN = 0.90
USAGE_EMERGENCY = 0.98

# AsyncEvictor semantics (peer.py): the 5s periodic sweep evicts whenever
# usage > min — it does NOT consult max; max only gates the request-path
# signal.  So the test fills to 89% (<= min, sweep-safe) in phase 1 and
# pushes past max (96%) in phase 2.
FILL_FRACT = 0.89        # phase 1: below min -> periodic sweep inert
PHASE2_SPAN_FRACT = 0.08   # phase 2: 89% -> 97% > max -> signal + sweep

CONFIGS = [
    # Fast mechanism check: 512M arena, 2MB blob = exactly one 2M arena
    # block (the C++ allocator reserves one block, so 255 blobs fit).
    dict(name="512M", arena_size="512M", blob_size=2 << 20),
    # Production capacity: 500G sparse arena, production-size 132MB pages.
    dict(name="500G", arena_size="500G", blob_size=132 << 20),
]

# Ports away from the benchmark stack (7890/7891, 21001/21002).
V6D_A_PORT = 7790
V6D_B_PORT = 7791
V6D_A_RPC = 21100
V6D_B_RPC = 21101
V6D_A_SOCK = "/tmp/vineyard.sock_pt_a"
V6D_B_SOCK = "/tmp/vineyard.sock_pt_b"
URL_A = f"http://127.0.0.1:{V6D_A_PORT}"
URL_B = f"http://127.0.0.1:{V6D_B_PORT}"

V6D_ENV = {
    **os.environ,
    # Sim daemon hooks: reserve_memory=false argv swap (mandatory for the
    # 500G sparse arena) + HitSource logging + load_data no-op.
    "SGLANG_SIMULATOR_ENABLE": "1",
    "SRPC_STREAM_DISABLE_RDMA": "1",
}

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
_results: list[tuple[str, bool, str]] = []

def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f": {detail}"
    print(msg)
    _results.append((name, cond, detail))

# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------

def http_get(base: str, path: str, timeout: float = 5.0):
    try:
        with urllib.request.urlopen(f"{base}{path}", timeout=timeout) as resp:
            body = resp.read()
            return resp.status, json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body}
    except Exception as e:
        return -1, {"error": str(e)}

def http_post(base: str, path: str, data: dict, timeout: float = 120.0):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{base}{path}", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_body = resp.read()
            return resp.status, json.loads(resp_body) if resp_body else None
    except urllib.error.HTTPError as e:
        err_body = e.read().decode() if e.fp else ""
        try:
            return e.code, json.loads(err_body)
        except Exception:
            return e.code, {"raw": err_body}
    except Exception as e:
        return -1, {"error": str(e)}

# --------------------------------------------------------------------------
# Process management
# --------------------------------------------------------------------------

def ensure_redis():
    """Ensure Redis is running."""
    try:
        r = subprocess.run(["redis-cli", "-p", str(REDIS_PORT), "ping"],
                           capture_output=True, text=True, timeout=2)
        if r.stdout.strip() == "PONG":
            print("Redis already running")
            return None
    except Exception:
        pass
    print("Starting Redis...", end=" ", flush=True)
    proc = subprocess.Popen(
        ["redis-server", "--port", str(REDIS_PORT), "--save", "",
         "--appendonly", "no", "--daemonize", "no", "--loglevel", "warning"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    deadline = time.time() + 5
    while time.time() < deadline:
        if proc.poll() is not None:
            print("FAILED")
            return proc
        try:
            r = subprocess.run(["redis-cli", "-p", str(REDIS_PORT), "ping"],
                               capture_output=True, text=True, timeout=2)
            if r.stdout.strip() == "PONG":
                print("OK")
                return proc
        except Exception:
            pass
        time.sleep(0.3)
    print("FAILED (timeout)")
    return proc

def start_v6d(cfg, name, port, rpc_port, sock, peer_id):
    args = [
        V6D_BIN, "serve",
        "--peer=tiered_vineyard",
        f"--vineyard-size={cfg['arena_size']}",
        "--memory-usage-max", str(USAGE_MAX),
        "--memory-usage-min", str(USAGE_MIN),
        "--memory-usage-emergency-min", str(USAGE_EMERGENCY),
        "--port", str(port),
        "--vineyard-socket", sock,
        "--vineyard-rpc-port", str(rpc_port),
        "--peer-id", peer_id,
        "--tracker-redis", REDIS_URL,
        "--tracker-ttl", "300",
        "--log-level", "debug",
    ]
    log_file = os.path.join(
        SIM_ROOT, "tmp.out", f"v6d_evict_{cfg['name']}_{name}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_fp = open(log_file, "w")
    print(f"Starting v6d {name} ({cfg['name']}, port {port}, "
          f"min={USAGE_MIN} max={USAGE_MAX})...")
    proc = subprocess.Popen(args, env=V6D_ENV, stdout=log_fp, stderr=subprocess.STDOUT)
    proc._log_fp = log_fp
    proc._log_file = log_file
    return proc

def wait_for_v6d(base_url, proc, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        status, _ = http_get(base_url, "/health", timeout=2)
        if status == 200:
            return True
        time.sleep(0.5)
    return False

def stop_proc(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
    if hasattr(proc, "_log_fp") and proc._log_fp:
        proc._log_fp.close()

def daemon_rss_mb(proc) -> float:
    try:
        with open(f"/proc/{proc.pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0

# --------------------------------------------------------------------------
# Object helpers
# --------------------------------------------------------------------------

def create_batch_and_seal(base, keys, size):
    """Create + seal + release a batch of objects in one acquire."""
    s, body = http_post(base, "/acquire", {
        "scope": "create",
        "object_keys": keys,
        "object_metas": [{"size": size} for _ in keys],
        "term": 60,
    })
    if s != 200 or not body:
        return False, ""
    lease_id = body.get("lease", {}).get("lease_id", "")
    if not lease_id:
        return False, ""
    s, _ = http_post(base, "/seal", {
        "lease_id": lease_id,
        "seal_object_keys": keys,
    })
    if s != 204:
        return False, lease_id
    http_post(base, "/release", {"lease_id": lease_id})
    return True, lease_id

def exists_on(base, key):
    """Check if object exists on a server."""
    s, body = http_post(base, "/exists", {"object_key": key})
    if s == 200 and body:
        return body.get("exists", False)
    return False

def _sample(keys, n=64):
    """Evenly sample keys so exists scans stay cheap at 500G scale."""
    if len(keys) <= n:
        return keys
    step = len(keys) / n
    return [keys[int(i * step)] for i in range(n)]

# --------------------------------------------------------------------------
# Per-config test
# --------------------------------------------------------------------------

def run_config(cfg):
    tag = cfg["name"]
    blob = cfg["blob_size"]
    arena = cfg["arena_bytes"]
    # Byte-basis counts: usage is computed against the arena size.
    n_phase1 = int(FILL_FRACT * arena / blob)
    n_phase2 = int(PHASE2_SPAN_FRACT * arena / blob)
    assert n_phase1 + n_phase2 <= arena // blob - 1

    redis_proc = None
    v6d_a = None
    v6d_b = None

    try:
        redis_proc = ensure_redis()
        if redis_proc is not None and redis_proc.poll() is not None:
            print("Redis failed to start")
            sys.exit(1)

        # Flush Redis to start clean
        subprocess.run(["redis-cli", "-p", str(REDIS_PORT), "FLUSHDB"],
                       capture_output=True, timeout=2)

        # Start v6d servers
        for sock in (V6D_A_SOCK, V6D_B_SOCK):
            try:
                os.unlink(sock)
            except FileNotFoundError:
                pass

        v6d_a = start_v6d(cfg, "A", V6D_A_PORT, V6D_A_RPC, V6D_A_SOCK, "peer_a")
        print("  Waiting for A...", end=" ", flush=True)
        if not wait_for_v6d(URL_A, v6d_a, 30):
            print("FAILED")
            check(f"{tag}_t0_server_a_start", False, "v6d A failed to start")
            return
        print("OK")

        v6d_b = start_v6d(cfg, "B", V6D_B_PORT, V6D_B_RPC, V6D_B_SOCK, "peer_b")
        print("  Waiting for B...", end=" ", flush=True)
        if not wait_for_v6d(URL_B, v6d_b, 30):
            print("FAILED")
            check(f"{tag}_t0_server_b_start", False, "v6d B failed to start")
            return
        print("OK")

        time.sleep(2)

        # ------------------------------------------------------------------
        # Phase 1: Fill A to 89% (<= min: the 5s periodic sweep stays inert)
        # ------------------------------------------------------------------
        print(f"\n--- Phase 1 [{tag}]: Fill A with {n_phase1} objects "
              f"({FILL_FRACT*100:.0f}%, below sweep min={USAGE_MIN}) ---")
        keys_phase1 = [f"evict_obj_{i:05d}" for i in range(n_phase1)]
        created = 0
        for i in range(0, n_phase1, 256):
            batch = keys_phase1[i:i+256]
            ok, _ = create_batch_and_seal(URL_A, batch, blob)
            if ok:
                created += len(batch)
            else:
                print(f"  batch {i//256} FAILED ({len(batch)} keys)")
        check(f"{tag}_t1_fill_below_min", created == n_phase1,
              f"created={created}/{n_phase1}")

        # Verify all exist on A (sampled)
        sample1 = _sample(keys_phase1)
        exist_count_a = sum(1 for k in sample1 if exists_on(URL_A, k))
        check(f"{tag}_t1b_all_exist_on_a", exist_count_a == len(sample1),
              f"exists={exist_count_a}/{len(sample1)} (sampled)")

        # Verify B can discover them via tracker (sampled)
        time.sleep(1)
        exist_count_b = sum(1 for k in sample1 if exists_on(URL_B, k))
        check(f"{tag}_t1c_all_discoverable_on_b",
              exist_count_b == len(sample1),
              f"discoverable={exist_count_b}/{len(sample1)} (sampled)")

        # ------------------------------------------------------------------
        # Phase 2: Push usage to 97% (> max=96%): request path signals the
        # evictor, which evicts down to min=90% (LRU: oldest first).
        # ------------------------------------------------------------------
        print(f"\n--- Phase 2 [{tag}]: Create {n_phase2} more objects "
              f"(exceed usage_max={USAGE_MAX}, trigger eviction) ---")
        keys_phase2 = [f"evict_new_{i:05d}" for i in range(n_phase2)]
        created2 = 0
        for i in range(0, n_phase2, 256):
            batch = keys_phase2[i:i+256]
            ok, _ = create_batch_and_seal(URL_A, batch, blob)
            if ok:
                created2 += len(batch)
        check(f"{tag}_t2_create_more", created2 == n_phase2,
              f"created={created2}/{n_phase2}")

        # Give the signalled evictor time to finish its cycle
        print("  Waiting for evictor cycle...", end=" ", flush=True)
        time.sleep(4)
        print("done")

        # ------------------------------------------------------------------
        # Phase 3: Verify eviction happened
        # ------------------------------------------------------------------
        print(f"\n--- Phase 3 [{tag}]: Verify eviction ---")

        # New objects should exist on A (LRU evicts the oldest first;
        # sampled — phase 2 can be hundreds of keys at 500G)
        sample2 = _sample(keys_phase2, 32)
        new_exist_a = sum(1 for k in sample2 if exists_on(URL_A, k))
        check(f"{tag}_t3a_new_objects_exist_on_a",
              new_exist_a >= len(sample2) // 2,
              f"exists={new_exist_a}/{len(sample2)} (sampled)")

        # Some old objects should be evicted from A (sampled)
        old_evicted = sum(1 for k in sample1 if not exists_on(URL_A, k))
        check(f"{tag}_t3b_old_objects_evicted", old_evicted >= 1,
              f"evicted={old_evicted}/{len(sample1)} (sampled, expected >=1)")

        # Evicted objects should be unannounced from tracker (B can't find)
        time.sleep(1)
        old_evicted_on_b = sum(1 for k in sample1
                               if not exists_on(URL_A, k)
                               and not exists_on(URL_B, k))
        check(f"{tag}_t3c_evicted_unannounced_from_tracker",
              old_evicted_on_b >= 1,
              f"unannounced={old_evicted_on_b} (expected >=1)")

        # New objects should be discoverable on B
        new_exist_b = sum(1 for k in sample2 if exists_on(URL_B, k))
        check(f"{tag}_t3d_new_objects_discoverable_on_b",
              new_exist_b >= len(sample2) // 2,
              f"discoverable={new_exist_b}/{len(sample2)} (sampled)")

        # Sparse-arena proof: physical RSS stays flat at 500G
        if cfg["name"] == "500G":
            rss_a = daemon_rss_mb(v6d_a)
            rss_b = daemon_rss_mb(v6d_b)
            check(f"{tag}_t3e_sparse_rss",
                  0 < rss_a < 2048 and 0 < rss_b < 2048,
                  f"daemon RSS A={rss_a:.0f}MB B={rss_b:.0f}MB with "
                  f"500G arenas (sparse memfd)")

        # ------------------------------------------------------------------
        # Phase 4: Summary
        # ------------------------------------------------------------------
        print(f"\n--- Phase 4 [{tag}]: Summary ---")
        evicted_keys = [k for k in sample1 if not exists_on(URL_A, k)]
        surviving_keys = [k for k in sample1 if exists_on(URL_A, k)]
        print(f"  Real arena: {cfg['arena_size']}, blob={blob} bytes")
        print(f"  min={USAGE_MIN} max={USAGE_MAX}: fill {FILL_FRACT*100:.0f}% "
              f"then push to ~{100*(FILL_FRACT+PHASE2_SPAN_FRACT):.0f}%")
        print(f"  Created: {len(keys_phase1)} + {len(keys_phase2)} "
              f"= {n_phase1 + n_phase2}")
        print(f"  Evicted from A (sampled): {len(evicted_keys)} objects")
        if evicted_keys:
            print(f"    e.g. {evicted_keys[:5]}")
        print(f"  Surviving on A (sampled): {len(surviving_keys)} old + "
              f"{new_exist_a} new")
        check(f"{tag}_t4_eviction_happened", len(evicted_keys) >= 1,
              f"evicted={len(evicted_keys)} (sampled)")

    finally:
        print("\nCleaning up daemons...")
        stop_proc(v6d_a)
        stop_proc(v6d_b)
        if redis_proc:
            stop_proc(redis_proc)
        for sock in (V6D_A_SOCK, V6D_B_SOCK):
            try:
                os.unlink(sock)
            except FileNotFoundError:
                pass

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _arena_bytes(s: str) -> int:
    if s.endswith("G"):
        return int(s[:-1]) << 30
    return int(s[:-1]) << 20

def main():
    print("=== V6D P2P Eviction Test (memory_usage path) ===")
    for cfg in CONFIGS:
        cfg["arena_bytes"] = _arena_bytes(cfg["arena_size"])
        cfg["capacity_blobs"] = cfg["arena_bytes"] // cfg["blob_size"] - 1
        print(f"config {cfg['name']}: arena={cfg['arena_size']} "
              f"blob={cfg['blob_size']} capacity={cfg['capacity_blobs']} "
              f"phase1={int(FILL_FRACT*cfg['arena_bytes']/cfg['blob_size'])} "
              f"phase2={int(PHASE2_SPAN_FRACT*cfg['arena_bytes']/cfg['blob_size'])}")
    print()

    for cfg in CONFIGS:
        print(f"\n########## Config: {cfg['name']} ##########")
        run_config(cfg)

    # Summary
    print(f"\n=== Summary ===")
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    total = len(_results)
    print(f"Passed: {passed}/{total}, Failed: {failed}")

    if failed > 0:
        print("\nFailed tests:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
