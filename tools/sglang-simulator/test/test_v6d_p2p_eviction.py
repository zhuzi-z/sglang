#!/usr/bin/env python3
"""V6D P2P eviction test (memory_usage path).

Starts Redis + two v6d servers with C_VineyardServerHook (virtual 500GB
capacity, 1G blob size) and --memory-usage-max to trigger eviction
via memory_usage path. Verifies evicted objects are removed from both
local store and tracker.

Run:
  python test/test_v6d_p2p_eviction.py
"""

import json
import os
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
V6D_BIN = os.path.join(SIM_ROOT, "scripts", "v6d")
LIB_DIR = os.path.join(SIM_ROOT, "scripts")

REDIS_PORT = 6379
REDIS_URL = f"redis://127.0.0.1:{REDIS_PORT}"

# Virtual capacity: 500GB, blob size: 1GB (virtual)
# With --memory-usage-max=0.05, eviction triggers at ~5% = ~25 blobs
# With --memory-usage-emergency-min=0.08, force-wait at ~8% = ~41 blobs
USAGE_MAX = 0.05
USAGE_MIN = 0.02
USAGE_EMERGENCY = 0.08
BLOB_SIZE = 1 << 30       # 1 GB (virtual size per blob)
VIRTUAL_CAP = 500 * (1 << 30)  # 500 GB virtual capacity (from --vineyard-size=500G)
# Eviction triggers at: USAGE_MAX * VIRTUAL_CAP / BLOB_SIZE = ~26 blobs
EVICT_THRESHOLD = int(USAGE_MAX * VIRTUAL_CAP / BLOB_SIZE)
# Create enough objects to trigger eviction
N_FILL = EVICT_THRESHOLD + 10  # ~36 objects

V6D_A_PORT = 7890
V6D_B_PORT = 7891
V6D_A_RPC = 21000
V6D_B_RPC = 21001
V6D_A_SOCK = "/tmp/vineyard.sock_a"
V6D_B_SOCK = "/tmp/vineyard.sock_b"
URL_A = f"http://127.0.0.1:{V6D_A_PORT}"
URL_B = f"http://127.0.0.1:{V6D_B_PORT}"

V6D_ENV = {
    **os.environ,
    "LD_LIBRARY_PATH": LIB_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    "SGLANG_SIMULATOR_V6D_CAPACITY_CONTROL": "1",
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

def http_post(base: str, path: str, data: dict, timeout: float = 30.0):
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

def start_v6d(name, port, rpc_port, sock, peer_id):
    args = [
        V6D_BIN, "serve",
        "--peer=tiered_vineyard",
        "--vineyard-size=500G",
        "--memory-usage-max", str(USAGE_MAX),
        "--memory-usage-min", str(USAGE_MIN),
        "--memory-usage-emergency-min", str(USAGE_EMERGENCY),
        "--port", str(port),
        "--vineyard-socket", sock,
        "--vineyard-rpc-port", str(rpc_port),
        "--peer-id", peer_id,
        "--tracker-redis", REDIS_URL,
        "--tracker-ttl", "60",
        "--log-level", "debug",
    ]
    log_file = os.path.join(SIM_ROOT, "tmp.out", f"v6d_evict_{name}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_fp = open(log_file, "w")
    print(f"Starting v6d {name} (port {port}, usage_max={USAGE_MAX}, "
          f"evict_threshold={EVICT_THRESHOLD} blobs)...")
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

# --------------------------------------------------------------------------
# Object helpers
# --------------------------------------------------------------------------

def create_and_seal(base, key, size=BLOB_SIZE):
    """Create + seal + release an object. Returns (success, lease_id)."""
    s, body = http_post(base, "/acquire", {
        "scope": "create",
        "object_keys": [key],
        "object_metas": [{"size": size}],
        "term": 60,
    })
    if s != 200 or not body:
        return False, ""
    lease_id = body.get("lease", {}).get("lease_id", "")
    if not lease_id:
        return False, ""
    s, _ = http_post(base, "/seal", {
        "lease_id": lease_id,
        "seal_object_keys": [key],
    })
    if s != 204:
        return False, lease_id
    # Release lease so object becomes evictable
    http_post(base, "/release", {"lease_id": lease_id})
    return True, lease_id

def exists_on(base, key):
    """Check if object exists on a server."""
    s, body = http_post(base, "/exists", {"object_key": key})
    if s == 200 and body:
        return body.get("exists", False)
    return False

# --------------------------------------------------------------------------
# Main test
# --------------------------------------------------------------------------

def main():
    print("=== V6D P2P Eviction Test (memory_usage path) ===")
    print(f"Virtual: {VIRTUAL_CAP} bytes cap, {BLOB_SIZE} bytes/blob, "
          f"usage_max={USAGE_MAX}")
    print(f"Eviction threshold: {EVICT_THRESHOLD} blobs")
    print()

    redis_proc = None
    v6d_a = None
    v6d_b = None

    try:
        # Start Redis
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

        v6d_a = start_v6d("A", V6D_A_PORT, V6D_A_RPC, V6D_A_SOCK, "peer_a")
        print("  Waiting for A...", end=" ", flush=True)
        if not wait_for_v6d(URL_A, v6d_a, 30):
            print("FAILED")
            sys.exit(1)
        print("OK")

        v6d_b = start_v6d("B", V6D_B_PORT, V6D_B_RPC, V6D_B_SOCK, "peer_b")
        print("  Waiting for B...", end=" ", flush=True)
        if not wait_for_v6d(URL_B, v6d_b, 30):
            print("FAILED")
            sys.exit(1)
        print("OK")

        time.sleep(2)

        # ------------------------------------------------------------------
        # Phase 1: Fill A below eviction threshold
        # Use EVICT_THRESHOLD - 1 to stay safely under usage_max
        # (e.g. 24 blobs = 4.8% < 5% with 500G/1G)
        # ------------------------------------------------------------------
        n_phase1 = EVICT_THRESHOLD - 1
        print(f"\n--- Phase 1: Fill A with {n_phase1} objects "
              f"(below eviction threshold) ---")
        keys_phase1 = [f"evict_obj_{i:03d}" for i in range(n_phase1)]
        created = 0
        for key in keys_phase1:
            ok, _ = create_and_seal(URL_A, key)
            if ok:
                created += 1
        check("t1_fill_below_threshold", created == n_phase1,
              f"created={created}/{n_phase1}")

        # Verify all exist on A
        exist_count_a = sum(1 for k in keys_phase1 if exists_on(URL_A, k))
        check("t1b_all_exist_on_a", exist_count_a == n_phase1,
              f"exists={exist_count_a}/{n_phase1}")

        # Verify B can discover them via tracker
        time.sleep(1)
        exist_count_b = sum(1 for k in keys_phase1 if exists_on(URL_B, k))
        check("t1c_all_discoverable_on_b", exist_count_b == n_phase1,
              f"discoverable={exist_count_b}/{n_phase1}")

        # ------------------------------------------------------------------
        # Phase 2: Create 10 more objects to exceed memory_usage_max
        # ------------------------------------------------------------------
        print(f"\n--- Phase 2: Create 10 more objects "
              f"(exceed usage_max={USAGE_MAX}, trigger eviction) ---")
        keys_phase2 = [f"evict_obj_{i:03d}" for i in range(EVICT_THRESHOLD,
                                                            EVICT_THRESHOLD + 10)]
        created2 = 0
        for key in keys_phase2:
            ok, _ = create_and_seal(URL_A, key)
            if ok:
                created2 += 1
        check("t2_create_more", created2 == 10,
              f"created={created2}/10")

        # Give background evictor time to process
        print("  Waiting for background evictor...", end=" ", flush=True)
        time.sleep(5)
        print("done")

        # ------------------------------------------------------------------
        # Phase 3: Verify eviction happened
        # ------------------------------------------------------------------
        print(f"\n--- Phase 3: Verify eviction ---")

        # New objects should exist on A
        new_exist_a = sum(1 for k in keys_phase2 if exists_on(URL_A, k))
        check("t3a_new_objects_exist_on_a", new_exist_a >= 5,
              f"exists={new_exist_a}/10")

        # Some old objects should be evicted from A
        old_evicted = sum(1 for k in keys_phase1 if not exists_on(URL_A, k))
        check("t3b_old_objects_evicted", old_evicted >= 1,
              f"evicted={old_evicted}/{EVICT_THRESHOLD} (expected >=1)")

        # Evicted objects should be unannounced from tracker (B can't find them)
        time.sleep(1)
        old_evicted_on_b = sum(1 for k in keys_phase1
                                if not exists_on(URL_A, k) and not exists_on(URL_B, k))
        check("t3c_evicted_unannounced_from_tracker",
              old_evicted_on_b >= 1,
              f"unannounced={old_evicted_on_b} (expected >=1)")

        # New objects should be discoverable on B
        new_exist_b = sum(1 for k in keys_phase2 if exists_on(URL_B, k))
        check("t3d_new_objects_discoverable_on_b", new_exist_b >= 5,
              f"discoverable={new_exist_b}/10")

        # ------------------------------------------------------------------
        # Phase 4: Summary
        # ------------------------------------------------------------------
        print(f"\n--- Phase 4: Summary ---")
        all_keys = keys_phase1 + keys_phase2
        total_on_a = sum(1 for k in all_keys if exists_on(URL_A, k))
        evicted_keys = [k for k in keys_phase1 if not exists_on(URL_A, k)]
        surviving_keys = [k for k in keys_phase1 if exists_on(URL_A, k)]
        print(f"  Virtual capacity: {VIRTUAL_CAP} bytes, blob={BLOB_SIZE} bytes")
        print(f"  usage_max={USAGE_MAX} → evict at {EVICT_THRESHOLD} blobs")
        print(f"  Created: {len(keys_phase1)} + {len(keys_phase2)} = {len(all_keys)}")
        print(f"  Evicted from A: {len(evicted_keys)} objects")
        if evicted_keys:
            print(f"    e.g. {evicted_keys[:5]}")
        print(f"  Surviving on A: {len(surviving_keys)} old + {new_exist_a} new = "
              f"{len(surviving_keys) + new_exist_a}")
        check("t4_eviction_happened", len(evicted_keys) >= 1,
              f"evicted={len(evicted_keys)}")

    finally:
        print("\nCleaning up...")
        stop_proc(v6d_a)
        stop_proc(v6d_b)
        if redis_proc:
            stop_proc(redis_proc)
        for sock in (V6D_A_SOCK, V6D_B_SOCK):
            try:
                os.unlink(sock)
            except FileNotFoundError:
                pass

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
        # Print log excerpts
        for name, proc in [("A", v6d_a), ("B", v6d_b)]:
            if hasattr(proc, "_log_file"):
                try:
                    with open(proc._log_file) as f:
                        log = f.read()
                    evict_lines = [l for l in log.split("\n")
                                   if any(k in l.lower() for k in ["evict", "discard", "unannounce"])]
                    if evict_lines:
                        print(f"\n--- v6d {name} eviction logs ---")
                        for l in evict_lines[-10:]:
                            print(f"  {l}")
                except Exception:
                    pass
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
