#!/usr/bin/env python3
"""V6D emergency eviction test.

Tests two emergency eviction paths with C_VineyardServerHook:
  1. Pre-create blocking eviction (USAGE_EMERGENCY threshold)
     - future_usage >= critical → wait_complete() blocks create
  2. OOM-triggered emergency eviction
     - create_blobs raises NotEnoughMemoryException → _trigger_emergency_eviction → retry

Run:
  python test/test_v6d_emergency_eviction.py
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

BLOB_SIZE = 2 << 20       # 2 MB
VIRTUAL_CAP = 1 << 30      # 1 GB

V6D_PORT = 7890
V6D_RPC = 21000
V6D_SOCK = "/tmp/vineyard.sock_emg"
URL_A = f"http://127.0.0.1:{V6D_PORT}"

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

def start_v6d(usage_max, usage_min, usage_emergency, log_suffix):
    """Start a v6d server with given memory_usage thresholds."""
    try:
        os.unlink(V6D_SOCK)
    except FileNotFoundError:
        pass
    args = [
        V6D_BIN, "serve",
        "--peer=tiered_vineyard",
        "--vineyard-size=256M",
        "--memory-usage-max", str(usage_max),
        "--memory-usage-min", str(usage_min),
        "--memory-usage-emergency-min", str(usage_emergency),
        "--port", str(V6D_PORT),
        "--vineyard-socket", V6D_SOCK,
        "--vineyard-rpc-port", str(V6D_RPC),
        "--peer-id", "peer_emg",
        "--tracker-redis", REDIS_URL,
        "--tracker-ttl", "60",
        "--log-level", "debug",
    ]
    log_file = os.path.join(SIM_ROOT, "tmp.out", f"v6d_emg_{log_suffix}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_fp = open(log_file, "w")
    print(f"Starting v6d (max={usage_max}, emergency={usage_emergency})...")
    proc = subprocess.Popen(args, env=V6D_ENV, stdout=log_fp, stderr=subprocess.STDOUT)
    proc._log_fp = log_fp
    proc._log_file = log_file
    return proc

def wait_for_v6d(proc, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        status, _ = http_get(URL_A, "/health", timeout=2)
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

def flush_redis():
    subprocess.run(["redis-cli", "-p", str(REDIS_PORT), "FLUSHDB"],
                   capture_output=True, timeout=2)

def grep_log(proc, pattern):
    """Return list of matching lines from the v6d log."""
    if not hasattr(proc, "_log_file"):
        return []
    try:
        with open(proc._log_file) as f:
            return [l.strip() for l in f if pattern in l]
    except Exception:
        return []

# --------------------------------------------------------------------------
# Object helpers
# --------------------------------------------------------------------------

def create_and_seal(base, key, size=1024):
    """Create + seal + release. Returns (success, lease_id)."""
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
    http_post(base, "/release", {"lease_id": lease_id})
    return True, lease_id

def create_batch_and_seal(base, keys, size=1024):
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
    s, body = http_post(base, "/exists", {"object_key": key})
    if s == 200 and body:
        return body.get("exists", False)
    return False

# --------------------------------------------------------------------------
# Test 1: Pre-create blocking eviction (USAGE_EMERGENCY threshold)
# --------------------------------------------------------------------------

def test_blocking_eviction(redis_proc):
    """Test that exceeding USAGE_EMERGENCY triggers blocking eviction.

    With max=3%, emergency=4%, min=1%:
    - Create 21 objects in ONE batch (jumps past both max and emergency)
      The pre-create check sees future_usage < max (needed_bytes are real
      sizes ~1024B, not virtual 2MB), so no eviction signal.
    - Create 1 more → used=21*2MB=4.1% >= emergency(4%) → [EVICT_FORCE]
    - Blocking eviction completes before create returns
    """
    print("\n=== Test 1: Blocking Eviction (USAGE_EMERGENCY) ===")
    # memory_usage_critical is hardcoded at 0.95 (no CLI flag).
    # USAGE_MAX must be <= critical so the pre-create check doesn't
    # return early before reaching the critical check.
    # Create 487 blobs (95.1%) in one batch → no eviction signal
    # (needed_bytes uses real sizes ~1024B, not virtual 2MB).
    # Create 1 more → used=95.1% >= critical(95%) → [EVICT_FORCE]
    USAGE_MAX = 0.94
    USAGE_MIN = 0.50
    USAGE_EMERGENCY = 0.60
    critical_blobs = int(0.95 * VIRTUAL_CAP / BLOB_SIZE)  # 486

    v6d = None
    try:
        flush_redis()
        v6d = start_v6d(USAGE_MAX, USAGE_MIN, USAGE_EMERGENCY, "blocking")
        print("  Waiting for v6d...", end=" ", flush=True)
        if not wait_for_v6d(v6d, 30):
            print("FAILED")
            check("t1_server_start", False, "v6d failed to start")
            return
        print("OK")
        time.sleep(1)

        # Phase A: Create critical_blobs+1 objects in ONE batch
        # Pre-create check uses real sizes (1024B), so future_usage stays
        # near 0% → no eviction signal → evictor doesn't run
        n_batch = critical_blobs + 1  # 487 blobs = 95.1%
        print(f"  Phase A: Create {n_batch} objects in one batch "
              f"({n_batch*BLOB_SIZE*100/VIRTUAL_CAP:.1f}%)...")
        keys_a = [f"blk_obj_{i:03d}" for i in range(n_batch)]
        ok, _ = create_batch_and_seal(URL_A, keys_a)
        check("t1a_batch_create", ok, f"created={ok}")

        # All should exist (evictor hasn't run yet)
        exist_a = sum(1 for k in keys_a if exists_on(URL_A, k))
        check("t1b_all_exist_before_trigger", exist_a == n_batch,
              f"exists={exist_a}/{n_batch}")

        # Phase B: Create 1 more → used=4.1% >= emergency(4%) → [EVICT_FORCE]
        print(f"  Phase B: Create 1 more (used={n_batch*BLOB_SIZE*100/VIRTUAL_CAP:.1f}% "
              f">= emergency={USAGE_EMERGENCY*100:.0f}%)...")
        ok, _ = create_and_seal(URL_A, "blk_trigger")
        check("t1c_trigger_create", ok, f"success={ok}")

        # KEY TEST: Check immediately (no 5s wait) that eviction happened
        # The blocking eviction should have already completed
        time.sleep(0.5)
        evicted_immediately = sum(1 for k in keys_a if not exists_on(URL_A, k))
        check("t1d_evicted_immediately", evicted_immediately >= 1,
              f"evicted={evicted_immediately} (blocking eviction should be immediate)")

        # Trigger object should exist
        check("t1e_trigger_exists", exists_on(URL_A, "blk_trigger"),
              "trigger object should exist")

        # Verify [EVICT_FORCE] in logs (blocking path)
        force_logs = grep_log(v6d, "EVICT_FORCE")
        check("t1f_force_log_present", len(force_logs) >= 1,
              f"EVICT_FORCE lines={len(force_logs)}")

        # Summary
        all_keys = keys_a + ["blk_trigger"]
        total_on_a = sum(1 for k in all_keys if exists_on(URL_A, k))
        evicted = [k for k in keys_a if not exists_on(URL_A, k)]
        print(f"  Result: {len(evicted)} old evicted, {total_on_a} total on A")
        if force_logs:
            print(f"  Log: {force_logs[-1][:120]}")

    finally:
        stop_proc(v6d)

# --------------------------------------------------------------------------
# Test 2: OOM-triggered emergency eviction
# --------------------------------------------------------------------------

def test_oom_emergency_eviction(redis_proc):
    """Test that OOM triggers emergency eviction and retry succeeds.

    With max=1.0, emergency=1.1 (never triggers pre-create blocking):
    - Fill to 512 blobs (100% virtual capacity)
    - Create 1 more → create_blobs OOM → _trigger_emergency_eviction → retry
    - Verify [EMERGENCY] in logs and the create succeeds
    """
    print("\n=== Test 2: OOM-Triggered Emergency Eviction ===")
    # USAGE_MAX=2.0 → background evictor never triggers (usage < max)
    # USAGE_EMERGENCY=0.6 → emergency eviction target (evict to 60%)
    # This forces the OOM path: create_blobs raises NotEnoughMemoryException
    # → _trigger_emergency_eviction → evict to emergency_min → retry
    USAGE_MAX = 2.0       # background evictor never triggers (>100%)
    USAGE_MIN = 0.5       # normal target (not used during emergency)
    USAGE_EMERGENCY = 0.6 # emergency target: evict to 60% = 307 blobs
    max_blobs = VIRTUAL_CAP // BLOB_SIZE  # 512

    v6d = None
    try:
        flush_redis()
        v6d = start_v6d(USAGE_MAX, USAGE_MIN, USAGE_EMERGENCY, "oom")
        print("  Waiting for v6d...", end=" ", flush=True)
        if not wait_for_v6d(v6d, 30):
            print("FAILED")
            check("t2_server_start", False, "v6d failed to start")
            return
        print("OK")
        time.sleep(1)

        # Phase A: Fill to capacity (512 blobs = 100%)
        # Use batch creates for speed
        batch_size = 64
        print(f"  Phase A: Fill to {max_blobs} objects (100%)...")
        keys_fill = [f"oom_obj_{i:04d}" for i in range(max_blobs)]
        created = 0
        for i in range(0, max_blobs, batch_size):
            batch = keys_fill[i:i+batch_size]
            ok, _ = create_batch_and_seal(URL_A, batch)
            if ok:
                created += len(batch)
        check("t2a_fill_to_capacity", created == max_blobs,
              f"created={created}/{max_blobs}")

        # Verify all exist
        sample_keys = keys_fill[::50]  # check every 50th key
        exist_count = sum(1 for k in sample_keys if exists_on(URL_A, k))
        check("t2b_filled_objects_exist",
              exist_count == len(sample_keys),
              f"sample exists={exist_count}/{len(sample_keys)}")

        # Phase B: Create 1 more → OOM → emergency eviction → retry
        print(f"  Phase B: Create 1 more (should OOM → emergency → retry)...")
        t0 = time.time()
        ok, lease_id = create_and_seal(URL_A, "oom_extra_obj")
        elapsed = time.time() - t0
        check("t2c_create_after_oom_succeeds", ok,
              f"success={ok}, elapsed={elapsed:.1f}s")

        # Verify the extra object exists (retry succeeded)
        check("t2d_extra_object_exists", exists_on(URL_A, "oom_extra_obj"),
              "extra object should exist after emergency eviction + retry")

        # Verify [EMERGENCY] in logs
        emergency_logs = grep_log(v6d, "EMERGENCY")
        check("t2e_emergency_log_present", len(emergency_logs) >= 1,
              f"EMERGENCY lines={len(emergency_logs)}")

        # Verify [POST_EMERGENCY_EVICT] in logs
        post_emg_logs = grep_log(v6d, "POST_EMERGENCY_EVICT")
        check("t2f_post_emergency_log", len(post_emg_logs) >= 1,
              f"POST_EMERGENCY_EVICT lines={len(post_emg_logs)}")

        # Some old objects should have been evicted
        evicted_count = sum(1 for k in keys_fill if not exists_on(URL_A, k))
        check("t2g_old_objects_evicted", evicted_count >= 1,
              f"evicted={evicted_count}/{max_blobs}")

        # Summary
        print(f"  Result: {evicted_count} old objects evicted by emergency eviction")
        if emergency_logs:
            print(f"  Log: {emergency_logs[-1][:120]}")
        if post_emg_logs:
            print(f"  Log: {post_emg_logs[-1][:120]}")

    finally:
        stop_proc(v6d)

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print("=== V6D Emergency Eviction Test ===")
    print(f"Virtual: {VIRTUAL_CAP//(1<<20)}MB cap, {BLOB_SIZE//(1<<20)}MB/blob")
    print()

    redis_proc = None
    try:
        redis_proc = ensure_redis()
        if redis_proc is not None and redis_proc.poll() is not None:
            print("Redis failed to start")
            sys.exit(1)

        # Run both tests
        test_blocking_eviction(redis_proc)
        test_oom_emergency_eviction(redis_proc)

    finally:
        print("\nCleaning up...")
        if redis_proc:
            stop_proc(redis_proc)
        try:
            os.unlink(V6D_SOCK)
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
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
