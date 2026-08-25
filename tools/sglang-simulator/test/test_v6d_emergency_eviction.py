#!/usr/bin/env python3
"""V6D emergency eviction test.

Tests two emergency eviction paths on a REAL arena (no sim capacity hooks —
the daemon's own accounting drives eviction), under two arena configs:
  - 512M / 2MB blobs   : fast mechanism check (255-blob capacity)
  - 500G / 132MB blobs : production capacity config.  The 500G arena is a
                         sparse memfd (reserve_memory=false via the sim
                         daemon hook): blob pages are never touched, so
                         physical cost is metadata-only (~10KB/object) and
                         132MB is the production page size.
Paths covered:
  1. Pre-create blocking eviction (memory_usage_critical, hardcoded 0.98)
     - future_usage >= critical → [EVICT_FORCE] + wait_complete blocks create
  2. OOM-triggered emergency eviction
     - create_blobs raises NotEnoughMemoryException → emergency eviction → retry

Note the AsyncEvictor's 5s periodic sweep evicts whenever usage > min (it
does NOT consult max); the tests set thresholds accordingly.

Run:
  python test/test_v6d_emergency_eviction.py
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

CONFIGS = [
    # Fast mechanism check: small arena, one 2M block per blob.
    # The C++ allocator reserves one block (measured) -> 255 blobs fit.
    dict(name="512M", arena_size="512M", blob_size=2 << 20),
    # Production capacity: 500G sparse arena, production-size 132MB pages.
    dict(name="500G", arena_size="500G", blob_size=132 << 20),
]

# Ports away from the benchmark stack (7890/7891, 21001/21002).
V6D_PORT = 7790
V6D_RPC = 21100
V6D_SOCK = "/tmp/vineyard.sock_emg"
URL_A = f"http://127.0.0.1:{V6D_PORT}"

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

def start_v6d(cfg, usage_max, usage_min, usage_emergency, log_suffix):
    """Start a v6d server with given memory_usage thresholds."""
    try:
        os.unlink(V6D_SOCK)
    except FileNotFoundError:
        pass
    args = [
        V6D_BIN, "serve",
        "--peer=tiered_vineyard",
        f"--vineyard-size={cfg['arena_size']}",
        "--memory-usage-max", str(usage_max),
        "--memory-usage-min", str(usage_min),
        "--memory-usage-emergency-min", str(usage_emergency),
        "--port", str(V6D_PORT),
        "--vineyard-socket", V6D_SOCK,
        "--vineyard-rpc-port", str(V6D_RPC),
        "--peer-id", "peer_emg",
        "--tracker-redis", REDIS_URL,
        "--tracker-ttl", "300",
        "--log-level", "debug",
    ]
    log_file = os.path.join(
        SIM_ROOT, "tmp.out", f"v6d_emg_{cfg['name']}_{log_suffix}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_fp = open(log_file, "w")
    print(f"Starting v6d ({cfg['name']}, max={usage_max}, "
          f"emergency={usage_emergency})...")
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

def create_and_seal(base, key, size):
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
# Test 1: Pre-create blocking eviction (USAGE_EMERGENCY threshold)
# --------------------------------------------------------------------------

def test_blocking_eviction(cfg):
    """Test that exceeding memory_usage_critical triggers blocking eviction.

    Threshold semantics (peer.py): memory_usage_critical is hardcoded at
    0.98; the request path logs [EVICT_FORCE] and blocks on the evictor
    (wait_complete) when future_usage >= critical.  NOTE: the 5s periodic
    sweep evicts whenever usage > min (0.5 here) — it does NOT consult max,
    so checks right after phase A tolerate a sweep having run.

    With max=94%, min=50%:
    - Phase A: create ~98.03% in ONE batch -> [EVICT_FORCE] fires during the
      create (used=0 -> nothing to evict -> proceeds)
    - Phase B: create 1 more -> [EVICT_FORCE] + wait_complete -> the evictor
      cycle completes before the create returns
    """
    tag = cfg["name"]
    blob = cfg["blob_size"]
    arena_cap = cfg["arena_bytes"]
    capacity_blobs = arena_cap // blob - 1
    print(f"\n=== Test 1 [{tag}]: Blocking Eviction (critical threshold) ===")
    # memory_usage_critical is hardcoded at 0.98 (peer.py, no CLI flag).
    # USAGE_MAX must be < critical so phase A crosses max (SIGNAL) first
    # and then critical (FORCE) within the same batch.
    USAGE_MAX = 0.94
    USAGE_MIN = 0.50
    USAGE_EMERGENCY = 0.60
    critical = 0.98
    # Byte-basis: usage is computed against the arena size, not blob count.
    n_batch = int(critical * arena_cap / blob) + 1
    assert n_batch + 1 <= capacity_blobs, (
        f"{tag}: {n_batch}+1 blobs exceed capacity {capacity_blobs}")

    v6d = None
    try:
        flush_redis()
        v6d = start_v6d(cfg, USAGE_MAX, USAGE_MIN, USAGE_EMERGENCY, "blocking")
        print("  Waiting for v6d...", end=" ", flush=True)
        if not wait_for_v6d(v6d, 30):
            print("FAILED")
            check(f"{tag}_t1_server_start", False, "v6d failed to start")
            return
        print("OK")
        time.sleep(1)

        # Phase A: Create ~98.03% of the arena in ONE batch.
        # Request path: future >= critical(98%) -> [EVICT_FORCE] ->
        # wait_complete on an empty store -> nothing to evict -> proceeds.
        print(f"  Phase A: Create {n_batch} objects in one batch "
              f"({n_batch*blob*100/arena_cap:.1f}%)...")
        keys_a = [f"blk_obj_{i:05d}" for i in range(n_batch)]
        ok, _ = create_batch_and_seal(URL_A, keys_a, blob)
        check(f"{tag}_t1a_batch_create", ok, f"created={ok}")

        # Creates landed.  A 5s sweep tick may already have evicted down to
        # min (the sweep ignores max), so accept >= half of the sample.
        sample_a = _sample(keys_a)
        exist_a = sum(1 for k in sample_a if exists_on(URL_A, k))
        check(f"{tag}_t1b_creates_landed",
              exist_a >= len(sample_a) // 2,
              f"exists={exist_a}/{len(sample_a)} (sampled; a periodic "
              f"sweep to min=50% may already have run)")

        # Phase B: Create 1 more -> future >= critical -> [EVICT_FORCE] +
        # wait_complete: the eviction cycle finishes before create returns.
        print(f"  Phase B: Create 1 more (used={n_batch*blob*100/arena_cap:.1f}% "
              f">= critical=98%)...")
        ok, _ = create_and_seal(URL_A, "blk_trigger", blob)
        check(f"{tag}_t1c_trigger_create", ok, f"success={ok}")

        # KEY TEST: blocking eviction completed inside the create call
        # (wait_complete) — or a sweep tick did; either way old objects
        # are already gone when the create returns.
        time.sleep(0.5)
        evicted_immediately = sum(1 for k in sample_a if not exists_on(URL_A, k))
        check(f"{tag}_t1d_evicted_immediately", evicted_immediately >= 1,
              f"evicted={evicted_immediately}/{len(sample_a)} (sampled; "
              f"blocking eviction should be immediate)")

        # Trigger object should exist
        check(f"{tag}_t1e_trigger_exists", exists_on(URL_A, "blk_trigger"),
              "trigger object should exist")

        # Verify [EVICT_FORCE] in logs (blocking path)
        force_logs = grep_log(v6d, "EVICT_FORCE")
        check(f"{tag}_t1f_force_log_present", len(force_logs) >= 1,
              f"EVICT_FORCE lines={len(force_logs)}")

        # Sparse-arena proof: physical RSS stays flat at 500G
        if cfg["name"] == "500G":
            rss = daemon_rss_mb(v6d)
            check(f"{tag}_t1g_sparse_rss", 0 < rss < 2048,
                  f"daemon RSS={rss:.0f}MB after {n_batch}+ blobs in a "
                  f"500G arena (sparse memfd)")

        # Summary
        total_on_a = sum(1 for k in _sample(keys_a + ["blk_trigger"])
                         if exists_on(URL_A, k))
        print(f"  Result: {total_on_a} exist on A (sampled)")
        if force_logs:
            print(f"  Log: {force_logs[-1][:120]}")

    finally:
        stop_proc(v6d)

# --------------------------------------------------------------------------
# Test 2: OOM-triggered emergency eviction
# --------------------------------------------------------------------------

def test_oom_emergency_eviction(cfg):
    """Test that OOM triggers emergency eviction and retry succeeds.

    With max=2.0 / min=0.9999: the request path never signals (future < max)
    and the 5s periodic sweep stays inert (fill tops out at ~99.95% <
    99.99% == min).  Phase B then creates one BIG blob (5% of the arena —
    larger than any block-level slack), forcing create_blobs to raise
    NotEnoughMemoryException -> _trigger_emergency_eviction (evict to
    emergency_min=60%) -> retry succeeds.
    """
    tag = cfg["name"]
    blob = cfg["blob_size"]
    capacity_blobs = cfg["arena_bytes"] // blob - 1
    print(f"\n=== Test 2 [{tag}]: OOM-Triggered Emergency Eviction ===")
    USAGE_MAX = 2.0        # request path never signals
    USAGE_MIN = 0.9999     # periodic sweep inert (fill peaks at ~99.95%)
    USAGE_EMERGENCY = 0.6  # emergency target: evict to 60%
    max_blobs = capacity_blobs
    big_blob = cfg["arena_bytes"] // 20  # 5% of arena >> any residual slack

    v6d = None
    try:
        flush_redis()
        v6d = start_v6d(cfg, USAGE_MAX, USAGE_MIN, USAGE_EMERGENCY, "oom")
        print("  Waiting for v6d...", end=" ", flush=True)
        if not wait_for_v6d(v6d, 30):
            print("FAILED")
            check(f"{tag}_t2_server_start", False, "v6d failed to start")
            return
        print("OK")
        time.sleep(1)

        # Phase A: Fill to capacity (100%), batched for speed
        batch_size = 256
        print(f"  Phase A: Fill to {max_blobs} objects (100%)...")
        keys_fill = [f"oom_obj_{i:05d}" for i in range(max_blobs)]
        created = 0
        for i in range(0, max_blobs, batch_size):
            batch = keys_fill[i:i+batch_size]
            ok, _ = create_batch_and_seal(URL_A, batch, blob)
            if ok:
                created += len(batch)
            else:
                print(f"  batch {i//batch_size} FAILED "
                      f"({len(batch)} keys)")
        check(f"{tag}_t2a_fill_to_capacity", created == max_blobs,
              f"created={created}/{max_blobs}")

        # Verify all exist (sampled)
        sample_fill = _sample(keys_fill)
        exist_count = sum(1 for k in sample_fill if exists_on(URL_A, k))
        check(f"{tag}_t2b_filled_objects_exist",
              exist_count == len(sample_fill),
              f"sample exists={exist_count}/{len(sample_fill)}")

        # Phase B: Create one BIG blob (5% of arena) -> OOM -> emergency
        # eviction (to 60%) -> retry succeeds.  A big blob is used so the
        # OOM does not depend on exact block-level slack after the fill.
        print(f"  Phase B: Create one {big_blob >> 20}MiB blob "
              f"(should OOM -> emergency -> retry)...")
        t0 = time.time()
        ok, lease_id = create_and_seal(URL_A, "oom_extra_obj", big_blob)
        elapsed = time.time() - t0
        check(f"{tag}_t2c_create_after_oom_succeeds", ok,
              f"success={ok}, elapsed={elapsed:.1f}s")

        # Verify the extra object exists (retry succeeded)
        check(f"{tag}_t2d_extra_object_exists",
              exists_on(URL_A, "oom_extra_obj"),
              "extra object should exist after emergency eviction + retry")

        # Verify [EMERGENCY] in logs
        emergency_logs = grep_log(v6d, "EMERGENCY")
        check(f"{tag}_t2e_emergency_log_present", len(emergency_logs) >= 1,
              f"EMERGENCY lines={len(emergency_logs)}")

        # Verify [POST_EMERGENCY_EVICT] in logs
        post_emg_logs = grep_log(v6d, "POST_EMERGENCY_EVICT")
        check(f"{tag}_t2f_post_emergency_log", len(post_emg_logs) >= 1,
              f"POST_EMERGENCY_EVICT lines={len(post_emg_logs)}")

        # Some old objects should have been evicted (sampled)
        evicted_count = sum(1 for k in sample_fill
                            if not exists_on(URL_A, k))
        check(f"{tag}_t2g_old_objects_evicted", evicted_count >= 1,
              f"evicted={evicted_count}/{len(sample_fill)} (sampled)")

        # Sparse-arena proof: physical RSS stays flat at 500G
        if cfg["name"] == "500G":
            rss = daemon_rss_mb(v6d)
            check(f"{tag}_t2h_sparse_rss", 0 < rss < 2048,
                  f"daemon RSS={rss:.0f}MB with {max_blobs} blobs in a "
                  f"500G arena (sparse memfd)")

        # Summary
        print(f"  Result: ~{evicted_count}/{len(sample_fill)} sampled old "
              f"objects evicted by emergency eviction")
        if emergency_logs:
            print(f"  Log: {emergency_logs[-1][:120]}")
        if post_emg_logs:
            print(f"  Log: {post_emg_logs[-1][:120]}")

    finally:
        stop_proc(v6d)

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _arena_bytes(s: str) -> int:
    if s.endswith("G"):
        return int(s[:-1]) << 30
    return int(s[:-1]) << 20

def main():
    print("=== V6D Emergency Eviction Test ===")
    for cfg in CONFIGS:
        cfg["arena_bytes"] = _arena_bytes(cfg["arena_size"])
        print(f"config {cfg['name']}: arena={cfg['arena_size']} "
              f"blob={cfg['blob_size']} "
              f"capacity={cfg['arena_bytes'] // cfg['blob_size'] - 1} blobs")
    print()

    redis_proc = None
    try:
        redis_proc = ensure_redis()
        if redis_proc is not None and redis_proc.poll() is not None:
            print("Redis failed to start")
            sys.exit(1)

        for cfg in CONFIGS:
            print(f"\n########## Config: {cfg['name']} ##########")
            test_blocking_eviction(cfg)
            test_oom_emergency_eviction(cfg)

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
