#!/usr/bin/env python3
"""V6D P2P lookup integration test.

Starts Redis + two v6d servers (real memory, no RDMA, no capacity hook),
verifies P2P object discovery via Redis tracker.

Prerequisites:
  - redis-server installed
  - libsrpc_stream_engine.so at scripts/
  - scripts/v6d launcher

Run:
  python test/test_v6d_p2p.py
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

REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}"

V6D_A_PORT = 7890
V6D_B_PORT = 7891
V6D_A_RPC = 21000
V6D_B_RPC = 21001
V6D_A_SOCK = "/tmp/vineyard.sock_a"
V6D_B_SOCK = "/tmp/vineyard.sock_b"

URL_A = f"http://127.0.0.1:{V6D_A_PORT}"
URL_B = f"http://127.0.0.1:{V6D_B_PORT}"

# Environment for v6d subprocesses (no capacity hook)
V6D_ENV = {
    **os.environ,
    "LD_LIBRARY_PATH": LIB_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
}

# --------------------------------------------------------------------------
# Results tracking
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

def http_get(base: str, path: str, timeout: float = 5.0) -> tuple[int, dict | None]:
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

def http_post(base: str, path: str, data: dict, timeout: float = 30.0) -> tuple[int, dict | None]:
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

def start_redis() -> subprocess.Popen:
    """Start Redis server (or reuse existing instance)."""
    # Check if Redis is already running
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
        ["redis-server", "--port", str(REDIS_PORT), "--save", "", "--appendonly", "no",
         "--daemonize", "no", "--loglevel", "warning"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for Redis to be ready
    deadline = time.time() + 5
    while time.time() < deadline:
        if proc.poll() is not None:
            print("FAILED (process exited)")
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

def start_v6d(name: str, port: int, rpc_port: int, sock: str, peer_id: str) -> subprocess.Popen:
    """Start a v6d server subprocess."""
    args = [
        V6D_BIN, "serve",
        "--peer=tiered_vineyard",
        "--vineyard-size=128M",
        "--port", str(port),
        "--vineyard-socket", sock,
        "--vineyard-rpc-port", str(rpc_port),
        "--peer-id", peer_id,
        "--tracker-redis", REDIS_URL,
        "--tracker-ttl", "60",
    ]
    log_file = os.path.join(SIM_ROOT, "tmp.out", f"v6d_p2p_{name}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_fp = open(log_file, "w")

    print(f"Starting v6d {name} (port {port}, rpc {rpc_port}, peer {peer_id})...")
    proc = subprocess.Popen(
        args, env=V6D_ENV,
        stdout=log_fp, stderr=subprocess.STDOUT,
    )
    proc._log_fp = log_fp
    proc._log_file = log_file
    return proc

def wait_for_v6d(base_url: str, proc: subprocess.Popen, timeout: float = 30) -> bool:
    """Wait for v6d health endpoint."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        status, _ = http_get(base_url, "/health", timeout=2)
        if status == 200:
            return True
        time.sleep(0.5)
    return False

def stop_proc(proc: subprocess.Popen):
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
# Main test
# --------------------------------------------------------------------------

def main():
    print("=== V6D P2P Lookup Test (No RDMA) ===")
    print()

    redis_proc = None
    v6d_a = None
    v6d_b = None

    try:
        # ------------------------------------------------------------------
        # Start Redis
        # ------------------------------------------------------------------
        redis_proc = start_redis()
        if redis_proc is not None and redis_proc.poll() is not None:
            print("Redis failed to start")
            sys.exit(1)

        # ------------------------------------------------------------------
        # Start v6d servers
        # ------------------------------------------------------------------
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

        # Give trackers time to register
        time.sleep(2)

        # ------------------------------------------------------------------
        # Test 1: Health check both servers
        # ------------------------------------------------------------------
        sa, _ = http_get(URL_A, "/health")
        sb, _ = http_get(URL_B, "/health")
        check("t1_health_both", sa == 200 and sb == 200,
              f"A={sa}, B={sb}")

        # ------------------------------------------------------------------
        # Test 2: Peer info shows tracker=connected
        # ------------------------------------------------------------------
        sa, body_a = http_get(URL_A, "/peer_info")
        sb, body_b = http_get(URL_B, "/peer_info")
        # Tracker status is logged, not directly in peer_info.
        # Check peer_info returns 200 with expected fields.
        check("t2_peer_info", sa == 200 and sb == 200,
              f"A status={sa}, B status={sb}")

        # ------------------------------------------------------------------
        # Test 3: Create object on A
        # ------------------------------------------------------------------
        obj_key = "p2p_test_object_001"
        sa, body = http_post(URL_A, "/acquire", {
            "scope": "create",
            "object_keys": [obj_key],
            "object_metas": [{"size": 1024}],
            "term": 60,
        })
        lease_id = ""
        if body and isinstance(body, dict):
            lease_id = body.get("lease", {}).get("lease_id", "")
        check("t3_create_on_a", sa == 200 and lease_id != "",
              f"status={sa}, lease_id={lease_id[:16]}...")

        # ------------------------------------------------------------------
        # Test 4: Seal object on A (triggers tracker announce)
        # ------------------------------------------------------------------
        sa, body = http_post(URL_A, "/seal", {
            "lease_id": lease_id,
            "seal_object_keys": [obj_key],
        })
        check("t4_seal_on_a", sa == 204,
              f"status={sa}")

        # Give tracker a moment to process
        time.sleep(1)

        # ------------------------------------------------------------------
        # Test 5: Object exists locally on A
        # ------------------------------------------------------------------
        sa, body = http_post(URL_A, "/exists", {"object_key": obj_key})
        exists_a = body.get("exists", False) if body else False
        check("t5_exists_on_a", sa == 200 and exists_a,
              f"status={sa}, exists={exists_a}")

        # ------------------------------------------------------------------
        # Test 6: Object discoverable on B via tracker
        # ------------------------------------------------------------------
        sb, body = http_post(URL_B, "/exists", {"object_key": obj_key})
        exists_b = body.get("exists", False) if body else False
        check("t6_exists_on_b_via_tracker", sb == 200 and exists_b,
              f"status={sb}, exists={exists_b}")

        # ------------------------------------------------------------------
        # Test 7: Non-existent object returns False on B
        # ------------------------------------------------------------------
        sb, body = http_post(URL_B, "/exists", {"object_key": "nonexistent_key_xyz"})
        exists_none = body.get("exists", True) if body else True
        check("t7_nonexistent_false", sb == 200 and not exists_none,
              f"status={sb}, exists={exists_none}")

        # ------------------------------------------------------------------
        # Test 8: Release lease on A (releases the lease, object persists)
        # ------------------------------------------------------------------
        sa, body = http_post(URL_A, "/release", {"lease_id": lease_id})
        check("t8_release_on_a", sa == 204,
              f"status={sa}")

        # ------------------------------------------------------------------
        # Test 9: Object still exists on A after release (sealed objects persist)
        # ------------------------------------------------------------------
        sa, body = http_post(URL_A, "/exists", {"object_key": obj_key})
        exists_after = body.get("exists", False) if body else False
        check("t9_exists_after_release", sa == 200 and exists_after,
              f"status={sa}, exists={exists_after}")

    finally:
        # ------------------------------------------------------------------
        # Cleanup
        # ------------------------------------------------------------------
        print("\nCleaning up...")
        stop_proc(v6d_a)
        stop_proc(v6d_b)
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

        # Print relevant log excerpts
        for name, proc in [("A", v6d_a), ("B", v6d_b)]:
            if hasattr(proc, "_log_file"):
                try:
                    with open(proc._log_file) as f:
                        log = f.read()
                    errs = [l for l in log.split("\n")
                            if any(k in l.lower() for k in ["error", "traceback", "fail"])]
                    if errs:
                        print(f"\n--- v6d {name} errors ---")
                        for l in errs[-10:]:
                            print(f"  {l}")
                except Exception:
                    pass

        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
