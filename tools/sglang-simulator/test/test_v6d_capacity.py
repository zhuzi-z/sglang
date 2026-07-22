#!/usr/bin/env python3
"""V6D capacity control integration test.

Starts v6d serve as a subprocess with the exact CLI command, then tests
the C_VineyardServerHook via the HTTP API. No vLLM or upper-layer services.

Prerequisites:
  - libsrpc_stream_engine.so at scripts/ (LD_LIBRARY_PATH=scripts)
  - v6d command in virtualenv bin/
  - PYTHONPATH includes simulator root (for sitecustomize.py)

Run:
  python test/test_v6d_capacity.py
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
PY = "/home/zhouhaizhu.zhz/.virtualenvs/pai-vllm/bin/python3"
V6D_BIN = "/home/zhouhaizhu.zhz/.virtualenvs/pai-vllm/bin/v6d"
LIB_DIR = os.path.join(SIM_ROOT, "scripts")

V6D_HOST = "127.0.0.1"
V6D_PORT = 7890
V6D_SOCKET = "/tmp/vineyard.sock_test"
BASE_URL = f"http://{V6D_HOST}:{V6D_PORT}"

MB = 1 << 20
BLOB_SIZE = 2 << 20       # 2 MB per blob (hook fixed)
VIRTUAL_CAP = 1 << 30      # 1 GB virtual capacity (hook fixed)
MAX_BLOBS = VIRTUAL_CAP // BLOB_SIZE  # 512

# Exact v6d serve CLI command (as specified by user)
V6D_ARGS = [
    V6D_BIN, "serve",
    "--peer=tiered_vineyard",
    "--vineyard-size=256M",
    "--port", str(V6D_PORT),
    "--vineyard-socket", V6D_SOCKET,
    "--vineyard-rpc-port", "21000",
    "--peer-id", "peer_test",
]

# Environment for v6d subprocess
V6D_ENV = {
    **os.environ,
    "LD_LIBRARY_PATH": LIB_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    "PYTHONPATH": SIM_ROOT + ":" + os.environ.get("PYTHONPATH", ""),
    "SGLANG_SIMULATOR_ENABLE_V6D_IPC_HOOK": "1",
    "SGLANG_SIMULATOR_V6D_CAPACITY_CONTROL": "1",
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

def http_get(path: str, timeout: float = 5.0) -> tuple[int, dict | None]:
    url = f"{BASE_URL}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
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

def http_post(path: str, data: dict, timeout: float = 30.0) -> tuple[int, dict | None]:
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                  headers={"Content-Type": "application/json"})
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
# V6D server management
# --------------------------------------------------------------------------

def start_v6d() -> subprocess.Popen:
    """Start v6d serve subprocess."""
    # Clean up stale socket
    try:
        os.unlink(V6D_SOCKET)
    except FileNotFoundError:
        pass

    log_file = os.path.join(SIM_ROOT, "tmp.out", "v6d_capacity_test.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_fp = open(log_file, "w")

    print(f"Starting v6d serve: {' '.join(V6D_ARGS[1:])}")
    print(f"  LD_LIBRARY_PATH={V6D_ENV['LD_LIBRARY_PATH']}")
    print(f"  PYTHONPATH={V6D_ENV['PYTHONPATH']}")
    print(f"  Log: {log_file}")

    proc = subprocess.Popen(
        V6D_ARGS,
        env=V6D_ENV,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
    )
    return proc, log_fp

def wait_for_ready(proc: subprocess.Popen, timeout: float = 30.0) -> bool:
    """Wait for v6d health endpoint to respond."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        status, _ = http_get("/health", timeout=2.0)
        if status == 200:
            return True
        time.sleep(0.5)
    return False

def stop_v6d(proc: subprocess.Popen):
    """Stop v6d subprocess."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def make_metas(count: int, size: int = 1024) -> list[dict]:
    """Create object_metas for acquire request."""
    return [{"size": size} for _ in range(count)]

def acquire(metas: list[dict], term: int = 60) -> tuple[int, dict | None]:
    """POST /acquire with scope=CREATE."""
    data = {
        "scope": "create",
        "object_metas": metas,
        "term": term,
    }
    return http_post("/acquire", data, timeout=60.0)

def discard(lease_id: str) -> tuple[int, dict | None]:
    """POST /discard to release objects."""
    return http_post("/discard", {"lease_id": lease_id}, timeout=10.0)

def main():
    print("=== V6D Capacity Control Integration Test ===")
    print(f"Virtual capacity: {VIRTUAL_CAP // MB} MB")
    print(f"Blob size: {BLOB_SIZE // MB} MB (fixed)")
    print(f"Max blobs: {MAX_BLOBS}")
    print()

    proc = None
    log_fp = None
    try:
        # ------------------------------------------------------------------
        # Start v6d serve
        # ------------------------------------------------------------------
        proc, log_fp = start_v6d()
        print("Waiting for v6d to be ready...", end=" ", flush=True)
        if not wait_for_ready(proc, timeout=30):
            print("FAILED")
            print("v6d did not start. Log:")
            log_fp.flush()
            with open(log_fp.name) as f:
                print(f.read()[-2000:])
            sys.exit(1)
        print("OK")

        # ------------------------------------------------------------------
        # Test 1: Health check
        # ------------------------------------------------------------------
        status, body = http_get("/health")
        check("t1_health", status == 200,
              f"status={status}, body={body}")

        # ------------------------------------------------------------------
        # Test 2: Peer info
        # ------------------------------------------------------------------
        status, body = http_get("/peer_info")
        check("t2_peer_info", status == 200,
              f"status={status}, peer_type={body.get('peer_type') if body else 'N/A'}")

        # ------------------------------------------------------------------
        # Test 3: Create single object — should succeed
        # ------------------------------------------------------------------
        status, body = acquire(make_metas(1))
        lease_id = body.get("lease", {}).get("lease_id", "") if body else ""
        check("t3_create_single",
              status == 200 and lease_id != "",
              f"status={status}, lease_id={lease_id[:16]}...")

        # ------------------------------------------------------------------
        # Test 4: Create batch of 10 objects — should succeed
        # ------------------------------------------------------------------
        status, body = acquire(make_metas(10))
        lease_id_4 = body.get("lease", {}).get("lease_id", "") if body else ""
        obj_count = len(body.get("objects", [])) if body else 0
        check("t4_create_batch_10",
              status == 200 and obj_count == 10,
              f"status={status}, objects={obj_count}")

        # ------------------------------------------------------------------
        # Test 5: Fill to 512 blobs (virtual capacity) in one batch
        # Already created 11 blobs (1 + 10). Need 501 more.
        # But capacity-based eviction (default=100) may evict some.
        # Instead, create 501 in one batch — total 512.
        # ------------------------------------------------------------------
        status, body = acquire(make_metas(501))
        obj_count = len(body.get("objects", [])) if body else 0
        check("t5_fill_to_512",
              status == 200 and obj_count == 501,
              f"status={status}, objects={obj_count}")

        # ------------------------------------------------------------------
        # Test 6: One more object should trigger OOM
        # Virtual used = 512 * 2MB = 1024MB = 1GB (full)
        # 513th blob → try_allocate(513) → 513 * 2MB > 1GB → OOM
        # ------------------------------------------------------------------
        status, body = acquire(make_metas(1))
        is_oom = status == 500 and body and (
            "Not enough memory" in str(body) or
            "Virtual capacity" in str(body) or
            "not enough memory" in str(body).lower()
        )
        check("t6_oom_at_capacity", is_oom,
              f"status={status}, error={str(body)[:200] if body else 'N/A'}")

        # ------------------------------------------------------------------
        # Test 7: Batch of 513 should also fail with OOM
        # ------------------------------------------------------------------
        status, body = acquire(make_metas(513))
        is_oom2 = status == 500 and body and (
            "Not enough memory" in str(body) or
            "Virtual capacity" in str(body) or
            "not enough memory" in str(body).lower()
        )
        check("t7_oom_batch_513", is_oom2,
              f"status={status}, error={str(body)[:200] if body else 'N/A'}")

        # ------------------------------------------------------------------
        # Test 8: Discard some objects, then create should succeed
        # ------------------------------------------------------------------
        if lease_id_4:
            status, _ = discard(lease_id_4)
            check("t8a_discard", status == 204,
                  f"status={status}")
        else:
            check("t8a_discard", False, "no lease_id to discard")

        # After discard, virtual used should decrease.
        # The discard releases the lease, which triggers del_blob on the
        # objects in that lease (10 blobs → -20MB).
        # Now create 1 object — should succeed.
        status, body = acquire(make_metas(1))
        new_lease = body.get("lease", {}).get("lease_id", "") if body else ""
        check("t8b_create_after_discard",
              status == 200 and new_lease != "",
              f"status={status}, lease_id={new_lease[:16]}...")

        # ------------------------------------------------------------------
        # Test 9: Verify hook is applied (check server logs)
        # ------------------------------------------------------------------
        log_fp.flush()
        with open(log_fp.name) as f:
            log_content = f.read()
        hook_installed = "C_VineyardServerHook installed" in log_content
        capacity_registered = "C_VineyardServerHook registered" in log_content
        check("t9_hook_in_logs",
              hook_installed or capacity_registered,
              f"installed={hook_installed}, registered={capacity_registered}")

    finally:
        if proc:
            stop_v6d(proc)
        if log_fp:
            log_fp.close()
        try:
            os.unlink(V6D_SOCKET)
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
