#!/usr/bin/env bash
# Validate that the CPU-only v6d IPC hook can start real vineyard IPC and
# that dashllm.utils.vineyard.launch_v6d can propagate the hook to the v6d child.
#
# Usage:
#   bash scripts/validate_v6d_ipc_hook.sh
#   SIMULATOR_PATH=/root/workspace/sglang-dev/tools/sglang-simulator \
#     bash scripts/validate_v6d_ipc_hook.sh

set -euo pipefail

SIMULATOR_PATH="${SIMULATOR_PATH:-/root/workspace/sglang-dev/tools/sglang-simulator}"
V6D_SIZE="${V6D_SIZE:-256M}"
DASH_TEST_REDIS_URI="${DASH_TEST_REDIS_URI:-redis://127.0.0.1:6379}"
HOST_LABEL="${HOST_LABEL:-$(hostname | tr -c '[:alnum:]_' '_')}"
TS="$(date +%Y%m%d_%H%M%S)_$$"
BASE_PORT=$((19000 + ($$ % 1000)))
BASE_RPC_PORT=$((31000 + ($$ % 1000)))

export PYTHONPATH="${SIMULATOR_PATH}:${PYTHONPATH:-}"
export SGLANG_SIMULATOR_ENABLE_V6D_IPC_HOOK=1

log() {
  echo "[validate_v6d_ipc_hook] $*"
}

check_hook_loaded() {
  log "checking sitecustomize hook from ${SIMULATOR_PATH}"
  python3 - <<'PY'
from v6d.server.peers.vineyard.peer import VineyardPeer
import v6d.common.transfer as transfer
assert getattr(VineyardPeer, "_sglang_simulator_cpu_ipc_hook", False), "VineyardPeer hook not installed"
assert getattr(transfer.init_srpc_transfer, "__name__", "") == "_skip_srpc_transfer", "init_srpc_transfer not patched"
print("HOOK_LOADED_OK")
PY
}

validate_direct_v6d() {
  local sock="/tmp/qoder_v6d_ipc_${HOST_LABEL}_${TS}.sock"
  local log_file="/tmp/qoder_v6d_ipc_${HOST_LABEL}_${TS}.log"
  local port="${BASE_PORT}"
  local rpc_port="${BASE_RPC_PORT}"

  log "starting direct v6d serve: socket=${sock}, port=${port}, rpc_port=${rpc_port}"
  v6d serve --peer tiered_vineyard \
    --vineyard-socket "${sock}" \
    --port "${port}" \
    --vineyard-rpc-port "${rpc_port}" \
    --vineyard-size "${V6D_SIZE}" >"${log_file}" 2>&1 &
  local pid=$!

  set +e
  python3 - "${sock}" <<'PY'
import sys
import time
import vineyard
sock = sys.argv[1]
last = None
for _ in range(40):
    try:
        client = vineyard.connect(sock)
        print("DIRECT_V6D_CONNECT_OK", type(client).__name__)
        raise SystemExit(0)
    except Exception as exc:
        last = exc
        time.sleep(0.5)
print("DIRECT_V6D_CONNECT_FAIL", repr(last))
raise SystemExit(2)
PY
  local status=$?
  set -e

  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  fi

  log "direct v6d log tail: ${log_file}"
  tail -60 "${log_file}" || true
  return "${status}"
}

validate_dash_launch_v6d() {
  local sock="/tmp/qoder_dash_launch_v6d_${HOST_LABEL}_${TS}.sock"
  local log_file="/tmp/qoder_dash_launch_v6d_${HOST_LABEL}_${TS}.log"
  local port=$((BASE_PORT + 1000))
  local rpc_port=$((BASE_RPC_PORT + 1000))

  log "starting dashllm launch_v6d helper: socket=${sock}, port=${port}, rpc_port=${rpc_port}"
  python3 - "${sock}" "${log_file}" "${port}" "${rpc_port}" <<'PY'
import os
import sys
import time
import vineyard
from dashllm.utils.vineyard import launch_v6d

sock, log_path, port, rpc_port = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.environ["DS_LLM_LAUNCH_V6D"] = "1"
os.environ["KVS_METASERVICE_REDIS_URI"] = os.environ.get("DASH_TEST_REDIS_URI", "redis://127.0.0.1:6379")
os.environ.setdefault("SPECTRUM_DEPLOYMENT_NAME", "qoder-v6d-ipc-hook-test")
condition = os.environ.get("DS_LLM_LAUNCH_V6D") in {"1", "true", "True"} and os.environ.get("KVS_METASERVICE_REDIS_URI") is not None
print("DASH_AUTO_LAUNCH_CONDITION", condition)

log_fp = open(log_path, "w")
proc = None
try:
    proc = launch_v6d(
        socket=sock,
        port=port,
        rpc_port=rpc_port,
        args="--peer=tiered_vineyard --vineyard-size=" + os.environ.get("V6D_SIZE", "256M"),
        envs_to_update={
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "SGLANG_SIMULATOR_ENABLE_V6D_IPC_HOOK": "1",
        },
        log_fp=log_fp,
        is_rank_0=True,
    )
    last = None
    for _ in range(40):
        try:
            client = vineyard.connect(sock)
            print("DASH_LAUNCH_V6D_CONNECT_OK", type(client).__name__)
            raise SystemExit(0)
        except Exception as exc:
            last = exc
            time.sleep(0.5)
    print("DASH_LAUNCH_V6D_CONNECT_FAIL", repr(last))
    raise SystemExit(2)
finally:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    log_fp.close()
PY
  local status=$?
  log "dash launch_v6d log tail: ${log_file}"
  tail -60 "${log_file}" || true
  return "${status}"
}

check_hook_loaded
validate_direct_v6d
validate_dash_launch_v6d
log "all checks passed"
