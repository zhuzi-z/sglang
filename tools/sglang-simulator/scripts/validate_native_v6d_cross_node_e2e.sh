#!/usr/bin/env bash
# Validate native V6D cross-node control-plane simulation without source hot patches.
#
# This script assumes node1/node2 services are already running with:
#   PYTHONPATH=<repo>/tools/sglang-simulator
#   SGLANG_SIMULATOR_ENABLE_VLLM_HOOK=1
#   SGLANG_SIMULATOR_ENABLE_V6D_IPC_HOOK=1
#   SGLANG_SIMULATOR_NATIVE_V6D_CONTROL_PLANE=1
#   _SIM_V6D_ACTIVE_WORKER_ID=worker_v6d_1 / worker_v6d_2
#
# It verifies three gates:
#   1. simulator code is consistent on both nodes
#   2. installed DashServing/vLLM code does not contain the old request_id hot patch
#   3. explicit kv_transfer_params probes can be sent to node1 and node2
#
# Optional log validation:
#   NODE1_LOG=/path/to/node1.log NODE2_LOG=/path/to/node2.log bash ...

set -euo pipefail

NODE1="${NODE1:-pai_vllm_v6d_1}"
NODE2="${NODE2:-pai_vllm_v6d_2}"
REPO="${REPO:-/root/workspace/sglang-dev}"
SIM="${SIM:-${REPO}/tools/sglang-simulator}"
PROBE="${PROBE:-${SIM}/scripts/send_native_v6d_probe.py}"
NODE1_ENDPOINT="${NODE1_ENDPOINT:-127.0.0.1:8001}"
NODE2_ENDPOINT="${NODE2_ENDPOINT:-127.0.0.1:8001}"
MODEL="${MODEL:-/root/workspace/models/Qwen/Qwen3___5-0___8B}"
REPEAT="${REPEAT:-187}"
KV_JSON="${KV_JSON:-{\"do_remote_decode\": true}}"
REQ_NS="${REQ_NS:-qoder-explicit-kv-$(date +%Y%m%d%H%M%S)}"

log() {
  echo "[native-v6d-e2e] $*"
}

ssh_node() {
  local node="$1"
  shift
  ssh -S none \
    -o ControlMaster=no \
    -o ControlPath=none \
    -o ConnectTimeout=30 \
    -o ServerAliveInterval=15 \
    "$node" "$@"
}

git_head() {
  local node="$1"
  ssh_node "$node" "cd '${REPO}' && git rev-parse --short HEAD"
}

checksum_simulator() {
  local node="$1"
  ssh_node "$node" "cd '${REPO}' && find tools/sglang-simulator -type f \
    ! -name '._*' ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum"
}

check_no_hot_patch() {
  local node="$1"
  ssh_node "$node" "python3 - <<'PY'
import inspect
import dashllm.core.backend.engine._vllm_v1 as m
src = inspect.getsource(m.vLLMEngine._generate_impl)
for token in ('native-v6d', 'rdecode', 'pprefill', 'force kv_transfer_params'):
    if token in src:
        raise SystemExit(f'HOT_PATCH_FOUND token={token} file={inspect.getsourcefile(m.vLLMEngine)}')
print('NO_REQUEST_ID_HOT_PATCH', inspect.getsourcefile(m.vLLMEngine))
PY"
}

run_probe() {
  local node="$1"
  local endpoint="$2"
  local request_id="$3"
  ssh_node "$node" "PYTHONPATH='${SIM}':\${PYTHONPATH:-} python3 '${PROBE}' \
    --endpoint '${endpoint}' \
    --model '${MODEL}' \
    --repeat '${REPEAT}' \
    --request-id '${request_id}' \
    --kv '${KV_JSON}' \
    --extra-params"
}

check_log_contains() {
  local node="$1"
  local log_path="$2"
  local pattern="$3"
  ssh_node "$node" "set -o pipefail; test -f '${log_path}' && grep -aF '${pattern}' '${log_path}' | tail -5"
}

log "checking simulator checksums"
NODE1_HEAD="$(git_head "$NODE1")"
NODE2_HEAD="$(git_head "$NODE2")"
NODE1_SUM="$(checksum_simulator "$NODE1")"
NODE2_SUM="$(checksum_simulator "$NODE2")"
echo "NODE1_HEAD=${NODE1_HEAD}"
echo "NODE2_HEAD=${NODE2_HEAD}"
echo "NODE1_TREE_SUM=${NODE1_SUM}"
echo "NODE2_TREE_SUM=${NODE2_SUM}"
if [[ "$NODE1_SUM" != "$NODE2_SUM" ]]; then
  echo "[ERROR] simulator file tree differs between nodes" >&2
  exit 2
fi

log "checking request_id hot patch is absent"
check_no_hot_patch "$NODE1"
check_no_hot_patch "$NODE2"

log "sending node1 store/control-plane probe with explicit kv_transfer_params"
run_probe "$NODE1" "$NODE1_ENDPOINT" "${REQ_NS}-node1-store"

log "sending node2 lookup/control-plane probe with explicit kv_transfer_params"
run_probe "$NODE2" "$NODE2_ENDPOINT" "${REQ_NS}-node2-lookup"

if [[ -n "${NODE1_LOG:-}" ]]; then
  log "checking node1 ownership allocation log"
  check_log_contains "$NODE1" "$NODE1_LOG" "batch_allocate"
fi

if [[ -n "${NODE2_LOG:-}" ]]; then
  log "checking node2 cross-node hit and no-op completion logs"
  check_log_contains "$NODE2" "$NODE2_LOG" "cross-node hit!"
  check_log_contains "$NODE2" "$NODE2_LOG" "get_finished: store="
fi

log "validation finished"
log "acceptance: same simulator file-tree checksum, no request_id hot patch, explicit kv params probe succeeds, node2 log has cross-node hit and completion when log paths are supplied"
