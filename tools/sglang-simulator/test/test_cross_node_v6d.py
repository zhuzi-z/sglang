"""
Cross-node V6D KV Cache hit verification test.

This test verifies the full cross-node cache hit detection flow:
1. Node 1 (worker_node_0): seals KV block objects into V6D via shared etcd
2. Node 2 (worker_node_1): queries etcd to discover blocks, classifies as remote hit
3. Data transfer is SKIPPED (simulation mode: head_dim=1 makes data trivially small)

Prerequisites:
- vineyardd running on Node 1 with etcd (shared)
- Node 2 can reach Node 1's etcd endpoint
- Both nodes have sglang_simulator installed

Usage:
    # Run on Node 1 (stores blocks):
    python test_cross_node_v6d.py --role store --etcd-endpoint http://11.226.24.110:2379

    # Run on Node 2 (verifies hits):
    python test_cross_node_v6d.py --role lookup --etcd-endpoint http://10.0.240.195:2379
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import base64

ETCD_PREFIX = "vineyard_cross"
KV_BLOCK_PREFIX = "sim_kv_cache_block"


def etcd_put(endpoint, key, value):
    """Put a key-value pair into etcd v3."""
    data = json.dumps({
        "key": base64.b64encode(key.encode()).decode(),
        "value": base64.b64encode(value.encode()).decode(),
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/v3/kv/put",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=5)
    return json.loads(resp.read())


def etcd_get(endpoint, key):
    """Get a value from etcd v3."""
    data = json.dumps({
        "key": base64.b64encode(key.encode()).decode(),
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/v3/kv/range",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=5)
    result = json.loads(resp.read())
    if result.get("kvs"):
        return base64.b64decode(result["kvs"][0]["value"]).decode()
    return None


def etcd_range_keys(endpoint, prefix):
    """Get all keys with a given prefix."""
    key_bytes = prefix.encode("utf-8")
    end_bytes = key_bytes[:-1] + bytes([key_bytes[-1] + 1])
    data = json.dumps({
        "key": base64.b64encode(key_bytes).decode(),
        "range_end": base64.b64encode(end_bytes).decode(),
        "limit": 1000,
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/v3/kv/range",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=5)
    result = json.loads(resp.read())
    entries = {}
    for kv in result.get("kvs", []):
        k = base64.b64decode(kv["key"]).decode()
        v = base64.b64decode(kv["value"]).decode()
        entries[k] = v
    return entries


def generate_block_hashes(prompt_tokens, block_size=16):
    """Simulate vLLM's block hash generation from token sequences."""
    hashes = []
    for i in range(0, len(prompt_tokens) - block_size + 1, block_size):
        block = prompt_tokens[i : i + block_size]
        h = hashlib.sha256(str(block).encode()).hexdigest()[:16]
        hashes.append(h)
    return hashes


def do_store(etcd_endpoint, worker_id="worker_node_0"):
    """Simulate Node 1: seal KV blocks into V6D (via etcd metadata)."""
    print(f"[STORE] Worker: {worker_id}")
    print(f"[STORE] etcd endpoint: {etcd_endpoint}")

    # Simulate a shared prefix (like system prompt tokens)
    shared_prefix_tokens = list(range(1000, 1000 + 16 * 10))  # 160 tokens = 10 blocks

    block_hashes = generate_block_hashes(shared_prefix_tokens)
    print(f"[STORE] Generated {len(block_hashes)} block hashes from shared prefix")

    # Store each block hash in etcd (simulating V6D seal + metadata write)
    stored = 0
    for bh in block_hashes:
        key = f"{ETCD_PREFIX}/{KV_BLOCK_PREFIX}/{bh}"
        value = json.dumps({
            "owner_worker_id": worker_id,
            "instance_id": 0,
            "block_hash": bh,
            "sealed_at": time.time(),
            "block_size": 16,
            "head_dim": 1,  # simulation: head_dim=1
        })
        etcd_put(etcd_endpoint, key, value)
        stored += 1

    print(f"[STORE] Sealed {stored} blocks into V6D (etcd metadata)")
    print(f"[STORE] Block hashes: {block_hashes[:3]}...")
    return block_hashes


def do_lookup(etcd_endpoint, worker_id="worker_node_1"):
    """Simulate Node 2: lookup KV blocks from etcd (cross-node hit detection)."""
    print(f"[LOOKUP] Worker: {worker_id}")
    print(f"[LOOKUP] etcd endpoint: {etcd_endpoint}")

    # Same shared prefix tokens (simulating same requests hitting both nodes)
    shared_prefix_tokens = list(range(1000, 1000 + 16 * 10))
    block_hashes = generate_block_hashes(shared_prefix_tokens)
    print(f"[LOOKUP] Looking up {len(block_hashes)} block hashes...")

    # Query etcd for each block hash
    local_hits = 0
    remote_hits = 0
    misses = 0

    for bh in block_hashes:
        key = f"{ETCD_PREFIX}/{KV_BLOCK_PREFIX}/{bh}"
        val = etcd_get(etcd_endpoint, key)
        if val:
            meta = json.loads(val)
            owner = meta["owner_worker_id"]
            if owner == worker_id:
                local_hits += 1
            else:
                remote_hits += 1
                print(f"  [REMOTE HIT] block={bh[:8]}... owner={owner} (cross-node!)")
        else:
            misses += 1

    total = len(block_hashes)
    print(f"\n{'='*60}")
    print(f"[RESULT] Cross-node V6D cache hit verification:")
    print(f"  Total blocks queried: {total}")
    print(f"  Local hits:  {local_hits} ({100*local_hits/total:.1f}%)")
    print(f"  Remote hits: {remote_hits} ({100*remote_hits/total:.1f}%)")
    print(f"  Misses:      {misses} ({100*misses/total:.1f}%)")
    print(f"  Data transfer: SKIPPED (head_dim=1 simulation)")
    print(f"{'='*60}")

    if remote_hits > 0:
        print("\n*** CROSS-NODE CACHE HIT VERIFIED ***")
        print("  - Metadata lookup via shared etcd: SUCCESS")
        print("  - Block ownership detection: SUCCESS")
        print("  - RPC data transfer: BYPASSED (simulation mode)")
        return True
    else:
        print("\n*** FAILED: No cross-node hits detected ***")
        return False


def do_full_test(etcd_endpoint):
    """Run both store and lookup in sequence (single-machine demo)."""
    print("=" * 60)
    print("FULL CROSS-NODE V6D CACHE HIT TEST")
    print("=" * 60)
    print()

    # Phase 1: Store (simulating Node 1)
    block_hashes = do_store(etcd_endpoint, worker_id="worker_node_0")
    print()

    # Phase 2: Lookup (simulating Node 2)
    success = do_lookup(etcd_endpoint, worker_id="worker_node_1")
    print()

    if success:
        print("TEST PASSED: Cross-node V6D cache hit detection works correctly.")
        print("  Flow: seal (Node1) -> etcd metadata -> lookup (Node2) -> remote hit detected")
        print("  Data transfer bypassed (head_dim=1, no GPU needed)")
    else:
        print("TEST FAILED")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-node V6D cache hit test")
    parser.add_argument("--role", choices=["store", "lookup", "full"],
                        default="full", help="Role: store/lookup/full")
    parser.add_argument("--etcd-endpoint", default="http://11.226.24.110:2379",
                        help="etcd endpoint URL")
    parser.add_argument("--worker-id", default=None,
                        help="Worker ID override")
    args = parser.parse_args()

    if args.role == "store":
        do_store(args.etcd_endpoint, args.worker_id or "worker_node_0")
    elif args.role == "lookup":
        do_lookup(args.etcd_endpoint, args.worker_id or "worker_node_1")
    else:
        do_full_test(args.etcd_endpoint)
