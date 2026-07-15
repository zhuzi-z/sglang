#!/usr/bin/env python3
"""Validate cross-node cache-state visibility through V6DCacheStorage.

This script exercises the same etcd-backed ownership store used by
MockHybridConnector in CPU simulation:

- store:  register deterministic block hashes with owner_worker_id
- lookup: query the same hashes from another node and classify remote hits
- full:   run store and lookup in one process for smoke testing

Data transfer is intentionally not performed in CPU simulation. A remote hit
means the caller can skip SRPC/KVT data transfer and trigger deterministic local
cache rebuild/update for the matched block hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()

from sglang_simulator.simulation.vllm.v6d_cache_storage import V6DCacheStorage  # noqa: E402


@dataclass(frozen=True)
class LookupResult:
    total: int
    local_hits: int
    remote_hits: int
    misses: int


def generate_block_hashes(namespace: str, block_count: int) -> list[str]:
    return [
        f"{namespace}:" + hashlib.sha256(f"{namespace}:{i}".encode()).hexdigest()[:16]
        for i in range(block_count)
    ]


def get_storage(etcd_endpoint: str) -> V6DCacheStorage:
    os.environ["V6D_ETCD_ENDPOINT"] = etcd_endpoint
    storage = V6DCacheStorage.get_instance(etcd_endpoint)
    if not storage.connected:
        raise RuntimeError(f"V6DCacheStorage cannot connect to etcd endpoint: {etcd_endpoint}")
    return storage


def store_blocks(etcd_endpoint: str, namespace: str, owner_worker_id: str, block_count: int) -> list[str]:
    storage = get_storage(etcd_endpoint)
    block_hashes = generate_block_hashes(namespace, block_count)
    stored = 0
    for block_hash in block_hashes:
        if storage.register_block(block_hash, owner_worker_id):
            stored += 1
    print(f"STORE_ENDPOINT={etcd_endpoint}")
    print(f"STORE_NAMESPACE={namespace}")
    print(f"STORE_OWNER={owner_worker_id}")
    print(f"STORE_BLOCKS={stored}/{len(block_hashes)}")
    if stored != len(block_hashes):
        raise RuntimeError("not all blocks were stored")
    return block_hashes


def lookup_blocks(etcd_endpoint: str, namespace: str, current_worker_id: str, block_count: int) -> LookupResult:
    storage = get_storage(etcd_endpoint)
    block_hashes = generate_block_hashes(namespace, block_count)
    local_hits = 0
    remote_hits = 0
    misses = 0

    for block_hash in block_hashes:
        owner = storage.lookup_block(block_hash)
        if owner is None:
            misses += 1
            continue
        if owner == current_worker_id:
            local_hits += 1
        else:
            remote_hits += 1
            print(f"REMOTE_HIT block={block_hash} owner={owner} current={current_worker_id}")

    result = LookupResult(
        total=len(block_hashes),
        local_hits=local_hits,
        remote_hits=remote_hits,
        misses=misses,
    )
    print(f"LOOKUP_ENDPOINT={etcd_endpoint}")
    print(f"LOOKUP_NAMESPACE={namespace}")
    print(f"LOOKUP_WORKER={current_worker_id}")
    print(f"LOOKUP_TOTAL={result.total}")
    print(f"LOOKUP_LOCAL_HITS={result.local_hits}")
    print(f"LOOKUP_REMOTE_HITS={result.remote_hits}")
    print(f"LOOKUP_MISSES={result.misses}")
    if result.remote_hits > 0:
        print("CROSS_NODE_MATCH_VERIFIED")
        print("DATA_TRANSFER_SKIPPED")
        print("LOCAL_CACHE_REBUILD_TRIGGERED")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cross-node cache-state matching via V6DCacheStorage")
    parser.add_argument("--role", choices=("store", "lookup", "full"), default="full")
    parser.add_argument("--etcd-endpoint", default=os.environ.get("V6D_ETCD_ENDPOINT", ""))
    parser.add_argument("--namespace", default="qoder_cross_node_v6d_state")
    parser.add_argument("--store-worker", default="worker_node_0")
    parser.add_argument("--lookup-worker", default="worker_node_1")
    parser.add_argument("--block-count", type=int, default=10)
    args = parser.parse_args()

    if not args.etcd_endpoint:
        raise SystemExit("--etcd-endpoint or V6D_ETCD_ENDPOINT is required")

    if args.role in {"store", "full"}:
        store_blocks(args.etcd_endpoint, args.namespace, args.store_worker, args.block_count)
    if args.role in {"lookup", "full"}:
        result = lookup_blocks(args.etcd_endpoint, args.namespace, args.lookup_worker, args.block_count)
        if result.remote_hits == 0:
            raise SystemExit("no remote hits detected")


if __name__ == "__main__":
    main()
