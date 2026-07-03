"""
V6DCacheStorage - Cross-node KV block state management via V6D's etcd backend.

Uses the same etcd that vineyardd connects to (shared etcd cluster) to store
and query block ownership metadata. This enables true cross-physical-machine
cache state visibility:

  Node 1: register_block(hash, worker_0) -> etcd v3 HTTP API -> shared etcd
  Node 2: lookup_block(hash) -> etcd v3 HTTP API -> shared etcd -> owner=worker_0

Architecture:
  - Each node runs vineyardd locally (IPC, for V6D object operations)
  - Block ownership registry is stored in vineyardd's etcd backend
  - Both nodes access the same etcd for cache state coordination
  - No SRPC/RPC required - only etcd HTTP API over network

The etcd endpoint is obtained from the running vineyardd process or
configured via V6D_ETCD_ENDPOINT environment variable.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import urllib.request
from typing import Optional

from sglang_simulator.utils import get_logger

logger = get_logger()

# Configuration via environment
_V6D_ETCD_ENDPOINT_ENV = "V6D_ETCD_ENDPOINT"
_V6D_SOCKET_DEFAULT = "/tmp/vineyard.sock"

# Prefix for block registry keys in etcd
_BLOCK_REGISTRY_PREFIX = "sim_kv_block"


def _etcd_put(endpoint: str, key: str, value: str, timeout: float = 5.0) -> bool:
    """Put a key-value pair into etcd via v3 HTTP API."""
    data = json.dumps({
        "key": base64.b64encode(key.encode()).decode(),
        "value": base64.b64encode(value.encode()).decode(),
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/v3/kv/put",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False


def _etcd_get(endpoint: str, key: str, timeout: float = 5.0) -> Optional[str]:
    """Get a value from etcd via v3 HTTP API. Returns None if not found."""
    data = json.dumps({
        "key": base64.b64encode(key.encode()).decode(),
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/v3/kv/range",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        result = json.loads(resp.read())
        kvs = result.get("kvs", [])
        if kvs:
            return base64.b64decode(kvs[0]["value"]).decode()
        return None
    except Exception:
        return None


def _etcd_delete_prefix(endpoint: str, prefix: str, timeout: float = 10.0) -> bool:
    """Delete all keys with given prefix from etcd."""
    prefix_bytes = prefix.encode()
    end_bytes = prefix_bytes[:-1] + bytes([prefix_bytes[-1] + 1])
    data = json.dumps({
        "key": base64.b64encode(prefix_bytes).decode(),
        "range_end": base64.b64encode(end_bytes).decode(),
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/v3/kv/deleterange",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False


def _discover_etcd_endpoint() -> Optional[str]:
    """Discover etcd endpoint from environment or running vineyardd process.

    Priority:
    1. V6D_ETCD_ENDPOINT env var (explicit configuration)
    2. Parse from running vineyardd process command line
    """
    # Check env var first
    endpoint = os.environ.get(_V6D_ETCD_ENDPOINT_ENV)
    if endpoint:
        return endpoint

    # Try to parse from vineyardd process
    try:
        import subprocess
        result = subprocess.run(
            ["bash", "-c", "ps aux | grep vineyardd | grep -v grep"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "-etcd_endpoint" in line:
                parts = line.split("-etcd_endpoint")
                if len(parts) > 1:
                    ep = parts[1].strip().split()[0]
                    return ep
    except Exception:
        pass

    return None


class V6DCacheStorage:
    """Cross-node KV block ownership registry backed by V6D's etcd.

    Uses etcd v3 HTTP API directly (the same etcd vineyardd connects to)
    for block state coordination across physical machines. This enables
    true cross-node cache hit detection with real network communication.

    Each block is stored as:
      Key:   {prefix}/{block_hash}
      Value: JSON {"owner_worker_id": "...", "block_hash": "..."}
    """

    _instance: Optional["V6DCacheStorage"] = None
    _lock = threading.Lock()

    def __init__(self, etcd_endpoint: Optional[str] = None):
        self._etcd_endpoint = etcd_endpoint or _discover_etcd_endpoint()
        self._connected = False

        if self._etcd_endpoint:
            # Verify connectivity with a health check write
            test_key = f"{_BLOCK_REGISTRY_PREFIX}/__health__"
            if _etcd_put(self._etcd_endpoint, test_key, "ok", timeout=3.0):
                self._connected = True
                logger.info(
                    "[V6DCacheStorage] Connected to etcd at %s (V6D backend)",
                    self._etcd_endpoint,
                )
            else:
                logger.warning(
                    "[V6DCacheStorage] Cannot connect to etcd at %s",
                    self._etcd_endpoint,
                )
        else:
            logger.warning(
                "[V6DCacheStorage] No etcd endpoint found. "
                "Set V6D_ETCD_ENDPOINT or start vineyardd with -etcd_endpoint."
            )

    @property
    def connected(self) -> bool:
        return self._connected

    @classmethod
    def get_instance(cls, etcd_endpoint: Optional[str] = None) -> "V6DCacheStorage":
        """Get or create the singleton storage instance."""
        if cls._instance is None or not cls._instance._connected:
            with cls._lock:
                if cls._instance is None or not cls._instance._connected:
                    cls._instance = cls(etcd_endpoint)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton (for test isolation)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear()
                cls._instance = None

    def register_block(self, block_hash: str, worker_id: str) -> bool:
        """Register block ownership in etcd (cross-node visible via network).

        Stores: {prefix}/{block_hash} -> {"owner_worker_id": worker_id, ...}
        This write goes through the real network to the shared etcd cluster.

        Args:
            block_hash: The hex string of the block hash.
            worker_id: The worker that owns this block.

        Returns:
            True if registered successfully.
        """
        if not self._connected:
            return False

        key = f"{_BLOCK_REGISTRY_PREFIX}/{block_hash}"
        value = json.dumps({"owner_worker_id": worker_id, "block_hash": block_hash})
        return _etcd_put(self._etcd_endpoint, key, value)

    def lookup_block(self, block_hash: str) -> Optional[str]:
        """Query block ownership from etcd (cross-node, real network I/O).

        Reads from the shared etcd cluster - if the block was registered
        by a worker on another physical machine, this involves real network
        communication to discover the remote cache state.

        Args:
            block_hash: The hex string of the block hash.

        Returns:
            owner_worker_id if found, None if not registered.
        """
        if not self._connected:
            return None

        key = f"{_BLOCK_REGISTRY_PREFIX}/{block_hash}"
        raw = _etcd_get(self._etcd_endpoint, key)
        if raw:
            try:
                meta = json.loads(raw)
                return meta.get("owner_worker_id")
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def clear(self):
        """Remove all registered block entries from etcd (test isolation)."""
        if not self._connected:
            return
        _etcd_delete_prefix(self._etcd_endpoint, _BLOCK_REGISTRY_PREFIX)
        logger.info("[V6DCacheStorage] Cleared all blocks from etcd")

    def get_stats(self) -> dict:
        """Return storage stats."""
        return {
            "connected": self._connected,
            "etcd_endpoint": self._etcd_endpoint,
            "backend": "etcd_v3_http",
        }
