"""Default v6d simulation hook for KVCM transfers with a real Manager and mock GA ledger.

The native SDK is never imported and pointers are never dereferenced. MemPool
tracks simulated written state; KVCM retains real allocation/commit/query APIs.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import math
import os
from urllib.parse import parse_qs, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from sglang_simulator.hook import BaseHook

logger = logging.getLogger("sglang_simulator")


def local_mock_url(value: str) -> str:
    """Never send simulated traffic to a real or non-loopback service."""
    parts = urlsplit(value)
    try:
        loopback = parts.hostname == "localhost" or ipaddress.ip_address(parts.hostname or "").is_loopback
    except ValueError:
        loopback = False
    if (parts.scheme != "http" or not loopback or parts.username or parts.password
            or parts.query or parts.fragment or parts.path not in ("", "/") or not parts.port):
        raise ValueError("KVCM simulation requires an explicit loopback HTTP mock endpoint")
    return value.rstrip("/")


class SimulatedKVCMTransfer:
    """Model SDK save/load against a shared MemPool GA ledger, never pointers."""

    def __init__(self, config):
        raw = config.storage_configs
        storages = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(storages, list):
            raise ValueError("storage_configs must be a JSON array")
        self.endpoints = {}
        self.closed = False
        self.timeout = float(os.environ.get("V6D_SIM_KVCM_TIMEOUT", "3"))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("V6D_SIM_KVCM_TIMEOUT must be finite and positive")
        self.opener = build_opener(ProxyHandler({}))
        for storage in storages:
            if not isinstance(storage, dict):
                raise ValueError("Each storage config must be an object")
            if storage.get("type", storage.get("storage_type")) not in ("pace", "ST_TAIRMEMPOOL"):
                continue
            spec = storage.get("storage_spec", storage.get("tair_mem_pool", {}))
            if not isinstance(spec, dict):
                raise ValueError("pace storage_spec must be an object")
            target = spec.get("service_discovery_url") or spec.get("domain")
            name = storage.get("global_unique_name")
            if not isinstance(name, str) or not isinstance(target, str) or not name or not target:
                raise ValueError("pace storage must declare its name and mock endpoint")
            self.endpoints[name] = local_mock_url(target)
        if not self.endpoints:
            raise ValueError("No ST_TAIRMEMPOOL mock storage was returned by KVCM")
        for endpoint in set(self.endpoints.values()):
            status = self.request(endpoint, "/admin/stats")
            if not str(status.get("version", "")).startswith("mock-pace-"):
                raise ValueError("Refusing simulated writes to a non-mock MemPool service")
        logger.info("[KVCM Sim] initialized storages=%s payload_bytes_moved=0", sorted(self.endpoints))

    def request(self, endpoint, path, payload=None):
        if self.closed:
            raise RuntimeError("Simulated KVCM transfer is closed")
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(endpoint + path, data=data, headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=self.timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict) or result.get("status") != "success":
            raise RuntimeError("Mock MemPool rejected simulated transfer")
        return result

    def location(self, uri, size):
        parts = urlsplit(uri)
        if parts.scheme != "pace" or parts.fragment or parts.username or parts.password:
            raise ValueError("Only Manager-issued pace URIs are supported")
        if not parts.path.startswith("/") or not parts.path[1:].isdigit():
            raise ValueError("pace URI must contain one decimal GA")
        ga = int(parts.path[1:])
        if ga <= 0:
            raise ValueError("pace GA must be positive")
        endpoint = self.endpoints.get(parts.netloc)
        if endpoint is None and not parts.netloc and len(self.endpoints) == 1:
            endpoint = next(iter(self.endpoints.values()))
        if endpoint is None:
            raise ValueError("pace URI references an unregistered storage")
        declared_size = parse_qs(parts.query).get("size")
        if declared_size and (len(declared_size) != 1 or int(declared_size[0]) != size):
            raise ValueError("URI size does not match the registered object size")
        return endpoint, {"ga": ga, "size": size}

    @staticmethod
    def validate_buffers(uris, pointers, sizes):
        if not len(uris) == len(pointers) == len(sizes):
            raise ValueError("uris, pointers, and sizes length mismatch")
        if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in sizes):
            raise ValueError("sizes must be positive integers")
        if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in pointers):
            raise ValueError("pointers must be positive integers (they are never dereferenced)")

    def transfer(self, operation, uris, pointers, sizes):
        self.validate_buffers(uris, pointers, sizes)
        results = [False] * len(uris)
        batches = {}
        for index, (uri, size) in enumerate(zip(uris, sizes)):
            try:
                endpoint, item = self.location(uri, size)
                batches.setdefault(endpoint, []).append((index, item))
            except (TypeError, ValueError) as exc:
                logger.warning("[KVCM Sim] invalid URI: %s", exc)
        for endpoint, batch in batches.items():
            try:
                response = self.request(endpoint, "/admin/transfer/" + operation,
                                        {"items": [item for _, item in batch]})
                items = response.get("results", [])
                if not isinstance(items, list) or len(items) != len(batch):
                    raise RuntimeError("Mock transfer result count mismatch")
                validated = []
                for (index, sent), received in zip(batch, items):
                    if (not isinstance(received, dict) or received.get("ga") != sent["ga"]
                            or not isinstance(received.get("success"), bool)):
                        raise RuntimeError("Mock transfer result identity/type mismatch")
                    validated.append((index, received["success"]))
                for index, success in validated:
                    results[index] = success
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("[KVCM Sim] %s failed: %s", operation, exc)
        logger.info("[KVCM Sim] operation=%s objects=%d succeeded=%d modeled_bytes=%d payload_bytes_moved=0",
                    operation, len(uris), sum(results), sum(size for size, ok in zip(sizes, results) if ok))
        return results

    def save(self, uris, pointers, sizes):
        return self.transfer("save", uris, pointers, sizes)

    def load(self, uris, pointers, sizes):
        return all(self.transfer("load", uris, pointers, sizes))

    def close(self):
        self.closed = True


class C_TairKVCMTransferRuntimeHook(BaseHook):
    HOOK_CLASS_NAME = "TairKVCMTransferRuntime"
    HOOK_MODULE_NAME = "v6d.common.tair_kvcm.transfer"

    @classmethod
    def hook(cls, target):
        if getattr(target, "_sim_kvcm_transfer_hook", False):
            return

        def initialize(self, config, *, transfer_client=None, pybind_module=None):
            if transfer_client is not None or pybind_module is not None:
                raise ValueError("Native SDK injection is incompatible with no-payload simulation")
            self.config = config
            self._cache_key = None
            self._closed = False
            self._pybind = None
            self._transfer_client = SimulatedKVCMTransfer(config)

        def save(self, uris, pointers, sizes):
            return self._transfer_client.save(uris, pointers, sizes)

        def load(self, uris, pointers, sizes):
            return self._transfer_client.load(uris, pointers, sizes)

        target.__init__ = initialize
        target.batch_save_from_memory = save
        target.batch_load_to_memory = load
        target._sim_kvcm_transfer_hook = True
        logger.info("[KVCM Sim] TairKVCMTransferRuntime hook installed; native SDK disabled")
