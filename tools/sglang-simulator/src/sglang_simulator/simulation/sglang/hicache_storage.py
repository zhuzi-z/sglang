import os
import time
from typing import Any, List, Optional

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import StateManager
from sglang_simulator.utils.logger import get_logger

logger = get_logger("hisim")


# Pool name string used for the default "KV" pool when no PoolTransfer.name
# is supplied. Matches sglang.srt.mem_cache.hicache_storage.PoolName.KV value.
_KV_POOL_NAME = "kv"

# Default page size used in records when we can't infer it from sglang at
# hook time. ConfigManager could supply this later if needed.
_DEFAULT_PAGE_SIZE = 256


def _extract_prefix_keys(extra_info) -> Optional[list]:
    """Pull prefix_keys out of HiCacheStorageExtraInfo (defaults to (None,))."""
    if extra_info is None:
        return None
    pk = getattr(extra_info, "prefix_keys", None)
    if pk is None or pk == (None,):
        return None
    # tolerate tuple/list
    return list(pk)


def _pool_name_str(name) -> str:
    """PoolName(str, Enum) members coerce to their string value; tolerate plain str too."""
    if name is None:
        return _KV_POOL_NAME
    return str(name)


class C_StorageBackendFactory(BaseHook):
    HOOK_CLASS_NAME = "StorageBackendFactory"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.storage.backend_factory"

    @classmethod
    def hook(cls, target):
        def override_create_backend(cls, *args, **kwargs):
            logger.info("Creating hijacked cache storage backend.")
            return MockHiCacheStorage()

        target.create_backend = override_create_backend


class MockHiCacheStorage:
    """In-memory L3 storage mock for hisim simulation.

    Mirrors the real `HiCacheStorage` v2 semantics for hybrid-SSM models
    (e.g. Qwen3.5) where the controller stores KV pages and mamba state as
    SEPARATE entries under their own pool namespace.

    Storage layout (in-memory):
        self.storage[pool_name] -> set[key]

    Persistence:
        One file per pool at /tmp/sglang_simulator/hicache_storage_keys_<POOL>.txt,
        each line is a hex/string key. Survives process restart.
    """

    STORAGE_DIR = "/tmp/sglang_simulator"

    def __init__(self, *args, **kwargs):
        self.storage: dict[str, set] = {}
        self.registered_pools: dict[str, Any] = {}
        os.makedirs(self.STORAGE_DIR, exist_ok=True)
        # Eager-load any pre-existing per-pool key files from prior runs.
        for fname in os.listdir(self.STORAGE_DIR):
            if fname.startswith("hicache_storage_keys_") and fname.endswith(".txt"):
                pool = fname[len("hicache_storage_keys_"): -len(".txt")]
                self._pool_set(pool)
                with open(os.path.join(self.STORAGE_DIR, fname)) as f:
                    for line in f:
                        k = line.strip()
                        if k:
                            self.storage[pool].add(k)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _pool_set(self, pool: str) -> set:
        if pool not in self.storage:
            self.storage[pool] = set()
        return self.storage[pool]

    def _pool_file(self, pool: str) -> str:
        return os.path.join(self.STORAGE_DIR, f"hicache_storage_keys_{pool}.txt")

    def _persist(self, pool: str, new_keys: List[str]) -> None:
        if not new_keys:
            return
        with open(self._pool_file(pool), "a") as f:
            for k in new_keys:
                f.write(k + "\n")

    # ------------------------------------------------------------------
    # registration (v1 + v2)
    # ------------------------------------------------------------------
    def register_mem_pool_host(self, mem_pool_host):
        # v1 — sglang single-pool path.
        pass

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        """v2 register API — HybridCacheController calls this once per host pool
        (KV pool, Mamba pool, …). Used by batch_get/set_v2 to locate the matching
        host buffer for that pool's transfers."""
        name = _pool_name_str(host_pool_name)
        self.registered_pools[name] = host_pool
        self._pool_set(name)

    # ------------------------------------------------------------------
    # v1 API — always operates on the KV namespace
    # ------------------------------------------------------------------
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        kv = self._pool_set(_KV_POOL_NAME)
        existed = key in kv
        if not existed:
            kv.add(key)
            self._persist(_KV_POOL_NAME, [key])
        return True

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        extra_info=None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        kv = self._pool_set(_KV_POOL_NAME)
        existed = [k in kv for k in keys]
        new_keys = [k for k, e in zip(keys, existed) if not e]
        kv.update(new_keys)
        self._persist(_KV_POOL_NAME, new_keys)
        # hits = "was newly written" per key (True = freshly added to L3)
        return True

    def exists(self, key: str) -> bool:
        present = key in self._pool_set(_KV_POOL_NAME)
        return present

    def batch_exists(self, keys: List[str], extra_info=None) -> int:
        """Longest-prefix exists count in the KV pool."""
        kv = self._pool_set(_KV_POOL_NAME)
        hit_len = 0
        for k in keys:
            if k not in kv:
                break
            hit_len += 1
        return hit_len

    def clear(self) -> bool:
        for pool in list(self.storage.keys()):
            self.storage[pool].clear()
            path = self._pool_file(pool)
            if os.path.exists(path):
                open(path, "w").close()
        return True

    # ------------------------------------------------------------------
    # v2 API — used by HybridCacheController for hybrid-SSM models (Qwen3.5,
    # Qwen3-Next, ...). Each PoolTransfer carries its own pool name + keys.
    # ------------------------------------------------------------------
    def batch_exists_v2(self, keys: List[str], pool_transfers=None, extra_info=None):
        """Mirror of HiCacheFile.batch_exists_v2 semantics.

        Algorithm:
          1. kv_pages = longest contiguous prefix of `keys` present in KV pool
          2. For each PoolTransfer with hit_policy=ALL_PAGES:
                 boundary = first index in [0, kv_pages) where keys[i] is NOT in that pool
                          (or kv_pages if all present)
             For TRAILING_PAGES (e.g. Mamba, trailing=len(transfer.keys) usually 1):
                 boundary = largest prefix_len in [1, kv_pages] such that
                            all keys[prefix_len-trailing .. prefix_len-1] exist in pool
                          (0 if none)
          3. final_pages = min(kv_pages, all extra boundaries)

        Returns the real PoolTransferResult dataclass when importable; falls
        back to a duck-typed namespace if sglang import is unavailable.
        """
        kv_pool = self._pool_set(_KV_POOL_NAME)
        kv_pages = 0
        for k in keys:
            if k not in kv_pool:
                break
            kv_pages += 1

        hit_count: dict = {_KV_POOL_NAME: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            pool_name = _pool_name_str(getattr(transfer, "name", None))
            pool = self._pool_set(pool_name)
            policy = getattr(transfer, "hit_policy", None)
            policy_value = getattr(policy, "value", policy)  # tolerate Enum or str
            t_keys = getattr(transfer, "keys", None) or []
            trailing = max(1, len(t_keys))

            if policy_value == "trailing_pages":
                boundary = 0
                # Walk down from kv_pages; the largest prefix_len where the
                # last `trailing` keys (in the OUTER KV-hash space) exist in
                # this pool's namespace.
                for prefix_len in range(kv_pages, 0, -1):
                    lo = max(0, prefix_len - trailing)
                    if all(keys[i] in pool for i in range(lo, prefix_len)):
                        boundary = prefix_len
                        break
            else:  # ALL_PAGES (default)
                boundary = kv_pages
                for i in range(kv_pages):
                    if keys[i] not in pool:
                        boundary = i
                        break

            if boundary:
                hit_count[pool_name] = boundary
            final_pages = min(final_pages, boundary)

        # Emit one record per pool_transfer so consumers can see per-pool hits.
        prefix = _extract_prefix_keys(extra_info)
        for transfer in pool_transfers or []:
            pool_name = _pool_name_str(getattr(transfer, "name", None))
            policy = getattr(transfer, "hit_policy", None)
            policy_value = getattr(policy, "value", policy)
            t_keys = getattr(transfer, "keys", None) or []

        # Try to return the real PoolTransferResult so consumers that
        # dataclass-introspect it (e.g. metrics) work; fall back to a
        # duck-typed namespace if sglang is not import-clean at this point.
        try:
            from sglang.srt.mem_cache.hicache_storage import PoolTransferResult
            return PoolTransferResult(
                kv_hit_pages=final_pages, extra_pool_hit_pages=hit_count
            )
        except Exception:
            from types import SimpleNamespace
            return SimpleNamespace(
                kv_hit_pages=final_pages, extra_pool_hit_pages=hit_count
            )

    def batch_get_v2(self, pool_transfers, extra_info=None) -> dict:
        """Per-pool per-key existence check, namespaced by pool name."""
        results: dict = {}
        prefix = _extract_prefix_keys(extra_info)
        for transfer in pool_transfers or []:
            pool_name = _pool_name_str(getattr(transfer, "name", None))
            pool = self._pool_set(pool_name)
            t_keys = getattr(transfer, "keys", None) or []
            policy = getattr(transfer, "hit_policy", None)
            policy_value = getattr(policy, "value", policy)
            per_key_hits = [k in pool for k in t_keys]
            results[pool_name] = per_key_hits
        return results

    def batch_set_v2(self, pool_transfers, extra_info=None) -> dict:
        """Per-pool per-key set, namespaced by pool name; persisted to file."""
        results: dict = {}
        prefix = _extract_prefix_keys(extra_info)
        for transfer in pool_transfers or []:
            pool_name = _pool_name_str(getattr(transfer, "name", None))
            pool = self._pool_set(pool_name)
            t_keys = getattr(transfer, "keys", None) or []
            policy = getattr(transfer, "hit_policy", None)
            policy_value = getattr(policy, "value", policy)
            existed = [k in pool for k in t_keys]
            new_keys = [k for k, e in zip(t_keys, existed) if not e]
            pool.update(new_keys)
            self._persist(pool_name, new_keys)
            results[pool_name] = [True] * len(t_keys)
            # hits[i]=True ⇒ this key was newly written this call (was missing)
        return results
