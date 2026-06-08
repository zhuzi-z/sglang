import os
from typing import Any, List, Optional

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager.env import Envs
from sglang_simulator.utils.logger import get_logger

logger = get_logger("hisim")


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
    def __init__(self, *args, **kwargs):

        self.storage: set = set()
        self.storage_file_path: str = Envs.hicache_storage_keys_path()
        os.makedirs(os.path.dirname(self.storage_file_path), exist_ok=True)

        if os.path.exists(self.storage_file_path):
            with open(self.storage_file_path) as f:
                line = f.readline()
                while line:
                    self.storage.add(line.strip())
                    line = f.readline()

    def register_mem_pool_host(self, mem_pool_host):
        pass

    def register_mem_host_pool_v2(self, *args, **kwargs):
        pass

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.exists(key):
            return True
        self.storage.add(key)
        with open(self.storage_file_path, "a+") as f:
            f.write(key + "\n")
        return True

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        extra_info=None,  # HiCacheStorageExtraInfo
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:

        for key in keys:
            if not self.set(key):
                return False
        return True

    def batch_set_v2(
        self,
        transfers: List,
        extra_info: Optional[Any] = None,
    ):

        results = {}
        for transfer in transfers:
            self.batch_set(transfer.keys)
            results[transfer.name] = [True] * len(transfer.keys)
        return results

    def exists(self, key: str) -> bool:
        return key in self.storage

    def batch_exists(self, keys: List[str], extra_info) -> int:
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)
    
    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List] = None,
        extra_info: Optional[Any] = None,
    ):
        
        from sglang.srt.mem_cache.hicache_storage import PoolTransferResult, PoolName

        kv_pages = self.batch_exists(keys, extra_info)

        hit_count: dict = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages
        return PoolTransferResult(final_pages, hit_count)

    def clear(self) -> bool:
        self.storage.clear()
        with open(self.storage_file_path, "w"):
            pass
        return True
