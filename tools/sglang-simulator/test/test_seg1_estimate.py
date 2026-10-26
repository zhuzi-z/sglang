#!/usr/bin/env python3
"""开发节点功能测试：seg1 估计模型 + per-request 条件化。"""
import json, os, sys, types

sys.path.insert(0, "/root/workspace/sglang-simtool/tools/sglang-simulator/src")
os.environ["SIMULATOR_HICACHE_BW_PROFILE"] = \
    "/root/workspace/sglang-simtool/tools/sglang-simulator/profiles/bandwidth_profile_948d_production.json"

from sglang_simulator.simulation.vllm.v6d.bandwidth import BandwidthModel

bw = BandwidthModel.get()
assert bw.enabled, "profile 未生效"
print("== T1 seg2 校准参数 ==")
print("  load(14 blk)  = %.1f ms (期望 ~24.2)" % (bw.latency_for(14, True) * 1000))
print("  load(62 blk)  = %.1f ms (期望 ~41.9)" % (bw.latency_for(62, True) * 1000))
print("== T2 seg1 估计模型 ==")
s14 = bw.seg1_latency(14) * 1000
s62 = bw.seg1_latency(62) * 1000
print("  seg1(14 blk) = %.1f ms (期望 22 + 14*24.56MiB/19.5GiBs ≈ 39.6)" % s14)
print("  seg1(62 blk) = %.1f ms (期望 ≈ 22 + 155.4 ≈ 177)" % s62)
assert 35 < s14 < 45 and 90 < s62 < 105
print("== T3 跨节点/本地 seg1 频段比 = %.1f%%（30%% 估计）==" % (s14 / max(bw.latency_for(14, True) * 1000, 1e-9) * 100))

print("== T4 backend per-request 条件化 ==")
import sglang_simulator.simulation.vllm.v6d.v6d_backend as vb
cls = None
for name in dir(vb):
    o = getattr(vb, name)
    if isinstance(o, type) and hasattr(o, "queue_load") and hasattr(o, "Store"):
        cls = o
print("  backend class:", cls.__name__)
be = object.__new__(cls)
be._seg1_scope = "local_registry"
be._local_keys = {"K1", "K2", "K3"}          # 本实例已存储的 key
be._loads = {}
import threading
be._lock = threading.RLock()
import time as _t
class FakeBw:
    def latency_for(self, n, swap_in): return 0.024
    def seg1_latency(self, n): return 0.040
vb.BandwidthModel = types.SimpleNamespace(get=lambda: FakeBw())
cls.now = lambda self: 100.0

class FakeReq:
    def __init__(self, rid): self.request_id = rid

# 场景 A：全部 key 本地已有 → 不加 seg1
be.queue_load(FakeReq("r_local"), 100, {"g": (["K1", "K2"], ["b1"])})
d_local = be._loads.pop("r_local")[0]
# 场景 B：含未知 key（远端 owner）→ 加 seg1
be.queue_load(FakeReq("r_cross"), 100, {"g": (["K1", "KX"], ["b2"])})
d_cross = be._loads.pop("r_cross")[0]
# 场景 C：空 registry 回退（新实例无本地 key）→ 按 all 处理
be._local_keys = set()
be.queue_load(FakeReq("r_fallback"), 100, {"g": (["K1"], ["b3"])})
d_fb = be._loads.pop("r_fallback")[0]
print("  local  deadline= %.3f (100+24, 无 seg1)" % d_local)
print("  cross  deadline= %.3f (100+24+40, 有 seg1)" % d_cross)
print("  fallback(deadline)= %.3f (100+24+40)" % d_fb)
assert abs(d_local - 100.024) < 1e-6
assert abs(d_cross - 100.064) < 1e-6
assert abs(d_fb - 100.064) < 1e-6
# 场景 D：scope=all 恢复旧行为
be._seg1_scope = "all"
be._local_keys = {"K1", "K2"}
be.queue_load(FakeReq("r_all"), 100, {"g": (["K1", "K2"], ["b4"])})
d_all = be._loads.pop("r_all")[0]
assert abs(d_all - 100.064) < 1e-6
print("  scope=all 恢复旧行为 OK")
print("ALL TESTS PASSED")
