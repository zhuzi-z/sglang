import os, subprocess, sys
from pathlib import Path
# stub 仅在子进程安装，pytest 收集不会加载模拟包或 torch。
if __name__ == "__main__":
    os.environ["_SGLANG_RELEASE_CHILD"] = "1"
from importlib import metadata, util
def _native_source(package, relative):
    fixture = os.environ.get("RELEASE_NATIVE_SOURCE_ROOT")
    if fixture:
        return Path(fixture) / relative
    try:
        return Path(metadata.distribution(package).locate_file(relative))
    except metadata.PackageNotFoundError:
        return Path(__file__).parent / "assets" / "native_sources" / relative
_NATIVE_FILES = (
    ("vllm", "vllm/distributed/kv_transfer/kv_connector/v1/v6d_object_connector.py"),
    ("v6d", "v6d/client/client.py"),
    ("v6d", "v6d/server/peers/vineyard/peer.py"),
)
def _native_missing():
    missing = [str(_native_source(p, r)) for p, r in _NATIVE_FILES if not _native_source(p, r).is_file()]
    missing += [p for p in ("v6d", "orjson") if util.find_spec(p) is None]
    return ", ".join(missing)
if __name__ == "__main__" and _native_missing():
    raise SystemExit("36 项原生回归未运行，缺少原生源码或依赖: " + _native_missing())
if os.environ.get("_SGLANG_RELEASE_CHILD") == "1":
    import ast, asyncio, copy, os, sys, types, unittest
    from pathlib import Path
    from unittest.mock import patch
    import test_release_regressions as base
    V=base.ROOT
    N=_native_source("vllm", "vllm/distributed/kv_transfer/kv_connector/v1/v6d_object_connector.py")
    def native_class(path, name, methods):
        tree=ast.parse(path.read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==name)
        cls=copy.deepcopy(cls)
        cls.bases=[]
        cls.body=[x for x in cls.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in methods]
        assert set(methods)=={x.name for x in cls.body}
        header=ast.parse("from __future__ import annotations").body
        module=ast.fix_missing_locations(ast.Module(body=header+[cls],type_ignores=[]))
        ns={"logger":base.logging.getLogger("native-locality-test")}
        for exception in ("ObjectBusyError","ObjectNotFoundError","ObjectFetchingError"):
            ns[exception]=type(exception,(Exception,),{})
        exec(compile(module,str(path),"exec"),ns)
        return ns[name]
    Manager=native_class(N,"V6dObjectManager",["_make_key","_hold_key_for_req","_process_lookup","lookup","async_lookup","get_key","async_get_key","cleanup_request"])
    Scheduler=native_class(N,"V6dObjectConnectorScheduler",["_prepare_fetch","_promote_fetched_objs","_fetch_intersection_blocks","_async_fetch_intersection_blocks"])
    Obj=native_class(_native_source("v6d", "v6d/client/client.py"),"Object",["complete"])
    def obj(key, location, size=100, scope="CREATE"):
        o=Obj()
        o.key=key
        o.meta={"location":location,"size":size}
        o._lease=types.SimpleNamespace(scope=scope,seal_count=0,seal_target_count=1,completed_object_keys=[])
        o._lease_id=key
        o._blob=types.SimpleNamespace(complete=lambda:None)
        o._client=types.SimpleNamespace(seal=lambda *a,**k:None)
        return o
    class Client:
        def __init__(self, objects):
            self.objects=objects
            self.calls=[]
        def get(self, keys, request_id=None):
            self.calls.append((keys,request_id))
            if isinstance(keys,str):
                return self.objects.get(keys)
            return [self.objects.get(k) for k in keys]
        async def async_get(self, keys, request_id=None):
            await asyncio.sleep(0)
            return self.get(keys,request_id)
    class NativeLocalityTests(base.ControllerTests):
        def setUp(self):
            super().setUp()
            old=self.controller
            cache=Scheduler()
            cache.__dict__.update(old.cache.__dict__)
            cache.num_groups=2
            cache.group_block_sizes={0:1,1:1}
            cache.hash_block_size=1
            cache.mamba_group_ids=set()
            cache.managers={}
            for gid in range(2):
                m=Manager()
                m._group_id=gid
                m._cached_objs={}
                m._cached_objs_reqs={}
                m.client=Client({})
                cache.managers[gid]=m
            def fetch(objects):
                for o in objects:
                    o.complete()
                return None
            async def async_fetch(objects):
                await asyncio.sleep(0)
                return fetch(objects)
            cache._fetch_helper=types.SimpleNamespace(fetch=fetch,async_fetch=async_fetch)
            cache._record_fetch_hit_counts=lambda *a:None
            cache.on_abort_req=lambda reqid,**kw:"aborted"
            cache._cleanup_req_tracking=lambda req_id:None
            old.hybrid._backend._scheduler=cache
            self.cache=cache
            self.controller=self.backend.SimulatedKVController(old.hybrid)
            self.controller._load_done=old._load_done
            self.controller._store_done=old._store_done
            self.cache.mamba_group_ids={0}

        def model(self):
            base.bw.BandwidthModel._instance=base.bw.BandwidthModel(base.profile())

        def deadline(self, rid, groups):
            self.controller.queue_load(types.SimpleNamespace(request_id=rid),10,groups)
            return self.controller._loads[rid][0]

        def native_fetch(self, rid, hashes, objects, asynchronous=False):
            self.cache.mamba_group_ids=set()
            request=types.SimpleNamespace(request_id=rid)
            if asynchronous:
                return asyncio.run(self.cache._async_fetch_intersection_blocks(request,hashes,objects))
            return self.cache._fetch_intersection_blocks(request,hashes,objects)

        def test_native_remote_snapshot_before_complete(self):
            self.model()
            h=b"r"
            k=h.hex()+"_0"
            o=obj(k,"26.226.49.82:9600",350)
            self.cache.managers[0].client.objects[k]=o
            staging={}
            self.assertEqual(self.cache.managers[0].lookup([h],"r",staging),1)
            self.assertTrue(self.native_fetch("r",[h],staging))
            self.assertEqual(o.meta["location"],"local")
            self.assertEqual(self.cache.managers[0].get_key(h,"r"),k)
            self.controller._local_keys.add(k)
            self.assertAlmostEqual(self.deadline("r",{0:([k],[0])}),.555)
            self.assertNotIn("r",self.controller._native_locality)
            self.state.clock=.554
            self.assertEqual(self.controller.progress(),0)
            self.state.clock=.555
            self.assertEqual(self.controller.progress(),1)
            self.assertEqual(self.controller.progress(),0)

        def test_native_local_beats_empty_registry(self):
            self.model()
            h=b"l"; k=h.hex()+"_0"
            o=obj(k,"local",100,"READ")
            self.cache.managers[0].client.objects[k]=o
            self.assertEqual(self.cache.managers[0].get_key(h,"l"),k)
            with self.assertLogs("release-test",level="INFO") as logs:
                self.assertAlmostEqual(self.deadline("l",{0:([k],[0])}),.2)
            self.assertIn("source=native",logs.output[0])
            self.assertIn("remote_keys=0",logs.output[0])

        def test_native_mixed_only_remote_bytes(self):
            self.model()
            h=b"m"; keys=[h.hex()+"_0",h.hex()+"_1"]
            objects={keys[0]:obj(keys[0],"peer:9600",350),keys[1]:obj(keys[1],"local",9999,"READ")}
            self.assertTrue(self.native_fetch("mixed",[h],objects,True))
            with self.assertLogs("release-test",level="INFO") as logs:
                self.assertAlmostEqual(self.deadline("mixed",{0:([keys[0]],[0]),1:([keys[1]],[1])}),.755)
            self.assertIn("native=2 fallback=0 remote_keys=1 seg1_bytes=350",logs.output[0])

        def test_fallback_per_missing_key_only(self):
            self.model()
            self.controller._capture_native_key("f",0,"n",obj("n","local"))
            self.controller._local_keys.update(["known"])
            with self.assertLogs("release-test",level="INFO") as logs:
                t=self.deadline("f",{0:(["n","known","unknown"],[0,1,2])})
            self.assertAlmostEqual(t,.805)
            self.assertIn("source=fallback native=1 fallback=2 remote_keys=1",logs.output[0])

        def test_missing_native_bytes_only_falls_back_size(self):
            self.model()
            self.controller._capture_native_key("s",0,"x",obj("x","peer",0))
            self.assertAlmostEqual(self.deadline("s",{0:(["x"],[0])}),.405)

        def test_historical_remote_read_cache_is_local(self):
            self.model()
            h=b"old"; k=h.hex()+"_0"
            self.cache.managers[0]._cached_objs[k]=obj(k,"peer:9600",500,"READ")
            self.assertEqual(self.cache.managers[0].get_key(h,"old"),k)
            self.assertAlmostEqual(self.deadline("old",{0:([k],[0])}),.2)
            self.assertFalse(self.cache.managers[0].client.calls)

        def test_request_group_key_isolation(self):
            self.model()
            self.controller._capture_native_key("one",0,"x",obj("x","local"))
            self.assertAlmostEqual(self.deadline("two",{0:(["x"],[0])}),.405)
            self.assertAlmostEqual(self.deadline("one",{1:(["x"],[0])}),.405)
            self.assertFalse(self.controller._native_locality)

        def test_abort_and_native_cleanup_remove_pending(self):
            self.model()
            for rid in ("abort","cleanup","finish"):
                self.controller._capture_native_key(rid,0,"x",obj("x","peer"))
                self.deadline(rid,{0:(["x"],[0])})
                self.controller._capture_native_key(rid,1,"unloaded",obj("unloaded","peer"))
            self.assertEqual(self.cache.on_abort_req("abort"),"aborted")
            self.cache.managers[0].cleanup_request("cleanup")
            self.cache._cleanup_req_tracking("finish")
            self.assertFalse(self.controller._native_locality)
            self.assertFalse(self.controller._loads)
            self.state.clock=99
            self.assertEqual(self.controller.progress(),0)
            self.assertFalse(self.done)

        def test_reset_clears_unloaded_snapshots_and_registry(self):
            self.controller._capture_native_key("miss",0,"x",obj("x","peer"))
            self.controller._local_keys.add("stale")
            self.assertTrue(self.controller.reset())
            self.assertFalse(self.controller._native_locality)
            self.assertFalse(self.controller._local_keys)

        def test_async_get_key_native_read_has_request_id(self):
            self.model()
            h=b"async"; k=h.hex()+"_1"
            m=self.cache.managers[1]
            m.client.objects[k]=obj(k,"peer",800)
            self.assertEqual(asyncio.run(m.async_get_key(h,"async")),k)
            self.assertEqual(m.client.calls,[([k],"async")])
            self.assertAlmostEqual(self.deadline("async",{1:([k],[0])}),1.005)

        def test_async_preparation_keeps_deadline_and_native_lookup(self):
            self.model()
            h=b"p"; k=h.hex()+"_1"
            m=self.cache.managers[1]
            m.client.objects[k]=obj(k,"peer",500)
            async def prepare():
                staging={}
                self.assertEqual(await m.async_lookup([h],"p",staging),1)
                self.assertTrue(await self.cache._async_fetch_intersection_blocks(types.SimpleNamespace(request_id="p"),[h],staging))
                self.deadline("p",{1:([k],[0])})
            asyncio.run(self.controller.prepare(prepare()))
            self.controller.settle_preparations()
            self.assertFalse(self.controller._preparing)
            self.assertEqual(self.controller.progress(),0)
            self.assertAlmostEqual(self.controller._loads["p"][0],.705)
        def test_no_profile_native_bytes_safe_and_cleanup(self):
            self.controller._capture_native_key("np",0,"x",obj("x","peer",12345))
            self.assertEqual(self.deadline("np",{0:(["x"],[0])}),0)
            self.assertEqual(self.controller.progress(),1)
            self.assertFalse(self.controller._native_locality)
            self.assertEqual(self.controller.progress(),0)


        def test_abort_during_rpc_cannot_revive_state(self):
            self.model()
            h=b"cancel"; k=h.hex()+"_1"
            m=self.cache.managers[1]
            m.client.objects[k]=obj(k,"peer",500)
            async def scenario():
                entered=asyncio.Event(); release=asyncio.Event()
                async def prepare():
                    entered.set()
                    await release.wait()
                    await m.async_get_key(h,"cancel")
                    self.controller.queue_load(types.SimpleNamespace(request_id="cancel"),10,{1:([k],[0])})
                task=asyncio.create_task(self.controller.prepare(prepare(),"cancel"))
                await entered.wait()
                self.cache.on_abort_req("cancel")
                release.set()
                await task
            asyncio.run(scenario())
            self.controller.settle_preparations()
            self.assertFalse(self.controller._loads)
            self.assertFalse(self.controller._native_locality)
            self.assertFalse(self.controller._preparing_requests)
            self.assertFalse(self.controller._cancelled_preparations)
            self.controller._capture_native_key("cancel",1,k,obj(k,"local"))
            self.assertIn("cancel",self.controller._native_locality)

        def test_failed_native_fetch_discards_snapshot(self):
            h=b"fail"; k=h.hex()+"_1"
            o=obj(k,"peer",200)
            def fail():
                raise ValueError("injected seal failure")
            o.complete=fail
            self.assertFalse(self.native_fetch("fail",[h],{k:o},True))
            self.assertFalse(self.controller._native_locality)

        def test_production_profile_effective_50_percent(self):
            path=V.parents[4]/"profiles/bandwidth_profile_948d_production.json"
            with patch.dict(os.environ,{"SIMULATOR_HICACHE_BW_PROFILE":str(path)},clear=True):
                base.bw.BandwidthModel.reset()
                model=base.bw.BandwidthModel.get()
                self.assertTrue(model.enabled)
                self.assertEqual(model.local_load.bw,69793218560)
                self.assertEqual(model._seg1_bw_bytes_per_s,34896609280)
                self.assertEqual(model._seg1_bw_bytes_per_s/model.local_load.bw,.5)
                self.assertAlmostEqual(model.seg1_latency(1,nbytes=34896609280),1.022)
                config=base.json.loads(path.read_text())
                self.assertIn("非实测",config["control_plane"]["seg1_cross_node"]["_estimate"])
                print("EFFECTIVE_BW local=69793218560 seg1=34896609280 ratio=0.5 floor_ms=22")

        def test_native_bytes_respects_env_and_zero_blocks(self):
            self.model()
            model=base.bw.BandwidthModel.get()
            with patch.dict(os.environ,{"SGLANG_SIMULATOR_SEG1_FLOOR_MS":"9"}):
                self.assertAlmostEqual(model.seg1_latency(2,nbytes=500),.509)
            with patch.dict(os.environ,{"SGLANG_SIMULATOR_SEG1_PER_BLK_MS":"3"}):
                self.assertAlmostEqual(model.seg1_latency(2,nbytes=99999),.011)
            self.assertEqual(model.seg1_latency(0,nbytes=9999),0)


        def eager_wire(self, key, rid, peer="peer:9600", size=350, shared=None):
            import orjson
            from v6d.common.schemas import Scope, AcquireResponse
            from v6d.server.core.object import Object as ServerObject
            from v6d.server.models import object_info_from_object
            p=_native_source("v6d", "v6d/server/peers/vineyard/peer.py")
            Peer=native_class(p,"VineyardPeer",["acquire"])
            Peer.acquire.__globals__["Scope"]=Scope
            target=Peer()
            meta=shared if shared is not None else {"location":"local","size":size}
            async def cleanup(): pass
            target._cleanup_expired_leases=cleanup
            lease=types.SimpleNamespace(scope=Scope.READ)
            def local(keys,*args,**kwargs):
                return lease,[ServerObject(k,meta,[]) for k in keys]
            async def remote(keys,scope,term,target,rid):
                if target == "fault":
                    raise RuntimeError("native acquire failure")
                return local(keys,scope,term,request_id=rid)
            target._acquire_local_read=local
            target._acquire_remote_read=remote
            self.capacity.C_VineyardPeerHook.hook(Peer)
            _,objects=asyncio.run(target.acquire([key],None,Scope.READ,request_id=rid,peer=peer))
            info=object_info_from_object(objects[0])
            wire=orjson.dumps({"lease":{"lease_id":"l","scope":"READ"},"objects":[info]},default=lambda o:o.model_dump())
            response=AcquireResponse.model_validate(orjson.loads(wire))
            Wire=native_class(_native_source("v6d", "v6d/client/client.py"),"Object",["__init__","_reconstruct_blob","complete"])
            return Wire(None,response.lease,response.objects[0])

        def test_eager_remote_local_location_rpc_read_cache(self):
            self.model()
            h=b"eager"; k=h.hex()+"_0"; rid="native-0"
            o=self.eager_wire(k,rid)
            self.assertEqual(o.meta["location"],"local")
            self.cache.managers[0]._cached_objs[k]=o
            self.assertEqual(self.cache.managers[0].get_key(h,rid),k)
            self.assertAlmostEqual(self.deadline(rid,{0:([k],[0])}),.555)


        def test_eager_repeat_preparation_and_next_cache_local(self):
            self.model()
            h=b"repeat"; k=h.hex()+"_0"; rid="repeat-0"
            o=self.eager_wire(k,rid)
            staging={k:o}
            self.assertTrue(self.native_fetch(rid,[h],staging))
            for _ in range(3):
                self.assertTrue(self.native_fetch(rid,[h],{}))
                self.cache.managers[0].get_key(h,rid)
            self.cache.managers[0]._cached_objs[k]=self.eager_wire(k,rid,"local")
            self.assertTrue(self.native_fetch(rid,[h],{}))
            self.assertAlmostEqual(self.deadline(rid,{0:([k],[0])}),.555)
            self.cache.managers[0].get_key(h,"next-0")
            with self.assertLogs("release-test",level="INFO") as logs:
                self.assertAlmostEqual(self.deadline("next-0",{0:([k],[0])}),.2)
            self.assertIn("remote_keys=0 seg1_bytes=0",logs.output[0])
            self.assertFalse(self.cache.managers[0].client.calls)

        def test_eager_metadata_clone_and_next_local_rpc(self):
            self.model()
            shared={"location":"local","size":350}
            before=copy.deepcopy(shared)
            remote=self.eager_wire("x","first-0",shared=shared)
            note=copy.deepcopy(remote.meta["_sim_native_read_origin"])
            self.assertEqual(shared,before)
            local=self.eager_wire("x","second-0",peer="local",shared=shared)
            self.assertEqual(remote.meta["_sim_native_read_origin"],note)
            self.assertEqual(local.meta["_sim_native_read_origin"]["peer"],"local")
            self.assertEqual(shared,before)
            self.controller._capture_native_key("first-0",0,"x",remote,True)
            self.controller._capture_native_key("second-0",0,"x",local,True)
            self.assertAlmostEqual(self.deadline("first-0",{0:(["x"],[0])}),.555)
            self.assertAlmostEqual(self.deadline("second-0",{0:(["x"],[0])}),.2)
            shared["_sim_native_read_origin"]=note
            old=copy.deepcopy(shared)
            local=self.eager_wire("x","third-0",peer="local",shared=shared)
            self.assertEqual(shared,old)
            self.assertEqual(local.meta["_sim_native_read_origin"]["request_id"],"third-0")
            self.assertEqual(local.meta["_sim_native_read_origin"]["peer"],"local")

        def test_eager_mixed_only_selected_remote_bytes(self):
            self.model()
            h=b"mixed"; k0=h.hex()+"_0"; k1=h.hex()+"_1"; extra=b"extra".hex()+"_0"
            staging={k0:self.eager_wire(k0,"mixed-0"),k1:self.eager_wire(k1,"mixed-0","local",9999),extra:self.eager_wire(extra,"mixed-0",size=88888)}
            self.assertTrue(self.native_fetch("mixed-0",[h],staging,True))
            self.assertIn(extra,staging)
            self.assertNotIn((0,extra),self.controller._native_locality["mixed-0"])
            with self.assertLogs("release-test",level="INFO") as logs:
                self.assertAlmostEqual(self.deadline("mixed-0",{0:([k0],[0]),1:([k1],[1])}),.755)
            self.assertIn("native=2 fallback=0 remote_keys=1 seg1_bytes=350",logs.output[0])

        def test_eager_current_note_survives_native_complete(self):
            self.model()
            wire=self.eager_wire("x","complete-0")
            o=obj("x","peer:9600",350)
            o.meta["_sim_native_read_origin"]=wire.meta["_sim_native_read_origin"]
            o.complete()
            self.assertEqual(o.meta["location"],"local")
            self.controller._capture_native_key("complete-0",0,"x",o,True)
            self.assertAlmostEqual(self.deadline("complete-0",{0:(["x"],[0])}),.555)
            self.controller._capture_native_key("complete-next-0",0,"x",o,True)
            self.assertAlmostEqual(self.deadline("complete-next-0",{0:(["x"],[0])}),.2)

        def test_eager_request_key_version_mismatch_is_not_remote(self):
            self.model()
            o=self.eager_wire("x","native-0")
            for field,value in (("request_id","native"),("key","other"),("version",0),("peer",None)):
                with self.subTest(field=field):
                    note=copy.deepcopy(o.meta["_sim_native_read_origin"])
                    o.meta["_sim_native_read_origin"][field]=value
                    self.controller._capture_native_key("native-0",0,"x",o,True)
                    self.assertAlmostEqual(self.deadline("native-0",{0:(["x"],[0])}),.2)
                    o.meta["_sim_native_read_origin"]=note

        def test_eager_missing_annotation_keeps_meta_and_registry_fallback(self):
            self.model()
            o=self.eager_wire("x","missing-0")
            o.meta.pop("_sim_native_read_origin")
            self.controller._capture_native_key("missing-0",0,"x",o,True)
            self.assertAlmostEqual(self.deadline("missing-0",{0:(["x"],[0])}),.2)
            o.meta["location"]="peer:9600"
            self.controller._capture_native_key("missing-meta-0",0,"x",o)
            self.assertAlmostEqual(self.deadline("missing-meta-0",{0:(["x"],[0])}),.555)
            o.meta.pop("location")
            self.controller._capture_native_key("missing-all-0",0,"x",o)
            self.assertAlmostEqual(self.deadline("missing-all-0",{0:(["x"],[0])}),.405)

        def test_eager_abort_during_matched_note_rpc_no_revival(self):
            self.model()
            o=self.eager_wire("x","abort-note-0")
            async def scenario():
                entered=asyncio.Event(); release=asyncio.Event()
                async def prepare():
                    entered.set(); await release.wait()
                    self.controller._capture_native_key("abort-note-0",0,"x",o,True)
                    self.controller.queue_load(types.SimpleNamespace(request_id="abort-note-0"),10,{0:(["x"],[0])})
                task=asyncio.create_task(self.controller.prepare(prepare(),"abort-note-0"))
                await entered.wait()
                self.cache.on_abort_req("abort-note-0")
                release.set(); await task
            asyncio.run(scenario())
            self.controller.settle_preparations()
            self.assertFalse(self.controller._native_locality)
            self.assertFalse(self.controller._loads)
            self.state.clock=99
            self.assertEqual(self.controller.progress(),0)
            self.assertFalse(self.done)

        def test_eager_failed_fetch_and_cleanup_forget_note(self):
            h=b"failure"; k=h.hex()+"_0"; rid="failure-0"
            o=self.eager_wire(k,rid)
            def fail(): raise ValueError("injected native completion failure")
            o.complete=fail
            self.assertFalse(self.native_fetch(rid,[h],{k:o},True))
            self.assertFalse(self.controller._native_locality)
            self.controller._capture_native_key(rid,0,k,o,True)
            self.cache.managers[0].cleanup_request(rid)
            self.assertFalse(self.controller._native_locality)
            self.assertEqual(self.controller.progress(),0)

        def test_eager_no_profile_current_note_safe_once(self):
            o=self.eager_wire("x","np-note-0")
            self.controller._capture_native_key("np-note-0",0,"x",o,True)
            self.assertEqual(self.deadline("np-note-0",{0:(["x"],[0])}),0)
            self.assertEqual(self.controller.progress(),1)
            self.assertEqual(self.controller.progress(),0)
            self.assertFalse(self.controller._native_locality)


        def test_eager_native_acquire_fault_propagates_without_note(self):
            shared={"location":"local","size":350}
            with self.assertRaisesRegex(RuntimeError,"native acquire failure"):
                self.eager_wire("x","fault-0","fault",shared=shared)
            self.assertEqual(shared,{"location":"local","size":350})
            self.assertFalse(self.controller._native_locality)

    if __name__=="__main__":
        unittest.main(verbosity=2)

else:
    import pytest
    @pytest.mark.skipif(bool(_native_missing()), reason="36 项原生回归未运行，缺少源码或依赖: " + _native_missing())
    def test_native_locality_subprocess():
        result = subprocess.run([sys.executable, str(Path(__file__).resolve())], capture_output=True, text=True, timeout=25)
        assert result.returncode == 0, result.stdout + result.stderr
