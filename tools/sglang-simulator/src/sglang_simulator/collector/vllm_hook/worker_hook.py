import os
import time
import torch
import json
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass, field, asdict

from sglang_simulator.hook import BaseHook


@dataclass
class RequestInfos:
    rid: str = ""
    created_time: Optional[float] = None
    client_created_time: Optional[float] = None  # client created time
    server_created_time: Optional[float] = None
    queue_start: float = 0.0
    queue_end: float = 0.0
    output_length: int = 0
    input_length: int = 0
    final_device_hit_len: int = 0
    local_kv_hit_len: int = 0
    ext_kv_hit_len: int = 0
    input_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)


SCHEDULE_INFOS: list[dict] = []
REQUEST_INFOS: dict[str, RequestInfos] = defaultdict(RequestInfos)
# RPC-2 (sample_tokens) window timings; aligns 1:1 with SCHEDULE_INFOS
# strictly by call order.
SAMPLE_TOKENS_LATENCIES: list[float] = []



class C_VLLMEngineArgsHook(BaseHook):
    """Hook EngineArgs to default the profiler to nsys (cudaProfilerApi)."""

    HOOK_CLASS_NAME = "EngineArgs"
    HOOK_MODULE_NAME = "vllm.engine.arg_utils"

    @classmethod
    def hook(cls, target):
        original_post_init = target.__post_init__

        def wrapped_post_init(self):
            original_post_init(self)
            if self.profiler_config.profiler is None:
                self.profiler_config.profiler = "cuda"

        target.__post_init__ = wrapped_post_init



class C_WorkerWrapperBaseHook(BaseHook):

    HOOK_CLASS_NAME = "WorkerWrapperBase"    
    HOOK_MODULE_NAME = "vllm.v1.worker.worker_base"

    @classmethod
    def hook(cls, target) -> None:

        original_execute_model = target.execute_model

        def wrapped_execute_model(self, scheduler_output: "SchedulerOutput"):

            request_infos = {}
            for req_id, sched_token in scheduler_output.num_scheduled_tokens.items():
                request_infos[req_id] = {"extend_input_len": sched_token}
            
            for req_id, completed_token, output_token in zip(
                scheduler_output.scheduled_cached_reqs.req_ids, 
                scheduler_output.scheduled_cached_reqs.num_computed_tokens,
                scheduler_output.scheduled_cached_reqs.num_output_tokens
            ):
                request_infos[req_id]["prefix_indices_len"] = completed_token - output_token
                request_infos[req_id]["output_ids_len"] = output_token
            
            for req in scheduler_output.scheduled_new_reqs:
                request_infos[req.req_id]["prefix_indices_len"] = req.num_computed_tokens
                request_infos[req.req_id]["output_ids_len"] = 0

            forward_mode = 2 if all([num_token == 1 for num_token in scheduler_output.num_scheduled_tokens.values()]) else 1

            # Per-step logprobs form (for independent logprobs-compensation
            # training). Only new reqs carry sampling_params in SchedulerOutput;
            # cached reqs (decode steps) would need engine-side hook or a
            # differential stress test to capture. total_num_scheduled_tokens
            # is the step-level token count used as the MTP-compensation feature.
            logprobs_req_count = 0
            logprobs_tokens = 0
            logprobs_n = 0
            for req in scheduler_output.scheduled_new_reqs:
                sp = getattr(req, "sampling_params", None)
                if sp is not None and getattr(sp, "logprobs", None):
                    logprobs_req_count += 1
                    logprobs_tokens += scheduler_output.num_scheduled_tokens.get(
                        req.req_id, 0
                    )
                    logprobs_n = max(logprobs_n, sp.logprobs)

            torch.cuda.synchronize()
            start = time.time()
            ret = original_execute_model(self, scheduler_output)
            torch.cuda.synchronize()
            end = time.time()

            if len(request_infos):
                for rid, req_info in request_infos.items():
                    req_info["rid"] = rid

                SCHEDULE_INFOS.append(
                    {
                        "start_timestamp": start,
                        "end_timestamp": end,
                        "forward_mode": forward_mode,
                        "request_infos": list(request_infos.values()),
                        "iter_latency": end - start,
                        "total_tokens": scheduler_output.total_num_scheduled_tokens,
                        "logprobs_req_count": logprobs_req_count,
                        "logprobs_tokens": logprobs_tokens,
                        "logprobs_n": logprobs_n,
                    }
                )

            return ret
        
        target.execute_model = wrapped_execute_model



class C_WorkerHook(BaseHook):
    HOOK_MODULE_NAME = "vllm.v1.worker.gpu_worker"
    HOOK_CLASS_NAME = "Worker"

    @classmethod
    def hook(cls, target) -> None:

        # Two-window collection (X1): if this vllm version has sample_tokens
        # (two-RPC architecture), clamp it with cuda.sync to measure the full
        # RPC-2 span (sampler + MTP draft forward + bookkeeping D2H + output
        # construction). It aligns 1:1 by call order with the execute_model
        # window (RPC-1, measured by C_WorkerWrapperBaseHook).
        # Skipped automatically on older vllm without this method; the
        # iter_latency semantics remain unchanged.
        original_sample_tokens = getattr(target, "sample_tokens", None)
        if original_sample_tokens is not None:

            def wrapped_sample_tokens(self, *args, **kwargs):
                torch.cuda.synchronize()
                start = time.time()
                ret = original_sample_tokens(self, *args, **kwargs)
                torch.cuda.synchronize()
                SAMPLE_TOKENS_LATENCIES.append(time.time() - start)
                return ret

            target.sample_tokens = wrapped_sample_tokens

        original_profile = getattr(target, "profile", None)

        def override_profile(self, *args, **kwrags):
            # Drive the real profiler first (cudaProfilerStart/Stop when
            # profiler_config.profiler == "cuda") so `nsys profile
            # -c cudaProfilerApi` captures the collection window, then
            # export the collected data.
            if original_profile is not None:
                is_start = args[0] if args else kwrags.get("is_start", True)
                try:
                    original_profile(self, is_start)
                except RuntimeError:
                    # Profiling is not enabled: fall back to data export only.
                    pass

            SIM_COLLECT_INFO_DIR = os.path.join(
                os.getenv("SIM_COLLECT_INFO_DIR", os.getcwd()),
                str(os.getpid()),
            )

            rank_suffix = f"rank{self.rank}"

            n_st = len(SAMPLE_TOKENS_LATENCIES)
            with open(
                f"{SIM_COLLECT_INFO_DIR}/{rank_suffix}.schedule_batch.jsonl",
                "w",
            ) as f:
                for i, batch_infos in enumerate(SCHEDULE_INFOS):
                    # New fields, backward compatible: under the old hook /
                    # old vllm, i >= n_st and no extra field is written.
                    if i < n_st:
                        st_lat = SAMPLE_TOKENS_LATENCIES[i]
                        batch_infos["sample_tokens_latency"] = st_lat
                        il = batch_infos.get("iter_latency")
                        if il is not None:
                            # X1 label: full GPU span = RPC-1 + RPC-2
                            batch_infos["full_step_latency"] = il + st_lat
                    f.write(json.dumps(batch_infos) + "\n")

            print(f"Schedule batch data has been saved to {SIM_COLLECT_INFO_DIR}/{rank_suffix}.schedule_batch.jsonl")
            SCHEDULE_INFOS.clear()
            SAMPLE_TOKENS_LATENCIES.clear()

        target.profile = override_profile



class C_EngineCoreHook(BaseHook):

    HOOK_MODULE_NAME = "vllm.v1.engine.core"
    HOOK_CLASS_NAME = "EngineCore"

    @classmethod
    def hook(cls, target) -> None:

        original_profile = target.profile

        def wrapped_profile(self, is_start: bool = True):

            SIM_COLLECT_INFO_DIR = os.getenv(
                "SIM_COLLECT_INFO_DIR", os.getcwd()
            )

            with open(
                f"{SIM_COLLECT_INFO_DIR}/rank0.requests.jsonl", "w"
            ) as f:
                for req_infos in REQUEST_INFOS.values():
                    f.write(json.dumps(asdict(req_infos)) + "\n")

            print(
                f"Request data has been saved to "
                f"{SIM_COLLECT_INFO_DIR}/rank0.requests.jsonl"
            )
            REQUEST_INFOS.clear()

            return original_profile(self, is_start)

        target.profile = wrapped_profile



class C_SchedulerHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "vllm.v1.core.sched.scheduler"

    @classmethod
    def hook(cls, target) -> None:

        original_add_request = target.add_request
        original_schedule = target.schedule
        original_free_request = target._free_request

        def wrapped_add_request(self, request):
            recv_time = time.time()
            req_info = REQUEST_INFOS[request.request_id]
            req_info.rid = request.request_id
            req_info.queue_start = recv_time
            req_info.server_created_time = request.arrival_time
            req_info.created_time = request.arrival_time
            return original_add_request(self, request)

        def wrapped_schedule(self):
            scheduler_output = original_schedule(self)

            if scheduler_output.scheduled_new_reqs:
                prefill_timestamp = time.time()
                for new_req_data in scheduler_output.scheduled_new_reqs:
                    req_id = new_req_data.req_id
                    request = self.requests.get(req_id)
                    if request is None:
                        continue
                    req_info = REQUEST_INFOS[req_id]
                    if req_info.queue_end == 0:
                        req_info.queue_end = prefill_timestamp
                        req_info.input_length = request.num_prompt_tokens
                        req_info.output_length = request.max_tokens

                        req_info.final_device_hit_len = (
                            request.num_cached_tokens
                        )
                        req_info.ext_kv_hit_len = (
                            request.num_external_computed_tokens
                        )
                        req_info.local_kv_hit_len = (
                            request.num_cached_tokens
                            - request.num_external_computed_tokens
                        )

            return scheduler_output

        def wrapped_free_request(self, request):
            if request.is_finished():
                req_info = REQUEST_INFOS[request.request_id]
                req_info.input_ids = (
                    list(request.prompt_token_ids)
                    if request.prompt_token_ids is not None
                    else []
                )
                req_info.output_ids = list(request.output_token_ids)
            return original_free_request(self, request)

        target.add_request = wrapped_add_request
        target.schedule = wrapped_schedule
        target._free_request = wrapped_free_request


# ===================== HCDETECT: hybrid-connector admission deadlock =====================
# Detects the admission-time full-preallocation pin deadlock (see
# root_cause_analysis.md): _step_waiting preallocates + pins full-prompt blocks
# per request at admission; prefill needs a SECOND allocation; when pins
# exhaust the free pool nothing can prefill, save receipts never fire and the
# pins are never freed -> livelock.
#
# Additive-only: wraps HybridScheduler._step_waiting, swallows its own
# exceptions, and rate-limits every log level (time-monotonic gates), so the
# high-frequency spin during the deadlock cannot flood the log:
#   main gate 10s (state read at most 1/10s; every other call returns in O(1))
#   WARN 10s/条, ERROR 60s/条, INFO 60s/条  ->  worst case ~480 条/hour.

import time as _hcd_time
import logging as _hcd_log

_hcd_logger = _hcd_log.getLogger("vllm.hcdetect")

_HCD_GATE_S = 10.0   # main gate: full state read at most once per 10s
_HCD_WARN_S = 10.0   # WARN min interval
_HCD_ERR_S = 60.0    # ERROR min interval
_HCD_INFO_S = 60.0   # INFO min interval
_HCD_SNAP_S = 30.0   # no-progress snapshot window


def _hcd_sched():
    """V1Scheduler via engine_proxy global (EngineCore process only)."""
    from vllm.v1.hybrid_connector import engine_proxy
    core = getattr(engine_proxy, "_g_core", None)
    return getattr(core, "scheduler", None)


def _hcd_state(hs):
    """Full snapshot; returns dict or None on any failure."""
    sched = _hcd_sched()
    if sched is None:
        return None
    bp = sched.kv_cache_manager.block_pool
    # NOTE: bp.free_blocks is a *method* (freeing blocks), NOT the free
    # count. Free count = len(bp.free_block_queue); total = len(bp.blocks).
    fq = getattr(bp, "free_block_queue", None)
    try:
        free_b = len(fq) if fq is not None else -1
    except TypeError:
        free_b = getattr(fq, "num_free_blocks", -1)
    total_b = len(getattr(bp, "blocks", ()) or ())
    saving = hs._saving          # dict reqid -> _SavingReq
    waiting = hs._waiting        # deque of (req, load_count, save_count)
    pin_blks = 0
    pin_prompt = 0
    np_reqs = 0
    items = []
    for rid, sreq in saving.items():
        req = getattr(sreq, "_req", None)
        kvblks = getattr(sreq, "kvblks", None)
        nb = sum(len(g) for g in getattr(kvblks, "blocks", ()) or ()) \
            if kvblks is not None else 0
        pin_blks += nb
        pt = getattr(req, "num_prompt_tokens", 0) or 0
        pin_prompt += pt
        computed = getattr(req, "num_computed_tokens", None)
        if computed is None or computed == 0:
            np_reqs += 1
        items.append((pt, nb, computed or 0, rid))
    items.sort(reverse=True)
    n_run = len(getattr(sched, "running", ()) or ())
    n_ws = len(getattr(sched, "waiting", ()) or ())
    n_wc = len(waiting)
    head_ask = None
    if waiting:
        req0 = waiting[0][0]
        head_ask = ((getattr(req0, "num_prompt_tokens", 0) or 0)
                    - (getattr(req0, "num_computed_tokens", 0) or 0))
    ncfg = getattr(getattr(sched, "kv_cache_config", None),
                   "num_gpu_blocks", None)
    return dict(
        total_b=total_b, free_b=free_b, ncfg=ncfg,
        pin_blks=pin_blks, pin_prompt=pin_prompt, n_save=len(saving),
        np_reqs=np_reqs, items=items, n_run=n_run, n_ws=n_ws, n_wc=n_wc,
        head_ask=head_ask,
    )


def _hcd_detect(hs, st):
    """Called after each _step_waiting; rate-limited; never raises."""
    now = _hcd_time.monotonic()
    if now - st["gate"] < _HCD_GATE_S:
        return
    st["gate"] = now
    s = _hcd_state(hs)
    if s is None:
        return
    # ---- INFO: periodic snapshot (60s) ----
    if now - st["info"] >= _HCD_INFO_S:
        st["info"] = now
        _hcd_logger.info(
            "[HCDETECT] free=%d/%d blk | pinned=%d blk (%d reqs, np=%d, "
            "prompt=%d tok) | run=%d | wait sched=%d conn=%d",
            s["free_b"], s["total_b"], s["pin_blks"], s["n_save"],
            s["np_reqs"], s["pin_prompt"], s["n_run"], s["n_ws"], s["n_wc"])
    # ---- WARN: admission backpressure (10s) ----
    np_ratio = (s["np_reqs"] / s["n_save"]) if s["n_save"] else 0.0
    if (s["n_wc"] > 0 and s["head_ask"] is not None and s["n_save"] > 0
            and now - st["warn"] >= _HCD_WARN_S):
        st["warn"] = now
        _hcd_logger.warning(
            "[HCDETECT] admission backpressure: head asks %d tok, "
            "free=%d/%d blk, save_pinned=%d blk in %d reqs (np=%d), "
            "run=%d, wait sched=%d conn=%d",
            s["head_ask"], s["free_b"], s["total_b"], s["pin_blks"],
            s["n_save"], s["np_reqs"], s["n_run"], s["n_ws"], s["n_wc"])
    # ---- ERROR: deadlock verdict (60s) ----
    # free nearly exhausted (<5%) + most pins never prefilled + nothing
    # running + no progress vs snapshot >=30s old.
    free_ok = s["total_b"] > 0 and s["free_b"] * 20 < s["total_b"]
    snap = (s["free_b"], s["n_save"], s["n_run"])
    no_progress = False
    if st["snap"] is not None and now - st["snap_t"] >= _HCD_SNAP_S:
        no_progress = (snap == st["snap"])
    if now - st["snap_t"] >= _HCD_SNAP_S:
        st["snap"], st["snap_t"] = snap, now
    if (free_ok and np_ratio >= 0.8 and s["n_run"] == 0 and no_progress
            and now - st["err"] >= _HCD_ERR_S):
        st["err"] = now
        top = ", ".join("%s p=%d blk=%d c=%d" % (r[:12], p, b, c)
                        for p, b, c, r in s["items"][:5])
        _hcd_logger.error(
            "[HCDETECT] HYBRID-CONNECTOR DEADLOCK SUSPECTED: save-pinned "
            "blocks exhausted free pool, no progress in %ds (pinned requests "
            "can never prefill -> save receipt never fires -> pinned blocks "
            "never freed). L1_total=%d blk (cfg num_gpu_blocks=%s) free=%d "
            "blk | save_pinned(v6d)=%d blk in %d reqs, never-prefilled=%d/%d, "
            "prompt_sum=%d tok | running(prefill)=%d | waiting: sched=%d "
            "conn=%d head_ask=%d tok | top5_stuck: %s",
            _HCD_SNAP_S, s["total_b"], s["ncfg"], s["free_b"], s["pin_blks"],
            s["n_save"], s["np_reqs"], s["n_save"], s["pin_prompt"],
            s["n_run"], s["n_ws"], s["n_wc"], s["head_ask"] or 0, top)


class C_HybridSchedulerHook(BaseHook):
    HOOK_CLASS_NAME = "HybridScheduler"
    HOOK_MODULE_NAME = "vllm.v1.hybrid_connector"

    @classmethod
    def hook(cls, target) -> None:
        original_step_waiting = target._step_waiting
        st = {"gate": 0.0, "warn": 0.0, "err": 0.0, "info": 0.0,
              "snap": None, "snap_t": 0.0}

        def wrapped_step_waiting(self):
            ret = original_step_waiting(self)
            try:
                _hcd_detect(self, st)
            except Exception:
                pass
            return ret

        target._step_waiting = wrapped_step_waiting
