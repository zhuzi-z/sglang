import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


def test_real_engine_core_pp_lifecycle(tmp_path):
    pytest.importorskip("vllm")
    pytest.importorskip("sklearn")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", VLLM_ENABLE_V1_MULTIPROCESSING="0",
               VLLM_WORKER_MULTIPROC_METHOD="spawn", SGLANG_SIMULATOR_ENABLE="0",
               SGLANG_SIMULATOR_OUTPUT_MODE="BLOCKING", SGLANG_SIMULATOR_RECORD_STATS="1",
               SGLANG_SIMULATOR_OUTPUT_DIR=str(tmp_path),
               SGLANG_SIMULATOR_LIFECYCLE_DIR=str(tmp_path), HISIM_LIFECYCLE_ENABLE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
                            env=env, text=True, capture_output=True, timeout=180)
    (tmp_path / "engine.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    events = []
    for line in result.stderr.splitlines():
        if line.startswith("[HISIM_"):
            tag, payload = line.split("] ", 1)
            events.append((tag[1:], json.loads(payload)))
    ready = [event for tag, event in events if tag == "HISIM_ENGINE_READY"]
    assert len(ready) == 1
    engine_ready = ready[0]
    assert all(engine_ready[key] for key in (
        "virtual_pp", "native_stepper", "step_fn_bound", "batch_queue_bound"))
    assert engine_ready["inner_class"] == "UniProcExecutor"
    assert engine_ready["stages"] == 4 and engine_ready["pcp"] == 8
    worker_ready = [event for tag, event in events if tag == "HISIM_WORKER_READY"]
    assert len(worker_ready) == 1 and worker_ready[0]["pid"] == engine_ready["pid"]
    assert all(worker_ready[0][key] for key in (
        "receiver_model_runner_stub", "receiver_init_device_match", "receiver_load_model_match", "receiver_execute_model_match"))
    loads = [event for tag, event in events if tag == "HISIM_WORKER_LOAD_MODEL"]
    assert len(loads) == 1 and loads[0]["pid"] == engine_ready["pid"]
    assert loads[0]["status"] == "skipped" and loads[0]["receiver_model_runner_stub"]
    stages = [event for tag, event in events if tag == "HISIM_STAGE_READY"]
    assert sorted(event["stage"] for event in stages) == list(range(4))
    assert len({event["pid"] for event in stages}) == 4
    assert all(event["ppid"] == engine_ready["pid"] for event in stages)
    configs = [event for tag, event in events if tag == "HISIM_ENGINE_CONFIG"]
    config_stages = {event["stage"] for event in configs}
    assert config_stages >= {"engine_args_input", "engine_args", "engine_core"}
    preserve_native = env.get("HISIM_TPU_PP_PRESERVE_DASHLLM_ENGINE_ARGS") == "1"
    assert ("create_engine_config" in config_stages) == (not preserve_native)
    stage = "engine_core" if preserve_native else "create_engine_config"
    created = next(event for event in configs if event["stage"] == stage)
    assert (worker_ready[0]["delegated_worker_module"] + "." +
            worker_ready[0]["delegated_worker_class"]) == created["worker_cls"]
    reports = [event for tag, event in events if tag == "HISIM_HOOK_REPORT"
               and event["stage"] == "engine_ready"]
    matches = [row for row in reports[0]["hooks"]
               if row["hook_name"] == "C_VLLMWorkerHook" and
               row["target_module"] + "." + row["target_class"] == created["worker_cls"]]
    assert len(matches) == 1 and matches[0]["status"] == "applied"


def _exercise(root):
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
        "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 4,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
        "vocab_size": 256, "max_position_embeddings": 8192, "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6, "tie_word_embeddings": False,
        "bos_token_id": 1, "eos_token_id": 2, "torch_dtype": "float16",
    }))
    paths = json.loads(os.environ.get("HISIM_TPU_PP_STAGE_MODELS", "[]"))
    if not paths:
        import joblib
        from sklearn.dummy import DummyRegressor
        from sglang_simulator.simulation.vllm.tpu_pp.stage_model import (
            FEATURE_NAMES, TAIL_FEATURES, MODEL_KIND,
        )
        for k in range(4):
            names = FEATURE_NAMES + (TAIL_FEATURES if k == 3 else [])
            estimator = DummyRegressor(strategy="constant", constant=0.01)
            estimator.fit([[0.0] * len(names)], [0.01])
            path = root / f"stage-{k}.pkl"
            joblib.dump(dict(model_kind=MODEL_KIND, production_usable=False,
                             stage=k, features=names, time_unit="seconds",
                             models=[estimator]), path)
            paths.append(str(path))
    prep_path = os.environ.get("HISIM_TPU_PP_HOST_PREP")
    config = {
        "platform": {"accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80}},
        "scheduler": {"tp_size": 1, "dp_size": 1, "ep_size": 1,
                      "pp_size": 4, "pcp_size": 8},
        "predictor": {"name": "tpu_pp_stage", "stage_model_paths": paths,
                      "pcp_size": 8, "interleave_size": 128,
                      "num_tokens_paddings": [256, 512, 1024],
                      "communication_s": 0.0008, "async_handoff": True,
                      "host_prep": json.loads(Path(prep_path).read_text()) if prep_path else None},
    }
    config_path = root / "config_pp4.json"
    config_path.write_text(json.dumps(config))
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(config_path)
    from sglang_simulator.simulation.vllm.startup import init_hook
    init_hook()
    from vllm import LLM, SamplingParams
    from sglang_simulator.simulation.vllm.engine_core_pipeline import C_VLLMEngineCoreHook
    from sglang_simulator.simulation.req_stats_manager import request_stats_manager
    from sglang_simulator.simulation.manager import StateManager

    started = time.monotonic()
    llm = LLM(model=str(model), skip_tokenizer_init=True, dtype="float16",
              load_format="dummy", enforce_eager=True, async_scheduling=False,
              max_model_len=8192, max_num_batched_tokens=2048, max_num_seqs=16,
              num_gpu_blocks_override=4096, block_size=16,
              enable_prefix_caching=True, disable_log_stats=False)
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    executor = core.model_executor
    assert type(executor).__name__ == "VirtualPPExecutor"
    assert type(executor.inner).__name__ == "UniProcExecutor"
    assert C_VLLMEngineCoreHook.TPU_PP_ACTIVE
    assert core.step_fn.__self__ is core._sim_tpu_pp_stepper
    assert not hasattr(core, "_sim_pipeline_stepper")

    def forbidden_whole_step(*args, **kwargs):
        raise AssertionError("old whole-step predictor was called")

    executor.predictor.predict_infer_time = forbidden_whole_step
    params = SamplingParams(max_tokens=3, ignore_eos=True, temperature=0, detokenize=False)
    rounds = []
    try:
        assert engine.collective_rpc("get_supported_tasks") == [("generate",)]
        for lengths, offset in (([512, 768, 1024, 1280], 3),
                                ([256 + (i % 4) * 128 for i in range(16)], 16),
                                ([6144, 4096, 2048], 40)):
            before = set(executor.step_records)
            prompts = [{"prompt_token_ids": [offset + i] * n} for i, n in enumerate(lengths)]
            outputs = llm.generate(prompts, params, use_tqdm=False)
            assert len(outputs) == len(prompts)
            assert all(o.finished and len(o.outputs[0].token_ids) == 3 for o in outputs)
            records = [v for k, v in executor.step_records.items() if k not in before]
            assert records and all(r["num_tokens"] <= 2048 for r in records)
            rounds.append({"lengths": lengths, "steps": len(records),
                           "request_ids": [o.request_id for o in outputs]})

        repeated = {"prompt_token_ids": [80] * 1024}
        llm.generate([repeated], params, use_tqdm=False)
        previous_requests = set(request_stats_manager.stats)
        llm.generate([repeated], params, use_tqdm=False)
        cache_stats = [st for rid, st in request_stats_manager.stats.items()
                       if rid not in previous_requests]
        assert len(cache_stats) == 1
        hit = cache_stats[0].local_kv_hit_len
        assert hit >= 1008

        abort_request_id = engine.add_request("abort-pp", {"prompt_token_ids": [81] * 4096}, params)
        engine.abort_request([abort_request_id], internal=True)
        assert not engine.has_unfinished_requests()

        request_id = engine.add_request("profile-inflight", {"prompt_token_ids": [82] * 4096}, params)
        core._sim_tpu_pp_stepper.step()
        assert core.batch_queue
        iteration = StateManager.get_iteration()
        before_export = (len(executor.step_records), len(C_VLLMEngineCoreHook.ITERATION_STATS),
                         len(request_stats_manager.stats), len(core.batch_queue), iteration)
        lifecycle_status = core.export_lifecycle(wait=True)
        assert before_export == (len(executor.step_records), len(C_VLLMEngineCoreHook.ITERATION_STATS),
                                 len(request_stats_manager.stats), len(core.batch_queue),
                                 StateManager.get_iteration())
        if lifecycle_status["enabled"]:
            assert lifecycle_status["flushed"] and not lifecycle_status["error"]
        core.profile(False)
        assert StateManager.get_iteration() == iteration
        deadline = time.monotonic() + 60
        finished = []
        while engine.has_unfinished_requests():
            assert time.monotonic() < deadline
            finished.extend(o for o in engine.step() if o.finished)
        assert len(finished) == 1 and len(finished[0].outputs[0].token_ids) == 3
        engine.collective_rpc("get_supported_tasks")
        stepper = core._sim_tpu_pp_stepper
        while stepper.in_flight or stepper.wave.open or executor.futures_queue:
            assert time.monotonic() < deadline
            stepper.step()
        assert not core.batch_queue
        assert not stepper.wave.open
        assert not executor.futures_queue
        records = list(executor.step_records.values())
        assert records and all("sample_ready_at" in r for r in records)
        core.profile(False)
        assert not executor.step_records
        assert not executor.stage_events
        assert not C_VLLMEngineCoreHook.ITERATION_STATS
        assert not request_stats_manager.stats
        assert StateManager.get_iteration() == 0
        exported = {name: [json.loads(line) for line in (root / name).read_text().splitlines()]
                    for name in ("request.jsonl", "iteration.jsonl", "tpu_pp_steps.jsonl",
                                 "tpu_pp_stage_events.jsonl")}
        assert len(exported["tpu_pp_steps.jsonl"]) == len(records)
        iterations = {r["tpu_pp_step_id"] for r in exported["iteration.jsonl"]}
        assert iterations == {r["step_id"] for r in records}
        active_request = [r for r in exported["request.jsonl"] if r["rid"] == request_id]
        assert len(active_request) == 1
        assert len(active_request[0]["gen_token_latencies"]) == 3
        computes = [e for e in exported["tpu_pp_stage_events.jsonl"] if e["kind"] == "compute"]
        assert len(computes) == 4 * len(records)
        assert not [e for e in exported["tpu_pp_stage_events.jsonl"] if e["kind"] == "worker_error"]
        lifecycle_checks = {}
        if lifecycle_status["enabled"]:
            from sglang_simulator.simulation.vllm.lifecycle import request_key
            status = core.export_lifecycle(wait=True)
            assert status["flushed"] and status["dropped"] == 0 and not status["error"]
            rows = [json.loads(line) for line in Path(status["path"]).read_text().splitlines()]
            scheduled = {r["step_id"]: r for r in rows if r["event"] == "scheduler.scheduled"}
            dispatches = [r for r in rows if r["event"] == "pp.dispatch"]
            ready = {r["step_id"]: r for r in rows if r["event"] == "pp.sample_ready"}
            accepted = [r for r in rows if r["event"] == "request.output_accepted"]
            updates = [r for r in rows if r["event"] == "scheduler.output_update"]
            assert len(updates) == len(records)
            assert {r["step_id"] for r in updates} == set(scheduled)
            assert all(r["native_ok"] and r["end_mono_ns"] >= r["start_mono_ns"] for r in updates)
            for dispatch in dispatches:
                step = scheduled[dispatch["step_id"]]
                assert dispatch["requests"] == step["requests"]
                assert step["membership_complete"]
                assert ready[dispatch["step_id"]]["sample_ready_at"] >= dispatch["dispatched_at"]
            completed = {r["request_id_sha256"] for r in accepted if r["finished"]}
            assert len(completed) == 26
            for key in completed:
                outputs = [r for r in accepted if r["request_id_sha256"] == key]
                assert sum(r["new_token_count"] for r in outputs) == 3
                assert all(r["monotonic_ns"] >= ready[r["step_id"]]["monotonic_ns"] for r in outputs)
            aborts = [r for r in rows if r["event"] == "request.logical_finish"
                      and r["request_id_sha256"] == request_key(abort_request_id)]
            assert len(aborts) == 1 and aborts[0]["status"] == "FINISHED_ABORTED"
            assert aborts[0]["observed_output_tokens"] == 0
            aliases = [r for r in rows if r["event"] == "request.identity_alias"]
            abort_aliases = [r for r in aliases if r["request_id_sha256"] == request_key(abort_request_id)]
            assert len(abort_aliases) == 1
            assert abort_aliases[0]["external_request_id_sha256"] == request_key("abort-pp")
            assert len(aliases) == 27
            intermediate = {r["step_id"] for r in rows if r["event"] == "scheduler.scheduled"
                            and any(not m["prefill_complete_after_forward"] for m in r["requests"])}
            assert intermediate
            for output in accepted:
                step_rows = scheduled[output["step_id"]]["requests"]
                member = next(m for m in step_rows if m["request_id_sha256"] == output["request_id_sha256"])
                assert member["prefill_complete_after_forward"] or output["new_token_count"] == 0
            text = Path(status["path"]).read_text()
            assert "prompt_token_ids" not in text and "new_token_ids" not in text
            assert "abort-pp" not in text and "profile-inflight" not in text
            from sglang_simulator import request_boundaries
            frontend_recorder = request_boundaries._SIM_RECORDER
            assert frontend_recorder is not None and frontend_recorder.flush()
            frontend_status = frontend_recorder.status()
            assert frontend_status["dropped"] == 0 and not frontend_status["error"]
            frontend_rows = [json.loads(line) for line in Path(frontend_status["path"]).read_text().splitlines()]
            assert {r["process_id"] for r in rows} == {r["process_id"] for r in frontend_rows}
            capabilities = [r for r in frontend_rows if r["event"] == "request.boundary_capability"]
            expected_sources = {
                "vllm.v1.engine.input_processor.InputProcessor.process_inputs",
                "vllm.v1.engine.input_processor.InputProcessor.assign_request_id",
                "vllm.v1.engine.output_processor.OutputProcessor.process_outputs",
                "vllm.v1.engine.output_processor.RequestState.make_request_output",
            }
            assert expected_sources <= {r["source"] for r in capabilities if r["installed"]}
            frontend_aliases = [r for r in frontend_rows if r["event"] == "request.identity_alias"]
            assert {(r["request_id_sha256"], r["external_request_id_sha256"]) for r in aliases} == {
                (r["request_id_sha256"], r["external_request_id_sha256"]) for r in frontend_aliases}
            materialized = [r for r in frontend_rows if r["event"] == "request.output_materialization"
                            and r["output_materialized"]]
            assert completed <= {r["request_id_sha256"] for r in materialized}
            output_batches = [r for r in frontend_rows if r["event"] == "request.output_processed"]
            assert output_batches
            assert all(r["engine_core_output_count"] > 0 for r in output_batches)
            assert all(r["returned_output_count"] is not None for r in output_batches)
            cache_capabilities = [r for r in rows if r["event"] == "request.cache_capabilities"]
            assert {r["component"] for r in cache_capabilities} >= {"cache_manager", "scheduler_transitions"}
            cache_manager = next(r for r in cache_capabilities if r["component"] == "cache_manager")
            assert {"get_computed_blocks", "allocate_slots", "free"} <= set(cache_manager["installed_methods"])
            for event in ("request.kv_lookup", "request.kv_allocation", "request.kv_release_call",
                          "pp.inner_execute", "pp.host_prepare", "pp.inner_sample", "request.engine_output_wait",
                          "request.engine_output_result"):
                assert any(r["event"] == event and r["native_ok"] for r in rows), event
            assert any(r["event"] == "request.engine_output_wait"
                       and r["source"] == "FutureWrapper._wait_for_response" for r in rows)
            lifecycle_checks = {"events": len(rows), "completed": len(completed),
                                "abort_observed": True, "chunked_membership": True,
                                "incremental_export_no_reset": True, "identity_aliases": len(aliases),
                                "frontend_events": len(frontend_rows), "frontend_identity_aliases": len(frontend_aliases),
                                "frontend_source_capabilities": sorted(expected_sources),
                                "frontend_dropped": frontend_status["dropped"],
                                "frontend_bytes": Path(frontend_status["path"]).stat().st_size,
                                "nonempty_output_batches": len(output_batches),
                                "empty_output_batch_events": sum(r["engine_core_output_count"] == 0 for r in output_batches),
                                "scheduler_output_updates": len(updates),
                                "cache_components": sorted(r["component"] for r in cache_capabilities),
                                "dropped": status["dropped"]}
        from sglang_simulator.simulation.vllm.tpu_pp import stage_runtime
        result = {"elapsed_s": time.monotonic() - started, "rounds": rounds,
                  "stage_runtime_path": stage_runtime.__file__, "completed_requests": 26,
                  "local_cache_hit_tokens": hit, "abort_passed": True,
                  "profile_inflight_passed": True, "profile_idle_reset_passed": True,
                  "exported_steps": len(records),
                  "compute_events": len(computes), "whole_step_predictor_calls": 0,
                  "stage_models": paths, "engine_core": type(core).__module__,
                  "scheduler": type(core.scheduler).__module__,
                  "lifecycle": lifecycle_checks}
        (root / "integration_result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    finally:
        core.shutdown()
        executor._collector.join(5)
    assert not executor._collector.is_alive()
    assert all(not worker.is_alive() for worker in executor._workers)


if __name__ == "__main__":
    _exercise(Path(sys.argv[1]))
