# PP0 scheduler-step collection

Enable the existing `SIM_COLLECTOR_ENABLE=1`, `SIM_COLLECTOR_MODE=pp_v3`,
`SIM_TPU_NATIVE_BOUNDARIES=1` and set `SIM_TRACE_FOCUS=pp0_step`.
Keep `SIM_TPU_DEVICE_PROBE=0`. No hardware profiler or new device wait is used.

## Boundaries

* T1: original TPUWorker.execute_model entry, recorded separately on all 8 PP0 PCP workers. Step summary takes the minimum.
* T2: return from the existing PPWave local synchronize(None,wait=True), BEFORE its global CPU barrier. Associate the helper thread with its frozen chunk identity. Step summary takes the maximum over every PP0 PCP worker and every chunk. This is device completion observed on host, not a hardware execution endpoint.
* Handoff: original handoff submit returns, after the existing barrier. This is submission, not PP1 receive completion.
* T3: successful sample_tokens future observed on the PP0 controller after reply aggregation. A custom done callback verifies success and a non-None result. The old `_sampled` callback also fires on failed futures and alone is not proof of success. Registration is after original dispatch; an already-completed future is observed at registration time.

T1..T2 includes input preparation, host submission, queuing, execution and completion-notification delay. T1..T3 additionally includes the rest of the pipeline, sampling, output aggregation and controller observation. T3 is not the final decode response to the client. Requests sharing a chunk share a window; never attribute the full duration as isolated execution to every request.

## Minimal records and features

All worker/control startup and installation diagnostics remain. Only PP0 workers record execute entry, native completion and handoff submission; only PP0 PCP0 snapshots chunk inputs. The controller records step inputs, request add/scheduled metadata and successful-result observation. Other runner, queue, output and stage events are filtered before recorder timestamp allocation and buffering. The native observer also skips filtered marks before timestamping. Some existing Python wrappers still execute; zero overhead is not claimed.

Per-request chunk features: req_id, input_length (runner CPU prompt length), past_length (computed KV tokens), query_length (tokens scheduled now), num_output_tokens, max_output_tokens and logprobs. Per chunk: request count, query sum, valid/padded global/local tokens and request indices. Scheduler request metadata is retained separately because its prompt-length semantics can differ from runner input_length.

## Files and export

Continuous raw events live in trace_v3/<host>/<pid-process>/pending/.../events.*.jsonl. Both start_profile and stop_profile cut/export accumulated records; neither starts or stops hardware profiling. Start is async; stop waits for receipts. Exports contain events.*.jsonl, process/export metadata and four tables: pp0_points.jsonl, step_inputs.jsonl, chunk_inputs.jsonl and requests.jsonl. No old schedule_batch/iter_latency table is generated in this focus mode.

The complete step table must join 8 worker processes and the controller OFFLINE. `pp0_steps.build_steps([(metadata, events), ...])` produces pp0_steps.jsonl rows; callers persist the returned list. Provide all relevant cuts when spans cross exports. No cross-process aggregation is added to serving's hot path.

Rows include T1/T2/handoff/T3, PP0/whole-step/remaining-pipeline windows in ms, chunk/request features and per-rank evidence. `valid` requires all 8 starts, all rank/chunk completions and handoffs, complete input features, matching chunk/step token sums, a successful result, ordered timestamps, and identical verified host clock domain + boot_id + time namespace. Missing evidence is explicit in issues; never silently accept partial spans. valid_pure_device_timing and per_request_isolated_latency remain false.

## Validation and rollback

Host tests cover 8-rank multi-chunk joins, interleaved steps, failed futures, clock mismatch, missing rank/features, token mismatch, both export triggers and original native wait-call preservation. Real TPU validation must separately prove requests succeed and rows satisfy these contracts. Long input/multiple chunk and overhead A/B remain independent acceptance items until exercised. To revert, restore only the captured collector dependency and SIM fields; preserve current resource/model/topology settings.
