# Four-stage lightweight continuous collection

Enable SIM_COLLECTOR_ENABLE=1, SIM_COLLECTOR_MODE=pp_v3,
SIM_TRACE_FOCUS=pp_stage, SIM_TPU_NATIVE_BOUNDARIES=1. Keep
SIM_TPU_DEVICE_PROBE=0. Existing pp0_step and full modes are preserved.

## Scope and non-interference

Only host observations of existing native calls are added. No device sync,
Event.record, tensor data read, compiler call, or hardware profiler is introduced.
Native exceptions and return values are preserved; observers fail open. Recorder
uses bounded nonblocking buffering and background serialization/write. Dropped,
rejected and observer-error counters must be checked; no zero-overhead claim.

All 32 workers retain execute/sample entry/exit and required native boundaries.
Only PCP0 of each stage records request inputs, compact layout and static config.
The controller records inputs, dispatch context, push/settle and successful result
observation. request features include req_id, query_length, pre-chunk past_length,
input_length, output counts, logprobs and length-derived phase/prompt completion.
Those derived flags describe lengths, not a guarantee that sampling ran.

## Protocol and payload

Native handoffs include process-local launch_id, native burst_id and observed
launch_ordinal. The ordinal counts actual _launch_blocking calls, not the
preincremented async helper enqueue count. Initial compiler/warmup handoffs stay silent. After a real step activates the
wave observer, untracked/zero-carry protocol marks (including that prologue)
are retained with context=null and protocol_session; never assign them to the next real chunk.
Helper carried identity is frozen at enqueue as in the previous implementation.

Each launch has payload_bytes when dtype width is known; shape/dtype metadata
is emitted only when its signature changes. Only tensor.shape and tensor.dtype
are read. Bytes are logical operand sizes, not physical wire/protocol traffic.
Unknown dtype produces unavailable bytes, never a guessed value.

## Export and offline aggregation

Both start_profile and stop_profile cut accumulated records and keep collecting;
start is asynchronous, stop waits for receipts. No hardware profiling is invoked.
Raw events also continuously exist under pending windows without a profile call.

Exports contain requests, step_inputs, chunk_inputs, layouts, stage_configs,
stage_points, stage_rank_windows, protocol_points and controller_points JSONL.
There is no legacy schedule_batch/2ms iter_latency table in this focus mode.

To build one sample per real stage/chunk, read process.json and all relevant raw
cuts, then call stage_collection.build_samples([(metadata, events), ...]). Save
its returned list as stage_samples.jsonl. This operation is OFFLINE; no cross-rank
aggregation runs inside inference workers.

Sample validity requires all8 rank starts/completions, complete request/step joins,
token/prefix consistency, ordered timestamps, and verified same-stage host clocks.
PP0–2 additionally require drain and handoff boundaries. PP3 completion scope
includes sampling/copy and is kept explicitly different. Multiple chunks that
lack independently observable completion stay invalid; no fabricated endpoints.
Duplicates across cuts are deduplicated by process+sequence, while actual duplicate
boundaries, missing ranks, clock mismatch and observer failures remain issues.

## Time and training contract

observed_stage_window_ms is host forward dispatch to completion observed on host.
It is NOT pure device service. Per-rank evidence separately records Python submit,
residual native wait, delay before native drain, global barrier wait and handoff
submit observation. None of these is claimed to be exact device communication.

All samples retain target_service_s=null, valid_service_label=false,
valid_pure_device_timing=false, device_elapsed_ns=null and device_queue_ns=null.
This release supplies four-stage inputs and causal boundaries, not validated
service labels. Before fitting a CPU deadline service model, validate/derive a
service target that does not double-count separately simulated waits. Never
rename this observation window into a validated service label.

Cross-stage hosts do not share an established monotonic clock; do not subtract
timestamps between machines. Protocol IDs and controller completion provide
causal joins. Static configuration is CPU metadata, emitted once per stage PCP0.

## Acceptance

Check all32 ranks startup, source SHA, all4 stage sample coverage and 8PCP evidence,
request-ID/token joins, observer/drop counters, protocol zero launches and payload
metadata; then inspect finite real requests and deployment restarts. Host tests
alone do not establish TPU runtime correctness or acceptable overhead. A/B,
nonzero-KV coverage and pure service timing remain separate validation items.
