# PP host breakdown contract

Enable `SIM_TPU_HOST_BREAKDOWN=1` together with the existing pp_stage/native
boundary flags. Default off. No hardware profiler, tensor value reads, added
device waits, scheduling changes or hot-path disk writes.

Existing synchronous and asynchronous native calls retain their arguments,
call order, count, return values and exceptions. Optional CPU clocks fail open.
The observer itself consumes CPU and buffer space; this is not zero overhead.

## CPU fields on existing completion/error points

`native.local_drain.ready/error`, producer/handoff flush, dependency ready,
launch return and output-get return carry:

* `call_start_mono_ns`, `call_end_mono_ns`: host observations immediately around
  the original call (not accelerator clock values).
* `call_wall_ns`: observed call duration.
* `calling_thread_cpu_ns`: calling thread's scheduled user+system CPU duration.
* `calling_thread_offcpu_ns`: max(0, wall minus calling-thread CPU). This can
  include lock/condition/device/network wait and OS descheduling. It cannot
  identify device wait or pure device compute.
* `process_cpu_ns`: all process threads' CPU consumption, possibly greater than
  wall time due to parallel threads. Background writer/runtime threads included.
* `thread_native_id`: OS calling-thread ID, not an accelerator stream ID.
* `cpu_accounting_kind=host_clocks_not_device_time`.

Absent CPU fields mean disabled/unavailable/untracked context, never zero time.
Clock sampling adds small endpoint skew; very short calls can have CPU greater
than wall due to sequential clock reads. Do not force process CPU below wall.

`native.handoff_barrier_done.data.drain_call_accounting` is a NESTED measurement
of the entire original drain (local synchronize plus global CPU barrier and
observer overhead). It is not an independent barrier-only CPU measurement.
Previous local-ready to barrier-done wall span remains a host approximation.

## Additional points

* `native.helper_request.enter/return`: PP0 async handoff request call. The
  return is NOT an exact enqueue instant and the helper may already be running.
* `native.launch_call.enter/return`: existing blocking handoff implementation,
  includes drain/barrier/submission. `launch_id` joins native protocol points.
  `execution_thread` identifies actual `_async_thread` versus caller thread.
* `native.helper_fence.enter/return`: original async condition wait. Enter has
  best-effort `waited_context`; the current event context is the caller, which
  may differ from the previous chunk. Never interpret it as current input-ready.
* `native.output_get.return`: original output getter, including existing copy
  wait if present. PP3 sampling/copy is composite, not isolated decoder compute.
* Existing PP method, runner phase and wrapper points retained on PCP0 only.
  Existing native points and worker execute/sample boundaries remain on all
  eight PCP ranks. Controller execute/sample RPC dispatch points retained.

All new host points export into `stage_points.jsonl`, controller RPC into
`controller_points.jsonl`; raw pending segments remain continuously written.
Start/stop profile retain export-and-continue behavior. Both can cut spans;
join by process/sequence/context/chunk/launch, never concatenate and double count.

## Modelling

Use input-time-known request lengths, past KV, scheduled tokens, padding and
batch features for service predictors. Measured CPU/off-CPU/barrier durations
are diagnostics/labels, not available-at-dispatch features. An end-to-end model
may explicitly model host queue/CPU/barrier alongside service; do not train on
total observed wait and then add the same simulated barrier again.

`valid_service_label` stays false. These clocks distinguish where the host
spends time but cannot solve compute-versus-device-queue attribution inside
the same native synchronization. Matched runtime completion/queue callbacks
would be needed for that distinction; no full hardware trace is required by
this contract.
