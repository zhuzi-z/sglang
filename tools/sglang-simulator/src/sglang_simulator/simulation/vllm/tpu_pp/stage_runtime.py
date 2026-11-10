"""TPU PP 逻辑流水线运行时：rank0 控制循环 + 每个 PP stage 一个 CPU worker 进程。

控制面完全复用原生代码：EngineCore.step_fn 换成 vllm_torchtpu 的
LazyPushStepper，stage 侧复用 PPWave 的 before_step/start_forward/
end_forward/push/settle。只替换硬件：
- 设备前向：stage worker 用本 stage 的 ML 模型预测时长，并把 device_ready
  推进到 max(查询时刻, 上一前向结束, 输入就绪) + 预测值；
- PP 通信：每条 PP 边一根单向 Pipe 传描述符，stage 间先过 WORLD 栅栏；
- 回复环：每个 stage 一根回复 Pipe，KV connector 存在时与原生一样等所有
  stage 回复，否则只等最后一个 stage。
ModelRunnerOutput 与 KV connector 簿记仍由内层 UniProcExecutor 同步产生，
其结果只在对应 stage 回复到达后才交给调度器。
"""

import dataclasses
import multiprocessing as mp
import os
import threading
import time
import traceback
from collections import deque

from sglang_simulator.simulation.vllm.tpu_pp import native_control
from sglang_simulator.simulation.vllm.tpu_pp.stage_model import (
    StageModel, StageSpan, empty_step_host_s, host_prep_s,
)
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")

_WORKER_READY_TIMEOUT_S = 300.0
_BARRIER_TIMEOUT_S = 60.0


@dataclasses.dataclass
class Descriptor:
    """代替 PP 边上的张量：只携带 step 身份与就绪时刻。"""

    step_id: int | None = None
    ready_at: float = 0.0

    def wait(self):
        remaining = self.ready_at - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    def clone(self):
        return dataclasses.replace(self)

    def __getitem__(self, key):
        return self.clone()

    def __add__(self, other):
        return self.clone()


@dataclasses.dataclass
class StagePayload:
    step_id: int
    num_tokens: int
    features: dict | None = None
    tail: dict | None = None
    host_prep_s: float | None = None
    empty_host_s: tuple | None = None


@dataclasses.dataclass
class WorkerConfig:
    stages: int
    model_paths: list
    communication_s: float
    async_handoff: bool
    execute_model_timeout_s: float
    record_events: bool = True


def _make_wave_class(base):
    class CpuWave(base):
        """原生 PPWave 方法原样使用，只覆盖硬件构造、下发与 padding。"""

        def __init__(self, rank, cfg, barrier, payloads, events):
            self.rank, self.size, self.last = rank, cfg.stages, cfg.stages - 1
            self.column, self.columns = 0, 1
            self.max_rows = 1
            self.carried = ("hidden_states",)
            self.drain_before_handoff = True
            # 原生异步交接只在 stage0 生效
            self.async_handoff = cfg.async_handoff and rank == 0
            self._async_cv = threading.Condition()
            self._async_carried = None
            self._async_busy = False
            self._async_error = self._async_out = self._async_thread = None
            self.zeros = (Descriptor(),)
            self.zero = Descriptor()
            self.launches = self.bursts = self.burst_launches = self.forwards = 0
            self.burst_open = False
            self.pending = None
            self.device_ready = 0.0
            self.barrier, self.payloads, self.events = barrier, payloads, events
            self.communication_s = cfg.communication_s
            self.record_events = cfg.record_events
            self.transport_burst, self.transport_index = 0, 0
            self.last_barrier_release = None

        def event(self, kind, **kwargs):
            if self.record_events:
                self.events.put(dict(kind=kind, stage=self.rank,
                                     time=time.monotonic(), **kwargs))

        def _pad(self, carried, rows):
            return carried

        def _launch_blocking(self, carried):
            # 原生顺序：本地设备排空 -> WORLD 栅栏 -> 提交 PP permute
            if self.transport_burst != self.bursts:
                self.transport_burst, self.transport_index = self.bursts, 0
            key = (self.bursts, self.transport_index)
            self.transport_index += 1
            Descriptor(ready_at=self.device_ready).wait()
            self.barrier.wait(timeout=_BARRIER_TIMEOUT_S)
            self.last_barrier_release = time.monotonic()
            incoming_pipe, outgoing_pipe = self.payloads
            outgoing_pipe.send((key, carried))
            incoming_key, incoming = incoming_pipe.recv()
            if incoming_key != key:
                raise AssertionError(
                    f"launch mismatch: expected {key}, received {incoming_key}")
            ready = time.monotonic() + self.communication_s
            return tuple(dataclasses.replace(d, ready_at=ready) for d in incoming)

    return CpuWave


def worker_main(rank, cfg, control, reply, barrier, payloads, events):
    native = native_control.load(cfg.async_handoff, cfg.execute_model_timeout_s)
    wave = _make_wave_class(native.PPWave)(rank, cfg, barrier, payloads, events)
    model = StageModel.from_file(cfg.model_paths[rank], rank, cfg.stages - 1)
    reply.send((0, True, {"ready": rank, "pid": os.getpid()}))
    pending = None
    try:
        while True:
            call_id, method, arg, wants_reply = control.recv()
            if method == "shutdown":
                wave._async_wait()
                break
            if method == "execute_model":
                payload = arg
                wave.before_step(payload.num_tokens)
                if payload.num_tokens:
                    if payload.host_prep_s is not None:
                        # 原生：stage 在 before_step 返回后串行构建本步输入
                        Descriptor(ready_at=time.monotonic()
                                   + payload.host_prep_s).wait()
                    incoming = wave.start_forward(1)
                    if rank:
                        desc = incoming["hidden_states"]
                        if desc.step_id != payload.step_id:
                            raise AssertionError(
                                f"stage {rank}: expected step {payload.step_id}, "
                                f"got {desc.step_id}")
                        input_ready = desc.ready_at
                    else:
                        input_ready = 0.0
                    query_at = time.monotonic()
                    features = payload.features
                    if rank == cfg.stages - 1:
                        features = {**features, **payload.tail}
                    duration = model.predict(features)
                    if not 0 <= duration < 60:
                        raise ValueError(f"invalid stage prediction {duration}")
                    # 起点已包含本次模型查询，不再额外叠加预测器 CPU 耗时
                    start = max(query_at, wave.device_ready, input_ready)
                    wave.device_ready = start + duration
                    wave.event("compute", step_id=payload.step_id,
                               start_at=start, end_at=wave.device_ready,
                               predicted_s=duration)
                    tensors = {"hidden_states": Descriptor(payload.step_id,
                                                           wave.device_ready),
                               "residual": Descriptor()}
                    wave.end_forward(None if rank == cfg.stages - 1 else tensors, 1)
                    pending = payload
                else:
                    # 原生空 step 在 before_step 之后仍串行执行 runner 簿记，占用本 stage worker
                    start = time.monotonic()
                    if payload.empty_host_s is not None:
                        Descriptor(ready_at=start + payload.empty_host_s[rank]).wait()
                    wave.event("empty", start_at=start, end_at=time.monotonic())
                result = payload.step_id
            elif method == "sample_tokens":
                if pending is None:
                    raise AssertionError("sample_tokens without execute_model")
                result, pending = pending.step_id, None
                if rank == cfg.stages - 1:
                    Descriptor(ready_at=wave.device_ready).wait()
            elif method == "pp_push":
                wave.push()
                result = "push"
            elif method == "pp_settle":
                wave.settle()
                result = "settle"
            else:
                raise ValueError(f"unknown stage method {method}")
            if wants_reply:
                reply.send((call_id, True, result))
    except BaseException:
        error = traceback.format_exc()
        events.put(dict(kind="worker_error", stage=rank, error=error,
                        time=time.monotonic()))
        barrier.abort()
        try:
            reply.send((-1, False, error))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise


class ReplyRing:
    """原生 MessageQueue 的远端读者形态：_reply_ready 只看 socket 是否可读。"""

    _is_remote_reader = True

    def __init__(self, connection):
        self.remote_socket = _MillisecondSocket(connection)


class _MillisecondSocket:
    """原生 remote_socket 是 zmq socket，poll 超时单位为毫秒。"""

    def __init__(self, connection):
        self.connection = connection

    def poll(self, timeout=0):
        return self.connection.poll(timeout / 1000.0)

    def recv(self):
        return self.connection.recv()


class VirtualPPExecutor:
    """以原生 MultiprocExecutor 的接口包住内层 UniProcExecutor。"""

    def __init__(self, inner, predictor, stages: int, on_step_done=None):
        self.inner = inner
        self._on_step_done = on_step_done
        self._req_info: dict[str, tuple] = {}
        self._closed = False
        self._workers, self._commands = [], []
        self.predictor = predictor
        self.stages = stages
        self.output_rank = stages - 1
        timeout = float(os.environ.get("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", 300))
        self.native = native_control.load(predictor.async_handoff, timeout)
        self._empty_host_s = empty_step_host_s(predictor.host_prep, stages)
        self.futures_queue = deque()
        self._tpu_wave_state = self.native.WaveState(stages)
        self._tpu_wave_state.lazy = True
        self._next_call = 0
        self._next_step = 0
        self._pending_sample = None
        self.step_records: dict[int, dict] = {}
        self.stage_events: list[dict] = []
        self._start_workers(timeout)
        self._patch_inner_collective_rpc()

    def _start_workers(self, timeout):
        ctx = mp.get_context("spawn")
        cfg = WorkerConfig(self.stages, self.predictor.stage_model_paths,
                           self.predictor.communication_s,
                           self.predictor.async_handoff, timeout)
        edges = [ctx.Pipe(duplex=False) for _ in range(self.stages)]
        self._barrier = ctx.Barrier(self.stages)
        self._events = ctx.Queue()
        self._commands, self._workers, readers = [], [], []
        for rank in range(self.stages):
            cmd_reader, cmd_writer = ctx.Pipe(duplex=False)
            rep_reader, rep_writer = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=worker_main, name=f"sim-pp{rank}", daemon=True,
                args=(rank, cfg, cmd_reader, rep_writer, self._barrier,
                      (edges[(rank - 1) % self.stages][0], edges[rank][1]),
                      self._events))
            proc.start()
            self._commands.append(cmd_writer)
            self._workers.append(proc)
            readers.append(rep_reader)
        self.response_mqs = [ReplyRing(r) for r in readers]
        deadline = time.monotonic() + _WORKER_READY_TIMEOUT_S
        for rank, reader in enumerate(readers):
            if not reader.poll(max(0.0, deadline - time.monotonic())):
                raise TimeoutError(f"TPU PP stage worker {rank} not ready")
            _, ok, value = reader.recv()
            if not ok:
                raise RuntimeError(f"TPU PP stage worker {rank} failed:\n{value}")
        self._collector = threading.Thread(
            target=self._collect_events, name="sim-pp-events", daemon=True)
        self._collector.start()
        logger.info("[TPU PP] %d stage workers ready: pids=%s", self.stages,
                    [p.pid for p in self._workers])

    def _collect_events(self):
        while True:
            event = self._events.get()
            if event is None:
                return
            if event["kind"] == "worker_error":
                logger.error("[TPU PP] stage %s failed:\n%s",
                             event["stage"], event["error"])
            self.stage_events.append(event)

    def _patch_inner_collective_rpc(self):
        # 原生 _patch_collective_rpc：非 step 调用前先 settle，并按序读完在途回复
        original = self.inner.collective_rpc
        step_methods = self.native.STEP_METHODS

        def collective_rpc(method, *args, **kwargs):
            if not (isinstance(method, str) and method in step_methods):
                self._tpu_wave_state.settle(self)
                self.drain()
            return original(method, *args, **kwargs)

        self.inner.collective_rpc = collective_rpc

    def drain(self):
        while self.futures_queue:
            self.futures_queue.pop()._wait_for_response()

    def __getattr__(self, name):
        # inner 未设置前（构造早期/反序列化）不能委托，否则无限递归
        if name == "inner" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def _aggregated(self) -> bool:
        return (getattr(self.inner, "kv_output_aggregator", None) is not None
                or getattr(self.inner, "ec_output_aggregator", None) is not None)

    def _stage_rpc(self, method, arg, all_ranks: bool, value):
        self._next_call += 1
        call_id = self._next_call
        ranks = range(self.stages) if all_ranks else (self.output_rank,)
        for rank, conn in enumerate(self._commands):
            conn.send((call_id, method, arg, rank in ranks))
        rings = tuple(self.response_mqs[r] for r in ranks)

        def response():
            for ring in rings:
                received, ok, result = ring.remote_socket.recv()
                if not ok:
                    raise RuntimeError(f"TPU PP stage worker failed:\n{result}")
                if received != call_id:
                    raise AssertionError((method, call_id, received))
            return value

        future = self.native.FutureWrapper(self.futures_queue, response)
        future._tpu_reply_rings = rings
        return future

    def collective_rpc(self, method, *args, non_block=False, **kwargs):
        if method in ("pp_push", "pp_settle"):
            future = self._stage_rpc(method, None, False, None)
            return future if non_block else future.result()
        return self.inner.collective_rpc(
            method, *args, non_block=non_block, **kwargs)

    def _spans(self, scheduler_output) -> list[StageSpan]:
        new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None) or ()
        computed = {r.req_id: r.num_computed_tokens for r in new_reqs}
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is not None:
            computed.update(zip(cached.req_ids, cached.num_computed_tokens))
        for r in new_reqs:
            prompt = getattr(r, "prompt_token_ids", None)
            sp = getattr(r, "sampling_params", None)
            self._req_info[r.req_id] = (
                len(prompt) if prompt is not None else None,
                bool(getattr(sp, "logprobs", None) is not None))
        for rid in getattr(scheduler_output, "finished_req_ids", None) or ():
            self._req_info.pop(rid, None)
        spans = []
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            prompt, logprobs = self._req_info.get(rid, (None, False))
            spans.append(StageSpan(int(n), max(0, int(computed.get(rid, 0))),
                                   prompt, logprobs))
        return spans

    def execute_model(self, scheduler_output, non_block=False):
        tokens = int(scheduler_output.total_num_scheduled_tokens)
        if tokens > 0:
            self._next_step += 1
            scheduler_output._sim_tpu_pp_step_id = self._next_step
        output = self.inner.execute_model(scheduler_output, non_block=False)
        if tokens <= 0:
            payload = StagePayload(0, 0, empty_host_s=self._empty_host_s)
            future = self._stage_rpc("execute_model", payload,
                                     self._aggregated(), output)
            return future if non_block else future.result()
        spans = self._spans(scheduler_output)
        features, tail = self.predictor.features_for(spans)
        step_id = self._next_step
        self._tpu_wave_state.dispatched()
        payload = StagePayload(step_id, tokens, features, tail,
                               host_prep_s(self.predictor.host_prep, spans))
        dispatched = time.monotonic()
        self.step_records[step_id] = dict(
            step_id=step_id, dispatched_at=dispatched,
            num_reqs=len(spans), num_tokens=tokens,
            spans=[(s.extend_length, s.past_kv_length) for s in spans])
        self._pending_sample = (step_id, output, scheduler_output)
        future = self._stage_rpc("execute_model", payload,
                                 self._aggregated(), None)
        return future if non_block else future.result()

    def sample_tokens(self, grammar_output, non_block=False):
        step_id, output, scheduler_output = self._pending_sample
        self._pending_sample = None
        sampled = self.inner.sample_tokens(grammar_output, non_block=False)
        if sampled is None:
            sampled = output
        future = self._stage_rpc("sample_tokens", None, self._aggregated(),
                                 sampled)
        record = self.step_records[step_id]
        on_done = self._on_step_done

        def done(_):
            record["sample_ready_at"] = time.monotonic()
            if on_done is not None:
                on_done(scheduler_output, time.time())

        future.add_done_callback(done)
        return future if non_block else future.result()

    def reset_stats(self):
        self.step_records.clear()
        self.stage_events.clear()

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        try:
            for conn in self._commands:
                try:
                    conn.send((0, "shutdown", None, False))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            for proc in self._workers:
                proc.join(5)
        finally:
            for proc in self._workers:
                if proc.is_alive():
                    proc.terminate()
            self._events.put(None)
            self.inner.shutdown()


def create_stepper(core, executor):
    native_stepper = executor.native.LazyPushStepper
    if hasattr(core, "capture_iteration_details"):
        return native_stepper(core)

    # 上游 vLLM 使用日志上下文，torchtpu fork 使用返回给前端的迭代详情。
    class LoggingStepper(native_stepper):
        def _process(self, step):
            with (
                core.log_error_detail(step.scheduler_output),
                core.log_iteration_details(step.scheduler_output),
            ):
                if not step.future.done():
                    method = "sample_tokens" if step.slot else "execute_model"
                    raise TimeoutError(f"RPC call to {method} timed out.")
                model_output = step.future.result()
                if model_output is None:
                    step.exec_future.result()
                    raise RuntimeError("unexpected error")
            core._process_aborts_queue()
            return core.scheduler.update_from_output(
                step.scheduler_output, model_output)

    return LoggingStepper(core)
