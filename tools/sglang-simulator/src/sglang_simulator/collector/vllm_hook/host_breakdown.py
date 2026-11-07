"""Optional CPU accounting around EXISTING calls; never device timing.

Process CPU includes every process thread and may exceed wall time. Off-CPU
includes arbitrary waits and OS descheduling, not just accelerator waits.
Clock collection is fail-open and never changes the wrapped call's outcome.
"""
import os
import threading
import time

ENABLED = os.getenv('SIM_TPU_HOST_BREAKDOWN') == '1'


def snapshot():
    if not ENABLED:
        return None
    try:
        return (time.monotonic_ns(), time.thread_time_ns(),
                time.process_time_ns(), threading.get_native_id())
    except Exception:
        return None


def elapsed(start):
    end = snapshot() if start is not None else None
    if end is None:
        return {}
    wall, cpu, process = (end[i] - start[i] for i in range(3))
    return {'call_start_mono_ns': start[0], 'call_end_mono_ns': end[0],
            'call_wall_ns': wall, 'calling_thread_cpu_ns': cpu,
            'process_cpu_ns': process,
            'calling_thread_offcpu_ns': max(0, wall - cpu),
            'thread_native_id': start[3],
            'cpu_accounting_kind': 'host_clocks_not_device_time'}
