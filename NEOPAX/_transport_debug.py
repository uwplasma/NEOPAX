from __future__ import annotations

import os
import threading
import time


_TIMER_LOCK = threading.Lock()
_TIMER_STACKS: dict[str, list[tuple[int, float]]] = {}
_TIMER_COUNTS: dict[str, int] = {}


def lagged_timing_enabled() -> bool:
    value = os.environ.get("NEOPAX_DEBUG_LAGGED_TIMING", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def lagged_timing_start(label: str) -> None:
    if not lagged_timing_enabled():
        return
    now = time.perf_counter()
    with _TIMER_LOCK:
        count = _TIMER_COUNTS.get(label, 0) + 1
        _TIMER_COUNTS[label] = count
        stack = _TIMER_STACKS.setdefault(label, [])
        stack.append((count, now))
    print(f"[lagged-timing] {label} start #{count}", flush=True)


def lagged_timing_end(label: str) -> None:
    if not lagged_timing_enabled():
        return
    now = time.perf_counter()
    with _TIMER_LOCK:
        stack = _TIMER_STACKS.get(label, [])
        if not stack:
            print(f"[lagged-timing] {label} end #? (no-start)", flush=True)
            return
        count, start = stack.pop()
    print(f"[lagged-timing] {label} end   #{count} dt={now - start:.3f}s", flush=True)
