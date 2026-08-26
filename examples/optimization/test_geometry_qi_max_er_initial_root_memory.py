#!/usr/bin/env python
"""Repeated-evaluation memory test for geometry plus initial-Er objectives.

This test intentionally does not call SciPy. It builds one optimizer problem,
warms it once, and evaluates the identical parameter vector eight times while
printing resident-memory data immediately after each completed evaluation.
"""

from __future__ import annotations

import gc
from pathlib import Path
import sys
import time

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX import optimization as opt  # noqa: E402
from optimize_geometry_qi_max_er_initial_root import (  # noqa: E402
    MAX_MODE_SCHEDULE,
    SEED_INPUT,
    build_initial_root_problem,
)


WARMUP = 1
REPEATS = 8


def main() -> int:
    if not np.isscalar(MAX_MODE_SCHEDULE):
        raise ValueError("The memory test requires one fixed MAX_MODE_SCHEDULE value.")
    problem = build_initial_root_problem(SEED_INPUT, int(MAX_MODE_SCHEDULE))
    x = np.asarray(jax.device_get(problem.x0), dtype=float)
    first_bytes: int | None = None

    def report(sample) -> None:
        nonlocal first_bytes
        if first_bytes is None:
            first_bytes = sample.resident_memory_bytes
        delta_mib = (
            None
            if first_bytes is None or sample.resident_memory_bytes is None
            else (sample.resident_memory_bytes - first_bytes) / 2**20
        )
        delta_text = "unavailable" if delta_mib is None else f"{delta_mib:+.1f} MiB"
        print(
            f"[memory test] trial={sample.iteration} elapsed_s={sample.elapsed_s:.3f} "
            f"rss_delta={delta_text} residual_norm={sample.residual_norm:.6e}",
            flush=True,
        )

    print(
        f"[memory test] warmup={WARMUP} repeats={REPEATS} "
        f"parameter_count={problem.parameter_count}",
        flush=True,
    )
    for warmup_index in range(WARMUP):
        print(f"[memory test] warmup={warmup_index} starting", flush=True)
        started = time.perf_counter()
        evaluation = problem.evaluate(x)
        jax.block_until_ready((evaluation.residuals, evaluation.jacobian))
        del evaluation
        gc.collect()
        print(
            f"[memory test] warmup={warmup_index} complete "
            f"elapsed_s={time.perf_counter() - started:.3f}",
            flush=True,
        )
    opt.repeated_evaluation_memory_samples(
        problem,
        warmup=0,
        repeats=REPEATS,
        scaled_parameter_values=x,
        on_sample=report,
    )
    print("[memory test] complete; SciPy was not run.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
