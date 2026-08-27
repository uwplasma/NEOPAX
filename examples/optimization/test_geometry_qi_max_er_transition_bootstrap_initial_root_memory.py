#!/usr/bin/env python
"""Memory test for geometry plus all four initial-Er root objective rows.

The problem is built by the matching optimization example, so it uses its
max-Er, left/right transition, and bootstrap-current terms and root options.
No SciPy iteration is run.
"""

from __future__ import annotations

import gc
import io
import argparse
from contextlib import redirect_stdout
from pathlib import Path
import sys
import time

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX import optimization as opt  # noqa: E402
import optimize_geometry_qi_max_er_transition_bootstrap_initial_root as example  # noqa: E402
from optimize_geometry_qi_max_er_transition_bootstrap_initial_root import (  # noqa: E402
    MAX_MODE_SCHEDULE,
    SEED_INPUT,
    build_transition_bootstrap_initial_root_problem,
)


WARMUP = 1
REPEATS = 8


class QuietProblem:
    """Suppress existing reverse-AD diagnostics only for this memory test."""

    def __init__(self, problem):
        self._problem = problem

    @property
    def x0(self):
        return self._problem.x0

    def evaluate(self, x):
        with redirect_stdout(io.StringIO()):
            return self._problem.evaluate(x)


def _live_jax_array_count() -> int | None:
    probe = getattr(jax, "live_arrays", None)
    if probe is None:
        return None
    try:
        return len(probe())
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("off", "vmex_like"), default="off")
    args = parser.parse_args()
    if not np.isscalar(MAX_MODE_SCHEDULE):
        raise ValueError("The memory test requires one fixed MAX_MODE_SCHEDULE value.")
    previous_mode = example.REVERSE_STAGE_MODE
    try:
        example.REVERSE_STAGE_MODE = args.mode
        problem = build_transition_bootstrap_initial_root_problem(SEED_INPUT, int(MAX_MODE_SCHEDULE))
    finally:
        example.REVERSE_STAGE_MODE = previous_mode
    quiet_problem = QuietProblem(problem)
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
        live_arrays = _live_jax_array_count()
        live_text = "unavailable" if live_arrays is None else str(live_arrays)
        print(
            f"[memory test] trial={sample.iteration} elapsed_s={sample.elapsed_s:.3f} "
            f"rss_delta={delta_text} live_jax_arrays={live_text} "
            f"residual_norm={sample.residual_norm:.6e}",
            flush=True,
        )

    print(
        f"[memory test] mode={args.mode} objectives=maxEr,Er_left,Er_right,J_bootstrap "
        f"warmup={WARMUP} repeats={REPEATS} parameter_count={problem.parameter_count}",
        flush=True,
    )
    for warmup_index in range(WARMUP):
        print(f"[memory test] warmup={warmup_index} starting", flush=True)
        started = time.perf_counter()
        evaluation = quiet_problem.evaluate(x)
        jax.block_until_ready((evaluation.residuals, evaluation.jacobian))
        del evaluation
        gc.collect()
        print(
            f"[memory test] warmup={warmup_index} complete "
            f"elapsed_s={time.perf_counter() - started:.3f}",
            flush=True,
        )
    opt.repeated_evaluation_memory_samples(
        quiet_problem,
        warmup=0,
        repeats=REPEATS,
        scaled_parameter_values=x,
        on_sample=report,
    )
    print("[memory test] complete; SciPy was not run.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
