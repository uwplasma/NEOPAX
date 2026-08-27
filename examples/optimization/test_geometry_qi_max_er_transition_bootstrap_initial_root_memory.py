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
from dataclasses import replace
from contextlib import redirect_stdout
from pathlib import Path
import sys
import time
from unittest.mock import patch

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX import optimization as opt  # noqa: E402
from NEOPAX import _reverse_ad_optimization as reverse_optimization  # noqa: E402
import optimize_geometry_qi_max_er_transition_bootstrap_initial_root as example  # noqa: E402
from optimize_geometry_qi_max_er_transition_bootstrap_initial_root import (  # noqa: E402
    MAX_MODE_SCHEDULE,
    SEED_INPUT,
    build_transition_bootstrap_initial_root_problem,
)


DEFAULT_WARMUP = 1
DEFAULT_REPEATS = 8


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


class EvaluationStructureCounter:
    """Count the shared primal/root boundaries without changing them.

    This is test-only monkeypatching: the wrappers delegate directly to the
    benchmark functions and exist solely to establish the call topology of one
    least-squares evaluation.
    """

    def __init__(self) -> None:
        self.raw_block_solve_calls = 0
        self.selected_root_calls = 0

    def reset(self) -> None:
        self.raw_block_solve_calls = 0
        self.selected_root_calls = 0

    def context(self):
        raw_block_solve = reverse_optimization.geometry_raw_block_solve_from_param_vector
        selected_root = reverse_optimization.initial_er_selected_root_profile

        def count_raw_block_solve(*args, **kwargs):
            self.raw_block_solve_calls += 1
            return raw_block_solve(*args, **kwargs)

        def count_selected_root(*args, **kwargs):
            self.selected_root_calls += 1
            return selected_root(*args, **kwargs)

        return (
            patch.object(
                reverse_optimization,
                "geometry_raw_block_solve_from_param_vector",
                count_raw_block_solve,
            ),
            patch.object(
                reverse_optimization,
                "initial_er_selected_root_profile",
                count_selected_root,
            ),
        )


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
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--diagnose-structure",
        action="store_true",
        help="Report benchmark-path raw VMEC solve and selected-root call counts per evaluation.",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("--warmup must be non-negative and --repeats must be positive.")
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
    structure_counter = EvaluationStructureCounter()

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
            f"residual_norm={sample.residual_norm:.6e}"
            + (
                " "
                f"raw_block_solve_calls={structure_counter.raw_block_solve_calls} "
                f"selected_root_calls={structure_counter.selected_root_calls}"
                if args.diagnose_structure
                else ""
            ),
            flush=True,
        )

    print(
        f"[memory test] mode={args.mode} objectives=maxEr,Er_left,Er_right,J_bootstrap "
        f"warmup={args.warmup} repeats={args.repeats} parameter_count={problem.parameter_count}",
        flush=True,
    )
    for warmup_index in range(args.warmup):
        print(f"[memory test] warmup={warmup_index} starting", flush=True)
        started = time.perf_counter()
        structure_counter.reset()
        patchers = structure_counter.context() if args.diagnose_structure else ()
        for patcher in patchers:
            patcher.start()
        try:
            evaluation = quiet_problem.evaluate(x)
        finally:
            for patcher in reversed(patchers):
                patcher.stop()
        jax.block_until_ready((evaluation.residuals, evaluation.jacobian))
        del evaluation
        gc.collect()
        print(
            f"[memory test] warmup={warmup_index} complete "
            f"elapsed_s={time.perf_counter() - started:.3f}",
            flush=True,
        )
    if args.diagnose_structure:
        for iteration in range(args.repeats):
            structure_counter.reset()
            patchers = structure_counter.context()
            for patcher in patchers:
                patcher.start()
            try:
                samples = opt.repeated_evaluation_memory_samples(
                    quiet_problem,
                    warmup=0,
                    repeats=1,
                    scaled_parameter_values=x,
                )
            finally:
                for patcher in reversed(patchers):
                    patcher.stop()
            report(replace(samples[0], iteration=iteration))
    else:
        opt.repeated_evaluation_memory_samples(
            quiet_problem,
            warmup=0,
            repeats=args.repeats,
            scaled_parameter_values=x,
            on_sample=report,
        )
    print("[memory test] complete; SciPy was not run.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
