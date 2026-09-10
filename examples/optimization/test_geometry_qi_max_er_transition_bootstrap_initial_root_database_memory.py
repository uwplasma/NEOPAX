#!/usr/bin/env python
"""Repeated-evaluation memory boundary test for the live NTX database path.

No SciPy optimization is run.  Each trial evaluates the same geometry vector
through the database-native selected-root reverse path, including its one
database-to-scan transpose.  This is intentionally separate from the exact
Lij staged-lane memory test.
"""

from __future__ import annotations

import argparse
import gc
import io
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
import optimize_geometry_qi_max_er_transition_bootstrap_initial_root as base_example  # noqa: E402


DATABASE_TRANSPORT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml"
)


class QuietProblem:
    """Suppress existing progress output; test output stays one line per trial."""

    def __init__(self, problem):
        self._problem = problem

    def evaluate(self, values):
        with redirect_stdout(io.StringIO()):
            return self._problem.evaluate(values)


def _live_jax_array_count() -> int | None:
    live_arrays = getattr(jax, "live_arrays", None)
    if live_arrays is None:
        return None
    try:
        return len(live_arrays())
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("--warmup must be non-negative and --repeats must be positive.")
    if not np.isscalar(base_example.MAX_MODE_SCHEDULE):
        raise ValueError("The database memory test requires one fixed MAX_MODE_SCHEDULE value.")

    # The benchmark/root evaluator contains the database-native one-fold
    # reverse boundary.  The exact-Lij staged/JIT mode cannot be selected
    # here because it owns an exact support tree.
    base_example.TRANSPORT_CONFIG = DATABASE_TRANSPORT_CONFIG
    base_example.REVERSE_STAGE_MODE = "database"
    base = base_example
    problem = base.build_transition_bootstrap_initial_root_problem(
        base.SEED_INPUT, int(base.MAX_MODE_SCHEDULE)
    )
    quiet_problem = QuietProblem(problem)
    x = np.asarray(jax.device_get(problem.x0), dtype=float)
    print(
        "[database memory test] "
        f"warmup={args.warmup} repeats={args.repeats} "
        f"parameter_count={problem.parameter_count} "
        "path=ntx_scan_runtime_database_selected_root_reverse",
        flush=True,
    )
    for warmup_index in range(args.warmup):
        print(f"[database memory test] warmup={warmup_index} starting", flush=True)
        started = time.perf_counter()
        evaluation = quiet_problem.evaluate(x)
        jax.block_until_ready((evaluation.residuals, evaluation.jacobian))
        del evaluation
        gc.collect()
        print(
            f"[database memory test] warmup={warmup_index} complete "
            f"elapsed_s={time.perf_counter() - started:.3f}",
            flush=True,
        )

    first_rss: int | None = None

    def report(sample) -> None:
        nonlocal first_rss
        if first_rss is None:
            first_rss = sample.resident_memory_bytes
        delta = (
            None
            if first_rss is None or sample.resident_memory_bytes is None
            else (sample.resident_memory_bytes - first_rss) / 2**20
        )
        rss_text = "unavailable" if delta is None else f"{delta:+.1f} MiB"
        arrays = _live_jax_array_count()
        array_text = "unavailable" if arrays is None else str(arrays)
        print(
            f"[database memory test] trial={sample.iteration} "
            f"elapsed_s={sample.elapsed_s:.3f} rss_delta={rss_text} "
            f"live_jax_arrays={array_text} residual_norm={sample.residual_norm:.6e}",
            flush=True,
        )

    opt.repeated_evaluation_memory_samples(
        quiet_problem,
        warmup=0,
        repeats=args.repeats,
        scaled_parameter_values=x,
        on_sample=report,
    )
    print("[database memory test] complete; SciPy was not run.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
