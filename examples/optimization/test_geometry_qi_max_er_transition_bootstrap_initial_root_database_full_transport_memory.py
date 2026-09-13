#!/usr/bin/env python
"""Repeated-memory test for four-step database full-transport reverse segments.

The selected initial-Er root is followed by four accepted Radau steps on the
small ``(5,25,31)`` database.  Each accepted step occupies one fixed reverse
segment.  This diagnostic intentionally does not require transport ``t_final``.
"""

from __future__ import annotations

import argparse
import gc
import io
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
import sys
import time

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX import optimization as opt  # noqa: E402
from NEOPAX import _reverse_ad_transport as reverse_transport  # noqa: E402
import test_geometry_qi_max_er_transition_bootstrap_initial_root_database_full_transport_parity as parity  # noqa: E402


def live_jax_array_count() -> int | None:
    live_arrays = getattr(jax, "live_arrays", None)
    if live_arrays is None:
        return None
    try:
        return len(live_arrays())
    except Exception:
        return None


def segment_cache_sizes() -> tuple[int | None, int | None]:
    cache_size = reverse_transport._jax_trace_cache_size
    return (
        cache_size(
            reverse_transport._radau_database_segment_reduced_cotangent_bwd_with_table_support_call
        ),
        cache_size(
            reverse_transport._radau_segment_replay_minimal_with_primal_records_call
        ),
    )


def evaluate(problem, x, *, show_progress: bool):
    context = nullcontext() if show_progress else redirect_stdout(io.StringIO())
    with context:
        evaluation = problem.evaluate(x)
    return jax.block_until_ready((evaluation.residuals, evaluation.jacobian))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--diagnose-segment-dispatch",
        action="store_true",
        help="Print segment-local JAX trace-cache sizes and existing progress output.",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("--warmup must be non-negative and --repeats must be positive.")

    problem = parity.build_problem(
        reverse_segment_length=parity.TRIAL_SEGMENT_LENGTH
    )
    if args.diagnose_segment_dispatch:
        problem.options["reverse_segment_jit_diagnostics"] = True
    x = np.asarray(jax.device_get(problem.x0), dtype=float)
    print(
        "[database full-transport memory] "
        f"grid=({parity.DATABASE_N_THETA},{parity.DATABASE_N_PHI},{parity.DATABASE_N_XI}) "
        f"accepted_steps={parity.ACCEPTED_STEP_LIMIT} "
        f"segment_length={parity.TRIAL_SEGMENT_LENGTH} "
        f"segments={parity.ACCEPTED_STEP_LIMIT // parity.TRIAL_SEGMENT_LENGTH} "
        "initial_er_root=jax_selected_root "
        f"warmup={args.warmup} repeats={args.repeats} "
        f"parameter_count={problem.parameter_count}",
        flush=True,
    )

    for warmup_index in range(args.warmup):
        started = time.perf_counter()
        cache_before = segment_cache_sizes()
        residuals, jacobian = evaluate(
            problem, x, show_progress=args.diagnose_segment_dispatch
        )
        cache_after = segment_cache_sizes()
        del residuals, jacobian
        gc.collect()
        print(
            "[database full-transport memory] "
            f"warmup={warmup_index} elapsed_s={time.perf_counter() - started:.3f} "
            f"segment_cache={cache_before}->{cache_after}",
            flush=True,
        )

    baseline_rss: int | None = None
    baseline_residuals = None
    baseline_jacobian = None
    for trial_index in range(args.repeats):
        started = time.perf_counter()
        cache_before = segment_cache_sizes()
        residuals, jacobian = evaluate(
            problem, x, show_progress=args.diagnose_segment_dispatch
        )
        cache_after = segment_cache_sizes()
        residuals_np = np.asarray(jax.device_get(residuals), dtype=float)
        jacobian_np = np.asarray(jax.device_get(jacobian), dtype=float)
        if baseline_residuals is None:
            baseline_residuals = residuals_np.copy()
            baseline_jacobian = jacobian_np.copy()
        residual_repeat_delta = float(
            np.max(np.abs(residuals_np - baseline_residuals))
        )
        jacobian_repeat_delta = float(
            np.max(np.abs(jacobian_np - baseline_jacobian))
        )
        del residuals, jacobian
        gc.collect()
        rss = opt._process_resident_memory_bytes()
        if baseline_rss is None:
            baseline_rss = rss
        rss_delta = (
            None
            if rss is None or baseline_rss is None
            else (rss - baseline_rss) / 2**20
        )
        rss_text = "unavailable" if rss_delta is None else f"{rss_delta:+.1f} MiB"
        arrays = live_jax_array_count()
        arrays_text = "unavailable" if arrays is None else str(arrays)
        print(
            "[database full-transport memory] "
            f"trial={trial_index} elapsed_s={time.perf_counter() - started:.3f} "
            f"rss_delta={rss_text} live_jax_arrays={arrays_text} "
            f"segment_cache={cache_before}->{cache_after} "
            f"residual_repeat_max_abs={residual_repeat_delta:.3e} "
            f"jacobian_repeat_max_abs={jacobian_repeat_delta:.3e}",
            flush=True,
        )

    print(
        "[database full-transport memory] complete; t_final was not required and SciPy was not run.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
