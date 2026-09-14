#!/usr/bin/env python
"""Repeated-memory test for the 16-step database full-transport reverse lane.

The selected initial-Er root is followed by 16 accepted Radau steps on the
small ``(5,25,31)`` database, divided into four fixed four-step reverse
segments. This diagnostic intentionally does not require transport ``t_final``.
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
from NEOPAX import _transport_solvers as transport_solvers  # noqa: E402
import test_geometry_qi_max_er_transition_bootstrap_initial_root_database_full_transport_parity as parity  # noqa: E402


def live_jax_array_count() -> int | None:
    live_arrays = getattr(jax, "live_arrays", None)
    if live_arrays is None:
        return None
    try:
        return len(live_arrays())
    except Exception:
        return None


def global_dispatch_cache_size() -> int | None:
    try:
        from jax._src import dispatch

        return int(dispatch.xla_primitive_callable.cache_info().currsize)
    except Exception:
        return None


def device_memory_text() -> str:
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        return "unavailable"
    if not stats:
        return "unavailable"
    parts = []
    for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
        value = stats.get(key)
        if value is not None:
            parts.append(f"{key}={int(value) / 2**20:.1f}MiB")
    return ",".join(parts) if parts else "unavailable"


def segment_cache_sizes(
    problem,
) -> tuple[int | None, ...]:
    cache_size = reverse_transport._jax_trace_cache_size
    optimization_replay_cache_size = getattr(
        problem.table_result_builder,
        "optimization_segment_replay_cache_size",
        lambda: None,
    )
    optimization_bwd_cache_size = getattr(
        problem.table_result_builder,
        "optimization_segment_bwd_cache_size",
        lambda: None,
    )
    optimization_support_cache_sizes = getattr(
        problem.table_result_builder,
        "optimization_support_cache_sizes",
        lambda: (None,) * 5,
    )
    return (
        cache_size(
            reverse_transport._radau_database_segment_reduced_cotangent_bwd_with_table_support_call
        ),
        cache_size(
            reverse_transport._radau_segment_replay_minimal_with_primal_records_call
        ),
        cache_size(
            transport_solvers._radau_segment_reduced_cotangent_bwd_batched_call
        ),
        optimization_replay_cache_size(),
        optimization_bwd_cache_size(),
        *optimization_support_cache_sizes(),
        global_dispatch_cache_size(),
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

    problem = parity.build_problem(reverse_stage_mode=parity.TRIAL_STAGE_MODE)
    phase_context = {
        "evaluation": "setup",
        "previous_rss": None,
        "previous_dispatch": None,
    }
    trial0_phase_rss: dict[str, int] = {}
    trial0_phase_dispatch: dict[str, int] = {}
    if args.diagnose_segment_dispatch:
        problem.options["reverse_segment_jit_diagnostics"] = True

        def _phase_probe(phase: str) -> None:
            rss = opt._process_resident_memory_bytes()
            rss_text = "unavailable" if rss is None else f"{rss / 2**20:.1f}MiB"
            arrays = live_jax_array_count()
            arrays_text = "unavailable" if arrays is None else str(arrays)
            dispatch_cache = global_dispatch_cache_size()
            dispatch_text = (
                "unavailable" if dispatch_cache is None else str(dispatch_cache)
            )
            previous_rss = phase_context["previous_rss"]
            rss_step_delta = (
                None
                if rss is None or previous_rss is None
                else (rss - previous_rss) / 2**20
            )
            trial0_rss = trial0_phase_rss.get(phase)
            rss_trial0_delta = (
                None
                if rss is None or trial0_rss is None
                else (rss - trial0_rss) / 2**20
            )
            previous_dispatch = phase_context["previous_dispatch"]
            dispatch_step_delta = (
                None
                if dispatch_cache is None or previous_dispatch is None
                else dispatch_cache - previous_dispatch
            )
            trial0_dispatch = trial0_phase_dispatch.get(phase)
            dispatch_trial0_delta = (
                None
                if dispatch_cache is None or trial0_dispatch is None
                else dispatch_cache - trial0_dispatch
            )
            if phase_context["evaluation"] == "trial:0":
                if rss is not None:
                    trial0_phase_rss[phase] = rss
                if dispatch_cache is not None:
                    trial0_phase_dispatch[phase] = dispatch_cache
            phase_context["previous_rss"] = rss
            phase_context["previous_dispatch"] = dispatch_cache

            def _delta_text(value, suffix=""):
                return "n/a" if value is None else f"{value:+.1f}{suffix}"

            print(
                "[database full-transport phase] "
                f"evaluation={phase_context['evaluation']} phase={phase} "
                f"rss={rss_text} live_jax_arrays={arrays_text} "
                f"rss_step_delta={_delta_text(rss_step_delta, 'MiB')} "
                f"rss_vs_trial0_phase={_delta_text(rss_trial0_delta, 'MiB')} "
                f"global_dispatch_cache={dispatch_text} "
                f"dispatch_step_delta={_delta_text(dispatch_step_delta)} "
                f"dispatch_vs_trial0_phase={_delta_text(dispatch_trial0_delta)} "
                f"device_memory={device_memory_text()}",
                flush=True,
            )

        problem.table_result_builder.optimization_phase_probe = _phase_probe
    x = np.asarray(jax.device_get(problem.x0), dtype=float)
    print(
        "[database full-transport memory] "
        f"grid=({parity.DATABASE_N_THETA},{parity.DATABASE_N_PHI},{parity.DATABASE_N_XI}) "
        f"accepted_steps={parity.ACCEPTED_STEP_LIMIT} "
        f"segment_length={parity.REVERSE_SEGMENT_LENGTH} "
        f"segments={parity.ACCEPTED_STEP_LIMIT // parity.REVERSE_SEGMENT_LENGTH} "
        "initial_er_root=jax_selected_root "
        f"stage={parity.TRIAL_STAGE_MODE} "
        "pullback_profile=current_optimized "
        "database_bwd_boundary=optimization_persistent_lean_context "
        f"warmup={args.warmup} repeats={args.repeats} "
        f"parameter_count={problem.parameter_count}",
        flush=True,
    )

    for warmup_index in range(args.warmup):
        phase_context["evaluation"] = f"warmup:{warmup_index}"
        phase_context["previous_rss"] = None
        phase_context["previous_dispatch"] = None
        started = time.perf_counter()
        cache_before = segment_cache_sizes(problem)
        residuals, jacobian = evaluate(
            problem, x, show_progress=args.diagnose_segment_dispatch
        )
        cache_after = segment_cache_sizes(problem)
        del residuals, jacobian
        gc.collect()
        print(
            "[database full-transport memory] "
            f"warmup={warmup_index} elapsed_s={time.perf_counter() - started:.3f} "
            "stage_cache=(benchmark_database_bwd,benchmark_replay,generic_bwd,"
            "optimization_replay,optimization_lean_bwd,profile_primal,"
            "profile_pullback,root_pullback,final_objectives,bootstrap_unpack,"
            "global_dispatch)="
            f"{cache_before}->{cache_after}",
            flush=True,
        )

    baseline_rss: int | None = None
    baseline_residuals = None
    baseline_jacobian = None
    for trial_index in range(args.repeats):
        phase_context["evaluation"] = f"trial:{trial_index}"
        phase_context["previous_rss"] = None
        phase_context["previous_dispatch"] = None
        started = time.perf_counter()
        cache_before = segment_cache_sizes(problem)
        residuals, jacobian = evaluate(
            problem, x, show_progress=args.diagnose_segment_dispatch
        )
        cache_after = segment_cache_sizes(problem)
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
            "stage_cache=(benchmark_database_bwd,benchmark_replay,generic_bwd,"
            "optimization_replay,optimization_lean_bwd,profile_primal,"
            "profile_pullback,root_pullback,final_objectives,bootstrap_unpack,"
            "global_dispatch)="
            f"{cache_before}->{cache_after} "
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
