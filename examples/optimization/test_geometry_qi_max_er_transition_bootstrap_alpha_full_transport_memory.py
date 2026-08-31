#!/usr/bin/env python
"""Repeated-evaluation memory audit for the full-transport geometry example.

This builds the same problem as
``optimize_geometry_qi_max_er_transition_bootstrap_alpha_full_transport.py``
and repeatedly evaluates its unchanged residual/Jacobian at ``x0``.  It does
not invoke SciPy, write geometry inputs, or alter the benchmark/full-transport
call graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX import optimization as opt  # noqa: E402
import optimize_geometry_qi_max_er_transition_bootstrap_alpha_full_transport as example  # noqa: E402


def _build_problem(*, accepted_step_limit: int | None):
    active_terms = tuple(term for term in example.terms if float(term[2]) != 0.0)
    max_mode_schedule = example.max_mode_schedule_values()
    if len(max_mode_schedule) != 1:
        raise ValueError(
            "The memory audit needs one fixed geometry stage; set "
            "MAX_MODE_SCHEDULE to one max_mode in the example."
        )
    return opt.geometry_full_transport_least_squares_problem(
        example.TRANSPORT_CONFIG,
        active_terms,
        vmec_input=example.SEED_INPUT,
        max_mode=max_mode_schedule[0],
        families=example.GEOMETRY_FAMILIES,
        scale_mode=example.SCALE_MODE,
        ess_alpha=example.ESS_ALPHA,
        mboz=example.QI_MBOZ,
        nboz=example.QI_NBOZ,
        surfaces=tuple(float(s) for s in example.SURFACES),
        geometry_max_iter=example.GEOMETRY_MAX_ITER,
        geometry_solver_device=example.SOLVER_DEVICE,
        device=example.SOLVER_DEVICE,
        accepted_step_limit=accepted_step_limit,
        reverse_segment_length=example.REVERSE_SEGMENT_LENGTH,
        max_reverse_accepted_steps=example.MAX_REVERSE_ACCEPTED_STEPS,
        initial_er_root_ad="jax_selected_root",
        radau_jacobian_reuse_mode="legacy",
        reverse_stage_adjoint_solve_mode="bicgstab",
        reverse_rhs_transpose_mode="explicit_ntx_interpolated",
        reverse_step_bwd_mode="reduced_cotangent",
    )


def _mib(value: int | None) -> str:
    return "unavailable" if value is None else f"{value / 2**20:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--accepted-step-limit",
        type=int,
        default=None,
        help="Override the example's full t_final solve only for a shorter diagnostic.",
    )
    args = parser.parse_args()

    accepted_step_limit = (
        example.FULL_TRANSPORT_ACCEPTED_STEP_LIMIT
        if args.accepted_step_limit is None
        else int(args.accepted_step_limit)
    )
    problem = _build_problem(accepted_step_limit=accepted_step_limit)
    x = np.asarray(jax.device_get(problem.x0), dtype=float)
    active_labels = [
        getattr(term[0], "name", getattr(term[0], "label", str(term[0])))
        for term in example.terms
        if float(term[2]) != 0.0
    ]
    print(
        "[full transport memory test] "
        f"warmup={args.warmup} repeats={args.repeats} "
        f"parameter_count={problem.parameter_count} "
        f"accepted_step_limit={accepted_step_limit} "
        f"objectives={','.join(active_labels)}",
        flush=True,
    )
    if args.warmup:
        print("[full transport memory test] warmup starting", flush=True)

    baseline_rss: int | None = None

    def report(sample: opt.RepeatedEvaluationMemorySample) -> None:
        nonlocal baseline_rss
        if baseline_rss is None:
            baseline_rss = sample.resident_memory_bytes
        if sample.resident_memory_bytes is None or baseline_rss is None:
            delta = "unavailable"
        else:
            delta = f"{(sample.resident_memory_bytes - baseline_rss) / 2**20:+.1f}"
        print(
            "[full transport memory test] "
            f"trial={sample.iteration} elapsed_s={sample.elapsed_s:.3f} "
            f"rss_delta={delta}MiB rss={_mib(sample.resident_memory_bytes)}MiB "
            f"residual_norm={sample.residual_norm:.6e} "
            f"jacobian_shape={sample.jacobian_shape}",
            flush=True,
        )

    samples = opt.repeated_evaluation_memory_samples(
        problem,
        warmup=args.warmup,
        repeats=args.repeats,
        scaled_parameter_values=x,
        on_sample=report,
    )
    if not samples:
        raise RuntimeError("The full-transport memory audit produced no samples.")
    print("[full transport memory test] complete; SciPy was not run.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
