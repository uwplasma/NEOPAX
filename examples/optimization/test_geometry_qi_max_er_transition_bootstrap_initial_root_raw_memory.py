#!/usr/bin/env python
"""Isolate retained memory from the established shared raw VMEX solve.

This is deliberately not an optimization mode.  It builds the same fixed
24-parameter geometry stage as the mixed initial-root example, extracts the
same VMEC boundary vector from its initial point, and repeatedly invokes only
``geometry_raw_block_solve_from_param_vector``.  There are no QI objectives,
Er roots, NTX calculations, transport residuals, or reverse pullbacks.

Every trial recreates the scaled-to-physical and physical-to-VMEC-vector
plumbing from the host optimizer vector before invoking the solve.  The test
therefore answers one question: can the raw VMEX/JAX boundary, including its
per-evaluation parameter setup, retain the memory seen after each full
least-squares evaluation?
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import time

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX import _reverse_ad_optimization as reverse_optimization  # noqa: E402
from NEOPAX import optimization as opt  # noqa: E402
from NEOPAX._geometry_autodiff import geometry_raw_block_solve_from_param_vector  # noqa: E402
from optimize_geometry_qi_max_er_transition_bootstrap_initial_root import (  # noqa: E402
    MAX_MODE_SCHEDULE,
    SEED_INPUT,
    build_transition_bootstrap_initial_root_problem,
)


def _dispatch_cache_size() -> int | None:
    """Return JAX's primitive-dispatch cache size when this JAX exposes it."""

    try:
        from jax._src import dispatch

        return int(dispatch.xla_primitive_callable.cache_info().currsize)
    except (AttributeError, ImportError):
        return None


def _live_jax_array_count() -> int | None:
    probe = getattr(jax, "live_arrays", None)
    if probe is None:
        return None
    try:
        return len(probe())
    except Exception:
        return None


def _run_raw_solve(problem, host_scaled_values, vmec_specs) -> None:
    """Recreate the benchmark VMEC input setup, solve, then release its result."""

    # This intentionally mirrors ``GeometryInitialErRootLeastSquaresProblem.evaluate``:
    # the optimizer supplies a host array, from which every evaluation creates its
    # physical mixed vector and the compact VMEC boundary vector.
    physical_values = problem._scaled_to_physical(host_scaled_values)
    vmec_values = reverse_optimization.vmec_parameter_values_from_parameter_vector(
        problem.parameter_set,
        physical_values,
    )

    raw = geometry_raw_block_solve_from_param_vector(
        problem.context,
        vmec_values,
        vmec_specs,
        max_iter=problem.geometry_max_iter,
        solver_device=problem.geometry_solver_device,
        stage=problem.raw_block_stage,
    )
    jax.block_until_ready((raw.state, raw.dof_mask))
    del raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--diagnose-jax-dispatch-cache",
        action="store_true",
        help="Also print JAX primitive-dispatch cache size after each raw solve.",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("--warmup must be non-negative and --repeats must be positive.")
    if not np.isscalar(MAX_MODE_SCHEDULE):
        raise ValueError("The raw memory test requires one fixed MAX_MODE_SCHEDULE value.")

    problem = build_transition_bootstrap_initial_root_problem(
        SEED_INPUT,
        int(MAX_MODE_SCHEDULE),
    )
    if problem.raw_block_stage is None:
        raise RuntimeError("The matching optimization problem did not create a raw VMEX stage.")
    host_scaled_values = np.asarray(jax.device_get(problem.x0), dtype=float)
    vmec_specs = tuple(spec.as_tuple() for spec in problem.parameter_set.vmec_boundary_specs)
    if len(vmec_specs) != problem.parameter_count:
        raise RuntimeError(
            "This isolation test requires geometry-only parameters; "
            f"got vmec_count={len(vmec_specs)}, parameter_count={problem.parameter_count}."
        )

    print(
        "[raw memory test] "
        f"warmup={args.warmup} repeats={args.repeats} parameter_count={problem.parameter_count} "
        "path=shared_raw_vmex_solve_only",
        flush=True,
    )
    for index in range(args.warmup):
        print(f"[raw memory test] warmup={index} starting", flush=True)
        started = time.perf_counter()
        _run_raw_solve(problem, host_scaled_values, vmec_specs)
        gc.collect()
        print(
            f"[raw memory test] warmup={index} complete elapsed_s={time.perf_counter() - started:.3f}",
            flush=True,
        )

    first_rss: int | None = None
    for index in range(args.repeats):
        started = time.perf_counter()
        _run_raw_solve(problem, host_scaled_values, vmec_specs)
        gc.collect()
        rss = opt._process_resident_memory_bytes()
        if first_rss is None:
            first_rss = rss
        delta_text = (
            "unavailable"
            if rss is None or first_rss is None
            else f"{(rss - first_rss) / 2**20:+.1f} MiB"
        )
        live_arrays = _live_jax_array_count()
        cache = _dispatch_cache_size() if args.diagnose_jax_dispatch_cache else None
        suffix = ""
        if args.diagnose_jax_dispatch_cache:
            suffix = f" jax_dispatch_cache_size={cache if cache is not None else 'unavailable'}"
        print(
            f"[raw memory test] trial={index} elapsed_s={time.perf_counter() - started:.3f} "
            f"rss_delta={delta_text} "
            f"live_jax_arrays={live_arrays if live_arrays is not None else 'unavailable'}"
            f"{suffix}",
            flush=True,
        )
    print("[raw memory test] complete; no roots, transport, objectives, or reverse AD were run.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
