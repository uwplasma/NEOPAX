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
import ctypes
import os
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
from NEOPAX._geometry_autodiff import geometry_raw_block_optimization_stage  # noqa: E402
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
        self.checkpoint_rss_bytes: dict[str, int | None] = {}

    def reset(self) -> None:
        self.raw_block_solve_calls = 0
        self.selected_root_calls = 0
        self.checkpoint_rss_bytes = {}

    def context(
        self,
        *,
        checkpoints: bool = False,
        optimization_raw_block_stage=None,
    ):
        raw_block_solve = reverse_optimization.geometry_raw_block_solve_from_param_vector
        selected_root = reverse_optimization.initial_er_selected_root_profile
        payload_builder = reverse_optimization.build_neopax_geometry_and_ntx_exact_lij_support_from_state
        payload_pullback = reverse_optimization.realtime_geometry_transport_reverse_table_from_payload_cotangents

        def record(name: str) -> None:
            if checkpoints:
                self.checkpoint_rss_bytes[name] = opt._process_resident_memory_bytes()

        def count_raw_block_solve(*args, **kwargs):
            self.raw_block_solve_calls += 1
            if optimization_raw_block_stage is not None:
                kwargs["stage"] = optimization_raw_block_stage.raw_block_stage
                kwargs["solve_with_aux_runner"] = optimization_raw_block_stage.solve_with_aux_runner
            result = raw_block_solve(*args, **kwargs)
            record("raw_block_solve")
            return result

        def count_selected_root(*args, **kwargs):
            self.selected_root_calls += 1
            result = selected_root(*args, **kwargs)
            record("selected_root")
            return result

        def count_payload_builder(*args, **kwargs):
            result = payload_builder(*args, **kwargs)
            record("geometry_ntx_payload")
            return result

        def count_payload_pullback(*args, **kwargs):
            result = payload_pullback(*args, **kwargs)
            record("payload_to_vmec")
            return result

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
            patch.object(
                reverse_optimization,
                "build_neopax_geometry_and_ntx_exact_lij_support_from_state",
                count_payload_builder,
            ),
            patch.object(
                reverse_optimization,
                "realtime_geometry_transport_reverse_table_from_payload_cotangents",
                count_payload_pullback,
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


def _trim_native_heap() -> bool:
    """Ask glibc to return currently free heap pages for this diagnostic only."""

    if os.name == "nt":
        return False
    try:
        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except OSError:
        return False


def _one_cache_cleared_sample(problem, x, iteration: int):
    """Evaluate the unchanged problem, then clear cache state before RSS sampling."""

    started = time.perf_counter()
    evaluation = problem.evaluate(x)
    residuals, jacobian = jax.block_until_ready(
        (evaluation.residuals, evaluation.jacobian)
    )
    residual_norm = float(np.linalg.norm(np.asarray(jax.device_get(residuals))))
    jacobian_shape = tuple(int(size) for size in jacobian.shape)
    elapsed_s = time.perf_counter() - started
    del evaluation, residuals, jacobian
    gc.collect()
    jax.clear_caches()
    gc.collect()
    heap_trimmed = _trim_native_heap()
    return (
        opt.RepeatedEvaluationMemorySample(
            iteration=iteration,
            elapsed_s=float(elapsed_s),
            resident_memory_bytes=opt._process_resident_memory_bytes(),
            residual_norm=residual_norm,
            jacobian_shape=jacobian_shape,
        ),
        heap_trimmed,
    )


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
    parser.add_argument(
        "--clear-jax-caches",
        action="store_true",
        help=(
            "Diagnostic only: clear JAX compilation caches and trim free native heap "
            "after every measured evaluation."
        ),
    )
    parser.add_argument(
        "--diagnose-checkpoints",
        action="store_true",
        help=(
            "Report the RSS delta immediately after the existing raw VMEC solve, "
            "geometry/NTX payload build, selected root, and payload-to-VMEC pullback."
        ),
    )
    parser.add_argument(
        "--stable-vmex-callback",
        action="store_true",
        help=(
            "Use the optimization-only, stage-bound VMEX raw-solve callback. "
            "The benchmark/default raw-solve call remains unchanged."
        ),
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
    optimization_raw_block_stage = None
    if args.stable_vmex_callback:
        if not problem.parameter_set.vmec_boundary_specs:
            raise ValueError("--stable-vmex-callback requires VMEC boundary parameters.")
        optimization_raw_block_stage = geometry_raw_block_optimization_stage(
            problem.context,
            tuple(spec.as_tuple() for spec in problem.parameter_set.vmec_boundary_specs),
            max_iter=problem.geometry_max_iter,
        )
    first_bytes: int | None = None
    first_checkpoint_bytes: dict[str, int | None] = {}
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
        if args.diagnose_checkpoints:
            checkpoint_text = []
            for name in (
                "raw_block_solve",
                "geometry_ntx_payload",
                "selected_root",
                "payload_to_vmec",
            ):
                checkpoint_bytes = structure_counter.checkpoint_rss_bytes.get(name)
                if name not in first_checkpoint_bytes:
                    first_checkpoint_bytes[name] = checkpoint_bytes
                base_bytes = first_checkpoint_bytes[name]
                if checkpoint_bytes is None or base_bytes is None:
                    value = "unavailable"
                else:
                    value = f"{(checkpoint_bytes - base_bytes) / 2**20:+.1f}"
                checkpoint_text.append(f"{name}={value}MiB")
            print(
                "[memory test] checkpoint_rss_delta " + " ".join(checkpoint_text),
                flush=True,
            )

    print(
        f"[memory test] mode={args.mode} objectives=maxEr,Er_left,Er_right,J_bootstrap "
        f"warmup={args.warmup} repeats={args.repeats} parameter_count={problem.parameter_count}",
        flush=True,
    )
    if args.clear_jax_caches:
        print(
            "[memory test] diagnostic_cleanup=jax.clear_caches+malloc_trim after each trial",
            flush=True,
        )
    if args.stable_vmex_callback:
        print("[memory test] raw_solve_callback=stable_optimization_only", flush=True)
    for warmup_index in range(args.warmup):
        print(f"[memory test] warmup={warmup_index} starting", flush=True)
        started = time.perf_counter()
        structure_counter.reset()
        patchers = (
            structure_counter.context(
                checkpoints=args.diagnose_checkpoints,
                optimization_raw_block_stage=optimization_raw_block_stage,
            )
            if args.diagnose_structure or args.diagnose_checkpoints or args.stable_vmex_callback
            else ()
        )
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
    if args.diagnose_structure or args.diagnose_checkpoints or args.stable_vmex_callback:
        for iteration in range(args.repeats):
            structure_counter.reset()
            patchers = structure_counter.context(
                checkpoints=args.diagnose_checkpoints,
                optimization_raw_block_stage=optimization_raw_block_stage,
            )
            for patcher in patchers:
                patcher.start()
            try:
                if args.clear_jax_caches:
                    sample, heap_trimmed = _one_cache_cleared_sample(
                        quiet_problem, x, iteration
                    )
                else:
                    samples = opt.repeated_evaluation_memory_samples(
                        quiet_problem,
                        warmup=0,
                        repeats=1,
                        scaled_parameter_values=x,
                    )
                    sample = replace(samples[0], iteration=iteration)
                    heap_trimmed = None
            finally:
                for patcher in reversed(patchers):
                    patcher.stop()
            report(sample)
            if heap_trimmed is not None:
                print(f"[memory test] trial={iteration} native_heap_trimmed={heap_trimmed}", flush=True)
    elif args.clear_jax_caches:
        for iteration in range(args.repeats):
            sample, heap_trimmed = _one_cache_cleared_sample(quiet_problem, x, iteration)
            report(sample)
            print(f"[memory test] trial={iteration} native_heap_trimmed={heap_trimmed}", flush=True)
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
