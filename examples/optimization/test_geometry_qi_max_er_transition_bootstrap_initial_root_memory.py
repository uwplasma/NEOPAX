#!/usr/bin/env python
"""Memory test for geometry plus selected initial-Er root objective rows.

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
import importlib
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


def _terms_for_objective_set(objective_set: str):
    """Select exactly one geometry/initial-root reverse branch for attribution."""

    selected = []
    for term in example.terms:
        objective = getattr(term[0], "objective", term[0])
        is_transport = objective.family == "transport"
        if objective_set == "geometry_only":
            if not is_transport:
                selected.append(term)
            continue
        if objective_set == "transport_er_only":
            if is_transport and objective.name != "bootstrap_current_softmax_abs_scaled":
                selected.append(term)
            continue
        if not is_transport:
            selected.append(term)
            continue
        is_bootstrap = objective.name == "bootstrap_current_softmax_abs_scaled"
        if objective_set == "all" or (objective_set == "er_only" and not is_bootstrap) or (
            objective_set == "bootstrap_only" and is_bootstrap
        ):
            selected.append(term)
    return selected


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
        self.vmex_jit_cache_sizes: dict[str, int | None] = {}
        self.jax_dispatch_cache_size: int | None = None
        self.raw_solve_dispatch_before: int | None = None
        self.raw_solve_dispatch_after: int | None = None
        self.raw_wrapper_dispatch_caches: dict[str, int | None] = {}
        self.pre_raw_dispatch_caches: dict[str, int | None] = {}
        self.transport_dispatch_caches: dict[str, int | None] = {}
        self.payload_dispatch_caches: dict[str, int | None] = {}
        self.support_reverse_dispatch_caches: dict[str, int | None] = {}
        self.support_reverse_rss_bytes: dict[str, int | None] = {}
        self.optimization_stage_dispatch_caches: dict[str, int | None] = {}
        self.geometry_dispatch_caches: dict[str, int | None] = {}

    def reset(self) -> None:
        self.raw_block_solve_calls = 0
        self.selected_root_calls = 0
        self.checkpoint_rss_bytes = {}
        self.vmex_jit_cache_sizes = {}
        self.jax_dispatch_cache_size = None
        self.raw_solve_dispatch_before = None
        self.raw_solve_dispatch_after = None
        self.raw_wrapper_dispatch_caches = {}
        self.pre_raw_dispatch_caches = {}
        self.transport_dispatch_caches = {}
        self.payload_dispatch_caches = {}
        self.support_reverse_dispatch_caches = {}
        self.support_reverse_rss_bytes = {}
        self.optimization_stage_dispatch_caches = {}
        self.geometry_dispatch_caches = {}

    def context(
        self,
        *,
        checkpoints: bool = False,
        optimization_raw_block_stage=None,
        reuse_base_implicit_params: bool = False,
        jit_boundary_parameter_updates: bool = False,
        vmex_cache_sizes: bool = False,
        jax_dispatch_cache: bool = False,
        raw_solve_dispatch: bool = False,
        raw_wrapper_dispatch: bool = False,
        pre_raw_dispatch: bool = False,
        transport_dispatch: bool = False,
        payload_dispatch: bool = False,
        support_reverse_dispatch: bool = False,
        support_reverse_rss: bool = False,
        optimization_stage_dispatch: bool = False,
        geometry_dispatch: bool = False,
        raw_implicit=None,
    ):
        raw_block_solve = reverse_optimization.geometry_raw_block_solve_from_param_vector
        selected_root = reverse_optimization.initial_er_selected_root_profile
        payload_builder = reverse_optimization.build_neopax_geometry_and_ntx_exact_lij_support_from_state
        payload_pullback = reverse_optimization.realtime_geometry_transport_reverse_table_from_payload_cotangents
        support_reverse = reverse_optimization.geometry_active_initial_er_root_only_reverse_table
        optimization_evaluator = opt.evaluate_geometry_initial_er_root_only_least_squares_optimization
        geometry_table = reverse_optimization.geometry_full_ad_reverse_table

        def record(name: str) -> None:
            if checkpoints:
                self.checkpoint_rss_bytes[name] = opt._process_resident_memory_bytes()

        def record_vmex_cache_sizes(implicit) -> None:
            if not vmex_cache_sizes:
                return
            module_name = f"{implicit.__name__.rsplit('.', 1)[0]}.solver"
            try:
                solver = importlib.import_module(module_name)
            except (ImportError, AttributeError):
                return
            for name in ("_block_lane", "_while_lane"):
                cache_size = getattr(getattr(solver, name, None), "_cache_size", None)
                if callable(cache_size):
                    try:
                        self.vmex_jit_cache_sizes[name] = int(cache_size())
                    except Exception:
                        self.vmex_jit_cache_sizes[name] = None

        def record_jax_dispatch_cache() -> None:
            if not jax_dispatch_cache:
                return
            try:
                from jax._src import dispatch

                info = dispatch.xla_primitive_callable.cache_info()
                self.jax_dispatch_cache_size = int(info.currsize)
            except (AttributeError, ImportError):
                self.jax_dispatch_cache_size = None

        def dispatch_cache_size() -> int | None:
            try:
                from jax._src import dispatch

                return int(dispatch.xla_primitive_callable.cache_info().currsize)
            except (AttributeError, ImportError):
                return None

        def record_raw_wrapper_dispatch(label: str) -> None:
            if raw_wrapper_dispatch:
                self.raw_wrapper_dispatch_caches[str(label)] = dispatch_cache_size()

        def record_pre_raw_dispatch(label: str) -> None:
            if pre_raw_dispatch:
                self.pre_raw_dispatch_caches[str(label)] = dispatch_cache_size()

        def record_transport_dispatch(label: str) -> None:
            if transport_dispatch:
                self.transport_dispatch_caches[str(label)] = dispatch_cache_size()

        def record_payload_dispatch(label: str) -> None:
            if payload_dispatch:
                self.payload_dispatch_caches[str(label)] = dispatch_cache_size()

        def record_support_reverse_dispatch(label: str) -> None:
            if support_reverse_dispatch:
                self.support_reverse_dispatch_caches[str(label)] = dispatch_cache_size()
            if support_reverse_rss:
                self.support_reverse_rss_bytes[str(label)] = opt._process_resident_memory_bytes()

        def record_optimization_stage_dispatch(label: str) -> None:
            if optimization_stage_dispatch:
                self.optimization_stage_dispatch_caches[str(label)] = dispatch_cache_size()

        def record_geometry_dispatch(label: str) -> None:
            if geometry_dispatch:
                self.geometry_dispatch_caches[str(label)] = dispatch_cache_size()

        original_scaled_to_physical = opt.GeometryInitialErRootLeastSquaresProblem._scaled_to_physical
        original_active_profile_values = reverse_optimization._active_profile_values_from_parameter_vector
        original_vmec_values = reverse_optimization.vmec_parameter_values_from_parameter_vector

        def count_scaled_to_physical(*args, **kwargs):
            record_pre_raw_dispatch("before_scaled_to_physical")
            result = original_scaled_to_physical(*args, **kwargs)
            record_pre_raw_dispatch("after_scaled_to_physical")
            return result

        def count_active_profile_values(*args, **kwargs):
            record_pre_raw_dispatch("before_active_profiles")
            result = original_active_profile_values(*args, **kwargs)
            record_pre_raw_dispatch("after_active_profiles")
            return result

        def count_vmec_values(*args, **kwargs):
            record_pre_raw_dispatch("before_vmec_extract")
            result = original_vmec_values(*args, **kwargs)
            record_pre_raw_dispatch("after_vmec_extract")
            return result

        def count_raw_block_solve(*args, **kwargs):
            self.raw_block_solve_calls += 1
            if raw_wrapper_dispatch:
                kwargs["dispatch_cache_probe"] = record_raw_wrapper_dispatch
            if optimization_raw_block_stage is not None:
                kwargs["stage"] = optimization_raw_block_stage.raw_block_stage
                if jit_boundary_parameter_updates:
                    kwargs["implicit_params_from_deltas_runner"] = (
                        optimization_raw_block_stage.implicit_params_from_deltas_runner
                    )
                    kwargs["state_mask_stop_gradient_runner"] = (
                        optimization_raw_block_stage.state_mask_stop_gradient_runner
                    )
                elif reuse_base_implicit_params:
                    kwargs["base_implicit_params"] = optimization_raw_block_stage.base_implicit_params
                else:
                    kwargs["solve_with_aux_runner"] = optimization_raw_block_stage.solve_with_aux_runner
            result = raw_block_solve(*args, **kwargs)
            if checkpoints:
                jax.block_until_ready((result.state, result.dof_mask))
            record_vmex_cache_sizes(result.implicit)
            record_jax_dispatch_cache()
            record_transport_dispatch("after_raw_block_solve")
            record("raw_block_solve")
            return result

        def count_selected_root(*args, **kwargs):
            self.selected_root_calls += 1
            result = selected_root(*args, **kwargs)
            if checkpoints:
                jax.block_until_ready(result)
            record_transport_dispatch("after_selected_root")
            record("selected_root")
            return result

        def count_payload_builder(*args, **kwargs):
            result = payload_builder(*args, **kwargs)
            if checkpoints:
                jax.block_until_ready(result)
            record_transport_dispatch("after_geometry_ntx_payload")
            record("geometry_ntx_payload")
            return result

        def count_payload_pullback(*args, **kwargs):
            if payload_dispatch:
                kwargs["dispatch_cache_probe"] = record_payload_dispatch
            result = payload_pullback(*args, **kwargs)
            if checkpoints:
                jax.block_until_ready(
                    (
                        result.table_result.objective_values,
                        result.table_result.geometry_gradient_matrix,
                    )
                )
            record_transport_dispatch("after_payload_to_vmec")
            record("payload_to_vmec")
            return result

        def count_support_reverse(*args, **kwargs):
            if support_reverse_dispatch or support_reverse_rss:
                kwargs["dispatch_cache_probe"] = record_support_reverse_dispatch
            return support_reverse(*args, **kwargs)

        def count_optimization_evaluator(*args, **kwargs):
            if optimization_stage_dispatch:
                kwargs["dispatch_cache_probe"] = record_optimization_stage_dispatch
            result = optimization_evaluator(*args, **kwargs)
            record_optimization_stage_dispatch("optimization_evaluator_return")
            return result

        def count_geometry_table(*args, **kwargs):
            record_geometry_dispatch("geometry_table_entry")
            if geometry_dispatch:
                kwargs["dispatch_cache_probe"] = record_geometry_dispatch
            result = geometry_table(*args, **kwargs)
            record_geometry_dispatch("geometry_table_return")
            return result

        patchers = [
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
            patch.object(
                reverse_optimization,
                "geometry_active_initial_er_root_only_reverse_table",
                count_support_reverse,
            ),
            patch.object(
                opt,
                "evaluate_geometry_initial_er_root_only_least_squares_optimization",
                count_optimization_evaluator,
            ),
            patch.object(
                reverse_optimization,
                "geometry_full_ad_reverse_table",
                count_geometry_table,
            ),
        ]
        if pre_raw_dispatch:
            patchers.extend(
                [
                    patch.object(
                        opt.GeometryInitialErRootLeastSquaresProblem,
                        "_scaled_to_physical",
                        count_scaled_to_physical,
                    ),
                    patch.object(
                        reverse_optimization,
                        "_active_profile_values_from_parameter_vector",
                        count_active_profile_values,
                    ),
                    patch.object(
                        reverse_optimization,
                        "vmec_parameter_values_from_parameter_vector",
                        count_vmec_values,
                    ),
                ]
            )
        if raw_solve_dispatch and raw_implicit is not None:
            original_solve = raw_implicit.solve_implicit_with_aux

            def count_implicit_solve(*args, **kwargs):
                self.raw_solve_dispatch_before = dispatch_cache_size()
                result = original_solve(*args, **kwargs)
                jax.block_until_ready(result)
                self.raw_solve_dispatch_after = dispatch_cache_size()
                return result

            patchers.append(
                patch.object(raw_implicit, "solve_implicit_with_aux", count_implicit_solve)
            )
        return tuple(patchers)


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


def _one_cleanup_sample(problem, x, iteration: int, *, clear_jax_caches: bool, trim_native_heap: bool):
    """Evaluate unchanged math, then apply explicitly selected diagnostic cleanup."""

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
    if clear_jax_caches:
        jax.clear_caches()
    gc.collect()
    heap_trimmed = _trim_native_heap() if trim_native_heap else None
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
    parser.add_argument(
        "--mode",
        choices=(
            "off",
            "optimization",
            "optimization_root_experiment",
            "optimization_root_strict_experiment",
            "optimization_root_per_radius_experiment",
            "optimization_payload_experiment",
            "optimization_payload_root_experiment",
            "optimization_payload_root_strict_experiment",
            "optimization_payload_root_scan_experiment",
            "optimization_payload_reverse_experiment",
            "vmex_like",
        ),
        default="off",
    )
    parser.add_argument(
        "--objective-set",
        choices=(
            "all",
            "er_only",
            "bootstrap_only",
            "geometry_only",
            "transport_er_only",
        ),
        default="all",
        help=(
            "Select all rows; geometry plus Er-only or bootstrap-only rows; or exactly "
            "geometry-only / Er-transport-only rows for memory attribution."
        ),
    )
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
            "Diagnostic only: clear JAX compilation caches after every measured evaluation."
        ),
    )
    parser.add_argument(
        "--trim-native-heap",
        action="store_true",
        help="Diagnostic only: call glibc malloc_trim after every measured evaluation.",
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
        "--persistent-raw-solve-jit",
        action="store_true",
        help=(
            "Use the optimization-only JIT compiled VMEX raw-solve boundary. "
            "Benchmark/default evaluations remain eager and unchanged."
        ),
    )
    parser.add_argument(
        "--persistent-raw-params",
        action="store_true",
        help=(
            "Use the optimization-only retained VMEX base-parameter pytree; "
            "only boundary deltas are rebuilt per evaluation."
        ),
    )
    parser.add_argument(
        "--persistent-raw-param-updates-jit",
        action="store_true",
        help=(
            "Use the optimization-only persistent JIT for the fixed VMEX "
            "boundary-parameter update map."
        ),
    )
    parser.add_argument(
        "--diagnose-vmex-caches",
        action="store_true",
        help="Report VMEX solver JIT trace-cache sizes after each raw-block solve.",
    )
    parser.add_argument(
        "--diagnose-jax-dispatch-cache",
        action="store_true",
        help="Report JAX's exposed primitive-dispatch cache size after each raw solve.",
    )
    parser.add_argument(
        "--diagnose-raw-solve-dispatch",
        action="store_true",
        help="Split the JAX dispatch-cache count immediately before/after VMEX solve_implicit_with_aux.",
    )
    parser.add_argument(
        "--diagnose-raw-wrapper-dispatch",
        action="store_true",
        help="Report JAX dispatch-cache sizes at exact raw-block wrapper boundaries.",
    )
    parser.add_argument(
        "--diagnose-pre-raw-dispatch",
        action="store_true",
        help="Report dispatch-cache sizes around the mixed evaluator's pre-VMEX vector operations.",
    )
    parser.add_argument(
        "--diagnose-transport-dispatch",
        action="store_true",
        help="Report dispatch-cache sizes after the root and payload/reverse transport boundaries.",
    )
    parser.add_argument(
        "--diagnose-payload-dispatch",
        action="store_true",
        help=(
            "Split the existing payload reverse pullback at the support VJP, "
            "geometry VJP, and raw-block-transpose boundaries."
        ),
    )
    parser.add_argument(
        "--diagnose-support-reverse-dispatch",
        action="store_true",
        help=(
            "Split the existing initial-root transport-support reverse before "
            "it constructs payload cotangents."
        ),
    )
    parser.add_argument(
        "--diagnose-support-reverse-rss",
        action="store_true",
        help=(
            "Report process RSS at the existing support-reverse probe labels; "
            "diagnostic only and does not alter reverse calculations."
        ),
    )
    parser.add_argument(
        "--diagnose-optimization-stage-dispatch",
        action="store_true",
        help=(
            "For optimization-only modes, split the dispatch-cache count across "
            "the root reverse, payload stage, geometry table, and final assembly."
        ),
    )
    parser.add_argument(
        "--diagnose-geometry-dispatch",
        action="store_true",
        help="Report dispatch-cache size immediately before and after the geometry reverse table.",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("--warmup must be non-negative and --repeats must be positive.")
    if not np.isscalar(MAX_MODE_SCHEDULE):
        raise ValueError("The memory test requires one fixed MAX_MODE_SCHEDULE value.")
    previous_mode = example.REVERSE_STAGE_MODE
    previous_terms = example.terms
    try:
        example.REVERSE_STAGE_MODE = args.mode
        example.terms = _terms_for_objective_set(args.objective_set)
        problem = build_transition_bootstrap_initial_root_problem(SEED_INPUT, int(MAX_MODE_SCHEDULE))
    finally:
        example.REVERSE_STAGE_MODE = previous_mode
        example.terms = previous_terms
    quiet_problem = QuietProblem(problem)
    x = np.asarray(jax.device_get(problem.x0), dtype=float)
    optimization_raw_block_stage = None
    if (
        args.persistent_raw_solve_jit
        or args.persistent_raw_params
        or args.persistent_raw_param_updates_jit
    ):
        if not problem.parameter_set.vmec_boundary_specs:
            raise ValueError("--persistent-raw-solve-jit requires VMEC boundary parameters.")
        optimization_raw_block_stage = geometry_raw_block_optimization_stage(
            problem.context,
            tuple(spec.as_tuple() for spec in problem.parameter_set.vmec_boundary_specs),
            max_iter=problem.geometry_max_iter,
            solver_device=problem.geometry_solver_device,
        )
    first_bytes: int | None = None
    first_checkpoint_bytes: dict[str, int | None] = {}
    first_support_reverse_rss_bytes: dict[str, int | None] = {}
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
        if args.diagnose_vmex_caches:
            entries = " ".join(
                f"{name}={value if value is not None else 'unavailable'}"
                for name, value in sorted(structure_counter.vmex_jit_cache_sizes.items())
            )
            print(f"[memory test] vmex_jit_cache_sizes {entries or 'unavailable'}", flush=True)
        if args.diagnose_jax_dispatch_cache:
            value = structure_counter.jax_dispatch_cache_size
            print(
                "[memory test] jax_dispatch_cache_size="
                + ("unavailable" if value is None else str(value)),
                flush=True,
            )
        if args.diagnose_raw_solve_dispatch:
            before = structure_counter.raw_solve_dispatch_before
            after = structure_counter.raw_solve_dispatch_after
            print(
                "[memory test] raw_solve_dispatch_cache "
                f"before={before if before is not None else 'unavailable'} "
                f"after={after if after is not None else 'unavailable'}",
                flush=True,
            )
        if args.diagnose_raw_wrapper_dispatch:
            entries = " ".join(
                f"{name}={value if value is not None else 'unavailable'}"
                for name, value in structure_counter.raw_wrapper_dispatch_caches.items()
            )
            print(f"[memory test] raw_wrapper_dispatch_cache {entries or 'unavailable'}", flush=True)
        if args.diagnose_pre_raw_dispatch:
            entries = " ".join(
                f"{name}={value if value is not None else 'unavailable'}"
                for name, value in structure_counter.pre_raw_dispatch_caches.items()
            )
            print(f"[memory test] pre_raw_dispatch_cache {entries or 'unavailable'}", flush=True)
        if args.diagnose_transport_dispatch:
            entries = " ".join(
                f"{name}={value if value is not None else 'unavailable'}"
                for name, value in structure_counter.transport_dispatch_caches.items()
            )
            print(f"[memory test] transport_dispatch_cache {entries or 'unavailable'}", flush=True)
        if args.diagnose_payload_dispatch:
            entries = " ".join(
                f"{name}={value if value is not None else 'unavailable'}"
                for name, value in structure_counter.payload_dispatch_caches.items()
            )
            print(f"[memory test] payload_dispatch_cache {entries or 'unavailable'}", flush=True)
        if args.diagnose_support_reverse_dispatch:
            entries = " ".join(
                f"{name}={value if value is not None else 'unavailable'}"
                for name, value in structure_counter.support_reverse_dispatch_caches.items()
            )
            print(
                f"[memory test] support_reverse_dispatch_cache {entries or 'unavailable'}",
                flush=True,
            )
        if args.diagnose_support_reverse_rss:
            entries = []
            for name, value in structure_counter.support_reverse_rss_bytes.items():
                baseline = first_support_reverse_rss_bytes.setdefault(name, value)
                if value is None or baseline is None:
                    delta = "unavailable"
                else:
                    delta = f"{(value - baseline) / 2**20:+.1f}MiB"
                entries.append(f"{name}={delta}")
            print(
                f"[memory test] support_reverse_rss_delta {' '.join(entries) or 'unavailable'}",
                flush=True,
            )
        if args.diagnose_optimization_stage_dispatch:
            entries = " ".join(
                f"{name}={value if value is not None else 'unavailable'}"
                for name, value in structure_counter.optimization_stage_dispatch_caches.items()
            )
            print(
                f"[memory test] optimization_stage_dispatch_cache {entries or 'unavailable'}",
                flush=True,
            )
        if args.diagnose_geometry_dispatch:
            entries = " ".join(
                f"{name}={value if value is not None else 'unavailable'}"
                for name, value in structure_counter.geometry_dispatch_caches.items()
            )
            print(
                f"[memory test] geometry_dispatch_cache {entries or 'unavailable'}",
                flush=True,
            )

    print(
        f"[memory test] mode={args.mode} objective_set={args.objective_set} "
        f"warmup={args.warmup} repeats={args.repeats} parameter_count={problem.parameter_count}",
        flush=True,
    )
    if args.clear_jax_caches or args.trim_native_heap:
        cleanup_parts = []
        if args.clear_jax_caches:
            cleanup_parts.append("jax.clear_caches")
        if args.trim_native_heap:
            cleanup_parts.append("malloc_trim")
        print(
            "[memory test] diagnostic_cleanup=" + "+".join(cleanup_parts) + " after each trial",
            flush=True,
        )
    if args.persistent_raw_solve_jit:
        print("[memory test] raw_solve_boundary=persistent_optimization_jit", flush=True)
    if args.persistent_raw_params:
        print("[memory test] raw_parameter_setup=persistent_optimization_base", flush=True)
    if args.persistent_raw_param_updates_jit:
        print("[memory test] raw_parameter_updates=persistent_optimization_jit", flush=True)
    for warmup_index in range(args.warmup):
        print(f"[memory test] warmup={warmup_index} starting", flush=True)
        started = time.perf_counter()
        structure_counter.reset()
        patchers = (
            structure_counter.context(
                checkpoints=args.diagnose_checkpoints,
                optimization_raw_block_stage=optimization_raw_block_stage,
                reuse_base_implicit_params=args.persistent_raw_params,
                jit_boundary_parameter_updates=args.persistent_raw_param_updates_jit,
                vmex_cache_sizes=args.diagnose_vmex_caches,
                jax_dispatch_cache=args.diagnose_jax_dispatch_cache,
                raw_solve_dispatch=args.diagnose_raw_solve_dispatch,
                raw_wrapper_dispatch=args.diagnose_raw_wrapper_dispatch,
                pre_raw_dispatch=args.diagnose_pre_raw_dispatch,
                transport_dispatch=args.diagnose_transport_dispatch,
                payload_dispatch=args.diagnose_payload_dispatch,
                support_reverse_dispatch=args.diagnose_support_reverse_dispatch,
                support_reverse_rss=args.diagnose_support_reverse_rss,
                optimization_stage_dispatch=args.diagnose_optimization_stage_dispatch,
                geometry_dispatch=args.diagnose_geometry_dispatch,
                raw_implicit=getattr(problem.raw_block_stage, "implicit", None),
            )
            if (
                args.diagnose_structure
                or args.diagnose_checkpoints
                or args.diagnose_vmex_caches
                or args.diagnose_jax_dispatch_cache
                or args.diagnose_payload_dispatch
                or args.diagnose_support_reverse_dispatch
                or args.diagnose_support_reverse_rss
                or args.diagnose_optimization_stage_dispatch
                or args.diagnose_geometry_dispatch
                or args.diagnose_raw_solve_dispatch
                or args.persistent_raw_solve_jit
                or args.persistent_raw_params
                or args.persistent_raw_param_updates_jit
            )
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
    if (
        args.diagnose_structure
        or args.diagnose_checkpoints
        or args.diagnose_vmex_caches
        or args.diagnose_jax_dispatch_cache
        or args.diagnose_payload_dispatch
        or args.diagnose_support_reverse_dispatch
        or args.diagnose_support_reverse_rss
        or args.diagnose_optimization_stage_dispatch
        or args.diagnose_geometry_dispatch
        or args.diagnose_raw_solve_dispatch
        or args.persistent_raw_solve_jit
        or args.persistent_raw_params
        or args.persistent_raw_param_updates_jit
    ):
        for iteration in range(args.repeats):
            structure_counter.reset()
            patchers = structure_counter.context(
                checkpoints=args.diagnose_checkpoints,
                optimization_raw_block_stage=optimization_raw_block_stage,
                reuse_base_implicit_params=args.persistent_raw_params,
                jit_boundary_parameter_updates=args.persistent_raw_param_updates_jit,
                vmex_cache_sizes=args.diagnose_vmex_caches,
                jax_dispatch_cache=args.diagnose_jax_dispatch_cache,
                raw_solve_dispatch=args.diagnose_raw_solve_dispatch,
                raw_wrapper_dispatch=args.diagnose_raw_wrapper_dispatch,
                pre_raw_dispatch=args.diagnose_pre_raw_dispatch,
                transport_dispatch=args.diagnose_transport_dispatch,
                payload_dispatch=args.diagnose_payload_dispatch,
                support_reverse_dispatch=args.diagnose_support_reverse_dispatch,
                support_reverse_rss=args.diagnose_support_reverse_rss,
                optimization_stage_dispatch=args.diagnose_optimization_stage_dispatch,
                geometry_dispatch=args.diagnose_geometry_dispatch,
                raw_implicit=getattr(problem.raw_block_stage, "implicit", None),
            )
            for patcher in patchers:
                patcher.start()
            try:
                if args.clear_jax_caches or args.trim_native_heap:
                    sample, heap_trimmed = _one_cleanup_sample(
                        quiet_problem,
                        x,
                        iteration,
                        clear_jax_caches=args.clear_jax_caches,
                        trim_native_heap=args.trim_native_heap,
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
    elif args.clear_jax_caches or args.trim_native_heap:
        for iteration in range(args.repeats):
            sample, heap_trimmed = _one_cleanup_sample(
                quiet_problem,
                x,
                iteration,
                clear_jax_caches=args.clear_jax_caches,
                trim_native_heap=args.trim_native_heap,
            )
            report(sample)
            if heap_trimmed is not None:
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
