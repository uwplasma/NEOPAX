r"""Isolate exact database stage Jacobian/solve/state costs without a rollout.

Run from the repository root, in the existing NEOPAX environment::

    python tests/benchmark_database_reverse_stage_cost.py --component independent \
        --n-radial 51 --objectives 10 --warmups 5 --repeats 20 \
        --save-results --dump-dir /tmp/database-stage-independent

Run --component shared in a separate process/directory for the production
candidate; --compare-results BASELINE.npz CANDIDATE.npz checks saved outputs
without importing JAX or preparing a fixture. The default is one independent
stage kernel. --component compare explicitly restores the old two-kernel run.
Support components are separate choices, never an implicit pair. For graph
inspection use --component support_table --lower-only --dump-dir DIRECTORY;
the selected kernel is not compiled, but fixture setup still evaluates its RHS.

This uses production equations and a fixed file-backed Monoenergetic database.
It neither solves VMEC nor builds an NTX scan or ambipolar root. Stage values
are constructed locally, not saved converged stages from the user's run.
Consequently timings diagnose this fixture, not the full reverse/forward ratio.
The optional shared-J candidate retains the same finite jacfwd derivative and
dense pivoted solve; it changes no production implementation or default flag.

The optional support kernels are NOT lightweight: on the local WSL host even
a fresh five-radius support-only compiler approached 7 GB RSS before being
stopped. Use sufficient host memory; do not combine every probe on a small host.
"""

from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def load_dependencies():
    """Keep --help dependency-free and saved-result comparisons NumPy-only."""
    global jax, jnp, np, NEOPAX, build_runtime_context
    global prepare_transport_solver_components, solvers
    started = time.perf_counter()
    import jax
    import jax.numpy as jnp
    import numpy as np
    import NEOPAX
    from NEOPAX._orchestrator import (
        build_runtime_context,
        prepare_transport_solver_components,
    )
    from NEOPAX import _transport_solvers as solvers
    return time.perf_counter() - started


def machine_metadata():
    memory = {}
    memory_file = Path("/proc/meminfo")
    if memory_file.is_file():
        for line in memory_file.read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                memory[key + "_bytes"] = int(value.split()[0]) * 1024
    versions = {}
    for package in ("jax", "jaxlib", "numpy", "scipy", "jax-cuda12-plugin", "jax-cuda12-pjrt"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "logical_cpus": os.cpu_count(), "python": sys.version,
        "executable": sys.executable, "pid": os.getpid(),
        "versions": versions, "host_memory": memory,
        "environment": {key: os.environ[key] for key in (
            "JAX_PLATFORMS", "JAX_PLATFORM_NAME", "CUDA_VISIBLE_DEVICES", "XLA_FLAGS",
            "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_ALLOCATOR",
        ) if key in os.environ},
    }


def write_json(args, filename, payload):
    if args.dump_dir:
        directory = Path(args.dump_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / filename).write_text(json.dumps(payload, indent=2))


def prepare_case(args):
    config = NEOPAX.prepare_config(
        ROOT / "examples/benchmarks/"
        "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml",
        device=args.device,
    )
    # Use stored equilibrium and database inputs. These overrides belong only
    # to this diagnostic fixture and cannot enter the production CLI.
    config["geometry"] = {
        "n_radial": args.n_radial,
        "vmec_file": str(ROOT / "examples/inputs/wout_QI_nfp2_newNT_opt_hires.nc"),
        "boozer_file": str(ROOT / "examples/inputs/boozermn_wout_QI_nfp2_newNT_opt_hires.nc"),
    }
    config["neoclassical"] = {
        "flux_model": "ntx_database",
        "entropy_model": "ntx_database",
        "neoclassical_file": str(ROOT / "examples/inputs/Dij_NTX_QI_Er_Optimized.h5"),
        "interpolation_mode": "generic",
    }
    # The runtime's documented ambipolarity mode constructs profiles without
    # invoking its optional initial root solve. No root derivative is measured.
    config["general"]["mode"] = "ambipolarity"
    config["transport_solver"]["debug_stage_markers"] = False
    config["transport_solver"]["debug_walltime_attempts"] = False
    args.fixture_config = {
        "geometry": dict(config["geometry"]),
        "neoclassical": dict(config["neoclassical"]),
        "energy_grid": dict(config["energy_grid"]),
        "general_mode": config["general"]["mode"],
    }
    runtime, state = build_runtime_context(config)
    if type(runtime.database).__name__ != "Monoenergetic":
        raise TypeError("This diagnostic requires the production Monoenergetic table representation.")
    components = prepare_transport_solver_components(config, runtime, state)
    prepared = solvers._build_prepared_radau_accepted_rollout(
        solver=components["solver"],
        state=components["solve_state"],
        vector_field=components["solve_vector_field"],
        species=runtime.species,
    )
    kernel = prepared.kernel_context
    physics = dataclasses.replace(
        prepared.physics_context,
        reverse_stage_adjoint_solve_mode="block",
        reverse_rhs_transpose_mode="explicit_database",
        reverse_stage_cotangent_mode="full",
        reverse_stage_adjoint_memory_mode="default",
        reverse_rhs_pullback_mode="separate",
        reverse_segment_input_diagnostics=False,
        reverse_database_stage_jacobian_mode="independent",
    )
    if kernel.use_transport_lagged_response:
        raise ValueError("This diagnostic must use the direct database RHS.")
    carry = prepared.initial_carry
    rhs = jax.jit(physics.flat_rhs)(carry.t, carry.y)
    jax.block_until_ready(rhs)
    history = jnp.tile(rhs, kernel.num_stages)
    y_bars = jnp.asarray(
        np.random.default_rng(20260912).normal(size=(args.objectives, kernel.state_dim)),
        dtype=carry.y.dtype,
    )
    inputs = (carry.t, carry.y, jnp.asarray(args.dt, carry.y.dtype), history, y_bars)
    partial_hooks = {}
    support_component = args.support_component if args.support_components else args.component
    if support_component in {"support_flux_geometry", "support_equation_geometry"}:
        # Bind genuine equation-system partials with exactly the projection and
        # cotangent transform used by _build_prepared_radau_accepted_rollout.
        vector_field = components["solve_vector_field"]
        temperature_mask, fixed_temperature = solvers._extract_fixed_temperature_projection(vector_field)
        density_floor, temperature_floor = solvers._extract_state_regularization(vector_field)
        _, unpack_flat, _, _, project_flat = solvers._make_solver_state_transform(
            components["solve_state"], runtime.species,
            temperature_active_mask=temperature_mask,
            fixed_temperature_profile=fixed_temperature,
            density_floor=density_floor, temperature_floor=temperature_floor,
        )
        partial = support_component.removeprefix("support_")
        partial_hooks[support_component] = solvers._flat_rhs_direct_database_payload_pullback_factory(
            unravel=unpack_flat, vector_field=vector_field,
            args=(runtime.species,), kwargs={}, project_flat=project_flat,
            hook_name=f"pullback_direct_rhs_database_{partial}_payload",
        )
    return runtime, kernel, physics, carry, inputs, partial_hooks


def build_kernels(kernel, physics, initial_carry, *, candidate="local"):
    def contexts(t, y, h, history):
        return (
            dataclasses.replace(initial_carry, t=t, y=y),
            SimpleNamespace(trial_dt=h, stage_history=history),
        )

    def jacobians(t, y, h, history):
        carry, primal = contexts(t, y, h, history)
        times, states = solvers._radau_exact_stage_times_states(kernel, carry, primal)
        return jax.vmap(
            lambda time_value, state: jax.jacfwd(
                lambda value: physics.flat_rhs(time_value, value)
            )(state)
        )(times, states)

    def matrix_from_jacobians(h, stage_jacobians):
        blocks = (
            jnp.eye(kernel.num_stages, dtype=kernel.dtype)[:, :, None, None]
            * jnp.eye(kernel.state_dim, dtype=kernel.dtype)[None, None, :, :]
            - h * kernel.a[:, :, None, None] * stage_jacobians[:, None, :, :]
        )
        return jnp.transpose(blocks, (0, 2, 1, 3)).reshape(
            (kernel.num_stages * kernel.state_dim,) * 2
        )

    def solve_matrix(matrix, h, bars):
        rhs = (h * kernel.b[None, :, None] * bars[:, None, :]).reshape((bars.shape[0], -1))
        # Retain the original vmap solve layout to isolate Jacobian reuse.
        return jax.vmap(lambda row: jnp.linalg.solve(matrix.T, -row))(rhs)

    def state_from_jacobians(stage_jacobians, residuals):
        staged = residuals.reshape((-1, kernel.num_stages, kernel.state_dim))
        return jax.vmap(
            lambda row: -jnp.sum(
                jax.vmap(lambda jacobian, bar: jacobian.T @ bar)(stage_jacobians, row),
                axis=0,
            )
        )(staged)

    def baseline(t, y, h, history, bars):
        carry, primal = contexts(t, y, h, history)
        rhs = (h * kernel.b[None, :, None] * bars[:, None, :]).reshape((bars.shape[0], -1))
        with jax.named_scope("production_stage_solve"):
            residuals = solvers._radau_solve_exact_stage_residual_transpose_batched(
                kernel, physics, carry, primal, None, rhs=rhs,
            )
        with jax.named_scope("production_outgoing_state"):
            outgoing = jax.vmap(
                lambda row: solvers._radau_exact_stage_residual_input_pullback(
                    kernel, physics, carry, primal, None, row, compute_dt_bar=False,
                )[0]
            )(residuals)
        return residuals, outgoing

    def shared_jacobian(t, y, h, history, bars):
        if candidate == "production":
            shared_physics = dataclasses.replace(
                physics, reverse_database_stage_jacobian_mode="shared"
            )
            carry, primal = contexts(t, y, h, history)
            rhs = (h * kernel.b[None, :, None] * bars[:, None, :]).reshape((bars.shape[0], -1))
            return solvers._radau_database_shared_stage_solve_and_state_pullback_batched(
                kernel, shared_physics, carry, primal, rhs=rhs,
            )
        with jax.named_scope("shared_stage_jacobians"):
            stage_jacobians = jacobians(t, y, h, history)
        matrix = matrix_from_jacobians(h, stage_jacobians)
        residuals = solve_matrix(matrix, h, bars)
        return residuals, state_from_jacobians(stage_jacobians, residuals)

    return {
        "baseline": baseline,
        "shared_jacobian": shared_jacobian,
        "stage_jacobians": jacobians,
        "matrix_from_jacobians": matrix_from_jacobians,
        "solve_matrix": solve_matrix,
        "state_from_jacobians": state_from_jacobians,
    }


def measure_support(runtime, kernel, physics, initial_carry, inputs, residuals, args, partial_hooks):
    """Measure one genuine partial using the production objective/stage layout.

    Standalone partials each prepare their own primals. Their timings are not
    additive with the shared-preparation table-and-geometry total.
    """
    support = {"geometry": runtime.geometry, "database": runtime.database}
    selected = args.support_component if args.support_components else args.component
    include_geometry = selected == "support_table_and_geometry"
    support_physics = dataclasses.replace(
        physics,
        reverse_database_table_only=True,
        reverse_database_include_direct_geometry=include_geometry,
    )
    required = (
        "flat_rhs_direct_database_split_support_pullback"
        if include_geometry else "flat_rhs_direct_database_table_pullback"
    )
    if selected in {"support_flux_geometry", "support_equation_geometry"}:
        # Reuse the exact scalar-vmap/stage-scan driver, replacing only its
        # selected scalar boundary. Never infer cost by projecting a total.
        support_physics = dataclasses.replace(
            support_physics,
            flat_rhs_direct_database_table_pullback=partial_hooks[selected],
            flat_rhs_direct_database_table_pullback_batched=None,
        )
    if getattr(support_physics, required, None) is None:
        raise RuntimeError(f"Production support hook unavailable for {selected}: {required}")

    def support_kernel(t, y, h, history, bars, payload):
        return solvers._radau_exact_stage_residual_database_table_support_pullback_batched(
            kernel, support_physics,
            dataclasses.replace(initial_carry, t=t, y=y),
            SimpleNamespace(trial_dt=h, stage_history=history), bars, payload,
        )

    # The production boundary returns flattened support leaves; save their
    # original names as well as the returned tuple structure for comparison.
    paths, _ = jax.tree_util.tree_flatten_with_path(solvers._radau_zero_support_delta_tree_like(support))
    args.output_leaf_labels = [jax.tree_util.keystr(path) for path, _ in paths]
    args.fixture_signature = {
        **args.fixture_signature,
        "support_residuals_sha256": input_fingerprint(residuals),
    }
    write_json(args, f"{selected}.fixture.json", {
        "fixture_signature": args.fixture_signature,
        "support_leaf_labels": args.output_leaf_labels,
        "seed": "solved stage adjoints" if args.support_components else "constructed finite cotangents",
    })
    measure(selected, support_kernel, (*inputs[:4], residuals, support), args)


def measure(name, function, inputs, args):
    # Exclude outstanding setup transfers/kernels from lowering and timings.
    jax.block_until_ready(inputs)
    print(json.dumps({"kernel": name, "phase": "lowering"}), flush=True)
    started = time.perf_counter()
    lowered = jax.jit(function).lower(*inputs)
    lowering_s = time.perf_counter() - started
    report = {"kernel": name, "phase": "lowered", "lowering_s": lowering_s}
    # Persist before compile: an OOM must not discard the lowered graph or the
    # evidence that compilation, rather than warm execution, was unfinished.
    write_json(args, f"{name}.lowering.json", report)
    if args.dump_dir:
        started = time.perf_counter()
        (Path(args.dump_dir) / f"{name}.stablehlo.txt").write_text(str(lowered.compiler_ir()))
        report["stablehlo_export_s"] = time.perf_counter() - started
    write_json(args, f"{name}.json", report)
    if args.lower_only:
        print(json.dumps(report), flush=True)
        return None
    report["phase"] = "compiling"
    write_json(args, f"{name}.json", report)
    print(json.dumps(report), flush=True)
    started = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - started
    report.update(phase="compiled", compile_s=compile_s)
    write_json(args, f"{name}.json", report)
    warmups = []
    for _ in range(args.warmups):
        started = time.perf_counter()
        result = jax.block_until_ready(compiled(*inputs))
        warmups.append(time.perf_counter() - started)
    samples = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        result = jax.block_until_ready(compiled(*inputs))
        samples.append(time.perf_counter() - started)
    hlo = compiled.as_text()
    opcodes = Counter(re.findall(r" = .*?\b([a-z][a-z0-9-]*)\(", hlo))
    report.update({
        "phase": "completed",
        "synchronization": "inputs and every warmup/repeat block_until_ready",
        "warmup_samples_s": warmups,
        "warm_median_s": statistics.median(samples),
        "warm_min_s": min(samples), "warm_max_s": max(samples),
        "warm_samples_s": samples,
        "optimized_hlo_bytes": len(hlo.encode()),
        "opcode_counts": dict(sorted(opcodes.items())),
        "memory_analysis": str(compiled.memory_analysis()),
        "cost_analysis": compiled.cost_analysis(),
        "cost_analysis_note": "compiler estimates, not physical counters; component timings are not additive",
    })
    if not all(np.all(np.isfinite(np.asarray(value))) for value in jax.tree_util.tree_leaves(result)):
        report["phase"] = "nonfinite_result"
        write_json(args, f"{name}.json", report)
        raise AssertionError(f"{name}: fixture produced nonfinite cotangents")
    if args.dump_dir:
        directory = Path(args.dump_dir)
        (directory / f"{name}.optimized.hlo.txt").write_text(hlo)
        if args.save_results:
            save_results(directory / f"{name}.npz", result, args)
    write_json(args, f"{name}.json", report)
    concise = dict(report)
    analysis = report["cost_analysis"]
    if isinstance(analysis, dict):
        concise["cost_analysis"] = {
            key: analysis[key]
            for key in ("flops", "transcendentals", "bytes accessed")
            if key in analysis
        }
    print(json.dumps(concise), flush=True)
    return result


def tree_metadata(tree):
    paths, structure = jax.tree_util.tree_flatten_with_path(tree)
    leaves = []
    for path, value in paths:
        array = np.asarray(value)
        leaves.append({"path": jax.tree_util.keystr(path), "shape": list(array.shape),
                       "dtype": str(array.dtype), "nbytes": int(array.nbytes)})
    return {"structure": str(structure), "leaves": leaves,
            "total_leaf_nbytes": sum(value["nbytes"] for value in leaves)}


def input_fingerprint(inputs):
    digest = hashlib.sha256()
    for value in jax.tree_util.tree_leaves(inputs):
        array = np.asarray(value)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_results(filename, result, args):
    paths, structure = jax.tree_util.tree_flatten_with_path(result)
    description = {
        "schema": 1, "structure": str(structure),
        "paths": [jax.tree_util.keystr(path) for path, _ in paths],
        "support_leaf_labels": getattr(args, "output_leaf_labels", None),
        "fixture_signature": args.fixture_signature,
    }
    arrays = {f"leaf_{index:04d}": np.asarray(value) for index, (_, value) in enumerate(paths)}
    arrays["__metadata__"] = np.asarray(json.dumps(description))
    np.savez(filename, **arrays)


def compare_results(filenames, *, rtol, atol):
    """Check saved structures and finite arrays without JAX or fixture setup."""
    import numpy as np

    with np.load(filenames[0], allow_pickle=False) as reference, np.load(filenames[1], allow_pickle=False) as result:
        reference_meta = json.loads(str(reference["__metadata__"]))
        result_meta = json.loads(str(result["__metadata__"]))
        if reference_meta != result_meta:
            raise AssertionError("Saved result structure or fixture signature differs")
        if set(reference.files) != set(result.files):
            raise AssertionError("Saved array keys differ")
        maximum = 0.0
        equal = True
        leaf_count = 0
        for key in sorted(set(reference.files) - {"__metadata__"}):
            expected, actual = reference[key], result[key]
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                raise AssertionError(f"{key}: shape/dtype differs")
            if not (np.all(np.isfinite(expected)) and np.all(np.isfinite(actual))):
                raise AssertionError(f"{key}: nonfinite saved result")
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
            maximum = max(maximum, float(np.max(np.abs(actual - expected), initial=0.0)))
            equal = equal and np.array_equal(actual, expected)
            leaf_count += 1
        print(json.dumps({"comparison": "passed", "leaves": leaf_count,
                          "exact_elementwise_equal": equal, "max_abs": maximum,
                          "rtol": rtol, "atol": atol}), flush=True)


SUPPORT_COMPONENTS = (
    "support_table", "support_flux_geometry", "support_equation_geometry",
    "support_table_and_geometry",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--component", choices=(
        "independent", "shared", "local_shared", "stage_jacobians", "compare",
        "production_parity", *SUPPORT_COMPONENTS,
    ), help="One kernel per process by default: independent; shared uses the production helper.")
    parser.add_argument("--n-radial", type=int, default=51)
    parser.add_argument("--objectives", type=int, default=10)
    parser.add_argument("--dt", type=float, default=1.0e-8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--components", action="store_true",
                        help="Legacy opt-in: compare stage kernels then measure four subcomponents in one process.")
    parser.add_argument("--support-components", action="store_true",
                        help="Legacy opt-in: compare stage kernels then measure ONE selected support component.")
    parser.add_argument("--support-component", choices=SUPPORT_COMPONENTS, default="support_table",
                        help="Component for legacy --support-only/--support-components; default table only.")
    parser.add_argument("--support-only", action="store_true",
                        help="Legacy alias: run ONE selected support component with constructed cotangents.")
    parser.add_argument("--candidate", choices=("local", "production"), default="local")
    parser.add_argument("--production-parity-only", action="store_true",
                        help="Legacy alias for --component production_parity.")
    parser.add_argument("--lower-only", action="store_true",
                        help="Export selected kernel StableHLO without compiling it; fixture setup still runs.")
    parser.add_argument("--save-results", action="store_true",
                        help="Export result leaves and structure to NPZ after timing (may be large for support).")
    parser.add_argument("--compare-results", nargs=2, metavar=("REFERENCE_NPZ", "RESULT_NPZ"),
                        help="NumPy-only saved result comparison; no fixture, JAX import, or compilation.")
    parser.add_argument("--rtol", type=float, default=2e-10)
    parser.add_argument("--atol", type=float, default=2e-12)
    parser.add_argument("--dump-dir")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmups < 1:
        parser.error("--repeats and --warmups must be positive")
    if args.n_radial < 3 or args.objectives < 1 or not (0 < args.dt < float("inf")):
        parser.error("--n-radial must be at least 3; --objectives and finite --dt must be positive")
    if not (0 <= args.rtol < float("inf") and 0 <= args.atol < float("inf")):
        parser.error("--rtol and --atol must be finite and nonnegative")
    legacy = args.components or args.support_components or args.support_only or args.production_parity_only
    if args.component and legacy:
        parser.error("Use --component or legacy selection flags, not both")
    if (args.support_only and (args.components or args.support_components or args.production_parity_only)
            or args.production_parity_only and (args.components or args.support_components)):
        parser.error("Conflicting legacy kernel selections")
    if args.component is None:
        args.component = (
            args.support_component if args.support_only else
            "production_parity" if args.production_parity_only else
            "compare" if args.components or args.support_components else "independent"
        )
    if (args.lower_only or args.save_results) and not args.dump_dir:
        parser.error("--lower-only and --save-results require --dump-dir")
    if args.lower_only and (args.save_results or args.component == "compare"):
        parser.error("--lower-only requires one kernel and cannot save numerical results")
    return args


def main():
    args = parse_args()
    if args.compare_results:
        compare_results(args.compare_results, rtol=args.rtol, atol=args.atol)
        return
    run = {"phase": "importing", "machine": machine_metadata(), "arguments": vars(args).copy()}
    write_json(args, "run.json", run)
    import_s = load_dependencies()
    jax.config.update("jax_enable_x64", True)
    run.update(phase="preparing", import_s=import_s)
    write_json(args, "run.json", run)
    preparation_started = time.perf_counter()
    runtime, kernel, physics, carry, inputs, partial_hooks = prepare_case(args)
    preparation_s = time.perf_counter() - preparation_started
    args.fixture_signature = {
        "state_inputs_sha256": input_fingerprint(inputs),
        "database_sha256": input_fingerprint(runtime.database),
        "geometry_sha256": input_fingerprint(runtime.geometry),
        "n_radial": args.n_radial, "objectives": args.objectives, "dt": args.dt,
        "database_type": type(runtime.database).__name__,
        "interpolation_mode": args.fixture_config["neoclassical"]["interpolation_mode"],
        "energy_grid": args.fixture_config["energy_grid"],
    }
    devices = [{"id": device.id, "platform": device.platform, "kind": device.device_kind,
                "memory_stats": device.memory_stats()} for device in jax.devices()]
    run.update({
        "phase": "prepared",
        "fixture": "stored equilibrium/database; constructed unconverged stages; no VMEC/scan/root/rollout",
        "jax": jax.__version__, "jax_enable_x64": bool(jax.config.jax_enable_x64), "devices": devices,
        "species": int(runtime.species.number_species), "radial": args.n_radial,
        "state_dim": kernel.state_dim, "stages": kernel.num_stages,
        "objectives": args.objectives, "database_type": type(runtime.database).__name__,
        "database": tree_metadata(runtime.database), "energy_grid": tree_metadata(runtime.energy_grid),
        "fixture_overrides": args.fixture_config, "fixture_signature": args.fixture_signature,
        "import_s": import_s, "preparation_s": preparation_s,
        "component": args.component,
        "support_note": "standalone partial preparation costs are not additive with shared-preparation total",
    })
    write_json(args, "run.json", run)
    print(json.dumps(run), flush=True)
    kernels = build_kernels(kernel, physics, carry, candidate=args.candidate)
    if args.component == "production_parity":
        local = build_kernels(kernel, physics, carry, candidate="local")["shared_jacobian"]
        production = build_kernels(kernel, physics, carry, candidate="production")["shared_jacobian"]

        def parity_kernel(*values):
            return local(*values), production(*values)

        outputs = measure("production_helper_parity", parity_kernel, inputs, args)
        if args.lower_only:
            return
        expected, actual = outputs
        for name, reference, result in zip(("residuals", "outgoing"), expected, actual, strict=True):
            reference, result = np.asarray(reference), np.asarray(result)
            if not (np.all(np.isfinite(reference)) and np.all(np.isfinite(result))):
                raise AssertionError(f"{name}: production fixture produced nonfinite values")
            np.testing.assert_array_equal(result, reference)
            print(json.dumps({"production_parity": name, "max_abs": float(np.max(np.abs(result - reference)))}), flush=True)
        return
    if args.component in SUPPORT_COMPONENTS:
        h, bars = inputs[2], inputs[4]
        residuals = (-h * kernel.b[None, :, None] * bars[:, None, :]).reshape((args.objectives, -1))
        print(json.dumps({"support_seed": "constructed finite cotangents, not solved stage adjoints"}), flush=True)
        measure_support(runtime, kernel, physics, carry, inputs, residuals, args, partial_hooks)
        return
    if args.component == "stage_jacobians":
        measure("stage_jacobians", kernels["stage_jacobians"], inputs[:4], args)
        return
    if args.component == "independent":
        measure("independent", kernels["baseline"], inputs, args)
        return
    if args.component in {"shared", "local_shared"}:
        candidate_name = "production" if args.component == "shared" else "local"
        selected = build_kernels(kernel, physics, carry, candidate=candidate_name)["shared_jacobian"]
        measure(args.component, selected, inputs, args)
        return
    baseline = measure("baseline", kernels["baseline"], inputs, args)
    candidate = measure("shared_jacobian", kernels["shared_jacobian"], inputs, args)
    for name, expected, actual in zip(("residuals", "outgoing"), baseline, candidate, strict=True):
        expected, actual = np.asarray(expected), np.asarray(actual)
        if not (np.all(np.isfinite(expected)) and np.all(np.isfinite(actual))):
            raise AssertionError(f"{name}: fixture produced nonfinite values")
        np.testing.assert_allclose(actual, expected, rtol=args.rtol, atol=args.atol)
        print(json.dumps({"parity": name, "max_abs": float(np.max(np.abs(actual - expected)))}), flush=True)
    if args.components:
        t, y, h, history, bars = inputs
        jacobians = measure("stage_jacobians", kernels["stage_jacobians"], (t, y, h, history), args)
        matrix = measure("matrix_from_jacobians", kernels["matrix_from_jacobians"], (h, jacobians), args)
        residuals = measure("solve_matrix", kernels["solve_matrix"], (matrix, h, bars), args)
        measure("state_from_jacobians", kernels["state_from_jacobians"], (jacobians, residuals), args)
    if args.support_components:
        measure_support(runtime, kernel, physics, carry, inputs, baseline[0], args, partial_hooks)


if __name__ == "__main__":
    main()
