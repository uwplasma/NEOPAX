"""Compare scalar-query AD and flux/state Jacobians in fresh GPU processes.

Uses saved coefficients and common synthetic profiles/mesh, with synthetic
NTSS fit geometry. No VMEC, NTX, ambipolar-root or transport solve runs.
Every method is checked against its own uncontracted derivative, not another
method's numerical rule. Outputs are CSV measurements and NPZ arrays.
"""
import argparse
import csv
import dataclasses
import inspect
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODES = ("generic", "generic_3d", "generic_loger_no_r", "preprocessed_3d",
         "preprocessed_3d_radial", "preprocessed_3d_radial_ntss1d",
         "preprocessed_3d_ntss1d_fixed", "preprocessed_ntss")


def collect_measurements(args, modes):
    rows = []
    for mode in modes:
        for implementation in ("uncontracted", "optimized"):
            root = ((args.reference_dir or args.out_dir) if implementation == "uncontracted"
                    else args.out_dir) / mode / implementation
            for name in ("queries.csv", "flux/measurements.csv"):
                if (root / name).exists():
                    with (root / name).open() as stream:
                        for row in csv.DictReader(stream):
                            row.update(mode=mode, implementation=implementation, status="completed")
                            row.setdefault("component", row.get("probe", "flux_jacobian"))
                            rows.append(row)
            if (root / "flux/workspace.csv").exists():
                with (root / "flux/workspace.csv").open() as stream:
                    row = next(csv.DictReader(stream))
                row.update(mode=mode, implementation=implementation, component="flux_jacobian",
                           status="directional_reference_only")
                rows.append(row)
    fields = ("mode", "implementation", "component", "status", "compile_s", "repeat_median_s",
              "temporary_bytes", "peak_bytes_in_use", "pool_bytes", "peak_pool_bytes")
    with (args.out_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_case(mode, database_path, n_radial=51):
    import jax
    import numpy as np
    from examples.benchmarks.benchmark_initial_er_coordinate_pullback import build_case as base_case
    from NEOPAX._monoenergetic import load_monoenergetic_database

    # Only small table/profile preparation uses the CPU. Measurements use GPU
    # arrays exclusively; no runtime or physics solve is constructed here.
    with jax.default_device(jax.devices("cpu")[0]):
        model, state, _ = base_case(n_radial)
        mesh = model.geometry
        geometry = SimpleNamespace(**mesh._asdict(), rho_grid=np.asarray(mesh.r_grid / mesh.a_b),
                                   curvature=np.full(n_radial, .1), iota=np.full(n_radial, .6),
                                   epsilon_t=np.asarray(mesh.r_grid) / 10., R0=10.)
        database = load_monoenergetic_database(geometry, str(database_path), mode)
    species, energy, mesh, database, state, density_bc, temperature_bc = jax.device_put(
        (model.species, model.energy_grid, model.geometry, database, state,
         model.bc_density, model.bc_temperature), jax.devices("gpu")[0])
    return dataclasses.replace(model, species=species, energy_grid=energy, geometry=mesh,
                               database=database, bc_density=density_bc, bc_temperature=temperature_bc), state


def worker(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from examples.benchmarks.benchmark_initial_er_coordinate_pullback import memory_stats
    from examples.benchmarks.benchmark_native_database_interpolation import _flux_jacobian_worker
    from NEOPAX._monoenergetic_interpolators import MONOENERGETIC_INTERPOLATION_KERNELS

    jax.config.update("jax_enable_x64", True)
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("A GPU is required; no CPU benchmark was started")
    kernel = MONOENERGETIC_INTERPOLATION_KERNELS[args.mode]
    if args.implementation == "uncontracted":
        kernel = jax.jit(inspect.unwrap(kernel))
        MONOENERGETIC_INTERPOLATION_KERNELS[args.mode] = kernel
    model, state = build_case(args.mode, args.database)
    output = args.out_dir / args.mode / args.implementation
    output.mkdir(parents=True, exist_ok=True)
    print(f"[{args.mode}/{args.implementation}] JAX={jax.__version__} device={jax.devices()[0]} "
          f"state={state.density.shape} prepared_memory={memory_stats()}", flush=True)

    def scalar_query(point, database):
        return jnp.asarray(kernel(point[0], 10.**point[1], 10.**point[2], database))

    radii = np.linspace(.073, 1.927, 51)
    radial_knots = np.asarray(model.database.rho * model.database.a_b)
    radii += np.where(np.min(np.abs(radii[:, None] - radial_knots[None, :]), axis=1) < .004, .013, 0.)
    # Central FD at a radial/linear interpolation knot averages two branches;
    # it is not the derivative of either fixed branch used by AD.
    jitter = np.random.default_rng(71).uniform(-.002, .002, (2, 51))
    points = jnp.stack((jnp.asarray(radii), jnp.linspace(-4.37, -2.63, 51) + jitter[0],
                        jnp.linspace(-5.17, -2.43, 51) + jitter[1]), axis=1)
    query = jax.jit(jax.vmap(scalar_query, in_axes=(0, None)))
    derivative = jax.jit(jax.vmap(jax.jacfwd(scalar_query), in_axes=(0, None)))
    values = {}
    rows = []
    for name, function in (("query", query), ("query_jacobian", derivative)):
        start = time.perf_counter()
        compiled = function.lower(points, model.database).compile()
        compile_s = time.perf_counter() - start
        samples = []
        for index in range(args.repeats + 1):
            start = time.perf_counter()
            result = jax.block_until_ready(compiled(points, model.database))
            elapsed = time.perf_counter() - start
            if index:
                samples.append(elapsed)
        values[name] = np.asarray(result)
        if not np.isfinite(values[name]).all():
            raise FloatingPointError(f"Nonfinite {name}")
        rows.append(dict(mode=args.mode, implementation=args.implementation, component=name,
                         compile_s=compile_s, repeat_median_s=float(np.median(samples)),
                         temporary_bytes=compiled.memory_analysis().temp_size_in_bytes, **memory_stats()))
        print(rows[-1], flush=True)

    direction = jnp.asarray([.012, -.17, .23])
    expected = np.einsum("qij,j->qi", values["query_jacobian"], np.asarray(direction))
    errors = []
    for step in (1e-4, 1e-5, 1e-6):
        fd = np.asarray((query(points + step * direction, model.database)
                         - query(points - step * direction, model.database)) / (2 * step))
        errors.append(float(np.max(np.abs(fd - expected) / np.maximum(1., np.abs(expected)))))
    if min(errors) > 3e-6:
        print(f"[{args.mode}] largest_fd_error_point="
              f"{np.unravel_index(np.argmax(np.abs(fd-expected)), fd.shape)} "
              f"points={np.asarray(points).tolist()}", flush=True)
        raise AssertionError(f"Query directional FD mismatch: {errors}")
    print(f"[{args.mode}] query_directional_fd_errors={errors}", flush=True)
    reverse_queries = jax.jit(jax.vmap(jax.jacrev(scalar_query), in_axes=(0, None)))
    reverse_result = np.asarray(reverse_queries(points[::17], model.database))
    np.testing.assert_allclose(reverse_result, values["query_jacobian"][::17], rtol=3e-10, atol=3e-10)
    values["query_reverse_jacobian"] = reverse_result

    physical_edges = jnp.asarray([[0., 1e-4, 0.], [.001, 1e-14, 1e-20],
                                  [.73, 1e-3, -3e-4], [1.99, 1e2, .1]])
    def physical_query(point, database):
        return jnp.asarray(kernel(*point, database))
    edges = jax.jit(jax.vmap(physical_query, in_axes=(0, None)))(physical_edges, model.database)
    edge_jacobians = jax.jit(jax.vmap(jax.jacfwd(physical_query), in_axes=(0, None)))(physical_edges, model.database)
    if not all(np.isfinite(np.asarray(value)).all() for value in (edges, edge_jacobians)):
        raise FloatingPointError("Nonfinite axis/floor/negative-field/extrapolation query or tangent")
    values["edge_values"], values["edge_jacobians"] = np.asarray(edges), np.asarray(edge_jacobians)

    # Check an active coefficient direction through the actual database type,
    # including integer NTSS metadata in the surrounding pytree.
    field = "ag13" if args.mode == "preprocessed_ntss" else "D13"
    table = getattr(model.database, field)
    table_direction = jnp.sin(jnp.arange(table.size).reshape(table.shape) * .17)
    weights = jnp.asarray([.2, -.3, .05])

    def table_score(table_value):
        database = dataclasses.replace(model.database, **{field: table_value})
        return jnp.vdot(weights, scalar_query(points[21], database))

    table_bar = jax.jit(jax.grad(table_score))(table)
    table_ad = float(jnp.vdot(table_bar, table_direction))
    coefficient_check = jax.jit(table_score)
    table_fds = [float((coefficient_check(table + step * table_direction)
                        - coefficient_check(table - step * table_direction)) / (2 * step))
                 for step in (1e-5, 1e-6, 1e-7, 1e-8)]
    table_fd = min(table_fds, key=lambda value: abs(value - table_ad))
    print(f"[{args.mode}] table_ad={table_ad:.17g} table_fd_sequence={table_fds}", flush=True)
    np.testing.assert_allclose(table_ad, table_fd, rtol=3e-6, atol=3e-8)
    values["table_directional_derivative"] = np.asarray(table_ad)
    np.savez(output / "queries.npz", **values)
    with (output / "queries.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    if not args.queries_only:
        flux_output = output / "flux"
        flux_output.mkdir(exist_ok=True)
        _flux_jacobian_worker(SimpleNamespace(worker_backend=f"{args.mode}/{args.implementation}",
                                              repeats=args.repeats,
                                              max_temp_bytes=args.max_temporary_mib * 1024**2,
                                              directional_reference_only=getattr(args, "directional_reference_only", False)),
                              model, state, flux_output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / "examples/inputs/Dij_NTX_QI_Er_Optimized.h5")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/interpolation_methods_ad")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--implementations", default="uncontracted,optimized")
    parser.add_argument("--reference-dir", type=Path,
                        help="Reuse previously checked uncontracted NPZ results when rerunning optimized workers.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--queries-only", action="store_true")
    parser.add_argument("--max-temporary-mib", type=int, default=2048,
                        help="Skip execution of flux Jacobians exceeding this compiled temporary workspace.")
    parser.add_argument("--worker-timeout", type=int, default=300)
    parser.add_argument("--directional-reference-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--implementation", choices=("uncontracted", "optimized"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.mode:
        worker(args)
        return
    import numpy as np
    modes = args.modes.split(",")
    if any(mode not in MODES for mode in modes):
        parser.error("Unknown interpolation mode")
    implementations = args.implementations.split(",")
    if any(value not in ("uncontracted", "optimized") for value in implementations):
        parser.error("Unknown implementation")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for mode in modes:
        for implementation in implementations:
            command = [sys.executable, str(Path(__file__).resolve()), "--mode", mode,
                       "--implementation", implementation, "--database", str(args.database),
                       "--out-dir", str(args.out_dir), "--repeats", str(args.repeats),
                       "--max-temporary-mib", str(args.max_temporary_mib)]
            if args.queries_only:
                command.append("--queries-only")
            with (args.out_dir / f"{mode}_{implementation}.log").open("w") as log:
                try:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=args.worker_timeout)
                    returncode = result.returncode
                except subprocess.TimeoutExpired:
                    returncode = 124
                    log.write("\nWorker exceeded the component-probe time limit.\n")
            print(f"[{mode}/{implementation}] exit={returncode}", flush=True)
            if returncode:
                failed.append(f"{mode}/{implementation}")
        for filename in (("queries.npz",) if args.queries_only else ("queries.npz", "flux/values.npz")):
            if "optimized" not in implementations:
                continue
            files = [(args.reference_dir or args.out_dir) / mode / "uncontracted" / filename,
                     args.out_dir / mode / "optimized" / filename]
            if not all(path.exists() for path in files):
                failed.append(f"{mode}/missing_comparison/{filename}")
            else:
                with np.load(files[0]) as reference, np.load(files[1]) as actual:
                    if "directions" in reference.files:
                        predicted = np.einsum("sij,ksj->ksi", actual["flux_jacobian"], reference["directions"])
                        np.testing.assert_allclose(predicted, reference["directional_derivatives"],
                                                   rtol=3e-10, atol=3e-10)
                    else:
                        for name in reference.files:
                            np.testing.assert_allclose(actual[name], reference[name], rtol=3e-10, atol=3e-10)
                print(f"[{mode}] {filename} derivative comparison PASS", flush=True)
    collect_measurements(args, modes)
    if failed:
        raise SystemExit("Failed workers; see logs: " + ", ".join(failed))


if __name__ == "__main__":
    main()
