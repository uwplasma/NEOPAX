"""Compare native and Interpax database queries in fresh GPU processes.

Uses saved QI monoenergetic coefficients, 51 state radii, four species/energy
nodes and 13 objective rows. No equilibrium, NTX or transport solve runs.
Reports first-call coordinate-VJP time/peak memory and warmed query timings.
The optional flux_jacobian probe measures seven stage flux/state Jacobians
without a transport solve, including compiled temporary workspace.
"""

import argparse
import csv
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _flux_jacobian_worker(args, model, state, output):
    import dataclasses
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.flatten_util import ravel_pytree
    from examples.benchmarks.benchmark_initial_er_coordinate_pullback import memory_stats

    flat, unpack = ravel_pytree(state)

    def flux(y, database):
        active_state = unpack(y)
        evaluator = dataclasses.replace(model, database=database).build_local_direct_flux_evaluator(active_state)
        result = jax.vmap(evaluator)(model.geometry.full_grid_indices, active_state.Er)
        return jnp.concatenate((result["Gamma"].reshape(-1) / 1e20,
                                result["Q"].reshape(-1) / 1e6))

    function = jax.jit(jax.vmap(jax.jacfwd(flux, argnums=0), in_axes=(0, None)))
    states = jnp.broadcast_to(flat, (7, flat.size))
    jax.block_until_ready((states, model.database))

    def directional_reference():
        @jax.jit
        def directional(active_states, direction, database):
            return jax.jvp(lambda value: jax.vmap(flux, in_axes=(0, None))(value, database),
                           (active_states,), (direction,))[1]
        directions = jnp.stack(tuple(states * .01 * jnp.sin(
            jnp.arange(states.size).reshape(states.shape) * (.31 + index))
            for index in range(3)))
        derivatives = jnp.stack(tuple(jax.block_until_ready(
            directional(states, direction, model.database)) for direction in directions))
        if not np.isfinite(np.asarray(derivatives)).all():
            raise FloatingPointError("Nonfinite uncontracted directional flux reference")
        np.savez(output / "values.npz", directions=np.asarray(directions),
                 directional_derivatives=np.asarray(derivatives))
        print(f"[{args.worker_backend}] three directional references passed; "
              "the oversized full reference Jacobian was not executed", flush=True)

    if getattr(args, "directional_reference_only", False):
        directional_reference()
        return
    start = time.perf_counter()
    compiled = function.lower(states, model.database).compile()
    compile_s = time.perf_counter() - start
    temporary_bytes = compiled.memory_analysis().temp_size_in_bytes
    print(f"[{args.worker_backend}] flux_jacobian compile_s={compile_s:.6f} "
          f"temporary_bytes={temporary_bytes} memory={memory_stats()}", flush=True)
    limit = getattr(args, "max_temp_bytes", None)
    if limit is not None and temporary_bytes > limit:
        with (output / "workspace.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("temporary_bytes", "limit_bytes", "status"))
            writer.writerow((temporary_bytes, limit, "execution_skipped"))
        if not str(args.worker_backend).endswith("uncontracted"):
            raise MemoryError(f"Optimized workspace {temporary_bytes} exceeds component-probe limit {limit}")
        directional_reference()
        return
    first_s = None
    samples = []
    for index in range(args.repeats + 1):
        active_states = states if index == 0 else states * jnp.asarray(1.000001)
        jax.block_until_ready(active_states)
        start = time.perf_counter()
        result = jax.block_until_ready(compiled(active_states, model.database))
        elapsed = time.perf_counter() - start
        if index == 0:
            first_s = elapsed
            saved = np.asarray(result)
            if not np.isfinite(saved).all():
                raise FloatingPointError("Nonfinite flux/state Jacobian")
            np.savez(output / "values.npz", flux_jacobian=saved)
        else:
            samples.append(elapsed)
        del result
    metrics = dict(backend=args.worker_backend, probe="flux_jacobian", compile_s=compile_s,
                   temporary_bytes=temporary_bytes, first_execution_s=first_s,
                   repeat_median_s=float(np.median(samples)), **memory_stats())
    with (output / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metrics)
        writer.writeheader()
        writer.writerow(metrics)
    print(f"[{args.worker_backend}] {metrics}", flush=True)


def _worker(args):
    import dataclasses
    import functools
    import inspect
    import os

    import h5py
    import interpax
    import jax
    import jax.numpy as jnp
    import numpy as np

    from examples.benchmarks.benchmark_initial_er_coordinate_pullback import build_case, memory_stats
    from NEOPAX import _interpolators as interpolation
    from NEOPAX._database import Monoenergetic

    jax.config.update("jax_enable_x64", True)
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("This memory benchmark requires a GPU; no CPU calculation was started")
    model, state, bars = build_case()
    # Apply the same stored NTX unit conversions as Monoenergetic.read_ntx,
    # vectorized here so loading does not compile per-element GPU updates.
    with h5py.File(args.database, "r") as stream:
        rho, nu, er, drds = (np.array(stream[key]) for key in ("rho", "nu_v", "Er", "drds"))
        d11, d13, d33 = (np.array(stream[key]) for key in ("D11", "D13", "D33"))
    database = Monoenergetic(
        a_b=model.database.a_b, rho=jnp.asarray(rho), nu_log=jnp.log10(nu),
        Er_list=jnp.log10(jnp.maximum(1e-8, np.abs(er[0])[None, :] / (model.database.a_b * rho[:, None]))),
        D11_log=jnp.log10(jnp.maximum(1e-20, d11 * drds[:, None, None] ** 2)),
        D13=jnp.asarray(d13 * drds[:, None, None]), D33=jnp.asarray(d33 * nu[None, :, None]),
    )
    model = dataclasses.replace(model, database=database)
    if args.worker_backend in {"interpax", "native_uncontracted"}:
        from NEOPAX._monoenergetic_interpolators import MONOENERGETIC_INTERPOLATION_KERNELS
        MONOENERGETIC_INTERPOLATION_KERNELS["generic"] = jax.jit(
            inspect.unwrap(MONOENERGETIC_INTERPOLATION_KERNELS["generic"]))
    if args.worker_backend == "native_uncontracted":
        from NEOPAX._native_interpolation import _bicubic_channels
        interpolation.bicubic_channels = _bicubic_channels
    if args.worker_backend == "interpax":
        from NEOPAX._transport_flux_models import NTXDatabaseTransportModel

        @jax.jit
        def reference_surface(index, xq, yq, db):
            return tuple(interpax.Interpolator2D(db.nu_log, db.Er_list[index], table[index], extrap=True)(xq, yq)
                         for table in (db.D11_log, db.D13, db.D33))
        # Reference-only substitution before any query compiles. Both workers
        # retain the same radial selection, physical state and AD boundary.
        interpolation.interpolator_nu_Er_general = reference_surface
        @functools.partial(jax.jit, static_argnames=("collisionality_model",))
        def reference_coordinate(species, energy, geometry, state, bars, db,
                                 bc_density, bc_temperature, floor, *, collisionality_model):
            reference_model = NTXDatabaseTransportModel(
                species, energy, geometry, db, collisionality_model=collisionality_model,
                bc_density=bc_density, bc_temperature=bc_temperature, density_floor=floor)
            return reference_model._local_particle_flux_database_coordinate_bars(state, bars, db)

        def reference_pullback(self, state, bars, db):
            return reference_coordinate(self.species, self.energy_grid, self.geometry, state, bars, db,
                                        self.bc_density, self.bc_temperature, self.density_floor,
                                        collisionality_model=self.collisionality_model)

        # Keep an independent generic VJP reference; it must not share the
        # native sparse coordinate transpose being checked by this benchmark.
        NTXDatabaseTransportModel._pullback_local_particle_flux_database_coordinates = reference_pullback

    backend = args.worker_backend
    output = args.out_dir / backend
    output.mkdir(parents=True, exist_ok=True)
    print(f"[{backend}] JAX={jax.__version__} Interpax={interpax.__version__} device={jax.devices()[0]} "
          f"state={state.density.shape} database={database.D11_log.shape}", flush=True)
    for name in ("XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_ENABLE_COMPILATION_CACHE"):
        print(f"[{backend}] {name}={os.environ.get(name, '<unset>')}", flush=True)
    if getattr(args, "probe", "coordinate") == "flux_jacobian":
        _flux_jacobian_worker(args, model, state, output)
        return
    jax.block_until_ready((state, bars, database))
    coordinate_times = []
    for _ in range(getattr(args, "coordinate_repeats", 1)):
        start = time.perf_counter()
        result = model._pullback_local_particle_flux_database_coordinates(state, bars, database)
        jax.block_until_ready(result)
        coordinate_times.append(time.perf_counter() - start)
    elapsed, memory = coordinate_times[0], memory_stats()
    print(f"[{backend}] coordinate_vjp_first_s={elapsed:.6f} memory={memory}", flush=True)
    if len(coordinate_times) > 1:
        print(f"[{backend}] coordinate_vjp_repeat_median_s={np.median(coordinate_times[1:]):.6f}", flush=True)

    surface = min(3, database.rho.size - 1)
    xq = jnp.linspace(database.nu_log[2], database.nu_log[-3], 51)
    yq = jnp.linspace(database.Er_list[surface, 1], database.Er_list[surface, -2], 51)

    def query(x, y):
        return jnp.stack(interpolation.interpolator_nu_Er_general(surface, x, y, database))

    forward = jax.jit(jax.vmap(query))
    jacobian = jax.jit(jax.vmap(jax.jacrev(query, argnums=(0, 1))))
    metrics = dict(backend=backend, coordinate_first_s=elapsed,
                   coordinate_implementation="generic_vjp" if backend == "interpax" else "native_sparse_stages",
                   coordinate_peak_bytes=memory["peak_bytes_in_use"],
                   coordinate_live_bytes=memory["bytes_in_use"],
                   coordinate_pool_bytes=memory["pool_bytes"],
                   coordinate_peak_pool_bytes=memory["peak_pool_bytes"])
    if len(coordinate_times) > 1:
        metrics["coordinate_repeat_median_s"] = float(np.median(coordinate_times[1:]))
    for label, function in (("forward", forward), ("query_jacobian", jacobian)):
        for _ in range(5):
            jax.block_until_ready(function(xq, yq))
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            jax.block_until_ready(function(xq, yq))
            samples.append(time.perf_counter() - start)
        metrics[label + "_median_s"] = float(np.median(samples))
        metrics[label + "_q25_s"], metrics[label + "_q75_s"] = np.quantile(samples, [.25, .75])
        print(f"[{backend}] {label} warm_median_s={np.median(samples):.9f}", flush=True)
    values = dict(a_b=np.asarray(result.a_b), Er_list=np.asarray(result.Er_list),
                  query_values=np.asarray(forward(xq, yq)), query_jacobian=np.asarray(jacobian(xq, yq)))
    if not all(np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError(f"Nonfinite {backend} result")
    np.savez(output / "values.npz", **values)
    with (output / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metrics)
        writer.writeheader()
        writer.writerow(metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / "examples/inputs/Dij_NTX_QI_Er_Optimized.h5")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/native_database_interpolation")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--coordinate-repeats", type=int, default=3,
                        help="Coordinate pullback calls, including its first compilation.")
    parser.add_argument("--probe", choices=("coordinate", "flux_jacobian"), default="coordinate")
    parser.add_argument("--reference-backend", choices=("interpax", "native_uncontracted"), default="interpax",
                        help="Use native_uncontracted to compare against native interpolation before its query-JVP fix.")
    parser.add_argument("--worker-backend", choices=("native", "interpax", "native_uncontracted"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.coordinate_repeats < 1:
        parser.error("repeat counts must be positive")
    if args.worker_backend:
        _worker(args)
        return
    # Do not initialize JAX in the parent: GPU contexts and allocations must
    # be released between the two measurements.
    for backend in (args.reference_backend, "native"):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--database", str(args.database),
                        "--out-dir", str(args.out_dir), "--repeats", str(args.repeats),
                        "--coordinate-repeats", str(args.coordinate_repeats),
                        "--probe", args.probe,
                        "--worker-backend", backend], check=True)
    import numpy as np
    with np.load(args.out_dir / args.reference_backend / "values.npz") as reference, np.load(args.out_dir / "native/values.npz") as actual:
        for key in reference.files:
            np.testing.assert_allclose(actual[key], reference[key], rtol=3e-10, atol=3e-10)
            scaled = np.max(np.abs(actual[key] - reference[key]) / np.maximum(1., np.abs(reference[key])))
            print(f"[comparison] {key}: max_scaled_difference={scaled:.6e}")
    print(f"PASS: values and derivatives agree. Measurements saved in {args.out_dir}")


if __name__ == "__main__":
    main()
