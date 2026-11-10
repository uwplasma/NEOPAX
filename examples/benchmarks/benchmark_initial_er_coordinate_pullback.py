"""Isolate the established initial-Er database-coordinate VJP, without a solve.

The default dimensions match the full-time AD benchmark (51 state radii,
7x16x11 scan table, four species/energy nodes, thirteen cotangent rows).
Smooth synthetic coefficients isolate compilation and allocation behavior;
this is not a transport or equilibrium validation. No VMEX/NTX solve runs.
"""
from __future__ import annotations

import argparse
from collections import namedtuple
import csv
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_case(n_radial=51, objective_count=13, *, small=False):
    import jax.numpy as jnp
    import numpy as np

    from NEOPAX._boundary_conditions import BoundaryConditionModel
    from NEOPAX._database import Monoenergetic
    from NEOPAX._species import Species
    from NEOPAX._state import TransportState
    from NEOPAX._transport_flux_models import NTXDatabaseTransportModel

    n_species, n_energy = (2, 2) if small else (4, 4)
    scan_rho = jnp.asarray([0.12247, .25, .375, .5, .625, .75, .875])
    nu = jnp.asarray([3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3,
                      3e-3, 1e-2, 3e-2, 1e-1, .3, 1., 3., 10.])
    er = jnp.asarray([0., 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, .003, .01, .03, .1])
    if small:
        scan_rho, nu, er = jnp.asarray([.1, .3, .5, .7, .9]), nu[::4], er[::3]
    a_b = jnp.asarray(2.0)
    shape = (scan_rho.size, nu.size, er.size)
    values = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape) / np.prod(shape)
    database = Monoenergetic(
        a_b=a_b, rho=scan_rho, nu_log=jnp.log10(nu),
        Er_list=jnp.log10(jnp.maximum(1e-8, er[None, :] / (a_b * scan_rho[:, None]))),
        D11_log=-3.0 + .1 * values, D13=.2 + .01 * values, D33=.3 + .02 * values,
    )
    faces = jnp.linspace(0., 1., n_radial + 1)
    rho = (faces[:-1] + faces[1:]) / 2
    mesh = namedtuple("CoordinateProbeMesh", "a_b r_grid r_grid_half dr full_grid_indices")(
        a_b, a_b * rho, a_b * faces, a_b / n_radial, jnp.arange(n_radial),
    )
    species = Species(
        number_species=n_species, species_indices=jnp.arange(n_species),
        mass_mp=jnp.asarray([.000544617, 2., 3., 4.])[:n_species],
        charge_qp=jnp.asarray([-1., 1., 1., 2.])[:n_species],
        names=("e", "D", "T", "He")[:n_species],
    )
    x, weights = map(jnp.asarray, np.polynomial.laguerre.laggauss(n_energy))
    energy = namedtuple("CoordinateProbeEnergy", (
        "xWeights L11_weight L12_weight L22_weight L13_weight L23_weight L33_weight v_norm"
    ))(weights, x**2, x**3, x**4, x**1.5, x**2.5, x, jnp.sqrt(x))
    factors = jnp.asarray([1., .5, .5, 1e-5])[:n_species]
    density = factors[:, None] * (2.7 - 2.3 * rho**4)[None, :]
    temperature = jnp.broadcast_to(12.5 - 11.8 * rho**2, density.shape)
    state = TransportState(density, density * temperature, 20.0 * (0.55 - rho))
    bc_density = BoundaryConditionModel(
        dr=mesh.dr, left_type="neumann", right_type="dirichlet",
        left_gradient=jnp.zeros(n_species), right_value=density[:, -1] * .9,
    )
    bc_temperature = BoundaryConditionModel(
        dr=mesh.dr, left_type="neumann", right_type="robin",
        left_gradient=jnp.zeros(n_species), right_value=jnp.ones(n_species),
        right_decay_length=jnp.full((n_species,), .05),
    )
    model = NTXDatabaseTransportModel(
        species, energy, mesh, database, bc_density=bc_density, bc_temperature=bc_temperature,
    )
    bars = jnp.sin(.7 + jnp.arange(objective_count * n_species * n_radial).reshape(
        objective_count, n_species, n_radial))
    return model, state, bars


def memory_stats():
    import jax
    stats = jax.devices()[0].memory_stats() or {}
    return {name: stats.get(name, "") for name in (
        "bytes_in_use", "peak_bytes_in_use", "bytes_limit", "pool_bytes", "peak_pool_bytes",
    )}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--small", action="store_true", help="Use a four-radius, two-species unit-size case.")
    parser.add_argument("--n-radial", type=int, default=51)
    parser.add_argument("--objectives", type=int, default=13)
    parser.add_argument("--compare-to", type=Path, help="Reference NPZ produced with identical probe arguments.")
    args = parser.parse_args(argv)
    if args.n_radial < 2 or args.objectives < 1:
        parser.error("n-radial must be >=2 and objectives must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import jax
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    if jax.devices()[0].platform != "gpu":
        parser.error("This GPU memory probe requires a GPU default device; no CPU calculation was started")
    for package in ("jax", "jaxlib", "interpax", "vmex", "ntx"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed as a distribution"
        print(f"[root-coordinate-probe] {package}={version}", flush=True)
    print("[root-coordinate-probe] device:", jax.devices()[0], flush=True)
    environment = {key: os.environ.get(key, "<unset>") for key in (
        "JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS",
    )}
    print("[root-coordinate-probe] environment:", environment, flush=True)
    try:
        print(subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
            "--format=csv"], text=True), flush=True)
    except (OSError, subprocess.CalledProcessError):
        pass
    model, state, bars = build_case(4 if args.small else args.n_radial, args.objectives, small=args.small)
    jax.block_until_ready((state, bars, model.database))
    before = memory_stats()
    print(f"[root-coordinate-probe] starting established coordinate VJP; "
          f"state_shape={state.density.shape}; cotangent_shape={bars.shape}; {before}", flush=True)
    # This intentionally calls the same unwrapped production method as the
    # failing benchmark. An outer jit would change its closed-over constants.
    start = time.perf_counter()
    result = model._pullback_local_particle_flux_database_coordinates(state, bars, model.database)
    jax.block_until_ready(result)
    first_elapsed = time.perf_counter() - start
    outputs = {"a_b": np.asarray(result.a_b), "Er_list": np.asarray(result.Er_list)}
    if not all(np.isfinite(value).all() for value in outputs.values()):
        raise FloatingPointError("Nonfinite established coordinate VJP")
    np.savez(args.out_dir / "coordinate_bars.npz", **outputs)
    after = memory_stats()
    with (args.out_dir / "measurements.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("phase", "elapsed_s", *before))
        writer.writerow(("before_coordinate_vjp", "", *before.values()))
        writer.writerow(("after_coordinate_vjp", first_elapsed, *after.values()))
    print(f"[root-coordinate-probe] finite established VJP; first_call_s={first_elapsed:.3f}; {after}", flush=True)
    if args.compare_to is not None:
        with np.load(args.compare_to) as reference:
            for name, value in outputs.items():
                if value.shape != reference[name].shape:
                    raise ValueError(f"Mismatched {name} shapes; use identical probe arguments")
                np.testing.assert_allclose(value, reference[name], rtol=3e-10, atol=3e-10)
                scaled_error = np.max(np.abs(value - reference[name]) / np.maximum(1., np.abs(reference[name])))
                print(f"[root-coordinate-probe] {name}: max_abs_difference="
                      f"{np.max(np.abs(value - reference[name])):.6e}; "
                      f"max_scaled_difference={scaled_error:.6e}; "
                      f"bitwise_equal={np.array_equal(value, reference[name])}", flush=True)
        print("[root-coordinate-probe] PASS: reference derivatives agree", flush=True)
    print("[root-coordinate-probe] PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
