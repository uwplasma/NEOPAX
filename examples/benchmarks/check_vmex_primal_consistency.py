"""Compare VMEX's forward equilibrium with the implicit AD equilibrium.

Runs two equilibrium solves and optional Boozer diagnostics, without building
an NTX scan, finding ambipolar roots or evolving transport. Intended for the
remote GPU machine. Results are CSV and NPZ, including both state arrays.
"""
import argparse
import csv
import dataclasses
from pathlib import Path
import sys
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compare_states(forward, implicit):
    """Compare like-for-like arrays; reject incomplete or invalid artifacts."""
    if forward.keys() != implicit.keys():
        raise ValueError("Equilibrium state fields differ")
    rows = []
    for name, a in forward.items():
        a, b = np.asarray(a), np.asarray(implicit[name])
        if a.shape != b.shape:
            raise ValueError(f"Equilibrium state shape differs: {name}")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise FloatingPointError(f"Nonfinite equilibrium field: {name}")
        delta = b - a
        rows.append(dict(field=name, max_abs_difference=float(np.max(np.abs(delta))),
                         relative_l2_difference=float(np.linalg.norm(delta.ravel()) /
                                                      max(np.linalg.norm(a.ravel()), 1e-30))))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "examples/benchmarks/"
                        "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/vmex_primal_consistency")
    parser.add_argument("--include-boozer", action="store_true",
                        help="Also compare QI/max-J and the Boozer-derived observables.")
    args = parser.parse_args()
    with args.config.open("rb") as stream:
        geometry_cfg = tomllib.load(stream)["geometry"]

    import jax
    import jax.numpy as jnp
    import vmex
    from NEOPAX._geometry_autodiff import (
        _solve_state_for_single_param, _vmec_booz_scalar_observables_from_state,
        _vmec_scalar_observables_from_state, build_geometry_autodiff_context,
        geometry_raw_block_solve_from_param_vector,
    )

    jax.config.update("jax_enable_x64", True)
    gpu = jax.devices("gpu")[0]  # Fail rather than silently run on a local CPU.
    print(f"[vmex-primal] JAX={jax.__version__} VMEX={vmex.__file__} device={gpu}", flush=True)
    input_path = Path(geometry_cfg["vmec_input_file"])
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    family = str(geometry_cfg.get("vmec_param_family", "RBC"))
    m, n = int(geometry_cfg.get("vmec_param_m", 1)), int(geometry_cfg.get("vmec_param_n", 0))
    delta = float(geometry_cfg.get("vmec_param_delta", 0.))
    print(f"[vmex-primal] configured_runtime_lane={geometry_cfg.get('vmec_lane', 'forward')} "
          "geometry_pullback_lane=ad", flush=True)
    with jax.default_device(gpu):
        context = build_geometry_autodiff_context(
            input_path, param_family=family, param_m=m, param_n=n,
            mboz=geometry_cfg.get("mboz", geometry_cfg.get("vmec_mboz")),
            nboz=geometry_cfg.get("nboz", geometry_cfg.get("vmec_nboz")))
        print("[vmex-primal] solving forward equilibrium", flush=True)
        forward = _solve_state_for_single_param(
            context, jnp.asarray(delta), lane="forward", max_iter=geometry_cfg.get("vmec_max_iter"),
            step_size=geometry_cfg.get("vmec_step_size"))
        jax.block_until_ready(forward)
        print("[vmex-primal] solving implicit equilibrium with its AD auxiliary data", flush=True)
        raw = geometry_raw_block_solve_from_param_vector(
            context, jnp.asarray([delta]), ((family, m, n),),
            max_iter=geometry_cfg.get("vmec_max_iter"),
            solver_device=str(geometry_cfg.get("vmec_implicit_solver_device", "default")))
        jax.block_until_ready(raw.state)
        arrays = [{field.name: np.asarray(getattr(state, field.name)) for field in dataclasses.fields(state)}
                  for state in (forward, raw.state)]
        rows = compare_states(*arrays)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.out_dir / "state_comparison.csv", rows)
        np.savez(args.out_dir / "states.npz", **{
            f"{label}__{name}": value
            for label, state in zip(("forward", "implicit"), arrays, strict=True)
            for name, value in state.items()})
        for row in rows:
            print(f"[vmex-primal] {row}", flush=True)

        evaluate = (_vmec_booz_scalar_observables_from_state if args.include_boozer
                    else _vmec_scalar_observables_from_state)
        values = [evaluate(context, state) for state in (forward, raw.state)]
        objective_rows = [dict(objective=name, forward=float(values[0][name]),
                               implicit=float(values[1][name]),
                               difference=float(values[1][name] - values[0][name]))
                          for name in values[0]]
        write_csv(args.out_dir / "objective_comparison.csv", objective_rows)
        for row in objective_rows:
            print(f"[vmex-primal] {row}", flush=True)
    print(f"[vmex-primal] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
