from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "examples" / "benchmarks" / "Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_benchmark.toml"
DEFAULT_VMEC_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_newNT_opt_hires"

import NEOPAX
from NEOPAX._orchestrator import run_transport
from NEOPAX._geometry_autodiff import (
    build_geometry_autodiff_context,
    build_runtime_context_for_geometry_param,
    central_fd_single_param,
    five_point_fd_single_param,
    geometry_observables_from_single_param,
    rel_error,
)
from benchmark_transport_autodiff_lagged_ntx import _objective_vector, _prepare_benchmark_config


def _parse_surface_s(text: str) -> tuple[float, ...]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one Boozer surface must be provided.")
    return tuple(values)


def _fd_step(base_value: float, *, fd_rel_step: float, fd_abs_step: float) -> float:
    return max(abs(float(base_value)) * float(fd_rel_step), float(fd_abs_step))


def _print_header(args, context, h: float) -> None:
    print(
        "[geometry-autodiff] "
        f"mode={args.mode} family={context.param_family} "
        f"m={context.param_m} n={context.param_n} "
        f"input={context.input_path} fd_step={h:.6e} "
        f"surfaces={','.join(f'{value:.3f}' for value in context.surface_s)}"
    )
    print(
        "[geometry-autodiff] "
        f"vmec max_iter={args.vmec_max_iter} step_size={args.vmec_step_size:.6e} "
        f"jacobian_penalty={args.vmec_jacobian_penalty:.6e} mboz={args.mboz} nboz={args.nboz}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare AD and FD for vmec_jax -> booz_xform_jax -> NEOPAX geometry/transport metrics."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="NEOPAX transport config used for transport-metric comparisons.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="transport_metrics",
        choices=("transport_metrics", "geometry_observables"),
        help="Compare the standard NEOPAX transport objectives or raw geometry observables.",
    )
    parser.add_argument(
        "--vmec-input",
        type=str,
        default=str(DEFAULT_VMEC_INPUT),
        help="VMEC input file used to seed the differentiable geometry path.",
    )
    parser.add_argument(
        "--param-family",
        type=str,
        default="RBC",
        choices=("RBC", "ZBS"),
        help="Boundary coefficient family to perturb.",
    )
    parser.add_argument("--param-m", type=int, default=1, help="VMEC coefficient m index.")
    parser.add_argument("--param-n", type=int, default=0, help="VMEC coefficient n index.")
    parser.add_argument(
        "--surface-s",
        type=str,
        default="0.25,0.5,0.75",
        help="Comma-separated Boozer surfaces in normalized toroidal flux s.",
    )
    parser.add_argument("--mboz", type=int, default=12, help="Boozer mboz.")
    parser.add_argument("--nboz", type=int, default=12, help="Boozer nboz.")
    parser.add_argument("--n-radial", type=int, default=51, help="Radial grid size for the reconstructed NEOPAX geometry.")
    parser.add_argument("--vmec-max-iter", type=int, default=2, help="Fixed-boundary VMEC iterations.")
    parser.add_argument("--vmec-step-size", type=float, default=5.0e-3, help="Fixed-boundary VMEC step size.")
    parser.add_argument(
        "--vmec-jacobian-penalty",
        type=float,
        default=1.0e3,
        help="Fixed-boundary VMEC jacobian penalty.",
    )
    parser.add_argument("--fd-rel-step", type=float, default=1.0e-6, help="Relative FD step.")
    parser.add_argument("--fd-abs-step", type=float, default=1.0e-8, help="Absolute FD step.")
    parser.add_argument(
        "--with-five-point",
        action="store_true",
        help="Also compute a five-point stencil FD estimate reusing the center evaluations.",
    )
    args = parser.parse_args()

    surface_s = _parse_surface_s(args.surface_s)
    context = build_geometry_autodiff_context(
        args.vmec_input,
        param_family=args.param_family,
        param_m=args.param_m,
        param_n=args.param_n,
        mboz=args.mboz,
        nboz=args.nboz,
        surface_s=surface_s,
    )
    h = _fd_step(context.baseline_coefficient, fd_rel_step=args.fd_rel_step, fd_abs_step=args.fd_abs_step)
    _print_header(args, context, h)

    if args.mode == "geometry_observables":
        func = lambda delta: geometry_observables_from_single_param(
            context,
            delta,
            max_iter=args.vmec_max_iter,
            step_size=args.vmec_step_size,
            jacobian_penalty=args.vmec_jacobian_penalty,
        )
    else:
        config = _prepare_benchmark_config(Path(args.config), device=None)

        def func(delta):
            runtime, state0 = build_runtime_context_for_geometry_param(
                config,
                context,
                delta,
                n_r=args.n_radial,
                max_iter=args.vmec_max_iter,
                step_size=args.vmec_step_size,
                jacobian_penalty=args.vmec_jacobian_penalty,
            )
            result = run_transport(config, runtime, state0)
            return _objective_vector(result["final_state"], runtime)

    print("[geometry-autodiff] progress: running custom/direct JAX derivative")
    _, ad = jax.jvp(func, (jnp.asarray(0.0, dtype=jnp.float64),), (jnp.asarray(1.0, dtype=jnp.float64),))
    print("[geometry-autodiff] progress: running centered finite difference")
    fd_center, minus, plus = central_fd_single_param(func, h)

    fd_five = None
    if args.with_five_point:
        print("[geometry-autodiff] progress: running five-point finite difference")
        fd_five = five_point_fd_single_param(func, h, minus=minus, plus=plus)

    if args.mode == "geometry_observables":
        print("[geometry-autodiff] observable errors:")
        for name, ad_value in ad.items():
            if name in {"surface_indices", "nfp"}:
                continue
            center_err = rel_error(ad_value, fd_center[name])
            line = f"  - {name}: ad_vs_center_rel_err={center_err:.6e}"
            if fd_five is not None:
                five_err = rel_error(ad_value, fd_five[name])
                center_vs_five = rel_error(fd_center[name], fd_five[name])
                line += (
                    f" ad_vs_five_rel_err={five_err:.6e}"
                    f" center_vs_five_rel_err={center_vs_five:.6e}"
                )
            print(line)
    else:
        labels = [
            "softmax_Er",
            "smooth_root_proxy",
            "Er2_volume_average",
            "Er_volume_average",
            "electron_temperature_volume_average_keV",
            "total_pressure_volume_average",
            "alpha_power_volume_average_mw_m3",
        ]
        print("[geometry-autodiff] objective errors:")
        ad_arr = jnp.asarray(ad)
        center_arr = jnp.asarray(fd_center)
        five_arr = None if fd_five is None else jnp.asarray(fd_five)
        for idx, label in enumerate(labels):
            center_err = rel_error(ad_arr[idx], center_arr[idx])
            line = (
                f"  - {label}: ad={float(ad_arr[idx]):.6e} fd_center={float(center_arr[idx]):.6e} "
                f"ad_vs_center_rel_err={center_err:.6e}"
            )
            if five_arr is not None:
                five_err = rel_error(ad_arr[idx], five_arr[idx])
                center_vs_five = rel_error(center_arr[idx], five_arr[idx])
                line += (
                    f" fd_five_point={float(five_arr[idx]):.6e}"
                    f" ad_vs_five_rel_err={five_err:.6e}"
                    f" center_vs_five_rel_err={center_vs_five:.6e}"
                )
            print(line)


if __name__ == "__main__":
    main()
