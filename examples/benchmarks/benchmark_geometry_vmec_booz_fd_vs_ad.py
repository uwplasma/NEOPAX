from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX._geometry_autodiff import (  # noqa: E402
    boundary_param_entries,
    build_geometry_autodiff_context,
    central_fd_single_param,
    exact_forward_scalar_observable_derivatives,
    exact_reverse_scalar_observable_derivatives,
    five_point_fd_single_param,
    rel_error,
    vmec_booz_scalar_observables_from_param_vector,
    vmec_booz_scalar_observables_from_single_param,
    vmec_scalar_observables_from_param_vector,
    vmec_scalar_observables_from_single_param,
)

DEFAULT_VMEC_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_newNT_opt_hires"


def _parse_surface_s(text: str) -> tuple[float, ...]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one Boozer surface must be provided.")
    return tuple(values)


def _fd_step(base_value: float, *, fd_rel_step: float, fd_abs_step: float) -> float:
    return max(abs(float(base_value)) * float(fd_rel_step), float(fd_abs_step))


def _parse_param_specs(text: str | None, *, default_family: str, default_m: int, default_n: int) -> tuple[tuple[str, int, int], ...]:
    if text is None or not str(text).strip():
        return ((str(default_family).strip().upper(), int(default_m), int(default_n)),)
    specs = []
    for raw_spec in str(text).split(","):
        spec = raw_spec.strip()
        if not spec:
            continue
        parts = [part.strip() for part in spec.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Parameter spec {spec!r} must have form FAMILY:m:n, e.g. RBC:1:0.")
        family = parts[0].upper()
        if family not in {"RBC", "ZBS"}:
            raise ValueError(f"Parameter spec {spec!r} has unsupported family {family!r}; use RBC or ZBS.")
        specs.append((family, int(parts[1]), int(parts[2])))
    if not specs:
        raise ValueError("--param-specs did not contain any valid specs.")
    return tuple(specs)


def _resolved_max_iter(context, user_value: int | None) -> int:
    return int(context.vmec_default_max_iter if user_value is None else user_value)


def _resolved_step_size(context, user_value: float | None) -> float:
    return float(context.vmec_default_step_size if user_value is None else user_value)


def _print_header(args, context, h: float, *, resolved_max_iter: int, resolved_step_size: float) -> None:
    print(
        "[geometry-fd-ad] "
        f"mode={args.mode} family={context.param_family} "
        f"m={context.param_m} n={context.param_n} "
        f"input={context.input_path} fd_step={h:.6e}",
        flush=True,
    )
    print(
        "[geometry-fd-ad] "
        f"jax_backend={jax.default_backend()} devices={jax.devices()}",
        flush=True,
    )
    print(
        "[geometry-fd-ad] "
        f"vmec forward_lane=run_fixed_boundary ad_backend={args.ad_backend} max_iter={resolved_max_iter} "
        f"step_size={resolved_step_size:.6e} exact_solver_device={args.exact_solver_device} "
        f"mboz={args.mboz} nboz={args.nboz} "
        f"surfaces={','.join(f'{value:.3f}' for value in context.surface_s)}",
        flush=True,
    )


def _observable_function(args, context, *, lane: str, resolved_max_iter: int, resolved_step_size: float):
    if args.mode == "vmec_scalar_observables":
        return lambda delta: vmec_scalar_observables_from_single_param(  # noqa: E731
            context,
            delta,
            lane=lane,
            max_iter=resolved_max_iter,
            step_size=resolved_step_size,
        )
    return lambda delta: vmec_booz_scalar_observables_from_single_param(  # noqa: E731
        context,
        delta,
        lane=lane,
        max_iter=resolved_max_iter,
        step_size=resolved_step_size,
    )


def _implicit_forward_jvp(func):
    _values, tangents = jax.jvp(
        func,
        (jnp.asarray(0.0, dtype=jnp.float64),),
        (jnp.asarray(1.0, dtype=jnp.float64),),
    )
    return tangents


def _implicit_reverse_gradients(func):
    baseline = func(jnp.asarray(0.0, dtype=jnp.float64))
    gradients = {}
    for name in baseline:
        gradients[name] = jax.grad(
            lambda delta, observable_name=name: jnp.asarray(func(delta)[observable_name], dtype=jnp.float64).reshape(())
        )(jnp.asarray(0.0, dtype=jnp.float64))
    return gradients


def _observable_vector_function(args, context, param_specs, *, lane: str, resolved_max_iter: int, resolved_step_size: float):
    if args.mode == "vmec_scalar_observables":
        return lambda deltas: vmec_scalar_observables_from_param_vector(  # noqa: E731
            context,
            deltas,
            param_specs,
            lane=lane,
            max_iter=resolved_max_iter,
            step_size=resolved_step_size,
        )
    return lambda deltas: vmec_booz_scalar_observables_from_param_vector(  # noqa: E731
        context,
        deltas,
        param_specs,
        lane=lane,
        max_iter=resolved_max_iter,
        step_size=resolved_step_size,
    )


def _implicit_reverse_jacobian(func, n_params: int):
    baseline = func(jnp.zeros((n_params,), dtype=jnp.float64))
    names = tuple(baseline.keys())

    def stacked_observables(deltas):
        values = func(deltas)
        return jnp.stack([jnp.asarray(values[name], dtype=jnp.float64).reshape(()) for name in names])

    jacobian_matrix = jax.jacrev(stacked_observables)(jnp.zeros((n_params,), dtype=jnp.float64))
    return {name: jacobian_matrix[i, :] for i, name in enumerate(names)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare VMEC/Boozer scalar observable derivatives between the NEOPAX AD lane and forward-lane finite differences."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="vmec_booz_scalar_observables",
        choices=("vmec_scalar_observables", "vmec_booz_scalar_observables"),
        help="Run the VMEC-only scalar gate or the VMEC -> Boozer scalar gate.",
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
        "--param-specs",
        type=str,
        default=None,
        help=(
            "Comma-separated list of geometry parameters FAMILY:m:n. "
            "Example: RBC:1:0,RBC:2:0,ZBS:1:0,ZBS:2:0. "
            "If omitted, --param-family/--param-m/--param-n are used."
        ),
    )
    parser.add_argument(
        "--surface-s",
        type=str,
        default="0.25,0.5,0.75",
        help="Comma-separated Boozer surfaces in normalized toroidal flux s.",
    )
    parser.add_argument("--mboz", type=int, default=12, help="Boozer mboz.")
    parser.add_argument("--nboz", type=int, default=12, help="Boozer nboz.")
    parser.add_argument("--vmec-max-iter", type=int, default=None, help="Override VMEC iterations. Default: from input file.")
    parser.add_argument(
        "--vmec-step-size",
        type=float,
        default=None,
        help="Override VMEC step size. Default: from input file.",
    )
    parser.add_argument("--fd-rel-step", type=float, default=1.0e-6, help="Relative FD step.")
    parser.add_argument("--fd-abs-step", type=float, default=1.0e-8, help="Absolute FD step.")
    parser.add_argument(
        "--fd-lane",
        type=str,
        default="forward",
        choices=("forward", "ad"),
        help=(
            "Lane used for centered finite differences. 'forward' preserves the "
            "historical solve_multigrid/run_fixed_boundary comparison; 'ad' "
            "finite-differences the same implicit primal map used by reverse AD."
        ),
    )
    parser.add_argument(
        "--ad-backend",
        type=str,
        default="exact_optimizer",
        choices=("exact_optimizer", "implicit"),
        help=(
            "'exact_optimizer' keeps the legacy accepted-point matrix-free benchmark. "
            "'implicit' compares FD against reverse AD through the current vmec_jax implicit custom-VJP lane."
        ),
    )
    parser.add_argument(
        "--skip-reverse-check",
        action="store_true",
        help="Skip reverse-mode observable derivative recovery.",
    )
    parser.add_argument(
        "--exact-solver-device",
        type=str,
        default="cpu",
        choices=("cpu", "gpu", "auto", "default"),
        help="Device used by the exact accepted-point forward/reverse callbacks. Default: cpu.",
    )
    parser.add_argument(
        "--with-five-point",
        action="store_true",
        help="Also compute a five-point stencil FD estimate.",
    )
    args = parser.parse_args()

    param_specs = _parse_param_specs(
        args.param_specs,
        default_family=args.param_family,
        default_m=args.param_m,
        default_n=args.param_n,
    )
    surface_s = _parse_surface_s(args.surface_s)
    if len(param_specs) > 1 and args.ad_backend == "implicit":
        _run_multi_parameter_implicit(args, param_specs=param_specs, surface_s=surface_s)
        return
    for param_index, (param_family, param_m, param_n) in enumerate(param_specs, start=1):
        if len(param_specs) > 1:
            print(
                f"[geometry-fd-ad] parameter {param_index}/{len(param_specs)}: "
                f"{param_family}:{param_m}:{param_n}",
                flush=True,
            )
        _run_single_parameter(args, param_family=param_family, param_m=param_m, param_n=param_n, surface_s=surface_s)


def _run_multi_parameter_implicit(args, *, param_specs: tuple[tuple[str, int, int], ...], surface_s: tuple[float, ...]) -> None:
    context = build_geometry_autodiff_context(
        args.vmec_input,
        param_family=param_specs[0][0],
        param_m=param_specs[0][1],
        param_n=param_specs[0][2],
        mboz=args.mboz,
        nboz=args.nboz,
        surface_s=surface_s,
    )
    entries = boundary_param_entries(context, param_specs)
    h_values = tuple(
        _fd_step(entry["baseline_coefficient"], fd_rel_step=args.fd_rel_step, fd_abs_step=args.fd_abs_step)
        for entry in entries
    )
    resolved_max_iter = _resolved_max_iter(context, args.vmec_max_iter)
    resolved_step_size = _resolved_step_size(context, args.vmec_step_size)
    print(
        "[geometry-fd-ad] "
        f"mode={args.mode} multi_parameter_count={len(param_specs)} "
        f"input={context.input_path}",
        flush=True,
    )
    print(
        "[geometry-fd-ad] "
        f"jax_backend={jax.default_backend()} devices={jax.devices()}",
        flush=True,
    )
    print(
        "[geometry-fd-ad] "
        f"vmec forward_lane=run_fixed_boundary fd_lane={args.fd_lane} "
        f"ad_backend={args.ad_backend} max_iter={resolved_max_iter} "
        f"step_size={resolved_step_size:.6e} mboz={args.mboz} nboz={args.nboz} "
        f"surfaces={','.join(f'{value:.3f}' for value in context.surface_s)}",
        flush=True,
    )

    print("[geometry-fd-ad] progress: running one implicit-lane reverse Jacobian for all geometry parameters", flush=True)
    ad_func = _observable_vector_function(
        args,
        context,
        param_specs,
        lane="ad",
        resolved_max_iter=resolved_max_iter,
        resolved_step_size=resolved_step_size,
    )
    reverse_jacobian = None if args.skip_reverse_check else _implicit_reverse_jacobian(ad_func, len(param_specs))

    print(f"[geometry-fd-ad] progress: running {args.fd_lane}-lane centered finite differences per parameter", flush=True)
    fd_by_param = []
    for i, (param_family, param_m, param_n) in enumerate(param_specs):
        single_context = build_geometry_autodiff_context(
            args.vmec_input,
            param_family=param_family,
            param_m=param_m,
            param_n=param_n,
            mboz=args.mboz,
            nboz=args.nboz,
            surface_s=surface_s,
        )
        fd_func = _observable_function(
            args,
            single_context,
            lane=args.fd_lane,
            resolved_max_iter=resolved_max_iter,
            resolved_step_size=resolved_step_size,
        )
        fd_center, _minus, _plus = central_fd_single_param(fd_func, h_values[i])
        fd_by_param.append(fd_center)

    names = reverse_jacobian.keys() if reverse_jacobian is not None else fd_by_param[0].keys()
    for i, (param_family, param_m, param_n) in enumerate(param_specs):
        print(
            f"[geometry-fd-ad] parameter {i + 1}/{len(param_specs)}: "
            f"{param_family}:{param_m}:{param_n} fd_step={h_values[i]:.6e}",
            flush=True,
        )
        for name in names:
            fd_value = fd_by_param[i][name]
            line = f"  - {name}: fd_center={float(jnp.asarray(fd_value)):.6e}"
            if reverse_jacobian is not None:
                reverse_value = reverse_jacobian[name][i]
                reverse_fd_err = rel_error(reverse_value, fd_value)
                line += (
                    f" reverse_ad={float(jnp.asarray(reverse_value)):.6e}"
                    f" reverse_vs_fd_rel_err={reverse_fd_err:.6e}"
                )
            print(line, flush=True)


def _run_single_parameter(args, *, param_family: str, param_m: int, param_n: int, surface_s: tuple[float, ...]) -> None:
    context = build_geometry_autodiff_context(
        args.vmec_input,
        param_family=param_family,
        param_m=param_m,
        param_n=param_n,
        mboz=args.mboz,
        nboz=args.nboz,
        surface_s=surface_s,
    )
    h = _fd_step(context.baseline_coefficient, fd_rel_step=args.fd_rel_step, fd_abs_step=args.fd_abs_step)
    resolved_max_iter = _resolved_max_iter(context, args.vmec_max_iter)
    resolved_step_size = _resolved_step_size(context, args.vmec_step_size)
    _print_header(args, context, h, resolved_max_iter=resolved_max_iter, resolved_step_size=resolved_step_size)

    observable_kind = args.mode
    fd_func = _observable_function(
        args,
        context,
        lane=args.fd_lane,
        resolved_max_iter=resolved_max_iter,
        resolved_step_size=resolved_step_size,
    )
    ad_func = _observable_function(
        args,
        context,
        lane="ad",
        resolved_max_iter=resolved_max_iter,
        resolved_step_size=resolved_step_size,
    )

    forward_ad = None
    if args.ad_backend == "exact_optimizer":
        print("[geometry-fd-ad] progress: running exact accepted-point matrix-free forward Jv", flush=True)
        forward_ad = exact_forward_scalar_observable_derivatives(
            context,
            observable_kind=observable_kind,
            max_iter=resolved_max_iter,
            step_size=resolved_step_size,
            solver_device=args.exact_solver_device,
        )
    else:
        print(
            "[geometry-fd-ad] progress: skipping forward JVP because implicit solve is custom_vjp-only",
            flush=True,
        )
    print(f"[geometry-fd-ad] progress: running {args.fd_lane}-lane centered finite difference", flush=True)
    fd_center, minus, plus = central_fd_single_param(fd_func, h)

    fd_five = None
    if args.with_five_point:
        print(f"[geometry-fd-ad] progress: running {args.fd_lane}-lane five-point finite difference", flush=True)
        fd_five = five_point_fd_single_param(fd_func, h, minus=minus, plus=plus)

    reverse_ad = None
    if not args.skip_reverse_check:
        if args.ad_backend == "implicit":
            print("[geometry-fd-ad] progress: running implicit-lane reverse gradients", flush=True)
            reverse_ad = _implicit_reverse_gradients(ad_func)
        else:
            print("[geometry-fd-ad] progress: running exact accepted-point matrix-free reverse J^T w recovery", flush=True)
            reverse_ad = exact_reverse_scalar_observable_derivatives(
                context,
                observable_kind=observable_kind,
                max_iter=resolved_max_iter,
                step_size=resolved_step_size,
                solver_device=args.exact_solver_device,
            )

    print("[geometry-fd-ad] observable comparison:")
    names = reverse_ad.keys() if forward_ad is None and reverse_ad is not None else forward_ad.keys()
    for name in names:
        center_value = fd_center[name]
        line = f"  - {name}: fd_center={float(jnp.asarray(center_value)):.6e}"
        if forward_ad is not None:
            forward_ad_value = forward_ad[name]
            forward_fd_err = rel_error(forward_ad_value, center_value)
            line += (
                f" forward_ad={float(jnp.asarray(forward_ad_value)):.6e}"
                f" forward_vs_fd_rel_err={forward_fd_err:.6e}"
            )
        if fd_five is not None:
            five_value = fd_five[name]
            center_vs_five = rel_error(center_value, five_value)
            line += f" fd_five_point={float(jnp.asarray(five_value)):.6e}"
            if forward_ad is not None:
                five_err = rel_error(forward_ad[name], five_value)
                line += f" forward_vs_five_rel_err={five_err:.6e}"
            line += f" center_vs_five_rel_err={center_vs_five:.6e}"
        if reverse_ad is not None:
            reverse_value = reverse_ad[name]
            reverse_fd_err = rel_error(reverse_value, center_value)
            line += f" reverse_ad={float(jnp.asarray(reverse_value)):.6e}"
            if forward_ad is not None:
                reverse_err = rel_error(reverse_value, forward_ad[name])
                line += f" reverse_vs_forward_rel_err={reverse_err:.6e}"
            line += f" reverse_vs_fd_rel_err={reverse_fd_err:.6e}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
