from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


NEOPAX_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = NEOPAX_ROOT.parent
for path in (NEOPAX_ROOT, WORKSPACE_ROOT / "vmec_jax"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vmec_jax.core import implicit as im  # noqa: E402
from vmec_jax.core import optimize as opt  # noqa: E402
from vmec_jax.core.input import VmecInput  # noqa: E402


DEFAULT_VMEC_INPUT = NEOPAX_ROOT / "examples" / "inputs" / "input.QI_nfp2_newNT_opt_hires_true"

OBJECTIVE_FUNCS = {
    "aspect_ratio": opt.aspect_ratio,
    "volume": opt.volume,
    "mean_iota": opt.mean_iota,
    "edge_iota": opt.edge_iota,
    "magnetic_well": opt.magnetic_well,
    "mirror_ratio": opt.mirror_ratio,
}


def _parse_param_spec(text: str) -> tuple[str, int, int]:
    parts = [part.strip() for part in str(text).split(":")]
    if len(parts) != 3:
        raise ValueError(f"Parameter spec {text!r} must be FAMILY:m:n, e.g. RBC:1:0.")
    family = parts[0].lower()
    if family not in {"rbc", "rbs", "zbc", "zbs"}:
        raise ValueError(f"Unsupported family {parts[0]!r}; use RBC, RBS, ZBC, or ZBS.")
    return family, int(parts[1]), int(parts[2])


def _param_index(inp: VmecInput, family: str, m: int, n: int) -> tuple[int, int]:
    row = int(n) + int(inp.ntor)
    col = int(m)
    arr = getattr(im.params_from_input(inp), family)
    if row < 0 or row >= arr.shape[0] or col < 0 or col >= arr.shape[1]:
        raise ValueError(
            f"{family.upper()}:{m}:{n} maps to index {(row, col)}, outside shape {arr.shape}."
        )
    return row, col


def _with_param_delta(
    params: im.ImplicitParams,
    family: str,
    row: int,
    col: int,
    delta,
) -> im.ImplicitParams:
    arr = getattr(params, family)
    return dataclasses.replace(params, **{family: arr.at[row, col].add(delta)})


def _param_value(params: im.ImplicitParams, family: str, row: int, col: int) -> float:
    return float(np.asarray(getattr(params, family))[row, col])


def _param_component(params: im.ImplicitParams, family: str, row: int, col: int) -> float:
    return float(jax.device_get(getattr(params, family)[row, col]))


def _param_unit_tangent_like(
    params: im.ImplicitParams,
    family: str,
    row: int,
    col: int,
) -> im.ImplicitParams:
    leaves = {
        field.name: jnp.zeros_like(getattr(params, field.name))
        for field in dataclasses.fields(params)
    }
    leaves[family] = leaves[family].at[row, col].set(1.0)
    return dataclasses.replace(params, **leaves)


def _fd_step(base_value: float, *, rel_step: float, abs_step: float) -> float:
    return max(abs(float(base_value)) * float(rel_step), float(abs_step))


def _relative_error(value: float, reference: float | None) -> float:
    if reference is None:
        return float("nan")
    return abs(value - reference) / max(abs(reference), 1.0e-300)


def _tree_l2_and_max(tree) -> tuple[float, float]:
    leaves = [jnp.ravel(jnp.asarray(x)) for x in jax.tree.leaves(tree)]
    if not leaves:
        return 0.0, 0.0
    vec = jnp.concatenate(leaves)
    return float(jax.device_get(jnp.linalg.norm(vec))), float(jax.device_get(jnp.max(jnp.abs(vec))))


def _objective_value(objective_name: str, state: im.SpectralState, params: im.ImplicitParams, cfg: im.ImplicitConfig):
    rt = im.runtime_from_params(params, cfg)
    return jnp.asarray(OBJECTIVE_FUNCS[objective_name](state, rt), dtype=jnp.float64).reshape(())


def _manual_implicit_forward_state_tangent(
    *,
    params: im.ImplicitParams,
    param_tangent: im.ImplicitParams,
    cfg: im.ImplicitConfig,
    x_star: im.SpectralState,
    dof_mask: im.SpectralState,
    formulation: str,
):
    frozen = jax.lax.stop_gradient(x_star)
    edge_mask = im._edge_mask(cfg)
    P = im._dof_projector(cfg, dof_mask)
    F = im.residual_fn(cfg, frozen, dof_mask, formulation=formulation)
    z_star = P(x_star)

    def F_z(z):
        return F(z, params)

    def F_p(prm):
        return F(z_star, prm)

    rhs = jax.tree.map(
        jnp.negative,
        jax.jvp(F_p, (params,), (param_tangent,))[1],
    )
    dz, _ = im._adjoint_solve(lambda v: jax.jvp(F_z, (z_star,), (v,))[1], rhs, cfg)

    def assemble_from_z_params(z, prm):
        return im._assemble(
            z,
            im.runtime_from_params(prm, cfg),
            frozen,
            P,
            edge_mask,
        )

    state_tangent = jax.jvp(
        assemble_from_z_params,
        (z_star, params),
        (dz, param_tangent),
    )[1]
    return dz, state_tangent


def _manual_implicit_pullback(
    *,
    params: im.ImplicitParams,
    cfg: im.ImplicitConfig,
    x_star: im.SpectralState,
    dof_mask: im.SpectralState,
    state_bar: im.SpectralState,
    direct_param_bar: im.ImplicitParams,
    rhs_mode: str,
    formulation: str,
) -> im.ImplicitParams:
    frozen = jax.lax.stop_gradient(x_star)
    edge_mask = im._edge_mask(cfg)
    P = im._dof_projector(cfg, dof_mask)
    F = im.residual_fn(cfg, frozen, dof_mask, formulation=formulation)
    z_star = P(x_star)

    _, vjp_z = jax.vjp(lambda z: F(z, params), z_star)
    if rhs_mode == "current":
        rhs = P(state_bar)
    elif rhs_mode == "assemble_vjp":
        _, assemble_vjp_z = jax.vjp(
            lambda z: im._assemble(
                z,
                im.runtime_from_params(params, cfg),
                frozen,
                P,
                edge_mask,
            ),
            z_star,
        )
        rhs = assemble_vjp_z(state_bar)[0]
    else:
        raise ValueError(f"Unknown rhs_mode={rhs_mode!r}.")

    lam, _ = im._adjoint_solve(lambda v: vjp_z(v)[0], rhs, cfg)
    _, vjp_p = jax.vjp(lambda prm: F(z_star, prm), params)
    implicit_param_bar = vjp_p(jax.tree.map(jnp.negative, lam))[0]

    _, assemble_vjp_p = jax.vjp(
        lambda prm: im._assemble(
            z_star,
            im.runtime_from_params(prm, cfg),
            frozen,
            P,
            edge_mask,
        ),
        params,
    )
    assemble_param_bar = assemble_vjp_p(state_bar)[0]

    return jax.tree.map(
        lambda a, b, c: a + b + c,
        implicit_param_bar,
        assemble_param_bar,
        direct_param_bar,
    )


def _manual_implicit_forward_jvp(
    *,
    objective_name: str,
    params: im.ImplicitParams,
    param_tangent: im.ImplicitParams,
    cfg: im.ImplicitConfig,
    x_star: im.SpectralState,
    dof_mask: im.SpectralState,
    formulation: str,
):
    _dz, state_tangent = _manual_implicit_forward_state_tangent(
        params=params,
        param_tangent=param_tangent,
        cfg=cfg,
        x_star=x_star,
        dof_mask=dof_mask,
        formulation=formulation,
    )
    value, objective_tangent = jax.jvp(
        lambda state, prm: _objective_value(objective_name, state, prm, cfg),
        (x_star, params),
        (state_tangent, param_tangent),
    )
    return value, objective_tangent


def _manual_residual_consistency_check(
    *,
    params: im.ImplicitParams,
    param_tangent: im.ImplicitParams,
    cfg: im.ImplicitConfig,
    x_star: im.SpectralState,
    dof_mask: im.SpectralState,
    formulation: str,
    steps: tuple[float, ...],
):
    frozen = jax.lax.stop_gradient(x_star)
    P = im._dof_projector(cfg, dof_mask)
    F = im.residual_fn(cfg, frozen, dof_mask, formulation=formulation)
    z_star = P(x_star)
    dz, _state_tangent = _manual_implicit_forward_state_tangent(
        params=params,
        param_tangent=param_tangent,
        cfg=cfg,
        x_star=x_star,
        dof_mask=dof_mask,
        formulation=formulation,
    )
    base_l2, base_max = _tree_l2_and_max(F(z_star, params))
    rows = [("base", 0.0, base_l2, base_max, float("nan"), float("nan"))]
    for step in steps:
        step_arr = jnp.asarray(step, dtype=jnp.float64)
        p_step = jax.tree.map(lambda p, t: p + step_arr * t, params, param_tangent)
        z_no_state = z_star
        z_plus = jax.tree.map(lambda z, t: z + step_arr * t, z_star, dz)
        z_minus = jax.tree.map(lambda z, t: z - step_arr * t, z_star, dz)
        no_l2, no_max = _tree_l2_and_max(F(z_no_state, p_step))
        plus_l2, plus_max = _tree_l2_and_max(F(z_plus, p_step))
        minus_l2, minus_max = _tree_l2_and_max(F(z_minus, p_step))
        rows.append(("no_state", step, no_l2, no_max, no_l2 / max(abs(step), 1.0e-300), no_max / max(abs(step), 1.0e-300)))
        rows.append(("+dz", step, plus_l2, plus_max, plus_l2 / max(abs(step) ** 2, 1.0e-300), plus_max / max(abs(step) ** 2, 1.0e-300)))
        rows.append(("-dz", step, minus_l2, minus_max, minus_l2 / max(abs(step) ** 2, 1.0e-300), minus_max / max(abs(step) ** 2, 1.0e-300)))
    return rows


def _manual_frozen_linearized_fd(
    *,
    objective_name: str,
    params: im.ImplicitParams,
    param_tangent: im.ImplicitParams,
    cfg: im.ImplicitConfig,
    x_star: im.SpectralState,
    dof_mask: im.SpectralState,
    formulation: str,
    step: float,
):
    _dz, state_tangent = _manual_implicit_forward_state_tangent(
        params=params,
        param_tangent=param_tangent,
        cfg=cfg,
        x_star=x_star,
        dof_mask=dof_mask,
        formulation=formulation,
    )
    step_arr = jnp.asarray(step, dtype=jnp.float64)
    p_minus = jax.tree.map(lambda p, t: p - step_arr * t, params, param_tangent)
    p_plus = jax.tree.map(lambda p, t: p + step_arr * t, params, param_tangent)
    state_minus = jax.tree.map(lambda x, t: x - step_arr * t, x_star, state_tangent)
    state_plus = jax.tree.map(lambda x, t: x + step_arr * t, x_star, state_tangent)
    minus = _objective_value(objective_name, state_minus, p_minus, cfg)
    plus = _objective_value(objective_name, state_plus, p_plus, cfg)
    fd = (plus - minus) / (2.0 * step_arr)
    _value, jvp = jax.jvp(
        lambda state, prm: _objective_value(objective_name, state, prm, cfg),
        (x_star, params),
        (state_tangent, param_tangent),
    )
    return minus, plus, fd, jvp


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Self-contained VMEC implicit pullback RHS diagnostic. "
            "Compares the current P(gbar) adjoint RHS against an _assemble VJP RHS "
            "without editing vmec_jax."
        )
    )
    parser.add_argument("--vmec-input", default=str(DEFAULT_VMEC_INPUT))
    parser.add_argument("--parameter", default="RBC:1:0")
    parser.add_argument("--objective", default="mean_iota", choices=tuple(OBJECTIVE_FUNCS))
    parser.add_argument("--fd-rel-step", type=float, default=3e-7)
    parser.add_argument("--fd-abs-step", type=float, default=1e-10)
    parser.add_argument("--skip-fd", action="store_true")
    parser.add_argument("--reference-fd", type=float, default=None)
    parser.add_argument("--run-builtin-grad", action="store_true")
    parser.add_argument(
        "--forward-jvp-only",
        action="store_true",
        help="Run only the manual implicit forward tangent diagnostic, skipping reverse pullbacks.",
    )
    parser.add_argument(
        "--residual-check-only",
        action="store_true",
        help="Run only the implicit residual consistency check for the selected parameter tangent.",
    )
    parser.add_argument(
        "--residual-check-steps",
        default="1e-5,1e-6",
        help="Comma-separated h values for --residual-check-only.",
    )
    parser.add_argument(
        "--frozen-linearized-fd-only",
        action="store_true",
        help="Evaluate centered FD along the frozen implicit tangent path, without perturbed VMEC solves.",
    )
    parser.add_argument(
        "--frozen-linearized-fd-step",
        type=float,
        default=None,
        help="Step used for --frozen-linearized-fd-only; defaults to the normal FD step.",
    )
    parser.add_argument("--ns", type=int, default=None)
    parser.add_argument("--ftol", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--mode", default="cli")
    parser.add_argument("--multigrid", action="store_true")
    parser.add_argument("--no-lconm1", action="store_true")
    parser.add_argument("--adjoint-tol", type=float, default=1e-11)
    parser.add_argument("--adjoint-restart", type=int, default=30)
    parser.add_argument("--adjoint-maxiter", type=int, default=300)
    parser.add_argument(
        "--formulation",
        default="preconditioned",
        choices=("preconditioned", "raw"),
        help="Residual formulation used for the manual adjoint diagnostic.",
    )
    args = parser.parse_args()

    vmec_input = Path(args.vmec_input).resolve()
    inp = VmecInput.from_file(str(vmec_input))
    cfg = im.make_config(
        inp,
        ns=args.ns,
        ftol=args.ftol,
        max_iterations=args.max_iterations,
        mode=args.mode,
        multigrid=bool(args.multigrid),
        lconm1=not bool(args.no_lconm1),
        adjoint_tol=args.adjoint_tol,
        adjoint_restart=args.adjoint_restart,
        adjoint_maxiter=args.adjoint_maxiter,
    )
    params0 = im.params_from_input(inp)
    family, m, n = _parse_param_spec(args.parameter)
    row, col = _param_index(inp, family, m, n)
    base_value = _param_value(params0, family, row, col)
    h = _fd_step(base_value, rel_step=args.fd_rel_step, abs_step=args.fd_abs_step)

    print(
        "[vmec-pullback-rhs] "
        f"input={vmec_input} objective={args.objective} parameter={family.upper()}:{m}:{n} "
        f"ns={cfg.resolution.ns} ftol={cfg.ftol:.6e} max_iterations={cfg.max_iterations} "
        f"mode={cfg.mode} multigrid={cfg.multigrid} lconm1={cfg.lconm1} "
        f"formulation={args.formulation}",
        flush=True,
    )
    print(
        f"[vmec-pullback-rhs] jax_backend={jax.default_backend()} devices={jax.devices()}",
        flush=True,
    )

    t0 = time.perf_counter()
    print("[vmec-pullback-rhs] progress: baseline solve with aux mask", flush=True)
    x_star, dof_mask = im.solve_implicit_with_aux(params0, cfg)
    baseline = float(jax.device_get(_objective_value(args.objective, x_star, params0, cfg)))
    print(
        f"[vmec-pullback-rhs] baseline value={baseline:.16e} elapsed_s={time.perf_counter() - t0:.3f}",
        flush=True,
    )

    fd = args.reference_fd
    if not args.skip_fd and fd is None:
        print("[vmec-pullback-rhs] progress: centered FD minus/plus", flush=True)

        def fd_value(delta):
            p = _with_param_delta(params0, family, row, col, jnp.asarray(delta, dtype=jnp.float64))
            s = im.solve_implicit(p, cfg)
            return _objective_value(args.objective, s, p, cfg)

        minus = float(jax.device_get(fd_value(-h)))
        plus = float(jax.device_get(fd_value(h)))
        fd = (plus - minus) / (2.0 * h)
        print(
            f"[vmec-pullback-rhs] fd_step={h:.6e} fd={fd:.16e} "
            f"minus={minus:.16e} plus={plus:.16e}",
            flush=True,
        )
    elif fd is not None:
        print(f"[vmec-pullback-rhs] reference_fd={fd:.16e}", flush=True)
    else:
        print("[vmec-pullback-rhs] fd skipped", flush=True)

    if args.forward_jvp_only:
        print("[vmec-pullback-rhs] progress: manual implicit forward JVP", flush=True)
        tangent = _param_unit_tangent_like(params0, family, row, col)
        t_jvp = time.perf_counter()
        _value, tangent_value = _manual_implicit_forward_jvp(
            objective_name=args.objective,
            params=params0,
            param_tangent=tangent,
            cfg=cfg,
            x_star=x_star,
            dof_mask=dof_mask,
            formulation=args.formulation,
        )
        tangent_float = float(jax.device_get(tangent_value))
        print(
            f"[vmec-pullback-rhs] forward_jvp={tangent_float:.16e} "
            f"rel_err_vs_fd={_relative_error(tangent_float, fd):.6e} "
            f"elapsed_s={time.perf_counter() - t_jvp:.3f}",
            flush=True,
        )
        print(f"[vmec-pullback-rhs] total_elapsed_s={time.perf_counter() - t0:.3f}", flush=True)
        return

    if args.residual_check_only:
        print("[vmec-pullback-rhs] progress: implicit residual consistency check", flush=True)
        tangent = _param_unit_tangent_like(params0, family, row, col)
        steps = tuple(
            float(raw.strip())
            for raw in str(args.residual_check_steps).split(",")
            if raw.strip()
        )
        t_res = time.perf_counter()
        rows = _manual_residual_consistency_check(
            params=params0,
            param_tangent=tangent,
            cfg=cfg,
            x_star=x_star,
            dof_mask=dof_mask,
            formulation=args.formulation,
            steps=steps,
        )
        for label, step, l2, max_abs, scaled_l2, scaled_max in rows:
            if label == "base":
                print(
                    f"[vmec-pullback-rhs] residual {label}: l2={l2:.6e} max={max_abs:.6e}",
                    flush=True,
                )
            else:
                scale_label = "over_h" if label == "no_state" else "over_h2"
                print(
                    f"[vmec-pullback-rhs] residual h={step:.1e} {label}: "
                    f"l2={l2:.6e} max={max_abs:.6e} "
                    f"l2_{scale_label}={scaled_l2:.6e} max_{scale_label}={scaled_max:.6e}",
                    flush=True,
                )
        print(
            f"[vmec-pullback-rhs] residual_check_elapsed_s={time.perf_counter() - t_res:.3f}",
            flush=True,
        )
        print(f"[vmec-pullback-rhs] total_elapsed_s={time.perf_counter() - t0:.3f}", flush=True)
        return

    if args.frozen_linearized_fd_only:
        print("[vmec-pullback-rhs] progress: frozen linearized FD along implicit tangent", flush=True)
        tangent = _param_unit_tangent_like(params0, family, row, col)
        step = h if args.frozen_linearized_fd_step is None else float(args.frozen_linearized_fd_step)
        t_lin = time.perf_counter()
        minus, plus, lin_fd, jvp = _manual_frozen_linearized_fd(
            objective_name=args.objective,
            params=params0,
            param_tangent=tangent,
            cfg=cfg,
            x_star=x_star,
            dof_mask=dof_mask,
            formulation=args.formulation,
            step=step,
        )
        minus_f = float(jax.device_get(minus))
        plus_f = float(jax.device_get(plus))
        lin_fd_f = float(jax.device_get(lin_fd))
        jvp_f = float(jax.device_get(jvp))
        print(
            f"[vmec-pullback-rhs] frozen_linearized_fd_step={step:.6e} "
            f"minus={minus_f:.16e} plus={plus_f:.16e}",
            flush=True,
        )
        print(
            f"[vmec-pullback-rhs] frozen_linearized_fd={lin_fd_f:.16e} "
            f"forward_jvp={jvp_f:.16e} "
            f"rel_err_linfd_vs_jvp={_relative_error(lin_fd_f, jvp_f):.6e} "
            f"rel_err_linfd_vs_reference_fd={_relative_error(lin_fd_f, fd):.6e} "
            f"elapsed_s={time.perf_counter() - t_lin:.3f}",
            flush=True,
        )
        print(f"[vmec-pullback-rhs] total_elapsed_s={time.perf_counter() - t0:.3f}", flush=True)
        return

    print("[vmec-pullback-rhs] progress: objective cotangent wrt (state, params)", flush=True)
    (_, pullback) = jax.vjp(
        lambda state, params: _objective_value(args.objective, state, params, cfg),
        x_star,
        params0,
    )
    state_bar, direct_param_bar = pullback(jnp.asarray(1.0, dtype=jnp.float64))

    if args.run_builtin_grad:
        print("[vmec-pullback-rhs] progress: built-in custom_vjp gradient", flush=True)

        def builtin_scalar(delta):
            p = _with_param_delta(params0, family, row, col, delta)
            s = im.solve_implicit(p, cfg)
            return _objective_value(args.objective, s, p, cfg)

        builtin = float(
            jax.device_get(jax.grad(builtin_scalar)(jnp.asarray(0.0, dtype=jnp.float64)))
        )
        print(
            f"[vmec-pullback-rhs] builtin_grad={builtin:.16e} "
            f"rel_err_vs_fd={_relative_error(builtin, fd):.6e}",
            flush=True,
        )

    for rhs_mode in ("current", "assemble_vjp"):
        print(f"[vmec-pullback-rhs] progress: manual pullback rhs_mode={rhs_mode}", flush=True)
        t_rhs = time.perf_counter()
        grad_tree = _manual_implicit_pullback(
            params=params0,
            cfg=cfg,
            x_star=x_star,
            dof_mask=dof_mask,
            state_bar=state_bar,
            direct_param_bar=direct_param_bar,
            rhs_mode=rhs_mode,
            formulation=args.formulation,
        )
        grad_value = _param_component(grad_tree, family, row, col)
        print(
            f"[vmec-pullback-rhs] rhs_mode={rhs_mode} ad={grad_value:.16e} "
            f"rel_err_vs_fd={_relative_error(grad_value, fd):.6e} "
            f"elapsed_s={time.perf_counter() - t_rhs:.3f}",
            flush=True,
        )

    print(f"[vmec-pullback-rhs] total_elapsed_s={time.perf_counter() - t0:.3f}", flush=True)


if __name__ == "__main__":
    main()
