from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = ROOT.parent
for path in reversed((ROOT, WORKSPACE_ROOT / "VMEX", WORKSPACE_ROOT / "vmex", WORKSPACE_ROOT / "vmec_jax")):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from NEOPAX._geometry_autodiff import (  # noqa: E402
    _geometry_full_ad_objectives_from_state,
    _implicit_params_from_input,
    _import_vmec_jax_implicit,
    build_geometry_autodiff_context,
    geometry_observable_kind_from_single_param,
)
from NEOPAX._reverse_ad_optimization import geometry_full_ad_reverse_table  # noqa: E402
from NEOPAX._reverse_ad_parameters import ReverseADParameterSet, VmecBoundaryParameterSpec  # noqa: E402

im = _import_vmec_jax_implicit()


DEFAULT_VMEC_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_newNT_opt_hires_true"

OBJECTIVE_ALIASES = {
    "aspect_ratio": "vmec_aspect_ratio",
    "volume": "vmec_volume_total",
    "mean_iota": "vmec_iota_mean",
    "magnetic_well": "vmec_magnetic_well",
    "mirror_ratio": "vmec_mirror_ratio",
    "beta_volume": "vmec_beta_volume",
    "iota_b_mean": "boozer_iota_b_mean",
    "b00_mean": "boozer_b00_mean",
    "buco_b_mean": "boozer_buco_b_mean",
    "bvco_b_mean": "boozer_bvco_b_mean",
    "aspect_proxy": "boozer_aspect_proxy",
    "b10_over_b00_mean": "boozer_b10_over_b00_mean",
    "qi_objective": "boozer_qi_objective",
    "maxj_objective": "boozer_maxj_objective",
}


def _parse_param_spec(text: str) -> tuple[str, int, int]:
    parts = [part.strip() for part in str(text).split(":")]
    if len(parts) != 3:
        raise ValueError(f"Parameter spec {text!r} must be FAMILY:m:n, e.g. RBC:1:0.")
    family = parts[0].upper()
    if family not in {"RBC", "ZBS"}:
        raise ValueError(f"Unsupported family {family!r}; use RBC or ZBS.")
    return family, int(parts[1]), int(parts[2])


def _param_field(family: str) -> str:
    if family == "RBC":
        return "rbc"
    if family == "ZBS":
        return "zbs"
    raise ValueError(f"Unsupported family {family!r}.")


def _param_index(inp, family: str, m: int, n: int) -> tuple[int, int]:
    row = int(n) + int(inp.ntor)
    col = int(m)
    arr = getattr(_implicit_params_from_input_for_script(inp), _param_field(family))
    if row < 0 or row >= arr.shape[0] or col < 0 or col >= arr.shape[1]:
        raise ValueError(
            f"{family}:{m}:{n} maps to index {(row, col)}, outside shape {arr.shape}."
        )
    return row, col


def _implicit_params_from_input_for_script(inp, solver_device: str | None = "default"):
    # Tiny local context shim so this diagnostic uses the same placement helper
    # as the NEOPAX geometry objective-table benchmark.
    return _implicit_params_from_input(
        SimpleNamespace(indata=inp),
        im,
        solver_device=solver_device,
    )


def _param_unit_tangent_like(params, family: str, row: int, col: int):
    leaves = {
        field.name: jnp.zeros_like(getattr(params, field.name))
        for field in dataclasses.fields(params)
    }
    field_name = _param_field(family)
    leaves[field_name] = leaves[field_name].at[row, col].set(1.0)
    return dataclasses.replace(params, **leaves)


def _param_value(params, family: str, row: int, col: int) -> float:
    return float(np.asarray(getattr(params, _param_field(family)))[row, col])


def _fd_step(base_value: float, *, rel_step: float, abs_step: float) -> float:
    return max(abs(float(base_value)) * float(rel_step), float(abs_step))


def _relative_error(value: float, reference: float | None) -> float:
    if reference is None:
        return float("nan")
    return abs(value - reference) / max(abs(reference), 1.0e-300)


def _tree_dot(left, right):
    leaves = jax.tree.leaves(
        jax.tree.map(
            lambda a, b: jnp.sum(jnp.asarray(a, dtype=jnp.float64) * jnp.asarray(b, dtype=jnp.float64)),
            left,
            right,
        )
    )
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float64)
    return sum(leaves, jnp.asarray(0.0, dtype=jnp.float64))


def _tree_l2(tree):
    leaves = jax.tree.leaves(
        jax.tree.map(lambda a: jnp.sum(jnp.asarray(a, dtype=jnp.float64) ** 2), tree)
    )
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float64)
    return jnp.sqrt(sum(leaves, jnp.asarray(0.0, dtype=jnp.float64)))


def _tree_sub(left, right):
    return jax.tree.map(
        lambda a, b: jnp.asarray(a, dtype=jnp.float64) - jnp.asarray(b, dtype=jnp.float64),
        left,
        right,
    )


def _solver_info_text(info) -> str:
    fields = []
    for name in (
        "converged",
        "iterations",
        "num_iters",
        "niter",
        "restarts",
        "num_restarts",
        "residual_norm",
        "relative_residual",
        "res_norm",
        "error",
    ):
        if not hasattr(info, name):
            continue
        value = getattr(info, name)
        try:
            arr = jax.device_get(value)
            arr_np = np.asarray(arr)
            if arr_np.shape == ():
                if arr_np.dtype == np.bool_:
                    value_text = str(bool(arr_np))
                elif np.issubdtype(arr_np.dtype, np.integer):
                    value_text = str(int(arr_np))
                else:
                    value_text = f"{float(arr_np):.6e}"
            else:
                value_text = f"shape={arr_np.shape}"
        except Exception:
            value_text = repr(value)
        fields.append(f"{name}={value_text}")
    if not fields:
        fields.append(f"type={type(info).__name__}")
    return " ".join(fields)


def _adjoint_solve_jax_scipy(A, b, cfg):
    return jax.scipy.sparse.linalg.gmres(
        A,
        b,
        tol=cfg.adjoint_tol,
        atol=0.0,
        restart=cfg.adjoint_restart,
        maxiter=cfg.adjoint_maxiter,
        solve_method="incremental",
    )


def _manual_implicit_pullback(
    *,
    params,
    cfg,
    x_star,
    dof_mask,
    state_bar,
    formulation: str,
    adjoint_solve=None,
):
    frozen = jax.lax.stop_gradient(x_star)
    edge_mask = im._edge_mask(cfg)
    P = im._dof_projector(cfg, dof_mask)
    F = im.residual_fn(cfg, frozen, dof_mask, formulation=formulation)
    z_star = P(x_star)

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

    _, vjp_z = jax.vjp(lambda z: F(z, params), z_star)
    solve = im._adjoint_solve if adjoint_solve is None else adjoint_solve
    lam, _ = solve(lambda v: vjp_z(v)[0], rhs, cfg)

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
    return jax.tree.map(lambda a, b: a + b, implicit_param_bar, assemble_param_bar)


def _manual_implicit_forward_state_tangent(
    *,
    params,
    param_tangent,
    cfg,
    x_star,
    dof_mask,
    formulation: str,
    linear_solve_mode: str,
    probe_chunk_size: int,
):
    frozen = jax.lax.stop_gradient(x_star)
    edge_mask = im._edge_mask(cfg)
    P = im._dof_projector(cfg, dof_mask)
    z_star = P(x_star)

    if linear_solve_mode == "raw_block":
        state_tangent = im.implicit_state_tangent_raw_block(
            params,
            cfg,
            x_star,
            dof_mask,
            param_tangent,
            probe_chunk_size=probe_chunk_size,
        )
        return (
            state_tangent,
            SimpleNamespace(solver="raw_block_direct"),
            jnp.asarray(float("nan"), dtype=jnp.float64),
            jnp.asarray(float("nan"), dtype=jnp.float64),
        )

    if linear_solve_mode == "block_corrected":
        state_tangent = im.implicit_state_tangent_block_corrected(
            params,
            cfg,
            x_star,
            dof_mask,
            param_tangent,
            probe_chunk_size=probe_chunk_size,
        )
        return (
            state_tangent,
            SimpleNamespace(solver="raw_block_plus_short_preconditioned_gmres"),
            jnp.asarray(float("nan"), dtype=jnp.float64),
            jnp.asarray(float("nan"), dtype=jnp.float64),
        )

    F = im.residual_fn(cfg, frozen, dof_mask, formulation=formulation)

    def F_z(z):
        return F(z, params)

    def F_p(prm):
        return F(z_star, prm)

    rhs = jax.tree.map(jnp.negative, jax.jvp(F_p, (params,), (param_tangent,))[1])
    forward_matvec = lambda v: jax.jvp(F_z, (z_star,), (v,))[1]
    dz, solve_info = im._adjoint_solve(forward_matvec, rhs, cfg)
    residual = _tree_sub(forward_matvec(dz), rhs)

    def assemble_from_z_params(z, prm):
        return im._assemble(
            z,
            im.runtime_from_params(prm, cfg),
            frozen,
            P,
            edge_mask,
        )

    return jax.jvp(
        assemble_from_z_params,
        (z_star, params),
        (dz, param_tangent),
    )[1], solve_info, _tree_l2(residual), _tree_l2(rhs)


def _objective(context, objective_name: str, state):
    values = _geometry_full_ad_objectives_from_state(context, state)
    if objective_name not in values:
        raise ValueError(f"Unknown objective {objective_name!r}; choices are {', '.join(values)}.")
    return jnp.asarray(values[objective_name], dtype=jnp.float64).reshape(())


def _normalize_objective_name(name: str) -> str:
    return OBJECTIVE_ALIASES.get(str(name), str(name))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only frozen-linearized FD for NEOPAX geometry_full_ad_objectives "
            "such as boozer_qi_objective and boozer_maxj_objective. "
            "Does not edit vmec_jax or NEOPAX production lanes."
        )
    )
    parser.add_argument("--vmec-input", default=str(DEFAULT_VMEC_INPUT))
    parser.add_argument("--parameter", default="RBC:1:0")
    parser.add_argument("--objective", default="boozer_qi_objective")
    parser.add_argument("--fd-rel-step", type=float, default=3e-7)
    parser.add_argument("--fd-abs-step", type=float, default=1e-10)
    parser.add_argument("--reference-fd", type=float, default=None)
    parser.add_argument("--run-full-fd", action="store_true")
    parser.add_argument("--mboz", type=int, default=18)
    parser.add_argument("--nboz", type=int, default=18)
    parser.add_argument("--surface-s", default="0.1,0.28,0.46,0.64,0.82,1.0")
    parser.add_argument("--mode", default="cli")
    parser.add_argument("--multigrid", action="store_true")
    parser.add_argument("--formulation", default="preconditioned", choices=("preconditioned", "raw"))
    parser.add_argument("--adjoint-tol", type=float, default=1e-11)
    parser.add_argument("--adjoint-restart", type=int, default=30)
    parser.add_argument("--adjoint-maxiter", type=int, default=300)
    parser.add_argument(
        "--forward-linear-maxiter",
        type=int,
        default=300,
        help=(
            "GMRES maxiter for the frozen forward tangent solve A dz = rhs. "
            "Kept separate from --adjoint-maxiter so reverse-solve budget "
            "sweeps do not move the forward-JVP reference."
        ),
    )
    parser.add_argument(
        "--forward-linear-solve-mode",
        type=str,
        default="gmres",
        choices=("gmres", "raw_block", "block_corrected"),
        help=(
            "Linear solver for the frozen forward tangent. 'gmres' uses the "
            "current preconditioned residual JVP; 'raw_block' uses the same "
            "raw block-tridiagonal operator family as raw_block_transpose_reverse; "
            "'block_corrected' mirrors VMEX optimization's raw block solve plus "
            "short preconditioned GMRES correction."
        ),
    )
    parser.add_argument(
        "--implicit-solver-device",
        type=str,
        default="default",
        choices=("default", "auto", "cpu", "gpu"),
        help=(
            "Device placement for VMEX implicit AD parameters. 'default' preserves "
            "old vmec_jax behavior by leaving placement to JAX; 'auto' uses VMEX policy."
        ),
    )
    parser.add_argument(
        "--block-transpose-probe-chunk-size",
        type=int,
        default=1,
        help="Probe chunk size for the optional raw-block-transpose reverse initializer.",
    )
    parser.add_argument(
        "--block-transpose-corrector-max-restarts",
        type=int,
        default=-1,
        help=(
            "Max GMRES restarts for the optional block-transpose-initialized reverse "
            "corrector. Use -1 to use the same budget as --adjoint-maxiter."
        ),
    )
    parser.add_argument(
        "--run-right-preconditioned-reverse-check",
        action="store_true",
        help="Also run the experimental right-preconditioned transpose reverse check.",
    )
    parser.add_argument("--skip-reverse-check", action="store_true")
    args = parser.parse_args()
    args.objective = _normalize_objective_name(args.objective)

    family, m, n = _parse_param_spec(args.parameter)
    surfaces = tuple(float(item.strip()) for item in str(args.surface_s).split(",") if item.strip())
    context = build_geometry_autodiff_context(
        args.vmec_input,
        param_family=family,
        param_m=m,
        param_n=n,
        mboz=args.mboz,
        nboz=args.nboz,
        surface_s=surfaces,
    )
    inp = context.indata
    cfg = im.make_config(
        inp,
        mode=args.mode,
        multigrid=bool(args.multigrid),
        lconm1=True,
        adjoint_tol=args.adjoint_tol,
        adjoint_restart=args.adjoint_restart,
        adjoint_maxiter=args.adjoint_maxiter,
    )
    forward_cfg = dataclasses.replace(cfg, adjoint_maxiter=int(args.forward_linear_maxiter))
    params0 = _implicit_params_from_input_for_script(inp, solver_device=args.implicit_solver_device)
    # VMEX runtime_from_params expects the p-independent template runtime to be
    # a host-built closure constant.  Prewarm both configs so JIT tracing never
    # enters setup.run_setup's discrete host logic.
    if hasattr(im, "_template_runtime"):
        im._template_runtime(cfg)
        im._template_runtime(forward_cfg)
    row, col = _param_index(inp, family, m, n)
    base_value = _param_value(params0, family, row, col)
    h = _fd_step(base_value, rel_step=args.fd_rel_step, abs_step=args.fd_abs_step)

    print(
        "[geometry-qi-linearized-fd] "
        f"input={Path(args.vmec_input).resolve()} objective={args.objective} "
        f"parameter={family}:{m}:{n} ns={cfg.resolution.ns} ftol={cfg.ftol:.6e} "
        f"mboz={args.mboz} nboz={args.nboz} surfaces={','.join(f'{s:.3f}' for s in surfaces)} "
        f"formulation={args.formulation} implicit_solver_device={args.implicit_solver_device} "
        f"forward_linear_solve_mode={args.forward_linear_solve_mode} "
        f"forward_linear_maxiter={forward_cfg.adjoint_maxiter} reverse_adjoint_maxiter={cfg.adjoint_maxiter}",
        flush=True,
    )
    print(
        f"[geometry-qi-linearized-fd] jax_backend={jax.default_backend()} devices={jax.devices()}",
        flush=True,
    )

    t0 = time.perf_counter()
    print("[geometry-qi-linearized-fd] progress: baseline solve with aux mask", flush=True)
    x_star, dof_mask = im.solve_implicit_with_aux(params0, cfg)
    baseline = float(jax.device_get(_objective(context, args.objective, x_star)))
    print(
        f"[geometry-qi-linearized-fd] baseline value={baseline:.16e} elapsed_s={time.perf_counter() - t0:.3f}",
        flush=True,
    )

    fd = args.reference_fd
    if args.run_full_fd:
        print("[geometry-qi-linearized-fd] progress: full nonlinear FD via geometry benchmark path", flush=True)
        minus = geometry_observable_kind_from_single_param(
            context,
            jnp.asarray(-h, dtype=jnp.float64),
            observable_kind="geometry_full_ad_objectives",
            lane="ad",
            max_iter=None,
            step_size=None,
        )[args.objective]
        plus = geometry_observable_kind_from_single_param(
            context,
            jnp.asarray(h, dtype=jnp.float64),
            observable_kind="geometry_full_ad_objectives",
            lane="ad",
            max_iter=None,
            step_size=None,
        )[args.objective]
        minus_f = float(jax.device_get(minus))
        plus_f = float(jax.device_get(plus))
        fd = (plus_f - minus_f) / (2.0 * h)
        print(
            f"[geometry-qi-linearized-fd] full_fd_step={h:.6e} full_fd={fd:.16e} "
            f"minus={minus_f:.16e} plus={plus_f:.16e}",
            flush=True,
        )
    elif fd is not None:
        print(f"[geometry-qi-linearized-fd] reference_fd={fd:.16e}", flush=True)
    else:
        print("[geometry-qi-linearized-fd] full FD skipped", flush=True)

    print("[geometry-qi-linearized-fd] progress: frozen linearized FD along implicit tangent", flush=True)
    tangent = _param_unit_tangent_like(params0, family, row, col)
    state_tangent, forward_solve_info, forward_res_l2, forward_rhs_l2 = _manual_implicit_forward_state_tangent(
        params=params0,
        param_tangent=tangent,
        cfg=forward_cfg,
        x_star=x_star,
        dof_mask=dof_mask,
        formulation=args.formulation,
        linear_solve_mode=args.forward_linear_solve_mode,
        probe_chunk_size=max(1, int(args.block_transpose_probe_chunk_size)),
    )
    forward_rel_res = forward_res_l2 / jnp.maximum(
        forward_rhs_l2,
        jnp.asarray(1.0e-300, dtype=jnp.float64),
    )
    step_arr = jnp.asarray(h, dtype=jnp.float64)
    state_minus = jax.tree.map(lambda x, t: x - step_arr * t, x_star, state_tangent)
    state_plus = jax.tree.map(lambda x, t: x + step_arr * t, x_star, state_tangent)
    minus = _objective(context, args.objective, state_minus)
    plus = _objective(context, args.objective, state_plus)
    lin_fd = (plus - minus) / (2.0 * step_arr)
    _value, jvp = jax.jvp(
        lambda state: _objective(context, args.objective, state),
        (x_star,),
        (state_tangent,),
    )
    lin_fd_f = float(jax.device_get(lin_fd))
    jvp_f = float(jax.device_get(jvp))
    print(
        f"[geometry-qi-linearized-fd] frozen_linearized_fd_step={h:.6e} "
        f"minus={float(jax.device_get(minus)):.16e} plus={float(jax.device_get(plus)):.16e}",
        flush=True,
    )
    print(
        f"[geometry-qi-linearized-fd] frozen_linearized_fd={lin_fd_f:.16e} "
        f"forward_jvp={jvp_f:.16e} "
        f"forward_linear_res_l2={float(jax.device_get(forward_res_l2)):.6e} "
        f"forward_linear_rel_res={float(jax.device_get(forward_rel_res)):.6e} "
        f"forward_linear_solver_info=({_solver_info_text(forward_solve_info)}) "
        f"rel_err_linfd_vs_jvp={_relative_error(lin_fd_f, jvp_f):.6e} "
        f"rel_err_linfd_vs_reference_fd={_relative_error(lin_fd_f, fd):.6e} "
        f"total_elapsed_s={time.perf_counter() - t0:.3f}",
        flush=True,
    )
    if not args.skip_reverse_check:
        print(
            "[geometry-qi-linearized-fd] progress: optimization-internal geometry reverse table check",
            flush=True,
        )
        parameter_set = ReverseADParameterSet(
            vmec_boundary_specs=(VmecBoundaryParameterSpec(family, m, n),)
        )
        internal_table = geometry_full_ad_reverse_table(
            context=context,
            parameter_set=parameter_set,
            objective_names=(args.objective,),
            parameter_values=jnp.zeros((1,), dtype=jnp.float64),
            lane="ad",
            max_iter=cfg.adjoint_maxiter,
            step_size=None,
            final_vmec_pullback_mode="raw_block_transpose",
            solver_device=args.implicit_solver_device,
        )
        internal_value_f = float(jax.device_get(internal_table.values[0]))
        internal_reverse_grad_f = float(jax.device_get(internal_table.jacobian[0, 0]))
        print(
            "[geometry-qi-linearized-fd] optimization_internal_reverse_table "
            f"value={internal_value_f:.16e} "
            f"raw_block_transpose_param_grad={internal_reverse_grad_f:.16e} "
            f"rel_err_internal_reverse_vs_jvp={_relative_error(internal_reverse_grad_f, jvp_f):.6e}",
            flush=True,
        )
        print("[geometry-qi-linearized-fd] progress: same-baseline reverse pullback check", flush=True)
        _objective_value, objective_vjp = jax.vjp(lambda state: _objective(context, args.objective, state), x_star)
        state_bar = objective_vjp(jnp.asarray(1.0, dtype=jnp.float64))[0]
        state_dot_tangent = _tree_dot(state_bar, state_tangent)
        frozen = jax.lax.stop_gradient(x_star)
        edge_mask = im._edge_mask(cfg)
        P = im._dof_projector(cfg, dof_mask)
        F = im.residual_fn(cfg, frozen, dof_mask, formulation=args.formulation)
        z_star = P(x_star)
        _, assemble_vjp_z = jax.vjp(
            lambda z: im._assemble(z, im.runtime_from_params(params0, cfg), frozen, P, edge_mask),
            z_star,
        )
        assembled_rhs = assemble_vjp_z(state_bar)[0]
        projected_rhs = P(state_bar)
        rhs_diff_l2 = _tree_l2(_tree_sub(assembled_rhs, projected_rhs))
        rhs_l2 = _tree_l2(projected_rhs)
        _, vjp_z = jax.vjp(lambda z: F(z, params0), z_star)
        builtin_lam, builtin_lam_info = im._adjoint_solve(lambda v: vjp_z(v)[0], projected_rhs, cfg)
        assembled_lam, assembled_lam_info = im._adjoint_solve(lambda v: vjp_z(v)[0], assembled_rhs, cfg)
        builtin_res_l2 = _tree_l2(_tree_sub(vjp_z(builtin_lam)[0], projected_rhs))
        assembled_res_l2 = _tree_l2(_tree_sub(vjp_z(assembled_lam)[0], assembled_rhs))
        rhs_l2_safe = jnp.maximum(rhs_l2, jnp.asarray(1.0e-300, dtype=jnp.float64))
        builtin_rel_res = builtin_res_l2 / rhs_l2_safe
        assembled_rel_res = assembled_res_l2 / rhs_l2_safe
        param_bar = _manual_implicit_pullback(
            params=params0,
            cfg=cfg,
            x_star=x_star,
            dof_mask=dof_mask,
            state_bar=state_bar,
            formulation=args.formulation,
        )
        scipy_param_bar = _manual_implicit_pullback(
            params=params0,
            cfg=cfg,
            x_star=x_star,
            dof_mask=dof_mask,
            state_bar=state_bar,
            formulation=args.formulation,
            adjoint_solve=_adjoint_solve_jax_scipy,
        )
        field_name = _param_field(family)
        reverse_grad = jnp.asarray(getattr(param_bar, field_name), dtype=jnp.float64)[row, col]
        scipy_reverse_grad = jnp.asarray(getattr(scipy_param_bar, field_name), dtype=jnp.float64)[row, col]
        builtin_param_bar = im.implicit_state_pullback_multi_rhs(
            params0,
            cfg,
            x_star,
            dof_mask,
            jax.tree.map(lambda leaf: jnp.expand_dims(leaf, axis=0), state_bar),
        )
        builtin_reverse_grad = jnp.asarray(
            getattr(builtin_param_bar, field_name),
            dtype=jnp.float64,
        )[0, row, col]
        block_reverse_grad_f = None
        raw_block_reverse_grad_f = None
        right_precond_reverse_grad_f = None
        if hasattr(im, "implicit_state_pullback_multi_rhs_raw_block_transpose"):
            raw_block_param_bar = im.implicit_state_pullback_multi_rhs_raw_block_transpose(
                params0,
                cfg,
                x_star,
                dof_mask,
                jax.tree.map(lambda leaf: jnp.expand_dims(leaf, axis=0), state_bar),
                probe_chunk_size=max(1, int(args.block_transpose_probe_chunk_size)),
            )
            raw_block_reverse_grad = jnp.asarray(
                getattr(raw_block_param_bar, field_name),
                dtype=jnp.float64,
            )[0, row, col]
            raw_block_reverse_grad_f = float(jax.device_get(raw_block_reverse_grad))
        if (
            bool(args.run_right_preconditioned_reverse_check)
            and hasattr(im, "implicit_state_pullback_multi_rhs_block_transpose_right_preconditioned")
        ):
            right_precond_param_bar = (
                im.implicit_state_pullback_multi_rhs_block_transpose_right_preconditioned(
                    params0,
                    cfg,
                    x_star,
                    dof_mask,
                    jax.tree.map(lambda leaf: jnp.expand_dims(leaf, axis=0), state_bar),
                    probe_chunk_size=max(1, int(args.block_transpose_probe_chunk_size)),
                )
            )
            right_precond_reverse_grad = jnp.asarray(
                getattr(right_precond_param_bar, field_name),
                dtype=jnp.float64,
            )[0, row, col]
            right_precond_reverse_grad_f = float(jax.device_get(right_precond_reverse_grad))
        if hasattr(im, "implicit_state_pullback_multi_rhs_block_transpose_init"):
            block_param_bar = im.implicit_state_pullback_multi_rhs_block_transpose_init(
                params0,
                cfg,
                x_star,
                dof_mask,
                jax.tree.map(lambda leaf: jnp.expand_dims(leaf, axis=0), state_bar),
                corrector_max_restarts=(
                    None
                    if int(args.block_transpose_corrector_max_restarts) < 0
                    else int(args.block_transpose_corrector_max_restarts)
                ),
                probe_chunk_size=max(1, int(args.block_transpose_probe_chunk_size)),
            )
            block_reverse_grad = jnp.asarray(
                getattr(block_param_bar, field_name),
                dtype=jnp.float64,
            )[0, row, col]
            block_reverse_grad_f = float(jax.device_get(block_reverse_grad))
        state_dot_f = float(jax.device_get(state_dot_tangent))
        reverse_grad_f = float(jax.device_get(reverse_grad))
        scipy_reverse_grad_f = float(jax.device_get(scipy_reverse_grad))
        builtin_reverse_grad_f = float(jax.device_get(builtin_reverse_grad))
        block_text = ""
        if block_reverse_grad_f is not None:
            block_text = (
                f" block_transpose_init_reverse_param_grad={block_reverse_grad_f:.16e}"
                f" rel_err_block_transpose_init_reverse_vs_jvp="
                f"{_relative_error(block_reverse_grad_f, jvp_f):.6e}"
            )
        raw_block_text = ""
        if raw_block_reverse_grad_f is not None:
            raw_block_text = (
                f" raw_block_transpose_reverse_param_grad={raw_block_reverse_grad_f:.16e}"
                f" rel_err_raw_block_transpose_reverse_vs_jvp="
                f"{_relative_error(raw_block_reverse_grad_f, jvp_f):.6e}"
            )
        right_precond_text = ""
        if right_precond_reverse_grad_f is not None:
            right_precond_text = (
                f" block_transpose_right_preconditioned_reverse_param_grad="
                f"{right_precond_reverse_grad_f:.16e}"
                f" rel_err_block_transpose_right_preconditioned_reverse_vs_jvp="
                f"{_relative_error(right_precond_reverse_grad_f, jvp_f):.6e}"
            )
        print(
            f"[geometry-qi-linearized-fd] implicit_pullback_diagnostics "
            f"rhs_diff_l2={float(jax.device_get(rhs_diff_l2)):.6e} "
            f"rhs_l2={float(jax.device_get(rhs_l2)):.6e} "
            f"builtin_adjoint_res_l2={float(jax.device_get(builtin_res_l2)):.6e} "
            f"builtin_adjoint_rel_res={float(jax.device_get(builtin_rel_res)):.6e} "
            f"assembled_adjoint_res_l2={float(jax.device_get(assembled_res_l2)):.6e} "
            f"assembled_adjoint_rel_res={float(jax.device_get(assembled_rel_res)):.6e} "
            f"builtin_solver_info=({_solver_info_text(builtin_lam_info)}) "
            f"assembled_solver_info=({_solver_info_text(assembled_lam_info)})",
            flush=True,
        )
        print(
            f"[geometry-qi-linearized-fd] reverse_state_dot_tangent={state_dot_f:.16e} "
            f"reverse_param_grad={reverse_grad_f:.16e} "
            f"jax_scipy_reverse_param_grad={scipy_reverse_grad_f:.16e} "
            f"implicit_reverse_param_grad={builtin_reverse_grad_f:.16e} "
            f"{raw_block_text}{right_precond_text}{block_text} "
            f"rel_err_state_dot_vs_jvp={_relative_error(state_dot_f, jvp_f):.6e} "
            f"rel_err_reverse_vs_jvp={_relative_error(reverse_grad_f, jvp_f):.6e} "
            f"rel_err_jax_scipy_reverse_vs_jvp={_relative_error(scipy_reverse_grad_f, jvp_f):.6e} "
            f"rel_err_implicit_reverse_vs_jvp={_relative_error(builtin_reverse_grad_f, jvp_f):.6e}",
            flush=True,
        )

if __name__ == "__main__":
    main()
