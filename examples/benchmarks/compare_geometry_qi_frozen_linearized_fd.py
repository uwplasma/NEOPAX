from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = ROOT.parent
for path in (ROOT, WORKSPACE_ROOT / "vmec_jax"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from NEOPAX._geometry_autodiff import (  # noqa: E402
    _booz_constants_and_grids_for_inputs,
    _booz_xform_inputs_from_state,
    _find_boozer_mode_index,
    _geometry_full_ad_objectives_from_state,
    _import_booz_xform_jax_api,
    _vmec_booz_qi_scalar_objective_from_boozer,
    build_geometry_autodiff_context,
    geometry_observable_kind_from_single_param,
)
from vmec_jax.core import implicit as im  # noqa: E402


DEFAULT_VMEC_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_newNT_opt_hires_true"


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
    arr = getattr(im.params_from_input(inp), _param_field(family))
    if row < 0 or row >= arr.shape[0] or col < 0 or col >= arr.shape[1]:
        raise ValueError(
            f"{family}:{m}:{n} maps to index {(row, col)}, outside shape {arr.shape}."
        )
    return row, col


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


def _manual_implicit_pullback(
    *,
    params,
    cfg,
    x_star,
    dof_mask,
    state_bar,
    formulation: str,
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
    return jax.tree.map(lambda a, b: a + b, implicit_param_bar, assemble_param_bar)


def _compact_qi_state_bar(context, state):
    booz_api = _import_booz_xform_jax_api()
    booz_inputs = _booz_xform_inputs_from_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=context.signgs,
        flux=context.flux,
    )
    booz_constants, booz_grids = _booz_constants_and_grids_for_inputs(context, booz_inputs)
    ixm_b = jnp.asarray(booz_grids.xm_b, dtype=jnp.int32)
    ixn_b = jnp.asarray(booz_grids.xn_b, dtype=jnp.int32)
    mode00 = _find_boozer_mode_index(booz_grids.xm_b, booz_grids.xn_b, m_value=0, n_value=0)
    mode10 = _find_boozer_mode_index(booz_grids.xm_b, booz_grids.xn_b, m_value=1, n_value=0)

    def booz_float_output_from_state(state_inner):
        inputs_inner = _booz_xform_inputs_from_state(
            state=state_inner,
            static=context.static,
            indata=context.indata,
            signgs=context.signgs,
            flux=context.flux,
        )
        out = booz_api.booz_xform_from_inputs(
            inputs=inputs_inner,
            constants=booz_constants,
            grids=booz_grids,
            surface_indices=context.surface_indices,
            jit=True,
        )
        return {
            key: jnp.asarray(out[key], dtype=jnp.float64)
            for key in ("iota_b", "buco_b", "bvco_b", "bmnc_b")
        }

    def booz_with_modes(booz_float):
        out = dict(booz_float)
        out["ixm_b"] = ixm_b
        out["ixn_b"] = ixn_b
        out["_mode00"] = mode00
        out["_mode10"] = mode10
        return out

    booz, booz_state_pullback = jax.vjp(booz_float_output_from_state, state)

    def qi_scalar(booz_inner):
        values = _vmec_booz_qi_scalar_objective_from_boozer(context, booz_with_modes(booz_inner))
        return jnp.asarray(values["qi_objective"], dtype=jnp.float64).reshape(())

    _qi_value, qi_pullback = jax.vjp(qi_scalar, booz)
    qi_boozer_bar = qi_pullback(jnp.asarray(1.0, dtype=jnp.float64))[0]
    return booz_state_pullback(qi_boozer_bar)[0]


def _manual_implicit_forward_state_tangent(
    *,
    params,
    param_tangent,
    cfg,
    x_star,
    dof_mask,
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

    rhs = jax.tree.map(jnp.negative, jax.jvp(F_p, (params,), (param_tangent,))[1])
    dz, _ = im._adjoint_solve(lambda v: jax.jvp(F_z, (z_star,), (v,))[1], rhs, cfg)

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
    )[1]


def _objective(context, objective_name: str, state):
    values = _geometry_full_ad_objectives_from_state(context, state)
    if objective_name not in values:
        raise ValueError(f"Unknown objective {objective_name!r}; choices are {', '.join(values)}.")
    return jnp.asarray(values[objective_name], dtype=jnp.float64).reshape(())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only frozen-linearized FD for NEOPAX geometry_full_ad_objectives "
            "such as boozer_qi_objective. Does not edit vmec_jax or NEOPAX production lanes."
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
    parser.add_argument("--skip-reverse-check", action="store_true")
    args = parser.parse_args()

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
    params0 = im.params_from_input(inp)
    row, col = _param_index(inp, family, m, n)
    base_value = _param_value(params0, family, row, col)
    h = _fd_step(base_value, rel_step=args.fd_rel_step, abs_step=args.fd_abs_step)

    print(
        "[geometry-qi-linearized-fd] "
        f"input={Path(args.vmec_input).resolve()} objective={args.objective} "
        f"parameter={family}:{m}:{n} ns={cfg.resolution.ns} ftol={cfg.ftol:.6e} "
        f"mboz={args.mboz} nboz={args.nboz} surfaces={','.join(f'{s:.3f}' for s in surfaces)} "
        f"formulation={args.formulation}",
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
    state_tangent = _manual_implicit_forward_state_tangent(
        params=params0,
        param_tangent=tangent,
        cfg=cfg,
        x_star=x_star,
        dof_mask=dof_mask,
        formulation=args.formulation,
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
        f"rel_err_linfd_vs_jvp={_relative_error(lin_fd_f, jvp_f):.6e} "
        f"rel_err_linfd_vs_reference_fd={_relative_error(lin_fd_f, fd):.6e} "
        f"total_elapsed_s={time.perf_counter() - t0:.3f}",
        flush=True,
    )
    if not args.skip_reverse_check:
        print("[geometry-qi-linearized-fd] progress: same-baseline reverse pullback check", flush=True)
        _objective_value, objective_vjp = jax.vjp(lambda state: _objective(context, args.objective, state), x_star)
        state_bar = objective_vjp(jnp.asarray(1.0, dtype=jnp.float64))[0]
        state_dot_tangent = _tree_dot(state_bar, state_tangent)
        compact_state_bar = _compact_qi_state_bar(context, x_star)
        compact_state_dot_tangent = _tree_dot(compact_state_bar, state_tangent)
        param_bar = _manual_implicit_pullback(
            params=params0,
            cfg=cfg,
            x_star=x_star,
            dof_mask=dof_mask,
            state_bar=state_bar,
            formulation=args.formulation,
        )
        field_name = _param_field(family)
        reverse_grad = jnp.asarray(getattr(param_bar, field_name), dtype=jnp.float64)[row, col]
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
        state_dot_f = float(jax.device_get(state_dot_tangent))
        compact_state_dot_f = float(jax.device_get(compact_state_dot_tangent))
        reverse_grad_f = float(jax.device_get(reverse_grad))
        builtin_reverse_grad_f = float(jax.device_get(builtin_reverse_grad))
        print(
            f"[geometry-qi-linearized-fd] reverse_state_dot_tangent={state_dot_f:.16e} "
            f"compact_reverse_state_dot_tangent={compact_state_dot_f:.16e} "
            f"reverse_param_grad={reverse_grad_f:.16e} "
            f"implicit_reverse_param_grad={builtin_reverse_grad_f:.16e} "
            f"rel_err_state_dot_vs_jvp={_relative_error(state_dot_f, jvp_f):.6e} "
            f"rel_err_compact_state_dot_vs_jvp={_relative_error(compact_state_dot_f, jvp_f):.6e} "
            f"rel_err_reverse_vs_jvp={_relative_error(reverse_grad_f, jvp_f):.6e} "
            f"rel_err_implicit_reverse_vs_jvp={_relative_error(builtin_reverse_grad_f, jvp_f):.6e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
