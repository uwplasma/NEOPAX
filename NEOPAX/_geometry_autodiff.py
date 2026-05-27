from __future__ import annotations

import dataclasses
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import interpax
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_local_stack_on_path() -> None:
    repo = _repo_root()
    candidates = (
        repo / "vmec_jax",
        repo / "booz_xform_jax" / "src",
    )
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _import_vmec_jax():
    _ensure_local_stack_on_path()
    import vmec_jax

    return vmec_jax


def _import_booz_xform_jax_api():
    _ensure_local_stack_on_path()
    from booz_xform_jax import jax_api

    return jax_api


def _resolve_vmec_attr(module, name: str, *, submodule: str | None = None):
    value = getattr(module, name, None)
    if value is not None:
        return value
    if submodule is None:
        raise AttributeError(f"vmec_jax does not provide '{name}'.")
    imported = __import__(f"vmec_jax.{submodule}", fromlist=[name])
    value = getattr(imported, name, None)
    if value is None:
        raise AttributeError(f"vmec_jax.{submodule} does not provide '{name}'.")
    return value


def _build_vmec_fixed_context(vmec_jax, *, static, indata, boundary):
    initial_guess_from_boundary = _resolve_vmec_attr(vmec_jax, "initial_guess_from_boundary", submodule="init_guess")
    eval_geom = _resolve_vmec_attr(vmec_jax, "eval_geom", submodule="geom")
    signgs_from_sqrtg = _resolve_vmec_attr(vmec_jax, "signgs_from_sqrtg", submodule="field")
    flux_profiles_from_indata = _resolve_vmec_attr(vmec_jax, "flux_profiles_from_indata", submodule="energy")
    eval_profiles = _resolve_vmec_attr(vmec_jax, "eval_profiles", submodule="profiles")
    booz_xform_inputs_from_state = _resolve_vmec_attr(vmec_jax, "booz_xform_inputs_from_state", submodule="booz_input")

    st_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=False)
    geom = eval_geom(st_guess, static)
    signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))
    flux = flux_profiles_from_indata(indata, jnp.asarray(static.s), signgs=signgs)
    prof = eval_profiles(indata, jnp.asarray(static.s))
    pressure = jnp.asarray(prof.get("pressure", jnp.zeros_like(jnp.asarray(static.s))))
    booz_inputs = booz_xform_inputs_from_state(
        state=st_guess,
        static=static,
        indata=indata,
        signgs=signgs,
        flux=flux,
    )
    return {
        "st_guess": st_guess,
        "signgs": signgs,
        "flux": flux,
        "pressure": pressure,
        "booz_inputs": booz_inputs,
    }


@dataclasses.dataclass(frozen=True)
class GeometryAutodiffContext:
    input_path: Path
    param_family: str
    param_m: int
    param_n: int
    cfg: Any
    indata: Any
    static: Any
    boundary: Any
    boundary_kind: str
    boundary_index: int
    signgs: int
    flux: Any
    pressure: jnp.ndarray
    surface_s: tuple[float, ...]
    surface_indices: jnp.ndarray
    mboz: int
    nboz: int
    booz_constants: Any
    booz_grids: Any
    baseline_coefficient: float
    vmec_default_max_iter: int
    vmec_default_step_size: float
    vmec_default_ftol: float | None


def _boundary_kind_for_family(family: str) -> str:
    family_upper = str(family).strip().upper()
    if family_upper == "RBC":
        return "rc"
    if family_upper == "ZBS":
        return "zs"
    raise ValueError("param_family must be 'RBC' or 'ZBS'.")


def _boundary_array_name_for_kind(kind: str) -> str:
    if kind == "rc":
        return "R_cos"
    if kind == "zs":
        return "Z_sin"
    raise ValueError(f"Unsupported boundary kind '{kind}'.")


def _vmec_default_max_iter_from_indata(indata: Any) -> int:
    niter_array = indata.get("NITER_ARRAY", None)
    if niter_array is not None:
        try:
            values = [int(v) for v in niter_array]
            if values:
                return int(values[-1])
        except Exception:
            pass
    try:
        return int(indata.get_int("NITER", 100))
    except Exception:
        return 100


def _vmec_default_step_size_from_indata(indata: Any) -> float:
    try:
        return float(indata.get_float("DELT", 1.0))
    except Exception:
        return 1.0


def _vmec_default_ftol_from_indata(indata: Any) -> float | None:
    ftol_array = indata.get("FTOL_ARRAY", None)
    if ftol_array is not None:
        try:
            values = [float(v) for v in ftol_array]
            if values:
                return float(values[-1])
        except Exception:
            pass
    try:
        value = indata.get_float("FTOL", None)
    except Exception:
        value = None
    return None if value is None else float(value)


def build_geometry_autodiff_context(
    input_path: str | Path,
    *,
    param_family: str,
    param_m: int,
    param_n: int,
    mboz: int = 12,
    nboz: int = 12,
    surface_s: Sequence[float] = (0.25, 0.5, 0.75),
) -> GeometryAutodiffContext:
    vmec_jax = _import_vmec_jax()
    booz_api = _import_booz_xform_jax_api()

    vmec_input = Path(input_path).expanduser().resolve()
    cfg, indata = vmec_jax.load_input(str(vmec_input))
    static = vmec_jax.build_static(cfg)
    boundary = vmec_jax.boundary_from_indata(indata, static.modes)

    kind = _boundary_kind_for_family(param_family)
    m_arr = jnp.asarray(static.modes.m)
    n_arr = jnp.asarray(static.modes.n)
    matches = jnp.where((m_arr == int(param_m)) & (n_arr == int(param_n)), size=2, fill_value=-1)[0]
    match_indices = [int(idx) for idx in np.asarray(matches) if int(idx) >= 0]
    if not match_indices:
        raise ValueError(
            f"Could not find a {param_family} coefficient with (m, n)=({param_m}, {param_n}) in {vmec_input}."
        )
    if len(match_indices) > 1:
        raise ValueError(
            f"Found multiple matches for {param_family}({param_m}, {param_n}); expected exactly one."
        )
    boundary_index = int(match_indices[0])

    try:
        prepare_fixed_boundary_context = _resolve_vmec_attr(vmec_jax, "prepare_fixed_boundary_context")
        fixed_context_obj = prepare_fixed_boundary_context(
            static=static,
            indata=indata,
            boundary=boundary,
            vmec_project=False,
        )
        fixed_context = {
            "signgs": int(fixed_context_obj.signgs),
            "flux": fixed_context_obj.flux,
            "pressure": jnp.asarray(fixed_context_obj.pressure),
            "booz_inputs": fixed_context_obj.booz_inputs,
        }
    except Exception:
        fixed_context = _build_vmec_fixed_context(vmec_jax, static=static, indata=indata, boundary=boundary)
    try:
        surface_indices_from_static = _resolve_vmec_attr(vmec_jax, "surface_indices_from_static")
        surface_indices, _ = surface_indices_from_static(static, list(surface_s))
    except Exception:
        s_half = 0.5 * (np.asarray(static.s[:-1], dtype=float) + np.asarray(static.s[1:], dtype=float))
        surface_indices = [int(np.argmin(np.abs(s_half - float(val)))) for val in surface_s]
    booz_constants, booz_grids = booz_api.prepare_booz_xform_constants_from_inputs(
        inputs=fixed_context["booz_inputs"],
        mboz=int(mboz),
        nboz=int(nboz),
        asym=bool(cfg.lasym),
    )

    boundary_array = jnp.asarray(getattr(boundary, _boundary_array_name_for_kind(kind)))

    return GeometryAutodiffContext(
        input_path=vmec_input,
        param_family=str(param_family).strip().upper(),
        param_m=int(param_m),
        param_n=int(param_n),
        cfg=cfg,
        indata=indata,
        static=static,
        boundary=boundary,
        boundary_kind=kind,
        boundary_index=boundary_index,
        signgs=int(fixed_context["signgs"]),
        flux=fixed_context["flux"],
        pressure=jnp.asarray(fixed_context["pressure"]),
        surface_s=tuple(float(val) for val in surface_s),
        surface_indices=jnp.asarray(surface_indices, dtype=jnp.int32),
        mboz=int(mboz),
        nboz=int(nboz),
        booz_constants=booz_constants,
        booz_grids=booz_grids,
        baseline_coefficient=float(boundary_array[boundary_index]),
        vmec_default_max_iter=_vmec_default_max_iter_from_indata(indata),
        vmec_default_step_size=_vmec_default_step_size_from_indata(indata),
        vmec_default_ftol=_vmec_default_ftol_from_indata(indata),
    )


def _solve_state_for_single_param(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> Any:
    vmec_jax = _import_vmec_jax()
    initial_guess_from_boundary = _resolve_vmec_attr(vmec_jax, "initial_guess_from_boundary", submodule="init_guess")
    solve_fixed_boundary_residual_iter = _resolve_vmec_attr(vmec_jax, "solve_fixed_boundary_residual_iter", submodule="solve")

    lane_key = str(lane).strip().lower()
    if lane_key not in {"forward", "ad"}:
        raise ValueError("lane must be 'forward' or 'ad'.")

    max_iter_value = int(context.vmec_default_max_iter if max_iter is None else max_iter)
    step_size_value = float(context.vmec_default_step_size if step_size is None else step_size)

    boundary_field = _boundary_array_name_for_kind(context.boundary_kind)
    boundary_array = jnp.asarray(getattr(context.boundary, boundary_field), dtype=jnp.float64)
    boundary_array = boundary_array.at[int(context.boundary_index)].add(jnp.asarray(param_delta, dtype=jnp.float64))
    boundary = dataclasses.replace(context.boundary, **{boundary_field: boundary_array})
    state0 = initial_guess_from_boundary(context.static, boundary, context.indata, vmec_project=True)

    if lane_key == "forward":
        result = solve_fixed_boundary_residual_iter(
            state0,
            context.static,
            indata=context.indata,
            signgs=int(context.signgs),
            ftol=context.vmec_default_ftol,
            max_iter=max_iter_value,
            step_size=step_size_value,
            vmec2000_control=True,
            strict_update=True,
            backtracking=True,
            limit_dt_from_force=True,
            limit_update_rms=True,
            verbose=False,
            verbose_vmec2000_table=False,
            jit_forces="auto",
            use_scan=True,
        )
        return result.state

    # AD lane: keep the implicit residual solve separate so we can add a
    # reverse-mode path later without changing the forward lane contract.
    del jacobian_penalty
    return vmec_jax.implicit.solve_fixed_boundary_state_implicit_vmec_residual(
        state0,
        context.static,
        indata=context.indata,
        signgs=int(context.signgs),
        max_iter=max_iter_value,
        step_size=step_size_value,
        ftol=context.vmec_default_ftol,
        edge_Rcos=state0.Rcos[-1, :],
        edge_Rsin=state0.Rsin[-1, :],
        edge_Zcos=state0.Zcos[-1, :],
        edge_Zsin=state0.Zsin[-1, :],
    )


def _find_mode_index(ixm_b: jnp.ndarray, ixn_b: jnp.ndarray, *, m: int, n: int) -> int | None:
    matches = jnp.where((ixm_b == int(m)) & (ixn_b == int(n)), size=1, fill_value=-1)[0]
    match = int(matches[0])
    return None if match < 0 else match


def solve_geometry_state_forward(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    max_iter: int | None = None,
    step_size: float | None = None,
):
    return _solve_state_for_single_param(
        context,
        param_delta,
        lane="forward",
        max_iter=max_iter,
        step_size=step_size,
    )


def solve_geometry_state_ad(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    max_iter: int | None = None,
    step_size: float | None = None,
):
    return _solve_state_for_single_param(
        context,
        param_delta,
        lane="ad",
        max_iter=max_iter,
        step_size=step_size,
    )


def vmec_scalar_observables_from_single_param(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> dict[str, jnp.ndarray]:
    vmec_jax = _import_vmec_jax()
    eval_geom = _resolve_vmec_attr(vmec_jax, "eval_geom", submodule="geom")
    volume_from_sqrtg = _resolve_vmec_attr(vmec_jax, "volume_from_sqrtg", submodule="integrals")
    equilibrium_iota_profiles_from_state = _resolve_vmec_attr(
        vmec_jax,
        "equilibrium_iota_profiles_from_state",
        submodule="profiles",
    )
    equilibrium_aspect_ratio_from_state = _resolve_vmec_attr(
        vmec_jax,
        "equilibrium_aspect_ratio_from_state",
        submodule="profiles",
    )

    state = _solve_state_for_single_param(
        context,
        param_delta,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    geom = eval_geom(state, context.static)
    _dvds, volume = volume_from_sqrtg(
        geom.sqrtg,
        context.static.s,
        context.static.grid.theta,
        context.static.grid.zeta,
        nfp=int(context.cfg.nfp),
    )
    _chips, _iotas, iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=int(context.signgs),
    )
    iota_mean = jnp.mean(iotaf[1:]) if int(iotaf.size) > 1 else iotaf[0]
    return {
        "aspect_ratio": jnp.asarray(equilibrium_aspect_ratio_from_state(state=state, static=context.static)),
        "volume_total": jnp.asarray(volume[-1]),
        "iota_mean": jnp.asarray(iota_mean),
        "edge_r00": jnp.asarray(state.Rcos[-1, 0]),
    }


def geometry_observables_from_single_param(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> dict[str, jnp.ndarray]:
    vmec_jax = _import_vmec_jax()
    booz_api = _import_booz_xform_jax_api()

    state = _solve_state_for_single_param(
        context,
        param_delta,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    inputs = vmec_jax.booz_xform_inputs_from_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=context.signgs,
        flux=context.flux,
    )
    out = booz_api.booz_xform_from_inputs(
        inputs=inputs,
        constants=context.booz_constants,
        grids=context.booz_grids,
        surface_indices=context.surface_indices,
        jit=True,
    )

    bmnc_b = jnp.asarray(out["bmnc_b"])
    ixm_b = jnp.asarray(out["ixm_b"], dtype=jnp.int32)
    ixn_b = jnp.asarray(out["ixn_b"], dtype=jnp.int32)
    nfp = int(jnp.asarray(out["nfp_b"]).reshape(()))

    mode00 = _find_mode_index(ixm_b, ixn_b, m=0, n=0)
    if mode00 is None:
        raise ValueError("Boozer output is missing the (m, n) = (0, 0) mode.")
    mode10 = _find_mode_index(ixm_b, ixn_b, m=1, n=0)

    b00 = bmnc_b[:, mode00]
    observables = {
        "iota_b": jnp.asarray(out["iota_b"]),
        "b00": b00,
        "buco_b": jnp.asarray(out["buco_b"]),
        "bvco_b": jnp.asarray(out["bvco_b"]),
    }
    if mode10 is not None:
        b10 = bmnc_b[:, mode10]
        observables["b10"] = b10
        observables["b10_over_b00"] = b10 / b00
    observables["aspect_proxy"] = jnp.asarray(state.Rcos[-1, mode00])
    observables["surface_indices"] = context.surface_indices.astype(jnp.float64)
    observables["nfp"] = jnp.asarray([float(nfp)], dtype=jnp.float64)
    return observables


def vmec_booz_scalar_observables_from_single_param(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> dict[str, jnp.ndarray]:
    observables = geometry_observables_from_single_param(
        context,
        param_delta,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    reduced = {
        "iota_b_mean": jnp.mean(jnp.asarray(observables["iota_b"])),
        "b00_mean": jnp.mean(jnp.asarray(observables["b00"])),
        "buco_b_mean": jnp.mean(jnp.asarray(observables["buco_b"])),
        "bvco_b_mean": jnp.mean(jnp.asarray(observables["bvco_b"])),
        "aspect_proxy": jnp.asarray(observables["aspect_proxy"]),
    }
    if "b10_over_b00" in observables:
        reduced["b10_over_b00_mean"] = jnp.mean(jnp.asarray(observables["b10_over_b00"]))
    return reduced


def _safe_divide(num, den):
    num_arr = jnp.asarray(num)
    den_arr = jnp.asarray(den)
    den_safe = jnp.where(jnp.abs(den_arr) > 0.0, den_arr, 1.0)
    return jnp.where(jnp.abs(den_arr) > 0.0, num_arr / den_safe, 0.0)


def _safe_reciprocal(values):
    arr = jnp.asarray(values)
    return jnp.where(jnp.abs(arr) > 0.0, 1.0 / arr, 0.0)


def _find_boozer_mode_index(ixm_b, ixn_b, *, m_value: int, n_value: int) -> int | None:
    matches = (jnp.asarray(ixm_b) == int(m_value)) & (jnp.asarray(ixn_b) == int(n_value))
    if not bool(jnp.any(matches)):
        return None
    return int(jnp.argmax(matches))


def _surface_indices_for_s_values(static, s_values: Sequence[float]):
    vmec_jax = _import_vmec_jax()
    try:
        surface_indices_from_static = _resolve_vmec_attr(vmec_jax, "surface_indices_from_static")
        surface_indices, _ = surface_indices_from_static(static, list(s_values))
        return jnp.asarray(surface_indices, dtype=jnp.int32)
    except Exception:
        s_half = 0.5 * (np.asarray(static.s[:-1], dtype=float) + np.asarray(static.s[1:], dtype=float))
        surface_indices = [int(np.argmin(np.abs(s_half - float(val)))) for val in s_values]
        return jnp.asarray(surface_indices, dtype=jnp.int32)


def _build_neopax_geometry_from_state(
    context: GeometryAutodiffContext,
    state,
    *,
    n_r: int,
):
    from NEOPAX._geometry_models import VmecBoozer
    from vmec_jax.energy import flux_profiles_from_indata
    from vmec_jax.integrals import cumrect_s_halfmesh
    from vmec_jax.vmec_forces import vmec_forces_rz_from_wout
    from vmec_jax.vmec_residue import vmec_force_norms_from_bcovar_dynamic

    rho_grid = jnp.linspace(0.0, 1.0, int(n_r))
    if int(n_r) > 1:
        rho_grid_half = jnp.concatenate(
            [
                jnp.array([0.0], dtype=rho_grid.dtype),
                0.5 * (rho_grid[:-1] + rho_grid[1:]),
                jnp.array([1.0], dtype=rho_grid.dtype),
            ]
        )
    else:
        rho_grid_half = jnp.array([0.0, 1.0], dtype=rho_grid.dtype)
    sample_rho = rho_grid[1:-1]

    wout_like = type(
        "WoutLike",
        (),
        {
            "nfp": int(context.static.cfg.nfp),
            "mpol": int(context.static.cfg.mpol),
            "ntor": int(context.static.cfg.ntor),
            "lasym": bool(context.static.cfg.lasym),
            "signgs": int(context.signgs),
        },
    )()
    kernels = vmec_forces_rz_from_wout(
        state=state,
        static=context.static,
        wout=wout_like,
        indata=context.indata,
        use_vmec_synthesis=True,
        trig=context.static.trig_vmec,
    )
    norms = vmec_force_norms_from_bcovar_dynamic(
        bc=kernels.bc,
        trig=context.static.trig_vmec,
        s=jnp.asarray(context.static.s),
        signgs=int(context.signgs),
    )
    volume_p = jnp.abs(jnp.asarray(norms.volume)) * (4.0 * jnp.pi**2)
    vp = jnp.abs(jnp.asarray(norms.vp))
    s_full = jnp.asarray(context.static.s)
    rho_half = jnp.concatenate(
        [jnp.zeros((1,), dtype=s_full.dtype), jnp.sqrt(jnp.maximum(0.5 * (s_full[1:] + s_full[:-1]), 0.0))],
        axis=0,
    )

    flux = flux_profiles_from_indata(context.indata, s_full, signgs=int(context.signgs))
    phi = cumrect_s_halfmesh(jnp.asarray(flux.phipf), s_full)
    psia = jnp.abs(phi[-1])

    r0_value = jnp.asarray(state.Rcos)[-1, 0]
    a_b = jnp.sqrt(volume_p / (2.0 * jnp.pi**2 * r0_value))
    r_grid = rho_grid * a_b
    r_grid_half = rho_grid_half * a_b
    dr = r_grid[1] - r_grid[0] if int(n_r) > 1 else jnp.asarray(0.0, dtype=r_grid.dtype)

    dVdr = interpax.Interpolator1D(rho_half[1:], jnp.asarray(vp)[1:], extrap=True)
    volume_scale = (2.0 * jnp.pi) ** 2
    vprime = dVdr(rho_grid) * 2.0 * rho_grid / a_b * volume_scale
    vprime_half = dVdr(rho_grid_half) * 2.0 * rho_grid_half / a_b * volume_scale
    over_vprime = _safe_reciprocal(vprime).at[0].set(0.0)

    booz_api = _import_booz_xform_jax_api()
    vmec_jax = _import_vmec_jax()
    inputs = vmec_jax.booz_xform_inputs_from_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=context.signgs,
        flux=context.flux,
    )
    out = booz_api.booz_xform_from_inputs(
        inputs=inputs,
        constants=context.booz_constants,
        grids=context.booz_grids,
        surface_indices=_surface_indices_for_s_values(
            context.static,
            tuple(float(rho_value**2) for rho_value in sample_rho),
        ),
        jit=True,
    )
    bmnc_b = jnp.asarray(out["bmnc_b"])
    ixm_b = jnp.asarray(out["ixm_b"], dtype=jnp.int32)
    ixn_b = jnp.asarray(out["ixn_b"], dtype=jnp.int32)
    mode00 = _find_boozer_mode_index(ixm_b, ixn_b, m_value=0, n_value=0)
    if mode00 is None:
        raise ValueError("Boozer output is missing the (0,0) mode.")
    mode10 = _find_boozer_mode_index(ixm_b, ixn_b, m_value=1, n_value=0)

    iota_samples = jnp.asarray(out["iota_b"])
    i_value_samples = jnp.asarray(out["buco_b"])
    g_value_samples = jnp.asarray(out["bvco_b"])
    b0_samples = bmnc_b[:, mode00]
    if mode10 is None:
        b10_samples = jnp.zeros_like(b0_samples)
    else:
        b10_samples = _safe_divide(bmnc_b[:, mode10], b0_samples)

    b0_surface = jnp.concatenate([b0_samples[:1], b0_samples, b0_samples[-1:]], axis=0)
    b10_surface = jnp.concatenate(
        [jnp.zeros((1,), dtype=b10_samples.dtype), b10_samples, b10_samples[-1:]],
        axis=0,
    )
    iota_surface = jnp.concatenate(
        [jnp.zeros((1,), dtype=iota_samples.dtype), iota_samples, iota_samples[-1:]],
        axis=0,
    )
    i_value_surface = jnp.concatenate(
        [jnp.zeros((1,), dtype=i_value_samples.dtype), i_value_samples, i_value_samples[-1:]],
        axis=0,
    )
    g_value_surface = jnp.concatenate(
        [jnp.zeros((1,), dtype=g_value_samples.dtype), g_value_samples, g_value_samples[-1:]],
        axis=0,
    )

    b0_interp = interpax.Interpolator1D(rho_grid, b0_surface, extrap=True)
    b10_interp = interpax.Interpolator1D(rho_grid, b10_surface, extrap=True)
    iota_interp = interpax.Interpolator1D(rho_grid, iota_surface, extrap=True)
    i_interp = interpax.Interpolator1D(rho_grid, i_value_surface, extrap=True)
    g_interp = interpax.Interpolator1D(rho_grid, g_value_surface, extrap=True)

    b0 = b0_interp(rho_grid)
    b_10 = b10_interp(rho_grid)
    iota = iota_interp(rho_grid)
    i_value = i_interp(rho_grid)
    g_value = g_interp(rho_grid)
    epsilon_t = rho_grid * a_b / r0_value
    curvature = _safe_divide(jnp.abs(b_10), epsilon_t).at[0].set(0.0)
    enlogation = jnp.square(_safe_divide(epsilon_t, b_10)).at[0].set(0.0)
    b0prime = jax.vmap(jax.grad(lambda r: b0_interp(r)))(r_grid)
    sqrtg00_value = _safe_divide(g_value + iota * i_value, jnp.maximum(jnp.square(b0), 1.0e-30))
    bsqav = _safe_divide(g_value + iota * i_value, sqrtg00_value * jnp.maximum(jnp.square(b0), 1.0e-30))
    iota_safe = jnp.where(jnp.abs(iota) > 0.0, jnp.abs(iota), 1.0)
    g_ps = (
        1.5
        * (4.0 / 3.0)
        * jnp.square(curvature / iota_safe)
        * (
            1.0
            + 3.4229 * jnp.power(epsilon_t, 3.6) * (1.0 - 2.5766 * jnp.power(jnp.abs(iota), 1.6))
            - 0.6039 * jnp.power(epsilon_t, 2.0) * (jnp.square(curvature) - 1.0)
        )
    )

    return VmecBoozer(
        n_r=int(n_r),
        a_b=a_b,
        Psia_value=psia,
        rho_grid=rho_grid,
        rho_grid_half=rho_grid_half,
        r_grid=r_grid,
        r_grid_half=r_grid_half,
        full_grid_indices=jnp.arange(int(n_r)),
        dr=dr,
        Vprime=vprime,
        Vprime_half=vprime_half,
        overVprime=over_vprime,
        epsilon_t=epsilon_t,
        B0=b0,
        B_10=b_10,
        enlogation=enlogation,
        iota=iota,
        R0=r0_value,
        B0prime=b0prime,
        curvature=curvature,
        G_PS=g_ps,
        sqrtg00_value=sqrtg00_value,
        Bsqav=bsqav,
        I_value=i_value,
        G_value=g_value,
    )


def _build_ntx_runtime_channels_from_surfaces(surfaces, *, rho, a_b, psia):
    from NEOPAX._transport_flux_models import NTXRuntimeScanChannels

    rho_arr = jnp.asarray(rho, dtype=jnp.float64)
    psia_value = float(jnp.asarray(psia))
    b00 = jnp.asarray([jnp.asarray(surface.b0 if surface.b0 is not None else surface.b_cos[0], dtype=jnp.float64) for surface in surfaces])
    boozer_i = jnp.asarray([jnp.asarray(surface.b_theta, dtype=jnp.float64) for surface in surfaces])
    boozer_g = jnp.asarray([jnp.asarray(surface.b_zeta, dtype=jnp.float64) for surface in surfaces])
    iota = jnp.asarray([jnp.asarray(surface.iota, dtype=jnp.float64) for surface in surfaces])
    drds = jnp.where(rho_arr > 0.0, jnp.asarray(a_b, dtype=jnp.float64) / (2.0 * rho_arr), 0.0)
    dpsi_drtilde = rho_arr * jnp.asarray(a_b, dtype=jnp.float64) * b00
    dr_tildedr = 2.0 * psia_value / (jnp.asarray(a_b, dtype=jnp.float64) ** 2 * b00)
    dr_tildeds = dr_tildedr * drds
    denom = boozer_g + iota * boozer_i
    fac_reference_to_sfincs_11 = 8.0 * denom * b00 * psia_value**2 / (jnp.sqrt(jnp.pi) * boozer_g**2)
    fac_reference_to_sfincs_31 = 4.0 * b00 * psia_value / (jnp.sqrt(jnp.pi) * boozer_g)
    fac_reference_to_sfincs_33 = -2.0 * b00 / (denom * jnp.sqrt(jnp.pi))
    fac_sfincs_to_dkes_11 = 1.0 / (8.0 * denom * dpsi_drtilde**2 / (boozer_g**2 * b00 * jnp.sqrt(jnp.pi)))
    fac_sfincs_to_dkes_31 = 1.0 / (4.0 * dpsi_drtilde / (boozer_g * jnp.sqrt(jnp.pi)))
    fac_sfincs_to_dkes_33 = 1.0 / (-2.0 * b00 / (denom * jnp.sqrt(jnp.pi)))
    return NTXRuntimeScanChannels(
        rho=rho_arr,
        a_b=float(a_b),
        psia=psia_value,
        b00=b00,
        r00=jnp.ones_like(rho_arr),
        boozer_i=boozer_i,
        boozer_g=boozer_g,
        iota=iota,
        drds=drds,
        dr_tildedr=dr_tildedr,
        dr_tildeds=dr_tildeds,
        fac_reference_to_sfincs_11=fac_reference_to_sfincs_11,
        fac_reference_to_sfincs_31=fac_reference_to_sfincs_31,
        fac_reference_to_sfincs_33=fac_reference_to_sfincs_33,
        fac_sfincs_to_dkes_11=fac_sfincs_to_dkes_11,
        fac_sfincs_to_dkes_31=fac_sfincs_to_dkes_31,
        fac_sfincs_to_dkes_33=fac_sfincs_to_dkes_33,
        fac_dkes_to_d11star=jnp.ones_like(rho_arr),
        fac_dkes_to_d31star=jnp.ones_like(rho_arr),
        fac_dkes_to_d33star=jnp.ones_like(rho_arr),
    )


def build_ntx_exact_lij_support_from_vmec_state(
    context: GeometryAutodiffContext,
    state,
    geometry,
    *,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
):
    _ensure_local_stack_on_path()
    ntx_src = _repo_root() / "NTX" / "src"
    ntx_src_str = str(ntx_src)
    if ntx_src.exists() and ntx_src_str not in sys.path:
        sys.path.insert(0, ntx_src_str)
    import ntx
    from NEOPAX._transport_flux_models import NTXExactLijRuntimeSupport

    rho_center = jnp.asarray(geometry.r_grid, dtype=jnp.float64) / jnp.asarray(geometry.a_b, dtype=jnp.float64)
    rho_face = jnp.asarray(geometry.r_grid_half, dtype=jnp.float64) / jnp.asarray(geometry.a_b, dtype=jnp.float64)
    center_surfaces = ntx.surfaces_from_vmec_jax_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=context.signgs,
        s_values=tuple(float(rho_value**2) for rho_value in np.asarray(rho_center, dtype=float)),
        mboz=int(context.mboz),
        nboz=int(context.nboz),
        psi_p=float(jnp.asarray(geometry.Psia_value)),
    )
    face_surfaces = ntx.surfaces_from_vmec_jax_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=context.signgs,
        s_values=tuple(float(rho_value**2) for rho_value in np.asarray(rho_face, dtype=float)),
        mboz=int(context.mboz),
        nboz=int(context.nboz),
        psi_p=float(jnp.asarray(geometry.Psia_value)),
    )
    grid_spec = ntx.GridSpec(n_theta=int(n_theta), n_zeta=int(n_zeta), n_xi=int(n_xi))

    def _stack_optional(*values):
        first = values[0]
        if first is None:
            return None
        return jnp.stack([jnp.asarray(value) for value in values], axis=0)

    center_prepared_tuple = tuple(ntx.prepare_monoenergetic_system(surface, grid_spec) for surface in center_surfaces)
    face_prepared_tuple = tuple(ntx.prepare_monoenergetic_system(surface, grid_spec) for surface in face_surfaces)
    center_prepared = jax.tree_util.tree_map(_stack_optional, *center_prepared_tuple)
    face_prepared = jax.tree_util.tree_map(_stack_optional, *face_prepared_tuple)
    return NTXExactLijRuntimeSupport(
        center_channels=_build_ntx_runtime_channels_from_surfaces(
            center_surfaces,
            rho=rho_center,
            a_b=geometry.a_b,
            psia=geometry.Psia_value,
        ),
        face_channels=_build_ntx_runtime_channels_from_surfaces(
            face_surfaces,
            rho=rho_face,
            a_b=geometry.a_b,
            psia=geometry.Psia_value,
        ),
        center_prepared=center_prepared,
        face_prepared=face_prepared,
        grid=grid_spec,
    )


def build_neopax_geometry_from_single_param(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    lane: str = "ad",
    n_r: int,
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
):
    state = _solve_state_for_single_param(
        context,
        param_delta,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    return _build_neopax_geometry_from_state(context, state, n_r=n_r)


def build_runtime_context_for_geometry_param(
    config: dict[str, Any],
    context: GeometryAutodiffContext,
    param_delta,
    *,
    lane: str = "ad",
    n_r: int,
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
):
    from NEOPAX._orchestrator import (
        Models,
        RuntimeContext,
        _apply_configured_er_dirichlet_boundaries,
        _build_database,
        _build_energy_grid,
        _build_flux_model,
        _build_species,
        _build_state,
        _maybe_initialize_er_from_ambipolarity,
        _normalize_solver_config,
    )
    from NEOPAX._source_models import build_source_models_from_config

    state_vmec = _solve_state_for_single_param(
        context,
        param_delta,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    geometry = _build_neopax_geometry_from_state(context, state_vmec, n_r=n_r)
    species = _build_species(config)
    energy_grid = _build_energy_grid(config)
    database = _build_database(config, geometry)
    state = _build_state(config, geometry, species)
    config_eff = deepcopy(config)
    config_eff["geometry"] = dict(config_eff.get("geometry", {}))
    config_eff["geometry"]["vmec_file"] = None
    config_eff["geometry"]["boozer_file"] = None
    neoclassical_cfg = dict(config_eff.get("neoclassical", {}))
    if str(neoclassical_cfg.get("flux_model", "")).strip().lower() == "ntx_exact_lij_runtime":
        neoclassical_cfg["ntx_exact_lij_support"] = build_ntx_exact_lij_support_from_vmec_state(
            context,
            state_vmec,
            geometry,
            n_theta=int(neoclassical_cfg.get("ntx_exact_n_theta", 25)),
            n_zeta=int(neoclassical_cfg.get("ntx_exact_n_zeta", 25)),
            n_xi=int(neoclassical_cfg.get("ntx_exact_n_xi", 64)),
        )
        neoclassical_cfg["preload_support"] = False
        neoclassical_cfg["vmec_file"] = None
        neoclassical_cfg["boozer_file"] = None
        config_eff["neoclassical"] = neoclassical_cfg
    solver_cfg = _normalize_solver_config(config_eff)
    source_models = build_source_models_from_config(config, species)
    models = Models(
        flux=_build_flux_model(config_eff, species, energy_grid, geometry, database, source_models=source_models),
        source=source_models,
    )
    runtime = RuntimeContext(
        species=species,
        energy_grid=energy_grid,
        geometry=geometry,
        database=database,
        solver_parameters=solver_cfg,
        models=models,
    )
    mode = str(config_eff.get("general", {}).get("mode", config_eff.get("mode", "transport"))).strip().lower()
    if mode != "ambipolarity":
        state = _maybe_initialize_er_from_ambipolarity(config_eff, runtime, state)
    state = _apply_configured_er_dirichlet_boundaries(config_eff, state)
    return runtime, state


def central_fd_single_param(
    func,
    h: float,
):
    minus = func(-h)
    plus = func(h)
    return jax.tree_util.tree_map(lambda p, m: (p - m) / (2.0 * h), plus, minus), minus, plus


def five_point_fd_single_param(
    func,
    h: float,
    *,
    minus=None,
    plus=None,
):
    minus2 = func(-2.0 * h)
    plus2 = func(2.0 * h)
    if minus is None:
        minus = func(-h)
    if plus is None:
        plus = func(h)
    return jax.tree_util.tree_map(
        lambda p2, p1, m1, m2: (-p2 + 8.0 * p1 - 8.0 * m1 + m2) / (12.0 * h),
        plus2,
        plus,
        minus,
        minus2,
    )


def rel_error(lhs, rhs, *, floor: float = 1.0e-14) -> float:
    lhs_arr = jnp.asarray(lhs, dtype=jnp.float64).reshape(-1)
    rhs_arr = jnp.asarray(rhs, dtype=jnp.float64).reshape(-1)
    numer = float(jnp.linalg.norm(lhs_arr - rhs_arr))
    denom = max(float(jnp.linalg.norm(rhs_arr)), float(floor))
    return numer / denom
