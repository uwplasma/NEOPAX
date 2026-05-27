from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import interpax


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
    spec: Any
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
    try:
        boundary_param_specs = vmec_jax.boundary_param_specs
    except AttributeError:
        from vmec_jax.optimization import boundary_param_specs

    specs = boundary_param_specs(
        boundary,
        static.modes,
        include=(kind,),
        fix=(),
        include_axis=True,
    )
    matches = [spec for spec in specs if int(spec.m) == int(param_m) and int(spec.n) == int(param_n) and spec.kind == kind]
    if not matches:
        raise ValueError(
            f"Could not find a {param_family} coefficient with (m, n)=({param_m}, {param_n}) in {vmec_input}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple matches for {param_family}({param_m}, {param_n}); expected exactly one."
        )
    spec = matches[0]

    fixed_context = vmec_jax.prepare_fixed_boundary_context(
        static=static,
        indata=indata,
        boundary=boundary,
        vmec_project=False,
    )
    surface_indices, _ = vmec_jax.surface_indices_from_static(static, list(surface_s))
    booz_constants, booz_grids = booz_api.prepare_booz_xform_constants_from_inputs(
        inputs=fixed_context.booz_inputs,
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
        spec=spec,
        signgs=int(fixed_context.signgs),
        flux=fixed_context.flux,
        pressure=jnp.asarray(fixed_context.pressure),
        surface_s=tuple(float(val) for val in surface_s),
        surface_indices=jnp.asarray(surface_indices, dtype=jnp.int32),
        mboz=int(mboz),
        nboz=int(nboz),
        booz_constants=booz_constants,
        booz_grids=booz_grids,
        baseline_coefficient=float(boundary_array[spec.index]),
    )


def _solve_state_for_single_param(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    max_iter: int = 2,
    step_size: float = 5.0e-3,
    jacobian_penalty: float = 1.0e3,
) -> Any:
    vmec_jax = _import_vmec_jax()
    params = jnp.asarray([param_delta], dtype=jnp.float64)
    try:
        apply_boundary_params = vmec_jax.apply_boundary_params
    except AttributeError:
        from vmec_jax.optimization import apply_boundary_params

    boundary = apply_boundary_params(context.boundary, (context.spec,), params)
    return vmec_jax.solve_fixed_boundary_from_boundary(
        boundary=boundary,
        static=context.static,
        indata=context.indata,
        flux=context.flux,
        pressure=context.pressure,
        signgs=context.signgs,
        max_iter=int(max_iter),
        step_size=float(step_size),
        jacobian_penalty=float(jacobian_penalty),
        differentiable=True,
        stop_grad_in_update=False,
        verbose=False,
        vmec_project=False,
    )


def _find_mode_index(ixm_b: jnp.ndarray, ixn_b: jnp.ndarray, *, m: int, n: int) -> int | None:
    matches = jnp.where((ixm_b == int(m)) & (ixn_b == int(n)), size=1, fill_value=-1)[0]
    match = int(matches[0])
    return None if match < 0 else match


def geometry_observables_from_single_param(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    max_iter: int = 2,
    step_size: float = 5.0e-3,
    jacobian_penalty: float = 1.0e3,
) -> dict[str, jnp.ndarray]:
    vmec_jax = _import_vmec_jax()
    booz_api = _import_booz_xform_jax_api()

    state = _solve_state_for_single_param(
        context,
        param_delta,
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


def build_neopax_geometry_from_single_param(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    n_r: int,
    max_iter: int = 2,
    step_size: float = 5.0e-3,
    jacobian_penalty: float = 1.0e3,
):
    from NEOPAX._geometry_models import VmecBoozer
    from vmec_jax.energy import flux_profiles_from_indata
    from vmec_jax.integrals import cumrect_s_halfmesh
    from vmec_jax.vmec_forces import vmec_forces_rz_from_wout
    from vmec_jax.vmec_residue import vmec_force_norms_from_bcovar_dynamic

    state = _solve_state_for_single_param(
        context,
        param_delta,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )

    rho_grid = jnp.linspace(0.0, 1.0, int(n_r))
    rho_grid_half0 = 0.5 * (rho_grid[0] + rho_grid[1]) if int(n_r) > 1 else jnp.asarray(0.0, dtype=rho_grid.dtype)
    rho_grid_half = jnp.linspace(rho_grid_half0, rho_grid_half0 + rho_grid[-1], int(n_r))
    sample_rho = rho_grid[1:-1]

    observables = geometry_observables_from_single_param(
        context,
        param_delta,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )

    # Build volume and flux profiles directly from the solved state.
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
    if int(n_r) > 1:
        r_grid = r_grid.at[0].set(0.5 * r_grid[1])
    r_grid_half = rho_grid_half * a_b
    dr = r_grid[2] - r_grid[1] if int(n_r) > 2 else jnp.asarray(0.0, dtype=r_grid.dtype)

    dVdr = interpax.Interpolator1D(rho_half[1:], jnp.asarray(vp)[1:], extrap=True)
    vprime = dVdr(rho_grid) * 2.0 * rho_grid / a_b
    vprime_half = dVdr(rho_grid_half) * 2.0 * rho_grid_half / a_b
    over_vprime = _safe_reciprocal(vprime).at[0].set(0.0)

    # Interpolate Boozer radial profiles from half-mesh surfaces onto the NEOPAX grid.
    s_values = tuple(float(rho_value**2) for rho_value in sample_rho)
    booz_outputs = geometry_observables_from_single_param(
        context,
        param_delta,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    del booz_outputs  # keep explicit naming below for clarity

    vmec_jax = _import_vmec_jax()
    booz_api = _import_booz_xform_jax_api()
    inputs = vmec_jax.booz_xform_inputs_from_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=context.signgs,
        flux=context.flux,
    )
    surface_indices, _ = vmec_jax.surface_indices_from_static(context.static, list(s_values))
    out = booz_api.booz_xform_from_inputs(
        inputs=inputs,
        constants=context.booz_constants,
        grids=context.booz_grids,
        surface_indices=jnp.asarray(surface_indices, dtype=jnp.int32),
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


def build_runtime_context_for_geometry_param(
    config: dict[str, Any],
    context: GeometryAutodiffContext,
    param_delta,
    *,
    n_r: int,
    max_iter: int = 2,
    step_size: float = 5.0e-3,
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

    geometry = build_neopax_geometry_from_single_param(
        context,
        param_delta,
        n_r=n_r,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    species = _build_species(config)
    energy_grid = _build_energy_grid(config)
    database = _build_database(config, geometry)
    state = _build_state(config, geometry, species)
    solver_cfg = _normalize_solver_config(config)
    source_models = build_source_models_from_config(config, species)
    models = Models(
        flux=_build_flux_model(config, species, energy_grid, geometry, database, source_models=source_models),
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
    mode = str(config.get("general", {}).get("mode", config.get("mode", "transport"))).strip().lower()
    if mode != "ambipolarity":
        state = _maybe_initialize_er_from_ambipolarity(config, runtime, state)
    state = _apply_configured_er_dirichlet_boundaries(config, state)
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
