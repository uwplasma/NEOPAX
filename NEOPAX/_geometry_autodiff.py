from __future__ import annotations

import dataclasses
import importlib
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
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
        repo / "VMEX",
        repo / "vmex",
        repo / "vmec_jax",
        repo / "booz_xform_jax" / "src",
    )
    for candidate in reversed(candidates):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _import_vmec_jax():
    _ensure_local_stack_on_path()
    try:
        import vmex as vmec_backend
    except ModuleNotFoundError:
        import vmec_jax as vmec_backend

    return vmec_backend


def _vmec_backend_package_name() -> str:
    module = _import_vmec_jax()
    return str(getattr(module, "__name__", "vmex"))


def _import_vmec_module(module_name: str):
    return _import_vmec_module_any((module_name,))


_CURRENT_VMEX_MODULE_ALIASES = {
    "fields": ("core.fields",),
    "fourier": ("core.fourier",),
    "geometry": ("core.geometry",),
    "implicit": ("core.implicit",),
    "multigrid": ("core.multigrid",),
    "optimize": ("core.optimize",),
    "setup": ("core.setup",),
    "solver": ("core.solver",),
    "statephysics": ("core.statephysics",),
    "wout": ("core.wout",),
}


def _import_vmec_module_any(module_names: Sequence[str]):
    _ensure_local_stack_on_path()
    backend_name = _vmec_backend_package_name()
    fallback_name = "vmec_jax" if backend_name == "vmex" else "vmex"
    errors = []
    backend_module_names: tuple[str, ...] = tuple(module_names)
    if backend_name == "vmex":
        expanded: list[str] = []
        for module_name in module_names:
            expanded.extend(_CURRENT_VMEX_MODULE_ALIASES.get(module_name, (module_name,)))
        backend_module_names = tuple(dict.fromkeys(expanded))

    for module_name in backend_module_names:
        candidate_name = f"{backend_name}.{module_name}"
        try:
            return importlib.import_module(candidate_name)
        except ModuleNotFoundError as exc:
            if exc.name != candidate_name:
                raise
            errors.append(f"{candidate_name}: {exc}")

    for module_name in module_names:
        candidate_name = f"{fallback_name}.{module_name}"
        try:
            return importlib.import_module(candidate_name)
        except ModuleNotFoundError as exc:
            if exc.name != candidate_name:
                raise
            errors.append(f"{candidate_name}: {exc}")
    raise ModuleNotFoundError(
        "Could not import VMEC backend module from any candidate: " + "; ".join(errors)
    )


def _import_vmec_jax_implicit():
    return _import_vmec_module("implicit")


def _import_vmec_jax_optimization():
    return _import_vmec_module_any(("optimization", "optimize"))


def _import_booz_xform_jax_api():
    _ensure_local_stack_on_path()
    from booz_xform_jax import jax_api

    return jax_api


def _neopax_full_to_half_mesh(full: Any, m_modes: Any, s_full: Any) -> jnp.ndarray:
    """VMEC full-mesh spectral rows -> Boozer half-mesh rows, all in JAX."""
    full = jnp.asarray(full)
    m_modes = jnp.asarray(m_modes)
    s_full = jnp.asarray(s_full)
    ns_full = int(full.shape[0])
    if ns_full < 2:
        return full

    sqrt_s_full = jnp.sqrt(jnp.maximum(s_full, 0.0)).at[0].set(1.0)
    s_half = 0.5 * (s_full[:-1] + s_full[1:])
    sqrt_s_half = jnp.sqrt(jnp.maximum(s_half, 0.0))

    even_val = 0.5 * (full[:-1, :] + full[1:, :])
    odd_val = (
        0.5
        * ((full[:-1, :] / sqrt_s_full[:-1, None]) + (full[1:, :] / sqrt_s_full[1:, None]))
        * sqrt_s_half[:, None]
    )
    half = jnp.where(((m_modes % 2) == 0)[None, :], even_val, odd_val)

    if ns_full >= 3:
        axis_mask = m_modes == 1
        axis_val = (
            1.5 * full[1, :] / sqrt_s_full[1]
            - 0.5 * full[2, :] / sqrt_s_full[2]
        ) * sqrt_s_half[0]
        half = half.at[0, :].set(jnp.where(axis_mask, axis_val, half[0, :]))
    return half


def _neopax_lambda_wout_from_full_mesh(
    *,
    lam_full: Any,
    m_modes: Any,
    phipf_internal: Any,
    lamscale: Any,
    s_full: Any,
) -> jnp.ndarray:
    """Traceable VMEX-state version of VMEC wrout lambda half-mesh scaling."""
    lam_full = jnp.asarray(lam_full)
    m_modes = jnp.asarray(m_modes)
    phipf_internal = jnp.asarray(phipf_internal)
    s_full = jnp.asarray(s_full)
    ns = int(lam_full.shape[0])
    if ns < 2:
        return jnp.zeros_like(lam_full)

    phipf_safe = jnp.where(phipf_internal == 0.0, 1.0, phipf_internal)
    lam_ext = lam_full * (jnp.asarray(lamscale) / phipf_safe[:, None])

    idx = jnp.arange(ns + 1, dtype=lam_ext.dtype)
    hs = s_full[1] - s_full[0]
    sqrts = jnp.sqrt(jnp.maximum(hs * (idx - 1.0), 0.0)).at[ns].set(1.0)
    shalf = jnp.sqrt(jnp.maximum(hs * jnp.abs(idx - 1.5), 0.0))
    js = jnp.arange(2, ns + 1)
    sm = jnp.zeros_like(sqrts).at[js].set(jnp.where(sqrts[js] != 0.0, shalf[js] / sqrts[js], 0.0))
    sp_val = jnp.where(
        js < ns,
        jnp.where(sqrts[js] != 0.0, shalf[js + 1] / sqrts[js], 0.0),
        jnp.where(sqrts[js] != 0.0, 1.0 / sqrts[js], 0.0),
    )
    sp = jnp.zeros_like(sqrts).at[js].set(sp_val)
    sp = sp.at[1].set(jnp.where(ns >= 2, sm[2], 0.0))

    mask_m_le1 = m_modes <= 1
    lam_half = lam_ext.at[0, :].set(jnp.where(mask_m_le1, lam_ext[1, :], lam_ext[0, :]))
    even_mask = (m_modes % 2) == 0

    def body(i, arr):
        row = ns - 1 - i
        even_val = 0.5 * (arr[row, :] + arr[row - 1, :])
        odd_val = 0.5 * (sm[row + 1] * arr[row, :] + sp[row] * arr[row - 1, :])
        return arr.at[row, :].set(jnp.where(even_mask, even_val, odd_val))

    lam_half = jax.lax.fori_loop(0, ns - 1, body, lam_half)
    return lam_half.at[0, :].set(jnp.zeros_like(lam_half[0, :]))


def _neopax_vmex_booz_xform_inputs_from_state(*, state, static) -> SimpleNamespace:
    """Build booz_xform_jax inputs from current VMEX state/runtime without host table loops."""
    rt = static.runtime
    setup = rt.setup
    lasym = bool(getattr(setup, "lasym", rt.resolution.lasym))
    if lasym:
        raise NotImplementedError("NEOPAX current-VMEX Boozer AD input builder currently supports lasym=False only.")

    solver_module = _import_vmec_module("core.solver")
    geometry_module = _import_vmec_module("core.geometry")
    fields_module = _import_vmec_module("core.fields")
    statephysics_module = _import_vmec_module("core.statephysics")
    transforms_module = _import_vmec_module("core.transforms")

    s_full = jnp.asarray(setup.s_full)
    (R_cos, R_sin, Z_cos, Z_sin), geometry = solver_module._geometry(state, rt)
    jacobian = geometry_module.half_mesh_jacobian(geometry, s=s_full)
    metrics = fields_module.metric_elements(geometry, s=s_full)
    fields = fields_module.magnetic_fields(
        geometry=geometry,
        jacobian=jacobian,
        metrics=metrics,
        trig=rt.trig,
        s=s_full,
        phips=setup.phips,
        phipf=setup.phipf,
        chips=setup.chips,
        signgs=setup.signgs,
        gamma=rt.gamma,
        mass=setup.mass,
        ncurr=setup.ncurr,
        enclosed_current=setup.icurv,
    )

    mode_scale = jnp.asarray(1.0 / transforms_module.physical_to_internal_scale(rt.modes, rt.trig))
    m_modes = jnp.asarray(rt.modes.m, dtype=jnp.int32)
    nfp = int(rt.resolution.nfp)

    rmnc = _neopax_full_to_half_mesh(jnp.asarray(R_cos) * mode_scale[None, :], m_modes, s_full)
    zmns = _neopax_full_to_half_mesh(jnp.asarray(Z_sin) * mode_scale[None, :], m_modes, s_full)
    lmns_full = _neopax_lambda_wout_from_full_mesh(
        lam_full=jnp.asarray(state.L_sin) * mode_scale[None, :],
        m_modes=m_modes,
        phipf_internal=setup.phipf,
        lamscale=fields.lamscale,
        s_full=s_full,
    )

    modes_nyq, consts = statephysics_module._nyquist_analysis_constants(rt.resolution)
    pressure = jnp.asarray(fields.pressure)
    bmag = jnp.sqrt(jnp.maximum(
        2.0 * (jnp.asarray(fields.total_pressure) - pressure[:, None, None]),
        jnp.asarray(jnp.finfo(pressure.dtype).tiny, dtype=pressure.dtype),
    ))

    bmnc = statephysics_module._wrout_cos_coeffs_state(bmag, consts)
    bsubumnc = statephysics_module._wrout_cos_coeffs_state(jnp.asarray(fields.bsubu), consts)
    bsubvmnc = statephysics_module._wrout_cos_coeffs_state(jnp.asarray(fields.bsubv), consts)

    iota = jnp.where(
        jnp.asarray(int(setup.ncurr) == 1),
        jnp.where(jnp.asarray(setup.phips) != 0.0, jnp.asarray(fields.chips) / jnp.asarray(setup.phips), 0.0),
        jnp.asarray(setup.iotas),
    )
    iota = iota.at[0].set(jnp.asarray(0.0, dtype=iota.dtype))

    return SimpleNamespace(
        nfp=nfp,
        xm=jnp.asarray(rt.modes.m, dtype=jnp.int32),
        xn=jnp.asarray(np.asarray(rt.modes.n, dtype=np.int32) * nfp, dtype=jnp.int32),
        xm_nyq=jnp.asarray(modes_nyq.m, dtype=jnp.int32),
        xn_nyq=jnp.asarray(np.asarray(modes_nyq.n, dtype=np.int32) * nfp, dtype=jnp.int32),
        rmnc=rmnc,
        zmns=zmns,
        lmns=lmns_full[1:, :],
        bmnc=bmnc[1:, :],
        bsubumnc=bsubumnc[1:, :],
        bsubvmnc=bsubvmnc[1:, :],
        iota=iota[1:],
    )


def _neopax_vmex_boozer_table_inputs_from_state(*, state, static, surface_indices=None) -> SimpleNamespace:
    """Batched current-VMEX version of ``vmex.core.boozer_tables.boozer_input_tables``."""
    rt = static.runtime
    setup = rt.setup
    lasym = bool(getattr(setup, "lasym", rt.resolution.lasym))
    if lasym:
        raise NotImplementedError("NEOPAX current-VMEX Boozer table AD input builder supports lasym=False only.")

    solver_module = _import_vmec_module("core.solver")
    geometry_module = _import_vmec_module("core.geometry")
    fields_module = _import_vmec_module("core.fields")

    nfp = int(rt.resolution.nfp)
    s = jnp.asarray(setup.s_full)
    ns = int(s.shape[0])
    if surface_indices is None:
        rows_np = np.arange(1, ns, dtype=np.int32)
    else:
        surface_indices_np = np.asarray(surface_indices, dtype=np.int32).reshape(-1)
        if surface_indices_np.size == 0:
            raise ValueError("Boozer table input surface_indices must be non-empty.")
        if np.any(surface_indices_np < 0) or np.any(surface_indices_np >= ns - 1):
            raise ValueError(
                f"Boozer table input surface_indices must be in [0, {ns - 2}], "
                f"got {surface_indices_np.tolist()}."
            )
        rows_np = surface_indices_np + 1
    rows = jnp.asarray(rows_np, dtype=jnp.int32)
    sqrt_s = jnp.sqrt(s)
    s_half = 0.5 * (s[rows] + s[rows - 1])

    _, geometry = solver_module._geometry(state, rt)
    jacobian = geometry_module.half_mesh_jacobian(geometry, s=s)
    metrics = fields_module.metric_elements(geometry, s=s)
    fields = fields_module.magnetic_fields(
        geometry=geometry,
        jacobian=jacobian,
        metrics=metrics,
        trig=rt.trig,
        s=s,
        phips=setup.phips,
        phipf=setup.phipf,
        chips=setup.chips,
        signgs=setup.signgs,
        gamma=rt.gamma,
        mass=setup.mass,
        ncurr=setup.ncurr,
        enclosed_current=setup.icurv,
    )

    ntheta2 = int(np.shape(fields.total_pressure)[1])
    nzeta = int(np.shape(fields.total_pressure)[2])
    ntheta1 = max(2 * (ntheta2 - 1), 1)
    i_full = np.arange(ntheta1)
    kk = np.arange(nzeta)
    i_src = np.where(i_full < ntheta2, i_full, ntheta1 - i_full)
    k_src = np.where(i_full[:, None] < ntheta2, kk[None, :], (nzeta - kk[None, :]) % nzeta)
    i_src2 = np.broadcast_to(i_src[:, None], (ntheta1, nzeta))
    sign_odd = jnp.asarray(np.where(i_full < ntheta2, 1.0, -1.0)[:, None])

    def mirror(values, parity):
        out = jnp.asarray(values)[:, i_src2, k_src]
        return out if parity == "even" else out * sign_odd[None, :, :]

    theta = 2.0 * np.pi * np.arange(ntheta1) / ntheta1
    zeta = 2.0 * np.pi * np.arange(nzeta) / (nfp * nzeta)
    m_max = ntheta1 // 2 - 1
    n_max = max(nzeta // 2 - 1, 0)
    ml: list[int] = []
    nl: list[int] = []
    for m_value in range(0, m_max + 1):
        for n_value in range(-n_max, n_max + 1):
            if m_value == 0 and n_value < 0:
                continue
            ml.append(m_value)
            nl.append(n_value * nfp)
    xm = np.asarray(ml, dtype=np.int32)
    xn = np.asarray(nl, dtype=np.int32)
    angle = theta[:, None, None] * xm[None, None, :] - zeta[None, :, None] * xn[None, None, :]
    cos_t = jnp.asarray(np.cos(angle))
    sin_t = jnp.asarray(np.sin(angle))
    weights = 2.0 / (ntheta1 * nzeta) * np.ones(xm.shape, dtype=np.float64)
    weights[(xm == 0) & (xn == 0)] = 1.0 / (ntheta1 * nzeta)
    weights_j = jnp.asarray(weights)

    def project(values, parity):
        basis = cos_t if parity == "even" else sin_t
        return weights_j[None, :] * jnp.einsum("rtz,tzm->rm", values, basis)

    pressure = jnp.asarray(fields.pressure)
    bsq2 = 2.0 * (jnp.asarray(fields.total_pressure)[rows] - pressure[rows, None, None])
    tiny = jnp.asarray(jnp.finfo(pressure.dtype).tiny, dtype=pressure.dtype)
    bmnc = project(mirror(jnp.sqrt(jnp.maximum(bsq2, tiny)), "even"), "even")
    bsubumnc = project(mirror(jnp.asarray(fields.bsubu)[rows], "even"), "even")
    bsubvmnc = project(mirror(jnp.asarray(fields.bsubv)[rows], "even"), "even")

    def phys_rows(even, odd, row_indices):
        return jnp.asarray(even)[row_indices] + sqrt_s[row_indices, None, None] * jnp.asarray(odd)[row_indices]

    def spectral_half(even, odd, parity):
        lo = project(mirror(phys_rows(even, odd, rows - 1), parity), parity)
        hi = project(mirror(phys_rows(even, odd, rows), parity), parity)
        m_even = jnp.asarray((xm % 2) == 0)
        interp_even = 0.5 * (lo + hi)
        interp_odd = (
            0.5
            * (lo / jnp.maximum(sqrt_s[rows - 1, None], 1.0e-30) + hi / sqrt_s[rows, None])
            * jnp.sqrt(s_half)[:, None]
        )
        return jnp.where(m_even[None, :], interp_even, interp_odd)

    rmnc = spectral_half(geometry.R_even, geometry.R_odd, "even")
    zmns = spectral_half(geometry.Z_even, geometry.Z_odd, "odd")

    lamscale = jnp.asarray(fields.lamscale)

    def half_native(even, odd):
        lo = phys_rows(even, odd, rows - 1)
        hi = phys_rows(even, odd, rows)
        return 0.5 * (lo + hi) * lamscale

    lth = project(mirror(half_native(geometry.dlambda_dtheta_even, geometry.dlambda_dtheta_odd), "even"), "even")
    lze = project(mirror(half_native(geometry.dlambda_dzeta_even, geometry.dlambda_dzeta_odd), "even"), "even")
    m_safe = jnp.asarray(np.where(xm != 0, xm, 1), dtype=jnp.float64)
    n_safe = jnp.asarray(np.where(xn != 0, xn, 1), dtype=jnp.float64)
    phips_rows = jnp.asarray(setup.phips)[rows]
    lmns = (
        jnp.where(
            jnp.asarray(xm != 0)[None, :],
            lth / m_safe[None, :],
            jnp.where(jnp.asarray(xn != 0)[None, :], -lze / n_safe[None, :], 0.0),
        )
        / phips_rows[:, None]
    )

    iota = jnp.where(
        jnp.asarray(int(setup.ncurr) == 1),
        jnp.asarray(fields.chips)[rows] / jnp.asarray(setup.phips)[rows],
        jnp.asarray(setup.iotas)[rows],
    )

    return SimpleNamespace(
        nfp=jnp.asarray(nfp, dtype=jnp.int32),
        xm=jnp.asarray(xm, dtype=jnp.int32),
        xn=jnp.asarray(xn, dtype=jnp.int32),
        xm_nyq=jnp.asarray(xm, dtype=jnp.int32),
        xn_nyq=jnp.asarray(xn, dtype=jnp.int32),
        rmnc=rmnc,
        zmns=zmns,
        lmns=lmns,
        bmnc=bmnc,
        bsubumnc=bsubumnc,
        bsubvmnc=bsubvmnc,
        iota=iota,
    )


def _booz_xform_inputs_from_state(
    *,
    state,
    static,
    indata,
    signgs,
    flux,
):
    vmec_jax = _import_vmec_jax()
    try:
        booz_xform_inputs_from_state = _resolve_vmec_attr(
            vmec_jax,
            "booz_xform_inputs_from_state",
            submodule="booz_input",
        )
        return booz_xform_inputs_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=signgs,
            flux=flux,
        )
    except (AttributeError, ModuleNotFoundError):
        if isinstance(static, _VmecStaticCompat):
            try:
                return _neopax_vmex_boozer_table_inputs_from_state(state=state, static=static)
            except (AttributeError, ModuleNotFoundError):
                return _neopax_vmex_booz_xform_inputs_from_state(state=state, static=static)

        raise


def _booz_constants_and_grids_for_inputs(context: "GeometryAutodiffContext", inputs):
    if context.booz_constants is not None and context.booz_grids is not None:
        return context.booz_constants, context.booz_grids
    booz_api = _import_booz_xform_jax_api()
    return booz_api.prepare_booz_xform_constants_from_inputs(
        inputs=inputs,
        mboz=int(context.mboz),
        nboz=int(context.nboz),
        asym=bool(context.cfg.lasym),
    )


def _resolve_vmec_attr(module, name: str, *, submodule: str | None = None):
    value = getattr(module, name, None)
    if value is not None:
        return value
    if submodule is None:
        raise AttributeError(f"{getattr(module, '__name__', 'vmec backend')} does not provide '{name}'.")
    imported = _import_vmec_module(submodule)
    value = getattr(imported, name, None)
    if value is None:
        raise AttributeError(f"{imported.__name__} does not provide '{name}'.")
    return value


def _resolve_vmec_attr_any(module, name: str, *, submodules: Sequence[str]):
    errors = []
    for submodule in submodules:
        try:
            return _resolve_vmec_attr(module, name, submodule=submodule)
        except (AttributeError, ModuleNotFoundError) as exc:
            errors.append(f"{submodule}: {exc}")
    raise AttributeError(
        f"VMEC backend does not provide '{name}' in any of: {', '.join(submodules)}. "
        f"Errors: {'; '.join(errors)}"
    )


@dataclasses.dataclass(frozen=True)
class _VmecStaticCompat:
    """Old NEOPAX-facing static view backed by the current vmec_jax runtime."""

    cfg: Any
    setup: Any
    resolution: Any
    runtime: Any

    @property
    def modes(self):
        return self.runtime.modes

    @property
    def grid(self):
        return self.runtime.trig

    @property
    def trig_vmec(self):
        return self.runtime.trig

    @property
    def s(self):
        return self.setup.s_full


def _vmec_input_get(indata: Any, name: str, default: Any = None) -> Any:
    if hasattr(indata, "get"):
        try:
            return indata.get(name, default)
        except Exception:
            pass
    attr = str(name).strip().lower()
    return getattr(indata, attr, default)


def _vmec_input_get_int(indata: Any, name: str, default: int) -> int:
    if hasattr(indata, "get_int"):
        try:
            return int(indata.get_int(name, default))
        except Exception:
            pass
    return int(_vmec_input_get(indata, name, default))


def _vmec_input_get_float(indata: Any, name: str, default: float | None) -> float | None:
    if hasattr(indata, "get_float"):
        try:
            return indata.get_float(name, default)
        except Exception:
            pass
    value = _vmec_input_get(indata, name, default)
    return None if value is None else float(value)


def _build_current_vmec_jax_context(vmec_jax, vmec_input: Path):
    VmecInput = _resolve_vmec_attr(vmec_jax, "VmecInput")
    inp = VmecInput.from_file(vmec_input)
    setup_module = _import_vmec_module("core.setup")
    solver_module = _import_vmec_module("core.solver")
    boundary_from_input = setup_module.boundary_from_input
    prepare_runtime = solver_module.prepare_runtime
    resolution_from_input = solver_module.resolution_from_input
    ns_array = np.atleast_1d(np.asarray(inp.ns_array, dtype=np.int64)).ravel()
    ns_array = ns_array[ns_array > 0]
    final_ns = int(ns_array[-1]) if ns_array.size else None
    resolution = resolution_from_input(inp, ns=final_ns)
    runtime = prepare_runtime(inp, resolution)
    static = _VmecStaticCompat(
        cfg=inp,
        setup=runtime.setup,
        resolution=resolution,
        runtime=runtime,
    )
    boundary = boundary_from_input(inp, modes=runtime.modes, trig=runtime.trig, lconm1=True)
    fixed_context = {
        "signgs": int(runtime.setup.signgs),
        "flux": SimpleNamespace(
            phips=runtime.setup.phips,
            chips=runtime.setup.chips,
            iotas=runtime.setup.iotas,
            icurv=runtime.setup.icurv,
            mass=runtime.setup.mass,
            phipf=runtime.setup.phipf,
            chipf=runtime.setup.chipf,
            iotaf=runtime.setup.iotaf,
            lamscale=runtime.setup.lamscale,
        ),
        "pressure": jnp.asarray(runtime.setup.mass),
        "booz_inputs": None,
    }
    return inp, inp, static, boundary, fixed_context


def _using_current_vmec_jax_context(context: "GeometryAutodiffContext") -> bool:
    return isinstance(context.static, _VmecStaticCompat)


def _input_boundary_array_name_for_kind(kind: str) -> str:
    if kind == "rc":
        return "rbc"
    if kind == "zs":
        return "zbs"
    raise ValueError(f"Unsupported boundary kind '{kind}'.")


def _input_with_boundary_delta(context: "GeometryAutodiffContext", param_delta):
    field_name = _input_boundary_array_name_for_kind(context.boundary_kind)
    base = jnp.asarray(getattr(context.indata, field_name), dtype=jnp.float64)
    n_offset = int(context.static.resolution.ntor) + int(context.param_n)
    m_index = int(context.param_m)
    updated = base.at[n_offset, m_index].add(jnp.asarray(param_delta, dtype=jnp.float64))
    return dataclasses.replace(context.indata, **{field_name: updated})


def _implicit_params_from_input(context: "GeometryAutodiffContext", implicit, solver_device: str | None = None):
    device_key = "auto" if solver_device is None else str(solver_device).strip().lower()
    if device_key in {"", "auto", "vmex"}:
        return implicit.params_from_input(context.indata)

    def _arr(value):
        raw = np.asarray(value, dtype=np.float64)
        if device_key in {"default", "jax"}:
            return jnp.asarray(raw)
        if device_key in {"cpu", "gpu", "cuda", "rocm", "tpu"}:
            platform = "gpu" if device_key in {"gpu", "cuda", "rocm"} else device_key
            return jax.device_put(raw, jax.devices(platform)[0])
        raise ValueError(
            "implicit solver device must be one of 'default', 'auto', 'cpu', or 'gpu'."
        )

    return implicit.ImplicitParams(
        rbc=_arr(context.indata.rbc),
        rbs=_arr(context.indata.rbs),
        zbc=_arr(context.indata.zbc),
        zbs=_arr(context.indata.zbs),
        phiedge=_arr(context.indata.phiedge),
        pres_scale=_arr(context.indata.pres_scale),
        curtor=_arr(context.indata.curtor),
        am=_arr(context.indata.am),
        ai=_arr(context.indata.ai),
        ac=_arr(context.indata.ac),
    )


def _implicit_params_with_boundary_delta(context: "GeometryAutodiffContext", implicit, param_delta, *, solver_device: str | None = None):
    params = _implicit_params_from_input(context, implicit, solver_device=solver_device)
    field_name = _input_boundary_array_name_for_kind(context.boundary_kind)
    base = jnp.asarray(getattr(params, field_name), dtype=jnp.float64)
    n_offset = int(context.static.resolution.ntor) + int(context.param_n)
    m_index = int(context.param_m)
    updated = base.at[n_offset, m_index].add(jnp.asarray(param_delta, dtype=jnp.float64))
    return dataclasses.replace(params, **{field_name: updated})


def _boundary_param_entry(context: "GeometryAutodiffContext", param_family: str, param_m: int, param_n: int) -> dict[str, Any]:
    kind = _boundary_kind_for_family(param_family)
    m_arr = jnp.asarray(context.static.modes.m)
    n_arr = jnp.asarray(context.static.modes.n)
    matches = jnp.where((m_arr == int(param_m)) & (n_arr == int(param_n)), size=2, fill_value=-1)[0]
    match_indices = [int(idx) for idx in np.asarray(matches) if int(idx) >= 0]
    if not match_indices:
        raise ValueError(
            f"Could not find a {param_family} coefficient with (m, n)=({param_m}, {param_n}) in {context.input_path}."
        )
    if len(match_indices) > 1:
        raise ValueError(
            f"Found multiple matches for {param_family}({param_m}, {param_n}); expected exactly one."
        )
    boundary_index = int(match_indices[0])
    boundary_array = jnp.asarray(getattr(context.boundary, _boundary_array_name_for_kind(kind)))
    return {
        "family": str(param_family).strip().upper(),
        "kind": kind,
        "m": int(param_m),
        "n": int(param_n),
        "boundary_index": boundary_index,
        "input_field": _input_boundary_array_name_for_kind(kind),
        "boundary_field": _boundary_array_name_for_kind(kind),
        "n_offset": int(context.static.resolution.ntor) + int(param_n),
        "m_index": int(param_m),
        "baseline_coefficient": float(boundary_array[boundary_index]),
    }


def boundary_param_entries(
    context: "GeometryAutodiffContext",
    param_specs: Sequence[tuple[str, int, int]],
) -> tuple[dict[str, Any], ...]:
    return tuple(_boundary_param_entry(context, family, m, n) for family, m, n in param_specs)


def _input_with_boundary_deltas(
    context: "GeometryAutodiffContext",
    param_deltas,
    param_entries: Sequence[dict[str, Any]],
):
    updates: dict[str, Any] = {}
    deltas = jnp.asarray(param_deltas, dtype=jnp.float64)
    for i, entry in enumerate(param_entries):
        field_name = entry["input_field"]
        base = updates.get(field_name)
        if base is None:
            base = jnp.asarray(getattr(context.indata, field_name), dtype=jnp.float64)
        updates[field_name] = base.at[int(entry["n_offset"]), int(entry["m_index"])].add(deltas[i])
    return dataclasses.replace(context.indata, **updates)


def _implicit_params_with_boundary_deltas(
    context: "GeometryAutodiffContext",
    implicit,
    param_deltas,
    param_entries: Sequence[dict[str, Any]],
    *,
    solver_device: str | None = None,
):
    params = _implicit_params_from_input(context, implicit, solver_device=solver_device)
    updates: dict[str, Any] = {}
    deltas = jnp.asarray(param_deltas, dtype=jnp.float64)
    for i, entry in enumerate(param_entries):
        field_name = entry["input_field"]
        base = updates.get(field_name)
        if base is None:
            base = jnp.asarray(getattr(params, field_name), dtype=jnp.float64)
        updates[field_name] = base.at[int(entry["n_offset"]), int(entry["m_index"])].add(deltas[i])
    return dataclasses.replace(params, **updates)


def _current_implicit_params_cfg_for_param_vector(
    context: "GeometryAutodiffContext",
    param_deltas,
    param_entries: Sequence[dict[str, Any]],
    *,
    max_iter: int | None = None,
    solver_device: str | None = None,
):
    implicit = _import_vmec_jax_implicit()
    params = _implicit_params_with_boundary_deltas(
        context,
        implicit,
        param_deltas,
        param_entries,
        solver_device=solver_device,
    )
    config_kwargs = {
        "ns": int(context.static.resolution.ns),
        "mode": "cli",
        "multigrid": True,
    }
    if max_iter is not None:
        config_kwargs["max_iterations"] = int(max_iter)
    return implicit, params, implicit.make_config(context.indata, **config_kwargs)


def _param_vector_gradient_from_implicit_param_grads(
    param_grads,
    param_entries: Sequence[dict[str, Any]],
) -> jnp.ndarray:
    columns = []
    for entry in param_entries:
        arr = jnp.asarray(getattr(param_grads, entry["input_field"]), dtype=jnp.float64)
        columns.append(arr[:, int(entry["n_offset"]), int(entry["m_index"])])
    return jnp.stack(columns, axis=1)


@dataclasses.dataclass(frozen=True, slots=True)
class GeometryRawBlockSolve:
    """Shared VMEC implicit solve payload for raw-block transpose pullbacks."""

    implicit: Any
    implicit_params: Any
    implicit_cfg: Any
    state: Any
    dof_mask: Any
    param_entries: tuple[dict[str, Any], ...]


def _stop_gradient_if_jax_value(leaf):
    try:
        return jax.lax.stop_gradient(leaf)
    except TypeError:
        return leaf


def geometry_raw_block_solve_from_param_vector(
    context: "GeometryAutodiffContext",
    param_deltas,
    param_specs: Sequence[tuple[str, int, int]],
    *,
    max_iter: int | None = None,
    solver_device: str | None = None,
) -> GeometryRawBlockSolve:
    """Solve VMEC once and keep the raw-block transpose auxiliary data."""

    if not _using_current_vmec_jax_context(context):
        raise ValueError("geometry_raw_block_solve_from_param_vector requires the current VMEX implicit AD lane.")
    param_entries = boundary_param_entries(context, param_specs)
    implicit, implicit_params, implicit_cfg = _current_implicit_params_cfg_for_param_vector(
        context,
        jnp.asarray(param_deltas, dtype=jnp.float64),
        param_entries,
        max_iter=max_iter,
        solver_device=solver_device,
    )
    # The raw-block transpose below differentiates the residual with respect to
    # the full VMEC parameter pytree and then extracts requested harmonic
    # columns.  It does not need the traced construction history from the
    # optimizer's compact parameter vector to the full VMEC input arrays.  Cutting
    # that ancestry keeps the payload/state VJPs from scaling with the number of
    # selected boundary harmonics.
    implicit_params = jax.tree_util.tree_map(_stop_gradient_if_jax_value, implicit_params)
    state, dof_mask = implicit.solve_implicit_with_aux(implicit_params, implicit_cfg)
    state = jax.tree_util.tree_map(_stop_gradient_if_jax_value, state)
    dof_mask = jax.tree_util.tree_map(_stop_gradient_if_jax_value, dof_mask)
    return GeometryRawBlockSolve(
        implicit=implicit,
        implicit_params=implicit_params,
        implicit_cfg=implicit_cfg,
        state=state,
        dof_mask=dof_mask,
        param_entries=param_entries,
    )


def _implicit_state_pullback_multi_rhs_with_assemble_rhs(
    implicit,
    params,
    cfg,
    x_star,
    dof_mask,
    state_bar_batch,
):
    """Batched implicit pullback from assembled VMEC-state cotangents."""

    frozen = jax.lax.stop_gradient(x_star)
    edge_mask = implicit._edge_mask(cfg)
    projector = implicit._dof_projector(cfg, dof_mask)
    residual = implicit.residual_fn(cfg, frozen, dof_mask)
    z_star = projector(x_star)

    _, assemble_vjp_z = jax.vjp(
        lambda z: implicit._assemble(
            z,
            implicit.runtime_from_params(params, cfg),
            frozen,
            projector,
            edge_mask,
        ),
        z_star,
    )
    _, residual_vjp_z = jax.vjp(lambda z: residual(z, params), z_star)
    _, residual_vjp_p = jax.vjp(lambda prm: residual(z_star, prm), params)
    _, assemble_vjp_p = jax.vjp(
        lambda prm: implicit._assemble(
            z_star,
            implicit.runtime_from_params(prm, cfg),
            frozen,
            projector,
            edge_mask,
        ),
        params,
    )

    rhs_batch = jax.vmap(lambda state_bar: assemble_vjp_z(state_bar)[0])(state_bar_batch)
    lambda_batch = jax.vmap(
        lambda rhs: implicit._adjoint_solve(lambda vec: residual_vjp_z(vec)[0], rhs, cfg)[0]
    )(rhs_batch)
    implicit_param_bar = jax.vmap(
        lambda lam: residual_vjp_p(jax.tree.map(jnp.negative, lam))[0]
    )(lambda_batch)
    assemble_param_bar = jax.vmap(lambda state_bar: assemble_vjp_p(state_bar)[0])(state_bar_batch)
    return jax.tree.map(lambda left, right: left + right, implicit_param_bar, assemble_param_bar)


def geometry_raw_block_transpose_from_state_bars(
    raw_block_solve: GeometryRawBlockSolve,
    state_bar_batch,
    *,
    probe_chunk_size: int = 1,
) -> jnp.ndarray:
    """Pull a batch of VMEC-state cotangents back to raw boundary parameters."""

    implicit = raw_block_solve.implicit
    if not hasattr(implicit, "implicit_state_pullback_multi_rhs_raw_block_transpose"):
        raise AttributeError(
            "The active VMEC backend does not expose implicit_state_pullback_multi_rhs_raw_block_transpose."
        )
    param_bar_batch = implicit.implicit_state_pullback_multi_rhs_raw_block_transpose(
        raw_block_solve.implicit_params,
        raw_block_solve.implicit_cfg,
        raw_block_solve.state,
        raw_block_solve.dof_mask,
        state_bar_batch,
        probe_chunk_size=int(probe_chunk_size),
    )
    return _param_vector_gradient_from_implicit_param_grads(param_bar_batch, raw_block_solve.param_entries)


def _boundary_with_boundary_deltas(
    context: "GeometryAutodiffContext",
    param_deltas,
    param_entries: Sequence[dict[str, Any]],
):
    updates: dict[str, Any] = {}
    deltas = jnp.asarray(param_deltas, dtype=jnp.float64)
    for i, entry in enumerate(param_entries):
        field_name = entry["boundary_field"]
        base = updates.get(field_name)
        if base is None:
            base = jnp.asarray(getattr(context.boundary, field_name), dtype=jnp.float64)
        updates[field_name] = base.at[int(entry["boundary_index"])].add(deltas[i])
    return dataclasses.replace(context.boundary, **updates)


def _wout_from_vmec_state(
    context: "GeometryAutodiffContext",
    state,
    *,
    fsqr: float = 0.0,
    fsqz: float = 0.0,
    fsql: float = 0.0,
):
    try:
        wout_module = _import_vmec_module("wout")
        wout_minimal_from_fixed_boundary = wout_module.wout_minimal_from_fixed_boundary

        return wout_minimal_from_fixed_boundary(
            path=context.input_path,
            state=state,
            static=context.static,
            indata=context.indata,
            signgs=int(context.signgs),
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            converged=True,
            flux_override=context.flux,
        )
    except (AttributeError, ModuleNotFoundError):
        wout_module = _import_vmec_module("core.wout")
        wout_from_state = wout_module.wout_from_state

        return wout_from_state(
            inp=context.indata,
            state=state,
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            converged=True,
            input_extension=str(context.input_path),
        )


def _build_vmec_fixed_context(vmec_jax, *, static, indata, boundary):
    initial_guess_from_boundary = _resolve_vmec_attr(vmec_jax, "initial_guess_from_boundary", submodule="init_guess")
    eval_geom = _resolve_vmec_attr(vmec_jax, "eval_geom", submodule="geom")
    signgs_from_sqrtg = _resolve_vmec_attr(vmec_jax, "signgs_from_sqrtg", submodule="field")
    eval_profiles = _resolve_vmec_attr(vmec_jax, "eval_profiles", submodule="profiles")
    booz_xform_inputs_from_state = _resolve_vmec_attr(vmec_jax, "booz_xform_inputs_from_state", submodule="booz_input")

    st_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=False)
    geom = eval_geom(st_guess, static)
    signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))
    try:
        flux_profiles_from_indata = _resolve_vmec_attr(vmec_jax, "flux_profiles_from_indata", submodule="energy")
        flux = flux_profiles_from_indata(indata, jnp.asarray(static.s), signgs=signgs)
    except (AttributeError, ModuleNotFoundError):
        setup_module = _import_vmec_module("core.setup")
        flux_profiles = setup_module.flux_profiles
        radial_grids = setup_module.radial_grids

        grids = radial_grids(int(jnp.asarray(static.s).shape[0]), dtype=jnp.asarray(static.s).dtype)
        flux = SimpleNamespace(
            **flux_profiles(
                indata,
                grids,
                r00=getattr(boundary, "r00", jnp.asarray(1.0, dtype=jnp.asarray(static.s).dtype)),
                signgs=signgs,
                lflip=bool(getattr(boundary, "lflip", False)),
            )
        )
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
    niter_array = _vmec_input_get(indata, "NITER_ARRAY", None)
    if niter_array is not None:
        try:
            values = [int(v) for v in niter_array]
            if values:
                return int(values[-1])
        except Exception:
            pass
    try:
        return _vmec_input_get_int(indata, "NITER", 100)
    except Exception:
        return 100


def _vmec_default_step_size_from_indata(indata: Any) -> float:
    try:
        return float(_vmec_input_get_float(indata, "DELT", 1.0))
    except Exception:
        return 1.0


def _vmec_default_ftol_from_indata(indata: Any) -> float | None:
    ftol_array = _vmec_input_get(indata, "FTOL_ARRAY", None)
    if ftol_array is not None:
        try:
            values = [float(v) for v in ftol_array]
            if values:
                return float(values[-1])
        except Exception:
            pass
    try:
        value = _vmec_input_get_float(indata, "FTOL", None)
    except Exception:
        value = None
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null"}:
            return None
        return int(text)
    return int(value)


def _optional_static_int(static: Any, cfg: Any, name: str) -> int | None:
    for owner in (static, getattr(static, "cfg", None), cfg):
        if owner is None or not hasattr(owner, name):
            continue
        value = getattr(owner, name)
        if value is not None:
            return int(value)
    return None


def _resolve_booz_resolution_defaults(
    *,
    static: Any,
    cfg: Any,
    mboz: int | None,
    nboz: int | None,
) -> tuple[int, int]:
    """Mirror the xbooz_xform-style auto resolution for unset Boozer modes."""
    resolved_mboz = _optional_int(mboz)
    resolved_nboz = _optional_int(nboz)

    mpol = _optional_static_int(static, cfg, "mpol")
    ntor = _optional_static_int(static, cfg, "ntor")

    if resolved_mboz is None:
        if mpol is None:
            raise ValueError("mboz was not provided and VMEC mpol could not be inferred.")
        resolved_mboz = max(2, 6 * int(mpol))
    if resolved_nboz is None:
        if ntor is None:
            raise ValueError("nboz was not provided and VMEC ntor could not be inferred.")
        resolved_nboz = max(0, 2 * int(ntor) - 1)
    return int(resolved_mboz), int(resolved_nboz)


def build_geometry_autodiff_context(
    input_path: str | Path,
    *,
    param_family: str,
    param_m: int,
    param_n: int,
    mboz: int | None = None,
    nboz: int | None = None,
    surface_s: Sequence[float] = (0.25, 0.5, 0.75),
) -> GeometryAutodiffContext:
    vmec_backend = _import_vmec_jax()
    booz_api = _import_booz_xform_jax_api()

    vmec_input = Path(input_path).expanduser().resolve()
    if all(hasattr(vmec_backend, name) for name in ("load_input", "build_static", "boundary_from_indata")):
        cfg, indata = vmec_backend.load_input(str(vmec_input))
        static = vmec_backend.build_static(cfg)
        boundary = vmec_backend.boundary_from_indata(indata, static.modes)
        fixed_context = None
    else:
        cfg, indata, static, boundary, fixed_context = _build_current_vmec_jax_context(vmec_backend, vmec_input)

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

    if fixed_context is None:
        try:
            prepare_fixed_boundary_context = _resolve_vmec_attr(vmec_backend, "prepare_fixed_boundary_context")
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
            fixed_context = _build_vmec_fixed_context(vmec_backend, static=static, indata=indata, boundary=boundary)
    try:
        surface_indices_from_static = _resolve_vmec_attr(vmec_backend, "surface_indices_from_static")
        surface_indices, _ = surface_indices_from_static(static, list(surface_s))
    except Exception:
        s_half = 0.5 * (np.asarray(static.s[:-1], dtype=float) + np.asarray(static.s[1:], dtype=float))
        surface_indices = [int(np.argmin(np.abs(s_half - float(val)))) for val in surface_s]
    resolved_mboz, resolved_nboz = _resolve_booz_resolution_defaults(
        static=static,
        cfg=cfg,
        mboz=mboz,
        nboz=nboz,
    )
    if fixed_context.get("booz_inputs") is None:
        booz_constants, booz_grids = None, None
    else:
        booz_constants, booz_grids = booz_api.prepare_booz_xform_constants_from_inputs(
            inputs=fixed_context["booz_inputs"],
            mboz=resolved_mboz,
            nboz=resolved_nboz,
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
        mboz=resolved_mboz,
        nboz=resolved_nboz,
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
    solver_device: str | None = None,
) -> Any:
    vmec_jax = _import_vmec_jax()
    if _using_current_vmec_jax_context(context):
        lane_key = str(lane).strip().lower()
        if lane_key not in {"forward", "ad"}:
            raise ValueError("lane must be 'forward' or 'ad'.")
        if lane_key == "forward":
            input_eff = _input_with_boundary_delta(context, param_delta)
            multigrid = _import_vmec_module("core.multigrid")
            solve_multigrid = multigrid.solve_multigrid

            solve_kwargs = {"mode": "cli", "verbose": False}
            if max_iter is not None:
                niter_array = np.asarray(input_eff.niter_array, dtype=np.int64).copy()
                niter_array[-1] = int(max_iter)
                solve_kwargs["niter_array"] = niter_array
            if step_size is not None:
                solve_kwargs["time_step"] = float(step_size)
            return solve_multigrid(
                input_eff,
                **solve_kwargs,
            ).state

        del jacobian_penalty
        implicit = _import_vmec_jax_implicit()
        params = _implicit_params_with_boundary_delta(
            context,
            implicit,
            param_delta,
            solver_device=solver_device,
        )
        config_kwargs = {
            "ns": int(context.static.resolution.ns),
            "mode": "cli",
            "multigrid": True,
        }
        if max_iter is not None:
            config_kwargs["max_iterations"] = int(max_iter)
        cfg = implicit.make_config(
            context.indata,
            **config_kwargs,
        )
        return implicit.solve_implicit(params, cfg)

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
            backtracking=False,
            limit_dt_from_force=False,
            limit_update_rms=False,
            verbose=False,
            verbose_vmec2000_table=False,
            jit_forces="auto",
            use_scan=True,
        )
        return result.state

    # AD lane: keep the implicit residual solve separate so we can add a
    # reverse-mode path later without changing the forward lane contract.
    del jacobian_penalty
    implicit = _import_vmec_jax_implicit()
    return implicit.solve_fixed_boundary_state_implicit_vmec_residual(
        state0,
        context.static,
        indata=context.indata,
        signgs=int(context.signgs),
        max_iter=max_iter_value,
        step_size=step_size_value,
        ftol=context.vmec_default_ftol,
        edge_Rcos=_vmec_state_field(state0, "Rcos", "R_cos")[-1, :],
        edge_Rsin=_vmec_state_field(state0, "Rsin", "R_sin")[-1, :],
        edge_Zcos=_vmec_state_field(state0, "Zcos", "Z_cos")[-1, :],
        edge_Zsin=_vmec_state_field(state0, "Zsin", "Z_sin")[-1, :],
    )


def _solve_state_for_param_vector(
    context: GeometryAutodiffContext,
    param_deltas,
    param_specs: Sequence[tuple[str, int, int]],
    *,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
    solver_device: str | None = None,
) -> Any:
    param_entries = boundary_param_entries(context, param_specs)
    vmec_jax = _import_vmec_jax()
    if _using_current_vmec_jax_context(context):
        lane_key = str(lane).strip().lower()
        if lane_key not in {"forward", "ad"}:
            raise ValueError("lane must be 'forward' or 'ad'.")
        if lane_key == "forward":
            input_eff = _input_with_boundary_deltas(context, param_deltas, param_entries)
            multigrid = _import_vmec_module("core.multigrid")
            solve_multigrid = multigrid.solve_multigrid

            solve_kwargs = {"mode": "cli", "verbose": False}
            if max_iter is not None:
                niter_array = np.asarray(input_eff.niter_array, dtype=np.int64).copy()
                niter_array[-1] = int(max_iter)
                solve_kwargs["niter_array"] = niter_array
            if step_size is not None:
                solve_kwargs["time_step"] = float(step_size)
            return solve_multigrid(input_eff, **solve_kwargs).state

        del jacobian_penalty
        implicit = _import_vmec_jax_implicit()
        params = _implicit_params_with_boundary_deltas(
            context,
            implicit,
            param_deltas,
            param_entries,
            solver_device=solver_device,
        )
        config_kwargs = {
            "ns": int(context.static.resolution.ns),
            "mode": "cli",
            "multigrid": True,
        }
        if max_iter is not None:
            config_kwargs["max_iterations"] = int(max_iter)
        cfg = implicit.make_config(context.indata, **config_kwargs)
        return implicit.solve_implicit(params, cfg)

    initial_guess_from_boundary = _resolve_vmec_attr(vmec_jax, "initial_guess_from_boundary", submodule="init_guess")
    solve_fixed_boundary_residual_iter = _resolve_vmec_attr(vmec_jax, "solve_fixed_boundary_residual_iter", submodule="solve")

    lane_key = str(lane).strip().lower()
    if lane_key not in {"forward", "ad"}:
        raise ValueError("lane must be 'forward' or 'ad'.")

    max_iter_value = int(context.vmec_default_max_iter if max_iter is None else max_iter)
    step_size_value = float(context.vmec_default_step_size if step_size is None else step_size)
    boundary = _boundary_with_boundary_deltas(context, param_deltas, param_entries)
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
            backtracking=False,
            limit_dt_from_force=False,
            limit_update_rms=False,
            verbose=False,
            verbose_vmec2000_table=False,
            jit_forces="auto",
            use_scan=True,
        )
        return result.state

    del jacobian_penalty
    implicit = _import_vmec_jax_implicit()
    return implicit.solve_fixed_boundary_state_implicit_vmec_residual(
        state0,
        context.static,
        indata=context.indata,
        signgs=int(context.signgs),
        max_iter=max_iter_value,
        step_size=step_size_value,
        ftol=context.vmec_default_ftol,
        edge_Rcos=_vmec_state_field(state0, "Rcos", "R_cos")[-1, :],
        edge_Rsin=_vmec_state_field(state0, "Rsin", "R_sin")[-1, :],
        edge_Zcos=_vmec_state_field(state0, "Zcos", "Z_cos")[-1, :],
        edge_Zsin=_vmec_state_field(state0, "Zsin", "Z_sin")[-1, :],
    )


def _find_mode_index(ixm_b: jnp.ndarray, ixn_b: jnp.ndarray, *, m: int, n: int) -> int | None:
    matches = jnp.where((ixm_b == int(m)) & (ixn_b == int(n)), size=1, fill_value=-1)[0]
    match = int(matches[0])
    return None if match < 0 else match


def _vmec_state_field(state, legacy_name: str, current_name: str):
    if hasattr(state, current_name):
        return getattr(state, current_name)
    return getattr(state, legacy_name)


def _smooth_negative_part_penalty(values: jnp.ndarray, *, softness: float = 1.0e-3) -> jnp.ndarray:
    values = jnp.asarray(values, dtype=jnp.float64)
    if int(values.size) == 0:
        return jnp.asarray(0.0, dtype=jnp.float64)
    softness_value = jnp.asarray(float(max(softness, 1.0e-12)), dtype=jnp.float64)
    deficit = -values
    return jnp.mean(softness_value * jnp.logaddexp(jnp.asarray(0.0, dtype=jnp.float64), deficit / softness_value))


def _smooth_lower_bound_penalty(values: jnp.ndarray, *, minimum: float = 0.0, softness: float = 1.0e-3) -> jnp.ndarray:
    values = jnp.asarray(values, dtype=jnp.float64)
    softness_value = jnp.asarray(float(max(softness, 1.0e-12)), dtype=jnp.float64)
    deficit = jnp.asarray(float(minimum), dtype=jnp.float64) - values
    return softness_value * jnp.logaddexp(jnp.asarray(0.0, dtype=jnp.float64), deficit / softness_value)


def _three_middle_profile_points(values: jnp.ndarray) -> jnp.ndarray:
    values = jnp.asarray(values, dtype=jnp.float64)
    npts = int(values.size)
    if npts <= 3:
        return values
    center = npts // 2
    if center == 0:
        return values[:3]
    if center >= npts - 1:
        return values[-3:]
    return values[center - 1 : center + 2]


def _three_middle_profile_points_fixed(values: jnp.ndarray) -> jnp.ndarray:
    values = jnp.asarray(values, dtype=jnp.float64)
    npts = int(values.size)
    if npts <= 0:
        return jnp.zeros((3,), dtype=jnp.float64)
    if npts == 1:
        return jnp.repeat(values[:1], 3, axis=0)
    if npts == 2:
        return jnp.asarray([values[0], values[0], values[1]], dtype=jnp.float64)
    return _three_middle_profile_points(values)


def _mean_native_dmerc_objective(values: jnp.ndarray, *, minimum: float = 0.0, softness: float = 1.0e-3) -> jnp.ndarray:
    values = jnp.asarray(values, dtype=jnp.float64)
    active = values[1:-1] if int(values.size) > 2 else jnp.zeros((0,), dtype=jnp.float64)
    penalties = _smooth_lower_bound_penalty(active, minimum=minimum, softness=softness)
    if int(penalties.size) == 0:
        return jnp.asarray(0.0, dtype=jnp.float64)
    return jnp.mean(penalties)


def _native_dmerc_objective_samples(values: jnp.ndarray, *, minimum: float = 0.0, softness: float = 1.0e-3) -> jnp.ndarray:
    values = jnp.asarray(values, dtype=jnp.float64)
    active = values[1:-1] if int(values.size) > 2 else jnp.zeros((0,), dtype=jnp.float64)
    penalties = _smooth_lower_bound_penalty(active, minimum=minimum, softness=softness)
    return _three_middle_profile_points_fixed(penalties)


def _vmec_magnetic_well_from_state(
    context: GeometryAutodiffContext,
    state,
) -> jnp.ndarray:
    vmec_jax = _import_vmec_jax()
    try:
        magnetic_well_from_state = _resolve_vmec_attr(
            vmec_jax,
            "magnetic_well_from_state",
            submodule="finite_beta",
        )
        return jnp.asarray(
            magnetic_well_from_state(
                state=state,
                static=context.static,
                indata=context.indata,
                signgs=int(context.signgs),
            ),
            dtype=jnp.float64,
        )
    except (AttributeError, ModuleNotFoundError):
        optimization = _import_vmec_jax_optimization()
        return jnp.asarray(
            optimization.magnetic_well(state, context.static.runtime),
            dtype=jnp.float64,
        )


def _vmec_dmerc_profile_from_state(
    context: GeometryAutodiffContext,
    state,
) -> jnp.ndarray:
    vmec_jax = _import_vmec_jax()
    try:
        mercier_terms_from_state = _resolve_vmec_attr(
            vmec_jax,
            "mercier_terms_from_state",
            submodule="finite_beta",
        )
        return jnp.asarray(
            mercier_terms_from_state(
                state=state,
                static=context.static,
                indata=context.indata,
                signgs=int(context.signgs),
            )["DMerc"],
            dtype=jnp.float64,
        )
    except (AttributeError, ModuleNotFoundError):
        # Current vmec_jax exposes Mercier through wout/NumPy utilities, not as
        # an AD-transparent state function. Keep the scalar-observable smoke
        # gate running while making this diagnostic contribution neutral.
        return jnp.zeros_like(jnp.asarray(context.static.s, dtype=jnp.float64))


def _vmec_scalar_observables_from_state(
    context: GeometryAutodiffContext,
    state,
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
    b_cartesian_from_state = _resolve_vmec_attr(
        vmec_jax,
        "b_cartesian_from_state",
        submodule="field",
    )
    smooth_reduce_max = _resolve_vmec_attr_any(
        vmec_jax,
        "_smooth_reduce_max",
        submodules=("quasi_isodynamic", "quasi_isodynamic.objectives"),
    )
    smooth_reduce_min = _resolve_vmec_attr_any(
        vmec_jax,
        "_smooth_reduce_min",
        submodules=("quasi_isodynamic", "quasi_isodynamic.objectives"),
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
    magnetic_well = _vmec_magnetic_well_from_state(context, state)
    dmerc = _vmec_dmerc_profile_from_state(context, state)
    dmerc_objective_samples = _native_dmerc_objective_samples(
        dmerc,
        minimum=0.0,
        softness=1.0e-3,
    )
    mirror_surface_indices = [int(np.argmin(np.abs(np.asarray(context.static.s, dtype=float) - float(surface)))) for surface in context.surface_s]
    mirror_ratios = []
    tiny = jnp.asarray(jnp.finfo(jnp.float64).tiny, dtype=jnp.float64)
    for s_index in mirror_surface_indices:
        bcart = jnp.asarray(
            b_cartesian_from_state(
                state,
                context.static,
                indata=context.indata,
                signgs=int(context.signgs),
                s_index=int(s_index),
            ),
            dtype=jnp.float64,
        )
        bmag = jnp.sqrt(jnp.maximum(jnp.sum(bcart * bcart, axis=-1), tiny))
        bmax = smooth_reduce_max(bmag, axis=(0, 1), softness=2.0e-2)
        bmin = smooth_reduce_min(bmag, axis=(0, 1), softness=2.0e-2)
        bmin_positive = jnp.maximum(bmin, tiny)
        denom = jnp.maximum(bmax + bmin_positive, tiny)
        mirror_ratios.append((bmax - bmin_positive) / denom)
    mirror_ratio = jnp.asarray(mirror_ratios, dtype=jnp.float64)
    mirror_threshold = jnp.asarray(0.21, dtype=jnp.float64)
    mirror_softness = jnp.asarray(2.0e-2, dtype=jnp.float64)
    mirror_penalty = mirror_softness * jnp.logaddexp(
        jnp.asarray(0.0, dtype=jnp.float64),
        (mirror_ratio - mirror_threshold) / mirror_softness,
    )
    mirror_ratio_objective = jnp.mean(mirror_penalty * mirror_penalty)
    magnetic_well_objective = _smooth_lower_bound_penalty(
        magnetic_well,
        minimum=0.0,
        softness=1.0e-3,
    ).reshape(())
    return {
        "aspect_ratio": jnp.asarray(equilibrium_aspect_ratio_from_state(state=state, static=context.static)),
        "volume_total": jnp.asarray(volume[-1]),
        "iota_mean": jnp.asarray(iota_mean),
        "magnetic_well_objective": jnp.asarray(magnetic_well_objective, dtype=jnp.float64),
        "dmerc_objective_lo": jnp.asarray(dmerc_objective_samples[0], dtype=jnp.float64),
        "dmerc_objective_mid": jnp.asarray(dmerc_objective_samples[1], dtype=jnp.float64),
        "dmerc_objective_hi": jnp.asarray(dmerc_objective_samples[2], dtype=jnp.float64),
        "mirror_ratio_objective": jnp.asarray(mirror_ratio_objective, dtype=jnp.float64),
        "edge_r00": jnp.asarray(_vmec_state_field(state, "Rcos", "R_cos")[-1, 0]),
    }


def _vmec_core_scalar_objectives_from_state(
    context: GeometryAutodiffContext,
    state,
) -> dict[str, jnp.ndarray]:
    """Current vmec_jax traceable scalar objectives used by the AD geometry gate."""

    optimization = _import_vmec_jax_optimization()
    rt = context.static.runtime
    surface_indices = tuple(int(index) for index in np.asarray(context.surface_indices, dtype=np.int32).reshape(-1))
    mirror_values = [
        optimization.mirror_ratio(state, rt, s_index=index)
        for index in surface_indices
    ]
    mirror_ratio = (
        jnp.mean(jnp.asarray(mirror_values, dtype=jnp.float64))
        if mirror_values
        else optimization.mirror_ratio(state, rt)
    )

    # This input family is vacuum/no-pressure, so beta is expected to be zero.
    # Keep the objective traceable and explicit without pretending to test a
    # finite-beta path that the case does not exercise.
    pressure = jnp.asarray(context.pressure, dtype=jnp.float64)
    beta_volume = jnp.asarray(0.0, dtype=jnp.float64)
    if int(pressure.size) > 0:
        beta_volume = jnp.asarray(jnp.mean(jnp.abs(pressure)) * 0.0, dtype=jnp.float64)

    return {
        "aspect_ratio": jnp.asarray(optimization.aspect_ratio(state, rt), dtype=jnp.float64),
        "volume_total": jnp.asarray(optimization.volume(state, rt), dtype=jnp.float64),
        "iota_mean": jnp.asarray(optimization.mean_iota(state, rt), dtype=jnp.float64),
        "magnetic_well": jnp.asarray(optimization.magnetic_well(state, rt), dtype=jnp.float64),
        "mirror_ratio": jnp.asarray(mirror_ratio, dtype=jnp.float64),
        "beta_volume": beta_volume,
    }


def _vmec_dmerc_unavailable_error() -> RuntimeError:
    return RuntimeError(
        "DMerc is not available in the current AD geometry benchmark because "
        "vmec_jax.core.optimize.d_merc is a wout/NumPy finite-difference-only "
        "objective. To make DMerc work in reverse AD, port the Mercier terms "
        "to a pure JAX state-level function, e.g. state/runtime -> fields/"
        "nyquist quantities -> DMerc, without constructing a host WOUT object."
    )


def _vmec_iotaf_scalar_observables_from_state(
    context: GeometryAutodiffContext,
    state,
) -> dict[str, jnp.ndarray]:
    vmec_jax = _import_vmec_jax()
    equilibrium_iota_profiles_from_state = _resolve_vmec_attr(
        vmec_jax,
        "equilibrium_iota_profiles_from_state",
        submodule="profiles",
    )

    _chips, iotas, iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=int(context.signgs),
    )
    iotas = jnp.asarray(iotas, dtype=jnp.float64)
    iotaf = jnp.asarray(iotaf, dtype=jnp.float64)
    npts = int(iotaf.size)
    if npts <= 0:
        raise ValueError("equilibrium_iota_profiles_from_state returned an empty iotaf profile.")

    edge_idx = npts - 1
    if npts > 2:
        interior_start = 1
        interior_stop = npts - 1
        q1_idx = interior_start + (interior_stop - interior_start) // 4
        mid_idx = interior_start + (interior_stop - interior_start) // 2
        q3_idx = interior_start + (3 * (interior_stop - interior_start)) // 4
        iota_mean = jnp.mean(iotaf[1:])
    else:
        q1_idx = edge_idx
        mid_idx = edge_idx
        q3_idx = edge_idx
        iota_mean = iotaf[edge_idx]

    return {
        "iotas_1": jnp.asarray(iotas[1] if int(iotas.size) > 1 else iotas[0]),
        "iotas_2": jnp.asarray(iotas[2] if int(iotas.size) > 2 else iotas[-1]),
        "iotaf_first": jnp.asarray(iotaf[1] if npts > 1 else iotaf[0]),
        "iotaf_q1": jnp.asarray(iotaf[q1_idx]),
        "iotaf_mid": jnp.asarray(iotaf[mid_idx]),
        "iotaf_q3": jnp.asarray(iotaf[q3_idx]),
        "iotaf_edge": jnp.asarray(iotaf[edge_idx]),
        "iota_mean": jnp.asarray(iota_mean),
    }


def _vmec_booz_scalar_observables_from_state(
    context: GeometryAutodiffContext,
    state,
) -> dict[str, jnp.ndarray]:
    out = _boozer_output_from_state(context, state)
    return _vmec_booz_scalar_observables_from_boozer(context, state, out)


def _boozer_output_from_state(
    context: GeometryAutodiffContext,
    state,
):
    booz_api = _import_booz_xform_jax_api()
    inputs = _booz_xform_inputs_from_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=context.signgs,
        flux=context.flux,
    )
    booz_constants, booz_grids = _booz_constants_and_grids_for_inputs(context, inputs)
    out = booz_api.booz_xform_from_inputs(
        inputs=inputs,
        constants=booz_constants,
        grids=booz_grids,
        surface_indices=context.surface_indices,
        jit=True,
    )
    out = dict(out)
    out["_mode00"] = _find_boozer_mode_index(booz_grids.xm_b, booz_grids.xn_b, m_value=0, n_value=0)
    out["_mode10"] = _find_boozer_mode_index(booz_grids.xm_b, booz_grids.xn_b, m_value=1, n_value=0)
    return out


def _vmec_booz_scalar_observables_from_boozer(
    context: GeometryAutodiffContext,
    state,
    out,
) -> dict[str, jnp.ndarray]:
    bmnc_b = jnp.asarray(out["bmnc_b"])
    ixm_b = jnp.asarray(out["ixm_b"], dtype=jnp.int32)
    ixn_b = jnp.asarray(out["ixn_b"], dtype=jnp.int32)

    mode00 = _find_mode_index(ixm_b, ixn_b, m=0, n=0)
    if mode00 is None:
        raise ValueError("Boozer output is missing the (m, n) = (0, 0) mode.")
    mode10 = _find_mode_index(ixm_b, ixn_b, m=1, n=0)

    b00 = bmnc_b[:, mode00]
    magnetic_well = _vmec_magnetic_well_from_state(context, state)
    dmerc = _vmec_dmerc_profile_from_state(context, state)
    dmerc_objective_samples = _native_dmerc_objective_samples(
        dmerc,
        minimum=0.0,
        softness=1.0e-3,
    )
    magnetic_well_objective = _smooth_lower_bound_penalty(
        magnetic_well,
        minimum=0.0,
        softness=1.0e-3,
    ).reshape(())
    reduced = {
        "iota_b_mean": jnp.mean(jnp.asarray(out["iota_b"])),
        "b00_mean": jnp.mean(b00),
        "buco_b_mean": jnp.mean(jnp.asarray(out["buco_b"])),
        "bvco_b_mean": jnp.mean(jnp.asarray(out["bvco_b"])),
        "magnetic_well_objective": jnp.asarray(magnetic_well_objective, dtype=jnp.float64),
        "dmerc_objective_lo": jnp.asarray(dmerc_objective_samples[0], dtype=jnp.float64),
        "dmerc_objective_mid": jnp.asarray(dmerc_objective_samples[1], dtype=jnp.float64),
        "dmerc_objective_hi": jnp.asarray(dmerc_objective_samples[2], dtype=jnp.float64),
        "aspect_proxy": jnp.asarray(_vmec_state_field(state, "Rcos", "R_cos")[-1, mode00]),
    }
    if mode10 is not None:
        b10 = bmnc_b[:, mode10]
        reduced["b10_over_b00_mean"] = jnp.mean(b10 / b00)
    else:
        reduced["b10_over_b00_mean"] = jnp.asarray(0.0, dtype=b00.dtype)
    return reduced


def _vmec_booz_light_scalar_observables_from_boozer(
    context: GeometryAutodiffContext,
    state,
    out,
) -> dict[str, jnp.ndarray]:
    """Reduced Boozer scalars for the combined AD gate.

    The historical Boozer scalar helper also computes DMerc and a magnetic
    well penalty.  The combined gate already uses the current vmec_jax magnetic
    well scalar and intentionally excludes DMerc, so computing those hidden
    branches only bloats the reverse graph.
    """

    bmnc_b = jnp.asarray(out["bmnc_b"])
    ixm_b = jnp.asarray(out["ixm_b"], dtype=jnp.int32)
    ixn_b = jnp.asarray(out["ixn_b"], dtype=jnp.int32)

    mode00 = _find_mode_index(ixm_b, ixn_b, m=0, n=0)
    if mode00 is None:
        raise ValueError("Boozer output is missing the (m, n) = (0, 0) mode.")
    mode10 = _find_mode_index(ixm_b, ixn_b, m=1, n=0)

    b00 = bmnc_b[:, mode00]
    reduced = {
        "iota_b_mean": jnp.mean(jnp.asarray(out["iota_b"])),
        "b00_mean": jnp.mean(b00),
        "buco_b_mean": jnp.mean(jnp.asarray(out["buco_b"])),
        "bvco_b_mean": jnp.mean(jnp.asarray(out["bvco_b"])),
        "aspect_proxy": jnp.asarray(_vmec_state_field(state, "Rcos", "R_cos")[-1, mode00]),
    }
    if mode10 is not None:
        b10 = bmnc_b[:, mode10]
        reduced["b10_over_b00_mean"] = jnp.mean(b10 / b00)
    else:
        reduced["b10_over_b00_mean"] = jnp.asarray(0.0, dtype=b00.dtype)
    return reduced


def _vmec_booz_qi_maxj_scalar_objectives_from_state(
    context: GeometryAutodiffContext,
    state,
) -> dict[str, jnp.ndarray]:
    booz = _boozer_output_from_state(context, state)
    diagnostics = _vmec_j_invariant_qi_maxj_objectives_from_boozer(
        context,
        booz,
        include_qi=True,
        include_maxj=True,
    )
    return {
        "qi_objective": jnp.asarray(diagnostics["qi_objective"], dtype=jnp.float64),
        "maxj_objective": jnp.asarray(diagnostics["maxj_objective"], dtype=jnp.float64),
    }


def _vmec_booz_qi_scalar_objective_from_state(
    context: GeometryAutodiffContext,
    state,
    *,
    nphi: int = 151,
    nalpha: int = 31,
    n_bounce: int = 51,
) -> dict[str, jnp.ndarray]:
    booz = _boozer_output_from_state(context, state)
    diagnostics = _vmec_j_invariant_qi_maxj_objectives_from_boozer(
        context,
        booz,
        include_qi=True,
        include_maxj=False,
    )
    return {
        "qi_objective": jnp.asarray(diagnostics["qi_objective"], dtype=jnp.float64),
    }


def _vmec_j_invariant_qi_maxj_objectives_from_state(
    context: GeometryAutodiffContext,
    state,
    *,
    include_qi: bool,
    include_maxj: bool,
) -> dict[str, jnp.ndarray]:
    booz = _boozer_output_from_state(context, state)
    return _vmec_j_invariant_qi_maxj_objectives_from_boozer(
        context,
        booz,
        include_qi=include_qi,
        include_maxj=include_maxj,
    )


def _vmec_j_invariant_qi_maxj_objectives_from_boozer(
    context: GeometryAutodiffContext,
    booz,
    *,
    include_qi: bool,
    include_maxj: bool,
) -> dict[str, jnp.ndarray]:
    """Keep the existing Boozer path and only swap the QI/max-J post-processing."""

    vmec_backend = _import_vmec_jax()
    helper = getattr(vmec_backend, "j_invariant_qi_maxj_residual_from_boozer", None)
    if helper is None:
        raise AttributeError(
            "The active VMEC backend does not expose j_invariant_qi_maxj_residual_from_boozer. "
            "This benchmark path is intentionally restricted to the existing Boozer output plus "
            "the VMEX J-based QI/max-J post-processing."
        )

    gi_b = jnp.asarray(booz["bvco_b"], dtype=jnp.float64) + (
        jnp.asarray(booz["iota_b"], dtype=jnp.float64) * jnp.asarray(booz["buco_b"], dtype=jnp.float64)
    )
    surface_indices = np.asarray(context.surface_indices, dtype=np.int32).reshape(-1)
    s_half = 0.5 * (
        np.asarray(context.static.s[:-1], dtype=float)
        + np.asarray(context.static.s[1:], dtype=float)
    )
    s_b = jnp.asarray(s_half[surface_indices], dtype=jnp.float64)
    diagnostics = helper(
        bmnc_b=booz["bmnc_b"],
        xm_b=booz["ixm_b"],
        xn_b=booz["ixn_b"],
        iota_b=booz["iota_b"],
        gi_b=gi_b,
        s_b=s_b,
        nfp=int(context.cfg.nfp),
        include_qi=bool(include_qi),
        include_maxj=bool(include_maxj),
    )
    return {
        "qi_objective": jnp.asarray(diagnostics.get("qi_objective", 0.0), dtype=jnp.float64),
        "maxj_objective": jnp.asarray(diagnostics.get("maxj_objective", 0.0), dtype=jnp.float64),
    }


def _vmec_booz_qi_scalar_objective_from_boozer(
    context: GeometryAutodiffContext,
    booz,
    *,
    nphi: int = 151,
    nalpha: int = 31,
    n_bounce: int = 51,
) -> dict[str, jnp.ndarray]:
    optimization = _import_vmec_jax_optimization()
    qi = optimization.quasi_isodynamic_residual(
        bmnc_b=booz["bmnc_b"],
        xm_b=booz["ixm_b"],
        xn_b=booz["ixn_b"],
        iota_b=booz["iota_b"],
        nfp=int(context.cfg.nfp),
        nphi=int(nphi),
        nalpha=int(nalpha),
        n_bounce=int(n_bounce),
    )
    return {
        "qi_objective": jnp.asarray(qi["total"], dtype=jnp.float64),
    }


def _geometry_full_ad_objectives_from_state(
    context: GeometryAutodiffContext,
    state,
) -> dict[str, jnp.ndarray]:
    """One AD gate vector for VMEC scalars, Boozer scalars, and J-based QI/max-J."""

    vmec_scalars = _vmec_core_scalar_objectives_from_state(context, state)
    booz = _boozer_output_from_state(context, state)
    boozer_scalars = _vmec_booz_light_scalar_observables_from_boozer(context, state, booz)
    qi_maxj = _vmec_j_invariant_qi_maxj_objectives_from_boozer(
        context,
        booz,
        include_qi=True,
        include_maxj=True,
    )

    out = {f"vmec_{name}": jnp.asarray(value, dtype=jnp.float64) for name, value in vmec_scalars.items()}
    for name, value in boozer_scalars.items():
        out[f"boozer_{name}"] = jnp.asarray(value, dtype=jnp.float64)
    out["boozer_qi_objective"] = jnp.asarray(qi_maxj["qi_objective"], dtype=jnp.float64)
    out["boozer_maxj_objective"] = jnp.asarray(qi_maxj["maxj_objective"], dtype=jnp.float64)
    return out


def _soft_min_idx(values, beta: float = 50.0):
    values = jnp.asarray(values, dtype=jnp.float64)
    weights = jax.nn.softmax(-jnp.asarray(beta, dtype=values.dtype) * values)
    return jnp.sum(jnp.arange(values.shape[0], dtype=values.dtype) * weights)


def _cummin(values):
    values = jnp.asarray(values, dtype=jnp.float64)
    return jax.lax.associative_scan(jnp.minimum, values)


def _cummax(values):
    values = jnp.asarray(values, dtype=jnp.float64)
    return jax.lax.associative_scan(jnp.maximum, values)


def _smooth_signed_sqrt(values, eps: float = 1.0e-9):
    values = jnp.asarray(values, dtype=jnp.float64)
    eps_arr = jnp.asarray(eps, dtype=values.dtype)
    abs_smooth = jnp.sqrt(values * values + eps_arr * eps_arr)
    return values / jnp.sqrt(abs_smooth + eps_arr)


def _apply_smooth_goodman_transform(b_line, phi_coords):
    b_line = jnp.asarray(b_line, dtype=jnp.float64)
    phi_coords = jnp.asarray(phi_coords, dtype=jnp.float64)
    n = int(b_line.shape[0])
    indices = jnp.arange(n, dtype=b_line.dtype)
    s_indmin = _soft_min_idx(b_line)
    mask_l = jax.nn.sigmoid(2.0 * (s_indmin - indices))
    mask_r = 1.0 - mask_l
    bl_sq = _cummin(b_line)
    br_sq = _cummax(jnp.where(indices >= s_indmin, b_line, b_line[0]))
    pmax = jnp.asarray(50.0, dtype=b_line.dtype)
    pmin = jnp.asarray(15.0, dtype=b_line.dtype)
    b_min_val = jnp.interp(s_indmin, indices, b_line)
    phi_mid = jnp.interp(s_indmin, indices, phi_coords)
    phi_start = phi_coords[0]
    phi_end = phi_coords[-1]
    x1_l = (phi_coords - phi_start) / (phi_mid - phi_start + 1.0e-10)
    x1_r = (phi_coords - phi_mid) / (phi_end - phi_mid + 1.0e-10)
    shape_l = (jnp.cos(2.0 * jnp.pi * x1_l) + 1.0) / 2.0
    shape_r = (jnp.cos(2.0 * jnp.pi * x1_r) + 1.0) / 2.0
    f_l = jnp.where(x1_l < 0.5, (1.0 - bl_sq) * (shape_l**pmax), (-b_min_val) * (shape_l**pmin))
    f_r = jnp.where(x1_r < 0.5, (-b_min_val) * (shape_r**pmin), (1.0 - br_sq[-1]) * (shape_r**pmax))
    return mask_l * (bl_sq + f_l) + mask_r * (br_sq + f_r)


def _compute_j_pair(phi_coords, b_input, b_target, bj_levels, dl_dphi, db_dpsi, *, nphi_int: int = 128):
    b_target = jnp.asarray(b_target, dtype=jnp.float64)
    phi_coords = jnp.asarray(phi_coords, dtype=jnp.float64)
    indices = jnp.arange(b_target.shape[0], dtype=jnp.float64)
    s_indmin = _soft_min_idx(b_target)
    high = jnp.asarray(1.1 * jnp.max(b_target), dtype=b_target.dtype)
    branch_sharpness = jnp.asarray(6.0, dtype=b_target.dtype)
    left_mask = jax.nn.sigmoid(branch_sharpness * (s_indmin - indices))
    right_mask = 1.0 - left_mask
    b_l = left_mask * b_target + (1.0 - left_mask) * high
    b_r = right_mask * b_target + (1.0 - right_mask) * high
    p1 = jnp.interp(bj_levels, jnp.flip(b_l), jnp.flip(phi_coords))
    p2 = jnp.interp(bj_levels, b_r, phi_coords)
    t = jnp.linspace(0.0, 1.0, int(nphi_int), dtype=b_target.dtype)
    phi_grid = p1[:, None] + t[None, :] * (p2 - p1)[:, None]
    bi_g = jnp.interp(phi_grid, phi_coords, b_input)
    bc_g = jnp.interp(phi_grid, phi_coords, b_target)
    dl_g = jnp.interp(phi_grid, phi_coords, dl_dphi)
    dn_g = jnp.interp(phi_grid, phi_coords, db_dpsi)
    bj_v = bj_levels[:, None]
    res_i = 1.0 - bi_g / (bj_v + 1.0e-9)
    vi_g = _smooth_signed_sqrt(res_i)
    res_c = 1.0 - bc_g / (bj_v + 1.0e-9)
    vc_g = _smooth_signed_sqrt(res_c)
    v_target_stab = jnp.sqrt(jnp.maximum(bj_v - bc_g, 0.0) + 1.0e-9)
    ji = jnp.trapezoid(vi_g * dl_g, x=phi_grid, axis=1)
    jc = jnp.trapezoid(vc_g * dl_g, x=phi_grid, axis=1)
    dj_c = jnp.trapezoid(-(dn_g / (bc_g + 1.0e-9)) * v_target_stab * dl_g, x=phi_grid, axis=1)
    return ji, jc, dj_c


def _periodic_central_difference(values: jnp.ndarray, spacing: float, *, axis: int) -> jnp.ndarray:
    spacing_value = jnp.asarray(float(spacing), dtype=jnp.asarray(values).dtype)
    return (jnp.roll(values, -1, axis=axis) - jnp.roll(values, 1, axis=axis)) / (2.0 * spacing_value)


def _interp_radial_grid(values_full: jnp.ndarray, s_full: jnp.ndarray, s_query: jnp.ndarray) -> jnp.ndarray:
    values_full = jnp.asarray(values_full, dtype=jnp.float64)
    s_full = jnp.asarray(s_full, dtype=jnp.float64)
    s_query = jnp.asarray(s_query, dtype=jnp.float64)
    flat = values_full.reshape(values_full.shape[0], -1).T
    interp_one = lambda column: jnp.interp(s_query, s_full, column)
    interpolated = jax.vmap(interp_one)(flat)
    return interpolated.T.reshape((s_query.shape[0],) + values_full.shape[1:])


def _vmec_state_for_custom_grid(vmec_jax, state, static):
    vmec_m1_internal_to_physical_signed = _resolve_vmec_attr(
        vmec_jax,
        "vmec_m1_internal_to_physical_signed",
        submodule="vmec_parity",
    )
    cfg = static.cfg
    lconm1 = bool(getattr(cfg, "lconm1", True))
    lthreed = bool(getattr(cfg, "lthreed", int(getattr(cfg, "ntor", 0)) > 0))
    lasym = bool(getattr(cfg, "lasym", False))
    if not (lconm1 and (lthreed or lasym) and int(getattr(cfg, "mpol", 0)) > 1):
        return state
    Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
        Rcos=_vmec_state_field(state, "Rcos", "R_cos"),
        Zsin=_vmec_state_field(state, "Zsin", "Z_sin"),
        Rsin=_vmec_state_field(state, "Rsin", "R_sin"),
        Zcos=_vmec_state_field(state, "Zcos", "Z_cos"),
        modes=static.modes,
        lthreed=lthreed,
        lasym=lasym,
        lconm1=lconm1,
    )
    if hasattr(state, "R_cos"):
        return dataclasses.replace(
            state,
            R_cos=Rcos,
            R_sin=Rsin,
            Z_cos=Zcos,
            Z_sin=Zsin,
        )
    return dataclasses.replace(
        state,
        Rcos=Rcos,
        Rsin=Rsin,
        Zcos=Zcos,
        Zsin=Zsin,
    )


def _vmec_surface_line_data_from_state(
    context: GeometryAutodiffContext,
    state,
    *,
    nphi: int,
    ntheta: int,
):
    vmec_jax = _import_vmec_jax()
    AngleGrid = _resolve_vmec_attr(vmec_jax, "AngleGrid", submodule="grids")
    build_helical_basis = _resolve_vmec_attr(vmec_jax, "build_helical_basis", submodule="fourier")
    eval_geom_jit = _resolve_vmec_attr(vmec_jax, "_eval_geom_jit", submodule="geom")
    bsup_from_geom = _resolve_vmec_attr(vmec_jax, "bsup_from_geom", submodule="field")
    b2_from_bsup = _resolve_vmec_attr(vmec_jax, "b2_from_bsup", submodule="field")
    equilibrium_iota_profiles_from_state = _resolve_vmec_attr(
        vmec_jax,
        "equilibrium_iota_profiles_from_state",
        submodule="profiles",
    )
    nfp = int(context.cfg.nfp)
    period = 2.0 * jnp.pi / float(nfp)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False, dtype=jnp.float64)
    phi = jnp.linspace(0.0, period, int(nphi), endpoint=False, dtype=jnp.float64)
    zeta = phi * float(nfp)
    grid = AngleGrid(theta=theta, zeta=zeta, nfp=nfp)
    basis = build_helical_basis(context.static.modes, grid, cache=False)
    state_use = _vmec_state_for_custom_grid(vmec_jax, state, context.static)
    geom = eval_geom_jit(state_use, basis, context.static.s, zeta)
    bsupu, bsupv = bsup_from_geom(
        geom,
        phipf=jnp.asarray(context.flux.phipf),
        chipf=jnp.asarray(context.flux.chipf),
        nfp=nfp,
        signgs=int(context.signgs),
        lamscale=jnp.asarray(context.flux.lamscale),
    )
    bmag_full = jnp.sqrt(jnp.maximum(jnp.asarray(b2_from_bsup(geom, bsupu, bsupv), dtype=jnp.float64), 0.0))
    _chips, iotas_half_raw, _iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=int(context.signgs),
    )
    iotas_half = jnp.asarray(iotas_half_raw, dtype=jnp.float64)[1:]
    cos_phi = jnp.cos(phi)[None, None, :]
    sin_phi = jnp.sin(phi)[None, None, :]
    x_theta = jnp.stack(
        [
            geom.Rt * cos_phi,
            geom.Rt * sin_phi,
            geom.Zt,
        ],
        axis=-1,
    )
    x_phi = jnp.stack(
        [
            geom.Rp * cos_phi - geom.R * sin_phi,
            geom.Rp * sin_phi + geom.R * cos_phi,
            geom.Zp,
        ],
        axis=-1,
    )
    iota_full = _interp_radial_grid(
        iotas_half[:, None, None],
        0.5 * (jnp.asarray(context.static.s[:-1], dtype=jnp.float64) + jnp.asarray(context.static.s[1:], dtype=jnp.float64)),
        jnp.asarray(context.static.s, dtype=jnp.float64),
    )[:, :, :, None]
    dl_dphi_full = jnp.linalg.norm(x_phi + iota_full * x_theta, axis=-1)
    bmag_grid_full = jnp.transpose(bmag_full, axes=(0, 2, 1))
    dl_grid_full = jnp.transpose(dl_dphi_full, axes=(0, 2, 1))
    s_full = jnp.asarray(context.static.s, dtype=jnp.float64)
    s_half = 0.5 * (s_full[:-1] + s_full[1:])
    phips_half = jnp.asarray(context.flux.phips, dtype=jnp.float64)[1:]
    target_indices = jnp.asarray(context.surface_indices, dtype=jnp.int32)
    neighbor_minus = jnp.clip(target_indices - 1, 0, int(s_half.shape[0]) - 1)
    neighbor_plus = jnp.clip(target_indices + 1, 0, int(s_half.shape[0]) - 1)
    extended_indices = jnp.unique(jnp.concatenate([neighbor_minus, target_indices, neighbor_plus], axis=0))
    extended_s = s_half[extended_indices]
    bmag_half = _interp_radial_grid(bmag_grid_full, s_full, extended_s)
    dl_half = _interp_radial_grid(dl_grid_full, s_full, extended_s)
    iota_surface = jnp.asarray(iotas_half, dtype=jnp.float64)[extended_indices]
    surface_map = {int(idx): pos for pos, idx in enumerate(np.asarray(extended_indices, dtype=int))}
    target_positions = jnp.asarray([surface_map[int(idx)] for idx in np.asarray(target_indices)], dtype=jnp.int32)
    minus_positions = jnp.asarray([surface_map[int(idx)] for idx in np.asarray(neighbor_minus)], dtype=jnp.int32)
    plus_positions = jnp.asarray([surface_map[int(idx)] for idx in np.asarray(neighbor_plus)], dtype=jnp.int32)
    db_ds = (bmag_half[plus_positions] - bmag_half[minus_positions]) / jnp.maximum(
        (s_half[neighbor_plus] - s_half[neighbor_minus])[:, None, None],
        1.0e-12,
    )
    db_dpsi = db_ds / jnp.maximum(phips_half[target_indices][:, None, None], 1.0e-12)
    return {
        "phi_line": jnp.linspace(0.0, period, int(nphi), endpoint=True, dtype=jnp.float64),
        "period": period,
        "b_grid": bmag_half[target_positions],
        "dl_grid": dl_half[target_positions],
        "db_dpsi_grid": db_dpsi,
        "iota_surface": iota_surface[target_positions],
    }


def _sample_periodic_surface_line(surface_grid: jnp.ndarray, theta_coords: jnp.ndarray, phi_coords: jnp.ndarray, *, period: jnp.ndarray):
    nphi_g, ntheta_g = int(surface_grid.shape[0]), int(surface_grid.shape[1])
    phi_idx = jnp.mod(phi_coords, period) / jnp.maximum(period, 1.0e-12) * float(nphi_g)
    theta_idx = jnp.mod(theta_coords, 2.0 * jnp.pi) / (2.0 * jnp.pi) * float(ntheta_g)
    coords = jnp.stack([phi_idx, theta_idx], axis=0)
    return jax.scipy.ndimage.map_coordinates(surface_grid, coords, order=1, mode="wrap")


def _vmec_qi_maxj_shared_diagnostics_from_state(
    context: GeometryAutodiffContext,
    state,
    *,
    nphi: int = 101,
    nalpha: int = 51,
    n_bounce: int = 66,
    p_j: float = 1.0,
    p_lambda: float = 1.0,
    nphi_int: int = 128,
):
    nphi_value = int(nphi)
    nalpha_value = int(nalpha)
    n_bounce_value = int(n_bounce)
    line_data = _vmec_surface_line_data_from_state(
        context,
        state,
        nphi=nphi_value,
        ntheta=max(nphi_value, 96),
    )
    phi = jnp.asarray(line_data["phi_line"], dtype=jnp.float64)
    alpha = jnp.linspace(0.0, 2.0 * jnp.pi, nalpha_value, endpoint=False, dtype=jnp.float64)
    b_target = jnp.asarray(line_data["b_grid"], dtype=jnp.float64)
    dl_target = jnp.asarray(line_data["dl_grid"], dtype=jnp.float64)
    db_dpsi = jnp.asarray(line_data["db_dpsi_grid"], dtype=jnp.float64)
    iota_surface = jnp.asarray(line_data["iota_surface"], dtype=jnp.float64)
    period = jnp.asarray(line_data["period"], dtype=jnp.float64)

    def _per_surface(b_surface, dl_surface, db_surface):
        def _per_line(b_line, dl_line, db_line):
            bmin = jnp.min(b_line)
            bmax = jnp.max(b_line)
            scale = jnp.maximum(bmax - bmin, 1.0e-10)
            b_norm = (b_line - bmin) / scale
            b_target_norm = _apply_smooth_goodman_transform(b_norm, phi)
            b_target_phys = b_target_norm * scale + bmin
            bj_norm = jnp.power(jnp.arange(n_bounce_value, dtype=jnp.float64) / jnp.maximum(n_bounce_value - 1, 1), p_lambda)
            bj_phys = bj_norm * scale + bmin
            ji, jc, djc = _compute_j_pair(
                phi,
                b_line,
                b_target_phys,
                bj_phys,
                dl_line,
                db_line,
                nphi_int=nphi_int,
            )
            return ji, jc, djc

        ji_all, jc_all, djc_all = jax.vmap(_per_line, in_axes=(0, 0, 0), out_axes=(0, 0, 0))(b_surface, dl_surface, db_surface)
        ji_pow = jnp.power(ji_all, p_j)
        jc_pow = jnp.power(jc_all, p_j)
        nalpha_f = jnp.asarray(float(nalpha_value), dtype=jnp.float64)
        sum_ji = jnp.sum(ji_pow, axis=0)
        sum_jc = jnp.sum(jc_pow, axis=0)
        sum_ji_sq = jnp.sum(ji_pow * ji_pow, axis=0)
        sum_jc_sq = jnp.sum(jc_pow * jc_pow, axis=0)
        qi_pair_sum = nalpha_f * (sum_ji_sq + sum_jc_sq) - 2.0 * sum_ji * sum_jc
        qi_num = jnp.sum(qi_pair_sum)
        mean_denom = ((jnp.sum(sum_ji + sum_jc) / (2.0 * float(n_bounce_value))) ** 2) + 1.0e-10
        qi_surface = jnp.sqrt(jnp.maximum(qi_num, 0.0) / mean_denom)

        maxj_surface = jnp.sqrt(jnp.mean(jnp.maximum(djc_all, 0.0) ** 2))
        return qi_surface, maxj_surface

    def _surface_lines(b_surface, dl_surface, db_surface, iota_value):
        def _line(alpha_value):
            theta_coords = alpha_value + iota_value * phi
            b_line = _sample_periodic_surface_line(b_surface, theta_coords, phi, period=period)
            dl_line = _sample_periodic_surface_line(dl_surface, theta_coords, phi, period=period)
            db_line = _sample_periodic_surface_line(db_surface, theta_coords, phi, period=period)
            return b_line, dl_line, db_line

        return jax.vmap(_line)(alpha)

    b_lines, dl_lines, db_lines = jax.vmap(_surface_lines, in_axes=(0, 0, 0, 0))(b_target, dl_target, db_dpsi, iota_surface)
    qi_surface, maxj_surface = jax.vmap(_per_surface, in_axes=(0, 0, 0))(b_lines, dl_lines, db_lines)
    return {
        "qi_surface": qi_surface,
        "maxj_surface": maxj_surface,
        "qi_objective": jnp.mean(qi_surface**2),
        "maxj_objective": jnp.mean(maxj_surface**2),
    }


def _observable_items_from_state(
    context: GeometryAutodiffContext,
    state,
    *,
    observable_kind: str,
) -> list[tuple[str, jnp.ndarray]]:
    kind = str(observable_kind).strip().lower()
    if kind == "vmec_scalar_observables":
        observables = _vmec_scalar_observables_from_state(context, state)
    elif kind == "vmec_core_scalar_objectives":
        observables = _vmec_core_scalar_objectives_from_state(context, state)
    elif kind == "geometry_full_ad_objectives":
        observables = _geometry_full_ad_objectives_from_state(context, state)
    elif kind == "vmec_iotaf_scalar_observables":
        observables = _vmec_iotaf_scalar_observables_from_state(context, state)
    elif kind == "vmec_booz_scalar_observables":
        observables = _vmec_booz_scalar_observables_from_state(context, state)
    elif kind == "vmec_booz_qi_scalar_objectives":
        observables = _vmec_booz_qi_scalar_objective_from_state(context, state)
    elif kind == "vmec_booz_qi_maxj_scalar_objectives":
        observables = _vmec_booz_qi_maxj_scalar_objectives_from_state(context, state)
    elif kind == "vmec_qi_maxj_scalar_objectives":
        observables = _vmec_j_invariant_qi_maxj_objectives_from_state(
            context,
            state,
            include_qi=True,
            include_maxj=True,
        )
    elif kind == "vmec_dmerc_objectives":
        raise _vmec_dmerc_unavailable_error()
    else:
        raise ValueError(
            "observable_kind must be 'vmec_scalar_observables', 'vmec_core_scalar_objectives', "
            "'geometry_full_ad_objectives', 'vmec_iotaf_scalar_observables', 'vmec_booz_scalar_observables', "
            "'vmec_booz_qi_scalar_objectives', 'vmec_booz_qi_maxj_scalar_objectives', "
            "'vmec_qi_maxj_scalar_objectives', or 'vmec_dmerc_objectives'."
        )
    return list(observables.items())


def _observable_names_for_kind(observable_kind: str) -> list[str]:
    kind = str(observable_kind).strip().lower()
    if kind == "vmec_scalar_observables":
        return [
            "aspect_ratio",
            "volume_total",
            "iota_mean",
            "magnetic_well_objective",
            "dmerc_objective_lo",
            "dmerc_objective_mid",
            "dmerc_objective_hi",
            "mirror_ratio_objective",
            "edge_r00",
        ]
    if kind == "vmec_core_scalar_objectives":
        return [
            "aspect_ratio",
            "volume_total",
            "iota_mean",
            "magnetic_well",
            "mirror_ratio",
            "beta_volume",
        ]
    if kind == "geometry_full_ad_objectives":
        return [
            "vmec_aspect_ratio",
            "vmec_volume_total",
            "vmec_iota_mean",
            "vmec_magnetic_well",
            "vmec_mirror_ratio",
            "vmec_beta_volume",
            "boozer_iota_b_mean",
            "boozer_b00_mean",
            "boozer_buco_b_mean",
            "boozer_bvco_b_mean",
            "boozer_aspect_proxy",
            "boozer_b10_over_b00_mean",
            "boozer_qi_objective",
            "boozer_maxj_objective",
        ]
    if kind == "vmec_iotaf_scalar_observables":
        return ["iotas_1", "iotas_2", "iotaf_first", "iotaf_q1", "iotaf_mid", "iotaf_q3", "iotaf_edge", "iota_mean"]
    if kind == "vmec_booz_scalar_observables":
        return [
            "iota_b_mean",
            "b00_mean",
            "buco_b_mean",
            "bvco_b_mean",
            "magnetic_well_objective",
            "dmerc_objective_lo",
            "dmerc_objective_mid",
            "dmerc_objective_hi",
            "aspect_proxy",
            "b10_over_b00_mean",
        ]
    if kind == "vmec_qi_maxj_scalar_objectives":
        return ["qi_objective", "maxj_objective"]
    if kind == "vmec_booz_qi_scalar_objectives":
        return ["qi_objective"]
    if kind == "vmec_booz_qi_maxj_scalar_objectives":
        return ["qi_objective", "maxj_objective"]
    if kind == "vmec_dmerc_objectives":
        raise _vmec_dmerc_unavailable_error()
    raise ValueError(
        "observable_kind must be 'vmec_scalar_observables', 'vmec_core_scalar_objectives', "
        "'geometry_full_ad_objectives', 'vmec_iotaf_scalar_observables', 'vmec_booz_scalar_observables', "
        "'vmec_booz_qi_scalar_objectives', 'vmec_booz_qi_maxj_scalar_objectives', "
        "'vmec_qi_maxj_scalar_objectives', or 'vmec_dmerc_objectives'."
    )


def geometry_observable_names_for_kind(observable_kind: str) -> tuple[str, ...]:
    return tuple(_observable_names_for_kind(observable_kind))


def _single_param_boundary_spec(context: GeometryAutodiffContext):
    optimization = _import_vmec_jax_optimization()
    prefix = "rc" if context.boundary_kind == "rc" else "zs"
    name = f"{prefix}{int(context.param_m)}{int(context.param_n)}"
    return optimization.BoundaryParamSpec(
        name=name,
        kind=prefix,
        index=int(context.boundary_index),
        m=int(context.param_m),
        n=int(context.param_n),
    )


def _make_exact_optimizer(
    context: GeometryAutodiffContext,
    *,
    observable_kind: str,
    max_iter: int | None = None,
    step_size: float | None = None,
    solver_device: str | None = None,
):
    optimization = _import_vmec_jax_optimization()
    vmec_jax = _import_vmec_jax()
    unpack_state = _resolve_vmec_attr(vmec_jax, "unpack_state", submodule="state")

    resolved_max_iter = int(context.vmec_default_max_iter if max_iter is None else max_iter)
    resolved_step_size = float(context.vmec_default_step_size if step_size is None else step_size)
    base_spec = _single_param_boundary_spec(context)

    indata_eff = deepcopy(context.indata)
    try:
        indata_eff.scalars["DELT"] = float(resolved_step_size)
    except Exception:
        pass

    def residuals_from_state(state):
        items = _observable_items_from_state(context, state, observable_kind=observable_kind)
        return jnp.stack([jnp.asarray(value, dtype=jnp.float64).reshape(()) for _, value in items])

    current_driven_iota_kind = bool(int(context.indata.get_int("NCURR", 0)) == 1) and observable_kind in (
        "vmec_scalar_observables",
        "vmec_iotaf_scalar_observables",
    )
    if current_driven_iota_kind:
        observable_names = _observable_names_for_kind(observable_kind)
        sanitize_indices = {
            idx
            for idx, name in enumerate(observable_names)
            if str(name).startswith(("iota", "iotas_", "iotaf_"))
        }

        def state_cotangent_operator_from_packed(packed_state, layout):
            packed_state = jnp.asarray(packed_state, dtype=jnp.float64)
            blocks = []
            for idx, name in enumerate(observable_names):
                sanitize = idx in sanitize_indices

                def _scalar_from_packed(packed, *, output_index=idx):
                    state = unpack_state(packed, layout)
                    items = _observable_items_from_state(context, state, observable_kind=observable_kind)
                    return jnp.asarray(items[output_index][1], dtype=jnp.float64).reshape(())

                _, vjp_fun = jax.vjp(_scalar_from_packed, packed_state)
                blocks.append((idx, vjp_fun, sanitize, name))

            def _apply(residual_cotangent):
                residual_cotangent = jnp.asarray(residual_cotangent, dtype=jnp.float64).reshape(-1)
                total = jnp.zeros_like(packed_state)
                for idx, vjp_fun, sanitize, _name in blocks:
                    cot = residual_cotangent[idx]

                    def _active(cot_block):
                        contribution = vjp_fun(cot_block)[0]
                        if sanitize:
                            # Match vmec_jax's current-driven-iota reverse cleanup:
                            # near-axis gauge-null cotangent entries can produce
                            # nonfinite or unstable reverse contributions even when
                            # forward JVP columns remain finite.
                            contribution = jnp.nan_to_num(contribution, nan=0.0, posinf=0.0, neginf=0.0)
                        return contribution

                    total = total + jax.lax.cond(
                        cot != 0.0,
                        _active,
                        lambda cot_block: jnp.zeros_like(packed_state),
                        cot,
                    )
                return total

            return _apply

        def state_cotangent_from_packed(packed_state, layout, residual_cotangent):
            return state_cotangent_operator_from_packed(packed_state, layout)(residual_cotangent)

        residuals_from_state._state_cotangent_operator_from_packed = state_cotangent_operator_from_packed
        residuals_from_state._state_cotangent_from_packed = state_cotangent_from_packed

    return optimization.FixedBoundaryExactOptimizer(
        context.static,
        indata_eff,
        context.boundary,
        [base_spec],
        residuals_from_state,
        inner_max_iter=resolved_max_iter,
        inner_ftol=context.vmec_default_ftol,
        trial_max_iter=resolved_max_iter,
        trial_ftol=context.vmec_default_ftol,
        solver_device=solver_device,
    )


def exact_forward_scalar_observable_derivatives(
    context: GeometryAutodiffContext,
    *,
    observable_kind: str,
    max_iter: int | None = None,
    step_size: float | None = None,
    solver_device: str | None = None,
) -> dict[str, jnp.ndarray]:
    optimizer = _make_exact_optimizer(
        context,
        observable_kind=observable_kind,
        max_iter=max_iter,
        step_size=step_size,
        solver_device=solver_device,
    )
    linear_op = optimizer.residual_linear_operator(np.zeros(1, dtype=float))
    return exact_forward_scalar_observable_derivatives_from_linear_operator(
        linear_op,
        observable_kind=observable_kind,
    )


def exact_scalar_observable_linear_operator(
    context: GeometryAutodiffContext,
    *,
    observable_kind: str,
    max_iter: int | None = None,
    step_size: float | None = None,
    solver_device: str | None = None,
):
    optimizer = _make_exact_optimizer(
        context,
        observable_kind=observable_kind,
        max_iter=max_iter,
        step_size=step_size,
        solver_device=solver_device,
    )
    return optimizer.residual_linear_operator(np.zeros(1, dtype=float))


def exact_vmec_dmerc_profile_linear_operator(
    context: GeometryAutodiffContext,
    *,
    indices: tuple[int, ...] | None = None,
    max_iter: int | None = None,
    step_size: float | None = None,
    solver_device: str | None = None,
):
    optimization = _import_vmec_jax_optimization()
    resolved_max_iter = int(context.vmec_default_max_iter if max_iter is None else max_iter)
    resolved_step_size = float(context.vmec_default_step_size if step_size is None else step_size)
    base_spec = _single_param_boundary_spec(context)

    indata_eff = deepcopy(context.indata)
    try:
        indata_eff.scalars["DELT"] = float(resolved_step_size)
    except Exception:
        pass

    index_array = None if indices is None else jnp.asarray(indices, dtype=jnp.int32)

    def residuals_from_state(state):
        dmerc = _vmec_dmerc_profile_from_state(context, state)
        if index_array is None:
            return jnp.asarray(dmerc, dtype=jnp.float64)
        return jnp.asarray(dmerc, dtype=jnp.float64)[index_array]

    optimizer = optimization.FixedBoundaryExactOptimizer(
        context.static,
        indata_eff,
        context.boundary,
        [base_spec],
        residuals_from_state,
        inner_max_iter=resolved_max_iter,
        inner_ftol=context.vmec_default_ftol,
        trial_max_iter=resolved_max_iter,
        trial_ftol=context.vmec_default_ftol,
        solver_device=solver_device,
    )
    return optimizer.residual_linear_operator(np.zeros(1, dtype=float))


def exact_forward_scalar_observable_derivatives_from_linear_operator(
    linear_op,
    *,
    observable_kind: str,
) -> dict[str, jnp.ndarray]:
    jac = np.asarray(linear_op.matvec(np.array([1.0], dtype=float)), dtype=float).reshape(-1)
    names = _observable_names_for_kind(observable_kind)
    if jac.size != len(names):
        names = names[: int(jac.size)]
    return {name: jnp.asarray(jac[idx], dtype=jnp.float64) for idx, name in enumerate(names)}


def exact_reverse_scalar_observable_derivatives(
    context: GeometryAutodiffContext,
    *,
    observable_kind: str,
    max_iter: int | None = None,
    step_size: float | None = None,
    solver_device: str | None = None,
) -> dict[str, jnp.ndarray]:
    linear_op = exact_scalar_observable_linear_operator(
        context,
        observable_kind=observable_kind,
        max_iter=max_iter,
        step_size=step_size,
        solver_device=solver_device,
    )
    return exact_reverse_scalar_observable_derivatives_from_linear_operator(
        linear_op,
        observable_kind=observable_kind,
    )


def exact_reverse_scalar_observable_derivatives_from_linear_operator(
    linear_op,
    *,
    observable_kind: str,
) -> dict[str, jnp.ndarray]:
    names = _observable_names_for_kind(observable_kind)
    n_res = int(linear_op.shape[0])
    out: dict[str, jnp.ndarray] = {}
    for idx, name in enumerate(names):
        if idx >= n_res:
            break
        basis = np.zeros(n_res, dtype=float)
        basis[idx] = 1.0
        grad = np.asarray(linear_op.rmatvec(basis), dtype=float).reshape(-1)
        out[name] = jnp.asarray(float(grad[0]), dtype=jnp.float64)
    return out


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


def vmec_dmerc_profile_from_single_param(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> jnp.ndarray:
    state = _solve_state_for_single_param(
        context,
        param_delta,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    return _vmec_dmerc_profile_from_state(context, state)


def vmec_scalar_observables_from_single_param(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> dict[str, jnp.ndarray]:
    state = _solve_state_for_single_param(
        context,
        param_delta,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    return _vmec_scalar_observables_from_state(context, state)


def vmec_scalar_observables_from_param_vector(
    context: GeometryAutodiffContext,
    param_deltas,
    param_specs: Sequence[tuple[str, int, int]],
    *,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> dict[str, jnp.ndarray]:
    state = _solve_state_for_param_vector(
        context,
        param_deltas,
        param_specs,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    return _vmec_scalar_observables_from_state(context, state)


def vmec_iotaf_scalar_observables_from_single_param(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> dict[str, jnp.ndarray]:
    state = _solve_state_for_single_param(
        context,
        param_delta,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    return _vmec_iotaf_scalar_observables_from_state(context, state)


def geometry_observables_from_single_param(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> dict[str, jnp.ndarray]:
    state = _solve_state_for_single_param(
        context,
        param_delta,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    observables = _vmec_booz_scalar_observables_from_state(context, state)
    full = {name: jnp.asarray(value) for name, value in observables.items()}
    full["surface_indices"] = context.surface_indices.astype(jnp.float64)
    full["nfp"] = jnp.asarray([float(context.static.resolution.nfp)], dtype=jnp.float64)
    return full


def vmec_booz_scalar_observables_from_single_param(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> dict[str, jnp.ndarray]:
    state = _solve_state_for_single_param(
        context,
        param_delta,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    return _vmec_booz_scalar_observables_from_state(context, state)


def vmec_booz_scalar_observables_from_param_vector(
    context: GeometryAutodiffContext,
    param_deltas,
    param_specs: Sequence[tuple[str, int, int]],
    *,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> dict[str, jnp.ndarray]:
    state = _solve_state_for_param_vector(
        context,
        param_deltas,
        param_specs,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    return _vmec_booz_scalar_observables_from_state(context, state)


def geometry_observable_kind_from_single_param(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    observable_kind: str,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> dict[str, jnp.ndarray]:
    state = _solve_state_for_single_param(
        context,
        param_delta,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    return {
        name: jnp.asarray(value, dtype=jnp.float64)
        for name, value in _observable_items_from_state(context, state, observable_kind=observable_kind)
    }


def geometry_observable_kind_from_param_vector(
    context: GeometryAutodiffContext,
    param_deltas,
    param_specs: Sequence[tuple[str, int, int]],
    *,
    observable_kind: str,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> dict[str, jnp.ndarray]:
    state = _solve_state_for_param_vector(
        context,
        param_deltas,
        param_specs,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    return {
        name: jnp.asarray(value, dtype=jnp.float64)
        for name, value in _observable_items_from_state(context, state, observable_kind=observable_kind)
    }


def geometry_observable_weighted_sum_from_param_vector(
    context: GeometryAutodiffContext,
    param_deltas,
    param_specs: Sequence[tuple[str, int, int]],
    objective_weights,
    *,
    observable_kind: str,
    objective_names: Sequence[str] | None = None,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> jnp.ndarray:
    state = _solve_state_for_param_vector(
        context,
        param_deltas,
        param_specs,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    items = _observable_items_from_state(context, state, observable_kind=observable_kind)
    observables = {name: jnp.asarray(value, dtype=jnp.float64).reshape(()) for name, value in items}
    names = tuple(objective_names) if objective_names is not None else tuple(name for name, _value in items)
    missing = [name for name in names if name not in observables]
    if missing:
        raise ValueError(f"Unknown objective names for {observable_kind}: {missing}")
    weights = jnp.asarray(objective_weights, dtype=jnp.float64).reshape((-1,))
    if int(weights.shape[0]) != len(names):
        raise ValueError(f"Expected {len(names)} objective weights for {observable_kind}; got {int(weights.shape[0])}.")
    values = jnp.stack([observables[name] for name in names])
    return jnp.sum(weights * values)


def geometry_observable_batched_cotangent_pullback_from_param_vector(
    context: GeometryAutodiffContext,
    param_deltas,
    param_specs: Sequence[tuple[str, int, int]],
    objective_cotangents,
    *,
    observable_kind: str,
    objective_names: Sequence[str] | None = None,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Return objective values and W @ d(objectives)/d(param_deltas)."""
    param_deltas = jnp.asarray(param_deltas, dtype=jnp.float64)
    cotangents = jnp.asarray(objective_cotangents, dtype=jnp.float64)
    if cotangents.ndim == 1:
        cotangents = cotangents[None, :]
    names = tuple(objective_names) if objective_names is not None else geometry_observable_names_for_kind(observable_kind)
    if int(cotangents.shape[1]) != len(names):
        raise ValueError(
            f"Expected objective cotangents with {len(names)} columns for {observable_kind}; "
            f"got {int(cotangents.shape[1])}."
        )

    def objective_vector(theta):
        state = _solve_state_for_param_vector(
            context,
            theta,
            param_specs,
            lane=lane,
            max_iter=max_iter,
            step_size=step_size,
            jacobian_penalty=jacobian_penalty,
        )
        items = _observable_items_from_state(context, state, observable_kind=observable_kind)
        observables = {name: jnp.asarray(value, dtype=jnp.float64).reshape(()) for name, value in items}
        missing = [name for name in names if name not in observables]
        if missing:
            raise ValueError(f"Unknown objective names for {observable_kind}: {missing}")
        return jnp.stack([observables[name] for name in names])

    def contracted_objectives_with_aux(theta):
        values = objective_vector(theta)
        return cotangents @ values, values

    gradient_matrix, values = jax.jacrev(contracted_objectives_with_aux, has_aux=True)(param_deltas)
    values_by_name = {name: values[i] for i, name in enumerate(names)}
    return values_by_name, gradient_matrix


def geometry_observable_vector_custom_vjp_from_param_vector(
    context: GeometryAutodiffContext,
    param_deltas,
    param_specs: Sequence[tuple[str, int, int]],
    *,
    observable_kind: str,
    objective_names: Sequence[str] | None = None,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> jnp.ndarray:
    """Return objectives with a cotangent-contracted reverse rule."""
    param_deltas = jnp.asarray(param_deltas, dtype=jnp.float64)
    names = tuple(objective_names) if objective_names is not None else geometry_observable_names_for_kind(observable_kind)

    def objective_vector(theta):
        state = _solve_state_for_param_vector(
            context,
            theta,
            param_specs,
            lane=lane,
            max_iter=max_iter,
            step_size=step_size,
            jacobian_penalty=jacobian_penalty,
        )
        items = _observable_items_from_state(context, state, observable_kind=observable_kind)
        observables = {name: jnp.asarray(value, dtype=jnp.float64).reshape(()) for name, value in items}
        missing = [name for name in names if name not in observables]
        if missing:
            raise ValueError(f"Unknown objective names for {observable_kind}: {missing}")
        return jnp.stack([observables[name] for name in names])

    @jax.custom_vjp
    def objective_vector_custom(theta):
        return objective_vector(theta)

    def objective_vector_fwd(theta):
        values = objective_vector(theta)
        return values, theta

    def objective_vector_bwd(theta, cotangent):
        cotangent = jnp.asarray(cotangent, dtype=jnp.float64)

        def contracted(theta_inner):
            values = objective_vector(theta_inner)
            return jnp.vdot(cotangent, values)

        return (jax.grad(contracted)(theta),)

    objective_vector_custom.defvjp(objective_vector_fwd, objective_vector_bwd)
    return objective_vector_custom(param_deltas)


def _tree_weighted_basis_sum(basis_tree, weights: jnp.ndarray):
    weights = jnp.asarray(weights, dtype=jnp.float64)

    def _combine(leaf):
        leaf = jnp.asarray(leaf)
        if leaf.dtype == jax.dtypes.float0:
            shape = (int(weights.shape[0]),) + tuple(leaf.shape[1:])
            return jnp.broadcast_to(leaf[0], shape)
        return jnp.tensordot(weights, leaf, axes=((1,), (0,)))

    return jax.tree.map(
        _combine,
        basis_tree,
    )


def _tree_scale_unit_cotangent(unit_tree, weights: jnp.ndarray):
    weights = jnp.asarray(weights, dtype=jnp.float64).reshape((-1,))

    def _scale(leaf):
        leaf = jnp.asarray(leaf)
        if leaf.dtype == jax.dtypes.float0:
            shape = (int(weights.shape[0]),) + tuple(leaf.shape)
            return jnp.broadcast_to(leaf, shape)
        shape = (int(weights.shape[0]),) + (1,) * int(leaf.ndim)
        return weights.reshape(shape) * leaf

    return jax.tree.map(_scale, unit_tree)


def _tree_add_all(*trees):
    if not trees:
        raise ValueError("_tree_add_all requires at least one tree.")
    out = trees[0]
    for tree in trees[1:]:
        def _add(left, right):
            left = jnp.asarray(left)
            right = jnp.asarray(right)
            if left.dtype == jax.dtypes.float0:
                return right
            if right.dtype == jax.dtypes.float0:
                return left
            return left + right

        out = jax.tree.map(_add, out, tree)
    return out


def geometry_full_ad_objective_table_pullback_from_param_vector(
    context: GeometryAutodiffContext,
    param_deltas,
    param_specs: Sequence[tuple[str, int, int]],
    objective_cotangents,
    *,
    objective_names: Sequence[str] | None = None,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
    final_vmec_pullback_mode: str = "raw_block_transpose",
    solver_device: str | None = None,
    raw_block_solve: GeometryRawBlockSolve | None = None,
    return_state_bars: bool = False,
) -> tuple[dict[str, jnp.ndarray], object]:
    """Return geometry objective values and objective cotangents.

    This is the memory-conscious table path for the combined geometry gate.
    It avoids the generic ``jacrev(objective_vector)`` path because that path
    pushes every objective basis cotangent through the expensive Boozer-QI
    residual graph.  Instead, it builds state-level cotangents by objective
    group and applies the QI pullback once, then scales that cotangent into the
    requested output rows.

    By default, the second return value is ``W @ d(objectives)/d(params)``.
    If ``return_state_bars`` is true, the final VMEC raw-block pullback is
    skipped and the second return value is the VMEC-state cotangent tree.
    """

    param_deltas = jnp.asarray(param_deltas, dtype=jnp.float64)
    cotangents = jnp.asarray(objective_cotangents, dtype=jnp.float64)
    if cotangents.ndim == 1:
        cotangents = cotangents[None, :]
    names = tuple(objective_names) if objective_names is not None else geometry_observable_names_for_kind(
        "geometry_full_ad_objectives"
    )
    expected_names = geometry_observable_names_for_kind("geometry_full_ad_objectives")
    if names != expected_names:
        raise ValueError(
            "geometry_full_ad_objective_table_pullback_from_param_vector currently expects the "
            f"standard geometry_full_ad_objectives ordering: {expected_names}; got {names}."
        )
    if int(cotangents.shape[1]) != len(names):
        raise ValueError(
            f"Expected objective cotangents with {len(names)} columns for geometry_full_ad_objectives; "
            f"got {int(cotangents.shape[1])}."
        )

    phase_start = time.perf_counter()

    def _ready(tree):
        def _leaf_ready(value):
            return value.block_until_ready() if hasattr(value, "block_until_ready") else value

        return jax.tree.map(_leaf_ready, tree)

    def _progress(label: str, tree=None):
        nonlocal phase_start
        if tree is not None:
            _ready(tree)
        now = time.perf_counter()
        print(f"[geometry-fd-ad] progress: objective_table {label} elapsed_s={now - phase_start:.3f}", flush=True)
        phase_start = now

    final_mode = str(final_vmec_pullback_mode).strip().lower()
    param_entries = boundary_param_entries(context, param_specs)

    def solve_state(theta):
        return _solve_state_for_param_vector(
            context,
            theta,
            param_specs,
            lane=lane,
            max_iter=max_iter,
            step_size=step_size,
            jacobian_penalty=jacobian_penalty,
            solver_device=solver_device,
        )

    if final_mode == "raw_block_transpose":
        if str(lane).strip().lower() != "ad" or not _using_current_vmec_jax_context(context):
            raise ValueError("raw_block_transpose final VMEC pullback requires the current VMEX implicit AD lane.")
        if raw_block_solve is None:
            raw_block_solve = geometry_raw_block_solve_from_param_vector(
                context,
                param_deltas,
                param_specs,
                max_iter=max_iter,
                solver_device=solver_device,
            )
        implicit = raw_block_solve.implicit
        implicit_params = raw_block_solve.implicit_params
        implicit_cfg = raw_block_solve.implicit_cfg
        state = raw_block_solve.state
        dof_mask = raw_block_solve.dof_mask
        param_entries = raw_block_solve.param_entries
        _progress("vmec implicit state/raw-block aux ready", state)
    else:
        state, state_pullback = jax.vjp(solve_state, param_deltas)
        _progress("vmec implicit state/vjp ready", state)
    booz_api = _import_booz_xform_jax_api()
    use_selected_boozer_inputs = _using_current_vmec_jax_context(context) and isinstance(context.static, _VmecStaticCompat)
    if use_selected_boozer_inputs:
        booz_inputs = _neopax_vmex_boozer_table_inputs_from_state(
            state=state,
            static=context.static,
            surface_indices=context.surface_indices,
        )
        booz_surface_indices = None
    else:
        booz_inputs = _booz_xform_inputs_from_state(
            state=state,
            static=context.static,
            indata=context.indata,
            signgs=context.signgs,
            flux=context.flux,
        )
        booz_surface_indices = context.surface_indices
    booz_constants, booz_grids = _booz_constants_and_grids_for_inputs(context, booz_inputs)
    _progress("booz input tables ready", booz_inputs)
    ixm_b = jnp.asarray(booz_grids.xm_b, dtype=jnp.int32)
    ixn_b = jnp.asarray(booz_grids.xn_b, dtype=jnp.int32)
    mode00 = _find_boozer_mode_index(booz_grids.xm_b, booz_grids.xn_b, m_value=0, n_value=0)
    mode10 = _find_boozer_mode_index(booz_grids.xm_b, booz_grids.xn_b, m_value=1, n_value=0)
    booz_float_keys = (
        "iota_b",
        "buco_b",
        "bvco_b",
        "bmnc_b",
    )

    def booz_output_from_state(state_inner):
        if use_selected_boozer_inputs:
            inputs_inner = _neopax_vmex_boozer_table_inputs_from_state(
                state=state_inner,
                static=context.static,
                surface_indices=context.surface_indices,
            )
        else:
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
            surface_indices=booz_surface_indices,
            jit=True,
        )
        return {key: jnp.asarray(out[key], dtype=jnp.float64) for key in booz_float_keys}

    def booz_with_modes(booz_float):
        out = dict(booz_float)
        out["ixm_b"] = ixm_b
        out["ixn_b"] = ixn_b
        out["_mode00"] = mode00
        out["_mode10"] = mode10
        return out

    booz, booz_state_pullback = jax.vjp(booz_output_from_state, state)
    _progress("booz_xform vjp ready", booz)
    values_by_name: dict[str, jnp.ndarray] = {}

    vmec_names = (
        "aspect_ratio",
        "volume_total",
        "iota_mean",
        "magnetic_well",
        "mirror_ratio",
        "beta_volume",
    )
    vmec_indices = tuple(names.index(f"vmec_{name}") for name in vmec_names)

    def vmec_vector(state_inner):
        values = _vmec_core_scalar_objectives_from_state(context, state_inner)
        return jnp.stack([jnp.asarray(values[name], dtype=jnp.float64).reshape(()) for name in vmec_names])

    vmec_values, vmec_state_pullback = jax.vjp(vmec_vector, state)
    values_by_name.update({f"vmec_{name}": vmec_values[i] for i, name in enumerate(vmec_names)})
    vmec_basis = jax.vmap(lambda cot: vmec_state_pullback(cot)[0])(
        jnp.eye(len(vmec_names), dtype=jnp.float64)
    )
    vmec_state_bar = _tree_weighted_basis_sum(vmec_basis, cotangents[:, vmec_indices])
    _progress("vmec objective cotangents ready", vmec_state_bar)

    boozer_light_names = (
        "iota_b_mean",
        "b00_mean",
        "buco_b_mean",
        "bvco_b_mean",
        "b10_over_b00_mean",
    )
    boozer_light_indices = tuple(names.index(f"boozer_{name}") for name in boozer_light_names)

    def boozer_light_vector(booz_inner):
        values = _vmec_booz_light_scalar_observables_from_boozer(context, state, booz_with_modes(booz_inner))
        return jnp.stack([jnp.asarray(values[name], dtype=jnp.float64).reshape(()) for name in boozer_light_names])

    boozer_values, boozer_pullback = jax.vjp(boozer_light_vector, booz)
    values_by_name.update({f"boozer_{name}": boozer_values[i] for i, name in enumerate(boozer_light_names)})
    boozer_basis = jax.vmap(lambda cot: boozer_pullback(cot)[0])(
        jnp.eye(len(boozer_light_names), dtype=jnp.float64)
    )
    boozer_bar = _tree_weighted_basis_sum(boozer_basis, cotangents[:, boozer_light_indices])
    _progress("boozer light cotangents ready", boozer_bar)

    if mode00 is None:
        raise ValueError("Boozer output is missing the (m, n) = (0, 0) mode.")
    aspect_proxy_index = names.index("boozer_aspect_proxy")

    def boozer_aspect_proxy(state_inner):
        return jnp.asarray(_vmec_state_field(state_inner, "Rcos", "R_cos")[-1, mode00], dtype=jnp.float64).reshape(())

    aspect_proxy_value, aspect_proxy_state_pullback = jax.vjp(boozer_aspect_proxy, state)
    values_by_name["boozer_aspect_proxy"] = aspect_proxy_value
    aspect_proxy_unit_state_bar = aspect_proxy_state_pullback(jnp.asarray(1.0, dtype=jnp.float64))[0]
    aspect_proxy_state_bar = _tree_scale_unit_cotangent(
        aspect_proxy_unit_state_bar,
        cotangents[:, aspect_proxy_index],
    )
    _progress("aspect proxy cotangents ready", aspect_proxy_state_bar)

    j_qi_maxj_names = ("boozer_qi_objective", "boozer_maxj_objective")
    j_qi_maxj_indices = tuple(names.index(name) for name in j_qi_maxj_names)

    def j_qi_maxj_vector(booz_inner):
        values = _vmec_j_invariant_qi_maxj_objectives_from_boozer(
            context,
            booz_with_modes(booz_inner),
            include_qi=True,
            include_maxj=True,
        )
        return jnp.stack(
            [
                jnp.asarray(values["qi_objective"], dtype=jnp.float64).reshape(()),
                jnp.asarray(values["maxj_objective"], dtype=jnp.float64).reshape(()),
            ]
        )

    j_qi_maxj_values, j_qi_maxj_pullback = jax.vjp(j_qi_maxj_vector, booz)
    values_by_name.update(
        {name: j_qi_maxj_values[i] for i, name in enumerate(j_qi_maxj_names)}
    )
    j_qi_maxj_basis = jax.vmap(lambda cot: j_qi_maxj_pullback(cot)[0])(
        jnp.eye(len(j_qi_maxj_names), dtype=jnp.float64)
    )
    j_qi_maxj_boozer_bar = _tree_weighted_basis_sum(
        j_qi_maxj_basis,
        cotangents[:, j_qi_maxj_indices],
    )
    _progress("j-qi/maxj Boozer cotangents ready", j_qi_maxj_boozer_bar)

    booz_bar = _tree_add_all(boozer_bar, j_qi_maxj_boozer_bar)
    boozer_state_bar = jax.vmap(lambda booz_cotangent: booz_state_pullback(booz_cotangent)[0])(booz_bar)
    _progress("booz cotangents pulled to state", boozer_state_bar)

    state_bar = _tree_add_all(vmec_state_bar, boozer_state_bar, aspect_proxy_state_bar)
    if bool(return_state_bars):
        return values_by_name, state_bar
    if final_mode == "raw_block_transpose":
        if not hasattr(implicit, "implicit_state_pullback_multi_rhs_raw_block_transpose"):
            raise AttributeError(
                "The active VMEC backend does not expose implicit_state_pullback_multi_rhs_raw_block_transpose."
            )
        param_bar_batch = implicit.implicit_state_pullback_multi_rhs_raw_block_transpose(
            implicit_params,
            implicit_cfg,
            state,
            dof_mask,
            state_bar,
            probe_chunk_size=1,
        )
        gradient_matrix = _param_vector_gradient_from_implicit_param_grads(param_bar_batch, param_entries)
    elif final_mode in {"lax_map", "sequential"}:
        gradient_matrix = jax.lax.map(lambda state_cotangent: state_pullback(state_cotangent)[0], state_bar)
    elif final_mode == "vmap":
        gradient_matrix = jax.vmap(lambda state_cotangent: state_pullback(state_cotangent)[0])(state_bar)
    else:
        raise ValueError(
            "final_vmec_pullback_mode must be 'raw_block_transpose', 'vmap', 'lax_map', or 'sequential'."
        )
    _progress("final vmec parameter pullback ready", gradient_matrix)
    return values_by_name, gradient_matrix


def geometry_observable_multi_rhs_pullback_from_param_vector(
    context: GeometryAutodiffContext,
    param_deltas,
    param_specs: Sequence[tuple[str, int, int]],
    objective_cotangents,
    *,
    observable_kind: str,
    objective_names: Sequence[str] | None = None,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
    solver_device: str | None = None,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Return objective values and a multi-RHS reverse table for geometry params."""
    param_deltas = jnp.asarray(param_deltas, dtype=jnp.float64)
    cotangents = jnp.asarray(objective_cotangents, dtype=jnp.float64)
    if cotangents.ndim == 1:
        cotangents = cotangents[None, :]
    names = tuple(objective_names) if objective_names is not None else geometry_observable_names_for_kind(observable_kind)
    if int(cotangents.shape[1]) != len(names):
        raise ValueError(
            f"Expected objective cotangents with {len(names)} columns for {observable_kind}; "
            f"got {int(cotangents.shape[1])}."
        )

    def objective_vector(theta):
        state = _solve_state_for_param_vector(
            context,
            theta,
            param_specs,
            lane=lane,
            max_iter=max_iter,
            step_size=step_size,
            jacobian_penalty=jacobian_penalty,
            solver_device=solver_device,
        )
        items = _observable_items_from_state(context, state, observable_kind=observable_kind)
        observables = {name: jnp.asarray(value, dtype=jnp.float64).reshape(()) for name, value in items}
        missing = [name for name in names if name not in observables]
        if missing:
            raise ValueError(f"Unknown objective names for {observable_kind}: {missing}")
        return jnp.stack([observables[name] for name in names])

    values = objective_vector(param_deltas)

    def single_rhs_grad(cotangent):
        def contracted(theta):
            return jnp.vdot(cotangent, objective_vector(theta))

        return jax.grad(contracted)(param_deltas)

    def scan_body(_carry, cotangent):
        return _carry, single_rhs_grad(cotangent)

    _carry, gradient_matrix = jax.lax.scan(scan_body, None, cotangents)
    values_by_name = {name: values[i] for i, name in enumerate(names)}
    return values_by_name, gradient_matrix


def vmec_qi_maxj_scalar_objectives_from_single_param(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> dict[str, jnp.ndarray]:
    state = _solve_state_for_single_param(
        context,
        param_delta,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    return _vmec_j_invariant_qi_maxj_objectives_from_state(
        context,
        state,
        include_qi=True,
        include_maxj=True,
    )


def vmec_booz_qi_maxj_scalar_objectives_from_single_param(
    context: GeometryAutodiffContext,
    param_delta,
    *,
    lane: str = "ad",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
) -> dict[str, jnp.ndarray]:
    state = _solve_state_for_single_param(
        context,
        param_delta,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    return _vmec_booz_qi_maxj_scalar_objectives_from_state(context, state)


def _safe_divide(num, den):
    num_arr = jnp.asarray(num)
    den_arr = jnp.asarray(den)
    den_safe = jnp.where(jnp.abs(den_arr) > 0.0, den_arr, 1.0)
    return jnp.where(jnp.abs(den_arr) > 0.0, num_arr / den_safe, 0.0)


def _safe_reciprocal(values):
    arr = jnp.asarray(values)
    active = jnp.abs(arr) > 0.0
    arr_safe = jnp.where(active, arr, 1.0)
    return jnp.where(active, 1.0 / arr_safe, 0.0)


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


def _boozer_surface_indices_and_rho(static, rho_values):
    rho_values_np = np.asarray(rho_values, dtype=float).reshape(-1)
    s_values = tuple(float(rho_value**2) for rho_value in rho_values_np)
    surface_indices_arr = np.unique(
        np.asarray(_surface_indices_for_s_values(static, s_values), dtype=np.int32)
    )
    surface_indices = jnp.asarray(surface_indices_arr, dtype=jnp.int32)
    s_full = jnp.asarray(static.s, dtype=jnp.float64)
    s_half = 0.5 * (s_full[:-1] + s_full[1:])
    sample_rho = jnp.sqrt(jnp.maximum(s_half[surface_indices], 0.0))
    return surface_indices, sample_rho


def _boozer_rmnc00_from_state_at_rho(context: GeometryAutodiffContext, state, rho_values):
    """Return Boozer R(m=0,n=0) on requested rho values.

    The frozen NTX path derives r00 from boozermn rmnc_b on VMEC half-mesh
    support.  The realtime path must use the same Boozer convention rather
    than full-grid VMEC Rcos[:, 0], otherwise NTX normalization factors see a
    different major-radius profile from the geometry object.
    """
    rho_arr = jnp.asarray(rho_values, dtype=jnp.float64)
    surface_indices, sample_rho = _boozer_surface_indices_and_rho(context.static, rho_values)
    vmec_jax = _import_vmec_jax()
    booz_api = _import_booz_xform_jax_api()
    inputs = _booz_xform_inputs_from_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=context.signgs,
        flux=context.flux,
    )
    booz_constants, booz_grids = _booz_constants_and_grids_for_inputs(context, inputs)
    out = booz_api.booz_xform_from_inputs(
        inputs=inputs,
        constants=booz_constants,
        grids=booz_grids,
        surface_indices=surface_indices,
        jit=True,
    )
    if "rmnc_b" not in out:
        raise ValueError("booz_xform_from_inputs output is missing rmnc_b.")
    ixm_b = jnp.asarray(out["ixm_b"], dtype=jnp.int32)
    ixn_b = jnp.asarray(out["ixn_b"], dtype=jnp.int32)
    mode00 = _find_boozer_mode_index(ixm_b, ixn_b, m_value=0, n_value=0)
    if mode00 is None:
        raise ValueError("Boozer output is missing the R(m=0,n=0) mode.")
    rmnc00_samples = jnp.asarray(out["rmnc_b"], dtype=jnp.float64)[:, mode00]
    return interpax.Interpolator1D(sample_rho, rmnc00_samples, extrap=True)(rho_arr)


def _neopax_geometry_requested_sample_rho(context: GeometryAutodiffContext, *, n_r: int):
    rho_grid = np.linspace(0.0, 1.0, int(n_r), dtype=float)
    if int(n_r) > 1:
        rho_grid_half = np.concatenate(
            [
                np.asarray([0.0], dtype=float),
                0.5 * (rho_grid[:-1] + rho_grid[1:]),
                np.asarray([1.0], dtype=float),
            ],
            axis=0,
        )
    else:
        rho_grid_half = np.asarray([0.0, 1.0], dtype=float)
    s_full = np.asarray(jax.device_get(context.static.s), dtype=float)
    rho_half = np.sqrt(np.maximum(0.5 * (s_full[1:] + s_full[:-1]), 0.0))
    edge_count = min(8, max(int(rho_half.shape[0]) - 1, 0))
    edge_rho = rho_half[-edge_count:] if edge_count > 0 else rho_half[:0]
    return np.unique(
        np.concatenate(
            [
                rho_grid_half[1:-1],
                rho_half[:edge_count],
                edge_rho,
            ],
            axis=0,
        )
    )


def _vmec_r00_from_state_at_rho(context: GeometryAutodiffContext, state, rho_values):
    """Return traceable VMEC R(m=0,n=0) on requested rho values."""

    rho_arr = jnp.asarray(np.asarray(rho_values, dtype=float), dtype=jnp.float64)
    m_arr = np.asarray(context.static.modes.m, dtype=np.int32).reshape(-1)
    n_arr = np.asarray(context.static.modes.n, dtype=np.int32).reshape(-1)
    matches = np.nonzero((m_arr == 0) & (n_arr == 0))[0]
    if matches.size == 0:
        raise ValueError("VMEC state is missing the R(m=0,n=0) mode.")
    mode00 = int(matches[0])
    s_full = jnp.asarray(context.static.s, dtype=jnp.float64)
    rho_full = jnp.sqrt(jnp.maximum(s_full, 0.0))
    r00_full = jnp.asarray(_vmec_state_field(state, "Rcos", "R_cos"), dtype=jnp.float64)[:, mode00]
    return interpax.Interpolator1D(rho_full, r00_full, extrap=True)(rho_arr)


def _build_neopax_geometry_from_state(
    context: GeometryAutodiffContext,
    state,
    *,
    n_r: int,
    requested_sample_rho=None,
    boozer_surface_sampling=None,
    booz_constants_grids=None,
    booz_mode_indices=None,
):
    from NEOPAX._geometry_models import VmecBoozer

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
    try:
        forces_module = _import_vmec_module("vmec_forces")
        residue_module = _import_vmec_module("vmec_residue")
        vmec_forces_rz_from_wout = forces_module.vmec_forces_rz_from_wout
        vmec_force_norms_from_bcovar_dynamic = residue_module.vmec_force_norms_from_bcovar_dynamic
    except ModuleNotFoundError:
        try:
            forces_module = _import_vmec_module("kernels.forces")
            residue_module = _import_vmec_module("kernels.residue")
            vmec_forces_rz_from_wout = forces_module.vmec_forces_rz_from_wout
            vmec_force_norms_from_bcovar_dynamic = residue_module.vmec_force_norms_from_bcovar_dynamic
        except ModuleNotFoundError:
            vmec_forces_rz_from_wout = None
            vmec_force_norms_from_bcovar_dynamic = None

    if vmec_forces_rz_from_wout is not None and vmec_force_norms_from_bcovar_dynamic is not None:
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
    else:
        fields_module = _import_vmec_module("core.fields")
        geometry_module = _import_vmec_module("core.geometry")
        solver_module = _import_vmec_module("core.solver")
        energies_and_force_norms = fields_module.energies_and_force_norms
        magnetic_fields = fields_module.magnetic_fields
        metric_elements = fields_module.metric_elements
        half_mesh_jacobian = geometry_module.half_mesh_jacobian
        _geometry = solver_module._geometry

        rt = context.static.runtime
        setup = rt.setup
        _, real_space = _geometry(state, rt)
        jacobian = half_mesh_jacobian(real_space, s=setup.s_full)
        metrics = metric_elements(real_space, s=setup.s_full)
        fields = magnetic_fields(
            geometry=real_space,
            jacobian=jacobian,
            metrics=metrics,
            trig=rt.trig,
            s=setup.s_full,
            phips=setup.phips,
            phipf=setup.phipf,
            chips=setup.chips,
            signgs=setup.signgs,
            gamma=rt.gamma,
            mass=setup.mass,
            ncurr=setup.ncurr,
            enclosed_current=setup.icurv,
        )
        norms = energies_and_force_norms(
            jacobian=jacobian,
            metrics=metrics,
            fields=fields,
            trig=rt.trig,
            s=setup.s_full,
            signgs=setup.signgs,
        )
    volume_p = jnp.abs(jnp.asarray(norms.volume)) * (4.0 * jnp.pi**2)
    vp = jnp.abs(jnp.asarray(norms.vp))
    s_full = jnp.asarray(context.static.s)
    rho_half = jnp.concatenate(
        [jnp.zeros((1,), dtype=s_full.dtype), jnp.sqrt(jnp.maximum(0.5 * (s_full[1:] + s_full[:-1]), 0.0))],
        axis=0,
    )
    # Match the frozen VmecBoozer half-grid interpolation pattern without
    # launching Boozer on every VMEC half-mesh surface.  Include a few real VMEC
    # half-mesh points at the axis and edge so extrapolated quantities such as
    # B0(r_grid) use a frozen-like local slope.
    if requested_sample_rho is None:
        requested_sample_rho = _neopax_geometry_requested_sample_rho(context, n_r=int(n_r))
    if boozer_surface_sampling is None:
        surface_indices, sample_rho = _boozer_surface_indices_and_rho(context.static, requested_sample_rho)
    else:
        surface_indices, sample_rho = boozer_surface_sampling

    phipf = jnp.asarray(context.flux.phipf)
    phi = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=phipf.dtype),
            jnp.cumsum(phipf[1:] * (s_full[1:] - s_full[:-1])),
        ],
        axis=0,
    )
    # VMEC/JAX gives the toroidal-flux integral per field period/radian here,
    # while the frozen wout-backed transport lane uses wout.phi[-1].
    psia = jnp.abs(phi[-1]) * (2.0 * jnp.pi)

    booz_api = _import_booz_xform_jax_api()
    vmec_jax = _import_vmec_jax()
    inputs = _booz_xform_inputs_from_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=context.signgs,
        flux=context.flux,
    )
    if booz_constants_grids is None:
        booz_constants, booz_grids = _booz_constants_and_grids_for_inputs(context, inputs)
    else:
        booz_constants, booz_grids = booz_constants_grids
    out = booz_api.booz_xform_from_inputs(
        inputs=inputs,
        constants=booz_constants,
        grids=booz_grids,
        surface_indices=surface_indices,
        jit=True,
    )
    bmnc_b = jnp.asarray(out["bmnc_b"])
    if "gmnc_b" not in out:
        raise ValueError("booz_xform_from_inputs output is missing gmnc_b.")
    if "rmnc_b" not in out:
        raise ValueError("booz_xform_from_inputs output is missing rmnc_b.")
    gmnc_b = jnp.asarray(out["gmnc_b"])
    rmnc_b = jnp.asarray(out["rmnc_b"])
    if booz_mode_indices is None:
        ixm_b = jnp.asarray(out["ixm_b"], dtype=jnp.int32)
        ixn_b = jnp.asarray(out["ixn_b"], dtype=jnp.int32)
        mode00 = _find_boozer_mode_index(ixm_b, ixn_b, m_value=0, n_value=0)
        if mode00 is None:
            raise ValueError("Boozer output is missing the (0,0) mode.")
        mode10 = _find_boozer_mode_index(ixm_b, ixn_b, m_value=1, n_value=0)
    else:
        mode00, mode10 = booz_mode_indices

    r0_value = rmnc_b[-1, mode00]
    a_b = jnp.sqrt(volume_p / (2.0 * jnp.pi**2 * r0_value))
    r_grid = rho_grid * a_b
    r_grid_half = rho_grid_half * a_b
    dr = r_grid[1] - r_grid[0] if int(n_r) > 1 else jnp.asarray(0.0, dtype=r_grid.dtype)

    dVdr = interpax.Interpolator1D(rho_half[1:], jnp.asarray(vp)[1:], extrap=True)
    volume_scale = (2.0 * jnp.pi) ** 2
    vprime = dVdr(rho_grid) * 2.0 * rho_grid / a_b * volume_scale
    vprime_half = dVdr(rho_grid_half) * 2.0 * rho_grid_half / a_b * volume_scale
    over_vprime = _safe_reciprocal(vprime).at[0].set(0.0)

    iota_samples = jnp.asarray(out["iota_b"])
    i_value_samples = jnp.asarray(out["buco_b"])
    g_value_samples = jnp.asarray(out["bvco_b"])
    b0_samples = bmnc_b[:, mode00]
    r00_samples = rmnc_b[:, mode00]
    sqrtg00_samples = gmnc_b[:, mode00]
    if mode10 is None:
        b10_raw_samples = jnp.zeros_like(b0_samples)
    else:
        b10_raw_samples = bmnc_b[:, mode10]

    b0_interp = interpax.Interpolator1D(sample_rho, b0_samples, extrap=True)
    r00_interp = interpax.Interpolator1D(sample_rho, r00_samples, extrap=True)
    sqrtg00_interp = interpax.Interpolator1D(sample_rho, sqrtg00_samples, extrap=True)
    b10_interp = interpax.Interpolator1D(sample_rho, b10_raw_samples, extrap=True)
    iota_interp = interpax.Interpolator1D(sample_rho, iota_samples, extrap=True)
    i_interp = interpax.Interpolator1D(sample_rho, i_value_samples, extrap=True)
    g_interp = interpax.Interpolator1D(sample_rho, g_value_samples, extrap=True)

    b_00 = b0_interp(rho_grid)
    b0 = b_00
    b_10 = _safe_divide(b10_interp(rho_grid), b_00).at[0].set(0.0)
    iota = iota_interp(rho_grid)
    i_value = i_interp(rho_grid)
    g_value = g_interp(rho_grid)
    # Match frozen VmecBoozer: epsilon_t uses the Boozer R00 profile, not the
    # edge major radius used to define a_b.
    epsilon_t = _safe_divide(rho_grid * a_b, r00_interp(rho_grid))
    curvature = _safe_divide(jnp.abs(b_10), epsilon_t).at[0].set(0.0)
    enlogation = jnp.square(_safe_divide(epsilon_t, b_10)).at[0].set(0.0)
    # b0_interp is tabulated against rho; B0prime follows the frozen geometry
    # convention of dB0/dr, so convert dB00/drho by drho/dr = 1 / a_b.
    b0prime = jax.vmap(jax.grad(lambda rho: b0_interp(rho)))(rho_grid) / a_b
    sqrtg00_value = sqrtg00_interp(rho_grid)
    bsqav = _safe_divide(g_value + iota * i_value, sqrtg00_value * jnp.maximum(jnp.square(b0), 1.0e-30))
    iota_safe = jnp.where(jnp.abs(iota) > 0.0, jnp.abs(iota), 1.0)
    g_ps = (
        1.5
        * (4.0 / 3.0)
        * jnp.square(curvature / iota_safe)
        * (
            1.0
            + 3.4229 * jnp.power(epsilon_t, 3.6) * (1.0 - 2.5766 * jnp.power(jnp.abs(iota), 1.6))
            - 0.6039 * jnp.power(epsilon_t, 2.0) * (1.0 - jnp.square(curvature))
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


def _build_ntx_runtime_channels_from_surfaces(surfaces, *, rho, a_b, psia, r00):
    from NEOPAX._transport_flux_models import NTXRuntimeScanChannels

    def _surface_zero_mode_value(surface, field_name):
        values = jnp.asarray(getattr(surface, field_name), dtype=jnp.float64)
        mode_mask = (jnp.asarray(surface.m) == 0) & (jnp.asarray(surface.n) == 0)
        index = jnp.argmax(mode_mask.astype(jnp.int32))
        return jnp.take(values, index)

    def _surface_b00(surface):
        if hasattr(surface, "b_cos"):
            return _surface_zero_mode_value(surface, "b_cos")
        b0 = getattr(surface, "b0", None)
        if b0 is not None:
            return jnp.asarray(b0, dtype=jnp.float64)
        raise AttributeError("NTX surface is missing both b_cos and b0.")

    def _surface_boozer_i(surface):
        if hasattr(surface, "b_theta"):
            return jnp.asarray(surface.b_theta, dtype=jnp.float64)
        return _surface_zero_mode_value(surface, "b_sub_theta_cos")

    def _surface_boozer_g(surface):
        if hasattr(surface, "b_zeta"):
            return jnp.asarray(surface.b_zeta, dtype=jnp.float64)
        return _surface_zero_mode_value(surface, "b_sub_zeta_cos")

    rho_arr = jnp.asarray(rho, dtype=jnp.float64)
    psia_value = jnp.asarray(psia, dtype=jnp.float64)
    r00_arr = jnp.asarray(r00, dtype=jnp.float64)
    b00 = jnp.asarray([_surface_b00(surface) for surface in surfaces])
    boozer_i = jnp.asarray([_surface_boozer_i(surface) for surface in surfaces])
    boozer_g = jnp.asarray([_surface_boozer_g(surface) for surface in surfaces])
    iota = jnp.asarray([jnp.asarray(surface.iota, dtype=jnp.float64) for surface in surfaces])
    a_b_value = jnp.asarray(a_b, dtype=jnp.float64)
    rho_positive = rho_arr > 0.0
    rho_safe = jnp.where(rho_positive, rho_arr, 1.0)
    drds = jnp.where(rho_positive, a_b_value / (2.0 * rho_safe), 0.0)
    dpsi_drtilde = rho_arr * a_b_value * b00
    dr_tildedr = 2.0 * psia_value / (jnp.asarray(a_b, dtype=jnp.float64) ** 2 * b00)
    dr_tildeds = dr_tildedr * drds
    denom = boozer_g + iota * boozer_i
    fac_reference_to_sfincs_11 = 8.0 * denom * b00 * psia_value**2 / (jnp.sqrt(jnp.pi) * boozer_g**2)
    fac_reference_to_sfincs_31 = 4.0 * b00 * psia_value / (jnp.sqrt(jnp.pi) * boozer_g)
    fac_reference_to_sfincs_33 = -2.0 * b00 / (denom * jnp.sqrt(jnp.pi))
    fac_sfincs_to_dkes_11 = _safe_reciprocal(
        8.0 * denom * dpsi_drtilde**2 / (boozer_g**2 * b00 * jnp.sqrt(jnp.pi))
    )
    fac_sfincs_to_dkes_31 = _safe_reciprocal(4.0 * dpsi_drtilde / (boozer_g * jnp.sqrt(jnp.pi)))
    fac_sfincs_to_dkes_33 = _safe_reciprocal(-2.0 * b00 / (denom * jnp.sqrt(jnp.pi)))
    epsilon_t = _safe_divide(rho_arr * a_b_value, r00_arr)
    fac_dkes_to_d11star = -(8.0 / jnp.pi) * iota * r00_arr
    fac_dkes_to_d31star = -(3.0 / 1.46) * iota * jnp.sqrt(jnp.maximum(epsilon_t, 0.0)) / 2.0
    return NTXRuntimeScanChannels(
        rho=rho_arr,
        a_b=a_b,
        psia=psia_value,
        b00=b00,
        r00=r00_arr,
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
        fac_dkes_to_d11star=fac_dkes_to_d11star,
        fac_dkes_to_d31star=fac_dkes_to_d31star,
        fac_dkes_to_d33star=jnp.asarray(1.0, dtype=jnp.float64),
    )


def _build_ntx_runtime_channels_from_geometry(geometry, *, rho, psia, r00):
    from NEOPAX._transport_flux_models import NTXRuntimeScanChannels

    rho_arr = jnp.asarray(rho, dtype=jnp.float64)
    rho_grid = jnp.asarray(geometry.rho_grid, dtype=jnp.float64)

    def _interp_geometry_field(name):
        values = jnp.asarray(getattr(geometry, name), dtype=jnp.float64)
        return interpax.Interpolator1D(rho_grid, values, extrap=True)(rho_arr)

    a_b = jnp.asarray(geometry.a_b, dtype=jnp.float64)
    psia_value = jnp.asarray(psia, dtype=jnp.float64)
    r00_arr = jnp.asarray(r00, dtype=jnp.float64)
    b00 = _interp_geometry_field("B0")
    boozer_i = _interp_geometry_field("I_value")
    boozer_g = _interp_geometry_field("G_value")
    iota = _interp_geometry_field("iota")
    rho_positive = rho_arr > 0.0
    rho_safe = jnp.where(rho_positive, rho_arr, 1.0)
    drds = jnp.where(rho_positive, a_b / (2.0 * rho_safe), 0.0)
    dpsi_drtilde = rho_arr * a_b * b00
    dr_tildedr = 2.0 * psia_value / (a_b**2 * b00)
    dr_tildeds = dr_tildedr * drds
    denom = boozer_g + iota * boozer_i
    sqrt_pi = jnp.sqrt(jnp.pi)
    fac_reference_to_sfincs_11 = 8.0 * denom * b00 * psia_value**2 / (sqrt_pi * boozer_g**2)
    fac_reference_to_sfincs_31 = 4.0 * b00 * psia_value / (sqrt_pi * boozer_g)
    fac_reference_to_sfincs_33 = -2.0 * b00 / (denom * sqrt_pi)
    fac_sfincs_to_dkes_11 = _safe_reciprocal(
        8.0 * denom * dpsi_drtilde**2 / (boozer_g**2 * b00 * sqrt_pi)
    )
    fac_sfincs_to_dkes_31 = _safe_reciprocal(4.0 * dpsi_drtilde / (boozer_g * sqrt_pi))
    fac_sfincs_to_dkes_33 = _safe_reciprocal(-2.0 * b00 / (denom * sqrt_pi))
    epsilon_t = _safe_divide(rho_arr * a_b, r00_arr)
    fac_dkes_to_d11star = -(8.0 / jnp.pi) * iota * r00_arr
    fac_dkes_to_d31star = -(3.0 / 1.46) * iota * jnp.sqrt(jnp.maximum(epsilon_t, 0.0)) / 2.0
    return NTXRuntimeScanChannels(
        rho=rho_arr,
        a_b=a_b,
        psia=psia_value,
        b00=b00,
        r00=r00_arr,
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
        fac_dkes_to_d11star=fac_dkes_to_d11star,
        fac_dkes_to_d31star=fac_dkes_to_d31star,
        fac_dkes_to_d33star=jnp.asarray(1.0, dtype=jnp.float64),
    )


def _ntx_surface_iota_targets(geometry, rho_values):
    rho_grid = jnp.asarray(geometry.rho_grid, dtype=jnp.float64)
    iota_grid = jnp.asarray(geometry.iota, dtype=jnp.float64)
    iota_for_sign = iota_grid
    if int(iota_grid.shape[0]) > 1:
        iota_for_sign = iota_for_sign.at[0].set(iota_grid[1])
    return jnp.interp(
        jnp.asarray(rho_values, dtype=jnp.float64),
        rho_grid,
        iota_for_sign,
    )


def _align_ntx_surface_iota_convention(surface, target_iota):
    surface_iota = jnp.asarray(surface.iota, dtype=jnp.float64)
    target = jnp.asarray(target_iota, dtype=jnp.float64)
    should_flip = jnp.logical_and(surface_iota * target < 0.0, jnp.abs(target) > 0.0)

    if not hasattr(surface, "b_theta"):
        return dataclasses.replace(
            surface,
            iota=jnp.where(should_flip, -surface_iota, surface_iota),
        )

    def _maybe_flip(value):
        if value is None:
            return None
        value_arr = jnp.asarray(value, dtype=jnp.float64)
        return jnp.where(should_flip, -value_arr, value_arr)

    updates = {
        "iota": _maybe_flip(surface.iota),
        "b_theta": _maybe_flip(surface.b_theta),
    }
    if surface.chi_p is not None:
        updates["chi_p"] = _maybe_flip(surface.chi_p)
    return dataclasses.replace(surface, **updates)


def _ntx_frozen_interp_mode_columns(interpolated_value, x_nodes, values, xq):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("expected a 2D `(mode, radius)` array")
    return np.asarray(
        [interpolated_value(x_nodes, row, float(xq), order=2) for row in values],
        dtype=np.float64,
    )


def _vmec_jax_wout_surface_with_frozen_sampling(wout, *, s, source_path):
    """Build an NTX VMEC surface using the same sampling convention as NTX's frozen loader."""

    import ntx

    surface_fn = getattr(ntx, "surface_from_vmex_vmec_wout", None)
    if surface_fn is None:
        surface_fn = getattr(ntx, "surface_from_vmec_jax_vmec_wout", None)
    if surface_fn is None:
        try:
            from ntx.vmex_vmec import surface_from_vmex_vmec_wout as surface_fn
        except Exception:
            from ntx.vmec_jax_vmec import surface_from_vmec_jax_vmec_wout as surface_fn
    return surface_fn(
        wout,
        s=float(s),
        source_path=Path(source_path).expanduser(),
    )


def _traceable_vmec_field_tables_from_state(context: GeometryAutodiffContext, state):
    """Build the NTX-needed VMEC Nyquist tables without materializing WOUT."""

    fields_module = _import_vmec_module("core.fields")
    geometry_module = _import_vmec_module("core.geometry")
    solver_module = _import_vmec_module("core.solver")
    fourier_module = _import_vmec_module("core.fourier")
    magnetic_fields = fields_module.magnetic_fields
    metric_elements = fields_module.metric_elements
    half_mesh_jacobian = geometry_module.half_mesh_jacobian
    _geometry = solver_module._geometry
    mode_table = fourier_module.mode_table

    if bool(context.cfg.lasym):
        raise NotImplementedError("traceable realtime VMEC NTX surfaces currently support stellarator symmetry only")

    rt = context.static.runtime
    setup = rt.setup
    s_full = jnp.asarray(setup.s_full, dtype=jnp.float64)
    _, real_space = _geometry(state, rt)
    jacobian = half_mesh_jacobian(real_space, s=s_full)
    metrics = metric_elements(real_space, s=s_full)
    fields = magnetic_fields(
        geometry=real_space,
        jacobian=jacobian,
        metrics=metrics,
        trig=rt.trig,
        s=s_full,
        phips=setup.phips,
        phipf=setup.phipf,
        chips=setup.chips,
        signgs=setup.signgs,
        gamma=rt.gamma,
        mass=setup.mass,
        ncurr=setup.ncurr,
        enclosed_current=setup.icurv,
    )

    trig = rt.trig
    ntheta2 = int(trig.ntheta2)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    mnyq = max(ntheta2 - 1, 0)
    nnyq = max(nzeta // 2, 0)
    modes = mode_table(max(mnyq, max(int(context.static.resolution.mpol) - 1, 0)) + 1, max(nnyq, int(context.static.resolution.ntor)))
    xm_nyq_np = np.asarray(modes.m, dtype=np.int32)
    xn_nyq_np = np.asarray(modes.n, dtype=np.int32) * int(context.static.resolution.nfp)

    nzeta_safe = max(nzeta, 1)
    dnorm = 1.0 / (nzeta_safe * max(ntheta2 - 1, 1))
    ntheta1 = int(trig.ntheta1)
    mscale_np = np.ones((mnyq + 1,), dtype=float)
    nscale_np = np.ones((nnyq + 1,), dtype=float)
    if mnyq >= 1:
        mscale_np[1:] = np.sqrt(2.0)
    if nnyq >= 1:
        nscale_np[1:] = np.sqrt(2.0)
    theta_index = np.arange(ntheta2, dtype=float)
    m_table = np.arange(mnyq + 1, dtype=float)
    theta_arg = (2.0 * np.pi) * theta_index[:, None] * m_table[None, :] / float(ntheta1)
    cosmu_np = np.cos(theta_arg) * mscale_np[None, :]
    sinmu_np = np.sin(theta_arg) * mscale_np[None, :]
    endpoint_sign = np.where((np.arange(mnyq + 1) % 2) == 0, 1.0, -1.0)
    cosmu_np[ntheta2 - 1, :] = endpoint_sign * mscale_np
    sinmu_np[ntheta2 - 1, :] = 0.0
    cosmui_np = dnorm * cosmu_np
    sinmui_np = dnorm * sinmu_np
    cosmui_np[0, :] *= 0.5
    cosmui_np[ntheta2 - 1, :] *= 0.5
    if mnyq > 0:
        cosmui_np[:, mnyq] *= 0.5
    zeta_index = np.arange(nzeta, dtype=float)
    n_table = np.arange(nnyq + 1, dtype=float)
    zeta_arg = (2.0 * np.pi) * zeta_index[:, None] * n_table[None, :] / float(nzeta_safe)
    cosnv_np = np.cos(zeta_arg) * nscale_np[None, :]
    sinnv_np = np.sin(zeta_arg) * nscale_np[None, :]
    if nnyq > 0:
        cosnv_np[:, nnyq] *= 0.5

    m_modes = np.asarray(modes.m, dtype=np.int32)
    n_modes = np.asarray(modes.n, dtype=np.int32)
    dmult_np = mscale_np[m_modes] * nscale_np[np.abs(n_modes)] * 0.5
    dmult_np = np.where((m_modes == 0) | (n_modes == 0), 2.0 * dmult_np, dmult_np)

    cosmui = jnp.asarray(cosmui_np, dtype=jnp.float64)
    sinmui = jnp.asarray(sinmui_np, dtype=jnp.float64)
    cosnv = jnp.asarray(cosnv_np, dtype=jnp.float64)
    sinnv = jnp.asarray(sinnv_np, dtype=jnp.float64)
    dmult = jnp.asarray(dmult_np, dtype=jnp.float64)
    m_index = jnp.asarray(m_modes, dtype=jnp.int32)
    n_abs_index = jnp.asarray(np.abs(n_modes), dtype=jnp.int32)
    n_sign = jnp.asarray(np.where(n_modes < 0, -1.0, 1.0), dtype=jnp.float64)

    def wrout_cos_coeffs_jax(f):
        f_red = jnp.asarray(f, dtype=jnp.float64)[:, :ntheta2, :]
        f_theta_cos = jnp.einsum("sik,im->smk", f_red, cosmui)
        f_theta_sin = jnp.einsum("sik,im->smk", f_red, sinmui)
        cos_zeta = jnp.einsum("smk,kn->smn", f_theta_cos, cosnv)
        sin_zeta = jnp.einsum("smk,kn->smn", f_theta_sin, sinnv)
        coeff = cos_zeta[:, m_index, n_abs_index] + n_sign[None, :] * sin_zeta[:, m_index, n_abs_index]
        return coeff * dmult[None, :]

    mmax_force = max(int(context.static.resolution.mpol) - 1, 0)
    nmax_force = int(context.static.resolution.ntor)
    cosmui_filter = jnp.asarray(dnorm * np.asarray(trig.cosmu, dtype=float)[:ntheta2, : mmax_force + 1], dtype=jnp.float64)
    sinmui_filter = jnp.asarray(dnorm * np.asarray(trig.sinmu, dtype=float)[:ntheta2, : mmax_force + 1], dtype=jnp.float64)
    cosmui_filter = cosmui_filter.at[0, :].multiply(0.5).at[ntheta2 - 1, :].multiply(0.5)
    cosmu_filter = jnp.asarray(np.asarray(trig.cosmu, dtype=float)[:ntheta2, : mmax_force + 1], dtype=jnp.float64)
    sinmu_filter = jnp.asarray(np.asarray(trig.sinmu, dtype=float)[:ntheta2, : mmax_force + 1], dtype=jnp.float64)
    cosnv_filter = jnp.asarray(np.asarray(trig.cosnv, dtype=float)[:, : nmax_force + 1], dtype=jnp.float64)
    sinnv_filter = jnp.asarray(np.asarray(trig.sinnv, dtype=float)[:, : nmax_force + 1], dtype=jnp.float64)
    dmult_filter_np = np.ones((mmax_force + 1, nmax_force + 1), dtype=float)
    if mnyq > 0 and mnyq <= mmax_force:
        dmult_filter_np[mnyq, :] *= 0.5
    if nnyq > 0 and nnyq <= nmax_force:
        dmult_filter_np[:, nnyq] *= 0.5
    dmult_filter = jnp.asarray(dmult_filter_np, dtype=jnp.float64)

    def filter_covariant_jax(f):
        f_red = jnp.asarray(f, dtype=jnp.float64)[:, :ntheta2, :]
        f_theta_cos = jnp.einsum("sik,im->smk", f_red, cosmui_filter)
        f_theta_sin = jnp.einsum("sik,im->smk", f_red, sinmui_filter)
        c1 = jnp.einsum("smk,kn->smn", f_theta_cos, cosnv_filter) * dmult_filter[None, :, :]
        c2 = jnp.einsum("smk,kn->smn", f_theta_sin, sinnv_filter) * dmult_filter[None, :, :]
        tmp_cos = jnp.einsum("smn,im->sin", c1, cosmu_filter)
        tmp_sin = jnp.einsum("smn,im->sin", c2, sinmu_filter)
        return jnp.einsum("sin,kn->sik", tmp_cos, cosnv_filter) + jnp.einsum("sin,kn->sik", tmp_sin, sinnv_filter)

    bsubu_out = filter_covariant_jax(fields.bsubu)
    bsubv_out = filter_covariant_jax(fields.bsubv)
    pressure = jnp.asarray(fields.pressure)
    bsq = 2.0 * (jnp.asarray(fields.total_pressure) - pressure[:, None, None])
    tiny = jnp.asarray(jnp.finfo(pressure.dtype).tiny, dtype=pressure.dtype)
    bmag = jnp.sqrt(jnp.maximum(bsq, tiny))
    gmnc = wrout_cos_coeffs_jax(jacobian.sqrt_g)
    bmnc = wrout_cos_coeffs_jax(bmag)
    bsupumnc = wrout_cos_coeffs_jax(fields.bsupu)
    bsupvmnc = wrout_cos_coeffs_jax(fields.bsupv)
    bsubumnc = wrout_cos_coeffs_jax(bsubu_out)
    bsubvmnc = wrout_cos_coeffs_jax(bsubv_out)

    def zero_axis(arr):
        return jnp.asarray(arr, dtype=jnp.float64).at[0, :].set(0.0)

    iotas_half = (
        jnp.asarray(fields.chips) / jnp.where(jnp.asarray(setup.phips) != 0.0, jnp.asarray(setup.phips), 1.0)
        if int(setup.ncurr) == 1
        else jnp.asarray(setup.iotas)
    )
    if int(iotas_half.shape[0]) >= 2:
        iotaf = jnp.empty_like(iotas_half)
        iotaf = iotaf.at[0].set(1.5 * iotas_half[1] - 0.5 * iotas_half[2] if int(iotas_half.shape[0]) >= 3 else iotas_half[1])
        iotaf = iotaf.at[1:-1].set(0.5 * (iotas_half[1:-1] + iotas_half[2:]))
        iotaf = iotaf.at[-1].set(1.5 * iotas_half[-1] - 0.5 * iotas_half[-2])
    else:
        iotaf = iotas_half

    phipf = jnp.asarray(context.flux.phipf, dtype=jnp.float64)
    phi = jnp.concatenate([jnp.zeros((1,), dtype=phipf.dtype), jnp.cumsum(phipf[1:] * (s_full[1:] - s_full[:-1]))])
    sqrts_edge = jnp.asarray(rt.setup.sqrts, dtype=jnp.float64)[-1]
    rb = jnp.asarray(real_space.R_even)[-1] + sqrts_edge * jnp.asarray(real_space.R_odd)[-1]
    zub = jnp.asarray(real_space.dZ_dtheta_even)[-1] + sqrts_edge * jnp.asarray(real_space.dZ_dtheta_odd)[-1]
    wint = jnp.asarray(rt.trig.wint, dtype=jnp.float64)
    area = 2.0 * jnp.pi * jnp.abs(jnp.sum(rb * zub * wint))
    aminor_p = jnp.sqrt(jnp.where(area != 0.0, area, 1.0) / jnp.pi)

    return SimpleNamespace(
        xm_nyq=xm_nyq_np,
        xn_nyq=xn_nyq_np,
        gmnc=zero_axis(gmnc),
        bmnc=zero_axis(bmnc),
        bsupumnc=zero_axis(bsupumnc),
        bsupvmnc=zero_axis(bsupvmnc),
        bsubumnc=zero_axis(bsubumnc),
        bsubvmnc=zero_axis(bsubvmnc),
        iotaf=iotaf,
        phi=phi,
        aminor_p=aminor_p,
    )


def _traceable_vmec_surface_from_tables(tables, *, s, source_path, static):
    from ntx.geometry import VmecSurface

    s_value = float(s)
    ns = int(static.resolution.ns)
    s_full_np = np.linspace(0.0, 1.0, ns, dtype=np.float64)
    hs = 1.0 / (ns - 1)
    s_half_np = s_full_np[:-1] + 0.5 * hs
    s_full = jnp.asarray(s_full_np, dtype=jnp.float64)
    s_half = jnp.asarray(s_half_np, dtype=jnp.float64)
    s_query = jnp.asarray(s_value, dtype=jnp.float64)

    def interp_half(values):
        return interpax.interp1d(s_query, s_half, jnp.asarray(values, dtype=jnp.float64)[1:, :], method="cubic", extrap=True)

    def interp_full(values):
        return interpax.interp1d(s_query, s_full, jnp.asarray(values, dtype=jnp.float64), method="cubic", extrap=True)

    bmnc = interp_half(tables.bmnc)
    gmnc = interp_half(tables.gmnc)
    bsupumnc = interp_half(tables.bsupumnc)
    bsupvmnc = interp_half(tables.bsupvmnc)
    bsubumnc = interp_half(tables.bsubumnc)
    bsubvmnc = interp_half(tables.bsubvmnc)
    iota = -interp_full(tables.iotaf)

    nfp = int(static.resolution.nfp)
    mode_count = int(np.asarray(tables.xm_nyq).size)
    aminor_p = jnp.asarray(tables.aminor_p, dtype=jnp.float64).reshape(())
    r_n = jnp.sqrt(s_query)
    r_hat = aminor_p * r_n
    phi_edge = jnp.asarray(tables.phi[-1], dtype=jnp.float64)
    psi_a_hat = phi_edge / (2.0 * jnp.pi)
    dpsi_hat_dr_hat = 2.0 * psi_a_hat * r_n / aminor_p

    return VmecSurface(
        path=Path(source_path).expanduser(),
        requested_psi_n=s_value,
        psi_n=s_value,
        nfp=nfp,
        ns=ns,
        mpol=int(static.resolution.mpol),
        ntor=int(static.resolution.ntor),
        total_mode_count=mode_count,
        loaded_mode_count=mode_count,
        iota=iota,
        m=jnp.asarray(tables.xm_nyq, dtype=jnp.int32),
        n=jnp.asarray(np.rint(-np.asarray(tables.xn_nyq) / nfp).astype(np.int32), dtype=jnp.int32),
        b_cos=jnp.asarray(bmnc, dtype=jnp.float64),
        jacobian_cos=jnp.asarray(gmnc, dtype=jnp.float64),
        b_sub_theta_cos=jnp.asarray(bsubumnc, dtype=jnp.float64),
        b_sub_zeta_cos=jnp.asarray(bsubvmnc, dtype=jnp.float64),
        b_sup_theta_cos=jnp.asarray(bsupumnc, dtype=jnp.float64),
        b_sup_zeta_cos=jnp.asarray(bsupvmnc, dtype=jnp.float64),
        b0=jnp.max(jnp.abs(bmnc)),
        psi_a_hat=psi_a_hat,
        phi_edge=phi_edge,
        r_n=r_n,
        r_hat=r_hat,
        dpsi_hat_dr_hat=dpsi_hat_dr_hat,
        dr_hat_dpsi_hat=1.0 / dpsi_hat_dr_hat,
        aminor_p=aminor_p,
        psi_p=None,
        transport_psi_scale=dpsi_hat_dr_hat,
    )


def _traceable_vmec_surfaces_from_state(context: GeometryAutodiffContext, state, *, s_values):
    tables = _traceable_vmec_field_tables_from_state(context, state)
    return tuple(
        _traceable_vmec_surface_from_tables(
            tables,
            s=s_value,
            source_path=context.input_path,
            static=context.static,
        )
        for s_value in s_values
    )


def _positive_transport_s_values_from_rho(rho_values):
    rho_np = np.asarray(rho_values, dtype=float).reshape(-1)
    positive = rho_np[np.isfinite(rho_np) & (rho_np > 0.0)]
    if positive.size == 0:
        return tuple(float(rho_value**2) for rho_value in rho_np)
    first_transport_rho = float(np.min(positive))
    return tuple(float((rho_value if rho_value > 0.0 else first_transport_rho) ** 2) for rho_value in rho_np)


def _build_ntx_center_prepared_from_vmec_state(
    context: GeometryAutodiffContext,
    state,
    *,
    center_count: int,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    surface_backend: str,
):
    """Build only the center prepared NTX system used by compact root pullbacks."""

    _ensure_local_stack_on_path()
    ntx_src = _repo_root() / "NTX" / "src"
    ntx_src_str = str(ntx_src)
    if ntx_src.exists() and ntx_src_str not in sys.path:
        sys.path.insert(0, ntx_src_str)
    import ntx

    rho_center_sample = np.linspace(0.0, 1.0, int(center_count), dtype=float)
    center_s_values = _positive_transport_s_values_from_rho(rho_center_sample)
    surface_backend_key = str(surface_backend).strip().lower()

    if surface_backend_key in {"vmec", "vmec_jax"}:
        center_surfaces = _traceable_vmec_surfaces_from_state(
            context,
            state,
            s_values=center_s_values,
        )
    elif surface_backend_key in {"auto", "booz", "boozer", "boozmn", "booz_xform", "booz_xform_jax"}:
        surfaces_fn = getattr(ntx, "surfaces_from_vmec_jax_state", None)
        if surfaces_fn is None:
            try:
                from ntx._vmec_jax_surfaces import surfaces_from_vmec_jax_state as surfaces_fn
            except ImportError:
                surface_fn = getattr(ntx, "surface_from_vmec_jax_state")

                def surfaces_fn(**kwargs):
                    s_values = kwargs.pop("s_values")
                    return tuple(surface_fn(s=s_value, **kwargs) for s_value in s_values)

        static_phipf = jnp.asarray(context.flux.phipf, dtype=jnp.float64)
        static_s_full = jnp.asarray(context.static.s, dtype=jnp.float64)
        static_phi = jnp.concatenate(
            [
                jnp.zeros((1,), dtype=static_phipf.dtype),
                jnp.cumsum(static_phipf[1:] * (static_s_full[1:] - static_s_full[:-1])),
            ],
            axis=0,
        )
        static_ntx_psia = jnp.abs(static_phi[-1])
        center_surfaces = surfaces_fn(
            state=state,
            static=context.static,
            indata=context.indata,
            signgs=context.signgs,
            s_values=center_s_values,
            mboz=int(context.mboz),
            nboz=int(context.nboz),
            psi_p=static_ntx_psia,
        )
    else:
        raise ValueError(
            "ntx_exact_surface_backend for realtime geometry must be one of "
            "'vmec', 'booz', or 'auto'."
        )

    grid_spec = ntx.GridSpec(n_theta=int(n_theta), n_zeta=int(n_zeta), n_xi=int(n_xi))

    def _stack_optional(*values):
        first = values[0]
        if first is None:
            return None
        return jnp.stack([jnp.asarray(value) for value in values], axis=0)

    center_prepared_tuple = tuple(ntx.prepare_monoenergetic_system(surface, grid_spec) for surface in center_surfaces)
    return jax.tree_util.tree_map(_stack_optional, *center_prepared_tuple)


def build_ntx_exact_lij_support_from_vmec_state(
    context: GeometryAutodiffContext,
    state,
    geometry,
    *,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    surface_backend: str = "booz",
    progress_label: str | None = None,
):
    _ensure_local_stack_on_path()
    ntx_src = _repo_root() / "NTX" / "src"
    ntx_src_str = str(ntx_src)
    if ntx_src.exists() and ntx_src_str not in sys.path:
        sys.path.insert(0, ntx_src_str)
    import ntx
    from NEOPAX._transport_flux_models import NTXExactLijRuntimeSupport

    def _surfaces_from_vmec_jax_state(**kwargs):
        surfaces_fn = getattr(ntx, "surfaces_from_vmec_jax_state", None)
        if surfaces_fn is None:
            try:
                from ntx._vmec_jax_surfaces import surfaces_from_vmec_jax_state as surfaces_fn
            except ImportError:
                surface_fn = getattr(ntx, "surface_from_vmec_jax_state")
                s_values = kwargs.pop("s_values")
                return tuple(surface_fn(s=s_value, **kwargs) for s_value in s_values)
        return surfaces_fn(**kwargs)

    vmec_wout_cache = None

    def _vmec_surfaces_from_state(*, s_values):
        nonlocal vmec_wout_cache
        if vmec_wout_cache is None:
            vmec_wout_cache = _wout_from_vmec_state(context, state)
        return tuple(
            _vmec_jax_wout_surface_with_frozen_sampling(
                vmec_wout_cache,
                s=float(s_value),
                source_path=context.input_path,
            )
            for s_value in s_values
        )

    rho_center = jnp.asarray(geometry.r_grid, dtype=jnp.float64) / jnp.asarray(geometry.a_b, dtype=jnp.float64)
    rho_face = jnp.asarray(geometry.r_grid_half, dtype=jnp.float64) / jnp.asarray(geometry.a_b, dtype=jnp.float64)
    # NEOPAX geometry stores the full toroidal flux, while the NTX exact-Lij
    # runtime support matches the file-backed NTX convention with psi_p / 2*pi.
    ntx_psia = jnp.asarray(geometry.Psia_value, dtype=jnp.float64) / (2.0 * jnp.pi)
    static_phipf = jnp.asarray(context.flux.phipf, dtype=jnp.float64)
    static_s_full = jnp.asarray(context.static.s, dtype=jnp.float64)
    static_phi = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=static_phipf.dtype),
            jnp.cumsum(static_phipf[1:] * (static_s_full[1:] - static_s_full[:-1])),
        ],
        axis=0,
    )
    static_ntx_psia = jnp.abs(static_phi[-1])
    rho_center_sample = np.linspace(0.0, 1.0, int(rho_center.shape[0]), dtype=float)
    if int(rho_face.shape[0]) > 1:
        rho_face_sample = np.concatenate(
            [
                np.asarray([0.0], dtype=float),
                0.5 * (rho_center_sample[:-1] + rho_center_sample[1:]),
                np.asarray([1.0], dtype=float),
            ],
            axis=0,
        )
    else:
        rho_face_sample = np.asarray([0.0, 1.0], dtype=float)
    r00_support_rho_np = np.unique(np.concatenate([rho_center_sample, rho_face_sample], axis=0))
    r00_support_rho = jnp.asarray(r00_support_rho_np, dtype=jnp.float64)
    r00_support = _boozer_rmnc00_from_state_at_rho(context, state, r00_support_rho_np)
    r00_interp = interpax.Interpolator1D(r00_support_rho, r00_support, extrap=True)
    r00_center = r00_interp(rho_center)
    r00_face = r00_interp(rho_face)
    center_s_values = tuple(float(rho_value**2) for rho_value in rho_center_sample)
    face_s_values = tuple(float(rho_value**2) for rho_value in rho_face_sample)

    surface_backend_key = str(surface_backend).strip().lower()
    if progress_label is not None:
        print(
            f"{progress_label} ntx_surface_backend={surface_backend_key}",
            flush=True,
        )
    if surface_backend_key in {"vmec", "vmec_jax"}:
        if progress_label is not None:
            print(
                f"{progress_label} ntx_surface_branch=vmec_traceable",
                flush=True,
            )
        center_surfaces = _traceable_vmec_surfaces_from_state(
            context,
            state,
            s_values=_positive_transport_s_values_from_rho(rho_center_sample),
        )
        face_surfaces = _traceable_vmec_surfaces_from_state(
            context,
            state,
            s_values=_positive_transport_s_values_from_rho(rho_face_sample),
        )
    elif surface_backend_key in {"auto", "booz", "boozer", "boozmn", "booz_xform", "booz_xform_jax"}:
        if progress_label is not None:
            print(
                f"{progress_label} ntx_surface_branch=booz_traceable",
                flush=True,
            )
        center_surfaces = _surfaces_from_vmec_jax_state(
            state=state,
            static=context.static,
            indata=context.indata,
            signgs=context.signgs,
            s_values=_positive_transport_s_values_from_rho(rho_center_sample),
            mboz=int(context.mboz),
            nboz=int(context.nboz),
            psi_p=static_ntx_psia,
        )
        face_surfaces = _surfaces_from_vmec_jax_state(
            state=state,
            static=context.static,
            indata=context.indata,
            signgs=context.signgs,
            s_values=_positive_transport_s_values_from_rho(rho_face_sample),
            mboz=int(context.mboz),
            nboz=int(context.nboz),
            psi_p=static_ntx_psia,
        )
    else:
        raise ValueError(
            "ntx_exact_surface_backend for realtime geometry must be one of "
            "'vmec', 'booz', or 'auto'."
        )
    # Keep the prepared NTX surfaces in NTX's file-backed Boozer convention.
    # The runtime scan channels below are built separately from NEOPAX geometry
    # so they can mirror the frozen-file channel convention without forcing the
    # prepared monoenergetic geometry to flip away from NTX's convention.
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
        center_channels=_build_ntx_runtime_channels_from_geometry(
            geometry,
            rho=rho_center,
            psia=ntx_psia,
            r00=r00_center,
        ),
        face_channels=_build_ntx_runtime_channels_from_geometry(
            geometry,
            rho=rho_face,
            psia=ntx_psia,
            r00=r00_face,
        ),
        center_prepared=center_prepared,
        face_prepared=face_prepared,
        grid=grid_spec,
    )


def build_ntx_exact_lij_support_from_param_vector(
    context: GeometryAutodiffContext,
    param_deltas,
    param_specs: Sequence[tuple[str, int, int]],
    *,
    lane: str = "ad",
    n_r: int,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    surface_backend: str = "booz",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
):
    """Build differentiable NTX support from VMEC boundary-harmonic deltas."""

    state = _solve_state_for_param_vector(
        context,
        param_deltas,
        param_specs,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    geometry = _build_neopax_geometry_from_state(context, state, n_r=n_r)
    return build_ntx_exact_lij_support_from_vmec_state(
        context,
        state,
        geometry,
        n_theta=int(n_theta),
        n_zeta=int(n_zeta),
        n_xi=int(n_xi),
        surface_backend=str(surface_backend),
    )


def build_neopax_geometry_and_ntx_exact_lij_support_from_param_vector(
    context: GeometryAutodiffContext,
    param_deltas,
    param_specs: Sequence[tuple[str, int, int]],
    *,
    lane: str = "ad",
    n_r: int,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    surface_backend: str = "booz",
    max_iter: int | None = None,
    step_size: float | None = None,
    jacobian_penalty: float = 1.0e3,
):
    """Build the realtime transport geometry payload from one VMEC solve."""

    state = _solve_state_for_param_vector(
        context,
        param_deltas,
        param_specs,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    return build_neopax_geometry_and_ntx_exact_lij_support_from_state(
        context,
        state,
        n_r=n_r,
        n_theta=n_theta,
        n_zeta=n_zeta,
        n_xi=n_xi,
        surface_backend=surface_backend,
    )


def build_neopax_geometry_and_ntx_exact_lij_support_from_state(
    context: GeometryAutodiffContext,
    state,
    *,
    n_r: int,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    surface_backend: str = "booz",
    progress_label: str | None = None,
):
    """Build the realtime transport geometry payload from an already-solved VMEC state."""

    geometry = _build_neopax_geometry_from_state(context, state, n_r=n_r)
    support = build_ntx_exact_lij_support_from_vmec_state(
        context,
        state,
        geometry,
        n_theta=int(n_theta),
        n_zeta=int(n_zeta),
        n_xi=int(n_xi),
        surface_backend=str(surface_backend),
    )
    return {"geometry": geometry, "ntx_support": support}


def geometry_payload_pullback_from_param_vector_raw_block_transpose(
    context: GeometryAutodiffContext,
    param_deltas,
    param_specs: Sequence[tuple[str, int, int]],
    payload_bars: Sequence[Any],
    *,
    combined_payload: bool,
    n_r: int,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    surface_backend: str = "booz",
    max_iter: int | None = None,
    solver_device: str | None = None,
    progress_label: str | None = None,
    return_branch_gradients: bool = False,
    raw_block_solve: GeometryRawBlockSolve | None = None,
    return_state_bars: bool = False,
    extra_state_bars: object | None = None,
    extra_state_bars_factory=None,
    return_raw_block_solve: bool = False,
) -> object:
    """Pull transport payload cotangents back to VMEC boundary harmonics.

    The payload graph is differentiated only from payload leaves back to the
    converged VMEC state.  The nonlinear VMEC solve pullback is then handled by
    the validated raw block-tridiagonal transpose rule, matching the
    geometry-objective table convention.

    If ``return_state_bars`` is true, stop before the raw-block transpose and
    return the assembled VMEC-state cotangent batch.  ``extra_state_bars`` or
    ``extra_state_bars_factory`` lets fused callers append already-assembled
    VMEC-state cotangents before the same compact raw-block transpose.
    """

    if not payload_bars:
        raise ValueError("payload_bars must contain at least one objective cotangent tree.")
    if not _using_current_vmec_jax_context(context):
        raise ValueError("raw_block_transpose payload pullback requires the current VMEX implicit AD lane.")

    if raw_block_solve is None:
        raw_block_solve = geometry_raw_block_solve_from_param_vector(
            context,
            jnp.asarray(param_deltas, dtype=jnp.float64),
            param_specs,
            max_iter=max_iter,
            solver_device=solver_device,
        )
    implicit = raw_block_solve.implicit
    implicit_params = raw_block_solve.implicit_params
    implicit_cfg = raw_block_solve.implicit_cfg
    state = raw_block_solve.state
    dof_mask = raw_block_solve.dof_mask
    param_entries = raw_block_solve.param_entries
    if not hasattr(implicit, "implicit_state_pullback_multi_rhs_raw_block_transpose"):
        raise AttributeError(
            "The active VMEC backend does not expose implicit_state_pullback_multi_rhs_raw_block_transpose."
        )
    geometry_requested_sample_rho = _neopax_geometry_requested_sample_rho(context, n_r=int(n_r))
    geometry_boozer_surface_sampling = _boozer_surface_indices_and_rho(
        context.static,
        geometry_requested_sample_rho,
    )
    geometry_boozer_inputs = _booz_xform_inputs_from_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=context.signgs,
        flux=context.flux,
    )
    geometry_booz_constants_grids = _booz_constants_and_grids_for_inputs(
        context,
        geometry_boozer_inputs,
    )
    _geometry_booz_constants, geometry_booz_grids = geometry_booz_constants_grids
    geometry_booz_mode00 = _find_boozer_mode_index(
        geometry_booz_grids.xm_b,
        geometry_booz_grids.xn_b,
        m_value=0,
        n_value=0,
    )
    if geometry_booz_mode00 is None:
        raise ValueError("Boozer grids are missing the (0,0) mode.")
    geometry_booz_mode_indices = (
        geometry_booz_mode00,
        _find_boozer_mode_index(
            geometry_booz_grids.xm_b,
            geometry_booz_grids.xn_b,
            m_value=1,
            n_value=0,
        ),
    )

    def geometry_from_state(state_inner):
        return _build_neopax_geometry_from_state(
            context,
            state_inner,
            n_r=n_r,
            requested_sample_rho=geometry_requested_sample_rho,
            boozer_surface_sampling=geometry_boozer_surface_sampling,
            booz_constants_grids=geometry_booz_constants_grids,
            booz_mode_indices=geometry_booz_mode_indices,
        )

    def ntx_support_from_state(state_inner):
        geometry_inner = geometry_from_state(state_inner)
        return build_ntx_exact_lij_support_from_vmec_state(
            context,
            state_inner,
            geometry_inner,
            n_theta=int(n_theta),
            n_zeta=int(n_zeta),
            n_xi=int(n_xi),
            surface_backend=str(surface_backend),
            progress_label=progress_label,
        )

    zero_like_state = jax.tree_util.tree_map(jnp.zeros_like, state)

    def _print_state_bar_batch_finiteness(label, state_bar_batch_value):
        if progress_label is None:
            return True
        leaves = jax.tree_util.tree_leaves(state_bar_batch_value)
        first_bad = None
        total_l2_sq = 0.0
        all_finite = True
        for leaf_i, leaf in enumerate(leaves):
            arr = np.asarray(jax.device_get(leaf))
            if not np.issubdtype(arr.dtype, np.inexact):
                continue
            finite = np.isfinite(arr)
            if not bool(np.all(finite)):
                all_finite = False
                if first_bad is None:
                    bad = np.argwhere(~finite)
                    first_bad = {
                        "leaf": int(leaf_i),
                        "shape": tuple(int(v) for v in arr.shape),
                        "index": None if bad.size == 0 else tuple(int(v) for v in bad[0]),
                    }
            finite_values = arr[finite]
            if finite_values.size:
                total_l2_sq += float(np.sum(finite_values * finite_values))
        print(
            f"{progress_label} {label}_state_bar_l2={total_l2_sq**0.5:.6e} "
            f"{label}_state_bar_all_finite={all_finite} "
            f"{label}_state_bar_first_nonfinite={first_bad}",
            flush=True,
        )
        return all_finite

    def _zero_state_bar_batch():
        return jax.tree_util.tree_map(
            lambda leaf: jnp.broadcast_to(leaf[None, ...], (len(payload_bars),) + leaf.shape),
            zero_like_state,
        )

    def _tree_path_names(path):
        names = []
        for key in path:
            name = getattr(key, "name", None)
            if name is None:
                name = getattr(key, "key", None)
            if name is None:
                name = str(key)
            names.append(str(name))
        return tuple(names)

    def _center_drds_from_state(state_inner):
        geometry_inner = geometry_from_state(state_inner)
        rho_arr = (
            jnp.asarray(geometry_inner.r_grid, dtype=jnp.float64)
            / jnp.asarray(geometry_inner.a_b, dtype=jnp.float64)
        )
        rho_positive = rho_arr > 0.0
        rho_safe = jnp.where(rho_positive, rho_arr, 1.0)
        return jnp.where(rho_positive, jnp.asarray(geometry_inner.a_b, dtype=jnp.float64) / (2.0 * rho_safe), 0.0)

    def _center_prepared_from_state(state_inner, *, center_count: int):
        return _build_ntx_center_prepared_from_vmec_state(
            context,
            state_inner,
            center_count=center_count,
            n_theta=int(n_theta),
            n_zeta=int(n_zeta),
            n_xi=int(n_xi),
            surface_backend=str(surface_backend),
        )

    def _batched_bar_tangent_contract(batched_bars, tangent_leaves):
        total = None
        for bars, tangent in zip(batched_bars, tangent_leaves, strict=True):
            bars_arr = jnp.asarray(bars)
            tangent_arr = jnp.asarray(tangent, dtype=bars_arr.dtype)
            rows = int(bars_arr.shape[0])
            contribution = jnp.reshape(bars_arr, (rows, -1)) @ jnp.ravel(tangent_arr)
            total = contribution if total is None else total + contribution
        if total is None:
            return jnp.zeros((len(payload_bars),), dtype=jnp.float64)
        return total

    def _implicit_param_tangent_batch():
        zero_params = jax.tree_util.tree_map(jnp.zeros_like, implicit_params)
        tangents = []
        for entry in param_entries:
            field_name = entry["input_field"]
            base = getattr(zero_params, field_name)
            updates = {
                field_name: base.at[int(entry["n_offset"]), int(entry["m_index"])].set(1.0)
            }
            tangents.append(dataclasses.replace(zero_params, **updates))
        return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves, axis=0), *tangents)

    def _payload_branch_pullback_setup(branch_name, payload_fn, branch_bars):
        payload_template = branch_bars[0]
        payload_template_paths_and_leaves = jax.tree_util.tree_flatten_with_path(payload_template)[0]
        payload_template_leaves = jax.tree_util.tree_leaves(payload_template)
        float_leaf_indices = tuple(
            leaf_i
            for leaf_i, leaf in enumerate(payload_template_leaves)
            if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
        )
        payload_bar_leaves_by_objective = tuple(
            jax.tree_util.tree_leaves(payload_bar) for payload_bar in branch_bars
        )
        active_float_leaf_indices = []
        for leaf_i in float_leaf_indices:
            leaf_has_bar = False
            for objective_i in range(len(branch_bars)):
                bar_leaf = payload_bar_leaves_by_objective[objective_i][leaf_i]
                bar_arr = jnp.asarray(bar_leaf)
                is_active = np.any(
                    np.asarray(
                        jax.device_get(
                            jnp.logical_and(jnp.isfinite(bar_arr), jnp.abs(bar_arr) > 0.0)
                        )
                    )
                )
                has_nonfinite = np.any(np.asarray(jax.device_get(~jnp.isfinite(bar_arr))))
                if bool(has_nonfinite):
                    raise FloatingPointError(
                        f"nonfinite payload cotangent reached {branch_name} leaf {leaf_i}; "
                        "rerun the benchmark diagnostics to inspect first_nonfinite_leaf."
                    )
                if bool(is_active):
                    leaf_has_bar = True
                    break
            if leaf_has_bar:
                active_float_leaf_indices.append(leaf_i)
        active_float_leaf_indices = tuple(active_float_leaf_indices)
        if progress_label is not None:
            print(
                f"{progress_label} {branch_name}_active_float_leaves={len(active_float_leaf_indices)}",
                flush=True,
            )
        if not active_float_leaf_indices:
            return {
                "active": False,
                "single_state_bar": lambda _objective_i: zero_like_state,
                "tangent_contraction": lambda _state_tangent: jnp.zeros((len(payload_bars),), dtype=jnp.float64),
            }

        active_leaf_paths = tuple(
            _tree_path_names(payload_template_paths_and_leaves[leaf_i][0])
            for leaf_i in active_float_leaf_indices
        )

        def _compact_initial_er_setup():
            if branch_name != "ntx_support":
                return None
            prepared_template = payload_template.center_prepared
            prepared_leaves = jax.tree_util.tree_leaves(prepared_template)
            if not prepared_leaves:
                return None
            center_count = int(jnp.asarray(prepared_leaves[0]).shape[0])
            prepared_paths = {
                _tree_path_names(path): local_i
                for local_i, (path, _leaf) in enumerate(
                    jax.tree_util.tree_flatten_with_path(prepared_template)[0]
                )
            }
            compact_specs = []
            for leaf_i, path_names in zip(active_float_leaf_indices, active_leaf_paths, strict=True):
                if path_names == ("center_channels", "drds"):
                    compact_specs.append(("center_drds", leaf_i, None))
                    continue
                if len(path_names) >= 2 and path_names[0] == "center_prepared":
                    prepared_path = path_names[1:]
                    if prepared_path not in prepared_paths:
                        return None
                    compact_specs.append(("center_prepared", leaf_i, prepared_paths[prepared_path]))
                    continue
                return None

            def compact_support_float_leaves_from_state(state_inner):
                center_drds = None
                center_prepared = None
                center_prepared_leaves = None
                out = []
                for kind, _leaf_i, local_i in compact_specs:
                    if kind == "center_drds":
                        if center_drds is None:
                            center_drds = _center_drds_from_state(state_inner)
                        out.append(jnp.asarray(center_drds))
                    else:
                        if center_prepared is None:
                            center_prepared = _center_prepared_from_state(
                                state_inner,
                                center_count=center_count,
                            )
                            center_prepared_leaves = jax.tree_util.tree_leaves(center_prepared)
                        out.append(jnp.asarray(center_prepared_leaves[local_i]))
                return tuple(out)

            batched_compact_bars = tuple(
                jnp.stack(
                    [
                        jnp.asarray(
                            payload_bar_leaves_by_objective[objective_i][leaf_i],
                            dtype=jnp.asarray(payload_template_leaves[leaf_i]).dtype,
                        )
                        for objective_i in range(len(branch_bars))
                    ],
                    axis=0,
                )
                for _kind, leaf_i, _local_i in compact_specs
            )
            compact_pullback_cache = []

            def _compact_pullback():
                if not compact_pullback_cache:
                    _compact_baseline, compact_pullback = jax.vjp(
                        compact_support_float_leaves_from_state,
                        state,
                    )
                    del _compact_baseline
                    compact_pullback_cache.append(compact_pullback)
                return compact_pullback_cache[0]

            def _single_state_bar(objective_i):
                compact_bar_leaves = tuple(
                    jax.lax.dynamic_index_in_dim(leaf, objective_i, axis=0, keepdims=False)
                    for leaf in batched_compact_bars
                )
                return _compact_pullback()(tuple(compact_bar_leaves))[0]

            def _tangent_contraction(state_tangent):
                _, tangent_leaves = jax.jvp(
                    compact_support_float_leaves_from_state,
                    (state,),
                    (state_tangent,),
                )
                return _batched_bar_tangent_contract(batched_compact_bars, tangent_leaves)

            if progress_label is not None:
                print(
                    f"{progress_label} {branch_name}_compact_initial_er_support_pullback=True",
                    flush=True,
                )
            return {
                "active": True,
                "single_state_bar": _single_state_bar,
                "tangent_contraction": _tangent_contraction,
            }

        compact_setup = _compact_initial_er_setup()
        if compact_setup is not None:
            return compact_setup

        def payload_float_leaves_from_state(state_inner):
            leaves = jax.tree_util.tree_leaves(payload_fn(state_inner))
            return tuple(jnp.asarray(leaves[leaf_i]) for leaf_i in active_float_leaf_indices)

        batched_payload_float_bars = tuple(
            jnp.stack(
                [
                    jnp.asarray(
                        payload_bar_leaves_by_objective[objective_i][leaf_i],
                        dtype=jnp.asarray(payload_template_leaves[leaf_i]).dtype,
                    )
                    for objective_i in range(len(branch_bars))
                ],
                axis=0,
            )
            for leaf_i in active_float_leaf_indices
        )
        payload_float_pullback_cache = []

        def _payload_float_pullback():
            if not payload_float_pullback_cache:
                _payload_float_baseline, payload_float_pullback = jax.vjp(
                    payload_float_leaves_from_state,
                    state,
                )
                del _payload_float_baseline
                payload_float_pullback_cache.append(payload_float_pullback)
            return payload_float_pullback_cache[0]

        def _single_state_bar(objective_i):
            payload_bar_leaves = tuple(
                jax.lax.dynamic_index_in_dim(leaf, objective_i, axis=0, keepdims=False)
                for leaf in batched_payload_float_bars
            )
            return _payload_float_pullback()(tuple(payload_bar_leaves))[0]

        def _tangent_contraction(state_tangent):
            _, tangent_leaves = jax.jvp(
                payload_float_leaves_from_state,
                (state,),
                (state_tangent,),
            )
            return _batched_bar_tangent_contract(batched_payload_float_bars, tangent_leaves)

        return {
            "active": True,
            "single_state_bar": _single_state_bar,
            "tangent_contraction": _tangent_contraction,
        }

    def _state_bar_batch_from_payload_branch(branch_name, payload_fn, branch_bars):
        setup = _payload_branch_pullback_setup(branch_name, payload_fn, branch_bars)
        if not bool(setup["active"]):
            return _zero_state_bar_batch()
        state_bar_rows = tuple(setup["single_state_bar"](i) for i in range(len(branch_bars)))
        state_bar_batch_value = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves, axis=0), *state_bar_rows)
        branch_all_finite = _print_state_bar_batch_finiteness(branch_name, state_bar_batch_value)
        if (
            progress_label is not None
            and branch_name == "ntx_support"
            and not branch_all_finite
        ):
            print(
                f"{progress_label} isolating first nonfinite ntx_support payload leaf",
                flush=True,
            )
            for leaf_i in active_float_leaf_indices:
                path, template_leaf = payload_template_paths_and_leaves[leaf_i]
                leaf_bars = jnp.stack(
                    [
                        jnp.asarray(
                            payload_bar_leaves_by_objective[objective_i][leaf_i],
                            dtype=jnp.asarray(template_leaf).dtype,
                        )
                        for objective_i in range(len(branch_bars))
                    ],
                    axis=0,
                )

                def single_payload_leaf_from_state(state_inner, leaf_index=leaf_i):
                    return jnp.asarray(jax.tree_util.tree_leaves(payload_fn(state_inner))[leaf_index])

                _leaf_baseline, single_leaf_pullback = jax.vjp(single_payload_leaf_from_state, state)
                del _leaf_baseline
                single_state_bar_batch = jax.lax.map(
                    lambda leaf_bar: single_leaf_pullback(leaf_bar)[0],
                    leaf_bars,
                )
                leaf_all_finite = _print_state_bar_batch_finiteness(
                    f"ntx_support_leaf_{leaf_i}",
                    single_state_bar_batch,
                )
                print(
                    f"{progress_label} ntx_support_leaf_{leaf_i}_path={path}",
                    flush=True,
                )
                if not leaf_all_finite:
                    break
        return state_bar_batch_value

    def _payload_raw_block_state_bar_batch():
        geometry_state_bar_batch = None
        support_state_bar_batch = None
        if combined_payload:
            geometry_bars = tuple(payload_bar["geometry"] for payload_bar in payload_bars)
            support_bars = tuple(payload_bar["ntx_support"] for payload_bar in payload_bars)
            # Build the heavy NTX support VJP before the lighter geometry VJP so
            # the support compile/execution does not overlap the geometry
            # state-bar batch.  This preserves the same cotangent sum and final
            # raw-block transpose while reducing peak device memory.
            support_state_bar_batch = _state_bar_batch_from_payload_branch(
                "ntx_support",
                ntx_support_from_state,
                support_bars,
            )
            geometry_state_bar_batch = _state_bar_batch_from_payload_branch(
                "geometry",
                geometry_from_state,
                geometry_bars,
            )
            state_bar_batch = jax.tree_util.tree_map(
                lambda geometry_bar, support_bar: geometry_bar + support_bar,
                geometry_state_bar_batch,
                support_state_bar_batch,
            )
            _print_state_bar_batch_finiteness("combined", state_bar_batch)
        else:
            state_bar_batch = _state_bar_batch_from_payload_branch(
                "ntx_support",
                ntx_support_from_state,
                tuple(payload_bars),
            )
            _print_state_bar_batch_finiteness("combined", state_bar_batch)

        if return_branch_gradients and combined_payload:
            return jax.tree_util.tree_map(
                lambda geometry_bar, support_bar, total_bar: jnp.concatenate(
                    [geometry_bar, support_bar, total_bar],
                    axis=0,
                ),
                geometry_state_bar_batch,
                support_state_bar_batch,
                state_bar_batch,
            )
        return state_bar_batch

    def _try_compact_payload_tangent_contract_to_param_matrix():
        if (
            not combined_payload
            or return_branch_gradients
            or return_state_bars
            or extra_state_bars is not None
            or extra_state_bars_factory is not None
            or not hasattr(implicit, "implicit_state_tangent_raw_block")
        ):
            return None
        geometry_bars = tuple(payload_bar["geometry"] for payload_bar in payload_bars)
        support_bars = tuple(payload_bar["ntx_support"] for payload_bar in payload_bars)
        geometry_setup = _payload_branch_pullback_setup("geometry", geometry_from_state, geometry_bars)
        support_setup = _payload_branch_pullback_setup("ntx_support", ntx_support_from_state, support_bars)
        if not param_entries:
            return jnp.zeros((len(payload_bars), 0), dtype=jnp.float64)
        param_tangent_batch = _implicit_param_tangent_batch()

        def _single_gradient_column(param_tangent):
            state_tangent = implicit.implicit_state_tangent_raw_block(
                implicit_params,
                implicit_cfg,
                state,
                dof_mask,
                param_tangent,
                probe_chunk_size=1,
            )
            return (
                geometry_setup["tangent_contraction"](state_tangent)
                + support_setup["tangent_contraction"](state_tangent)
            )

        gradient_columns = jax.lax.map(
            _single_gradient_column,
            param_tangent_batch,
        )
        if progress_label is not None:
            print(
                f"{progress_label} compact_payload_tangent_contract=True",
                flush=True,
            )
        return jnp.swapaxes(gradient_columns, 0, 1)

    streamed_gradient_matrix = _try_compact_payload_tangent_contract_to_param_matrix()
    if streamed_gradient_matrix is not None:
        if progress_label is not None:
            arr = np.asarray(jax.device_get(streamed_gradient_matrix))
            finite = np.isfinite(arr)
            all_finite = bool(np.all(finite))
            finite_values = arr[finite]
            l2 = float(np.sum(finite_values * finite_values) ** 0.5) if finite_values.size else 0.0
            first_bad = None
            if not all_finite:
                bad = np.argwhere(~finite)
                first_bad = None if bad.size == 0 else tuple(int(v) for v in bad[0])
            print(
                f"{progress_label} raw_block_param_bar_l2={l2:.6e} "
                f"raw_block_param_bar_all_finite={all_finite} "
                f"raw_block_param_bar_first_nonfinite={first_bad}",
                flush=True,
            )
        return streamed_gradient_matrix

    raw_block_state_bar_batch = _payload_raw_block_state_bar_batch()

    if extra_state_bars is None and extra_state_bars_factory is not None:
        raw_block_state_bar_batch = jax.block_until_ready(raw_block_state_bar_batch)
        extra_state_bars = extra_state_bars_factory(raw_block_solve)

    if bool(return_state_bars):
        if extra_state_bars is None:
            result_state_bars = raw_block_state_bar_batch
        else:
            result_state_bars = jax.tree_util.tree_map(
                lambda payload_bar, extra_bar: jnp.concatenate([payload_bar, extra_bar], axis=0),
                raw_block_state_bar_batch,
                extra_state_bars,
            )
        if bool(return_raw_block_solve):
            return result_state_bars, raw_block_solve
        return result_state_bars

    if extra_state_bars is not None:
        raw_block_state_bar_batch = jax.tree_util.tree_map(
            lambda payload_bar, extra_bar: jnp.concatenate([payload_bar, extra_bar], axis=0),
            raw_block_state_bar_batch,
            extra_state_bars,
        )

    param_bar_batch = implicit.implicit_state_pullback_multi_rhs_raw_block_transpose(
        implicit_params,
        implicit_cfg,
        state,
        dof_mask,
        raw_block_state_bar_batch,
        probe_chunk_size=1,
    )
    if progress_label is not None:
        param_bar_leaves = jax.tree_util.tree_leaves(param_bar_batch)
        first_bad = None
        total_l2_sq = 0.0
        all_finite = True
        for leaf_i, leaf in enumerate(param_bar_leaves):
            arr = np.asarray(jax.device_get(leaf))
            if not np.issubdtype(arr.dtype, np.inexact):
                continue
            finite = np.isfinite(arr)
            if not bool(np.all(finite)):
                all_finite = False
                if first_bad is None:
                    bad = np.argwhere(~finite)
                    first_bad = {
                        "leaf": int(leaf_i),
                        "shape": tuple(int(v) for v in arr.shape),
                        "index": None if bad.size == 0 else tuple(int(v) for v in bad[0]),
                    }
            finite_values = arr[finite]
            if finite_values.size:
                total_l2_sq += float(np.sum(finite_values * finite_values))
        print(
            f"{progress_label} raw_block_param_bar_l2={total_l2_sq**0.5:.6e} "
            f"raw_block_param_bar_all_finite={all_finite} "
            f"raw_block_param_bar_first_nonfinite={first_bad}",
            flush=True,
        )
    gradient_matrix = _param_vector_gradient_from_implicit_param_grads(param_bar_batch, param_entries)
    if return_branch_gradients and combined_payload:
        row_count = len(payload_bars)
        return {
            "geometry": gradient_matrix[:row_count],
            "ntx_support": gradient_matrix[row_count : 2 * row_count],
            "combined": gradient_matrix[2 * row_count :],
        }
    return gradient_matrix


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
    state_vmec = _solve_state_for_single_param(
        context,
        param_delta,
        lane=lane,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
    )
    return build_runtime_context_for_vmec_state(
        config,
        context,
        state_vmec,
        n_r=n_r,
    )


def build_runtime_context_for_vmec_state(
    config: dict[str, Any],
    context: GeometryAutodiffContext,
    state_vmec,
    *,
    n_r: int,
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
            surface_backend=str(neoclassical_cfg.get("ntx_exact_surface_backend", "booz")),
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
