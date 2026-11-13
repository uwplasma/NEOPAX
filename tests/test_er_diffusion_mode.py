"""Eight-cell diffusion/AD checks; no geometry, NTX or transport solve."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from NEOPAX._boundary_conditions import build_boundary_condition_model, left_constraints_from_bc_model
from NEOPAX._cell_variable import make_profile_cell_variable
from NEOPAX._fem import conservative_update
from NEOPAX._orchestrator import _normalize_solver_config
from NEOPAX._state import TransportState
from NEOPAX._transport_equations import build_electric_field_equation, build_equation_system


def field(scale=1.):
    faces = scale*jnp.linspace(0., 1., 9)
    centres = .5*(faces[:-1]+faces[1:])
    return SimpleNamespace(
        r_grid=centres, r_grid_half=faces, Vprime=centres, Vprime_half=faces,
        B0=jnp.ones(8), enlogation=jnp.ones(8), iota=jnp.ones(8),
    )


def species():
    return SimpleNamespace(
        names=("e", "D", "T", "He"), number_species=4,
        mass=jnp.array([.000544617, 2., 3., 4.]),
        charge_qp=jnp.array([-1., 1., 1., 2.]),
    )


def boundary(right=2., floating=False):
    return build_boundary_condition_model({
        "left": {"type": "dirichlet", "value": 0.},
        "right": {"type": "neumann", "gradient": 0.} if floating
                 else {"type": "dirichlet", "value": right},
    }, .125)


def equation(mode=None, *, geometry=None, bc=None, **kwargs):
    sp = species()
    settings = dict(
        Er_relax=.1, DEr=2., source_mode="ambipolar_local",
        permitivity_mode="ntss_like_midpoint",
    )
    if mode is not None:
        settings["diffusion_mode"] = mode
    settings.update(kwargs)
    return build_electric_field_equation(
        field() if geometry is None else geometry, sp.names, None,
        sp.mass, sp.charge_qp, None, boundary() if bc is None else bc, **settings,
    )


def state(er):
    return TransportState(jnp.ones((4, 8)), jnp.ones((4, 8)), er)


def zero_fluxes():
    return {"Gamma": jnp.zeros((4, 8)), "Gamma_faces": jnp.zeros((4, 9))}


@pytest.mark.parametrize("mode", [None, "neopax"])
def test_default_diffusion_bitwise_preserves_the_original_operator(mode):
    geometry = field()
    profile = jnp.sin(4*geometry.r_grid) + .2
    left_value, left_gradient = left_constraints_from_bc_model(
        boundary(), profile[0], profile=profile, face_centers=geometry.r_grid_half,
    )
    # The original factory's complete arithmetic, including both boundaries.
    original_flux = -make_profile_cell_variable(
        profile, geometry.r_grid_half,
        left_face_constraint=jnp.asarray(left_value).reshape(-1)[0],
        left_face_grad_constraint=jnp.asarray(left_gradient).reshape(-1)[0],
        right_face_constraint=jnp.asarray(2.),
    ).face_grad()
    original_rhs = conservative_update(
        original_flux, jnp.diff(geometry.r_grid_half), geometry.Vprime, geometry.Vprime_half,
    )
    actual_flux, actual_rhs = equation(mode)._er_diffusion(profile)
    np.testing.assert_array_equal(actual_flux, original_flux)
    np.testing.assert_array_equal(actual_rhs, original_rhs)


def test_ntss_linear_radial_field_has_zero_diffusion_including_the_axis():
    geometry = field()
    eq = equation("ntssfusion")
    profile = 2*geometry.r_grid
    flux, diffusion = eq._er_diffusion(profile)
    np.testing.assert_allclose(flux, 0., atol=1e-13)
    np.testing.assert_allclose(diffusion, 0., atol=1e-12)
    np.testing.assert_allclose(eq(state(profile), zero_fluxes()), 0., atol=1e-12)


def test_quadratic_field_distinguishes_ntss_from_scalar_radial_diffusion():
    profile = field().r_grid**2
    _, ntss = equation("ntssfusion", bc=boundary(1.))._er_diffusion(profile)
    _, legacy = equation("neopax", bc=boundary(1.))._er_diffusion(profile)
    # Away from boundary stencils: (r*E')'/r=4; (r*(E'-E/r))'/r=2.
    np.testing.assert_allclose(ntss[1:-2], 2., atol=1e-12)
    np.testing.assert_allclose(legacy[1:-2], 4., atol=1e-12)


def test_ntss_flux_preserves_the_conservative_volume_balance():
    geometry = field()
    profile = jnp.cos(3*geometry.r_grid)
    flux, diffusion = equation("ntssfusion")._er_diffusion(profile)
    integrated = jnp.sum(diffusion*geometry.Vprime*jnp.diff(geometry.r_grid_half))
    expected = -(geometry.Vprime_half[-1]*flux[-1]-geometry.Vprime_half[0]*flux[0])
    np.testing.assert_allclose(integrated, expected, atol=1e-12)
    assert float(flux[0]) == 0.


def test_explicit_outer_node_is_used_in_the_er_over_r_face_term():
    profile = jnp.sin(field().r_grid)
    edge = jnp.asarray(2.7)
    old_flux, _ = equation("neopax", bc=boundary(floating=True))._er_diffusion(profile, edge)
    new_flux, _ = equation("ntssfusion", bc=boundary(floating=True))._er_diffusion(profile, edge)
    np.testing.assert_allclose(new_flux[-1]-old_flux[-1], edge/field().r_grid_half[-1], atol=1e-13)


@pytest.mark.parametrize("mode", ["neopax", "ntssfusion"])
def test_zero_DEr_bypasses_the_flux_and_keeps_nonfinite_fields_out(mode):
    flux, diffusion = equation(mode, DEr=0.)._er_diffusion(jnp.full(8, jnp.nan))
    np.testing.assert_array_equal(flux, jnp.zeros(9))
    np.testing.assert_array_equal(diffusion, jnp.zeros(8))


def test_floating_outer_edge_remains_pure_ambipolar_relaxation():
    fluxes = zero_fluxes()
    fluxes["Gamma_faces"] = fluxes["Gamma_faces"].at[1, -1].set(1e20)
    current = state(jnp.sin(field().r_grid))
    old = equation("neopax", bc=boundary(floating=True), boundary_mode="floating_ambipolar_edge")
    new = equation("ntssfusion", bc=boundary(floating=True), boundary_mode="floating_ambipolar_edge")
    np.testing.assert_array_equal(old(current, fluxes)[-1], new(current, fluxes)[-1])
    assert float(new(current, fluxes)[-1]) != 0.


def test_ntss_axis_and_outer_node_vjp_agree_with_directional_fd():
    eq = equation("ntssfusion", bc=boundary(floating=True), boundary_mode="floating_ambipolar_edge_node")
    values = jnp.concatenate([jnp.sin(4*field().r_grid), jnp.array([.7])])
    direction = jnp.linspace(-.8, 1.1, 9)
    weights = jnp.linspace(.2, 1.3, 8)

    def objective(x):
        return jnp.vdot(weights, eq(state(x[:-1]), zero_fluxes(), er_edge_override=x[-1]))

    value, gradient = jax.jit(jax.value_and_grad(objective))(values)
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
    assert float(gradient[-1]) != 0.
    step = 1e-6
    fd = (objective(values+step*direction)-objective(values-step*direction))/(2*step)
    np.testing.assert_allclose(jnp.vdot(gradient, direction), fd, rtol=1e-8, atol=1e-8)


def test_ntss_geometry_scale_vjp_has_the_expected_inverse_square_scaling():
    profile = jnp.sin(3*field().r_grid)
    weights = jnp.linspace(.3, 1.1, 8)

    def objective(scale):
        _, rhs = equation("ntssfusion", geometry=field(scale))._er_diffusion(profile)
        return jnp.vdot(weights, rhs)

    scale = jnp.asarray(1.3)
    value, derivative = jax.value_and_grad(objective)(scale)
    np.testing.assert_allclose(derivative, -2*value/scale, rtol=1e-12, atol=1e-11)


def test_ntss_supports_the_legacy_ghost_boundary_builder():
    class GhostBoundary:
        def apply_ghost(self, profile):
            return jnp.concatenate([-profile[:1], profile, 4-profile[-1:]])

    eq = equation("ntssfusion", bc=GhostBoundary())
    profile = 2*field().r_grid
    flux, rhs = eq._er_diffusion(profile)
    np.testing.assert_allclose(flux, 0., atol=1e-12)
    np.testing.assert_allclose(rhs, 0., atol=1e-12)
    node_flux, _ = eq._er_diffusion(profile, jnp.asarray(2.3))
    np.testing.assert_allclose(node_flux[-1], -(2.3-profile[-1])/.0625+2.3, atol=1e-12)


@pytest.mark.parametrize("mode", [None, "neopax", "ntssfusion"])
def test_toml_solver_option_reaches_the_equation_builder(mode):
    cfg = {"equations": {"toggle_density": [False]*4, "toggle_temperature": [False]*4, "toggle_Er": True}}
    if mode is not None:
        cfg["transport_solver"] = {"Er_diffusion_mode": mode}
    settings = _normalize_solver_config(cfg)
    equations = build_equation_system(
        config=cfg, species=species(), field=field(), flux_model=None,
        solver_cfg=settings, boundary_models={"Er": boundary()},
    )
    assert len(equations) == 1
    assert equations[0].diffusion_mode == ("neopax" if mode is None else mode)


@pytest.mark.parametrize("mode", ["ntss", True, 0, "", None])
def test_invalid_diffusion_modes_fail_in_config_and_direct_factory(mode):
    with pytest.raises(ValueError, match="Er_diffusion_mode"):
        _normalize_solver_config({"transport_solver": {"Er_diffusion_mode": mode}})
    with pytest.raises(ValueError, match="Er_diffusion_mode"):
        equation(diffusion_mode=mode)
