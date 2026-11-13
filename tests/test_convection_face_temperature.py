"""Small face-convection and reverse checks; no NTX or transport solve."""
import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from NEOPAX._boundary_conditions import build_boundary_condition_model
from NEOPAX._orchestrator import _normalize_solver_config
from NEOPAX._state import TransportState
from NEOPAX._transport_equations import (
    ComposedEquationSystem, TemperatureEquation, build_equation_system,
    build_temperature_equation,
)


def species():
    return SimpleNamespace(names=("e", "D"), number_species=2,
        charge_qp=jnp.array([-1., 1.]), ion_indices=(1,))


def field(scale=1.):
    faces=scale*jnp.array([0., .2, .6, 1.])
    centres=.5*(faces[:-1]+faces[1:])
    return SimpleNamespace(r_grid=centres, r_grid_half=faces, dr=scale/3,
        Vprime=1+centres, Vprime_half=1+faces)


def state():
    n=jnp.array([[2., 1.5, 1.], [2., 1.5, 1.]])
    t=jnp.array([[10., 8., 6.], [4., 6., 8.]])
    return TransportState(n, n*t, jnp.zeros(3))


def fluxes():
    g=jnp.array([[1., -2., 0., -1.], [-3., 1., -2., 2.]])
    return {
        "Gamma": jnp.zeros((2,3)), "Q": jnp.zeros((2,3)),
        "Q_faces": jnp.array([[1., 2., 3., 4.], [-2., 1., -1., 3.]])*1e23,
        "Gamma_neo_faces": g*1e20,
        "Gamma_turb_faces": -.25*g*1e20,
        "Gamma_classical_faces": .15*g*1e20,
    }


def direct_equation(mode=None, **kwargs):
    options=dict(
        dr_cells=jnp.full(3,.5), Vprime=jnp.ones(3), Vprime_half=jnp.ones(4),
        flux_model=None, flux_faces_builder=lambda f,*_: f,
        temperature_ghost_builder=lambda t: jnp.pad(t,((0,0),(1,1)),mode="edge"),
        charge_qp=species().charge_qp, active_species_mask=jnp.ones(2,dtype=bool),
        species=species(), include_work_term=False,
        heat_flux_reconstruction="closure_face_flux", convection_reconstruction="closure_face_flux",
        temperature_faces_builder=lambda s: jnp.array([[11.,9.,7.,5.],[3.,5.,7.,9.]]),
    )
    if mode is not None:
        options["convection_temperature_mode"]=mode
    options.update(kwargs)
    return TemperatureEquation(**options)


def factory(mode="face", geometry=None, model=None, bc=None, **kwargs):
    options=dict(include_work_term=False, include_neo_convection=True,
        include_turbulent_convection=False, include_classical_convection=False,
        heat_flux_reconstruction="closure_face_flux", convection_reconstruction="closure_face_flux",
        convection_temperature_mode=mode, charge_qp=species().charge_qp)
    options.update(kwargs)
    return build_temperature_equation(
        field() if geometry is None else geometry, model, None, species(), bc, **options)


@pytest.mark.parametrize("mode", [None, "upwind"])
def test_default_keeps_original_upwind_arithmetic_for_all_flux_channels(mode):
    def unexpected_builder(s):
        raise AssertionError("Default mode must not construct another face state")
    eq=direct_equation(mode,temperature_faces_builder=unexpected_builder)
    f=fluxes()
    ghost=jnp.pad(state().temperature,((0,0),(1,1)),mode="edge")
    convection=jnp.zeros((2,4))
    for key in ("Gamma_neo_faces","Gamma_turb_faces","Gamma_classical_faces"):
        gamma=1e-20*f[key]
        convection=convection+jnp.where(gamma>=0,ghost[:,:-1],ghost[:,1:])*gamma
    energy=1e-23*f["Q_faces"]+convection
    expected=(2./3.)*(-jnp.diff(energy,axis=1)/.5)
    np.testing.assert_array_equal(eq(state(),f),expected)
    np.testing.assert_allclose(eq.debug_components(state(),f)["pressure_rhs"],expected,atol=1e-13)


def test_face_mode_uses_one_face_temperature_for_all_channels_and_both_flow_signs():
    eq=direct_equation("face")
    f=fluxes()
    face=eq.temperature_faces_builder(state())
    diagnostics=eq.debug_components(state(),f)
    convection=jnp.zeros((2,4))
    for channel in ("neo","turb","classical"):
        expected=(1e-20*f[f"Gamma_{channel}_faces"])*face
        np.testing.assert_allclose(diagnostics[f"convective_{channel}_faces"],expected,atol=1e-13)
        convection=convection+expected
    expected_rhs=(2./3.)*(-jnp.diff(1e-23*f["Q_faces"]+convection,axis=1)/.5)
    np.testing.assert_allclose(jax.jit(eq)(state(),f),expected_rhs,atol=1e-12)
    np.testing.assert_allclose(diagnostics["pressure_rhs"],expected_rhs,atol=1e-12)


def test_factory_uses_the_transport_face_temperature_boundaries_and_floor_without_extra_flux_calls():
    class CaptureFlux:
        calls=0
        def __call__(self,s):
            return {"Gamma":jnp.zeros_like(s.density),"Q":jnp.zeros_like(s.pressure)}
        def evaluate_face_fluxes(self,s,face,**kwargs):
            self.calls+=1
            self.face_temperature=face.temperature
            return {"Q_faces":2*face.temperature*1e23,
                "Gamma_faces":fluxes()["Gamma_neo_faces"],
                "Gamma_neo_faces":fluxes()["Gamma_neo_faces"]}
    bc=build_boundary_condition_model({"left":{"type":"neumann","gradient":{"default":0.}},
        "right":{"type":"robin","value":{"default":1.},"decay_length":{"e":.2,"D":.35}}},1/3,species_names=species().names)
    model=CaptureFlux()
    current=dataclasses.replace(state(),pressure=state().density*jnp.array([[.5,2.,4.],[4.,2.,.25]]))
    eq=factory(model=model,bc=bc,temperature_floor=1.)
    diagnostics=eq.debug_components(current)
    assert model.calls==1
    np.testing.assert_allclose(eq.temperature_faces_builder(current),model.face_temperature,rtol=1e-14)
    np.testing.assert_allclose(diagnostics["Q_faces"],2*model.face_temperature,atol=1e-13)
    np.testing.assert_allclose(diagnostics["convective_neo_faces"],
        1e-20*fluxes()["Gamma_neo_faces"]*model.face_temperature,atol=1e-13)
    assert model.calls==1


def test_face_mode_does_not_require_a_builder_when_all_convection_is_disabled():
    eq=direct_equation("face",temperature_faces_builder=None,
        include_neo_convection=False,include_turbulent_convection=False,include_classical_convection=False)
    np.testing.assert_allclose(eq(state(),fluxes()),(2./3.)*(-jnp.diff(1e-23*fluxes()["Q_faces"],axis=1)/.5),atol=1e-13)


def test_face_temperature_and_flux_joint_vjp_matches_fd_including_zero_flux():
    eq=factory()
    current=state()
    gamma=fluxes()["Gamma_neo_faces"]*1e-20
    dp=jnp.array([[.2,-.1,.3],[-.2,.4,.1]])
    dg=jnp.linspace(-.7,.8,8).reshape(2,4)
    weights=jnp.array([[.2,-.3,.7],[.1,.4,-.5]])
    def objective(pressure,g):
        return jnp.sum(eq(dataclasses.replace(current,pressure=pressure),
            {**fluxes(),"Gamma_neo_faces":g*1e20})*weights)
    bp,bg=jax.jit(jax.grad(objective,argnums=(0,1)))(current.pressure,gamma)
    assert np.isfinite(bp).all() and np.isfinite(bg).all()
    assert float(bg[0,2])!=0.  # d(Gamma*T)/dGamma remains T at Gamma=0.
    step=1e-5
    fd=(objective(current.pressure+step*dp,gamma+step*dg)-objective(current.pressure-step*dp,gamma-step*dg))/(2*step)
    np.testing.assert_allclose(jnp.vdot(bp,dp)+jnp.vdot(bg,dg),fd,rtol=1e-8,atol=1e-8)


def test_shared_flux_reverse_and_direct_profile_reverse_include_face_temperature():
    eq=factory()
    current=state()
    system=ComposedEquationSystem(equations=(eq,),temperature_equation=eq,species=species())
    weights=jnp.array([[.2,-.3,.7],[.1,.4,-.5]])
    bar=TransportState(jnp.zeros((2,3)),weights,jnp.zeros(3))
    working,eidx=system._prepare_working_state(current)
    f=fluxes()
    actual_flux=system.pullback_shared_fluxes(current,f,bar)
    expected_flux=jax.grad(lambda g:jnp.sum(eq(working,{**f,"Gamma_neo_faces":g*1e20})*weights))(f["Gamma_neo_faces"]*1e-20)
    np.testing.assert_allclose(actual_flux["Gamma_neo_faces"]*1e20,expected_flux,atol=1e-12)
    actual_state=system._pullback_shared_flux_rhs_state(current,working,eidx,f,bar)
    _,pullback=jax.vjp(lambda s:eq(s,f),working)
    expected_state=pullback(weights)[0]
    assert np.any(np.asarray(actual_state.pressure)!=0.)
    for actual,expected in zip(jax.tree_util.tree_leaves(actual_state),jax.tree_util.tree_leaves(expected_state),strict=True):
        np.testing.assert_allclose(actual,expected,atol=1e-12)


def test_face_mode_geometry_vjp_matches_fd_on_nonuniform_grid():
    def objective(scale):
        return jnp.sum(factory(geometry=field(scale))(state(),fluxes())*jnp.array([[.2,-.3,.7],[.1,.4,-.5]]))
    scale=jnp.asarray(1.3)
    actual=jax.grad(objective)(scale)
    step=1e-5
    expected=(objective(scale+step)-objective(scale-step))/(2*step)
    np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-8)


@pytest.mark.parametrize("mode", [None,"upwind","face"])
def test_option_is_wired_from_solver_config_into_temperature_equation(mode):
    cfg={"equations":{"toggle_density":[False,False],"toggle_temperature":[True,True],"toggle_Er":False}}
    if mode is not None:
        cfg["transport_solver"]={"temperature_convection_temperature_mode":mode}
    settings=_normalize_solver_config(cfg)
    eqs=build_equation_system(cfg,species(),field(),None,solver_cfg=settings)
    assert eqs[0].convection_temperature_mode == ("upwind" if mode is None else mode)
    assert eqs[0].temperature_faces_builder is not None


@pytest.mark.parametrize("mode", ["centered","ntssfusion",True,None,""])
def test_invalid_modes_fail_before_runtime_construction(mode):
    with pytest.raises(ValueError,match="temperature_convection_temperature_mode"):
        _normalize_solver_config({"transport_solver":{"temperature_convection_temperature_mode":mode}})
    with pytest.raises(ValueError,match="temperature_convection_temperature_mode"):
        factory(mode=mode)
