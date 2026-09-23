"""Small real-table checks for the direct-centre physical-mesh transpose."""

import collections
import dataclasses
import types

import jax
import jax.numpy as jnp
import pytest

from NEOPAX._database import Monoenergetic
from NEOPAX._species import Species
from NEOPAX._state import TransportState
from NEOPAX._transport_flux_models import (
    CombinedTransportFluxModel,
    NTXDatabaseTransportModel,
    NTXRuntimeScanTransportModel,
    ZeroTransportModel,
    _database_geometry_with_constrained_axis_face,
    _float_delta_tree_like,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _Mesh:
    a_b: object
    rho_grid: object
    rho_grid_half: object
    r_grid: object
    r_grid_half: object
    dr: object
    metric: object
    full_grid_indices: object


def _fixture():
    rho = jnp.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
    base = jnp.arange(80, dtype=jnp.float64).reshape((5, 4, 4))
    database = Monoenergetic(
        a_b=jnp.asarray(1.0), rho=rho,
        nu_log=jnp.asarray([-5.0, -3.0, -1.0, 1.0]),
        Er_list=jnp.broadcast_to(jnp.asarray([-8.0, -5.0, -2.0, 1.0]), (5, 4)),
        D11_log=-3.0 + 0.001 * base,
        D13=0.2 + 0.001 * base,
        D33=0.3 + 0.002 * base,
    )
    centers = jnp.asarray([0.2, 0.4, 0.6, 0.8])
    geometry = _Mesh(jnp.asarray(1.0), centers, rho, centers, rho,
                     jnp.asarray(0.2), jnp.asarray(1.7), jnp.arange(4))
    species = Species(
        number_species=2, species_indices=jnp.asarray([0, 1]),
        mass_mp=jnp.asarray([5.446e-4, 2.0]), charge_qp=jnp.asarray([-1.0, 1.0]),
        names=("e", "D"),
    )
    energy = collections.namedtuple(
        "CenterGeometryEnergy",
        "xWeights L11_weight L12_weight L22_weight L13_weight L23_weight L33_weight v_norm",
    )(
        jnp.asarray([0.25, 0.75]), jnp.asarray([1.0, 0.7]),
        jnp.asarray([0.1, -0.2]), jnp.asarray([0.8, 1.2]),
        jnp.asarray([0.4, 0.5]), jnp.asarray([-0.3, 0.2]),
        jnp.asarray([1.1, 0.6]), jnp.asarray([1.2, 1.8]),
    )
    density = jnp.asarray([[1.0, 1.05, 1.1, 1.15], [0.9, 0.95, 1.0, 1.05]])
    temperature = jnp.asarray([[2.0, 2.1, 2.2, 2.3], [1.6, 1.7, 1.8, 1.9]])
    state = TransportState(
        density=density, pressure=density * temperature,
        Er=jnp.asarray([1.0e-4, -1.2e-4, 1.5e-4, -1.8e-4]),
    )
    model = NTXDatabaseTransportModel(species, energy, geometry, database)
    return model, state, geometry


def _bars(rows):
    # Independent channel/row weights, not repeated copies of one objective.
    values = jnp.arange(rows * 3 * 8, dtype=jnp.float64).reshape(rows, 3, 2, 4)
    values = jnp.sin(0.7 + values) * jnp.asarray([1.0, 0.03, 2.0])[None, :, None, None]
    return {name: values[:, i] if rows > 1 else values[0, i]
            for i, name in enumerate(("Gamma", "Q", "Upar"))}


def _assert_tree_close(actual, expected):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for one, two in zip(jax.tree_util.tree_leaves(actual),
                        jax.tree_util.tree_leaves(expected), strict=True):
        assert one.shape == two.shape
        assert jnp.all(jnp.isfinite(one))
        assert jnp.all(jnp.isfinite(two))
        assert jnp.allclose(one, two, rtol=3e-10, atol=3e-10)


@pytest.mark.parametrize("rows", [1, 10])
def test_database_center_mesh_jvp_matches_radius_vjp_and_direct_flux(rows):
    """The compact rule equals the old rule AND the built-in forward flux VJP."""
    model, state, geometry = _fixture()
    bars = _bars(rows)
    zero = _float_delta_tree_like(geometry)

    @jax.jit
    def _reference(state, geometry, bars):
        def _forward(delta):
            physical = _database_geometry_with_constrained_axis_face(geometry, delta)
            # Fixed table means its scale and derived radial limits do not move.
            return dataclasses.replace(model, geometry=physical)(state)

        _, forward_pullback = jax.vjp(_forward, zero)
        return (jax.vmap(lambda bar: forward_pullback(bar)[0])(bars)
                if rows > 1 else forward_pullback(bars)[0])

    expected = _reference(state, geometry, bars)
    old = jax.jit(lambda state, bars, geometry: model.pullback_direct_rhs_geometry_by_radius(
        state, bars, geometry, center_geometry_mode="radial_vjp"
    ))(state, bars, geometry)
    compact = jax.jit(lambda state, bars, geometry: model.pullback_direct_rhs_geometry_by_radius(
        state, bars, geometry, center_geometry_mode="scalar_jvp"
    ))(state, bars, geometry)
    _assert_tree_close(compact, old)
    _assert_tree_close(compact, expected)
    assert jnp.any(jnp.abs(compact.a_b) > 1e-8)


def test_database_center_mesh_jvp_supports_batched_heat_only_bars():
    """Absent Gamma/Upar channels must not remove the objective axis."""
    model, state, geometry = _fixture()
    bars = _bars(3)
    heat_only = {"Q": bars["Q"]}
    filled = {"Gamma": jnp.zeros_like(bars["Q"]), "Q": bars["Q"],
              "Upar": jnp.zeros_like(bars["Q"])}
    actual = jax.jit(model.pullback_direct_rhs_geometry_by_radius)(state, heat_only, geometry)
    expected = jax.jit(model._pullback_direct_rhs_geometry_by_radius_vjp)(state, filled, geometry)
    _assert_tree_close(actual, expected)


def test_database_composite_center_geometry_batches_non_neoclassical_rows():
    """The opt-in matrix-RHS path retains turbulent centre geometry bars."""

    @dataclasses.dataclass(frozen=True)
    class _GeometryTurbulence:
        field: object

        def __call__(self, state):
            heat = self.field.metric * state.density
            zero = jnp.zeros_like(heat)
            return {"Gamma": zero, "Q": heat, "Upar": zero}

    neo, state, geometry = _fixture()
    model = CombinedTransportFluxModel(
        neo,
        _GeometryTurbulence(geometry),
        ZeroTransportModel(),
        geometry=geometry,
    )
    bars = _bars(3)
    actual = jax.jit(
        lambda values: model.pullback_direct_rhs_geometry_by_radius(
            state, values, geometry
        )
    )(bars)
    expected = jax.jit(
        jax.vmap(
            lambda one_row: model.pullback_direct_rhs_geometry_by_radius(
                state, one_row, geometry
            )
        )
    )(bars)
    _assert_tree_close(actual, expected)
    assert jnp.any(jnp.abs(actual.metric) > 1e-8)


def test_database_center_mesh_jvp_preserves_custom_geometry_dependencies():
    """A custom local evaluator may depend on more than the radial mesh."""
    class CustomModel(NTXDatabaseTransportModel):
        def build_local_direct_flux_evaluator(self, state):
            def evaluate(index, er):
                value = self.geometry.metric * state.density[:, index] + er
                return {"Gamma": value, "Q": 2 * value, "Upar": -value}
            return evaluate

        def _pullback_direct_rhs_physical_mesh_geometry(self, *_args):
            raise AssertionError("Custom evaluators must retain the general VJP")

    base, state, geometry = _fixture()
    model = CustomModel(base.species, base.energy_grid, geometry, base.database)
    bars = _bars(1)
    actual = jax.jit(model.pullback_direct_rhs_geometry_by_radius)(state, bars, geometry)
    expected_metric = jnp.sum(
        state.density * (bars["Gamma"] + 2 * bars["Q"] - bars["Upar"])
    )
    assert jnp.allclose(actual.metric, expected_metric)
    assert jnp.allclose(actual.a_b, 0.0)


@pytest.mark.parametrize("mode, expected", [(None, "scalar_jvp"), ("radial_vjp", "radial_vjp"), ("scalar_jvp", "scalar_jvp")])
def test_database_center_geometry_modes_dispatch_through_composite_and_runtime(mode, expected):
    """Explicit modes traverse both wrappers; the omitted mode retains current dispatch."""
    model, state, geometry = _fixture()
    seen = []

    def _selected(selected):
        def _pullback(_state, _bars, value):
            seen.append(selected)
            return _float_delta_tree_like(value)
        return _pullback

    object.__setattr__(model, "_pullback_direct_rhs_geometry_by_radius_vjp", _selected("radial_vjp"))
    object.__setattr__(model, "_pullback_direct_rhs_physical_mesh_geometry", _selected("scalar_jvp"))
    runtime = types.SimpleNamespace(_database_model=lambda: model)
    runtime.pullback_direct_rhs_geometry_by_radius = types.MethodType(
        NTXRuntimeScanTransportModel.pullback_direct_rhs_geometry_by_radius, runtime
    )
    composite = CombinedTransportFluxModel(runtime, ZeroTransportModel(), ZeroTransportModel())
    actual = jax.jit(lambda bars: composite.pullback_direct_rhs_geometry_by_radius(
        state, bars, geometry, center_geometry_mode=mode
    ))(_bars(1))
    assert seen == [expected]
    _assert_tree_close(actual, _float_delta_tree_like(geometry))


@pytest.mark.parametrize("mode", ["unknown", "", "radial"])
def test_database_center_geometry_rejects_unknown_mode(mode):
    model, state, geometry = _fixture()
    with pytest.raises(ValueError, match="center_geometry_mode"):
        model.pullback_direct_rhs_geometry_by_radius(
            state, _bars(1), geometry, center_geometry_mode=mode
        )
