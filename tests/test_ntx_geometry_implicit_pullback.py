"""Exact local checks for the opt-in geometry-only NTX support pullback."""

from types import SimpleNamespace
import dataclasses
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import ntx
import pytest

import NEOPAX._neoclassical as neoclassical_module
import NEOPAX._reverse_ad_initial_er as initial_er_module

from NEOPAX._transport_flux_models import (
    CombinedTransportFluxModel,
    NTXDatabaseTransportModel,
    NTXExactLijRuntimeSupport,
    NTXExactLijRuntimeTransportModel,
    NTXRuntimeScanChannels,
    _extract_right_constraints,
    _sanitize_float_delta_bar_tree,
)
from NEOPAX._database_preprocessed import PreprocessedMonoenergetic3DNTSSRadius
from NEOPAX._database import Monoenergetic
from NEOPAX._neoclassical import (
    _collisionality_kind,
    _ntss_radial_flux_correction_terms,
    get_A_matrix,
    get_Matrix,
    get_correction_matrix,
    get_corrected_fluxes,
)
from NEOPAX._energy_grid_models import StandardLaguerreEnergyGrid
from NEOPAX._species import Species
from NEOPAX._state import TransportState, get_v_thermal, safe_density
from NEOPAX._transport_equations import ComposedEquationSystem
from NEOPAX._transport_solvers import (
    _flat_rhs_build_support_pullback_batched_interpolated_faces_factory,
    _flat_rhs_state_and_lagged_response_pullback_factory,
    _lagged_response_build_state_and_support_pullback_batched_interpolated_faces_hook,
    _radau_exact_stage_residual_support_pullback,
    _radau_prepare_lagged_response_with_compact_coefficient_record,
)
from NEOPAX._geometry_autodiff import (
    _native_vmec_coefficient_tangent_contraction,
    _ntx_runtime_channel_payload_bars,
)
from NEOPAX._reverse_ad_transport import (
    _initial_cache_support_pullback_from_rebuild_dispatch,
    _initial_direct_rhs_support_pullback_batched,
    _initial_lagged_response_joint_state_and_support_pullback,
    _merge_rebuild_ntx_channels_into_generic_payload_bar,
    _objective_vector_vjp_rows,
    _realized_reverse_slot_branches,
    _run_realized_reverse_slot_dispatch,
    _take_batched_pytree_row,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _MockSupportBar:
    center_channels: object
    face_channels: object
    center_prepared: object
    face_prepared: object


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _ToyBootstrapGeometry:
    """Minimal differentiable geometry payload for the terminal-VJP gate."""

    scale: object


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _TestMomentumGeometry:
    """Minimum JAX-pytree geometry accepted by the momentum matrix JIT."""

    a_b: object
    r_grid: object
    r_grid_half: object
    Bsqav: object
    G_PS: object
    B0: object
    full_grid_indices: object = None
    dr: object = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _ToyBootstrapChannels:
    """Only the sparse ``drds`` channel touched by the compact helper."""

    drds: object


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _ToyBootstrapSupport:
    center_channels: object
    face_channels: object
    center_prepared: object
    face_prepared: object


@dataclasses.dataclass(frozen=True)
class _ToyBootstrapModel:
    """Small analytic model exercising the real compact-helper contracts."""

    geometry: object
    support: object

    def with_support_payload(self, support):
        return dataclasses.replace(self, support=support)

    def _momentum_corrected_upar_one_radius(self, state, radius_index, *, support=None):
        support = self.support if support is None else support
        prepared = support.center_prepared[0][radius_index]
        drds = support.center_channels.drds[radius_index]
        # Keep state, sparse support, and geometry genuinely coupled, while
        # remaining a no-rollout analytic fixture.
        return (
            state.pressure[:, radius_index] * (1.0 + prepared)
            + state.density[:, radius_index] * drds
            + self.geometry.scale * (prepared + 0.5 * drds)
        )


def test_database_initial_root_support_batches_charge_weighted_particle_bars(monkeypatch):
    """Selected-root database bars use exactly ``Z_a * residual_bar``."""

    class _ScanModel:
        def pullback_direct_rhs_support_payload(self, state, flux_bar, support):
            del state, support
            return {"database": {"gamma": 2.0 * flux_bar["Gamma"]}}

        pullback_local_particle_flux_support_payload = (
            pullback_direct_rhs_support_payload
        )

    scan_model = _ScanModel()
    monkeypatch.setattr(
        initial_er_module,
        "find_ntx_runtime_scan_model_in_model",
        lambda _model: scan_model,
    )
    runtime = SimpleNamespace(
        species=SimpleNamespace(charge_qp=jnp.asarray([-1.0, 2.0])),
        models=SimpleNamespace(flux=object()),
    )
    state = TransportState(
        density=jnp.ones((2, 3)),
        pressure=2.0 * jnp.ones((2, 3)),
        Er=jnp.asarray([0.1, 0.2, 0.3]),
    )
    residual_bars = jnp.asarray([[0.4, -0.2, 0.1], [-0.3, 0.5, 0.2]])
    actual = initial_er_module.compact_initial_er_database_support_bars(
        runtime=runtime,
        state=state,
        er_profile=state.Er,
        residual_bars=residual_bars,
        support={"database": object()},
    )
    expected = 2.0 * runtime.species.charge_qp[None, :, None] * residual_bars[:, None, :]
    assert jnp.allclose(actual["gamma"], expected, rtol=0.0, atol=0.0)


def test_database_initial_root_geometry_uses_recorded_database_payload(monkeypatch):
    """The compact root geometry hook must not request a new NTX scan."""

    seen = {}

    class _ScanModel:
        def with_support_payload(self, support):
            seen["support"] = support
            return self

        def pullback_local_particle_flux_geometry_by_radius(
            self, state, er_profile, residual_bars, geometry
        ):
            assert seen["support"]["database"] is database
            assert geometry is support_geometry
            assert state is root_state
            assert er_profile is root_state.Er
            return {"scale": 3.0 * residual_bars}

    database = object()
    support_geometry = object()
    root_state = TransportState(
        density=jnp.ones((1, 2)), pressure=jnp.ones((1, 2)), Er=jnp.asarray([0.1, 0.2])
    )
    residual_bars = jnp.asarray([[0.3, -0.2], [-0.4, 0.5]])
    scan_model = _ScanModel()
    monkeypatch.setattr(
        initial_er_module,
        "find_ntx_runtime_scan_model_in_model",
        lambda _model: scan_model,
    )
    actual = initial_er_module.compact_initial_er_database_geometry_bars(
        runtime=SimpleNamespace(models=SimpleNamespace(flux=object())),
        state=root_state,
        er_profile=root_state.Er,
        residual_bars=residual_bars,
        support={"geometry": support_geometry, "database": database},
    )
    assert seen["support"]["database"] is database
    assert jnp.allclose(actual["scale"], 3.0 * residual_bars, rtol=0.0, atol=0.0)


def test_momentum_correction_matrix_is_square_for_four_species():
    """The wHe bootstrap system has three Sonine unknowns per species."""

    n_species = 4
    geometry = _TestMomentumGeometry(
        a_b=jnp.asarray(1.0),
        r_grid=jnp.asarray([0.5]),
        r_grid_half=jnp.asarray([0.25, 0.75]),
        full_grid_indices=jnp.asarray([0]),
        Bsqav=jnp.asarray([1.2]),
        G_PS=jnp.asarray([1.0]),
        B0=jnp.asarray([1.0]),
    )
    grid = StandardLaguerreEnergyGrid(n_x=2)
    lij = jnp.eye(5, dtype=jnp.float64)
    eij = 0.1 * jnp.eye(5, dtype=jnp.float64)
    cm_ab = jnp.ones((n_species, n_species, 3, 3), dtype=jnp.float64)
    cn_ab = 0.2 * jnp.ones((n_species, n_species, 3, 3), dtype=jnp.float64)
    tau = jnp.ones((n_species, n_species), dtype=jnp.float64)
    v_thermal = jnp.ones((n_species, 1), dtype=jnp.float64)
    species_indices = jnp.arange(n_species, dtype=jnp.int32)

    rows = jax.vmap(
        lambda species_index: get_Matrix(
            grid,
            geometry,
            species_index,
            0,
            lij,
            eij,
            cm_ab,
            cn_ab,
            tau,
            v_thermal,
        )
    )(species_indices)
    matrix = jnp.reshape(rows, (n_species * 3, n_species * 3))

    assert rows.shape == (n_species, 3, n_species * 3)
    assert matrix.shape == (n_species * 3, n_species * 3)


def test_momentum_flux_wrapper_preserves_species_radius_contract(monkeypatch):
    """The corrected-flux wrapper maps species and radius independently."""

    n_species = 4
    geometry = _TestMomentumGeometry(
        a_b=jnp.asarray(1.0),
        r_grid=jnp.asarray([0.25, 0.75]),
        r_grid_half=jnp.asarray([0.0, 0.5, 1.0]),
        Bsqav=jnp.ones(2),
        G_PS=jnp.ones(2),
        B0=jnp.ones(2),
        full_grid_indices=jnp.asarray([0, 1]),
        dr=jnp.asarray(0.5),
    )
    species = Species(
        number_species=n_species,
        species_indices=jnp.arange(n_species),
        mass_mp=jnp.asarray([5.446e-4, 2.0, 3.0, 4.0]),
        charge_qp=jnp.asarray([-1.0, 1.0, 1.0, 2.0]),
        names=("e", "D", "T", "He"),
    )
    energy_grid = StandardLaguerreEnergyGrid(n_x=2)
    density = jnp.ones((n_species, 2))
    temperature = 2.0 * jnp.ones((n_species, 2))
    er = jnp.zeros(2)

    def _fake_lij(*_args):
        return jnp.zeros((5, 5)), jnp.zeros((5, 5)), jnp.zeros(3)

    def _fake_momentum(
        species_arg, energy_grid_arg, geometry_arg, _radius_index,
        lij, eij, nu_average, _v_thermal, _density, _temperature,
        a1, a2, a3, *_args,
    ):
        assert species_arg is species
        assert energy_grid_arg is energy_grid
        assert geometry_arg is geometry
        assert lij.shape == (n_species, 5, 5)
        assert eij.shape == (n_species, 5, 5)
        assert nu_average.shape == (n_species, 3)
        assert a1.shape == (n_species, 2)
        assert a2.shape == (n_species, 2)
        assert a3.shape == (2,)
        value = jnp.arange(n_species, dtype=jnp.float64)
        return value, value + 1.0, value + 2.0, value + 3.0, value + 4.0

    monkeypatch.setattr(neoclassical_module, "get_Lij_matrix_with_momentum_correction", _fake_lij)
    monkeypatch.setattr(neoclassical_module, "get_momentum_Correction", _fake_momentum)
    corrected = neoclassical_module.get_Neoclassical_Fluxes_With_Momentum_Correction.__wrapped__(
        species, energy_grid, geometry, None, er, temperature, density
    )
    for component in corrected:
        assert component.shape == (n_species, 2)
    assert jnp.allclose(corrected[2][:, 0], jnp.arange(n_species) + 2.0)


def test_four_species_momentum_assembly_keeps_full_collision_partner_axes(monkeypatch):
    """Each species block must retain all collision partners while assembling M."""

    n_species = 4
    n_sonine = 3
    species = Species(
        number_species=n_species,
        species_indices=jnp.arange(n_species),
        mass_mp=jnp.asarray([5.446e-4, 2.0, 3.0, 4.0]),
        charge_qp=jnp.asarray([-1.0, 1.0, 1.0, 2.0]),
        names=("e", "D", "T", "He"),
    )
    grid = StandardLaguerreEnergyGrid(n_x=2)
    geometry = _TestMomentumGeometry(
        a_b=jnp.asarray(1.0), r_grid=jnp.asarray([0.5]),
        r_grid_half=jnp.asarray([0.25, 0.75]), Bsqav=jnp.ones(1),
        G_PS=jnp.ones(1), B0=jnp.ones(1), full_grid_indices=jnp.asarray([0]),
        dr=jnp.asarray(0.5),
    )
    density = jnp.ones((n_species, 1))
    temperature = 2.0 * jnp.ones((n_species, 1))
    v_thermal = jnp.ones((n_species, 1))
    lij = jnp.broadcast_to(jnp.eye(5), (n_species, 5, 5))
    eij = 0.1 * lij
    nu_average = jnp.ones((n_species, 3))
    gradients = jnp.ones((n_species, 1))

    def _fake_collision(*_args):
        return jnp.zeros((n_sonine, n_sonine)), jnp.zeros((n_sonine, n_sonine)), jnp.asarray(1.0)

    def _fake_matrix(_grid, _geometry, species_index, _radius_index, _lij, _eij, cm, cn, tau, _vthermal):
        assert cm.shape == (n_species, n_species, n_sonine, n_sonine)
        assert cn.shape == (n_species, n_species, n_sonine, n_sonine)
        assert tau.shape == (n_species, n_species)
        return jax.lax.dynamic_slice(jnp.eye(n_species * n_sonine), (n_sonine * species_index, 0), (n_sonine, n_species * n_sonine))

    def _fake_fluxes(_grid, _geometry, species_index, *_args):
        value = species_index.astype(jnp.float64)
        return value, value, value, value, value

    monkeypatch.setattr(neoclassical_module, "get_Collision_Operator_terms", _fake_collision)
    monkeypatch.setattr(neoclassical_module, "get_Matrix", _fake_matrix)
    monkeypatch.setattr(neoclassical_module, "get_corrected_fluxes", _fake_fluxes)
    result = neoclassical_module.get_momentum_Correction.__wrapped__(
        species, grid, geometry, 0, lij, eij, nu_average, v_thermal,
        density, temperature, gradients, gradients, jnp.zeros(1),
        species.mass, species.charge, gradients, gradients,
    )
    assert all(component.shape == (n_species,) for component in result)


def test_four_species_momentum_blocks_use_species_equality_not_sonine_indices():
    """The fourth species must not alias the final Sonine index.

    NTSSfusion distinguishes the diagonal ``a == b`` block explicitly.  A
    3-by-3 Sonine identity cannot implement that predicate once there are four
    kinetic species.
    """

    n_species = 4
    geometry = _TestMomentumGeometry(
        a_b=jnp.asarray(1.0),
        r_grid=jnp.asarray([0.5]),
        r_grid_half=jnp.asarray([0.25, 0.75]),
        Bsqav=jnp.asarray([1.0]),
        G_PS=jnp.asarray([1.0]),
        B0=jnp.asarray([1.0]),
    )
    grid = StandardLaguerreEnergyGrid(n_x=2)
    coeff = jnp.eye(3, dtype=jnp.float64)
    nucoeff = jnp.zeros((3, 3), dtype=jnp.float64)
    cm_ab = jnp.ones((n_species, n_species, 3, 3), dtype=jnp.float64)
    cn_ab = 0.2 * jnp.ones((n_species, n_species, 3, 3), dtype=jnp.float64)
    tau = jnp.ones((n_species, n_species), dtype=jnp.float64)
    v_thermal = jnp.ones((n_species, 1), dtype=jnp.float64)
    sonine = grid.Sonine_expansion
    factor = 2.0

    # ``a=3, b=2`` is an off-diagonal species block.  The old ``I[a,b]``
    # selector was clipped by JAX to I[2,2] and incorrectly made it diagonal.
    actual = get_A_matrix(
        grid, jnp.asarray(3), jnp.asarray(2), coeff, nucoeff, cn_ab,
        jnp.zeros((3, 3)), tau, v_thermal, geometry, 0,
    )
    expected = -factor * jnp.multiply(coeff.T @ cn_ab[3, 2], sonine)
    assert jnp.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    # The matching fourth-species diagonal must still take the diagonal block
    # (the only valid use of the 3-by-3 Sonine identity in this expression).
    diagonal_sum = 0.3 * jnp.eye(3, dtype=jnp.float64)
    diagonal = get_A_matrix(
        grid, jnp.asarray(3), jnp.asarray(3), coeff, nucoeff, cn_ab,
        diagonal_sum, tau, v_thermal, geometry, 0,
    )
    expected_diagonal = jnp.eye(3) - factor * jnp.multiply(
        coeff.T @ diagonal_sum, sonine
    )
    assert jnp.allclose(diagonal, expected_diagonal, rtol=1.0e-12, atol=1.0e-12)

    correction = jnp.ones((n_species, 3), dtype=jnp.float64)
    density = jnp.ones((n_species, 1), dtype=jnp.float64)
    temperature = 2.0 * jnp.ones((n_species, 1), dtype=jnp.float64)
    dndr = jnp.zeros((n_species, 1), dtype=jnp.float64)
    dtdr = jnp.asarray([[0.1], [0.2], [0.3], [0.4]], dtype=jnp.float64)
    charge = jnp.asarray([-1.0, 1.0, 1.0, 2.0], dtype=jnp.float64)
    correction_term, add1, *_ = get_correction_matrix(
        grid, jnp.asarray(3), jnp.asarray(2), coeff, nucoeff, cm_ab, cn_ab,
        jnp.zeros((3, 3)), tau, factor, correction, 0, dndr, dtdr,
        temperature, density, charge,
    )
    expected_term = -factor * ((coeff.T @ cn_ab[3, 2]) @ sonine)
    expected_add1 = (
        dtdr[3, 0] / temperature[3, 0]
        - charge[3] / charge[2] * temperature[2, 0] / temperature[3, 0]
        * dtdr[2, 0] / temperature[2, 0]
    )
    assert jnp.allclose(correction_term, expected_term, rtol=1.0e-12, atol=1.0e-12)
    assert jnp.allclose(add1, expected_add1, rtol=1.0e-12, atol=1.0e-12)


def test_momentum_corrected_fluxes_accept_four_collision_species():
    """Each correction contribution is evaluated for one collision species.

    This is the second half of the wHe (four-species) bootstrap contract:
    the correction vector has three Sonine entries, while the outer vmap is
    over the four collision-species indices.
    """

    n_species = 4
    geometry = _TestMomentumGeometry(
        a_b=jnp.asarray(1.0),
        r_grid=jnp.asarray([0.5]),
        r_grid_half=jnp.asarray([0.25, 0.75]),
        Bsqav=jnp.asarray([1.2]),
        G_PS=jnp.asarray([1.0]),
        B0=jnp.asarray([1.0]),
    )
    grid = StandardLaguerreEnergyGrid(n_x=2)
    lij = jnp.reshape(jnp.linspace(0.1, 1.5, 15), (5, 3))
    eij = jnp.reshape(jnp.linspace(0.2, 0.8, 15), (5, 3))
    cm_ab = jnp.ones((n_species, n_species, 3, 3), dtype=jnp.float64)
    cn_ab = 0.2 * jnp.ones((n_species, n_species, 3, 3), dtype=jnp.float64)
    tau = jnp.ones((n_species, n_species), dtype=jnp.float64)
    correction = jnp.reshape(jnp.linspace(0.01, 0.12, 12), (n_species, 3))
    v_thermal = jnp.ones((n_species, 1), dtype=jnp.float64)
    density = jnp.ones((n_species, 1), dtype=jnp.float64)
    temperature = 2.0 * jnp.ones((n_species, 1), dtype=jnp.float64)
    gradients = 0.1 * jnp.ones((n_species, 1), dtype=jnp.float64)

    outputs = get_corrected_fluxes(
        grid, geometry, jnp.asarray(0, dtype=jnp.int32), 0,
        lij, eij, jnp.ones(3), cm_ab, cn_ab, tau, correction,
        v_thermal, density, temperature, gradients, gradients,
        jnp.ones(1), jnp.ones(n_species), jnp.asarray([-1.0, 1.0, 1.0, 2.0]), gradients, gradients,
    )

    assert all(bool(jnp.all(jnp.isfinite(value))) for value in outputs)


@pytest.mark.parametrize("scan_generated_monoenergetic", (False, True))
def test_database_local_bootstrap_state_pullback_matches_full_upar_jvp(
    monkeypatch, scan_generated_monoenergetic
):
    """The compact database bootstrap state rule is the full wHe JVP.

    This guards the new per-radius boundary directly.  In particular it
    covers the production axis convention (coefficient matrices at the axis
    borrow index one) and keeps the database tables out of the differentiated
    input tree.
    """
    species = Species(
        number_species=4,
        species_indices=jnp.asarray([0, 1, 2, 3]),
        mass_mp=jnp.asarray([5.446e-4, 2.0, 3.0, 4.0]),
        charge_qp=jnp.asarray([-1.0, 1.0, 1.0, 2.0]),
        names=("e", "D", "T", "He"),
    )
    geometry = _TestMomentumGeometry(
        a_b=jnp.asarray(1.0),
        r_grid=jnp.asarray([0.25, 0.75]),
        r_grid_half=jnp.asarray([0.0, 0.5, 1.0]),
        Bsqav=jnp.asarray([1.2, 1.3]),
        G_PS=jnp.asarray([1.0, 1.1]),
        B0=jnp.asarray([1.0, 1.0]),
        full_grid_indices=jnp.arange(2, dtype=jnp.int32),
        dr=jnp.asarray(0.5),
    )
    rho = jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    nu_v = jnp.asarray([1.0e-4, 1.0e-2, 1.0])
    er_grid = jnp.asarray([[0.0, 1.0e-4, 1.0e-3, 1.0e-2]])
    table_shape = (rho.size, nu_v.size, er_grid.shape[1])
    table_seed = jnp.reshape(
        jnp.arange(int(jnp.prod(jnp.asarray(table_shape))), dtype=jnp.float64),
        table_shape,
    )
    if scan_generated_monoenergetic:
        database = Monoenergetic(
            a_b=1.0,
            rho=rho,
            nu_log=jnp.log10(nu_v),
            Er_list=jnp.broadcast_to(
                jnp.asarray([-8.0, -5.0, -2.0, 1.0]), table_shape[:1] + (4,)
            ),
            D11_log=-3.0 + 1.0e-3 * table_seed,
            D13=0.2 + 1.0e-4 * table_seed,
            D33=0.3 + 2.0e-4 * table_seed,
        )
    else:
        database = PreprocessedMonoenergetic3DNTSSRadius.read_data(
            a_b=1.0,
            rho=rho,
            nu_v=nu_v,
            Er=er_grid,
            drds=jnp.ones_like(rho),
            D11=1.0 + 1.0e-3 * table_seed,
            D13=0.2 + 1.0e-4 * table_seed,
            D33=0.3 + 2.0e-4 * table_seed,
        )
    model = NTXDatabaseTransportModel(
        species=species,
        energy_grid=StandardLaguerreEnergyGrid(n_x=2),
        geometry=geometry,
        database=database,
    )
    state = TransportState(
        density=jnp.asarray(
            [[1.0, 1.1], [0.9, 1.0], [0.8, 0.9], [1.0e-4, 1.1e-4]]
        ),
        pressure=jnp.asarray(
            [[1.5, 1.76], [1.2, 1.5], [0.9, 1.17], [1.5e-4, 1.76e-4]]
        ),
        Er=jnp.asarray([2.0e-4, 2.5e-4]),
    )
    state_direction = dataclasses.replace(
        state,
        density=jnp.asarray(
            [[0.03, -0.02], [-0.01, 0.04], [0.02, -0.03], [1.0e-6, -2.0e-6]]
        ),
        pressure=jnp.asarray(
            [[0.04, -0.01], [0.02, 0.03], [-0.03, 0.01], [2.0e-6, -1.0e-6]]
        ),
        Er=jnp.asarray([2.0e-5, -1.0e-5]),
    )
    upar_bar = jnp.asarray(
        [[0.2, -0.3], [-0.4, 0.1], [0.3, 0.2], [-0.1, 0.25]]
    )
    upar_tangent = jax.jvp(
        model.evaluate_momentum_corrected_upar_only,
        (state,),
        (state_direction,),
    )[1]
    state_bar = model.pullback_momentum_corrected_upar_state_by_radius(
        state, upar_bar
    )
    lhs = sum(
        jnp.vdot(jnp.asarray(bar), jnp.asarray(direction))
        for bar, direction in zip(
            jax.tree_util.tree_leaves(state_bar),
            jax.tree_util.tree_leaves(state_direction),
            strict=True,
        )
        if (
            jnp.issubdtype(jnp.asarray(bar).dtype, jnp.inexact)
            and jnp.issubdtype(jnp.asarray(direction).dtype, jnp.inexact)
        )
    )
    assert jnp.allclose(lhs, jnp.vdot(upar_bar, upar_tangent), rtol=2.0e-10, atol=2.0e-10)

    def _geometry_direction_leaf(value):
        array = jnp.asarray(value)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            return jnp.full_like(array, 0.013)
        return jnp.zeros(array.shape, dtype=jax.dtypes.float0)

    geometry_direction = jax.tree_util.tree_map(_geometry_direction_leaf, geometry)
    geometry_upar_tangent = jax.jvp(
        lambda geometry_value: dataclasses.replace(
            model, geometry=geometry_value
        ).evaluate_momentum_corrected_upar_only(state),
        (geometry,),
        (geometry_direction,),
    )[1]
    geometry_bar = model.pullback_momentum_corrected_upar_geometry_by_radius(
        state, upar_bar, geometry
    )
    geometry_lhs = sum(
        jnp.vdot(jnp.asarray(bar), jnp.asarray(direction))
        for bar, direction in zip(
            jax.tree_util.tree_leaves(geometry_bar),
            jax.tree_util.tree_leaves(geometry_direction),
            strict=True,
        )
        if (
            jnp.issubdtype(jnp.asarray(bar).dtype, jnp.inexact)
            and jnp.issubdtype(jnp.asarray(direction).dtype, jnp.inexact)
        )
    )
    assert jnp.allclose(
        geometry_lhs,
        jnp.vdot(upar_bar, geometry_upar_tangent),
        rtol=2.0e-10,
        atol=2.0e-10,
    )

    def _upar_from_tables(d11_log, d13, d33):
        table_model = dataclasses.replace(
            model,
            database=dataclasses.replace(
                database, D11_log=d11_log, D13=d13, D33=d33
            ),
        )
        return table_model.evaluate_momentum_corrected_upar_only(state)

    _, generic_table_pullback = jax.vjp(
        _upar_from_tables, database.D11_log, database.D13, database.D33
    )
    expected_table_bars = generic_table_pullback(upar_bar)
    actual_table_bars = model.pullback_momentum_corrected_upar_database_by_radius(
        state, upar_bar
    )
    for actual, expected in zip(actual_table_bars, expected_table_bars, strict=True):
        assert jnp.allclose(actual, expected, rtol=2.0e-10, atol=2.0e-10)

    # The selected-root helper reduces exactly the local charge-flux residual,
    # rather than the transport RHS.  Test that complete database boundary
    # against its generic table VJP before the recorded scan fold.
    root_runtime = SimpleNamespace(
        species=species,
        models=SimpleNamespace(flux=model),
    )
    root_residual_bars = jnp.asarray(
        [[0.35, -0.2], [-0.15, 0.45]], dtype=jnp.float64
    )

    def _root_residuals_from_tables(d11_log, d13, d33):
        table_model = dataclasses.replace(
            model,
            database=dataclasses.replace(
                database, D11_log=d11_log, D13=d13, D33=d33
            ),
        )
        runtime = SimpleNamespace(
            species=species, models=SimpleNamespace(flux=table_model)
        )
        return initial_er_module.initial_er_charge_flux_residuals(
            state, state.Er, runtime=runtime
        )

    _, generic_root_pullback = jax.vjp(
        _root_residuals_from_tables,
        database.D11_log,
        database.D13,
        database.D33,
    )
    expected_root_table_bars = jax.vmap(generic_root_pullback)(root_residual_bars)
    monkeypatch.setattr(
        initial_er_module,
        "find_ntx_runtime_scan_model_in_model",
        lambda _model: model,
    )
    actual_root_database_bars = initial_er_module.compact_initial_er_database_support_bars(
        runtime=root_runtime,
        state=state,
        er_profile=state.Er,
        residual_bars=root_residual_bars,
        support={"database": database},
    )
    for actual, expected in zip(
        (
            actual_root_database_bars.D11_log,
            actual_root_database_bars.D13,
            actual_root_database_bars.D33,
        ),
        expected_root_table_bars,
        strict=True,
    ):
        assert jnp.allclose(actual, expected, rtol=2.0e-10, atol=2.0e-10)

    # The geometry half of the same selected-root boundary must use the
    # pointwise database primitive, not build all radial flux columns and
    # select one afterwards.  Check both its primal value and its geometry
    # transpose against the established complete centre-flux calculation.
    root_density = safe_density(state.density, model.density_floor)
    density_right, density_right_grad = _extract_right_constraints(
        model.bc_density, root_density, geometry.r_grid_half
    )
    temperature_right, temperature_right_grad = _extract_right_constraints(
        model.bc_temperature, state.temperature, geometry.r_grid_half
    )
    _, generic_gamma, _, _ = neoclassical_module.get_Neoclassical_Fluxes(
        species,
        model.energy_grid,
        geometry,
        database,
        state.Er,
        state.temperature,
        root_density,
        density_right_constraint=density_right,
        density_right_grad_constraint=density_right_grad,
        temperature_right_constraint=temperature_right,
        temperature_right_grad_constraint=temperature_right_grad,
    )
    local_gamma = model.build_local_particle_flux_evaluator(state)
    for radius_index in range(state.Er.shape[0]):
        assert jnp.allclose(
            local_gamma(jnp.asarray(radius_index), state.Er[radius_index]),
            generic_gamma[:, radius_index],
            rtol=2.0e-10,
            atol=2.0e-10,
        )

    geometry_root_tangent = jax.jvp(
        lambda geometry_value: jnp.sum(
            species.charge_qp[:, None]
            * neoclassical_module.get_Neoclassical_Fluxes(
                species,
                model.energy_grid,
                geometry_value,
                database,
                state.Er,
                state.temperature,
                root_density,
                density_right_constraint=density_right,
                density_right_grad_constraint=density_right_grad,
                temperature_right_constraint=temperature_right,
                temperature_right_grad_constraint=temperature_right_grad,
            )[1],
            axis=0,
        ),
        (geometry,),
        (geometry_direction,),
    )[1]
    geometry_root_bar = model.pullback_local_particle_flux_geometry_by_radius(
        state, state.Er, root_residual_bars, geometry
    )
    geometry_root_lhs = sum(
        jnp.vdot(jnp.asarray(bar), jnp.asarray(direction))
        for bar, direction in zip(
            jax.tree_util.tree_leaves(geometry_root_bar),
            jax.tree_util.tree_leaves(geometry_direction),
            strict=True,
        )
        if (
            jnp.issubdtype(jnp.asarray(bar).dtype, jnp.inexact)
            and jnp.issubdtype(jnp.asarray(direction).dtype, jnp.inexact)
        )
    )
    assert jnp.allclose(
        geometry_root_lhs,
        jnp.vdot(root_residual_bars, geometry_root_tangent),
        rtol=2.0e-10,
        atol=2.0e-10,
    )


def test_ntss_radial_flux_correction_terms_match_taguchi_formula():
    """Port the four additive Taguchi terms without changing Upar."""

    geometry = _TestMomentumGeometry(
        a_b=jnp.asarray(1.0),
        r_grid=jnp.asarray([0.5]),
        r_grid_half=jnp.asarray([0.25, 0.75]),
        Bsqav=jnp.asarray([1.0]),
        G_PS=jnp.asarray([1.7]),
        B0=jnp.asarray([2.3]),
    )
    temperature = jnp.asarray([[1.2], [2.4]], dtype=jnp.float64)
    mass = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    charge = jnp.asarray([-1.5, 2.0], dtype=jnp.float64)
    dtdr = jnp.asarray([[0.3], [-0.4]], dtype=jnp.float64)
    a1 = jnp.asarray([[0.8], [-0.6]], dtype=jnp.float64)
    a2 = jnp.asarray([[0.2], [0.5]], dtype=jnp.float64)
    sum_matrix = jnp.asarray(
        [[0.1, 0.4, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    add1, add2, add3, add4 = (0.7, -0.2, 0.15, -0.35)
    nu_av = jnp.asarray([0.9, 1.1, 1.3], dtype=jnp.float64)

    particle, heat = _ntss_radial_flux_correction_terms(
        geometry, 1, 0, sum_matrix, add1, add2, add3, add4, nu_av,
        a1, a2, dtdr, temperature, mass, charge,
    )
    from NEOPAX._state import JOULE_PER_KEV

    prefactor = (
        mass[1] * temperature[1, 0] * JOULE_PER_KEV * geometry.G_PS[0]
        / (charge[1] * geometry.B0[0]) ** 2
    )
    expected_particle = prefactor * (
        add1 - dtdr[1, 0] / temperature[1, 0] * sum_matrix[0, 1] - add3
        + a1[1, 0] * nu_av[0] / 1.5 + a2[1, 0] * nu_av[1] / 1.5
    )
    expected_heat = prefactor * (
        add2 - dtdr[1, 0] / temperature[1, 0]
        * (2.5 * sum_matrix[0, 1] - sum_matrix[1, 1]) - add4
        + a1[1, 0] * nu_av[1] / 1.5 + a2[1, 0] * nu_av[2] / 1.5
    )
    assert jnp.allclose(particle, expected_particle, rtol=1.0e-12, atol=1.0e-30)
    assert jnp.allclose(heat, expected_heat, rtol=1.0e-12, atol=1.0e-30)


def _small_runtime_model(n_energy=1):
    """Construct only the moment-weight state needed by the local adapter."""

    model = object.__new__(NTXExactLijRuntimeTransportModel)
    weights = jnp.linspace(0.8, 1.2, n_energy)
    object.__setattr__(
        model,
        "energy_grid",
        SimpleNamespace(
            # The production energy grid carries one normalized speed per
            # energy.  Keep that axis in the small two-energy fixture so its
            # ``lax.map`` reference path has the same contract.
            v_norm=jnp.linspace(0.9, 1.1, n_energy),
            xWeights=weights,
            L11_weight=weights,
            L12_weight=weights,
            L22_weight=weights,
            L13_weight=weights,
            L23_weight=weights,
            L33_weight=weights,
        ),
    )
    object.__setattr__(model, "derivative_mode", "direct")
    object.__setattr__(model, "scan_batch_size", None)
    object.__setattr__(model, "use_remat", False)
    return model


def _small_vmec_prepared():
    """A traceable VMEC prepared system for native-bridge contract gates."""

    surface = ntx.VmecSurface(
        path=Path("native-bridge-fixture.nc"),
        requested_psi_n=0.2,
        psi_n=0.2,
        nfp=2,
        ns=3,
        mpol=2,
        ntor=1,
        total_mode_count=2,
        loaded_mode_count=2,
        iota=jnp.asarray(0.6),
        m=jnp.asarray([0, 1]),
        n=jnp.asarray([0, 1]),
        b_cos=jnp.asarray([1.0, 0.1]),
        jacobian_cos=jnp.asarray([1.0, 0.02]),
        b_sub_theta_cos=jnp.asarray([0.2, 0.01]),
        b_sub_zeta_cos=jnp.asarray([1.1, 0.03]),
        b_sup_theta_cos=jnp.asarray([0.3, 0.04]),
        b_sup_zeta_cos=jnp.asarray([1.2, 0.05]),
        b0=jnp.asarray(1.0),
        psi_a_hat=jnp.asarray(1.0),
        phi_edge=jnp.asarray(1.0),
        r_n=jnp.asarray(0.5),
        r_hat=jnp.asarray(0.5),
        dpsi_hat_dr_hat=jnp.asarray(1.0),
        dr_hat_dpsi_hat=jnp.asarray(1.0),
        transport_psi_scale=jnp.asarray(1.0),
    )
    return surface, ntx.prepare_monoenergetic_system(surface, ntx.GridSpec(5, 5, 4))


def _assert_float_tree_allclose(actual, expected, *, rtol=1e-9, atol=1e-11):
    for leaf_index, (actual_leaf, expected_leaf) in enumerate(zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    )):
        if jnp.issubdtype(jnp.asarray(expected_leaf).dtype, jnp.inexact):
            if not jnp.allclose(actual_leaf, expected_leaf, rtol=rtol, atol=atol):
                difference = jnp.abs(actual_leaf - expected_leaf)
                scale = jnp.maximum(jnp.abs(expected_leaf), 1.0e-30)
                raise AssertionError(
                    "float tree mismatch: "
                    f"leaf={leaf_index} "
                    f"max_abs={float(jnp.max(difference)):.16e} "
                    f"max_rel={float(jnp.max(difference / scale)):.16e}"
                )


def test_joint_momentum_corrected_upar_pullback_matches_separate_helpers():
    """One local bootstrap VJP preserves the three established contracts.

    This is deliberately analytic and contains no transport rollout, VMEC
    solve, profiler, or filesystem activity.  It exercises the exact sparse
    prepared-plus-``drds`` support layout used by the production helper.
    """

    geometry = _ToyBootstrapGeometry(scale=jnp.asarray(1.7))
    support = _ToyBootstrapSupport(
        center_channels=_ToyBootstrapChannels(
            drds=jnp.asarray([0.8, 1.1]),
        ),
        face_channels=(jnp.asarray([0.2, -0.4]),),
        center_prepared=(jnp.asarray([0.3, -0.25]),),
        face_prepared=(jnp.asarray([0.1, 0.5, -0.2]),),
    )
    model = _ToyBootstrapModel(geometry=geometry, support=support)
    state = TransportState(
        density=jnp.asarray([[1.0, 1.2], [0.9, 1.1]]),
        pressure=jnp.asarray([[1.3, 1.5], [1.1, 1.4]]),
        Er=jnp.asarray([2.0e-4, 2.5e-4]),
    )
    upar_bar = jnp.asarray([[0.4, -0.3], [-0.2, 0.5]])

    state_expected = (
        NTXExactLijRuntimeTransportModel.pullback_momentum_corrected_upar_state_by_radius(
            model, state, upar_bar
        )
    )
    support_expected = (
        NTXExactLijRuntimeTransportModel.pullback_momentum_corrected_upar_support_by_radius(
            model, state, upar_bar, support
        )
    )
    geometry_expected = (
        NTXExactLijRuntimeTransportModel.pullback_momentum_corrected_upar_geometry_by_radius(
            model, state, upar_bar, geometry, support
        )
    )
    state_actual, support_actual, geometry_actual = (
        NTXExactLijRuntimeTransportModel.pullback_momentum_corrected_upar_state_support_geometry_by_radius(
            model, state, upar_bar, geometry, support
        )
    )

    _assert_float_tree_allclose(state_actual, state_expected, rtol=1e-12, atol=1e-12)
    _assert_float_tree_allclose(support_actual, support_expected, rtol=1e-12, atol=1e-12)
    _assert_float_tree_allclose(geometry_actual, geometry_expected, rtol=1e-12, atol=1e-12)


def test_native_vmec_coefficient_bridge_extracts_runtime_channels_only():
    """The diagnostic channel projection cannot retain prepared leaves."""

    support_bars = (
        SimpleNamespace(
            center_channels=(jnp.asarray([1.0]),),
            face_channels=(jnp.asarray([2.0]),),
            center_prepared="center-prepared-must-not-pass",
            face_prepared="face-prepared-must-not-pass",
        ),
        SimpleNamespace(
            center_channels=(jnp.asarray([3.0]),),
            face_channels=(jnp.asarray([4.0]),),
            center_prepared="second-center-prepared-must-not-pass",
            face_prepared="second-face-prepared-must-not-pass",
        ),
    )

    actual = _ntx_runtime_channel_payload_bars(support_bars)
    assert len(actual) == 2
    _assert_float_tree_allclose(
        actual,
        (
            (support_bars[0].center_channels, support_bars[0].face_channels),
            (support_bars[1].center_channels, support_bars[1].face_channels),
        ),
    )
    assert "prepared" not in repr(actual).lower()


def test_native_vmec_bridge_merges_rebuild_channels_and_direct_geometry():
    """Native coefficients replace only face-prepared, never direct geometry."""

    generic_ntx = _MockSupportBar(
        center_channels=(jnp.asarray([1.0, -2.0]),),
        face_channels=(jnp.asarray([3.0, 4.0]),),
        center_prepared="retain-center-prepared",
        face_prepared="retain-face-prepared",
    )
    rebuild_ntx = _MockSupportBar(
        center_channels=(jnp.asarray([0.2, 0.3]),),
        face_channels=(jnp.asarray([-0.4, 0.5]),),
        center_prepared="native-replaces-only-this-center-branch",
        face_prepared="native-replaces-only-this-face-branch",
    )
    merged = _merge_rebuild_ntx_channels_into_generic_payload_bar(
        {"geometry": {"direct": jnp.asarray([1.3, -0.4])}, "ntx_support": generic_ntx},
        {"geometry": {"direct": jnp.asarray([-0.2, 0.7])}, "ntx_support": rebuild_ntx},
    )
    _assert_float_tree_allclose(
        merged["geometry"], {"direct": jnp.asarray([1.1, 0.3])}
    )
    assert merged["ntx_support"].center_prepared == "retain-center-prepared"
    assert merged["ntx_support"].face_prepared == "retain-face-prepared"
    _assert_float_tree_allclose(
        merged["ntx_support"].center_channels,
        (jnp.asarray([1.2, -1.7]),),
    )
    _assert_float_tree_allclose(
        merged["ntx_support"].face_channels,
        (jnp.asarray([2.6, 4.5]),),
    )


def test_grouped_objective_vjp_rows_match_scalar_objective_gradients():
    """The grouped terminal-rule algebra preserves independent objective rows."""

    primal = jnp.asarray([0.4, -0.7, 1.2])

    def objective_vector(value):
        return jnp.asarray(
            [
                value[0] ** 2 + value[1],
                value[0] * value[2] - jnp.sin(value[1]),
                jnp.sum(value**3),
            ]
        )

    values, grouped_bars = _objective_vector_vjp_rows(objective_vector, primal)
    expected_bars = jnp.stack(
        [jax.grad(lambda value, row=row: objective_vector(value)[row])(primal) for row in range(3)]
    )
    assert jnp.allclose(values, objective_vector(primal), rtol=1e-12, atol=1e-12)
    assert jnp.allclose(grouped_bars, expected_bars, rtol=1e-12, atol=1e-12)


def test_take_batched_pytree_row_slices_dataclass_leaves():
    """Grouped geometry bars are pytrees, not directly subscriptable arrays."""

    rows = _MockSupportBar(
        center_channels=(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),),
        face_channels=(jnp.asarray([[5.0], [6.0]]),),
        center_prepared=jnp.asarray([[7.0, 8.0], [9.0, 10.0]]),
        face_prepared=jnp.asarray([[11.0], [12.0]]),
    )
    actual = _take_batched_pytree_row(rows, 1)
    expected = _MockSupportBar(
        center_channels=(jnp.asarray([3.0, 4.0]),),
        face_channels=(jnp.asarray([6.0]),),
        center_prepared=jnp.asarray([9.0, 10.0]),
        face_prepared=jnp.asarray([12.0]),
    )
    _assert_float_tree_allclose(actual, expected)


def test_initial_joint_lagged_pullback_retains_rhs_induced_support_bar():
    """The initial support bar must use cache plus initial-RHS lagged bars."""

    # Small exact analogue of the initial-carry graph:
    # lagged = 2 * state + 3 * support; rhs = 5 * lagged;
    # carry = (state + rhs, lagged-cache).  The carry cotangent reaches the
    # lagged response through both the cache and RHS paths.
    state = jnp.asarray([0.4, -0.7])
    support = jnp.asarray([1.1, -0.2])
    carry_y_bar = jnp.asarray([0.6, -0.3])
    cache_bar = jnp.asarray([-0.4, 0.9])

    def carry_from_state_and_support(state_value, support_value):
        lagged = 2.0 * state_value + 3.0 * support_value
        return state_value + 5.0 * lagged, lagged

    _, full_pullback = jax.vjp(carry_from_state_and_support, state, support)
    expected_state_bar, expected_support_bar = full_pullback((carry_y_bar, cache_bar))
    rhs_lagged_bar = 5.0 * carry_y_bar

    def joint_pullback(flat_y, total_lagged_bars, support_payload):
        del flat_y, support_payload
        return 2.0 * total_lagged_bars, 3.0 * total_lagged_bars

    joint_state_bar, joint_support_bar = (
        _initial_lagged_response_joint_state_and_support_pullback(
            flat_y=state,
            cache_lagged_bars=cache_bar,
            rhs_lagged_bars=rhs_lagged_bar,
            support_payload=support,
            joint_pullback=joint_pullback,
        )
    )
    # Add the direct carry-to-state contribution, as the real initial-carry
    # reverse does before its lagged-response transpose.
    joint_state_bar = joint_state_bar + carry_y_bar
    assert jnp.allclose(joint_state_bar, expected_state_bar, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(joint_support_bar, expected_support_bar, rtol=1e-12, atol=1e-12)

    # The present split support-only path would omit this real contribution.
    assert not jnp.allclose(3.0 * cache_bar, expected_support_bar)


def test_initial_rebuild_dispatch_uses_active_hook_and_keeps_vmec_channel_separate():
    """The initial-edge selector mirrors the separate rebuild hook contract."""

    calls = []

    def native_hook(flat_y, lagged_bars, support):
        calls.append((flat_y, lagged_bars, support))
        return {"support": lagged_bars + support}, {"b_cos": 2.0 * lagged_bars}

    context = SimpleNamespace(
        reverse_rebuild_support_pullback_mode=(
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_"
            "shared_primal_with_vmec_coefficients"
        ),
        flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients=native_hook,
    )
    support_bars, native_bars = _initial_cache_support_pullback_from_rebuild_dispatch(
        physics_context=context,
        flat_y=jnp.asarray([1.0, 2.0]),
        lagged_response_bars=jnp.asarray([0.3, -0.4]),
        support_payload=jnp.asarray([0.5, 0.7]),
    )
    assert len(calls) == 1
    assert jnp.allclose(support_bars["support"], jnp.asarray([0.8, 0.3]))
    assert jnp.allclose(native_bars["b_cos"], jnp.asarray([0.6, -0.8]))


def test_initial_rebuild_dispatch_selects_direct_native_vmec_hook():
    """The direct product-rule selector cannot fall back to the old hook."""

    calls = []

    def direct_hook(flat_y, lagged_bars, support):
        calls.append((flat_y, lagged_bars, support))
        return {"support": lagged_bars + support}, {"b_cos": 3.0 * lagged_bars}

    context = SimpleNamespace(
        reverse_rebuild_support_pullback_mode=(
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_"
            "shared_primal_with_vmec_coefficients_direct_directional_product_rule"
        ),
        flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule=direct_hook,
    )
    support_bars, native_bars = _initial_cache_support_pullback_from_rebuild_dispatch(
        physics_context=context,
        flat_y=jnp.asarray([1.0, 2.0]),
        lagged_response_bars=jnp.asarray([0.3, -0.4]),
        support_payload=jnp.asarray([0.5, 0.7]),
    )
    assert len(calls) == 1
    assert jnp.allclose(support_bars["support"], jnp.asarray([0.8, 0.3]))
    assert jnp.allclose(native_bars["b_cos"], jnp.asarray([0.9, -1.2]))


def test_initial_rebuild_dispatch_selects_direct_coefficient_native_vmec_hook():
    """The coefficient-transpose selector cannot fall back after setup."""

    calls = []

    def direct_hook(flat_y, lagged_bars, support):
        calls.append((flat_y, lagged_bars, support))
        return {"support": lagged_bars + support}, {"b_cos": 4.0 * lagged_bars}

    context = SimpleNamespace(
        reverse_rebuild_support_pullback_mode=(
            "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_"
            "shared_primal_with_vmec_coefficients_direct_coefficient_pullback"
        ),
        flat_rhs_build_support_pullback_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_coefficient_pullback=direct_hook,
    )
    support_bars, native_bars = _initial_cache_support_pullback_from_rebuild_dispatch(
        physics_context=context,
        flat_y=jnp.asarray([1.0, 2.0]),
        lagged_response_bars=jnp.asarray([0.3, -0.4]),
        support_payload=jnp.asarray([0.5, 0.7]),
    )
    assert len(calls) == 1
    assert jnp.allclose(support_bars["support"], jnp.asarray([0.8, 0.3]))
    assert jnp.allclose(native_bars["b_cos"], jnp.asarray([1.2, -1.6]))


def test_initial_rebuild_dispatch_matches_ordinary_batched_support_contract():
    """Non-native initial dispatch is exactly its active rebuild hook."""

    def ordinary_hook(flat_y, lagged_bars, support):
        return {
            "state_marker": flat_y * 0.0,
            "support": 1.7 * lagged_bars - 0.25 * support,
        }

    context = SimpleNamespace(
        reverse_rebuild_support_pullback_mode="ntx_batched_interpolated_faces",
        flat_rhs_build_support_pullback_batched_interpolated_faces=ordinary_hook,
    )
    flat_y = jnp.asarray([0.4, -0.8])
    lagged_bars = jnp.asarray([[0.3, -0.4], [0.2, 0.6]])
    support = jnp.asarray([1.1, -0.7])
    expected = ordinary_hook(flat_y, lagged_bars, support)
    actual, native_bars = _initial_cache_support_pullback_from_rebuild_dispatch(
        physics_context=context,
        flat_y=flat_y,
        lagged_response_bars=lagged_bars,
        support_payload=support,
    )
    _assert_float_tree_allclose(actual, expected)
    assert native_bars is None


def test_native_vmec_coefficient_tangent_contraction_matches_vjp_duality():
    """The compact tangent term equals the corresponding coefficient VJP dot."""

    coefficient_bars = (
        jnp.asarray([[0.2, -0.4], [0.7, 0.1], [-0.3, 0.5]]),
        jnp.asarray([[[0.1, 0.2], [-0.5, 0.3]], [[0.4, -0.2], [0.6, 0.1]], [[-0.3, 0.7], [0.2, -0.6]]]),
        jnp.asarray([0.6, -0.2, 0.9]),
    )
    coefficient_tangents = (
        jnp.asarray([1.5, -0.8]),
        jnp.asarray([[0.3, -0.1], [0.9, 0.2]]),
        jnp.asarray(1.7),
    )
    actual = _native_vmec_coefficient_tangent_contraction(
        coefficient_bars, coefficient_tangents
    )
    expected = jnp.asarray(
        [
            jnp.vdot(coefficient_bars[0][row], coefficient_tangents[0])
            + jnp.vdot(coefficient_bars[1][row], coefficient_tangents[1])
            + jnp.vdot(coefficient_bars[2][row], coefficient_tangents[2])
            for row in range(3)
        ]
    )
    assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_native_vmec_bridge_forwards_bridge_only_flag_to_local_ntx_helper():
    """Catch a missing bridge-only keyword before a compiled Radau segment."""

    captured = {}

    class _Model:
        def _pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs(
            self, prepared, **kwargs
        ):
            captured["prepared"] = prepared
            captured.update(kwargs)
            return "native-result"

    method = (
        NTXExactLijRuntimeTransportModel.
        _pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_reuse_moment_drds_jvp_with_vmec_coefficients
    )
    # This is the next hop used by the selected direct-directional rebuild
    # mode.  Keep the keyword at this boundary: otherwise the failure occurs
    # only after the expensive Radau reverse-segment trace has started.
    batched_method = (
        NTXExactLijRuntimeTransportModel.
        pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal
    )
    assert "native_vmec_coefficient_bars_only" in inspect.signature(method).parameters
    assert "native_vmec_direct_directional_product_rule" in inspect.signature(method).parameters
    assert (
        "native_vmec_direct_directional_product_rule"
        in inspect.signature(batched_method).parameters
    )
    result = method(
        _Model(),
        "prepared",
        drds_value="drds",
        reference_nu_hat="nu",
        reference_epsi_hat="epsi",
        vth_a="vth",
        field_bars="bars",
        return_case_bars=True,
        native_vmec_coefficient_bars_only=True,
        native_vmec_direct_directional_product_rule=True,
    )
    assert result == "native-result"
    assert captured["return_native_vmec_coefficient_bars"] is True
    assert captured["native_vmec_coefficient_bars_only"] is True
    assert captured["native_vmec_direct_directional_product_rule"] is True
    assert captured["return_case_bars"] is True


def test_native_multi_rhs_composite_forwarding_hook_is_exposed_to_radau():
    """The vector-field owner must expose the native inner NTX hook.

    Radau owns a :class:`CombinedTransportFluxModel` method, rather than the
    exact-NTX component method directly.  This pure mock prevents a missing
    outer forwarding wrapper from becoming a late reverse-segment failure.
    """

    calls = []

    class _InnerNTX:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_shared_primal(
            self,
            state,
            response_bars,
            support,
        ):
            calls.append((state, response_bars, support))
            return "native-result"

    composite = object.__new__(CombinedTransportFluxModel)
    object.__setattr__(composite, "neoclassical_model", _InnerNTX())
    response = SimpleNamespace(neoclassical_response="ntx-bars")

    assert composite.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_shared_primal(
        "state",
        response,
        "support",
        ignored_outer_keyword=True,
    ) == "native-result"
    assert calls == [("state", "ntx-bars", "support")]


def test_native_multi_rhs_reused_drds_composite_forwarding_hook_is_exposed_to_radau():
    """The new selector reaches only its dedicated inner hook."""

    calls = []

    class _InnerNTX:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal(
            self, state, response_bars, support,
        ):
            calls.append((state, response_bars, support))
            return "reused-drds-result"

    composite = object.__new__(CombinedTransportFluxModel)
    object.__setattr__(composite, "neoclassical_model", _InnerNTX())
    response = SimpleNamespace(neoclassical_response="ntx-bars")
    assert composite.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal(
        "state", response, "support", ignored_outer_keyword=True,
    ) == "reused-drds-result"
    assert calls == [("state", "ntx-bars", "support")]


def test_black_box_direct_support_replacement_dispatches_only_payload_owner():
    """Direct black-box support preserves unrelated composite submodels.

    This is deliberately a dataclass-only gate: it validates the capability
    dispatch without constructing a transport rollout, VMEC surface, or NTX
    solve.
    """

    @dataclasses.dataclass(frozen=True)
    class _PayloadOwner:
        payload: object = None

        def with_support_payload(self, payload):
            return dataclasses.replace(self, payload=payload)

    @dataclasses.dataclass(frozen=True)
    class _Passive:
        label: str

    @dataclasses.dataclass(frozen=True)
    class _Composite:
        neoclassical_model: object
        turbulent_model: object
        classical_model: object

    original = _Composite(_PayloadOwner(), _Passive("turb"), _Passive("classical"))
    scan_payload = {"geometry": "g", "channels": "c", "surfaces": "s"}
    scan_replaced = ComposedEquationSystem._flux_model_with_realtime_support_payload(
        original, scan_payload
    )
    assert scan_replaced.neoclassical_model.payload is scan_payload
    assert scan_replaced.turbulent_model is original.turbulent_model
    assert scan_replaced.classical_model is original.classical_model

    exact_payload = {"geometry": "g", "ntx_support": "prepared-support"}
    exact_replaced = ComposedEquationSystem._flux_model_with_realtime_support_payload(
        original, exact_payload
    )
    assert exact_replaced.neoclassical_model.payload == "prepared-support"


def test_black_box_direct_stage_support_uses_direct_rhs_hook_without_lagged_cache():
    """A one-stage mock proves black-box support is no longer silently zero."""

    calls = []

    def _direct_hook(t_value, flat_y, rhs_bar, support):
        calls.append((t_value, flat_y, rhs_bar, support))
        return jnp.sum(rhs_bar) * support

    kernel = SimpleNamespace(
        num_stages=1,
        state_dim=1,
        dtype=jnp.float64,
        c=jnp.asarray([0.0]),
        a=jnp.asarray([[0.0]]),
    )
    physics = SimpleNamespace(flat_rhs_direct_support_pullback=_direct_hook)
    carry = SimpleNamespace(t=jnp.asarray(2.0), y=jnp.asarray([7.0]))
    primal = SimpleNamespace(
        stage_history=jnp.asarray([0.0]),
        trial_dt=jnp.asarray(3.0),
    )
    result = _radau_exact_stage_residual_support_pullback(
        kernel,
        physics,
        carry,
        primal,
        None,
        jnp.asarray([5.0]),
        jnp.asarray(4.0),
    )
    assert jnp.allclose(result, -20.0)
    assert len(calls) == 1


def test_black_box_initial_direct_rhs_support_contracts_each_objective():
    """Carry-zero support uses the same stage-bar contraction as its VJP."""

    carry0 = SimpleNamespace(t=jnp.asarray(1.5), y=jnp.asarray([2.0, 3.0]))
    carry0_bars = SimpleNamespace(
        prev_stages=jnp.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[-2.0, 1.0], [5.0, -3.0]],
            ]
        )
    )
    result = _initial_direct_rhs_support_pullback_batched(
        carry0=carry0,
        carry0_bars=carry0_bars,
        kernel_context=SimpleNamespace(num_stages=2),
        flat_rhs_direct_support_pullback=lambda _t, _y, rhs_bar, support: support * jnp.sum(rhs_bar),
        support_payload=jnp.asarray(0.5),
    )
    assert jnp.allclose(result, jnp.asarray([5.0, 0.5]))


def test_black_box_database_initial_direct_rhs_support_uses_objective_batch():
    """Recorded database payloads keep the initial direct bar numerical."""

    carry0 = SimpleNamespace(t=jnp.asarray(1.5), y=jnp.asarray([2.0, 3.0]))
    carry0_bars = SimpleNamespace(
        prev_stages=jnp.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[-2.0, 1.0], [5.0, -3.0]],
            ]
        )
    )
    calls = []

    def _direct_hook(_t, _y, rhs_bar, support):
        calls.append(jnp.asarray(rhs_bar).shape)
        return {"database": support["database"] * jnp.sum(rhs_bar)}

    result = _initial_direct_rhs_support_pullback_batched(
        carry0=carry0,
        carry0_bars=carry0_bars,
        kernel_context=SimpleNamespace(num_stages=2),
        flat_rhs_direct_support_pullback=_direct_hook,
        support_payload={"database": jnp.asarray(0.5)},
    )
    assert calls == [(2,)]
    assert jnp.allclose(result["database"], jnp.asarray([5.0, 0.5]))


def test_black_box_exact_direct_support_split_matches_generic_payload_vjp():
    """The exact split is a partition of the generic support VJP.

    The geometry leg is deliberately evaluated with ``ntx_support`` held
    fixed.  This small analytic owner proves both the equality and the
    no-double-count contract without a transport rollout or an NTX solve.
    """

    class _ExactFluxOwner:
        def __call__(self, _state):
            return jnp.asarray(3.0)

        def pullback_direct_rhs_support_payload(self, _state, flux_bar, _support):
            # Exact NTX support contribution only: 2 * ntx_support.
            return 2.0 * flux_bar

    equations = object.__new__(ComposedEquationSystem)
    object.__setattr__(equations, "shared_flux_model", _ExactFluxOwner())
    object.__setattr__(equations, "_prepare_working_state", lambda state: (state, None))
    object.__setattr__(equations, "pullback_shared_fluxes", lambda _state, _fluxes, rhs_bar: rhs_bar)
    seen_payloads = []

    def _rebuild(payload):
        seen_payloads.append(payload)
        # This represents the direct equation geometry term plus the exact
        # NTX response supplied separately by the owner above.
        return lambda _t, _state, _runtime: payload["geometry"] + 2.0 * payload["ntx_support"]

    object.__setattr__(equations, "with_realtime_geometry_support_payload", _rebuild)
    support = {"geometry": jnp.asarray(5.0), "ntx_support": jnp.asarray(7.0)}
    actual = equations.pullback_direct_rhs_support_payload(
        jnp.asarray(0.0), None, None, jnp.asarray(11.0), support
    )
    _, generic_pullback = jax.vjp(
        lambda geometry, ntx_support: geometry + 2.0 * ntx_support,
        support["geometry"],
        support["ntx_support"],
    )
    expected_geometry, expected_ntx_support = generic_pullback(jnp.asarray(11.0))
    assert jnp.allclose(actual["geometry"], expected_geometry)
    assert jnp.allclose(actual["ntx_support"], expected_ntx_support)
    assert all(
        jnp.array_equal(payload["ntx_support"], support["ntx_support"])
        for payload in seen_payloads
    )


def test_black_box_recorded_database_direct_support_split_matches_generic_payload_vjp():
    """Recorded scan databases take the same split geometry/direct path."""

    class _RecordedDatabaseOwner:
        def __call__(self, _state):
            return jnp.asarray(3.0)

        def pullback_direct_rhs_support_payload(self, _state, flux_bar, support):
            del support
            return {
                "database": 2.0 * flux_bar,
            }

    equations = object.__new__(ComposedEquationSystem)
    object.__setattr__(equations, "shared_flux_model", _RecordedDatabaseOwner())
    object.__setattr__(equations, "_prepare_working_state", lambda state: (state, None))
    object.__setattr__(equations, "pullback_shared_fluxes", lambda _state, _fluxes, rhs_bar: rhs_bar)
    object.__setattr__(
        equations,
        "with_realtime_geometry_support_payload",
        lambda payload: lambda _t, _state, _runtime: payload["geometry"] + 2.0 * payload["database"],
    )
    support = {
        "geometry": jnp.asarray(5.0),
        "channels": jnp.asarray(7.0),
        "surfaces": jnp.asarray(11.0),
        "database": jnp.asarray(13.0),
    }
    actual = equations.pullback_direct_rhs_support_payload(
        jnp.asarray(0.0), None, None, jnp.asarray(17.0), support
    )
    assert jnp.allclose(actual["geometry"], 17.0)
    assert jnp.allclose(actual["database"], 34.0)
    assert jnp.allclose(actual["channels"], 0.0)
    assert jnp.allclose(actual["surfaces"], 0.0)


def test_native_multi_rhs_equation_system_forwarding_hook_is_exposed_to_radau():
    """The real Radau vector-field owner forwards the native hook once."""

    calls = []

    class _SharedFlux:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_shared_primal(
            self,
            state,
            response_bars,
            support,
        ):
            calls.append((state, response_bars, support))
            return "native-result"

    equations = object.__new__(ComposedEquationSystem)
    object.__setattr__(equations, "shared_flux_model", _SharedFlux())
    object.__setattr__(
        equations,
        "_split_realtime_geometry_payload",
        lambda support: (support, None),
    )
    object.__setattr__(equations, "_prepare_working_state", lambda state: (state, None))
    object.__setattr__(equations, "_shared_flux_call_kwargs", lambda kwargs: {})
    response = SimpleNamespace(flux_response="flux-bars")

    assert equations.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_shared_primal(
        "state",
        response,
        "support",
        ignored_outer_keyword=True,
    ) == "native-result"
    assert calls == [("state", "flux-bars", "support")]


def test_native_multi_rhs_reused_drds_equation_system_forwarding_hook_is_exposed_to_radau():
    """The Radau vector-field owner reaches the dedicated reuse hook."""

    calls = []

    class _SharedFlux:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal(
            self, state, response_bars, support,
        ):
            calls.append((state, response_bars, support))
            return "reused-drds-result"

    equations = object.__new__(ComposedEquationSystem)
    object.__setattr__(equations, "shared_flux_model", _SharedFlux())
    object.__setattr__(equations, "_split_realtime_geometry_payload", lambda support: (support, None))
    object.__setattr__(equations, "_prepare_working_state", lambda state: (state, None))
    object.__setattr__(equations, "_shared_flux_call_kwargs", lambda kwargs: {})
    response = SimpleNamespace(flux_response="flux-bars")
    assert equations.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal(
        "state", response, "support", ignored_outer_keyword=True,
    ) == "reused-drds-result"
    assert calls == [("state", "flux-bars", "support")]


def test_direct_directional_vmec_composite_and_equation_hooks_are_exposed_to_radau():
    """Prevent an inner-only direct selector from failing after setup."""

    calls = []

    class _InnerNTX:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule(
            self, state, response_bars, support,
        ):
            calls.append((state, response_bars, support))
            return "direct-result"

    composite = object.__new__(CombinedTransportFluxModel)
    object.__setattr__(composite, "neoclassical_model", _InnerNTX())
    combined_response = SimpleNamespace(neoclassical_response="ntx-bars")
    assert composite.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule(
        "state", combined_response, "support", ignored_outer_keyword=True,
    ) == "direct-result"

    equations = object.__new__(ComposedEquationSystem)
    object.__setattr__(equations, "shared_flux_model", composite)
    object.__setattr__(equations, "_split_realtime_geometry_payload", lambda support: (support, None))
    object.__setattr__(equations, "_prepare_working_state", lambda state: (state, None))
    object.__setattr__(equations, "_shared_flux_call_kwargs", lambda kwargs: {})
    equation_response = SimpleNamespace(flux_response=combined_response)
    assert equations.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule(
        "state", equation_response, "support", ignored_outer_keyword=True,
    ) == "direct-result"
    assert calls == [
        ("state", "ntx-bars", "support"),
        ("state", "ntx-bars", "support"),
    ]


def test_per_energy_call_boundary_vmec_composite_and_equation_hooks_are_exposed_to_radau():
    """The local HLO-boundary selector must not silently fall back."""

    calls = []

    class _InnerNTX:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule_per_energy_call_boundary(
            self, state, response_bars, support,
        ):
            calls.append((state, response_bars, support))
            return "per-energy-boundary-result"

    composite = object.__new__(CombinedTransportFluxModel)
    object.__setattr__(composite, "neoclassical_model", _InnerNTX())
    combined_response = SimpleNamespace(neoclassical_response="ntx-bars")
    assert composite.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule_per_energy_call_boundary(
        "state", combined_response, "support", ignored_outer_keyword=True,
    ) == "per-energy-boundary-result"

    equations = object.__new__(ComposedEquationSystem)
    object.__setattr__(equations, "shared_flux_model", composite)
    object.__setattr__(equations, "_split_realtime_geometry_payload", lambda support: (support, None))
    object.__setattr__(equations, "_prepare_working_state", lambda state: (state, None))
    object.__setattr__(equations, "_shared_flux_call_kwargs", lambda kwargs: {})
    equation_response = SimpleNamespace(flux_response=combined_response)
    assert equations.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule_per_energy_call_boundary(
        "state", equation_response, "support", ignored_outer_keyword=True,
    ) == "per-energy-boundary-result"
    assert calls == [
        ("state", "ntx-bars", "support"),
        ("state", "ntx-bars", "support"),
    ]


def test_direct_coefficient_vmec_composite_and_equation_hooks_are_exposed_to_radau():
    """The coefficient-transpose selector is wired through both wrappers."""

    calls = []

    class _InnerNTX:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_coefficient_pullback(
            self, state, response_bars, support,
        ):
            calls.append((state, response_bars, support))
            return "direct-coefficient-result"

    composite = object.__new__(CombinedTransportFluxModel)
    object.__setattr__(composite, "neoclassical_model", _InnerNTX())
    combined_response = SimpleNamespace(neoclassical_response="ntx-bars")
    assert composite.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_coefficient_pullback(
        "state", combined_response, "support", ignored_outer_keyword=True,
    ) == "direct-coefficient-result"

    equations = object.__new__(ComposedEquationSystem)
    object.__setattr__(equations, "shared_flux_model", composite)
    object.__setattr__(equations, "_split_realtime_geometry_payload", lambda support: (support, None))
    object.__setattr__(equations, "_prepare_working_state", lambda state: (state, None))
    object.__setattr__(equations, "_shared_flux_call_kwargs", lambda kwargs: {})
    equation_response = SimpleNamespace(flux_response=combined_response)
    assert equations.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_coefficient_pullback(
        "state", equation_response, "support", ignored_outer_keyword=True,
    ) == "direct-coefficient-result"
    assert calls == [
        ("state", "ntx-bars", "support"),
        ("state", "ntx-bars", "support"),
    ]


def test_mock_native_per_energy_call_boundary_is_a_distinct_hlo_call():
    """A local boundary stays inside the mapped body without changing values.

    This is deliberately a pure array/HLO-structure gate, not a transport or
    NTX production run.  The extra call in the bounded lowering is the
    property needed before applying the same boundary around the expensive
    native per-energy support operation.
    """

    def _local_energy_work(value):
        return (jnp.sin(value) @ value + jnp.cos(value),)

    plain = jax.jit(lambda values: jax.lax.map(_local_energy_work, values))
    bounded_work = jax.jit(_local_energy_work, inline=False)
    bounded = jax.jit(lambda values: jax.lax.map(bounded_work, values))
    values = jnp.ones((3, 4, 4), dtype=jnp.float64)
    assert jnp.allclose(bounded(values)[0], plain(values)[0])
    plain_hlo = plain.lower(values).compiler_ir(dialect="hlo").as_hlo_text()
    bounded_hlo = bounded.lower(values).compiler_ir(dialect="hlo").as_hlo_text()
    assert bounded_hlo.count("call(") == plain_hlo.count("call(") + 1


def test_realized_reverse_slot_branches_dispatch_only_active_slots_in_reverse_order():
    """Host dispatch reads only the fixed schedule, never objective data."""

    assert _realized_reverse_slot_branches(
        [True, True, False, True],
        [True, False, True, True],
        False,
    ) == (
        (3, "reuse"),
        (1, "reuse"),
        (0, "rebuild"),
    )


def test_realized_reverse_slot_dispatch_preserves_branch_and_device_value_order():
    """A no-transport oracle for the static host-dispatch seam."""

    calls = []

    def _step(slot_index, branch, carry, record, reduced):
        calls.append((slot_index, branch, int(carry), int(record), int(reduced)))
        increment = jnp.asarray(slot_index + (10 if branch == "rebuild" else 1))
        return reduced + increment, (increment,)

    reduced, support = _run_realized_reverse_slot_dispatch(
        slot_active=[True, True, False, True],
        slot_next_lagged_valid=[True, False, True, True],
        segment_start_lagged_valid=False,
        step_start_carries=jnp.asarray([10, 20, 30, 40]),
        step_primal_records=jnp.asarray([1, 2, 3, 4]),
        next_reduced_bars=jnp.asarray(5),
        initial_support_bars=(jnp.asarray(0),),
        take_axis0=lambda values, index: values[index],
        step_fn=_step,
    )
    assert calls == [
        (3, "reuse", 40, 4, 5),
        (1, "reuse", 20, 2, 9),
        (0, "rebuild", 10, 1, 11),
    ]
    assert int(reduced) == 21
    assert int(support[0]) == 16


def test_native_split_joint_no_prepared_carry_hook_selects_only_its_wrapper():
    """The compact initial mode cannot silently select the older joint hook."""

    calls = []

    class _Owner:
        def vector_field(self):
            return None

        def pullback_build_lagged_response_state_and_ntx_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_no_prepared_carry(
            self, state, bars, support, **kwargs,
        ):
            calls.append((state, bars, support, kwargs))
            return "compact-result"

    owner = _Owner()
    hook = _lagged_response_build_state_and_support_pullback_batched_interpolated_faces_hook(
        owner.vector_field,
        native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_no_prepared_carry=True,
    )
    assert callable(hook)
    assert hook("state", "bars", "support") == "compact-result"
    assert calls == [("state", "bars", "support", {})]


def test_fused_fixed_lagged_state_response_factory_calls_one_owner_hook():
    """The initial-only selector cannot silently revert to two RHS hooks."""

    calls = []

    class _Owner:
        def vector_field(self):
            return None

        def pullback_evaluate_with_lagged_response_state_and_response(
            self, t_value, state, *args, lagged_response, rhs_bar, **kwargs
        ):
            calls.append((t_value, state, lagged_response, rhs_bar, args, kwargs))
            return state * 2.0, {"lagged": rhs_bar * -3.0}

    owner = _Owner()
    pullback = _flat_rhs_state_and_lagged_response_pullback_factory(
        unravel=lambda value: value,
        pack_flat=lambda value: value,
        vector_field=owner.vector_field,
        args=(),
        kwargs={},
    )
    state_bar, lagged_bar = pullback(
        jnp.asarray(0.25),
        jnp.asarray([1.0, -2.0]),
        {"lagged": jnp.asarray([4.0, 5.0])},
        jnp.asarray([0.3, -0.7]),
    )
    assert jnp.allclose(state_bar, jnp.asarray([2.0, -4.0]))
    assert jnp.allclose(lagged_bar["lagged"], jnp.asarray([-0.9, 2.1]))
    assert len(calls) == 1


def test_native_multi_rhs_compact_residual_equation_system_forwarding_hook_is_exposed_to_radau():
    """The split-residual selector must be exposed by the vector-field owner."""

    calls = []

    class _SharedFlux:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal(
            self, state, response_bars, support,
        ):
            calls.append((state, response_bars, support))
            return "compact-residual-result"

    equations = object.__new__(ComposedEquationSystem)
    object.__setattr__(equations, "shared_flux_model", _SharedFlux())
    object.__setattr__(equations, "_split_realtime_geometry_payload", lambda support: (support, None))
    object.__setattr__(equations, "_prepare_working_state", lambda state: (state, None))
    object.__setattr__(equations, "_shared_flux_call_kwargs", lambda kwargs: {})
    response = SimpleNamespace(flux_response="flux-bars")
    assert equations.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal(
        "state", response, "support", ignored_outer_keyword=True,
    ) == "compact-residual-result"
    assert calls == [("state", "flux-bars", "support")]


def test_native_multi_rhs_compact_residual_factory_finds_vector_field_owner_hook():
    """Catch a missing owner wrapper before a compiled reverse segment starts."""

    calls = []

    class _SharedFlux:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal(
            self, state, response_bars, support,
        ):
            calls.append((state, response_bars, support))
            return "compact-residual-result"

    equations = object.__new__(ComposedEquationSystem)
    object.__setattr__(equations, "shared_flux_model", _SharedFlux())
    object.__setattr__(equations, "_split_realtime_geometry_payload", lambda support: (support, None))
    object.__setattr__(equations, "_prepare_working_state", lambda state: (state, None))
    object.__setattr__(equations, "_shared_flux_call_kwargs", lambda kwargs: {})
    pullback = _flat_rhs_build_support_pullback_batched_interpolated_faces_factory(
        unravel=lambda value: value,
        vector_field=equations.vector_field,
        args=(),
        kwargs={},
        native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal=True,
    )
    assert callable(pullback)
    assert pullback("state", SimpleNamespace(flux_response="flux-bars"), "support") == "compact-residual-result"
    assert calls == [("state", "flux-bars", "support")]


def test_native_multi_rhs_compact_residual_physical_wrapper_forwards_its_flag():
    """The outer physical hook must reach the compact-residual local adapter."""

    calls = []

    class _PhysicalLike:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal(
            self, state, response_bars, support, **kwargs,
        ):
            calls.append((state, response_bars, support, kwargs))
            return "compact-residual-result"

    method = NTXExactLijRuntimeTransportModel.pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal
    assert "native_compact_residual_ntx_rhs" in inspect.signature(method).parameters
    wrapper = NTXExactLijRuntimeTransportModel.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_compact_residual_reuse_moment_drds_jvp_shared_primal
    assert wrapper(_PhysicalLike(), "state", "response-bars", "support") == "compact-residual-result"
    assert calls == [(
        "state",
        "response-bars",
        "support",
        {
            "native_factorized_ntx_rhs": True,
            "native_compact_residual_ntx_rhs": True,
            "reuse_joint_moment_drds_jvp": True,
        },
    )]


def test_geometry_implicit_local_support_pullback_matches_prepared_support_path():
    """The new representation changes no active support or response value."""

    model = _small_runtime_model()
    prepared = ntx.prepare_monoenergetic_system(
        ntx.example_surface(),
        ntx.GridSpec(5, 5, 4),
    )
    args = dict(
        drds_value=jnp.asarray(1.2),
        reference_nu_hat=jnp.asarray([1.0e-2]),
        reference_epsi_hat=jnp.asarray([1.0e-3]),
        vth_a=jnp.asarray([1.1]),
        field_bars=(
            jnp.asarray(0.0),
            jnp.asarray([0.3, -0.2, 0.1, 0.4, -0.1, 0.2]),
            jnp.asarray([-0.3, 0.1, 0.2, -0.2, 0.3, -0.1]),
            jnp.asarray([0.2, 0.4, -0.3, 0.1, 0.2, -0.4]),
        ),
    )

    prepared_result = model._pullback_interpolated_moment_prepared_support_and_drds_only(
        prepared,
        **args,
    )
    geometry_result = model._pullback_interpolated_moment_prepared_support_and_drds_only(
        prepared,
        **args,
        geometry_implicit_ntx_two_directional=True,
    )

    _assert_float_tree_allclose(
        geometry_result[0].geometry,
        prepared_result[0].geometry,
    )
    _assert_float_tree_allclose(geometry_result[1], prepared_result[1])
    _assert_float_tree_allclose(geometry_result[2], prepared_result[2])

    fixed_leaf_pairs = zip(
        jax.tree_util.tree_leaves(geometry_result[0]),
        jax.tree_util.tree_leaves(prepared),
        strict=True,
    )
    for bar_leaf, primal_leaf in fixed_leaf_pairs:
        if jnp.issubdtype(jnp.asarray(primal_leaf).dtype, jnp.inexact):
            assert jnp.all(jnp.isfinite(jnp.asarray(bar_leaf)))


def test_geometry_implicit_local_support_pullback_keeps_objective_axis_device_batchable():
    """The adapter has no scalar objective loop or static-leaf batching error."""

    model = _small_runtime_model()
    prepared = ntx.prepare_monoenergetic_system(
        ntx.example_surface(),
        ntx.GridSpec(5, 5, 4),
    )
    base_field_bars = (
        jnp.asarray(0.0),
        jnp.asarray([0.3, -0.2, 0.1, 0.4, -0.1, 0.2]),
        jnp.asarray([-0.3, 0.1, 0.2, -0.2, 0.3, -0.1]),
        jnp.asarray([0.2, 0.4, -0.3, 0.1, 0.2, -0.4]),
    )

    def _one_objective(scale):
        result = model._pullback_interpolated_moment_prepared_support_and_drds_only(
            prepared,
            drds_value=jnp.asarray(1.2),
            reference_nu_hat=jnp.asarray([1.0e-2]),
            reference_epsi_hat=jnp.asarray([1.0e-3]),
            vth_a=jnp.asarray([1.1]),
            field_bars=tuple(scale * value for value in base_field_bars),
            geometry_implicit_ntx_two_directional=True,
        )
        return (
            tuple(
                jax.tree_util.tree_leaves(
                    _sanitize_float_delta_bar_tree(prepared, result[0])
                )
            ),
            result[1],
            result[2],
        )

    batched_bars = jax.vmap(_one_objective)(jnp.asarray([0.5, 1.5]))
    for leaf in jax.tree_util.tree_leaves(batched_bars):
        assert leaf.shape[0] == 2


def test_mock_multi_rhs_anchor_adapter_uses_one_local_primal_context():
    """The planned adapter keeps primal state local and batches only RHS work.

    This is intentionally a pure-array mock: it models the ownership and axis
    contract of the future NEOPAX/NTX anchor adapter without constructing an
    NTX system or invoking any solver.
    """

    call_count = {"local_primal": 0}

    def _build_local_primal_context(anchor_input):
        call_count["local_primal"] += 1
        return (
            2.0 * anchor_input,
            anchor_input + 3.0,
            anchor_input - 5.0,
        )

    def _support_adjoint_from_context(context, rhs_bar):
        base, d_er, d_log_nu = context
        return {
            "prepared": rhs_bar * (base + d_er),
            "drds": rhs_bar * (base - d_log_nu),
        }

    def _multi_rhs_anchor_adapter(anchor_input, rhs_bars):
        # This is the intended adapter ownership: one local primal context,
        # with only cotangent-dependent work carrying the leading RHS axis.
        local_context = _build_local_primal_context(anchor_input)
        return jax.vmap(
            lambda rhs_bar: _support_adjoint_from_context(local_context, rhs_bar)
        )(rhs_bars)

    anchor_input = jnp.asarray(1.25)
    rhs_bars = jnp.asarray([0.5, -1.0, 2.0, 0.25])
    actual = _multi_rhs_anchor_adapter(anchor_input, rhs_bars)
    assert call_count["local_primal"] == 1
    assert actual["prepared"].shape == (4,)
    assert actual["drds"].shape == (4,)

    expected = jax.vmap(
        lambda rhs_bar: _support_adjoint_from_context(
            (2.0 * anchor_input, anchor_input + 3.0, anchor_input - 5.0),
            rhs_bar,
        )
    )(rhs_bars)
    _assert_float_tree_allclose(actual, expected)

    traced = jax.make_jaxpr(_multi_rhs_anchor_adapter)(anchor_input, rhs_bars)
    assert "scan[" not in str(traced)
    assert "map[" not in str(traced)


def test_multi_rhs_prepared_support_adapter_matches_scalar_local_pullbacks():
    """Exact local gate for the unselected multi-RHS NTX adapter.

    The ordinary helper is evaluated once per RHS only as the test oracle.
    The implementation under test receives the complete RHS batch and must
    preserve the scalar prepared, ``drds``, and interpolation-primal values.
    This test intentionally has no transport rollout, filesystem output, or
    profiling side effect.
    """

    prepared = ntx.prepare_monoenergetic_system(
        ntx.example_surface(),
        ntx.GridSpec(5, 5, 4),
    )
    scalar_field_bars = (
        (
            jnp.asarray(0.13),
            jnp.asarray([0.3, -0.2, 0.1, 0.4, -0.1, 0.2]),
            jnp.asarray([-0.3, 0.1, 0.2, -0.2, 0.3, -0.1]),
            jnp.asarray([0.2, 0.4, -0.3, 0.1, 0.2, -0.4]),
        ),
        (
            jnp.asarray(-0.31),
            jnp.asarray([-0.1, 0.2, 0.3, -0.4, 0.5, -0.2]),
            jnp.asarray([0.4, -0.3, 0.2, 0.1, -0.5, 0.3]),
            jnp.asarray([-0.2, 0.3, 0.4, -0.1, 0.2, 0.5]),
        ),
        (
            jnp.asarray(0.21),
            jnp.asarray([0.2, -0.4, 0.5, -0.3, 0.1, 0.3]),
            jnp.asarray([-0.5, 0.2, -0.1, 0.3, 0.4, -0.2]),
            jnp.asarray([0.1, -0.3, 0.2, 0.5, -0.4, 0.3]),
        ),
    )
    batched_field_bars = tuple(
        jnp.stack([field_bars[field_index] for field_bars in scalar_field_bars])
        for field_index in range(4)
    )
    # The transport benchmark uses an energy scan, whereas the original
    # gate covered only one energy.  Exercise both shapes: an error in the
    # energy reduction can leave profile bars apparently correct while
    # corrupting a geometry-only prepared leaf.
    for n_energy in (1, 2):
        model = _small_runtime_model(n_energy=n_energy)
        args = dict(
            drds_value=jnp.asarray(1.2),
            reference_nu_hat=jnp.linspace(1.0e-2, 1.8e-2, n_energy),
            reference_epsi_hat=jnp.linspace(1.0e-3, 2.0e-3, n_energy),
            vth_a=jnp.linspace(1.1, 1.2, n_energy),
        )
        actual_prepared, actual_drds, actual_primal = (
            model._pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs(
                prepared,
                field_bars=batched_field_bars,
                **args,
            )
        )
        native_prepared, native_drds, native_primal = (
            model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs(
                prepared,
                field_bars=batched_field_bars,
                **args,
            )
        )
        for rhs_index, field_bars in enumerate(scalar_field_bars):
            expected_prepared, expected_drds, expected_primal = (
                model._pullback_interpolated_moment_prepared_support_and_drds_only(
                    prepared,
                    field_bars=field_bars,
                    **args,
                )
            )
            _assert_float_tree_allclose(
                jax.tree_util.tree_map(lambda value: value[rhs_index], actual_prepared),
                expected_prepared,
            )
            _assert_float_tree_allclose(actual_drds[rhs_index], expected_drds)
            _assert_float_tree_allclose(actual_primal, expected_primal)
            _assert_float_tree_allclose(
                jax.tree_util.tree_map(lambda value: value[rhs_index], native_prepared),
                expected_prepared,
            )
            _assert_float_tree_allclose(native_drds[rhs_index], expected_drds)
            _assert_float_tree_allclose(native_primal, expected_primal)


def test_compact_and_reused_drds_native_multi_rhs_adapters_match_full_native_adapter():
    """Opt-in native payload and drds reductions preserve local cotangents."""

    model = _small_runtime_model(n_energy=2)
    prepared = ntx.prepare_monoenergetic_system(
        ntx.example_surface(), ntx.GridSpec(5, 5, 4)
    )
    field_bars = (
        jnp.asarray([0.13, -0.31, 0.21]),
        jnp.asarray([[0.3, -0.2, 0.1, 0.4, -0.1, 0.2],
                     [-0.1, 0.2, 0.3, -0.4, 0.5, -0.2],
                     [0.2, -0.4, 0.5, -0.3, 0.1, 0.3]]),
        jnp.asarray([[-0.3, 0.1, 0.2, -0.2, 0.3, -0.1],
                     [0.4, -0.3, 0.2, 0.1, -0.5, 0.3],
                     [-0.5, 0.2, -0.1, 0.3, 0.4, -0.2]]),
        jnp.asarray([[0.2, 0.4, -0.3, 0.1, 0.2, -0.4],
                     [-0.2, 0.3, 0.4, -0.1, 0.2, 0.5],
                     [0.1, -0.3, 0.2, 0.5, -0.4, 0.3]]),
    )
    args = dict(
        drds_value=jnp.asarray(1.2),
        reference_nu_hat=jnp.asarray([1.0e-2, 1.8e-2]),
        reference_epsi_hat=jnp.asarray([1.0e-3, 2.0e-3]),
        # Production supplies one species thermal speed; the local energy
        # scan must broadcast this scalar before attaching the RHS axis.
        vth_a=jnp.asarray(1.1),
        field_bars=field_bars,
        return_case_bars=True,
    )
    full = model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs(
        prepared, **args
    )
    compact = model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_compact(
        prepared, **args
    )
    reused_drds = model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_reuse_moment_drds_jvp(
        prepared, **args
    )
    compact_residual = model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_compact_residual_reuse_moment_drds_jvp(
        prepared, **args
    )
    for candidate in (compact, reused_drds, compact_residual):
        _assert_float_tree_allclose(candidate[0], full[0])
        _assert_float_tree_allclose(candidate[1], full[1])
        _assert_float_tree_allclose(candidate[2], full[2])
        _assert_float_tree_allclose(candidate[3], full[3])


def test_native_per_energy_call_boundary_matches_unbounded_native_vmec_pullback():
    """The local compilation boundary preserves native support and face bars.

    This is one small local NTX gate, not a transport rollout.  It covers the
    exact direct-directional algebra selected by the experimental boundary.
    """

    model = _small_runtime_model(n_energy=2)
    _surface, prepared = _small_vmec_prepared()
    field_bars = (
        jnp.asarray([0.0, 0.0]),
        jnp.asarray(
            [[0.3, -0.2, 0.1, 0.4, -0.1, 0.2],
             [-0.1, 0.2, 0.3, -0.4, 0.5, -0.2]]
        ),
        jnp.asarray(
            [[-0.3, 0.1, 0.2, -0.2, 0.3, -0.1],
             [0.4, -0.3, 0.2, 0.1, -0.5, 0.3]]
        ),
        jnp.asarray(
            [[0.2, 0.4, -0.3, 0.1, 0.2, -0.4],
             [-0.2, 0.3, 0.4, -0.1, 0.2, 0.5]]
        ),
    )
    args = dict(
        drds_value=jnp.asarray(1.2),
        reference_nu_hat=jnp.asarray([1.0e-2, 1.8e-2]),
        reference_epsi_hat=jnp.asarray([1.0e-3, 2.0e-3]),
        vth_a=jnp.asarray(1.1),
        field_bars=field_bars,
        return_case_bars=True,
        native_vmec_coefficient_bars_only=True,
        native_vmec_direct_directional_product_rule=True,
    )
    ordinary = model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_reuse_moment_drds_jvp_with_vmec_coefficients(
        prepared,
        native_per_energy_call_boundary=False,
        **args,
    )
    bounded = model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_reuse_moment_drds_jvp_with_vmec_coefficients(
        prepared,
        native_per_energy_call_boundary=True,
        **args,
    )
    _assert_float_tree_allclose(bounded, ordinary)


@pytest.mark.parametrize(
    ("field_index", "component_name"),
    (
        (1, "transport_moments"),
        (2, "dtransport_moments_d_er"),
        (3, "dtransport_moments_d_log_nu_star"),
    ),
)
def test_native_vmec_face_rebuild_component_oracle_matches_prepared_vjp(
    field_index,
    component_name,
):
    """Compare each actual NEOPAX low-dot component at the replacement boundary.

    This is the missing integration-level oracle: it retains the production
    local multi-energy/moment assembly but stops before interpolation, a Radau
    step, a VMEC solve, or any filesystem output.  It asks whether the native
    coefficient return is exactly the `face_prepared -> VmecSurface` VJP for
    one response component.
    """

    model = _small_runtime_model(n_energy=2)
    surface, prepared = _small_vmec_prepared()
    all_field_bars = (
        jnp.asarray([0.0, 0.0]),
        jnp.asarray(
            [[0.3, -0.2, 0.1, 0.4, -0.1, 0.2],
             [-0.1, 0.2, 0.3, -0.4, 0.5, -0.2]]
        ),
        jnp.asarray(
            [[-0.3, 0.1, 0.2, -0.2, 0.3, -0.1],
             [0.4, -0.3, 0.2, 0.1, -0.5, 0.3]]
        ),
        jnp.asarray(
            [[0.2, 0.4, -0.3, 0.1, 0.2, -0.4],
             [-0.2, 0.3, 0.4, -0.1, 0.2, 0.5]]
        ),
    )
    field_bars = tuple(
        value if index == field_index else jnp.zeros_like(value)
        for index, value in enumerate(all_field_bars)
    )
    args = dict(
        drds_value=jnp.asarray(1.2),
        reference_nu_hat=jnp.asarray([1.0e-2, 1.8e-2]),
        reference_epsi_hat=jnp.asarray([1.0e-3, 2.0e-3]),
        vth_a=jnp.asarray(1.1),
        field_bars=field_bars,
        return_case_bars=True,
    )
    prepared_bar, _drds_bar, _primal, _case_bars, native_bars = (
        model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_reuse_moment_drds_jvp_with_vmec_coefficients(
            prepared,
            native_vmec_coefficient_bars_only=False,
            **args,
        )
    )
    _, surface_pullback = jax.vjp(
        lambda surface_value: ntx.prepare_monoenergetic_system(
            surface_value, prepared.grid
        ),
        surface,
    )
    names = (
        "b_cos",
        "jacobian_cos",
        "b_sub_theta_cos",
        "b_sub_zeta_cos",
        "b_sup_theta_cos",
        "b_sup_zeta_cos",
        "b0",
    )
    expected = tuple(
        surface_pullback(
            jax.tree_util.tree_map(lambda value: value[rhs_index], prepared_bar)
        )[0]
        for rhs_index in range(int(jnp.asarray(field_bars[0]).shape[0]))
    )
    for name in names:
        expected_value = jnp.stack([getattr(value, name) for value in expected])
        actual_value = jnp.asarray(native_bars[name])
        if not jnp.allclose(actual_value, expected_value, rtol=1e-10, atol=1e-12):
            difference = jnp.abs(actual_value - expected_value)
            scale = jnp.maximum(jnp.abs(expected_value), 1.0e-30)
            raise AssertionError(
                f"{component_name} {name}: "
                f"max_abs={float(jnp.max(difference)):.16e} "
                f"max_rel={float(jnp.max(difference / scale)):.16e}"
            )
    # The native bridge exposes precisely the seven coefficient fields above.
    # Prove that every other differentiable VmecSurface field has no local
    # prepared-system cotangent in the explicit-epsi_hat transport contract;
    # otherwise this bridge would silently omit a state-dependent channel.
    for name in (
        "requested_psi_n",
        "psi_n",
        "nfp",
        "iota",
        "psi_a_hat",
        "phi_edge",
        "r_n",
        "r_hat",
        "dpsi_hat_dr_hat",
        "dr_hat_dpsi_hat",
        "aminor_p",
        "psi_p",
        "transport_psi_scale",
    ):
        values = tuple(getattr(value, name) for value in expected)
        if values[0] is None:
            continue
        expected_value = jnp.stack([jnp.asarray(value) for value in values])
        if jnp.issubdtype(expected_value.dtype, jnp.inexact):
            assert jnp.allclose(expected_value, 0.0, rtol=0.0, atol=1e-12), (
                component_name,
                name,
                expected_value,
            )


@pytest.mark.parametrize(
    ("field_index", "component_name"),
    (
        (1, "transport_moments"),
        (2, "dtransport_moments_d_er"),
        (3, "dtransport_moments_d_log_nu_star"),
    ),
)
def test_native_vmec_face_rebuild_er_drds_chain_matches_local_response_vjp(
    field_index,
    component_name,
):
    """Exercise ``Er * drds -> epsi_hat`` through the native local reverse.

    Unlike the coefficient-only oracle above, this starts with the physical
    local inputs and includes the return-case-bars chain back through
    ``_pullback_local_scan_inputs_and_drds_from_primitives``.  It remains a
    tiny in-memory two-energy check and does not construct a transport state
    or solve VMEC.
    """

    model = _small_runtime_model(n_energy=2)
    # The actual code reads this only if a velocity floor is configured.
    object.__setattr__(model, "er_v_floor", None)
    # ``_nu_over_vnew_local`` is jitted, so the fixture must use NEOPAX's
    # registered Species pytree rather than a host-only SimpleNamespace.
    object.__setattr__(
        model,
        "species",
        Species(
            # Although this gate evaluates ion index zero only, lax.cond
            # traces the Zeff collisionality branch too.  That branch needs
            # a physical electron lane, so retain the smallest valid pair.
            number_species=2,
            species_indices=jnp.asarray([0, 1]),
            mass_mp=jnp.asarray([2.0, 5.446e-4]),
            charge_qp=jnp.asarray([1.0, -1.0]),
            names=("ion", "e"),
        ),
    )
    _surface, prepared = _small_vmec_prepared()
    drds_value = jnp.asarray(1.2)
    er_value = jnp.asarray(-2.7e-3)
    temperature_local = jnp.asarray([1.4, 1.1])
    density_local = jnp.asarray([0.9, 0.9])
    # Match the production helper exactly.  In particular, temperatures are
    # stored in keV and Species.mass is in SI units.
    vthermal_local = get_v_thermal(model.species.mass, temperature_local)
    collisionality_kind = _collisionality_kind("default")
    all_field_bars = (
        jnp.asarray([0.0, 0.0]),
        jnp.asarray(
            [[0.3, -0.2, 0.1, 0.4, -0.1, 0.2],
             [-0.1, 0.2, 0.3, -0.4, 0.5, -0.2]]
        ),
        jnp.asarray(
            [[-0.3, 0.1, 0.2, -0.2, 0.3, -0.1],
             [0.4, -0.3, 0.2, 0.1, -0.5, 0.3]]
        ),
        jnp.asarray(
            [[0.2, 0.4, -0.3, 0.1, 0.2, -0.4],
             [-0.2, 0.3, 0.4, -0.1, 0.2, 0.5]]
        ),
    )
    field_bars = tuple(
        value if index == field_index else jnp.zeros_like(value)
        for index, value in enumerate(all_field_bars)
    )
    nu_hat, epsi_hat, vth_a = model._interpolated_moment_local_scan_primitives(
        drds_value=drds_value,
        species_index=0,
        er_value=er_value,
        temperature_local=temperature_local,
        density_local=density_local,
        vthermal_local=vthermal_local,
        collisionality_kind=collisionality_kind,
    )
    (
        native_prepared,
        native_direct_drds,
        _native_primal,
        (native_nu_bar, native_epsi_bar, native_vth_a_bar),
    ) = model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs_reuse_moment_drds_jvp(
        prepared,
        drds_value=drds_value,
        reference_nu_hat=nu_hat,
        reference_epsi_hat=epsi_hat,
        vth_a=vth_a,
        field_bars=field_bars,
        return_case_bars=True,
    )
    (
        native_primitive_drds,
        native_er,
        _native_temperature,
        _native_density,
    ) = jax.vmap(
        lambda nu_bar, epsi_bar, vth_bar: model._pullback_local_scan_inputs_and_drds_from_primitives(
            drds_value=drds_value,
            species_index=0,
            er_value=er_value,
            temperature_local=temperature_local,
            density_local=density_local,
            collisionality_kind=collisionality_kind,
            reference_nu_hat_bar=nu_bar,
            reference_epsi_hat_bar=epsi_bar,
            vth_a_bar=vth_bar,
        )
    )(native_nu_bar, native_epsi_bar, native_vth_a_bar)

    def _local_response(prepared_value, drds_local, er_local):
        return model._build_interpolated_moment_response_local(
            prepared_value,
            drds_value=drds_local,
            species_index=0,
            er_value=er_local,
            temperature_local=temperature_local,
            density_local=density_local,
            vthermal_local=vthermal_local,
            collisionality_kind=collisionality_kind,
        )

    _value, pullback = jax.vjp(_local_response, prepared, drds_value, er_value)
    for rhs_index in range(int(jnp.asarray(field_bars[0]).shape[0])):
        _expected_prepared, expected_drds, expected_er = pullback(
            tuple(value[rhs_index] for value in field_bars)
        )
        # The fixed-epsi component oracle already establishes the prepared
        # surface path.  This gate isolates the previously omitted physical
        # ``Er * drds -> epsi_hat`` case chain; some unrelated prepared leaves
        # carry NaNs for this deliberately tiny collision fixture.
        assert jnp.allclose(
            native_direct_drds[rhs_index] + native_primitive_drds[rhs_index],
            expected_drds,
            rtol=1e-10,
            atol=1e-12,
        ), component_name
        assert jnp.allclose(
            native_er[rhs_index], expected_er, rtol=1e-10, atol=1e-12
        ), component_name


def test_native_joint_local_multi_rhs_matches_explicit_state_support_vjp():
    """Native joint local bars equal explicit state/prepared VJPs per RHS.

    This is the numerical gate for the unselected initial-carry adapter.  It
    deliberately stops at one prepared system: there is no support build,
    interpolation, Radau step, VMEC solve, profile, or filesystem output.
    The Python loops are oracle-only; the implementation under test receives
    the complete on-device RHS batch at once.
    """

    model = _small_runtime_model(n_energy=2)
    object.__setattr__(model, "er_v_floor", None)
    object.__setattr__(
        model,
        "species",
        Species(
            number_species=2,
            species_indices=jnp.asarray([0, 1]),
            mass_mp=jnp.asarray([2.0, 5.446e-4]),
            charge_qp=jnp.asarray([1.0, -1.0]),
            names=("ion", "e"),
        ),
    )
    _surface, prepared = _small_vmec_prepared()
    drds_value = jnp.asarray(1.2)
    er_value = jnp.asarray(-2.7e-3)
    temperature_local = jnp.asarray([1.4, 1.1])
    density_local = jnp.asarray([0.9, 0.9])
    collisionality_kind = _collisionality_kind("default")
    rhs_fields = (
        jnp.asarray([0.17, -0.23]),
        jnp.asarray(
            [[0.3, -0.2, 0.1, 0.4, -0.1, 0.2],
             [-0.1, 0.2, 0.3, -0.4, 0.5, -0.2]]
        ),
        jnp.asarray(
            [[-0.3, 0.1, 0.2, -0.2, 0.3, -0.1],
             [0.4, -0.3, 0.2, 0.1, -0.5, 0.3]]
        ),
        jnp.asarray(
            [[0.2, 0.4, -0.3, 0.1, 0.2, -0.4],
             [-0.2, 0.3, 0.4, -0.1, 0.2, 0.5]]
        ),
    )
    # Species is the leading axis expected by the local joint helper; RHS is
    # next, and remains the matrix dimension passed to NTX.
    field_bars = tuple(
        jnp.stack((values, -0.7 * values), axis=0) for values in rhs_fields
    )
    actual = (
        model._pullback_interpolated_moment_response_local_fields_and_prepared_support_and_drds_flat_prepared(
            prepared,
            drds_value=drds_value,
            er_value=er_value,
            temperature_local=temperature_local,
            density_local=density_local,
            collisionality_kind=collisionality_kind,
            field_bars=field_bars,
            native_factorized_ntx_rhs=True,
            reuse_joint_moment_drds_jvp=True,
        )
    )

    def _add_trees(*trees):
        def _add_leaves(*leaves):
            first = jnp.asarray(leaves[0])
            if first.dtype == jax.dtypes.float0:
                return first
            return sum(leaves[1:], first)

        return jax.tree_util.tree_map(_add_leaves, *trees)

    rhs_count = int(field_bars[0].shape[1])
    expected_rows = []
    for rhs_index in range(rhs_count):
        species_rows = []
        for species_index in range(int(model.species.number_species)):
            def _local_response(
                prepared_value,
                drds_local,
                er_local,
                temperature_value,
                density_value,
            ):
                return model._build_interpolated_moment_response_local(
                    prepared_value,
                    drds_value=drds_local,
                    species_index=species_index,
                    er_value=er_local,
                    temperature_local=temperature_value,
                    density_local=density_value,
                    vthermal_local=get_v_thermal(model.species.mass, temperature_value),
                    collisionality_kind=collisionality_kind,
                )

            _, pullback = jax.vjp(
                _local_response,
                prepared,
                drds_value,
                er_value,
                temperature_local,
                density_local,
            )
            species_rows.append(
                pullback(tuple(field[species_index, rhs_index] for field in field_bars))
            )
        expected_rows.append(_add_trees(*species_rows))

    def _stack_rhs_leaves(*leaves):
        first = jnp.asarray(leaves[0])
        if first.dtype == jax.dtypes.float0:
            return first
        return jnp.stack(leaves)

    expected_prepared = jax.tree_util.tree_map(
        _stack_rhs_leaves, *(row[0] for row in expected_rows)
    )
    expected_state = tuple(
        jnp.stack([row[index] for row in expected_rows])
        for index in range(1, 5)
    )
    _assert_float_tree_allclose(actual[:4], expected_state, rtol=1e-9, atol=1e-11)
    _assert_float_tree_allclose(
        actual[4], tuple(jax.tree_util.tree_leaves(expected_prepared)),
        rtol=1e-9,
        atol=1e-11,
    )


def test_native_joint_local_vmec_contract_omits_only_zero_prepared_carry():
    """The compact joint local contract retains all non-prepared channels.

    This is deliberately a single local NTX system.  It proves the new carry
    reduction changes only the generic prepared return, which the native VMEC
    coefficient channel replaces, without starting a support build or a
    transport rollout.
    """

    model = _small_runtime_model(n_energy=2)
    object.__setattr__(model, "er_v_floor", None)
    object.__setattr__(
        model,
        "species",
        Species(
            number_species=2,
            species_indices=jnp.asarray([0, 1]),
            mass_mp=jnp.asarray([2.0, 5.446e-4]),
            charge_qp=jnp.asarray([1.0, -1.0]),
            names=("ion", "e"),
        ),
    )
    _surface, prepared = _small_vmec_prepared()
    field_values = (
        jnp.asarray([0.17, -0.23]),
        jnp.asarray([[0.3, -0.2, 0.1, 0.4, -0.1, 0.2], [-0.1, 0.2, 0.3, -0.4, 0.5, -0.2]]),
        jnp.asarray([[-0.3, 0.1, 0.2, -0.2, 0.3, -0.1], [0.4, -0.3, 0.2, 0.1, -0.5, 0.3]]),
        jnp.asarray([[0.2, 0.4, -0.3, 0.1, 0.2, -0.4], [-0.2, 0.3, 0.4, -0.1, 0.2, 0.5]]),
    )
    field_bars = tuple(jnp.stack((values, -0.7 * values), axis=0) for values in field_values)
    kwargs = dict(
        drds_value=jnp.asarray(1.2),
        er_value=jnp.asarray(-2.7e-3),
        temperature_local=jnp.asarray([1.4, 1.1]),
        density_local=jnp.asarray([0.9, 0.9]),
        collisionality_kind=_collisionality_kind("default"),
        field_bars=field_bars,
        native_factorized_ntx_rhs=True,
        reuse_joint_moment_drds_jvp=True,
        return_native_vmec_coefficient_bars=True,
    )
    full = model._pullback_interpolated_moment_response_local_fields_and_prepared_support_and_drds_flat_prepared(
        prepared, **kwargs
    )
    compact = model._pullback_interpolated_moment_response_local_fields_and_prepared_support_and_drds_flat_prepared(
        prepared, omit_generic_prepared_carry=True, **kwargs
    )
    _assert_float_tree_allclose(compact[:4], full[:4])
    assert compact[4] == ()
    _assert_float_tree_allclose(compact[5], full[5])


def test_native_vmec_face_rebuild_accumulation_matches_generic_prepared_vjp():
    """Compare the real native replacement boundary without a transport rollout.

    The earlier component oracle proves one local ``face_prepared`` system.
    This gate adds the production face-anchor interpolation, species sum, and
    objective/RHS batching, then checks that converting the generic stacked
    prepared cotangent to VMEC surface coefficients equals the native stacked
    coefficient return.  It deliberately has no VMEC solve or raw-block
    pullback: the two sides receive identical fixed traceable surfaces.
    """

    surface, _prepared = _small_vmec_prepared()
    grid = ntx.GridSpec(5, 5, 4)
    geometry = _TestMomentumGeometry(
        a_b=jnp.asarray(1.0),
        r_grid=jnp.asarray([0.3, 0.7]),
        r_grid_half=jnp.asarray([0.1, 0.5, 0.9]),
        Bsqav=jnp.asarray([1.2, 1.3]),
        G_PS=jnp.asarray([1.0, 1.0]),
        B0=jnp.asarray([1.0, 1.0]),
    )

    def _channels(rho):
        rho = jnp.asarray(rho, dtype=jnp.float64)
        ones = jnp.ones_like(rho)
        return NTXRuntimeScanChannels.from_mapping(
            rho,
            {
                "a_b": 1.0, "psia": 1.0, "b00": ones, "r00": ones,
                "boozer_i": ones, "boozer_g": ones, "iota": ones,
                "drds": ones, "dr_tildedr": ones, "dr_tildeds": ones,
                "fac_reference_to_sfincs_11": ones,
                "fac_reference_to_sfincs_31": ones,
                "fac_reference_to_sfincs_33": ones,
                "fac_sfincs_to_dkes_11": ones,
                "fac_sfincs_to_dkes_31": ones,
                "fac_sfincs_to_dkes_33": ones,
                "fac_dkes_to_d11star": ones,
                "fac_dkes_to_d31star": ones,
                "fac_dkes_to_d33star": ones,
            },
        )

    # Unlike the earlier carrier-only version of this gate, each face must
    # have a distinct prepared VMEC surface.  The production support builder
    # does exactly that, and a re-derived native coefficient rule could agree
    # on repeated copies while failing once Fourier data vary with radius.
    face_surfaces = tuple(
        dataclasses.replace(
            surface,
            b_cos=surface.b_cos + jnp.asarray([0.03 * face, -0.01 * face]),
            jacobian_cos=surface.jacobian_cos + jnp.asarray([0.02 * face, 0.004 * face]),
            b_sub_theta_cos=surface.b_sub_theta_cos + jnp.asarray([0.01 * face, -0.003 * face]),
            b_sub_zeta_cos=surface.b_sub_zeta_cos + jnp.asarray([-0.02 * face, 0.005 * face]),
            b_sup_theta_cos=surface.b_sup_theta_cos + jnp.asarray([0.015 * face, 0.002 * face]),
            b_sup_zeta_cos=surface.b_sup_zeta_cos + jnp.asarray([-0.01 * face, 0.003 * face]),
            b0=surface.b0 + jnp.asarray(0.02 * face),
        )
        for face in range(3)
    )
    face_prepared_values = tuple(
        ntx.prepare_monoenergetic_system(face_surface, grid)
        for face_surface in face_surfaces
    )

    def _stack_prepared(values):
        return jax.tree_util.tree_map(
            lambda *leaves: (
                None if leaves[0] is None
                else jnp.stack([jnp.asarray(leaf) for leaf in leaves], axis=0)
            ),
            *values,
        )

    support = NTXExactLijRuntimeSupport(
        center_channels=_channels(geometry.r_grid),
        face_channels=_channels(geometry.r_grid_half),
        center_prepared=_stack_prepared(face_prepared_values[:2]),
        face_prepared=_stack_prepared(face_prepared_values),
        grid=grid,
    )
    species = Species(
        # Keep this fixture at the active-wHe dimensionality.  The compact
        # bootstrap VJP has a separate local implementation from the fast
        # Upar-only primal, so a three-species parity check cannot validate
        # the four-species correction matrix introduced for wHe.
        number_species=4,
        species_indices=jnp.asarray([0, 1, 2, 3]),
        mass_mp=jnp.asarray([5.446e-4, 2.0, 3.0, 4.0]),
        charge_qp=jnp.asarray([-1.0, 1.0, 1.0, 2.0]),
        names=("e", "D", "T", "He"),
    )
    # ``get_Matrix`` is JIT compiled, so this needs the same JAX-pytree
    # energy-grid object used by production rather than a SimpleNamespace.
    energy_grid = StandardLaguerreEnergyGrid(n_x=2)
    model = NTXExactLijRuntimeTransportModel(
        species=species, energy_grid=energy_grid, geometry=geometry,
        vmec_file=None, boozer_file=None, support=support,
        center_response_mode="interpolate_from_faces", response_anchor_count=2,
    )
    state = TransportState(
        density=jnp.asarray(
            [[1.0, 1.15], [1.0, 1.15], [0.8, 0.9], [0.3, 0.34]]
        ),
        pressure=jnp.asarray(
            [[1.3, 1.61], [1.1, 1.38], [0.7, 0.87], [0.24, 0.29]]
        ),
        Er=jnp.asarray([2.0e-4, 2.5e-4]),
    )
    # The bootstrap-only primal may omit Gamma/Q/qpar/Upar2, but must retain
    # exactly the regularized Upar consumed by the production objective.
    _assert_float_tree_allclose(
        model.evaluate_momentum_corrected_upar_only(state),
        model.evaluate_momentum_corrected_fluxes(state)["Upar"],
        rtol=1e-10,
        atol=1e-12,
    )
    # The production ``joint_local_vjp_upar_only`` mode evaluates the fast
    # Upar-only primal but obtains its derivative from the local corrected-
    # flux VJP.  Check that exact duality at wHe dimensionality, including a
    # density/pressure/Er direction; this is the direct oracle for a claimed
    # bootstrap sign error before involving any transport rollout.
    upar_bar = jnp.asarray(
        [[0.3, -0.2], [-0.4, 0.5], [0.2, 0.1], [-0.15, 0.25]]
    )
    state_direction = dataclasses.replace(
        state,
        density=jnp.asarray(
            [[0.12, -0.07], [-0.08, 0.03], [0.05, 0.09], [-0.02, 0.04]]
        ),
        pressure=jnp.asarray(
            [[0.17, -0.06], [-0.04, 0.11], [0.08, -0.03], [0.02, 0.05]]
        ),
        Er=jnp.asarray([3.0e-5, -2.0e-5]),
    )
    upar_tangent = jax.jvp(
        model.evaluate_momentum_corrected_upar_only,
        (state,),
        (state_direction,),
    )[1]
    compact_state_bar = model.pullback_momentum_corrected_upar_state_by_radius(
        state,
        upar_bar,
    )
    def _float_tree_vdot(bar_tree, direction_tree):
        return sum(
            jnp.vdot(jnp.asarray(bar), jnp.asarray(direction))
            for bar, direction in zip(
                jax.tree_util.tree_leaves(bar_tree),
                jax.tree_util.tree_leaves(direction_tree),
                strict=True,
            )
            if jnp.issubdtype(jnp.asarray(bar).dtype, jnp.inexact)
        )

    compact_contraction = _float_tree_vdot(compact_state_bar, state_direction)
    assert jnp.allclose(
        compact_contraction,
        jnp.vdot(upar_bar, upar_tangent),
        rtol=1e-10,
        atol=1e-12,
    )
    geometry_direction = _TestMomentumGeometry(
        a_b=jnp.asarray(0.03),
        r_grid=jnp.asarray([0.01, -0.02]),
        r_grid_half=jnp.asarray([0.01, -0.02, 0.03]),
        Bsqav=jnp.asarray([0.04, -0.03]),
        G_PS=jnp.asarray([0.02, -0.01]),
        B0=jnp.asarray([-0.02, 0.01]),
    )
    upar_geometry_tangent = jax.jvp(
        lambda geometry_value: dataclasses.replace(
            model, geometry=geometry_value
        ).evaluate_momentum_corrected_upar_only(state),
        (geometry,),
        (geometry_direction,),
    )[1]
    compact_geometry_bar = model.pullback_momentum_corrected_upar_geometry_by_radius(
        state,
        upar_bar,
        geometry,
        support,
    )
    assert jnp.allclose(
        _float_tree_vdot(compact_geometry_bar, geometry_direction),
        jnp.vdot(upar_bar, upar_geometry_tangent),
        rtol=1e-10,
        atol=1e-12,
    )
    def _jvp_zero_like(value):
        array = jnp.asarray(value)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            return jnp.zeros_like(array)
        return jnp.zeros(array.shape, dtype=jax.dtypes.float0)

    def _prepared_support_direction(value):
        array = jnp.asarray(value)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            return 0.013 * jnp.ones_like(array)
        return jnp.zeros(array.shape, dtype=jax.dtypes.float0)

    support_zero_direction = jax.tree_util.tree_map(_jvp_zero_like, support)
    support_direction = dataclasses.replace(
        support_zero_direction,
        center_channels=dataclasses.replace(
            support_zero_direction.center_channels,
            drds=jnp.asarray([0.07, -0.04]),
        ),
        center_prepared=jax.tree_util.tree_map(
            _prepared_support_direction,
            support.center_prepared,
        ),
    )
    upar_support_tangent = jax.jvp(
        lambda support_value: model.with_support_payload(
            support_value
        ).evaluate_momentum_corrected_upar_only(state),
        (support,),
        (support_direction,),
    )[1]
    compact_support_leaves = model.pullback_momentum_corrected_upar_support_by_radius(
        state,
        upar_bar,
        support,
    )
    compact_support_contraction = sum(
        jnp.vdot(jnp.asarray(bar), jnp.asarray(direction))
        for bar, direction in zip(
            compact_support_leaves,
            jax.tree_util.tree_leaves(support_direction),
            strict=True,
        )
        if (
            jnp.issubdtype(jnp.asarray(bar).dtype, jnp.inexact)
            and jnp.issubdtype(jnp.asarray(direction).dtype, jnp.inexact)
        )
    )
    assert jnp.allclose(
        compact_support_contraction,
        jnp.vdot(upar_bar, upar_support_tangent),
        rtol=1e-10,
        atol=1e-12,
    )
    response_bars = jax.tree_util.tree_map(
        lambda value: jnp.stack(
            (jnp.full_like(jnp.asarray(value), 0.17),
             jnp.full_like(jnp.asarray(value), -0.23)),
            axis=0,
        ),
        model.build_lagged_response(state),
    )
    generic = model.pullback_build_lagged_response_support_payload_batched_interpolated_faces_multi_rhs_shared_primal(
        state, response_bars, support,
        native_factorized_ntx_rhs=True,
        reuse_joint_moment_drds_jvp=True,
    )
    native_support, native_bars = (
        model.pullback_build_lagged_response_support_payload_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients(
            state, response_bars, support,
        )
    )
    _assert_float_tree_allclose(
        native_support.face_channels, generic.face_channels,
        rtol=1e-10, atol=1e-12,
    )
    # This selector deliberately carries the face-prepared contribution only
    # through ``native_bars``.  A nonzero generic prepared payload here would
    # double-count it when the later VMEC-state bridge is applied.
    _assert_float_tree_allclose(
        native_support.face_prepared,
        jax.tree_util.tree_map(jnp.zeros_like, native_support.face_prepared),
        rtol=0.0,
        atol=0.0,
    )
    names = (
        "b_cos", "jacobian_cos", "b_sub_theta_cos", "b_sub_zeta_cos",
        "b_sup_theta_cos", "b_sup_zeta_cos", "b0",
    )
    expected_by_name = {name: [] for name in names}
    for rhs_index in range(2):
        expected_by_face = {name: [] for name in names}
        for face_index in range(3):
            _, surface_pullback = jax.vjp(
                lambda surface_value: ntx.prepare_monoenergetic_system(
                    surface_value, grid
                ),
                face_surfaces[face_index],
            )
            prepared_bar = jax.tree_util.tree_map(
                lambda value: jnp.asarray(value)[rhs_index, face_index],
                generic.face_prepared,
            )
            surface_bar = surface_pullback(prepared_bar)[0]
            for name in names:
                expected_by_face[name].append(jnp.asarray(getattr(surface_bar, name)))
        for name in names:
            expected_by_name[name].append(jnp.stack(expected_by_face[name], axis=0))
    for name in names:
        assert jnp.allclose(
            native_bars[name], jnp.stack(expected_by_name[name], axis=0),
            rtol=1e-10, atol=1e-12,
        ), name

    # This is the exact post-NTX replacement used by the geometry bridge.
    # The generic route contracts ``face_prepared`` bars with a JVP of
    # ``prepare_monoenergetic_system(face_surface(state))``.  The native route
    # contracts the parallel coefficient bars with the JVP of just those same
    # face coefficients.  Use one common, two-direction state tangent and
    # require row-by-row equality for the complete two-RHS accumulation.
    def _surfaces_from_mock_state(mock_state):
        first, second = mock_state
        return tuple(
            dataclasses.replace(
                face_surface,
                b_cos=face_surface.b_cos + first * jnp.asarray([0.11, -0.07]),
                jacobian_cos=face_surface.jacobian_cos + second * jnp.asarray([0.05, 0.03]),
                b_sub_theta_cos=face_surface.b_sub_theta_cos + first * jnp.asarray([-0.02, 0.04]),
                b_sub_zeta_cos=face_surface.b_sub_zeta_cos + second * jnp.asarray([0.06, -0.01]),
                b_sup_theta_cos=face_surface.b_sup_theta_cos + first * jnp.asarray([0.03, 0.02]),
                b_sup_zeta_cos=face_surface.b_sup_zeta_cos + second * jnp.asarray([-0.04, 0.05]),
                b0=face_surface.b0 + first * 0.09 - second * 0.06,
            )
            for face_surface in face_surfaces
        )

    def _prepared_from_mock_state(mock_state):
        return _stack_prepared(
            tuple(
                ntx.prepare_monoenergetic_system(face_surface, grid)
                for face_surface in _surfaces_from_mock_state(mock_state)
            )
        )

    def _coefficient_tuple_from_mock_state(mock_state):
        surfaces = _surfaces_from_mock_state(mock_state)
        return tuple(
            jnp.stack([jnp.asarray(getattr(face_surface, name)) for face_surface in surfaces])
            for name in names
        )

    mock_state = (jnp.asarray(0.0), jnp.asarray(0.0))
    mock_tangent = (jnp.asarray(0.37), jnp.asarray(-0.29))
    _, prepared_tangent = jax.jvp(
        _prepared_from_mock_state,
        (mock_state,),
        (mock_tangent,),
    )
    prepared_bar_leaves = jax.tree_util.tree_leaves(generic.face_prepared)
    prepared_tangent_leaves = jax.tree_util.tree_leaves(prepared_tangent)
    generic_jvp_contraction = []
    for rhs_index in range(2):
        total = jnp.asarray(0.0)
        for bar_leaf, tangent_leaf in zip(
            prepared_bar_leaves, prepared_tangent_leaves, strict=True
        ):
            tangent_array = jnp.asarray(tangent_leaf)
            if jnp.issubdtype(tangent_array.dtype, jnp.inexact):
                total = total + jnp.vdot(jnp.asarray(bar_leaf)[rhs_index], tangent_array)
        generic_jvp_contraction.append(total)
    native_jvp_contraction = _native_vmec_coefficient_tangent_contraction(
        tuple(jnp.asarray(native_bars[name]) for name in names),
        jax.jvp(
            _coefficient_tuple_from_mock_state,
            (mock_state,),
            (mock_tangent,),
        )[1],
    )
    assert jnp.allclose(
        native_jvp_contraction,
        jnp.stack(generic_jvp_contraction),
        rtol=1e-10,
        atol=1e-12,
    )


def test_native_multi_rhs_support_retains_local_drds_case_chain():
    """The native support rule must include ``drds -> epsi_hat`` exactly.

    The former native selector only compared against a support-only NTX
    oracle, which deliberately holds the local case fixed.  This miniature
    response instead makes ``epsi_hat`` depend on ``drds`` and compares the
    corrected native case bars plus their primitive transpose to the actual
    local response VJP.  It is in-memory only and has no transport rollout.
    """

    model = _small_runtime_model(n_energy=2)
    prepared = ntx.prepare_monoenergetic_system(
        ntx.example_surface(),
        ntx.GridSpec(5, 5, 4),
    )
    epsi_per_drds = jnp.asarray([1.0e-3, 1.6e-3])

    def _mock_local_primitives(*, drds_value, **_unused):
        return (
            jnp.asarray([1.0e-2, 1.8e-2]),
            epsi_per_drds * drds_value,
            jnp.asarray(1.1),
        )

    object.__setattr__(model, "_interpolated_moment_local_scan_primitives", _mock_local_primitives)
    scalar_field_bars = (
        (
            jnp.asarray(0.0),
            jnp.asarray([0.3, -0.2, 0.1, 0.4, -0.1, 0.2]),
            jnp.asarray([-0.3, 0.1, 0.2, -0.2, 0.3, -0.1]),
            jnp.asarray([0.2, 0.4, -0.3, 0.1, 0.2, -0.4]),
        ),
        (
            jnp.asarray(0.0),
            jnp.asarray([-0.1, 0.2, 0.3, -0.4, 0.5, -0.2]),
            jnp.asarray([0.4, -0.3, 0.2, 0.1, -0.5, 0.3]),
            jnp.asarray([-0.2, 0.3, 0.4, -0.1, 0.2, 0.5]),
        ),
    )
    batched_field_bars = tuple(
        jnp.stack([bars[field_index] for bars in scalar_field_bars])
        for field_index in range(4)
    )
    drds = jnp.asarray(1.2)
    reference_nu_hat, reference_epsi_hat, vth_a = _mock_local_primitives(
        drds_value=drds,
    )
    (
        native_prepared,
        native_direct_drds,
        _native_primal,
        (_native_nu_hat, native_epsi_hat, _native_vth_a),
    ) = model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs(
        prepared,
        drds_value=drds,
        reference_nu_hat=reference_nu_hat,
        reference_epsi_hat=reference_epsi_hat,
        vth_a=vth_a,
        field_bars=batched_field_bars,
        return_case_bars=True,
    )
    native_drds = native_direct_drds + jnp.sum(native_epsi_hat * epsi_per_drds, axis=1)

    def _response(prepared_value, drds_value):
        return model._build_interpolated_moment_response_local(
            prepared_value,
            drds_value=drds_value,
            species_index=0,
            er_value=jnp.asarray(0.0),
            temperature_local=jnp.asarray([1.0]),
            density_local=jnp.asarray([1.0]),
            vthermal_local=jnp.asarray([1.1]),
            collisionality_kind="unused",
        )

    # The complete RHS below is linear.  Gate its three NTX-dependent output
    # channels independently so a failure identifies the low-dot term that
    # needs correction, rather than merely reporting a summed prepared bar.
    component_names = (
        "transport_moments",
        "dtransport_moments_d_er",
        "dtransport_moments_d_log_nu_star",
    )
    reference_field_bars = scalar_field_bars[0]
    for field_index, component_name in zip((1, 2, 3), component_names, strict=True):
        component_field_bars = tuple(
            value if index == field_index else jnp.zeros_like(value)
            for index, value in enumerate(reference_field_bars)
        )
        component_batch = tuple(value[None, ...] for value in component_field_bars)
        (
            generic_component_prepared,
            _generic_component_direct_drds,
            _generic_component_primal,
        ) = model._pullback_interpolated_moment_prepared_support_and_drds_only_multi_rhs(
            prepared,
            drds_value=drds,
            reference_nu_hat=reference_nu_hat,
                reference_epsi_hat=reference_epsi_hat,
                vth_a=vth_a,
                field_bars=component_batch,
                include_second_direction_base_prepared=False,
            )
        (
            native_component_prepared,
            _native_component_direct_drds,
            _native_component_primal,
            _native_component_case_bars,
        ) = model._pullback_interpolated_moment_prepared_support_and_drds_only_native_multi_rhs(
            prepared,
            drds_value=drds,
            reference_nu_hat=reference_nu_hat,
            reference_epsi_hat=reference_epsi_hat,
            vth_a=vth_a,
            field_bars=component_batch,
            return_case_bars=True,
        )
        _response_value, component_pullback = jax.vjp(_response, prepared, drds)
        expected_component_prepared, _expected_component_drds = component_pullback(
            component_field_bars
        )
        try:
            _assert_float_tree_allclose(
                jax.tree_util.tree_map(
                    lambda value: value[0], generic_component_prepared
                ),
                expected_component_prepared,
                rtol=5e-9,
            )
        except AssertionError as error:
            raise AssertionError(
                "Generic corrected low-dot mismatch in "
                f"{component_name}; the remaining issue is shared algebra, "
                "not native matrix-RHS packing."
            ) from error
        try:
            _assert_float_tree_allclose(
                jax.tree_util.tree_map(
                    lambda value: value[0], native_component_prepared
                ),
                expected_component_prepared,
                rtol=5e-9,
            )
        except AssertionError as error:
            raise AssertionError(
                "Native matrix-RHS prepared-support low-dot mismatch in "
                f"{component_name}."
            ) from error
        native_component_epsi_hat = _native_component_case_bars[1]
        native_component_drds = (
            _native_component_direct_drds
            + jnp.sum(native_component_epsi_hat * epsi_per_drds, axis=1)
        )
        try:
            _assert_float_tree_allclose(
                native_component_drds[0],
                _expected_component_drds,
                rtol=5e-9,
            )
        except AssertionError as error:
            raise AssertionError(
                "Native matrix-RHS drds support-chain mismatch in "
                f"{component_name}."
            ) from error

    for rhs_index, field_bars in enumerate(scalar_field_bars):
        _response_value, pullback = jax.vjp(_response, prepared, drds)
        expected_prepared, expected_drds = pullback(field_bars)
        _assert_float_tree_allclose(
            jax.tree_util.tree_map(lambda value: value[rhs_index], native_prepared),
            expected_prepared,
            rtol=5e-9,
        )
        _assert_float_tree_allclose(
            native_drds[rhs_index], expected_drds, rtol=5e-9
        )


def test_compact_local_coefficient_record_matches_ordinary_response():
    """The record adapter exposes existing coefficient primitives exactly."""

    model = _small_runtime_model(n_energy=2)
    prepared = ntx.prepare_monoenergetic_system(
        ntx.example_surface(),
        ntx.GridSpec(5, 5, 4),
    )
    args = dict(
        drds_value=jnp.asarray(1.2),
        nu_hat_a=jnp.asarray([1.0e-2, 1.8e-2]),
        epsi_hat_a=jnp.asarray([1.0e-3, 2.0e-3]),
        vth_a=jnp.asarray([1.1, 1.2]),
    )

    reference = model._interpolated_moment_reduced_local_outputs_from_primitives(
        prepared,
        **args,
    )
    actual, record = (
        model._interpolated_moment_reduced_local_outputs_with_coefficient_record_from_primitives(
            prepared,
            **args,
        )
    )

    _assert_float_tree_allclose(actual, reference)
    assert record.coefficient_scan.shape == (2, 5)
    assert record.dcoefficient_scan_d_er.shape == (2, 5)
    assert record.dcoefficient_scan_d_log_nu_star.shape == (2, 5)
    assert all(
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree_util.tree_leaves(record)
    )

    shaped_response, shaped_record = jax.eval_shape(
        lambda nu_hat_a, epsi_hat_a, vth_a: model._interpolated_moment_reduced_local_outputs_with_coefficient_record_from_primitives(
            prepared,
            drds_value=jnp.asarray(1.2),
            nu_hat_a=nu_hat_a,
            epsi_hat_a=epsi_hat_a,
            vth_a=vth_a,
        ),
        args["nu_hat_a"],
        args["epsi_hat_a"],
        args["vth_a"],
    )
    assert jax.tree_util.tree_structure(shaped_response) == jax.tree_util.tree_structure(reference)
    assert jax.tree_util.tree_structure(shaped_record) == jax.tree_util.tree_structure(record)
    assert shaped_record.coefficient_scan.shape == record.coefficient_scan.shape


def test_compact_record_lagged_preparation_preserves_rebuild_and_reuse_contract():
    """The experimental Radau hook returns one paired result without carry growth."""

    kernel_context = SimpleNamespace(use_transport_lagged_response=True)
    zero_record = {"coefficients": jnp.zeros((1, 1, 1, 5))}

    def build_with_record(state):
        return {
            "response": state + 2.0
        }, {"coefficients": jnp.broadcast_to(state[None, None, :, None], (1, 1, 1, 5))}

    rebuild_carry = SimpleNamespace(
        y=jnp.asarray([3.0]),
        lagged_response_valid=jnp.asarray(False),
        lagged_response_cache={"response": jnp.asarray([-1.0])},
        lagged_reference_y=jnp.asarray([-2.0]),
    )
    response, reference_y, reused, record = _radau_prepare_lagged_response_with_compact_coefficient_record(
        kernel_context,
        rebuild_carry,
        lambda y: y,
        None,
        build_with_record,
        lambda: zero_record,
    )
    assert jnp.allclose(response["response"], jnp.asarray([5.0]))
    assert jnp.allclose(reference_y, rebuild_carry.y)
    assert not bool(reused)
    assert jnp.allclose(record["coefficients"], jnp.full((1, 1, 1, 5), 3.0))

    reuse_carry = SimpleNamespace(
        y=jnp.asarray([4.0]),
        lagged_response_valid=jnp.asarray(True),
        lagged_response_cache={"response": jnp.asarray([7.0])},
        lagged_reference_y=jnp.asarray([6.0]),
    )
    response, reference_y, reused, record = _radau_prepare_lagged_response_with_compact_coefficient_record(
        kernel_context,
        reuse_carry,
        lambda y: y,
        None,
        build_with_record,
        lambda: zero_record,
    )
    assert jnp.allclose(response["response"], reuse_carry.lagged_response_cache["response"])
    assert jnp.allclose(reference_y, reuse_carry.lagged_reference_y)
    assert bool(reused)
    _assert_float_tree_allclose(record, zero_record)
