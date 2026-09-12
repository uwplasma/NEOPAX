"""Fast assembly checks for shared-primal database RHS support transposes.

These fixtures use the real fixed-flux capture, assembly and equation-to-flux
VJP.  Only the local flux laws are small polynomials; no NTX/VMEC solve or
recorded-scan transpose is needed.
"""

from collections import Counter

import jax
import jax.numpy as jnp
import pytest

from NEOPAX._state import TransportState
from NEOPAX._transport_equations import ComposedEquationSystem


def _face_values(values):
    return jnp.concatenate((values, values[..., -1:]), axis=-1)


def _center_fluxes(state, geometry, database):
    return {
        "Gamma": (
            state.density * database["values"]
            + state.pressure * database["coordinate"]
            + geometry["scale"] * state.Er
        ),
        "Q": (
            state.pressure**2 * database["values"]
            + geometry["metric"] * state.density
            + database["coordinate"] * state.Er
        ),
    }


def _face_fluxes(kind, state, geometry, database):
    density = _face_values(state.density)
    pressure = _face_values(state.pressure)
    table = _face_values(database["values"])
    weights = jnp.asarray([0.5, 1.5, 2.5])
    if kind == "density":
        return {
            "Gamma": (
                density * (table + database["coordinate"])
                + geometry["scale"] * weights
            )
        }
    return {
        "Gamma": (
            density * (2.0 * table - database["coordinate"])
            + jnp.sum(geometry["metric"]) * weights
        ),
        "Q": (
            pressure * (table**2 + 3.0 * database["coordinate"])
            + geometry["scale"] * pressure * weights
        ),
    }


class _CompactOwner:
    def __init__(self, support, calls):
        self.support = support
        self.calls = calls

    def __call__(self, state):
        self.calls["centers"] += 1
        return _center_fluxes(state, **self.support)

    def pullback_direct_rhs_support_payload(self, state, flux_bar, support):
        _, pullback = jax.vjp(
            lambda database: _center_fluxes(state, support["geometry"], database),
            support["database"],
        )
        return {"database": pullback(flux_bar)[0]}

    def pullback_direct_rhs_geometry_by_radius(self, state, flux_bar, geometry):
        _, pullback = jax.vjp(
            lambda value: _center_fluxes(state, value, self.support["database"]),
            geometry,
        )
        return pullback(flux_bar)[0]


class _NativeFaces:
    def __init__(self, kind, support, calls):
        self.kind = kind
        self.support = support
        self.calls = calls

    def __call__(self, state, *, center_fluxes):
        assert set(center_fluxes) == {"Gamma", "Q"}
        self.calls[f"{self.kind}_faces"] += 1
        return _face_fluxes(self.kind, state, **self.support)

    def database_table_pullback(self, state, _centers, face_bar, support):
        _, pullback = jax.vjp(
            lambda database: _face_fluxes(
                self.kind, state, support["geometry"], database
            ),
            support["database"],
        )
        return {"database": pullback(face_bar)[0]}

    def database_geometry_pullback(self, state, _centers, face_bar, support):
        _, pullback = jax.vjp(
            lambda geometry: _face_fluxes(
                self.kind, state, geometry, support["database"]
            ),
            support["geometry"],
        )
        return pullback(face_bar)[0]


class _Density:
    name = "density"

    def __init__(self, support, calls):
        self.geometry = support["geometry"]
        self.face_flux_builder = _NativeFaces("density", support, calls)

    @staticmethod
    def _use_model_face_particle_fluxes():
        return True

    def __call__(self, state, *, fluxes):
        return (
            self.geometry["scale"] * jnp.diff(fluxes["Gamma_faces"], axis=-1)
            + self.geometry["metric"] * fluxes["Gamma"]
            + 0.2 * state.pressure
        )


class _Temperature:
    name = "temperature"

    def __init__(self, support, calls):
        self.geometry = support["geometry"]
        self.face_flux_builder = _NativeFaces("temperature", support, calls)

    @staticmethod
    def _use_model_face_heat_fluxes():
        return True

    @staticmethod
    def _use_model_face_particle_fluxes():
        return True

    @staticmethod
    def _use_face_completed_work_term():
        return True

    def __call__(self, state, *, fluxes):
        return (
            self.geometry["metric"] * jnp.diff(fluxes["Q_faces"], axis=-1)
            + self.geometry["scale"] * fluxes["Q"]
            + state.Er * fluxes["Gamma_faces"][..., 1:]
        )


class _Er:
    name = "Er"

    def __init__(self, support):
        self.geometry = support["geometry"]

    def __call__(self, state, *, fluxes, er_edge_override=None):
        assert er_edge_override is None
        return self.geometry["scale"] * jnp.sum(fluxes["Gamma"], axis=0) + state.Er


def _equations(support, calls):
    density = _Density(support, calls)
    temperature = _Temperature(support, calls)
    er = _Er(support)
    equations = ComposedEquationSystem(
        equations=(density, temperature, er),
        density_equation=density,
        temperature_equation=temperature,
        er_equation=er,
        shared_flux_model=_CompactOwner(support, calls),
        config={},
        solver_cfg={},
        boundary_models={},
    )

    def _bind(_model, payload):
        calls["bind"] += 1
        return _CompactOwner(payload, calls)

    def _prepare(state):
        calls["prepare"] += 1
        # A non-None electron index makes accidental normalization of the
        # existing table/flux (None) and equation-geometry (1) contracts visible.
        return state, 1

    def _capture(state, center_fluxes):
        calls["capture"] += 1
        return ComposedEquationSystem._capture_database_primal_fixed_flux_payloads(
            equations, state, center_fluxes
        )

    def _flux_pullback(state, eidx, reference, rhs_bar, payload):
        calls["flux_vjp"] += 1
        assert eidx is None
        return ComposedEquationSystem._pullback_database_fixed_flux_payloads(
            equations, state, eidx, reference, rhs_bar, payload
        )

    def _at_geometry(geometry, _owner):
        result = _equations({"geometry": geometry, "database": support["database"]}, calls)
        base_evaluate = result._evaluate_database_fixed_fluxes_from_working_state

        def _evaluate(state, eidx, reference, payload, **kwargs):
            assert eidx == 1
            return base_evaluate(state, eidx, reference, payload, **kwargs)

        object.__setattr__(result, "_evaluate_database_fixed_fluxes_from_working_state", _evaluate)
        return result

    object.__setattr__(equations, "_flux_model_with_realtime_support_payload", _bind)
    object.__setattr__(equations, "_prepare_working_state", _prepare)
    object.__setattr__(equations, "_capture_database_primal_fixed_flux_payloads", _capture)
    object.__setattr__(equations, "_pullback_database_fixed_flux_payloads", _flux_pullback)
    object.__setattr__(equations, "_with_database_equation_geometry_and_fixed_flux", _at_geometry)
    object.__setattr__(
        equations, "with_realtime_geometry_support_payload", lambda payload: _equations(payload, calls)
    )
    return equations


def _inputs():
    state = TransportState(
        density=jnp.asarray([[1.1, 1.8], [0.8, 1.4]]),
        pressure=jnp.asarray([[2.0, 2.5], [1.5, 2.1]]),
        Er=jnp.asarray([0.3, -0.4]),
    )
    support = {
        "geometry": {"scale": jnp.asarray(1.3), "metric": jnp.asarray([0.7, 1.2])},
        "database": {"values": jnp.asarray([1.4, 2.3]), "coordinate": jnp.asarray(0.6)},
    }
    rows = jax.tree_util.tree_map(
        lambda value: jnp.stack((jnp.ones_like(value), 0.3 + value)), state
    )
    return state, support, rows


def _independent_support_bar(equations, state, rhs_bar, support):
    args = (0.0, state, None, rhs_bar, support)
    table = equations.pullback_direct_rhs_database_table_payload(*args)
    flux = equations.pullback_direct_rhs_database_flux_geometry_payload(*args)
    equation = equations.pullback_direct_rhs_database_equation_geometry_payload(*args)
    return {
        "database": table["database"],
        "geometry": jax.tree_util.tree_map(jnp.add, flux["geometry"], equation["geometry"]),
    }


def _assert_same_tree(actual, expected):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True
    ):
        assert jnp.all(jnp.isfinite(actual_leaf))
        assert jnp.allclose(actual_leaf, expected_leaf, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("preparation_mode", [None, "separate", "shared"])
def test_database_split_support_shared_primal_matches_independent_jit_vmap(preparation_mode):
    """Reusing values preserves all table/coordinate and native-face geometry rows."""
    calls = Counter()
    reference_calls = Counter()

    def _split(state, support, rhs_bar):
        return _equations(support, calls).pullback_direct_rhs_database_split_support_payload(
            0.0, state, None, rhs_bar, support,
            support_preparation_mode=preparation_mode,
        )

    def _independent(state, support, rhs_bar):
        return _independent_support_bar(_equations(support, reference_calls), state, rhs_bar, support)

    split = jax.jit(jax.vmap(_split, in_axes=(None, None, 0)))
    independent = jax.jit(jax.vmap(_independent, in_axes=(None, None, 0)))
    state, support, rows = _inputs()
    for factor in (1.0, 1.25):
        dynamic_state = jax.tree_util.tree_map(lambda value: factor * value, state)
        dynamic_support = jax.tree_util.tree_map(lambda value: factor * value, support)
        actual = split(dynamic_state, dynamic_support, rows)
        expected = independent(dynamic_state, dynamic_support, rows)
        _assert_same_tree(actual, expected)
        assert jnp.any(jnp.abs(actual["database"]["coordinate"]) > 0.0)
        assert jnp.any(jnp.abs(actual["geometry"]["metric"]) > 0.0)

    # Count trace-time construction, not compiled runtime work (XLA can CSE).
    for name in ("bind", "prepare", "centers", "capture", "density_faces", "temperature_faces"):
        assert calls[name] == (3 if preparation_mode == "separate" else 1)
        assert reference_calls[name] == 3
    assert calls["flux_vjp"] == (2 if preparation_mode == "separate" else 1)
    assert reference_calls["flux_vjp"] == 2


@pytest.mark.parametrize(
    "overridden_hook",
    [
        "pullback_direct_rhs_database_table_payload",
        "pullback_direct_rhs_database_flux_geometry_payload",
        "pullback_direct_rhs_database_equation_geometry_payload",
    ],
)
@pytest.mark.parametrize("preparation_mode", [None, "separate", "shared"])
def test_database_split_support_shared_primal_preserves_overridden_hooks(
    overridden_hook, preparation_mode
):
    """Concrete owners with public hook overrides keep their original dispatch."""
    state, support, rows = _inputs()
    rhs_bar = jax.tree_util.tree_map(lambda value: value[0], rows)
    calls = Counter()
    equations = _equations(support, calls)
    original = getattr(equations, overridden_hook)
    sentinel = 0.375
    field = "database" if overridden_hook.endswith("table_payload") else "geometry"

    # Deliberately accept only the established public signature.  An internal
    # prepared keyword must not be passed to third-party overrides.
    def _override(t, state_value, runtime, bar, payload):
        calls["override"] += 1
        result = original(t, state_value, runtime, bar, payload)
        result[field] = jax.tree_util.tree_map(lambda value: value + sentinel, result[field])
        return result

    object.__setattr__(equations, overridden_hook, _override)
    actual = equations.pullback_direct_rhs_database_split_support_payload(
        0.0, state, None, rhs_bar, support,
        support_preparation_mode=preparation_mode,
    )
    assert calls["override"] == 1
    assert calls["capture"] == 3
    expected = _independent_support_bar(equations, state, rhs_bar, support)
    _assert_same_tree(actual, expected)


@pytest.mark.parametrize("option", ["support_preparation_mode", "center_geometry_mode"])
def test_database_split_support_rejects_unknown_mode(option):
    state, support, rows = _inputs()
    rhs_bar = jax.tree_util.tree_map(lambda value: value[0], rows)
    equations = _equations(support, Counter())
    with pytest.raises(ValueError, match=option):
        equations.pullback_direct_rhs_database_split_support_payload(
            0.0, state, None, rhs_bar, support, **{option: "unknown"}
        )


@pytest.mark.parametrize("override_partial", [False, True])
def test_database_split_support_explicit_center_mode_rejects_legacy_hook(override_partial):
    """An explicit mode must not be silently ignored by an old public hook."""
    state, support, rows = _inputs()
    rhs_bar = jax.tree_util.tree_map(lambda value: value[0], rows)
    equations = _equations(support, Counter())
    if override_partial:
        original = equations.pullback_direct_rhs_database_flux_geometry_payload

        def _legacy(t, value, runtime, bar, payload):
            return original(t, value, runtime, bar, payload)

        object.__setattr__(equations, "pullback_direct_rhs_database_flux_geometry_payload", _legacy)
    # With no partial override, the model's legacy signature is the unsupported boundary.
    with pytest.raises(TypeError, match="center_geometry_mode.*keyword"):
        equations.pullback_direct_rhs_database_split_support_payload(
            0.0, state, None, rhs_bar, support, center_geometry_mode="radial_vjp"
        )


@pytest.mark.parametrize("center_mode", ["radial_vjp", "scalar_jvp"])
@pytest.mark.parametrize("preparation_mode", ["separate", "shared"])
def test_database_split_support_forwards_explicit_center_mode(
    monkeypatch, center_mode, preparation_mode
):
    """The option reaches the compact centre hook without changing face/table bars."""
    seen = []
    original = _CompactOwner.pullback_direct_rhs_geometry_by_radius

    def _selected(self, state, bar, geometry, *, center_geometry_mode=None):
        seen.append(center_geometry_mode)
        return original(self, state, bar, geometry)

    monkeypatch.setattr(_CompactOwner, "pullback_direct_rhs_geometry_by_radius", _selected)
    state, support, rows = _inputs()
    rhs_bar = jax.tree_util.tree_map(lambda value: value[0], rows)
    equations = _equations(support, Counter())
    expected = _independent_support_bar(equations, state, rhs_bar, support)
    seen.clear()
    actual = equations.pullback_direct_rhs_database_split_support_payload(
        0.0, state, None, rhs_bar, support,
        support_preparation_mode=preparation_mode, center_geometry_mode=center_mode,
    )
    assert seen == [center_mode]
    _assert_same_tree(actual, expected)


@pytest.mark.parametrize("center_mode", ["radial_vjp", "scalar_jvp"])
def test_database_generic_initial_support_forwards_center_mode(monkeypatch, center_mode):
    """The independent initial-support selector cannot hide the centre selector."""
    state, support, rows = _inputs()
    rhs_bar = jax.tree_util.tree_map(lambda value: value[0], rows)
    equations = _equations(support, Counter())
    seen = []

    def _flux_geometry(_self, _t, _state, _runtime, _bar, payload, *, center_geometry_mode=None):
        seen.append(center_geometry_mode)
        return {"geometry": jax.tree_util.tree_map(jnp.ones_like, payload["geometry"])}

    # Keep this a hook-wiring check; split-support equivalence is exercised above.
    monkeypatch.setattr(ComposedEquationSystem, "pullback_direct_rhs_database_flux_geometry_payload", _flux_geometry)
    object.__setattr__(equations, "pullback_shared_fluxes", lambda *_args: _center_fluxes(state, **support))
    actual = equations.pullback_direct_rhs_support_payload(
        0.0, state, None, rhs_bar, support, center_geometry_mode=center_mode
    )
    assert seen == [center_mode]
    assert set(actual) == {"geometry", "database"}
    assert all(jnp.all(jnp.isfinite(value)) for value in jax.tree_util.tree_leaves(actual))
