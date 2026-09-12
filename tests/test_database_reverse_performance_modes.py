"""Small numerical and dispatch checks for initial database support modes.

The reference differentiates a complete, repeated initial-stage payload.
No transport rollout, NTX scan, or VMEC solve is needed.
"""

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from NEOPAX import _reverse_ad_transport as reverse_transport
from NEOPAX._reverse_ad_transport import (
    _configure_database_reverse_performance,
    _initial_direct_rhs_support_pullback_batched,
)
from NEOPAX._state import TransportState
from NEOPAX._transport_solvers import (
    RADAUSolver,
    _RadauAcceptedStepPhysicsContext,
    _RadauAcceptedStepReducedCotangent,
    _build_prepared_radau_accepted_rollout,
)


def _fixture():
    carry = SimpleNamespace(t=jnp.asarray(1.5), y=jnp.asarray([2.0, 3.0]))
    stage_bars = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, -4.0], [-0.5, 1.0]],
            [[-2.0, 1.0], [5.0, -3.0], [0.25, -1.5]],
        ]
    )
    support = {
        "geometry": {
            "scale": jnp.asarray(0.8),
            "metric": jnp.asarray([0.3, -0.2]),
        },
        "database": {
            "values": jnp.asarray([1.1, -0.7]),
            "coordinate": jnp.asarray(0.25),
        },
    }
    return carry, stage_bars, support


def _rhs(t, y, support):
    geometry, database = support["geometry"], support["database"]
    scale, metric = geometry["scale"], geometry["metric"]
    values, coordinate = database["values"], database["coordinate"]
    return jnp.stack(
        (
            t * scale * y[0] ** 2
            + jnp.sin(values[0] * y[1])
            + metric[0] * coordinate**2,
            jnp.exp(scale * coordinate) * y[1]
            + jnp.dot(metric, values)
            + y[0] * values[1] ** 2,
        )
    )


def _generic_hook(t, y, rhs_bar, support):
    _, pullback = jax.vjp(lambda payload: _rhs(t, y, payload), support)
    return pullback(rhs_bar)[0]


def _split_hook(t, y, rhs_bar, support):
    result = {}
    for name in ("geometry", "database"):
        _, pullback = jax.vjp(
            lambda value: _rhs(t, y, {**support, name: value}), support[name]
        )
        result[name] = pullback(rhs_bar)[0]
    return result


def _unexpected_hook(*_args, **_kwargs):
    raise AssertionError("The unselected support hook must not be traced.")


def _reference(carry, stage_bars, support):
    def _initial_stages(payload):
        value = _rhs(carry.t, carry.y, payload)
        return jnp.broadcast_to(value, stage_bars.shape[1:])

    _, pullback = jax.vjp(_initial_stages, support)
    return jax.vmap(lambda objective_bar: pullback(objective_bar)[0])(stage_bars)


def _call(carry, stage_bars, support, *, generic=None, split=None, **options):
    return _initial_direct_rhs_support_pullback_batched(
        carry0=carry,
        carry0_bars=SimpleNamespace(prev_stages=stage_bars),
        kernel_context=SimpleNamespace(num_stages=stage_bars.shape[1]),
        flat_rhs_direct_support_pullback=generic,
        flat_rhs_direct_database_split_support_pullback=split,
        support_payload=support,
        **options,
    )


def _assert_same_finite_tree(actual, expected):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for value, reference in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True
    ):
        assert bool(jnp.all(jnp.isfinite(value)))
        np.testing.assert_allclose(value, reference, rtol=2.0e-6, atol=2.0e-6)


def test_generic_mode_uses_generic_hook_with_nonzero_stage_cotangents():
    carry, stage_bars, support = _fixture()
    expected = _reference(carry, stage_bars, support)
    actual = jax.jit(
        lambda payload: _call(
            carry,
            stage_bars,
            payload,
            generic=_generic_hook,
            split=_unexpected_hook,
            mode="generic",
        )
    )(support)
    _assert_same_finite_tree(actual, expected)
    assert any(bool(jnp.any(value != 0)) for value in jax.tree_util.tree_leaves(actual))


@pytest.mark.parametrize("mode", [None, "split"])
def test_split_mode_and_current_default_preserve_nonzero_support_cotangents(mode):
    carry, stage_bars, support = _fixture()
    options = {} if mode is None else {"mode": mode}
    actual = jax.jit(
        lambda payload: _call(
            carry,
            stage_bars,
            payload,
            generic=_unexpected_hook,
            split=_split_hook,
            **options,
        )
    )(support)
    _assert_same_finite_tree(actual, _reference(carry, stage_bars, support))


def test_split_mode_preserves_generic_fallback_when_split_hook_is_unavailable():
    carry, stage_bars, support = _fixture()
    actual = _call(carry, stage_bars, support, generic=_generic_hook, mode="split")
    _assert_same_finite_tree(actual, _reference(carry, stage_bars, support))


@pytest.mark.parametrize("mode", ["generic", "split"])
def test_live_support_keeps_ordinary_hook(mode):
    carry, stage_bars, _ = _fixture()
    actual = _call(
        carry,
        stage_bars,
        jnp.asarray(0.5),
        generic=lambda _t, _y, rhs_bar, support: support * jnp.sum(rhs_bar),
        split=_unexpected_hook,
        mode=mode,
    )
    np.testing.assert_allclose(actual, 0.5 * jnp.sum(stage_bars, axis=(1, 2)))


def test_reduced_zero_mode_returns_batched_support_zeros_without_tracing_hooks():
    carry, stage_bars, support = _fixture()
    stage_bars = jnp.zeros_like(stage_bars)
    actual = jax.jit(
        lambda payload: _call(
            carry,
            stage_bars,
            payload,
            generic=_unexpected_hook,
            split=_unexpected_hook,
            mode="reduced_zero",
            reduced_prev_stages_are_zero=True,
        )
    )(support)
    expected = jax.tree_util.tree_map(
        lambda value: jnp.zeros((stage_bars.shape[0],) + value.shape, dtype=value.dtype),
        support,
    )
    _assert_same_finite_tree(actual, expected)


@pytest.mark.parametrize("zero_stage_bars", [False, True])
def test_reduced_zero_mode_requires_structural_proof_even_without_hooks(zero_stage_bars):
    carry, stage_bars, support = _fixture()
    if zero_stage_bars:
        stage_bars = jnp.zeros_like(stage_bars)
    with pytest.raises(ValueError):
        _call(carry, stage_bars, support, mode="reduced_zero")


def test_reduced_zero_mode_rejects_live_support_even_with_structural_proof():
    carry, stage_bars, _ = _fixture()
    with pytest.raises(ValueError):
        _call(
            carry,
            jnp.zeros_like(stage_bars),
            jnp.asarray(0.5),
            mode="reduced_zero",
            reduced_prev_stages_are_zero=True,
        )


def test_invalid_mode_is_rejected_before_missing_hook_exit():
    carry, stage_bars, support = _fixture()
    with pytest.raises(ValueError):
        _call(carry, stage_bars, support, mode="unknown")


def _physics_context():
    def _unpack(value):
        return 2.0 * value

    _unpack.cotangent = lambda value: 5.0 * value
    return _RadauAcceptedStepPhysicsContext(
        unpack_flat=_unpack,
        pack_flat=lambda value: value,
        project_flat=lambda value: value + 1.0,
        build_lagged_response=_unexpected_hook,
        pullback_build_lagged_response=_unexpected_hook,
        flat_rhs=_unexpected_hook,
        flat_rhs_with_lagged_response=_unexpected_hook,
        flat_rhs_direct_support_pullback=_unexpected_hook,
        flat_rhs_direct_database_split_support_pullback=_unexpected_hook,
    )


class _SupportOwner:
    def __init__(self, species):
        self.species = species
        self.calls = []

    def __call__(self, *_args):
        raise AssertionError("Configuring reverse hooks must not evaluate the primal RHS.")

    def _result(self, t, state, species, rhs_bar, support):
        assert species is self.species
        scale = (t + jnp.sum(state)) * jnp.sum(rhs_bar)
        return jax.tree_util.tree_map(lambda value: value * scale, support)

    def pullback_direct_rhs_support_payload(
        self, t, state, species, *, rhs_bar, support, center_geometry_mode="scalar_jvp"
    ):
        self.calls.append(("generic", center_geometry_mode))
        return self._result(t, state, species, rhs_bar, support)

    def pullback_direct_rhs_database_split_support_payload(
        self,
        t,
        state,
        species,
        *,
        rhs_bar,
        support,
        center_geometry_mode="scalar_jvp",
        support_preparation_mode="shared",
    ):
        self.calls.append(("split", center_geometry_mode, support_preparation_mode))
        return self._result(t, state, species, rhs_bar, support)


def test_configure_current_defaults_preserves_context_and_callable_identities(monkeypatch):
    physics = _physics_context()
    monkeypatch.setattr(
        reverse_transport, "_flat_rhs_direct_support_pullback_factory", _unexpected_hook
    )
    monkeypatch.setattr(
        reverse_transport,
        "_flat_rhs_direct_database_payload_pullback_factory",
        _unexpected_hook,
    )
    actual = _configure_database_reverse_performance(
        physics, vector_field=_unexpected_hook, species=object()
    )
    assert actual is physics
    for field in dataclasses.fields(physics):
        value = getattr(physics, field.name)
        if callable(value):
            assert getattr(actual, field.name) is value


def test_configure_legacy_modes_rebinds_and_forwards_both_support_hooks():
    physics = _physics_context()
    species = object()
    owner = _SupportOwner(species)
    actual = _configure_database_reverse_performance(
        physics,
        vector_field=owner.__call__,
        species=species,
        initial_support_mode="generic",
        support_preparation_mode="separate",
        center_geometry_mode="radial_vjp",
    )
    assert actual is not physics
    assert actual.reverse_database_initial_support_mode == "generic"
    assert actual.reverse_database_support_preparation_mode == "separate"
    assert actual.reverse_database_center_geometry_mode == "radial_vjp"
    for name in (
        "flat_rhs",
        "flat_rhs_with_lagged_response",
        "build_lagged_response",
        "pullback_build_lagged_response",
        "unpack_flat",
        "pack_flat",
        "project_flat",
    ):
        assert getattr(actual, name) is getattr(physics, name)

    t, state, rhs_bar = jnp.asarray(1.5), jnp.asarray([2.0, 3.0]), jnp.asarray([1.0, -0.5])
    support = {"geometry": jnp.asarray(2.0), "database": jnp.asarray([3.0, -1.0])}
    # Projection and primal/cotangent unpacking are deliberately different.
    expected_scale = (1.5 + 2.0 * (3.0 + 4.0)) * (5.0 * (1.0 - 0.5))
    expected = jax.tree_util.tree_map(lambda value: value * expected_scale, support)
    generic = actual.flat_rhs_direct_support_pullback(t, state, rhs_bar, support)
    split = actual.flat_rhs_direct_database_split_support_pullback(t, state, rhs_bar, support)
    _assert_same_finite_tree(generic, expected)
    _assert_same_finite_tree(split, expected)
    assert owner.calls == [("generic", "radial_vjp"), ("split", "radial_vjp", "separate")]


@pytest.mark.parametrize(
    "option",
    ["initial_support_mode", "support_preparation_mode", "center_geometry_mode", "stage_jacobian_mode"],
)
def test_configure_rejects_unknown_modes(option):
    with pytest.raises(ValueError):
        _configure_database_reverse_performance(
            _physics_context(),
            vector_field=_unexpected_hook,
            species=object(),
            **{option: "unknown"},
        )


def test_configure_nondefault_modes_require_database_support_capability():
    physics = dataclasses.replace(
        _physics_context(), flat_rhs_direct_database_split_support_pullback=None
    )
    with pytest.raises(ValueError):
        _configure_database_reverse_performance(
            physics,
            vector_field=_unexpected_hook,
            species=object(),
            initial_support_mode="generic",
        )


def test_reduced_zero_preserves_real_initial_carry_root_and_profile_pullbacks():
    """Skip only predictor support, preserving the composed physical gradient.

    This builds a real Radau initial carry and uses its production custom VJP,
    but never takes a transport step. Two analytic positive roots provide an
    independent reference for the subsequent implicit state/support rules.
    """
    profile = jnp.asarray([2.0, 1.5])
    support = {"geometry": jnp.asarray(0.5), "database": jnp.asarray(0.7)}

    def _pre_root(values, payload):
        return TransportState(
            density=(values[0] + jnp.asarray([0.1, 0.3]) * payload["geometry"])[None, :],
            pressure=(values[1] ** 2 + jnp.asarray([0.2, 0.4]) * payload["geometry"])[None, :],
            Er=jnp.zeros((2,), dtype=values.dtype),
        )

    def _root_target(state, payload):
        return (
            state.density[0]
            + state.pressure[0] ** 2
            + payload["geometry"] * payload["database"]
        )

    def _rooted(values, payload):
        state = _pre_root(values, payload)
        return dataclasses.replace(state, Er=jnp.sqrt(_root_target(state, payload)))

    class _PolynomialOwner:
        @staticmethod
        def rhs(state, payload):
            return TransportState(
                density=-0.1 * state.density + payload["database"] * state.pressure,
                pressure=-0.2 * state.pressure + payload["geometry"] * state.density**2,
                Er=-0.3 * state.Er + payload["geometry"] * payload["database"],
            )

        def __call__(self, _t, state, *_args):
            return self.rhs(state, support)

    owner = _PolynomialOwner()
    state = _rooted(profile, support)
    pre_root = _pre_root(profile, support)
    solver = RADAUSolver(t0=0.0, t1=1.0e-3, dt=1.0e-4, rhs_mode="black_box")
    prepared = _build_prepared_radau_accepted_rollout(
        solver=solver, state=state, vector_field=owner.__call__, species=None
    )

    def _initial_carry(state_value):
        return reverse_transport.reverse_initial_carry_from_state_with_static_setup(
            solver=solver,
            state=state_value,
            solve_vector_field=owner.__call__,
            species=None,
            prepared_rollout_static=prepared,
        )

    carry, initial_state_pullback = jax.vjp(_initial_carry, state)
    y_seeds = jnp.stack(
        (jnp.arange(1, carry.y.size + 1, dtype=carry.y.dtype),
         -jnp.arange(2, carry.y.size + 2, dtype=carry.y.dtype))
    )

    def _zero_tangent(value):
        value = jnp.asarray(value)
        dtype = value.dtype if jnp.issubdtype(value.dtype, jnp.inexact) else jax.dtypes.float0
        return jnp.zeros(value.shape, dtype=dtype)

    # Execute the actual production expansion, not a copied version of its
    # invariant. This catches a future change that restores predictor bars.
    source = Path(reverse_transport.__file__)
    module = ast.parse(source.read_text(encoding="utf-8"))
    expansion_nodes = [
        node for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "_full_carry_bar_from_reduced"
    ]
    assert len(expansion_nodes) == 1
    namespace = {"jax": jax, "dataclasses": dataclasses, "carry0": carry,
                 "_zero_tangent_like": _zero_tangent}
    exec(compile(ast.Module(body=expansion_nodes, type_ignores=[]), str(source), "exec"), namespace)
    production_expand = namespace["_full_carry_bar_from_reduced"]

    def _expand(seed):
        reduced = _RadauAcceptedStepReducedCotangent(
            y=seed, lagged_response_cache=None, lagged_reference_y=0.25 * seed
        )
        return production_expand(reduced)

    carry_bars = jax.vmap(_expand)(y_seeds)
    assert bool(jnp.all(carry_bars.prev_stages == 0))
    assert bool(jnp.all(carry_bars.y != 0))
    assert bool(jnp.all(carry_bars.lagged_reference_y != 0))

    def _fixed_support_pullback(_t, flat_y, rhs_bar, payload):
        state_value = prepared.physics_context.unpack_flat(flat_y)
        _, pullback = jax.vjp(
            lambda value: prepared.physics_context.pack_flat(owner.rhs(state_value, value)),
            payload,
        )
        return pullback(rhs_bar)[0]

    _, profile_pullback = jax.vjp(_pre_root, profile, support)

    def _one_root_and_profile(state_bar):
        root_state_bar = jax.tree_util.tree_map(jnp.zeros_like, pre_root)
        root_support_bar = jax.tree_util.tree_map(jnp.zeros_like, support)
        for radius in range(2):
            def _residual(state_value, root_value, payload):
                return root_value**2 - _root_target(state_value, payload)[radius]

            contribution = reverse_transport.implicit_scalar_root_state_pullback(
                lambda state_value, root_value: _residual(state_value, root_value, support),
                pre_root,
                state.Er[radius],
                state_bar.Er[radius],
            )
            support_contribution = reverse_transport.implicit_scalar_root_support_pullback(
                _residual, pre_root, state.Er[radius], state_bar.Er[radius], support
            )
            root_state_bar = reverse_transport.add_trees(root_state_bar, contribution)
            root_support_bar = reverse_transport.add_trees(root_support_bar, support_contribution)
        pre_root_bar = reverse_transport.add_trees(
            dataclasses.replace(state_bar, Er=jnp.zeros_like(state_bar.Er)), root_state_bar
        )
        profile_bar, initial_profile_support_bar = profile_pullback(pre_root_bar)
        return (
            profile_bar,
            root_support_bar,
            reverse_transport.add_trees(root_support_bar, initial_profile_support_bar),
        )

    results = {}
    for mode in ("split", "reduced_zero"):
        direct_support_bars = _initial_direct_rhs_support_pullback_batched(
            carry0=carry,
            carry0_bars=carry_bars,
            kernel_context=prepared.kernel_context,
            flat_rhs_direct_support_pullback=_unexpected_hook,
            flat_rhs_direct_database_split_support_pullback=(
                _fixed_support_pullback if mode == "split" else _unexpected_hook
            ),
            support_payload=support,
            mode=mode,
            reduced_prev_stages_are_zero=True,
        )
        state_bars = jax.vmap(lambda bar: initial_state_pullback(bar)[0])(carry_bars)
        profile_bars, root_support_bars, remaining_support_bars = jax.vmap(
            _one_root_and_profile
        )(state_bars)
        total_support_bars = reverse_transport.add_trees(
            direct_support_bars, remaining_support_bars
        )
        assert bool(jnp.all(state_bars.Er != 0))
        assert bool(jnp.all(profile_bars != 0))
        assert all(bool(jnp.all(value != 0)) for value in jax.tree_util.tree_leaves(root_support_bars))
        results[mode] = (state_bars, profile_bars, root_support_bars, total_support_bars)

    _assert_same_finite_tree(results["reduced_zero"], results["split"])

    def _objectives(values, payload):
        return 1.25 * (y_seeds @ prepared.physics_context.pack_flat(_rooted(values, payload)))

    reference_profile, reference_support = jax.jacrev(_objectives, argnums=(0, 1))(
        profile, support
    )
    _assert_same_finite_tree(results["reduced_zero"][1], reference_profile)
    _assert_same_finite_tree(results["reduced_zero"][3], reference_support)
