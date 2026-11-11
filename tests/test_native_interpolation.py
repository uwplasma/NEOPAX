"""Small-array comparisons with Interpax; no equilibrium/transport solves."""

import dataclasses

import interpax
import jax
import jax.numpy as jnp
import numpy as np

from NEOPAX import _interpolators as interpolation
from NEOPAX._database import Monoenergetic
from NEOPAX._monoenergetic import database_with_geometry_scale
from NEOPAX._native_interpolation import (
    _cell_upper_index, bicubic_channels, cubic1d, tricubic_channels, contract_scalar_queries,
)

jax.config.update("jax_enable_x64", True)


def _assert_close(actual, expected, *, rtol=5e-11, atol=5e-11):
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(left, right, rtol=rtol, atol=atol)


def _reference(x, y, tables, xq, yq):
    return jnp.stack(tuple(interpax.Interpolator2D(x, y, table, extrap=True)(xq, yq) for table in tables))


def _native(x, y, tables, xq, yq):
    return jnp.stack(bicubic_channels(x, y, tuple(tables), xq, yq))


def _surface_data():
    x = jnp.asarray([-3., -1.6, -.5, .7, 2.])
    y = jnp.asarray([-4.8, -2.7, -1.1, .4, 1.8])
    values = jnp.asarray(np.random.default_rng(42).normal(size=(3, x.size, y.size)))
    return x, y, values


def test_cell_lookup_matches_searchsorted_at_knots_and_extreme_queries():
    near_uniform = np.linspace(-2., 2., 17) + .05 * np.sin(np.linspace(0., 2. * np.pi, 17))
    for knots in (np.asarray([-.3, 1.7]), near_uniform, np.asarray([-2., -.7, -.7, .1, 2.]),
                  np.ones(5), np.linspace(0., 2., 129) ** 2, 1e20 + np.linspace(0., 65536., 17)):
        queries = np.concatenate((knots, np.nextafter(knots, -np.inf), np.nextafter(knots, np.inf),
                                  np.linspace(knots[0] - .3, knots[-1] + .3, 200),
                                  [-1e100, 1e100, -np.inf, np.inf, np.nan]))
        actual = jax.jit(jax.vmap(lambda q: _cell_upper_index(jnp.asarray(knots), q)))(jnp.asarray(queries))
        np.testing.assert_array_equal(actual, np.clip(np.searchsorted(knots, queries, side="right"), 1, len(knots) - 1))


def test_cubic1d_values_query_derivatives_and_extrapolation():
    for knots in (jnp.asarray([-.3, 1.7]), jnp.asarray([-2., -.7, .1, .6, 2.])):
        values = jnp.sin(2. * knots)
        queries = jnp.concatenate((knots, knots[:1] - .4, knots[-1:] + .3, jnp.asarray([.37])))
        for extrap in (False, True):
            native = lambda q: cubic1d(knots, values, q, extrap=extrap)
            reference = interpax.Interpolator1D(knots, values, extrap=extrap)
            _assert_close(jax.vmap(native)(queries), jax.vmap(reference)(queries))
            _assert_close(jax.vmap(jax.grad(native))(queries), jax.vmap(jax.grad(reference))(queries))


def test_bicubic_values_and_vjp_for_queries_knots_and_all_channels():
    x, y, values = _surface_data()
    queries = jnp.asarray([[-3.2, -5.1], [-3., -4.8], [-.5, -1.1], [-.23, -.67], [2.3, 2.2]])
    cotangent = jnp.asarray([.3, -.7, 1.1])
    for evaluate in (_native, _reference):
        objective = lambda a, b, c, q, r: jnp.vdot(cotangent, evaluate(a, b, c, q, r))
        result = jax.vmap(jax.value_and_grad(objective, argnums=(0, 1, 2, 3, 4)),
                          in_axes=(None, None, None, 0, 0))(x, y, values, queries[:, 0], queries[:, 1])
        if evaluate is _native:
            actual = result
        else:
            _assert_close(actual, result)


def test_bicubic_jvp_hessian_and_finite_difference():
    x, y, values = _surface_data()
    point = jnp.asarray([-.23, -.67, -1.6, -2.7])
    direction = jnp.asarray([.2, -.3, .1, -.15])
    functions = tuple(
        lambda p, evaluate=evaluate: jnp.sum(evaluate(
            x.at[1].set(p[2]), y.at[1].set(p[3]), values, p[0], p[1]) ** 2)
        for evaluate in (_native, _reference)
    )
    _assert_close(jax.jvp(functions[0], (point,), (direction,)),
                  jax.jvp(functions[1], (point,), (direction,)))
    _assert_close(jax.hessian(functions[0])(point), jax.hessian(functions[1])(point))
    derivative = jax.jvp(functions[0], (point,), (direction,))[1]
    fd = (functions[0](point + 1e-5 * direction) - functions[0](point - 1e-5 * direction)) / 2e-5
    _assert_close(derivative, fd, rtol=2e-8, atol=2e-8)
    joint_native = contract_scalar_queries(_native, argnums=(3, 4), joint=True)
    joint_objective = lambda p: jnp.sum(joint_native(
        x.at[1].set(p[2]), y.at[1].set(p[3]), values, p[0], p[1]) ** 2)
    _assert_close(jax.jvp(joint_objective, (point,), (direction,)),
                  jax.jvp(functions[1], (point,), (direction,)))
    _assert_close(jax.hessian(joint_objective)(point), jax.hessian(functions[1])(point))


def test_bicubic_two_knot_axes_and_repeated_knot_gradients():
    for x in (jnp.asarray([-2., 1.]), jnp.asarray([-3., -1., -1., 0., 2.])):
        y = jnp.asarray([-2., 1.])
        table = jnp.arange(x.size * y.size, dtype=jnp.float64).reshape(x.size, y.size)
        query = jnp.asarray([-.3, .2])
        actual = _native(x, y, table[None], *query)
        _assert_close(actual, _reference(x, y, table[None], *query))
        bars = jax.grad(lambda knots: jnp.sum(_native(knots, y, table[None], *query)))(x)
        assert np.isfinite(bars).all()


def test_bicubic_batched_state_jacobian_matches_reference_for_query_and_data_directions():
    x, y, values = _surface_data()
    point = jnp.linspace(-.03, .04, 9)
    query_directions = jnp.asarray(np.random.default_rng(19).normal(size=(2, 9)))
    for vary_queries, vary_data in ((True, False), (False, True), (True, True)):
        def evaluate(p, surface):
            queries = jnp.asarray([-.23, -.67])
            if vary_queries:
                queries = queries + query_directions @ p
            knots_x, knots_y, tables = x, y, values
            if vary_data:
                knots_x = x.at[1].add(p[0])
                knots_y = y.at[2].add(p[1])
                tables = values.at[0, 2, 2].add(p[2])
            return surface(knots_x, knots_y, tables, *queries)

        def gradients(p, surface):
            return jax.jacfwd(lambda value: evaluate(value, surface))(p)

        points = jnp.stack((point, point * .8, point * 1.1))
        actual = jax.jit(jax.vmap(lambda p: gradients(p, _native)))(points)
        expected = jax.jit(jax.vmap(lambda p: gradients(p, _reference)))(points)
        _assert_close(actual, expected)


def test_tricubic_values_data_vjp_and_mixed_hessian_match_interpax():
    x, y, _ = _surface_data()
    z = jnp.asarray([-1.3, -.1, .8, 2.1])
    tables = jnp.asarray(np.random.default_rng(31).normal(size=(2, x.size, y.size, z.size)))

    def native(a, b, c, values, q):
        return jnp.stack(tricubic_channels(a, b, c, tuple(values), *q))

    def reference(a, b, c, values, q):
        return jnp.stack(tuple(interpax.Interpolator3D(a, b, c, value, extrap=True)(*q)
                               for value in values))

    for point in ([-3.2, -5., -1.5], [-.23, -.67, .31], [2.2, 2., 2.4]):
        point = jnp.asarray(point)
        results = []
        for evaluate in (native, reference):
            objective = lambda a, b, c, values, q: jnp.vdot(
                jnp.asarray([.3, -.7]), evaluate(a, b, c, values, q))
            results.append(jax.value_and_grad(objective, argnums=(0, 1, 2, 3, 4))(
                x, y, z, tables, point))
        _assert_close(*results)

    point = jnp.asarray([-.23, -.67, .31, .02, .01])
    def objective(p, evaluate):
        values = tables.at[0, 1, 2, 1].add(p[3])
        return jnp.sum(evaluate(x.at[1].add(p[4]), y, z, values, p[:3]) ** 2)
    _assert_close(jax.hessian(lambda p: objective(p, native))(point),
                  jax.hessian(lambda p: objective(p, reference))(point))

    # Integer coordinate metadata must supply float0 zeros when a table is
    # differentiated; numerical zeros with the integer primal dtype are invalid.
    knots = jnp.arange(3, dtype=jnp.int32)
    cube = jnp.arange(27., dtype=jnp.float64).reshape((3, 3, 3))
    gradient = jax.grad(lambda value: tricubic_channels(
        knots, knots, knots, (value,), .7, .8, .9)[0])(cube)
    assert np.isfinite(gradient).all()
    _assert_close(jnp.sum(gradient), jnp.asarray(1.))


def test_fixed_ntss_interpolator_rejects_wrong_field_node_count():
    import inspect
    import pytest
    from types import SimpleNamespace
    from NEOPAX._interpolators_preprocessed import get_Dij_preprocessed_3d_ntss_radius_ntss1d_fixednu

    original = inspect.unwrap(get_Dij_preprocessed_3d_ntss_radius_ntss1d_fixednu)
    for count in (11, 13):
        database = SimpleNamespace(Er_grid=jnp.ones((5, count)))
        with pytest.raises(ValueError, match="exactly 12 electric-field nodes"):
            original(.5, .001, .001, database)


def _database():
    rho = jnp.asarray([.1, .3, .5, .7, .9])
    x, y, _ = _surface_data()
    shape = (rho.size, x.size, y.size)
    tables = jnp.asarray(np.random.default_rng(9).normal(scale=.03, size=(3, *shape)))
    return Monoenergetic(a_b=jnp.asarray(1.3), rho=rho, nu_log=x,
                         Er_list=y[None, :] - .2 * rho[:, None],
                         D11_log=tables[0] - 3., D13=tables[1] + .2, D33=tables[2] + .4)


def test_database_radial_branches_floors_and_transposes(monkeypatch):
    database = _database()
    points = ((0., 0., 0.), (.12, .02, .003), (.65, .02, -.003),
              (1.2, .02, .003), (.65, 1e-14, 0.), (.65, 1e3, 10.))
    native_values = [interpolation.get_Dij(*map(jnp.asarray, point), database) for point in points]

    def reference_surface(index, nu, er, db):
        return tuple(_reference(db.nu_log, db.Er_list[index],
                                (db.D11_log[index], db.D13[index], db.D33[index]), nu, er))

    # Disable jit for the reference so changing this surface evaluator cannot
    # reuse a compiled native radial query and accidentally compare it to itself.
    with monkeypatch.context() as context, jax.disable_jit():
        context.setattr(interpolation, "interpolator_nu_Er_general", reference_surface)
        reference_values = [interpolation.get_Dij(*map(jnp.asarray, point), database) for point in points]
    _assert_close(native_values, reference_values)

    bar = jnp.asarray([.2, -.4, .7])
    for radius in (.12, .65, 1.2):
        def objective(scale, er_grid, table):
            db = dataclasses.replace(database_with_geometry_scale(database, scale),
                                     Er_list=er_grid, D13=table)
            return jnp.vdot(bar, interpolation.get_Dij(jnp.asarray(radius), .02, .003, db))

        actual = jax.grad(objective, argnums=(0, 1, 2))(database.a_b, database.Er_list, database.D13)
        with monkeypatch.context() as context, jax.disable_jit():
            context.setattr(interpolation, "interpolator_nu_Er_general", reference_surface)
            expected = jax.grad(objective, argnums=(0, 1, 2))(database.a_b, database.Er_list, database.D13)
        _assert_close(actual, expected, rtol=5e-10, atol=5e-10)
        table_bar = interpolation.monoenergetic_interpolation_table_bar(
            jnp.asarray(radius), jnp.asarray(.02), jnp.asarray(.003), bar[1], database.D13, database)
        _assert_close(actual[2], table_bar)


def test_coordinate_pullback_reuses_compilation_for_changed_arrays():
    from types import SimpleNamespace
    from examples.benchmarks.benchmark_initial_er_coordinate_pullback import build_case
    from NEOPAX._neoclassical import _particle_flux_coordinate_basis
    from NEOPAX._transport_flux_models import (
        _contract_database_particle_flux_coordinates,
        _database_particle_flux_coordinate_queries,
    )

    model, state, bars = build_case(4, 2, small=True)
    # Use the same explicit coordinate dtypes as the rebuilt runtime database;
    # Python weak scalars and explicitly typed arrays have different JIT keys.
    model = dataclasses.replace(model, database=database_with_geometry_scale(model.database, model.database.a_b))

    def compare(current_model, current_state, current_bars):
        actual = current_model._pullback_local_particle_flux_database_coordinates(
            current_state, current_bars, current_model.database)
        expected = current_model._local_particle_flux_database_coordinate_bars(
            current_state, current_bars, current_model.database)
        _assert_close(actual, expected, rtol=3e-10, atol=3e-10)
        assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(actual))
        return actual

    def compiled_counts():
        return tuple(function._cache_size() for function in (
            _database_particle_flux_coordinate_queries, _particle_flux_coordinate_basis,
            _contract_database_particle_flux_coordinates))

    compare(model, state, bars)
    compiled_count = compiled_counts()
    geometry = model.geometry._replace(
        a_b=model.geometry.a_b * 1.01, r_grid=model.geometry.r_grid * 1.01,
        r_grid_half=model.geometry.r_grid_half * 1.01, dr=model.geometry.dr * 1.01)
    database = dataclasses.replace(
        database_with_geometry_scale(model.database, model.database.a_b * 1.01),
        Er_list=model.database.Er_list + .003, D11_log=model.database.D11_log + .002)
    changed_model = dataclasses.replace(
        model, geometry=geometry, database=database, density_floor=.8,
        species=dataclasses.replace(model.species, mass_mp=model.species.mass_mp * 1.001),
        energy_grid=model.energy_grid._replace(xWeights=model.energy_grid.xWeights * .99),
        bc_density=dataclasses.replace(model.bc_density, right_value=model.bc_density.right_value * .97),
        bc_temperature=dataclasses.replace(
            model.bc_temperature, right_decay_length=model.bc_temperature.right_decay_length * .9))
    changed_state = dataclasses.replace(
        state, density=state.density * 1.03, pressure=state.pressure * .96, Er=state.Er + .01)
    updated_bars = bars * .8
    batched = compare(changed_model, changed_state, updated_bars)
    assert compiled_counts() == compiled_count
    scalar = compare(changed_model, changed_state, updated_bars[0])
    _assert_close((scalar.a_b, scalar.Er_list), (batched.a_b[0], batched.Er_list[0]))
    assert compiled_counts()[:2] == compiled_count[:2]
    for kind in ("ntss_like", "ntss_legacy", "first_principles", "ntss_zeff"):
        compare(dataclasses.replace(model, collisionality_model=kind), state, bars)

    def pressure_direction(step, original):
        varied = dataclasses.replace(state, pressure=state.pressure * (1. + step))
        evaluate = (model._local_particle_flux_database_coordinate_bars if original
                    else model._pullback_local_particle_flux_database_coordinates)
        result = evaluate(varied, bars, model.database)
        return jnp.concatenate((jnp.ravel(result.a_b), jnp.ravel(result.Er_list)))

    _assert_close(jax.jvp(lambda p: pressure_direction(p, False), (jnp.asarray(0.),), (jnp.asarray(1.),)),
                  jax.jvp(lambda p: pressure_direction(p, True), (jnp.asarray(0.),), (jnp.asarray(1.),)),
                  rtol=2e-8, atol=2e-8)
    custom_bc = SimpleNamespace(right_type="neumann", right_gradient=jnp.zeros((2,)))
    compare(dataclasses.replace(model, bc_density=custom_bc), state, bars)

    class CustomModel(type(model)):
        pass

    compare(CustomModel(**vars(model)), state, bars)
    from NEOPAX._database import Monoenergetic3D, MonoenergeticLogErNoR
    for database_type in (Monoenergetic3D, MonoenergeticLogErNoR):
        compare(dataclasses.replace(model, database=database_type(**vars(model.database))), state, bars)
    other_database = {"table": jnp.ones((2, 3))}
    fallback = model._pullback_local_particle_flux_database_coordinates(state, bars, other_database)
    np.testing.assert_array_equal(fallback["table"], np.zeros((2, 3)))
