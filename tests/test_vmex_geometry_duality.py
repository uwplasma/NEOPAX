"""Common geometry-response diagnostic; tiny exact algebra, no VMEX solves."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_common_geometry_check_compares_same_state_mask_and_boundary_directions():
    path = Path(__file__).resolve().parents[1] / 'examples/benchmarks/check_vmex_geometry_duality.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'compare_geometry_duality')
    # Two independent boundary directions, three state cotangents.
    tangent_map = jnp.asarray([[2., -1.], [3., 4.]])
    bars = {'y': jnp.asarray([[.2, .7], [1., -.3], [-.4, .8]])}
    entries = ({'column': 0}, {'column': 1})
    raw = SimpleNamespace(implicit_params=object(), implicit_cfg=object(), state=object(),
                          dof_mask=object(), param_entries=entries)
    tangent_fn = Mock(side_effect=lambda params, cfg, state, mask, d, **kw: {'y': tangent_map @ d})
    raw.implicit = SimpleNamespace(implicit_state_tangent_raw_block=tangent_fn)
    transpose = Mock(return_value=bars['y'] @ tangent_map)
    ns = dict(jax=jax, jnp=jnp, np=np,
              geometry_raw_block_transpose_from_state_bars=transpose,
              _param_unit_tangent_like=lambda params, entry: jnp.eye(2)[entry['column']])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), ns)
    rows = ns['compare_geometry_duality'](raw, bars)
    assert len(rows) == 6 and max(row['relative_error'] for row in rows) < 1e-14
    transpose.assert_called_once_with(raw, bars)
    assert tangent_fn.call_count == 2
    for call in tangent_fn.call_args_list:
        assert all(a is b for a, b in zip(call.args[:4],
                   (raw.implicit_params, raw.implicit_cfg, raw.state, raw.dof_mask), strict=True))
        assert call.kwargs == {'probe_chunk_size': 1}
    transpose.return_value = bars['y'] @ tangent_map + 1.
    assert max(row['absolute_error'] for row in ns['compare_geometry_duality'](raw, bars)) > .9
    transpose.return_value = jnp.full((3, 2), jnp.nan)
    with pytest.raises(FloatingPointError, match='Nonfinite geometry response'):
        ns['compare_geometry_duality'](raw, bars)


def test_main_response_comparison_distinguishes_changed_map_from_broken_transpose():
    path = Path(__file__).resolve().parents[1] / 'examples/benchmarks/check_vmex_geometry_duality.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'compare_geometry_duality')
    legacy = jnp.array([[2., -1.], [3., 4.]])
    certified = legacy + jnp.array([[.2, -.3], [.1, .4]])
    bars = {'y': jnp.array([[.2, .7], [1., -.3], [-.4, .8]])}
    entries = tuple(dict(input_field='boundary', n_offset=0, m_index=i, column=i) for i in range(2))
    state, mask, cfg, params = (object() for _ in range(4))

    def forward(p, c, x, m, directions, **kw):
        assert (p, c, x, m) == (params, cfg, state, mask)
        assert kw == dict(probe_chunk_size=1, response_chunk_size=1)
        return {'y': directions @ certified.T}, SimpleNamespace(converged=jnp.array([True]))

    def reverse(p, c, x, m, cotangents, **kw):
        assert (p, c, x, m) == (params, cfg, state, mask)
        assert kw == dict(solver='block', probe_chunk_size=1, response_chunk_size=1)
        return SimpleNamespace(boundary=(cotangents['y'] @ certified)[:, None, :])

    im = SimpleNamespace(
        implicit_state_tangent_raw_block=lambda p, c, x, m, d, **kw: {'y': legacy @ d},
        implicit_state_tangent_multi_rhs=forward,
        implicit_state_pullback_multi_rhs=reverse,
    )
    raw = SimpleNamespace(implicit=im, implicit_params=params, implicit_cfg=cfg,
                          state=state, dof_mask=mask, param_entries=entries)
    ns = dict(jax=jax, jnp=jnp, np=np,
              geometry_raw_block_transpose_from_state_bars=lambda raw, b: b['y'] @ legacy,
              _param_unit_tangent_like=lambda p, entry: jnp.eye(2)[entry['column']])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), ns)
    rows = ns['compare_geometry_duality'](raw, bars)
    assert len(rows) == 12 and max(row['absolute_error'] for row in rows) < 1e-14
    # Different maps can each have a correct transpose. A changed gradient
    # alone must not be reported as proof that the new map is incorrect.
    assert rows[0]['tangent_contraction'] != rows[6]['tangent_contraction']
    im.implicit_state_pullback_multi_rhs = lambda *args, **kw: SimpleNamespace(
        boundary=(bars['y'] @ certified + 1.)[:, None, :])
    rows = ns['compare_geometry_duality'](raw, bars)
    assert max(row['absolute_error'] for row in rows[:6]) < 1e-14
    assert min(row['absolute_error'] for row in rows[6:]) > .99
