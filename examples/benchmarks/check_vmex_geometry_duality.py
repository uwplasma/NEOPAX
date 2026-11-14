"""Check the shared geometry tangent/adjoint at the benchmark equilibrium.

Runs one equilibrium solve and its linear responses on the GPU. There is no
NTX scan, ambipolar root solve or transport simulation. The check covers the
VMEX link shared by every geometry derivative, including Er and bootstrap.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
from pathlib import Path
import sys
import tomllib

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_transport_reverse_ad_only import (  # noqa: E402
    _baseline_geometry_delta_vector_for_specs,
    _geometry_context_from_config,
    _geometry_param_specs_from_parameter_name,
)
from benchmark_transport_realtime_geometry_forward_fd import _param_unit_tangent_like  # noqa: E402
from NEOPAX._geometry_autodiff import (  # noqa: E402
    geometry_raw_block_solve_from_param_vector,
    geometry_raw_block_transpose_from_state_bars,
)

DEFAULT_CONFIG = ROOT / 'examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml'


def compare_geometry_duality(raw, bars):
    """Compare raw and main-certified responses at exactly the same point."""
    methods = [("neopax_raw_block", None)]
    if hasattr(raw.implicit, 'implicit_state_tangent_multi_rhs'):
        methods.append(("main_certified_block", "block"))
    rows = []
    for method, certified in methods:
        if certified is None:
            adjoint = geometry_raw_block_transpose_from_state_bars(raw, bars)
        else:
            param_bars = raw.implicit.implicit_state_pullback_multi_rhs(
                raw.implicit_params, raw.implicit_cfg, raw.state, raw.dof_mask, bars,
                solver=certified, probe_chunk_size=1, response_chunk_size=1,
            )
            adjoint = jnp.stack([getattr(param_bars, entry['input_field'])[
                :, entry['n_offset'], entry['m_index']] for entry in raw.param_entries], axis=1)
        adjoint = np.asarray(jax.device_get(adjoint))
        for column, entry in enumerate(raw.param_entries):
            direction = _param_unit_tangent_like(raw.implicit_params, entry)
            if certified is None:
                tangent = raw.implicit.implicit_state_tangent_raw_block(
                    raw.implicit_params, raw.implicit_cfg, raw.state, raw.dof_mask,
                    direction, probe_chunk_size=1,
                )
            else:
                tangent, report = raw.implicit.implicit_state_tangent_multi_rhs(
                    raw.implicit_params, raw.implicit_cfg, raw.state, raw.dof_mask,
                    jax.tree.map(lambda value: value[None], direction),
                    probe_chunk_size=1, response_chunk_size=1,
                )
                if not bool(np.all(np.asarray(jax.device_get(report.converged)))):
                    raise FloatingPointError(f'Unconverged {method} tangent, column {column}.')
                tangent = jax.tree.map(lambda value: value[0], tangent)
            products = [jnp.sum(b * t[None, ...], axis=tuple(range(1, b.ndim)))
                        for b, t in zip(jax.tree_util.tree_leaves(bars),
                                        jax.tree_util.tree_leaves(tangent), strict=True)]
            forward = np.asarray(jax.device_get(sum(products)))
            reverse = adjoint[:, column]
            if not np.isfinite(forward).all() or not np.isfinite(reverse).all():
                raise FloatingPointError(f'Nonfinite geometry response ({method}) in parameter column {column}.')
            for row, (left, right) in enumerate(zip(forward, reverse, strict=True)):
                absolute = float(abs(left-right))
                rows.append(dict(response_method=method, parameter_column=column, cotangent_row=row,
                                 tangent_contraction=float(left), adjoint=float(right),
                                 absolute_error=absolute,
                                 relative_error=absolute/abs(float(left)) if left != 0 else None))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--parameters', default='RBC:1:0,ZBS:1:0')
    parser.add_argument('--bar-count', type=int, default=3)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'outputs/vmex_geometry_duality')
    args = parser.parse_args(argv)
    if args.bar_count <= 0:
        parser.error('--bar-count must be positive.')
    if jax.default_backend() != 'gpu':
        parser.error('This diagnostic requires a GPU; CPU execution is disabled.')
    if not jax.config.x64_enabled:
        parser.error('Set JAX_ENABLE_X64=true.')
    with args.config.open('rb') as stream:
        config = tomllib.load(stream)
    geometry = config['geometry']
    if str(geometry.get('vmec_lane', '')).lower() != 'ad':
        parser.error('Use the current benchmark TOML with geometry.vmec_lane="ad".')
    input_path = Path(geometry['vmec_input_file'])
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    geometry['vmec_input_file'] = str(input_path.resolve())
    specs = _geometry_param_specs_from_parameter_name(args.parameters)
    context = _geometry_context_from_config(config, args.parameters)
    deltas = _baseline_geometry_delta_vector_for_specs(geometry, specs)
    print(f'[geometry-duality] JAX={jax.__version__} device={jax.devices()[0]} '
          f'input_sha256={hashlib.sha256(input_path.read_bytes()).hexdigest()}', flush=True)
    print('[geometry-duality] ONE benchmark equilibrium; no NTX/root/transport solve', flush=True)
    raw = geometry_raw_block_solve_from_param_vector(
        context, deltas, specs, max_iter=geometry.get('vmec_max_iter'),
        solver_device=str(geometry.get('vmec_implicit_solver_device', 'default')),
    )
    bars = jax.tree_util.tree_map(
        lambda value: jnp.stack([
            jnp.sin(jnp.arange(value.size, dtype=value.dtype).reshape(value.shape)
                    * (.37 + row * .13) + .71) / jnp.sqrt(float(value.size))
            for row in range(args.bar_count)]), raw.state,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Save the actual linearization point for further checks without repeating
    # the equilibrium. Only array-valued dataclass fields are serialized.
    arrays = dict(parameter_labels=np.asarray([f'{f}:{m}:{n}' for f, m, n in specs]),
                  input_sha256=np.asarray(hashlib.sha256(input_path.read_bytes()).hexdigest()))
    for prefix, tree in (('state', raw.state), ('params', raw.implicit_params),
                         ('mask', raw.dof_mask), ('bars', bars)):
        for field in dataclasses.fields(tree):
            value = getattr(tree, field.name)
            if value is not None:
                array = np.asarray(jax.device_get(value))
                if array.dtype.kind in 'biufc':
                    arrays[f'{prefix}__{field.name}'] = array
    np.savez_compressed(args.out_dir / 'linearization.npz', **arrays)
    print('[geometry-duality] saved linearization; comparing production tangent and adjoint', flush=True)
    rows = compare_geometry_duality(raw, bars)
    for row in rows:
        row['parameter'] = ':'.join(map(str, specs[row['parameter_column']]))
        relative = f"{row['relative_error']:.6e}" if row['relative_error'] is not None else 'undefined (zero contraction)'
        print(f"[geometry-duality] method={row['response_method']} {row['parameter']} row={row['cotangent_row']} "
              f"tangent={row['tangent_contraction']:.16e} adjoint={row['adjoint']:.16e} "
              f"relative_error={relative}", flush=True)
    with (args.out_dir / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'[geometry-duality] wrote {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
