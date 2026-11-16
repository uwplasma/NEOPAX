"""Replay two saved NTX cases at the observed one/two-ULP field changes.

GPU only. No equilibrium, root, transport or derivative solve. The production
single-point scan body is used on one fixed prepared surface per radius.
The scalar diagnostic additionally returns the primary-source retained Schur
residual; it does not certify the parallel source or every Legendre block row.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POINTS = ((5, 0, 6), (6, 0, 7))


def snapshot(directory):
    paths = list(directory.glob("generic_database_baseline_*.npz"))
    if len(paths) != 1:
        raise ValueError(f"Expected one baseline snapshot in {directory}; found {len(paths)}")
    with np.load(paths[0], allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def field_at(arrays, er_tilde, r, e):
    # Reproduce the supplied NumPy readout's multiplication order exactly.
    c = lambda name: arrays["scan_inputs['channels']."+name]
    return np.float64((np.float64(er_tilde[e])*c("dr_tildeds")[r])*c("b00")[r])


def surface_fields(arrays, radius):
    prefix = f"scan_inputs['surfaces'][{radius}]."
    fields = {k[len(prefix):]: v for k, v in arrays.items() if k.startswith(prefix)}
    if not fields or "b_cos" not in fields:
        raise ValueError(f"No saved VMEC surface {radius}")
    return fields


def restore_surface(fields, context, surface_type, asarray):
    values = {k: asarray(v) for k, v in fields.items()}
    # Match the original surface builder's Python/static scalar fields.
    for name in ("requested_psi_n", "psi_n"):
        values[name] = float(fields[name])
    values["nfp"] = int(fields["nfp"])
    resolution = context.static.resolution
    count = fields["b_cos"].size
    values.update(path=context.input_path, ns=int(resolution.ns),
        mpol=int(resolution.mpol), ntor=int(resolution.ntor),
        total_mode_count=count, loaded_mode_count=count)
    return surface_type(**values)


def head_matrix(diagonal, lower, upper, last_schur, xp):
    """Three original head rows with the eliminated tail folded into row two."""
    zero = xp.zeros_like(last_schur)
    return xp.concatenate((
        xp.concatenate((diagonal[0], upper[0], zero), axis=1),
        xp.concatenate((lower[1], diagonal[1], upper[1]), axis=1),
        xp.concatenate((zero, lower[2], last_schur), axis=1)), axis=0)


def scaled_solve(matrix, rhs, xp):
    """Diagnostic diagonal scaling; solves the same reduced equations."""
    a, b = matrix, rhs
    columns = xp.ones(matrix.shape[1], dtype=matrix.dtype)
    for _ in range(5):
        row_max = xp.max(xp.abs(a), axis=1)
        rows = 1./xp.sqrt(xp.where(row_max > 0., row_max, 1.))
        a, b = rows[:, None]*a, rows[:, None]*b
        column_max = xp.max(xp.abs(a), axis=0)
        scale = 1./xp.sqrt(xp.where(column_max > 0., column_max, 1.))
        a, columns = a*scale[None, :], columns*scale
    return columns[:, None]*xp.linalg.solve(a, b), a


def make_head_audit():
    import jax
    import jax.numpy as jnp
    from ntx import _solver as solver
    from solvax import direct

    def audit(prepared, nu, es):
        ctx = solver._operator_context(prepared.surface, prepared.geometry, prepared.grid, nu, es)
        s1, s3 = solver.source_modes(ctx, prepared.grid.n_xi)
        rhs_low = jnp.stack((s1[:3], s3[:3]), axis=-1)
        block = solver._operator_block_fn(ctx, prepared.d_theta, prepared.d_zeta)
        native, lu, piv, _, _ = direct._block_thomas_truncated_fn_state(
            block, prepared.grid.n_xi+1, rhs_low, 3)
        size = lu.shape[1]
        pivoted = (jnp.tril(lu, -1)+jnp.eye(size)) @ jnp.triu(lu)
        permutation = jax.lax.linalg.lu_pivots_to_permutation(piv[2], size)
        last_schur = pivoted[2][jnp.argsort(permutation)]
        lower, diagonal, upper = jax.lax.map(block, jnp.arange(3, dtype=jnp.int32))
        matrix = head_matrix(diagonal, lower, upper, last_schur, jnp)
        rhs = rhs_low.reshape(3*size, 2)
        dense = jnp.linalg.solve(matrix, rhs)
        scaled, scaled_matrix = scaled_solve(matrix, rhs, jnp)

        def coefficients(value):
            modes = value.reshape(3, size, 2)
            return jnp.stack(solver.coefficients_from_modes(prepared.geometry,
                modes[...,0], modes[...,1], nu))

        def errors(value):
            residual = jnp.linalg.norm(matrix @ value-rhs, axis=0)
            tiny = jnp.finfo(matrix.dtype).tiny
            rhs_norm = jnp.linalg.norm(rhs, axis=0)
            relative = residual/jnp.maximum(rhs_norm, tiny)
            backward = residual/jnp.maximum(jnp.linalg.norm(matrix)*jnp.linalg.norm(value, axis=0)+rhs_norm, tiny)
            return relative, backward

        def condition(value):
            singular = jnp.linalg.svd(value, compute_uv=False)
            return singular[...,0]/singular[...,-1]

        values = (native.reshape(3*size, 2), dense, scaled)
        return dict(matrix=matrix, rhs=rhs, pivots=piv,
            coefficients=jnp.stack([coefficients(x) for x in values]),
            relative_residual=jnp.stack([errors(x)[0] for x in values]),
            backward_error=jnp.stack([errors(x)[1] for x in values]),
            schur_condition=condition(pivoted), head_condition=condition(matrix),
            scaled_head_condition=condition(scaled_matrix))
    return jax.jit(audit)


def compile_flat_inputs(function, leaves, structure, backend):
    # AOT argument metadata contains ArgInfo objects, which fail NTX's surface
    # validation. Rebuild the dataclasses only inside tracing, from array leaves.
    def flat_function(*values):
        return function(*backend.tree_util.tree_unflatten(structure, values))
    return backend.jit(flat_function).lower(*leaves).compile()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--audit-head", action="store_true",
        help="Audit the retained Schur system and compare equivalent pivoted/scaled reduced solves.")
    args = parser.parse_args()
    config_path = args.run_dir/"config.toml"
    config = tomllib.loads(config_path.read_text())
    nc = config["neoclassical"]
    if nc.get("flux_model") != "ntx_scan_runtime" or nc.get("interpolation_mode", "generic") != "generic":
        parser.error("Use the saved GENERIC baseline-check directory.")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Choose a new empty output directory.")
    a, b = snapshot(args.run_dir/"ad"), snapshot(args.run_dir/"fd_2")
    for r, _, _ in POINTS:
        fa, fb = surface_fields(a, r), surface_fields(b, r)
        if fa.keys() != fb.keys() or any(not np.array_equal(fa[k], fb[k], equal_nan=True) for k in fa):
            parser.error(f"Saved surface {r} differs; this is not an identical-surface field check.")

    import jax
    if jax.default_backend() != "gpu" or not jax.config.x64_enabled:
        parser.error("Run on the large GPU with JAX_ENABLE_X64=true; no CPU fallback.")
    import jax.numpy as jnp
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ntx
    from ntx.geometry import VmecSurface
    from ntx._solver_scan import _scan_coefficients_serial
    from benchmark_transport_reverse_ad_only import _geometry_context_from_config
    from NEOPAX._transport_flux_models import _build_ntx_field_channels
    from NEOPAX._optimization_diagnostics import save_gradient_audit

    metadata = json.loads((args.run_dir/"ad/metadata.json").read_text())
    for name, expected in metadata["sources"].items():
        path = Path(importlib.import_module(name).__file__)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise RuntimeError(f"Source changed since the saved baseline: {name}")
    for name, expected in metadata["versions"].items():
        if expected is not None and importlib.metadata.version(name) != expected:
            raise RuntimeError(f"Package version changed since the saved baseline: {name}")

    input_path = Path(config["geometry"]["vmec_input_file"])
    config["geometry"]["vmec_input_file"] = str((ROOT/input_path).resolve())
    # This constructs static metadata from the input; it does not solve VMEC.
    context = _geometry_context_from_config(config, "RBC:1:0")
    grid = ntx.GridSpec(n_theta=int(nc["ntx_scan_n_theta"]),
        n_zeta=int(nc["ntx_scan_n_zeta"]), n_xi=int(nc["ntx_scan_n_xi"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/"config.toml").write_bytes(config_path.read_bytes())
    os.environ["NEOPAX_GRADIENT_AUDIT_DIR"] = str(args.out_dir.resolve())
    (args.out_dir/"baseline_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")

    @jax.jit
    def diagnostic(prepared, nu, es):
        result = ntx.solve_prepared(prepared, ntx.MonoenergeticCase(nu_hat=nu, epsi_hat=es))
        values = jnp.stack((result.D11, result.D31, result.D13, result.D33, result.D33_spitzer))
        return values, result.residual_l2, result.onsager_residual

    rows = []
    head_rows = []
    audit_head = make_head_audit() if args.audit_head else None
    head_executable = None
    head_input_structure = None
    with jax.default_device(jax.local_devices(backend="gpu")[0]):
        production_fields = {}
        for name, arrays in (("ad", a), ("fd_2", b)):
            prefix = "scan_inputs['channels']."
            channels = {k[len(prefix):]: jnp.asarray(v) for k, v in arrays.items() if k.startswith(prefix)}
            production_fields[name] = np.asarray(_build_ntx_field_channels(
                jnp.asarray(nc["ntx_scan_rho"]), jnp.asarray(nc["ntx_scan_er_tilde"]), channels)[1])
        for r, n, e in POINTS:
            surface = restore_surface(surface_fields(a, r), context, VmecSurface, jnp.asarray)
            print(f"[field-sensitivity] preparing saved surface {r}; no equilibrium solve", flush=True)
            prepared = ntx.prepare_monoenergetic_system(surface, grid)
            nu_value = np.float64(nc["ntx_scan_nu_v"][n])
            nu = jnp.asarray(nu_value, dtype=jnp.float64)
            save_gradient_audit(f"frozen_prepared_r{r}", prepared=prepared,
                radius_index=np.asarray(r), nu=nu)
            baseline = None
            for label, arrays, es_value in (
                ("ad_numpy", a, field_at(a, nc["ntx_scan_er_tilde"], r, e)),
                ("ad_repeat", a, field_at(a, nc["ntx_scan_er_tilde"], r, e)),
                ("fd2_numpy", b, field_at(b, nc["ntx_scan_er_tilde"], r, e)),
                ("ad_production_field", a, production_fields["ad"][r,e]),
                ("fd2_production_field", b, production_fields["fd_2"][r,e]),
            ):
                es = jnp.asarray(es_value, dtype=jnp.float64)
                values = np.asarray(_scan_coefficients_serial(prepared, nu[None], es[None]))[0]
                diag, residual, onsager = jax.device_get(diagnostic(prepared, nu, es))
                if not np.isfinite(values).all() or not np.isfinite(diag).all():
                    raise FloatingPointError(f"Nonfinite coefficient at {(r,n,e,label)}")
                if baseline is None:
                    baseline = values.copy()
                drds = float(arrays["scan_inputs['channels'].drds"][r])
                observed = float(arrays["support['database'].D13"][r,n,e])/drds
                rows.append(dict(radius_index=r, nu_index=n, field_index=e, case=label,
                    Es_hex=float(es_value).hex(), nu_over_v=float(nu_value),
                    scan_raw_D11=float(values[0]), scan_raw_D13=float(values[2]), scan_raw_D33=float(values[3]),
                    saved_raw_D13=observed, raw_D13_minus_saved=float(values[2])-observed,
                    raw_D13_minus_first=float(values[2]-baseline[2]),
                    diagnostic_raw_D13=float(diag[2]),
                    diagnostic_minus_scan_D13=float(diag[2]-values[2]),
                    primary_schur_residual_l2=float(residual), onsager_residual=float(onsager)))
                row = rows[-1]
                print(f"[field-sensitivity] r={r} case={label} Es={row['Es_hex']} "
                    f"raw_D13={row['scan_raw_D13']:.16e} delta={row['raw_D13_minus_first']:.9e} "
                    f"saved_gap={row['raw_D13_minus_saved']:.9e} "
                    f"diagnostic_gap={row['diagnostic_minus_scan_D13']:.9e} "
                    f"primary_schur_residual={row['primary_schur_residual_l2']:.9e}", flush=True)
            if audit_head is not None:
                previous_pivots = None
                for label, arrays in (("ad", a), ("fd2", b)):
                    es = jnp.asarray(production_fields["ad" if label == "ad" else "fd_2"][r,e])
                    head_values, structure = jax.tree_util.tree_flatten((prepared, nu, es))
                    if head_executable is None:
                        print("[head-audit] compiling bounded three-block audit", flush=True)
                        head_input_structure = structure
                        head_executable = compile_flat_inputs(audit_head, head_values, structure, jax)
                        memory = head_executable.memory_analysis()
                        needed = memory.temp_size_in_bytes+memory.argument_size_in_bytes+memory.output_size_in_bytes-memory.alias_size_in_bytes
                        print(f"[head-audit] compiled buffer estimate={needed} bytes", flush=True)
                        if needed > 4*1024**3:
                            raise RuntimeError("Head-audit compiled buffers exceed the 4 GiB limit; audit not executed.")
                    if structure != head_input_structure:
                        raise ValueError("Prepared-system static structure changed between saved cases.")
                    data = jax.device_get(head_executable(*head_values))
                    np.savez_compressed(args.out_dir/f"head_r{r}_{label}.npz", **data)
                    changed = 0 if previous_pivots is None else int(np.count_nonzero(data['pivots'] != previous_pivots))
                    previous_pivots = data['pivots'].copy()
                    print(f"[head-audit] r={r} case={label} Schur_conditions={data['schur_condition']} "
                        f"head_condition={float(data['head_condition']):.9e} "
                        f"scaled_condition={float(data['scaled_head_condition']):.9e} changed_head_pivots={changed}", flush=True)
                    saved = float(arrays["support['database'].D13"][r,n,e])/float(arrays["scan_inputs['channels'].drds"][r])
                    for i, method in enumerate(("production_head", "pivoted_head", "scaled_pivoted_head")):
                        coeff = data['coefficients'][i]
                        item = dict(radius_index=r, case=label, method=method,
                            raw_D11=float(coeff[0]), raw_D31=float(coeff[1]), raw_D13=float(coeff[2]),
                            raw_D33=float(coeff[3]), onsager_residual=float(abs(coeff[1]+coeff[2])),
                            raw_D13_minus_saved=float(coeff[2])-saved,
                            head_condition=float(data['head_condition']), scaled_condition=float(data['scaled_head_condition']),
                            primary_relative_residual=float(data['relative_residual'][i,0]),
                            parallel_relative_residual=float(data['relative_residual'][i,1]),
                            primary_backward_error=float(data['backward_error'][i,0]),
                            parallel_backward_error=float(data['backward_error'][i,1]))
                        head_rows.append(item)
                        print(f"[head-audit] r={r} case={label} method={method} D11={item['raw_D11']:.9e} "
                            f"D13={item['raw_D13']:.16e} Onsager={item['onsager_residual']:.9e} "
                            f"primary_relres={item['primary_relative_residual']:.9e} "
                            f"parallel_relres={item['parallel_relative_residual']:.9e}", flush=True)
    with (args.out_dir/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: f"{v:.16e}" if isinstance(v, float) else v for k,v in row.items()} for row in rows)
    print(f"[field-sensitivity] saved {args.out_dir/'comparison.csv'}", flush=True)
    if head_rows:
        with (args.out_dir/"head_comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(head_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows({k: f"{v:.16e}" if isinstance(v, float) else v for k,v in row.items()} for row in head_rows)


if __name__ == "__main__":
    main()
