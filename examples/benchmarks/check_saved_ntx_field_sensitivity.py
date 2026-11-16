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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
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
    with (args.out_dir/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: f"{v:.16e}" if isinstance(v, float) else v for k,v in row.items()} for row in rows)
    print(f"[field-sensitivity] saved {args.out_dir/'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
