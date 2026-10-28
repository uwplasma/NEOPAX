"""Compare existing saved-geometry transport outputs without any physics solve.

Imports only NumPy/stdlib and h5py when reading the existing HDF5 report.
Never imports NEOPAX, JAX, VMEX or NTX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def file_provenance(path, recorded_sha256=None):
    path = Path(path)
    available = path.is_file()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if available else None
    return dict(path=str(path), available=available, current_sha256=digest,
                recorded_sha256=recorded_sha256,
                matches_recorded=(bool(available and digest == recorded_sha256)
                                  if recorded_sha256 is not None else None))


def source_provenance(manifest, directory):
    """Compare recorded files with current files; hashes describe bytes only."""
    source = manifest.get("saved_geometry_result", {})
    seed = file_provenance(manifest["resolved_config"]["geometry"]["vmec_input_file"],
                           manifest.get("seed_sha256"))
    files = dict(baseline_input=seed)
    for key, hash_key in (("accepted_geometry_input", "accepted_geometry_sha256"),
                          ("summary_path", "summary_sha256")):
        if source.get(key):
            files[key] = file_provenance(source[key], source.get(hash_key))
    transport_input = file_provenance(Path(directory) /
        "optimized/transport/input.QI_neopax_geometry_full_transport_optimized")
    transport_input["matches_current_baseline_input_bytes"] = (
        bool(transport_input["current_sha256"] == seed["current_sha256"])
        if transport_input["available"] and seed["available"] else None)
    files["saved_transport_input"] = transport_input
    checks = [item["matches_recorded"] for item in files.values()
              if item["recorded_sha256"] is not None]
    return dict(files=files, all_recorded_sources_unchanged=all(checks) if checks else None,
                note="Different byte hashes can reflect formatting or numerical changes. "
                     "The historical baseline input cannot be reconstructed from its hash.")


def field_summary(er, rho, settings):
    er, rho = np.asarray(er, dtype=float), np.asarray(rho, dtype=float)
    if (er.ndim != 1 or er.shape != rho.shape or len(er) < 3 or not np.isfinite(er).all()
            or not np.isfinite(rho).all() or np.any(np.diff(rho) <= 0)):
        raise ValueError("Expected a finite terminal field on a strictly increasing rho grid.")
    lower, upper, target = settings["inner_radius"], settings["outer_radius"], settings["target"]
    mask = (rho >= lower) & (rho <= upper)
    inner, outer = mask & (rho < target), mask & (rho > target)
    if not np.any(inner) or not np.any(outer):
        raise ValueError("The saved grid must cover both sides of the target.")
    faces = np.flatnonzero((er[:-1] >= 0) & (er[1:] < 0))
    roots = rho[faces] + np.diff(rho)[faces] * er[faces] / (er[faces] - er[faces + 1])
    roots = roots[(roots >= lower) & (roots <= upper)]
    return dict(Er_at_target=float(np.interp(target, rho, er)), inner_mean_Er=float(np.mean(er[inner])),
                inner_positive_fraction=float(np.mean(er[inner] > 0)),
                outer_negative_fraction=float(np.mean(er[outer] < 0)),
                annular_min_Er=float(np.min(er[mask])), annular_max_Er=float(np.max(er[mask])),
                actual_downcrossings=roots.tolist(),
                nearest_downcrossing=float(roots[np.argmin(abs(roots-target))]) if len(roots) else None)


def read_terminal_hdf5(path):
    import h5py
    with h5py.File(path, "r") as stream:
        rho, er = np.asarray(stream["rho"]), np.asarray(stream["Er"])
        times = np.asarray(stream["ts"]).reshape(-1) if "ts" in stream else None
        if er.ndim == 2:
            if times is None or len(times) != len(er):
                raise ValueError("Saved transport Er rows must have matching times.")
            valid = np.isfinite(times) & np.isfinite(er).all(axis=1)
            if not np.any(valid):
                raise ValueError("The saved transport file has no finite saved time/field.")
            indices = np.flatnonzero(valid)
            index = int(indices[np.argmax(times[valid])])
            final_time, er = float(times[index]), er[index]
        elif er.ndim == 1 and "final_time" in stream:
            final_time = float(np.asarray(stream["final_time"]))
        else:
            raise ValueError("Unsupported saved transport field/time layout.")
    return dict(final_time=final_time, terminal_rho=rho, terminal_Er=er)


def compare_saved_transport(directory, snapshot, settings):
    reference = Path(directory) / "optimized/transport/transport_solution.h5"
    if not reference.exists():
        return dict(available=False, reference=str(reference))
    saved = read_terminal_hdf5(reference)
    old_rho, new_rho = np.asarray(saved["terminal_rho"]), np.asarray(snapshot["terminal_rho"])
    same_grid = old_rho.shape == new_rho.shape and np.allclose(old_rho, new_rho, rtol=1e-10, atol=1e-12)
    same_time = bool(np.isclose(saved["final_time"], snapshot["final_time"], rtol=0, atol=1e-10))
    difference = np.asarray(snapshot["terminal_Er"]) - saved["terminal_Er"] if same_grid else None
    er_matches = bool(same_grid and np.allclose(snapshot["terminal_Er"], saved["terminal_Er"], rtol=1e-5, atol=1e-6))
    return dict(available=True, reference=str(reference), reference_time=saved["final_time"],
                baseline_time=snapshot["final_time"], same_grid=bool(same_grid), same_time=same_time,
                terminal_Er_matches=er_matches, matches=bool(same_time and er_matches),
                max_abs_Er_difference=float(np.max(abs(difference))) if difference is not None else None,
                saved_field=field_summary(saved["terminal_Er"], old_rho, settings),
                baseline_field=field_summary(snapshot["terminal_Er"], new_rho, settings))


def runtime_resolution(manifest):
    geometry = manifest["resolved_config"]["geometry"]
    path = Path(geometry["vmec_input_file"])
    text = re.sub(r"!.*", "", path.read_text())
    modes = {}
    for name in ("mpol", "ntor"):
        match = re.search(r"\b" + name + r"\s*=\s*(\d+)", text, re.IGNORECASE)
        modes[name] = int(match[1]) if match else None
    effective = {}
    for name, alias, auto in (("mboz", "vmec_mboz", max(2, 6*modes["mpol"]) if modes["mpol"] is not None else None),
                              ("nboz", "vmec_nboz", max(0, 2*modes["ntor"]-1) if modes["ntor"] is not None else None)):
        raw = geometry.get(name, geometry.get(alias))
        effective[name] = auto if raw is None or str(raw).strip().lower() in {"", "none"} else int(raw)
    return dict(configured={key: geometry.get(key, geometry.get("vmec_"+key)) for key in ("mboz", "nboz")},
                effective=effective, declared={key: manifest[key] for key in ("mboz", "nboz")},
                matches_declared=all(effective[key] == manifest[key] for key in effective),
                vmec_input=str(path), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-result-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Directory containing case_manifest.json and baseline_forward_observation.json.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads((args.baseline_dir / "case_manifest.json").read_text())
    snapshot = json.loads((args.baseline_dir / "baseline_forward_observation.json").read_text())
    summary = dict(source_provenance=source_provenance(manifest, args.geometry_result_dir),
                   runtime_resolution=runtime_resolution(manifest),
                   terminal_comparison=compare_saved_transport(args.geometry_result_dir, snapshot, manifest["settings"]),
                   note="Reads existing files only. A mismatch does not identify which solver, initializer or runtime change caused it.")
    text = json.dumps(summary, indent=2, allow_nan=False)
    print(text)
    path = args.output or args.baseline_dir / "saved_geometry_baseline_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


if __name__ == "__main__":
    main()
