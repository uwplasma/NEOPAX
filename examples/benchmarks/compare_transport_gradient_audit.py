"""Compare saved terminal and optional pre-scan-fold adjoints with FD changes.

Reads files only; no JAX import or physics solve. The optional support-chain
check tests the fixed-support transport adjoint along the saved FD direction;
it does not itself validate the later NTX/VMEX parameter pullback.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _one(directory, pattern):
    paths = tuple(Path(directory).glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} in {directory}; found {len(paths)}. Use one run per directory.")
    return np.load(paths[0], allow_pickle=False)


def _labels(archive):
    return tuple(str(archive[key].item()) for key in archive.files if key.startswith("labels["))


def compare(ad, baseline, minus, plus, report):
    labels = _labels(ad)
    if any(_labels(archive) != labels for archive in (baseline, minus, plus)) or tuple(report["objective_labels"]) != labels:
        raise ValueError("Objective labels/order differ.")
    # Establish the comparison point from arrays, not just rounded objectives.
    keys = ("initial_y", "final_y", "final_time", "values") + tuple(
        key for key in ad.files if key.startswith("final_state."))
    for key in keys:
        a, b = ad[key], baseline[key]
        if a.shape != b.shape:
            raise ValueError(f"Different baseline {key} shapes: {a.shape} versus {b.shape}")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Nonfinite baseline {key}.")
        difference = float(np.max(np.abs(a-b), initial=0.))
        print(f"[gradient-audit] baseline {key}: max_abs_difference={difference:.9e}")
        if not np.allclose(a, b, rtol=1e-12, atol=1e-13):
            raise ValueError(f"AD/FD baseline {key} differs; refusing a derivative-error comparison.")
    if not np.allclose(report["objective_values"], baseline["values"], rtol=1e-12, atol=1e-13):
        raise ValueError("FD summary does not match the saved FD baseline.")
    h = float(report["fd_step"])
    if not np.isfinite(h) or h <= 0:
        raise ValueError("fd_step must be finite and positive.")
    dy = (plus["final_y"] - minus["final_y"]) / (2*h)
    state_contribution = ad["final_y_bars"] @ dy
    if not np.isfinite(state_contribution).all():
        raise ValueError("Nonfinite terminal state contribution.")
    geometry_split = report.get("baseline_geometry_final_state_fd")
    profile = report["parameter_kind"] == "profile"
    reference = report["gradient_fd"] if profile else geometry_split
    rows = []
    for index, label in enumerate(labels):
        actual = float(state_contribution[index])
        expected = None if reference is None else float(reference[index])
        difference = None if expected is None else abs(actual-expected)
        relative = None if expected in (None, 0.) else difference/abs(expected)
        rows.append(dict(objective=label, terminal_bar_dot_fd_final_y=actual,
            fd_reference=expected, absolute_difference=difference, relative_difference=relative))
    # Fixed-temperature projections may also depend on profile parameters
    # outside final_y. A disagreement here localizes that boundary; it must
    # not be called a failed transport reverse sweep without checking it.
    return rows


def compare_support_chain(ad, fold, baseline, minus, plus, report):
    """Separate the terminal cost and full fixed-support transport adjoint."""
    labels = _labels(ad)
    if any(_labels(data) != labels for data in (fold, baseline, minus, plus)) or tuple(report["objective_labels"]) != labels:
        raise ValueError("Pre-scan-fold objective labels/order differ.")
    if report.get("parameter_kind") != "realtime_geometry":
        raise ValueError("Support-chain comparison requires a geometry parameter; profiles have a separate direct term.")
    h = float(report["fd_step"])
    if not np.isfinite(h) or h <= 0:
        raise ValueError("fd_step must be finite and positive.")
    if any(k.startswith("native_vmec_face_coefficient_bars") for k in fold.files):
        raise ValueError("This check requires the geometry/database-only support contract.")
    directions = {}
    for key in ad.files:
        if not key.startswith(("support['geometry'].", "support['database'].")):
            continue
        if any(key not in data for data in (baseline, minus, plus)):
            raise ValueError(f"Missing endpoint support: {key}; update FD audit capture before rerunning FD.")
        arrays = [data[key] for data in (ad, baseline, minus, plus)]
        if any(a.shape != arrays[0].shape or a.dtype != arrays[0].dtype or not np.isfinite(a).all() for a in arrays):
            raise ValueError(f"Invalid support shape, dtype or values: {key}")
        if arrays[0].tobytes() != arrays[1].tobytes():
            raise ValueError(f"AD/FD baseline support differs: {key}")
        if arrays[0].dtype.kind in "fc":
            directions[key] = (arrays[3]-arrays[2])/(2*h)
            if not np.isfinite(directions[key]).all():
                raise ValueError(f"Nonfinite support direction: {key}")
    if not directions:
        raise ValueError("Missing saved geometry/database support.")
    fixed = {k for k in baseline.files if k.startswith("fixed_inputs[")}
    if not fixed or any({k for k in data.files if k.startswith("fixed_inputs[")} != fixed for data in (minus, plus)):
        raise ValueError("Missing fixed species/energy input capture; update FD audit recording.")
    for key in fixed:
        a = baseline[key]
        if any(a.shape != data[key].shape or a.dtype != data[key].dtype
               or a.tobytes() != data[key].tobytes() for data in (minus, plus)):
            raise ValueError(f"FD changed a supposedly fixed input: {key}")

    def contract(data, prefix):
        terms = []
        for key, direction in directions.items():
            bar_key = prefix + key[len("support"):]
            if bar_key not in data:
                raise ValueError(f"Missing support cotangent: {bar_key}")
            bar = data[bar_key]
            if bar.shape != direction.shape or not np.isfinite(bar).all():
                raise ValueError(f"Invalid support cotangent: {bar_key}")
            terms.append(math.fsum(np.real(bar*direction).reshape(-1)))
        total = math.fsum(terms)
        if not np.isfinite(total):
            raise ValueError(f"Nonfinite support contraction: {prefix}")
        return total

    dy = (plus["final_y"]-minus["final_y"])/(2*h)
    rows = []
    for i, label in enumerate(labels):
        terminal = float(ad["final_y_bars"][i] @ dy) + contract(ad, f"explicit_support_bars[{i}]")
        support = contract(fold, f"support_bars[{i}]")
        fd = float(report["gradient_fd"][i])
        if not all(math.isfinite(value) for value in (terminal, support, fd)):
            raise ValueError(f"Nonfinite chain comparison: {label}")
        relative = lambda value: abs(value-fd)/abs(fd) if fd else (0. if value == 0. else float("inf"))
        rows.append(dict(objective=label, terminal_full_chain=terminal,
            terminal_full_chain_relative_difference=relative(terminal),
            support_chain=support, support_chain_relative_difference=relative(support),
            full_gradient_fd=fd))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ad-dir", type=Path, required=True)
    parser.add_argument("--fd-dir", type=Path, required=True)
    parser.add_argument("--fd-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ad-before-scan-fold", type=Path,
        help="Also contract the saved pre-scan-fold adjoint with FD geometry/table changes; requires endpoint support capture.")
    args = parser.parse_args()
    with _one(args.ad_dir, "ad_terminal_*.npz") as ad, \
         _one(args.fd_dir, "fd_baseline_*.npz") as baseline, \
         _one(args.fd_dir, "fd_*_minus_*.npz") as minus, \
         _one(args.fd_dir, "fd_*_plus_*.npz") as plus:
        report = json.loads(args.fd_report.read_text())
        rows = compare(ad, baseline, minus, plus, report)
        if args.ad_before_scan_fold is not None:
            with np.load(args.ad_before_scan_fold, allow_pickle=False) as fold:
                support_rows = compare_support_chain(ad, fold, baseline, minus, plus, report)
            for row, extra in zip(rows, support_rows, strict=True):
                row.update(extra)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        error = row["relative_difference"]
        print(f"{row['objective']}: terminal_state={row['terminal_bar_dot_fd_final_y']:.9e} "
              f"relative_difference={error:.9e}" if error is not None else
              f"{row['objective']}: reference unavailable or zero; see CSV")
        if "support_chain" in row:
            print(f"{row['objective']}: terminal_full_chain={row['terminal_full_chain']:.12e} "
                  f"terminal_relative={row['terminal_full_chain_relative_difference']:.6e} "
                  f"support_chain={row['support_chain']:.12e} "
                  f"support_relative={row['support_chain_relative_difference']:.6e} "
                  f"full_FD={row['full_gradient_fd']:.12e}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
