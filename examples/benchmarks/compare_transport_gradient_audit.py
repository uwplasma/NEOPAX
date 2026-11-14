"""Compare saved AD terminal cotangents with FD endpoint state changes.

Reads files only; no JAX import or physics solve. The dot product checks the
terminal state contribution, not the entire transport or geometry derivative.
"""
import argparse
import csv
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ad-dir", type=Path, required=True)
    parser.add_argument("--fd-dir", type=Path, required=True)
    parser.add_argument("--fd-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with _one(args.ad_dir, "ad_terminal_*.npz") as ad, \
         _one(args.fd_dir, "fd_baseline_*.npz") as baseline, \
         _one(args.fd_dir, "fd_*_minus_*.npz") as minus, \
         _one(args.fd_dir, "fd_*_plus_*.npz") as plus:
        rows = compare(ad, baseline, minus, plus, json.loads(args.fd_report.read_text()))
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
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
