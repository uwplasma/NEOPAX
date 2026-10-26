"""Inspect saved Er root-candidate results without running any physics.

Only JSON files and NumPy arrays are read. No NEOPAX, JAX, VMEC, NTX or
transport modules are imported. The crossing/slope decomposition is an exact
piecewise-linear identity when the target and crossing share a grid cell;
it cannot establish physical causation or numerical grid pinning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

LOCAL = "audit_root_local_cost"
ANNULAR = "audit_root_annular_cost"
ORIGINAL = ("audit_root_original_cost", "audit_root_bootstrap_cost", "audit_root_original_bootstrap_cost")


def read_json(path):
    return json.loads(path.read_text())


def point_key(values):
    return tuple(float(value) for value in values)


def crossing(er, rho, *, target, inner_radius, outer_radius):
    er, rho = np.asarray(er, dtype=float), np.asarray(rho, dtype=float)
    if (er.ndim != 1 or er.shape != rho.shape or len(er) < 3
            or not np.isfinite(er).all() or not np.isfinite(rho).all()
            or np.any(np.diff(rho) <= 0)):
        raise ValueError("Expected matching finite Er and strictly increasing rho arrays.")
    if not rho[0] < target < rho[-1] or not inner_radius < target < outer_radius:
        raise ValueError("Target must be inside both the grid and the analysis annulus.")
    indices = np.flatnonzero((er[:-1] >= 0) & (er[1:] < 0))
    roots = rho[indices] + np.diff(rho)[indices] * er[indices] / (er[indices] - er[indices + 1])
    keep = (roots >= inner_radius) & (roots <= outer_radius)
    roots, indices = roots[keep], indices[keep]
    result = dict(root=None, bracket=None, fraction=None, target_in_crossing_cell=False,
                  Er_at_target=float(np.interp(target, rho, er)), downcrossings=roots.tolist())
    if not len(roots):
        return result
    nearest = int(np.argmin(abs(roots - target)))
    i, q = int(indices[nearest]), float(roots[nearest])
    width, contrast = float(rho[i + 1] - rho[i]), float(er[i] - er[i + 1])
    result.update(root=q, root_error=q - target, bracket=i, rho_left=float(rho[i]),
                  rho_right=float(rho[i + 1]), Er_left=float(er[i]), Er_right=float(er[i + 1]),
                  cell_width=width, contrast=contrast, downward_slope=contrast / width,
                  fraction=float(er[i] / contrast),
                  target_in_crossing_cell=bool(rho[i] <= target <= rho[i + 1]))
    return result


def compare_profiles(before, after, target):
    """Separate local contrast changes from crossing displacement algebraically."""
    a, b = before["crossing"], after["crossing"]
    result = dict(root_shift=(b["root"] - a["root"] if a["root"] is not None and b["root"] is not None else None),
                  target_Er_change=b["Er_at_target"] - a["Er_at_target"], decomposition_available=False)
    ra, rb = np.asarray(before["rho"]), np.asarray(after["rho"])
    if ra.shape != rb.shape or not np.allclose(ra, rb, rtol=1e-12, atol=1e-14):
        result["reason"] = "Radial grids differ."
        return result
    if a["bracket"] is None or b["bracket"] != a["bracket"]:
        result["reason"] = "The selected crossing is missing or changes bracket."
        return result
    i = a["bracket"]
    indices = slice(max(0, i - 1), min(len(ra), i + 3))
    ea, eb = np.asarray(before["Er"])[indices], np.asarray(after["Er"])[indices]
    denominator = float(ea @ ea)
    if denominator > 0:
        factor = float(ea @ eb / denominator)
        delta_norm = float(np.linalg.norm(eb - ea))
        result.update(local_rescaling_factor=factor,
                      non_scaling_fraction_of_local_change=(float(np.linalg.norm(eb - factor * ea)) / delta_norm
                                                            if delta_norm > 0 else 0.))
    if not (a["target_in_crossing_cell"] and b["target_in_crossing_cell"]):
        result["reason"] = "The target is outside the selected crossing cell."
        return result
    slope0, slope1 = a["downward_slope"], b["downward_slope"]
    offset0, offset1 = a["root"] - target, b["root"] - target
    motion = .5 * (slope0 + slope1) * (offset1 - offset0)
    slope = .5 * (offset0 + offset1) * (slope1 - slope0)
    change = result["target_Er_change"]
    result.update(decomposition_available=True, crossing_motion_component=motion,
                  local_slope_component=slope, closure_error=change - motion - slope,
                  crossing_motion_fraction=motion / change if abs(change) > 1e-15 else None,
                  local_slope_fraction=slope / change if abs(change) > 1e-15 else None)
    return result


def candidate_rows(report, root):
    """Compare physical descent directions using rows already differentiated."""
    if not report.get("all_finite"):
        return []
    audit = report["root_loss_audit"]
    labels = [name.removeprefix("transport:") for name in report["residual_labels"]]
    jac = np.asarray(report["jacobian"], dtype=float)
    scales = np.asarray(audit["coordinate_scales"], dtype=float)
    parameters = report["parameter_order"]
    if jac.shape != (len(labels), len(parameters)) or scales.shape != (len(parameters),):
        raise ValueError("AD table/scales do not match the recorded row/column labels.")
    if not np.isfinite(jac).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Invalid stored AD table/scales.")
    gq = jac[labels.index("audit_root_nearest_downcrossing_rho")]
    ge = jac[labels.index("audit_root_interpolated_Er")]
    costs = audit.get("baseline_metrics", {}).get("candidates", {})
    rows = []
    reported = {item["name"]: item for item in audit.get("comparisons", [])}
    names = (LOCAL, ANNULAR, *(name for name in ORIGINAL if name in reported))
    for name in names:
        gc = (jac[labels.index(name)] if name in labels else
              np.asarray([reported[name]["physical_parameter_gradient"][parameter] for parameter in parameters]))
        metadata = costs.get(name, {})
        zero = metadata.get("zero_cost")
        # d[0.5*(E/s)^2]/dp = (2*C_zero/E)*dE/dp, without guessing the saved Er scale.
        gz = (2 * zero / root["Er_at_target"] * ge
              if zero is not None and root["Er_at_target"] != 0 else
              np.zeros_like(ge) if zero == 0 else None)
        for group, indices in (("profiles", [i for i, p in enumerate(parameters) if ":" not in p]),
                               ("geometry", [i for i, p in enumerate(parameters) if ":" in p])):
            g, gr = gc[indices] * scales[indices], gq[indices] * scales[indices]
            norm, root_norm = float(np.linalg.norm(g)), float(np.linalg.norm(gr))
            direction = -g / norm if norm > 0 else np.zeros_like(g)
            q_rate = float(gr @ direction) if root["root"] is not None and norm > 0 else None
            e_rate = float((ge[indices] * scales[indices]) @ direction) if norm > 0 else None
            error = root["root"] - audit["target"] if root["root"] is not None else None
            improvement = -float(np.sign(error)) * q_rate if error is not None and q_rate is not None else None
            row = dict(candidate=name, group=group, cost=metadata.get("cost"), zero_cost=zero,
                       sign_cost=metadata.get("sign_cost"), scaled_cost_gradient_l2=norm,
                       scaled_root_gradient_l2=root_norm if root["root"] is not None else None,
                       root_change_per_unit_descent=q_rate, Er_at_target_change_per_unit_descent=e_rate,
                       root_error_reduction_per_unit_descent=improvement,
                       root_direction_alignment=(improvement / root_norm
                                                 if improvement is not None and root_norm > 0 and error != 0 else None),
                       scaled_sign_gradient_l2=(float(np.linalg.norm((gc - gz)[indices] * scales[indices]))
                                                if gz is not None else None))
            row.update({key: metadata[key] for key in ("position_cost", "strength_cost", "bootstrap_cost") if key in metadata})
            if root["target_in_crossing_cell"] and q_rate is not None and e_rate is not None:
                motion = root["downward_slope"] * q_rate
                row.update(Er_rate_crossing_motion_component=motion, Er_rate_local_slope_component=e_rate - motion,
                           Er_rate_crossing_motion_fraction=motion / e_rate if abs(e_rate) > 1e-15 else None)
            rows.append(row)
    return rows


def load_points(seed, run_dir, descent_dir=None):
    audit = seed["root_loss_audit"]
    settings = {key: audit["settings"][key] for key in ("target", "inner_radius", "outer_radius")}
    baseline_key = point_key(audit["experiment"]["parameter_values"])
    entries = {baseline_key: dict(order=0, label="baseline", report=seed)}
    for path in sorted(run_dir.glob("evaluation_*.json")):
        match = re.fullmatch(r"evaluation_(\d+)_(ad|forward)\.json", path.name)
        if not match:
            continue
        value = read_json(path)
        is_ad = match[2] == "ad"
        values = value["root_loss_audit"]["experiment"]["parameter_values"] if is_ad else value["parameter_values"]
        if is_ad and (value["root_loss_audit"]["settings"] != audit["settings"]
                      or value["parameter_order"] != seed["parameter_order"]):
            raise ValueError(f"Root settings or parameter order differ in {path}.")
        key = point_key(values)
        entry = entries.setdefault(key, dict(order=int(match[1]), label=path.stem))
        entry["report" if is_ad else "forward"] = value
    events = []
    event_path = run_dir / "events.jsonl"
    if event_path.exists():
        lines = event_path.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise
                # The optimizer may still be appending the final line.
    for event in events:
        if "parameters" in event and point_key(event["parameters"]) in entries:
            entries[point_key(event["parameters"])]["event"] = event
    if descent_dir is not None:
        for path in descent_dir.glob("*_task.json"):
            task = read_json(path)
            key = point_key(task.get("parameter_values", []))
            result_path = path.with_name(path.name.replace("_task.json", ".json"))
            if (key not in entries or not result_path.exists()
                    or task.get("profile_contract") != "ad_initial_profiles_fixed_config_selected_root"
                    or task.get("resolved_config") != audit["experiment"]["resolved_config"]):
                continue
            result = read_json(result_path)
            digest = hashlib.sha256(json.dumps(task, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if result.get("task_digest") == digest:
                entries[key].setdefault("forward", result)
    points = []
    for key, entry in sorted(entries.items(), key=lambda item: item[1]["order"]):
        report, forward = entry.get("report"), entry.get("forward")
        event = entry.get("event", {})
        if report is not None:
            saved = report["root_loss_audit"]
            er, rho = saved["terminal_Er"], saved["terminal_rho"]
        else:
            er, rho = forward["terminal_Er"], forward["terminal_rho"]
        status = ("baseline" if key == baseline_key else "accepted" if event.get("event") == "accepted"
                  else "rejected" if event.get("event") == "rejected"
                  else "AD_ready_not_accepted" if report is not None else "forward_trial_AD_pending")
        point = dict(label=entry["label"], status=status, iteration=event.get("iteration"), step=event.get("step"),
                     parameter_values=list(key), rho=rho, Er=er, crossing=crossing(er, rho, **settings))
        if report is not None:
            point["candidates"] = candidate_rows(report, point["crossing"])
        if forward is not None:
            point["initial_Er"] = forward.get("initial_state", {}).get("Er")
            if report is not None:
                af, aa = np.asarray(forward["terminal_Er"]), np.asarray(er)
                point["forward_AD_max_abs_difference"] = float(np.max(abs(af - aa))) if af.shape == aa.shape else None
        points.append(point)
    if len(points) < 2:
        raise ValueError("No saved trial endpoints found; check --run-dir.")
    previous = points[0]
    initial0 = points[0].get("initial_Er")
    for point in points[1:]:
        point["change_from_previous_accepted"] = compare_profiles(previous, point, settings["target"])
        point["change_from_baseline"] = compare_profiles(points[0], point, settings["target"])
        initial = point.get("initial_Er")
        if initial0 is not None and initial is not None and len(initial0) == len(initial):
            a, b = np.asarray(initial0), np.asarray(initial)
            band = (np.asarray(point["rho"]) >= settings["inner_radius"]) & (np.asarray(point["rho"]) <= settings["outer_radius"])
            point["initial_Er_max_abs_change"] = float(np.max(abs(b - a)))
            point["initial_Er_annular_sign_change_indices"] = np.flatnonzero(band & (np.sign(a) != np.sign(b))).tolist()
        if point["status"] == "accepted":
            previous = point
    return dict(settings=settings, parameters=seed["parameter_order"], points=points,
                interpretation="Read-only analysis of saved arrays/gradients. Same-bracket persistence alone does not prove grid pinning; "
                               "the decomposition is algebraic and does not separate physical from numerical causes.")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(analysis, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(analysis, indent=2, allow_nan=False))
    summary, cells, candidates = [], [], []
    for point in analysis["points"]:
        base = dict(label=point["label"], status=point["status"], iteration=point["iteration"])
        summary.append({**base, **{k: v for k, v in point["crossing"].items() if k != "downcrossings"},
                        **point.get("change_from_previous_accepted", {})})
        candidates.extend({**base, **row} for row in point.get("candidates", []))
        bracket = point["crossing"]["bracket"]
        indices = range(max(0, bracket - 2), min(len(point["rho"]), bracket + 4)) if bracket is not None else []
        for i in indices:
            cells.append({**base, "index": i, "rho": point["rho"][i], "Er": point["Er"][i]})
    write_csv(out / "crossing_summary.csv", summary)
    write_csv(out / "crossing_cells.csv", cells)
    write_csv(out / "candidate_comparison.csv", candidates)


def plot_profiles(analysis, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    target = analysis["settings"]["target"]
    points = analysis["points"]
    for point in points:
        label = f"{point['status']} {point['iteration'] if point['iteration'] is not None else ''}"
        for axis in axes:
            axis.plot(point["rho"], point["Er"], marker=".", label=label)
        if point["crossing"]["root"] is not None:
            axes[1].plot(point["crossing"]["root"], 0., "o", color=axes[1].lines[-1].get_color())
    for axis in axes:
        axis.axvline(target, color="black", linestyle=":", label="Target")
        axis.axhline(0., color="0.5", linewidth=.7)
        axis.set(xlabel="rho", ylabel="Er [kV/m]")
    axes[0].set(xlim=(analysis["settings"]["inner_radius"], analysis["settings"]["outer_radius"]), title="Saved terminal profiles")
    root = points[0]["crossing"]
    if root["bracket"] is not None:
        axes[1].set_xlim(min(target, root["rho_left"]) - root["cell_width"],
                         max(target, root["rho_right"]) + root["cell_width"])
    axes[1].set_title("Crossing neighborhood; markers are interpolated zeros")
    axes[0].legend(fontsize=7)
    fig.savefig(out / "terminal_profiles.svg")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ad-report", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--descent-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    analysis = load_points(read_json(args.ad_report), args.run_dir, args.descent_dir)
    out = args.out_dir or args.run_dir / "saved_profile_analysis"
    write_outputs(analysis, out)
    if args.plot:
        plot_profiles(analysis, out)
    for point in analysis["points"]:
        root = point["crossing"]
        print("[saved-root-analysis] " + json.dumps(dict(label=point["label"], status=point["status"], crossing=root)))
        if "change_from_previous_accepted" in point:
            print("[saved-root-change] " + json.dumps(point["change_from_previous_accepted"]))
        for candidate in point.get("candidates", []):
            print("[saved-root-candidate] " + json.dumps(candidate))
    print(f"Wrote {out / 'summary.json'}; crossing_summary.csv, crossing_cells.csv, candidate_comparison.csv")


if __name__ == "__main__":
    main()
