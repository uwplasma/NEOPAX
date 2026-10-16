"""Reproduce a host-only analysis of the archived diagnostic JSON excerpts.

Only JSON, small NumPy matrices, and the archived console CSV are read. No
NEOPAX, JAX, VMEC, NTX, transport solve, or finite-difference run is imported.
"""

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = "transport:Er_transition_location_moment"
BOOT = "bootstrap_current_penalty"
POWER = "transport:net_total_power_volume_average_mw_m3"
PROFILE_NAMES = ("n0", "T0", "density_shape_power", "temperature_shape_power",
                 "density_shape_alpha", "temperature_shape_alpha")


def profile_analysis(points, matrices):
    """Compare recorded steps with hypothetical directions in the saved model."""
    sensitivities, directions, chosen_steps = [], [], []
    for trial, (jacobian, residuals, labels) in sorted(matrices.items()):
        jp = jacobian[:, :6]
        ir, ib, ip = [labels.index(name) for name in (ROOT, BOOT, POWER)]
        root, boot, power = jp[[ir, ib, ip]]
        total_gradient = jp.T @ residuals
        for k, name in enumerate(PROFILE_NAMES):
            sensitivities.append({
                "trial": trial, "parameter": name,
                "root_rho_derivative_per_relative_coordinate": root[k] / np.sqrt(1000),
                "root_cost_derivative": residuals[ir] * root[k],
                "bootstrap_cost_derivative": residuals[ib] * boot[k],
                "power_cost_derivative": residuals[ip] * power[k],
                "total_cost_derivative": total_gradient[k],
            })

        def perpendicular(row, other_rows):
            return row - np.linalg.lstsq(other_rows.T, row, rcond=1e-12)[0] @ other_rows

        candidates = {
            "root_descent": -root,
            "total_cost_descent": -total_gradient,
            "root_descent_preserving_power": -perpendicular(root, power[None, :]),
            "root_descent_preserving_power_and_bootstrap": -perpendicular(root, np.array([power, boot])),
        }
        for name, candidate in candidates.items():
            direction = candidate / np.linalg.norm(candidate)
            # A common tiny coordinate step is only a linear-model comparison.
            # No function evaluation is made at any hypothetical point.
            step_norm = 1e-4
            change = jp @ (step_norm * direction)
            directions.append({
                "trial": trial, "direction": name, "step_norm": step_norm,
                "direction_n0": direction[0], "direction_T0": direction[1],
                "direction_density_shape_power": direction[2],
                "direction_temperature_shape_power": direction[3],
                "direction_density_shape_alpha": direction[4],
                "direction_temperature_shape_alpha": direction[5],
                "root_rho_linear_change": change[ir] / np.sqrt(1000),
                "root_cost_directional_change": residuals[ir] * change[ir],
                "bootstrap_cost_directional_change": residuals[ib] * change[ib],
                "power_cost_directional_change": residuals[ip] * change[ip],
                "total_cost_directional_change": residuals @ change,
                "total_cost_quadratic_change": residuals @ change + .5 * change @ change,
            })

    for trial, event in sorted(points.items()):
        model = event["model"]
        anchor = model["from_accepted_trial"]
        if anchor not in matrices:
            continue
        j0, r0, labels = matrices[anchor]
        ir, ib, ip = [labels.index(name) for name in (ROOT, BOOT, POWER)]
        dp = np.array(model["step"][:6])
        dg = np.array(model["step"][6:])
        profile_change, geometry_change = j0[:, :6] @ dp, j0[:, 6:] @ dg
        jp_root = j0[ir, :6]
        root_profile_gradient = r0[ir] * jp_root
        chosen_steps.append({
            "trial": trial, "from_trial": anchor,
            "profile_step_norm": np.linalg.norm(dp), "geometry_step_norm": np.linalg.norm(dg),
            "root_profile_step_cosine_with_descent": -root_profile_gradient @ dp / (
                np.linalg.norm(root_profile_gradient) * np.linalg.norm(dp)),
            "profile_root_rho_linear_change": profile_change[ir] / np.sqrt(1000),
            "geometry_root_rho_linear_change": geometry_change[ir] / np.sqrt(1000),
            "profile_root_cost_directional_change": r0[ir] * profile_change[ir],
            "profile_bootstrap_cost_directional_change": r0[ib] * profile_change[ib],
            "profile_power_cost_directional_change": r0[ip] * profile_change[ip],
            "profile_total_cost_directional_change": r0 @ profile_change,
            "largest_root_rho_reduction_using_entire_actual_step_in_profiles": (
                np.linalg.norm(jp_root) * model["trust_step_norm"] / np.sqrt(1000)),
        })
    write_csv("profile_sensitivities.csv", sensitivities)
    write_csv("profile_direction_models.csv", directions)
    write_csv("chosen_profile_steps.csv", chosen_steps)
    return {
        "warning": "Hypothetical steps use the saved linear model only; no physical trial is evaluated.",
        "trial_28_sensitivities": [r for r in sensitivities if r["trial"] == 28],
        "trial_28_directions": [r for r in directions if r["trial"] == 28],
        "selected_chosen_steps": [r for r in chosen_steps if r["trial"] in (12, 22, 25, 29, 30)],
    }


def write_csv(name, records):
    with (HERE / name).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    events = [json.loads(line) for line in (HERE / "complete_events_excerpt.jsonl").read_text().splitlines()]
    points = {e["trial"]: e for e in events if e["event"] == "trial"}
    failures = {e["trial"]: e for e in events if e["event"] == "failed_trial"}
    finished = next(e for e in events if e["event"] == "finished")
    accepted = set(finished["accepted_trials"])
    with (HERE.parent / "combined_from_geometry_result_diagnostics_2026-10-16.csv").open(encoding="utf-8-sig") as source:
        console = {int(row["evaluation"]): row for row in csv.DictReader(source)}

    matrices = {}
    rows_out, points_out, models_out, failures_out = [], [], [], []
    for trial, event in sorted(points.items()):
        labels = [row["label"] for row in event["rows"]]
        residuals = np.array([row["residual"] for row in event["rows"]])
        jacobian = []
        for row in event["rows"]:
            gradient = np.array(row["gradient_profile"] + row["gradient_geometry"])
            if row["residual"] == 0:
                assert row["jacobian_norm_geometry"] == row["jacobian_norm_profile"] == 0
                jacobian.append(np.zeros_like(gradient))
            else:
                jacobian.append(gradient / row["residual"])
        jacobian = np.array(jacobian)
        # The recorder stores g_i = r_i J_i, so all nonzero-residual rows can
        # be recovered without their NPZ snapshot. Zero-weight rows are zero.
        np.testing.assert_allclose(np.linalg.norm(jacobian[:, :6], axis=1),
                                   [r["jacobian_norm_profile"] for r in event["rows"]], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(np.linalg.norm(jacobian[:, 6:], axis=1),
                                   [r["jacobian_norm_geometry"] for r in event["rows"]], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(jacobian.T @ residuals, event["gradient"], rtol=1e-11, atol=1e-10)
        np.testing.assert_allclose(.5 * residuals @ residuals, event["cost"], rtol=1e-13)
        np.testing.assert_allclose(event["cost"], float(console[trial]["total_cost"]), rtol=1e-8)
        matrices[trial] = (jacobian, residuals, labels)

        active = np.array([row["cost"] > 1e-16 for row in event["rows"]])
        singular = np.linalg.svd(jacobian[active], compute_uv=False)
        iroot, iboot, ipower = [labels.index(name) for name in (ROOT, BOOT, POWER)]
        root_j = jacobian[iroot]
        power_j = jacobian[ipower]
        power_free_root = root_j - (root_j @ power_j) / (power_j @ power_j) * power_j
        others = jacobian[active & (np.arange(len(labels)) != iroot)]
        independent_root = root_j - np.linalg.lstsq(others.T, root_j, rcond=1e-12)[0] @ others
        gn_step = np.linalg.lstsq(jacobian[active], -residuals[active], rcond=1e-12)[0]
        model = event["model"]
        points_out.append({
            "trial": trial, "accepted": trial in accepted, "cost": event["cost"],
            "step_from": model["from_accepted_trial"], "step_norm": model["trust_step_norm"],
            "geometry_step_fraction_squared": model["geometry_step_norm"] ** 2 / model["trust_step_norm"] ** 2,
            "model_ratio": model["actual_over_predicted"],
            "root_power_cosine": event["row_sensitivity_cosines"][iroot][ipower],
            "root_bootstrap_cosine": event["row_sensitivity_cosines"][iroot][iboot],
            "active_rows": int(active.sum()), "sigma_max": singular[0], "sigma_min": singular[-1],
            "active_condition_number": singular[0] / singular[-1],
            "root_sensitivity_norm": np.linalg.norm(root_j),
            "root_sensitivity_power_preserving": np.linalg.norm(power_free_root),
            "root_sensitivity_other_rows_preserving": np.linalg.norm(independent_root),
            "unconstrained_linear_root_only_step_lower_bound": abs(residuals[iroot]) / np.linalg.norm(root_j),
            "unconstrained_linear_root_step_preserving_others": abs(residuals[iroot]) / np.linalg.norm(independent_root),
            "unconstrained_joint_gauss_newton_step_norm": np.linalg.norm(gn_step),
        })
        for row in event["rows"]:
            if row["label"] not in (ROOT, BOOT, POWER):
                continue
            rows_out.append({"trial": trial, "accepted": trial in accepted,
                             **{k: row[k] for k in ("label", "residual", "cost", "jacobian_norm_geometry",
                                "jacobian_norm_profile", "gradient_norm_geometry", "gradient_norm_profile")}})

    for trial, event in sorted(points.items()):
        model = event["model"]
        anchor = model["from_accepted_trial"]
        if anchor in matrices:
            j0, r0, _ = matrices[anchor]
            np.testing.assert_allclose(j0 @ model["step"],
                                       [r["linear_residual_change"] for r in model["rows"]], atol=2e-12, rtol=1e-10)
            np.testing.assert_allclose(model["predicted_cost"], .5 * (r0 + j0 @ model["step"]) ** 2, atol=2e-12)
        for i, row in enumerate(model["rows"]):
            if row["label"] not in (ROOT, BOOT, POWER):
                continue
            actual = event["rows"][i]["residual"]
            prior = np.sqrt(2 * model["old_cost"][i])  # These three residuals are positive in the excerpt.
            models_out.append({"trial": trial, "accepted": trial in accepted, "from_trial": anchor,
                               "label": row["label"], "step_norm": model["trust_step_norm"],
                               "geometry_predicted_residual_change": row["geometry_linear_change"],
                               "profile_predicted_residual_change": row["profile_linear_change"],
                               "predicted_residual_change": row["linear_residual_change"],
                               "actual_residual_change": actual - prior,
                               "predicted_cost_change": model["predicted_cost"][i] - model["old_cost"][i],
                               "actual_cost_change": model["actual_cost_change"][i]})

    for trial, event in sorted(failures.items()):
        values = event["transport_objective_values"]
        anchor = event["model"]["from_accepted_trial"]
        new = [500 * values["Er_transition_location_moment"] ** 2,
               2 * max(values["bootstrap_current_softmax_abs_scaled"] - .1, 0) ** 2,
               5000 * (values["net_total_power_volume_average_mw_m3"] - 300 / 331.0187969899648) ** 2]
        if anchor in matrices:
            _, r0, labels = matrices[anchor]
            old = [.5 * r0[labels.index(label)] ** 2 for label in (ROOT, BOOT, POWER)]
        else:
            old = [float(console[anchor][key]) for key in ("root_location_cost", "bootstrap_cost", "net_power_cost")]
        failures_out.append({"trial": trial, "from_trial": anchor, "first_bad": event["first_nonfinite_boundary"],
                             "step_norm": event["model"]["trust_step_norm"],
                             "root_cost": new[0], "root_cost_change": new[0] - old[0],
                             "bootstrap_cost": new[1], "bootstrap_cost_change": new[1] - old[1],
                             "power_cost": new[2], "power_cost_change": new[2] - old[2],
                             "three_transport_cost_change": sum(new) - sum(old)})

    write_csv("objective_rows.csv", rows_out)
    write_csv("point_analysis.csv", points_out)
    write_csv("step_predictions.csv", models_out)
    write_csv("failed_transport_values.csv", failures_out)
    profile_result = profile_analysis(points, matrices)
    result = {"physics_solves": 0, "complete_trial_records": sorted(points),
              "partial_failure_records": sorted(failures), "active_mask_at_finish": finished["active_mask"],
              "selected_points": [p for p in points_out if p["trial"] in (11, 12, 13, 22, 25, 28, 29, 30)],
              "failed_transport_values": failures_out,
              "profile_direction_analysis": profile_result,
              "warning": "Local linear projections ignore nonlinear validity and bounds; they are diagnostics, not proposed steps."}
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
