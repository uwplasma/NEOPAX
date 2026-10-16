"""Analyze transcribed small matrices; never import NEOPAX/JAX or run transport.

The Jacobian entries have the precision printed in the user's console, while
the separate spectrum CSV preserves the reported full-matrix singular values.
Optional --linear-trf runs SciPy on 11 linear residual rows and six variables,
with three active rows, to compare the first bounded step. It is not a
physics/AD validation. Constant geometry costs are omitted from this model.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PARAMETERS = ("n0", "T0", "density_shape_power", "temperature_shape_power",
              "density_shape_alpha", "temperature_shape_alpha")
OBJECTIVES = ("root_location", "bootstrap", "net_power")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-trf", action="store_true")
    args = parser.parse_args()
    with (HERE / "profiles_only_initial_jacobian_comparison_2026-10-16.csv").open(newline="") as stream:
        entries = {(e["objective"], e["parameter"]): e for e in csv.DictReader(stream)}
    with (HERE / "profiles_only_trial_0002_coordinate_comparison_2026-10-16.csv").open(newline="") as stream:
        coordinates = {e["parameter"]: e for e in csv.DictReader(stream)}
    with (HERE / "profiles_only_initial_jacobian_spectrum_2026-10-16.csv").open(newline="") as stream:
        spectrum = list(csv.DictReader(stream))

    baseline = np.array([float(coordinates[p]["baseline"]) for p in PARAMETERS])
    old_step = np.array([float(coordinates[p]["replay_delta"]) for p in PARAMETERS])
    new_step = np.array([float(coordinates[p]["new_delta"]) for p in PARAMETERS])
    step_difference = new_step - old_step
    lower = np.array([0.6, 5.0, 0.1, 0.1, 0.1, 0.1]) / baseline - 1
    upper = np.array([10.0, 25.0, 12.0, 12.0, 12.0, 12.0]) / baseline - 1
    np.set_printoptions(precision=10, suppress=True)
    for mode, step in (("old", old_step), ("new", new_step)):
        jacobian = np.array([[float(entries[o, p][mode + "_jacobian"])
                              for p in PARAMETERS] for o in OBJECTIVES])
        residual = np.array([float(entries[o, PARAMETERS[0]][mode + "_residual"])
                             for o in OBJECTIVES])
        print(f"\n{mode}: columns={PARAMETERS}")
        for i, label in enumerate(OBJECTIVES):
            norm = np.linalg.norm(jacobian[i])
            print(f"  {label}: J norm={norm:.10g} cost-gradient norm={abs(residual[i]) * norm:.10g}")
        predicted_change = jacobian @ step
        print("  predicted root-radius change:", predicted_change[0] / np.sqrt(1000))
        print("  predicted power change, MW:", predicted_change[2] * 331.0187969899648 / 100)
        _, _, vt = np.linalg.svd(jacobian, full_matrices=True)
        null_part = vt[3:] @ step_difference
        print("  step-difference squared fraction in this model's null space:",
              np.dot(null_part, null_part) / np.dot(step_difference, step_difference))
        print("  reported full-matrix spectrum:", [float(e[mode + "_singular_value"]) for e in spectrum])
        print("  reported largest/third singular-value ratio:",
              float(spectrum[0][mode + "_singular_value"]) / float(spectrum[2][mode + "_singular_value"]))
        if args.linear_trf:
            from scipy.optimize import least_squares

            calls = []
            # Preserve m=11 >= n=6 and the production active-row positions.
            # SciPy's dense trust-region solver uses m in its full-rank test;
            # using just three rows changes that numerical decision.
            full_jacobian = np.zeros((11, 6))
            full_residual = np.zeros(11)
            full_jacobian[[6, 9, 10]] = jacobian
            full_residual[[6, 9, 10]] = residual

            def linear_residual(x):
                calls.append(x.copy())
                return full_residual + full_jacobian @ x

            least_squares(linear_residual, np.zeros(6), jac=lambda x: full_jacobian,
                          bounds=(lower, upper), x_scale=1.0, max_nfev=2,
                          ftol=1e-6, xtol=1e-10)
            proposed = calls[1]
            print("  linear TRF first proposal:", proposed)
            print("  recorded optimizer proposal:", step)
            print("  maximum coordinate difference:", np.max(np.abs(proposed - step)))


if __name__ == "__main__":
    main()
