#!/usr/bin/env python
"""Geometry-only QI optimization using NEOPAX reverse-AD internals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX import optimization as opt  # noqa: E402


SEED_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_initial"
OUT_DIR = ROOT / "outputs" / "geometry_qi_only_optimization"

MAX_MODE_SCHEDULE = (1,)
MAX_NFEV = 1
FTOL = 1.0e-6

QI_WEIGHT = 1.0
MAXJ_WEIGHT = 0.0
ASPECT_TARGET = 10.0
ASPECT_WEIGHT = 0.0
IOTA_TARGET = -0.61
IOTA_WEIGHT = 0.0
MIRROR_TARGET = 0.25
MIRROR_WEIGHT = 0.0


def qi_terms():
    return [
        (opt.geometry.boozer_qi_objective, 0.0, QI_WEIGHT),
        (opt.geometry.boozer_maxj_objective, 0.0, MAXJ_WEIGHT),
        (opt.geometry.vmec_aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
        (opt.geometry.vmec_iota_mean, IOTA_TARGET, IOTA_WEIGHT),
        (opt.geometry.vmec_mirror_ratio, MIRROR_TARGET, MIRROR_WEIGHT),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vmec-input", default=str(SEED_INPUT))
    parser.add_argument("--max-mode", type=int, default=None)
    parser.add_argument("--parameters", default=None)
    parser.add_argument("--max-nfev", type=int, default=MAX_NFEV)
    parser.add_argument("--ftol", type=float, default=FTOL)
    parser.add_argument("--xtol", type=float, default=1.0e-10)
    parser.add_argument("--mboz", type=int, default=18)
    parser.add_argument("--nboz", type=int, default=18)
    parser.add_argument("--geometry-max-iter", type=int, default=None)
    parser.add_argument("--solver-device", default="default")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def report(tag, problem, x):
    evaluation = problem.evaluate(x)
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    jacobian = np.asarray(jax.device_get(evaluation.jacobian), dtype=float)
    values = {
        label: float(np.asarray(jax.device_get(value), dtype=float))
        for label, value in evaluation.result.objective_values.items()
    }
    print(f"[{tag}] elapsed_s={evaluation.elapsed_s:.3f}")
    for label, value in values.items():
        print(f"  - {label}: value={value:.16e}")
    print(f"  residual_norm={float(np.linalg.norm(residuals)):.6e}")
    print(f"  jacobian_shape={jacobian.shape}")
    return evaluation


def main() -> int:
    args = parse_args()
    vmec_input = Path(args.vmec_input)
    max_mode_schedule = (int(args.max_mode),) if args.max_mode is not None else MAX_MODE_SCHEDULE
    terms = tuple(term for term in qi_terms() if float(term[2]) != 0.0)
    x = None
    last_problem = None
    last_result = None

    for max_mode in max_mode_schedule:
        print(f"\n===== NEOPAX geometry-only QI stage, max_mode={max_mode} =====", flush=True)
        problem = opt.geometry_least_squares_problem(
            vmec_input,
            terms,
            max_mode=max_mode if args.parameters is None else None,
            parameters=args.parameters,
            mboz=args.mboz,
            nboz=args.nboz,
            max_iter=args.geometry_max_iter,
            solver_device=args.solver_device,
        )
        if x is None or len(x) != problem.parameter_count:
            x = np.zeros((problem.parameter_count,), dtype=float)
        print(
            f"[setup] parameter_count={problem.parameter_count} "
            f"parameters={list(problem.parameter_labels)}",
            flush=True,
        )
        report("initial", problem, x)
        last_result = opt.least_squares(
            problem,
            max_nfev=int(args.max_nfev),
            ftol=float(args.ftol),
            xtol=float(args.xtol),
            verbose=1,
        )
        x = np.asarray(last_result.x, dtype=float)
        report("final", problem, x)
        last_problem = problem

    output_path = args.output
    if output_path is None:
        output_path = OUT_DIR / "geometry_qi_only_summary.json"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "vmec_input": str(vmec_input),
        "terms": [(term[0].label, float(term[1]), float(term[2])) for term in terms],
        "parameter_labels": [] if last_problem is None else list(last_problem.parameter_labels),
        "x_scaled": [] if last_result is None else np.asarray(last_result.x, dtype=float).tolist(),
        "cost": None if last_result is None else float(last_result.cost),
        "optimality": None if last_result is None else float(last_result.optimality),
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
