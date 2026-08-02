#!/usr/bin/env python
"""Geometry QI + full-transport Er/bootstrap/alpha-power optimization."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vmex as vj  # noqa: E402
from vmex import optimize as vmex_opt  # noqa: E402

from NEOPAX import optimization as opt  # noqa: E402
from NEOPAX._orchestrator import load_config, run_config  # noqa: E402


# --------------------------- parameters ------------------------------------
SEED_INPUT = ROOT / "examples" / "inputs" / "input.QI_nfp2_initial"
TRANSPORT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark_QI_optimization.toml"
)
OUT_DIR = ROOT / "outputs" / "geometry_qi_max_er_transition_bootstrap_alpha_full_transport_optimization"

SURFACES = np.asarray(
    [1 / 51, 10 / 51, 20 / 51, 30 / 51, 40 / 51, 51 / 51],
    dtype=float,
)
QI_MBOZ = 18
QI_NBOZ = 18

MAX_MODE_SCHEDULE = 2
GEOMETRY_FAMILIES = "RBC,ZBS"
SCALE_MODE = "ess"
ESS_ALPHA = 1.2

# None means the full transport solve uses t_final from the TOML.
FULL_TRANSPORT_ACCEPTED_STEP_LIMIT = None
REVERSE_SEGMENT_LENGTH = "auto_quarter"

ASPECT_TARGET = 10.0
IOTA_TARGET = -0.61
MIRROR_TARGET = 0.19
MAX_ER_TARGET = 25.0
ER_TRANSITION_LEFT_TARGET = 26.0
ER_TRANSITION_RIGHT_TARGET = -10.0
BOOTSTRAP_LIMIT_SCALED = 0.1
ALPHA_POWER_TARGET_MW = 300.0
ALPHA_REFERENCE_VOLUME_M3 = 331.0187969899648
ALPHA_POWER_TARGET_MW_M3 = ALPHA_POWER_TARGET_MW / ALPHA_REFERENCE_VOLUME_M3

QI_WEIGHT = 1.0
MAXJ_WEIGHT = 0.0001
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 1.0
MIRROR_WEIGHT = 100.0
MAX_ER_WEIGHT = 0.6
ER_TRANSITION_LEFT_WEIGHT = 1.0
ER_TRANSITION_RIGHT_WEIGHT = 1.0
BOOTSTRAP_WEIGHT = 1.0
ALPHA_POWER_WEIGHT = 1.0

NFEV = 6
FTOL = 1.0e-6
XTOL = 1.0e-10
GEOMETRY_MAX_ITER = None
SOLVER_DEVICE = "default"

MAKE_WOUT_PLOTS = True
MAKE_INITIAL_PLOTS = True
MAKE_TRANSPORT_REPORTS = True


def max_mode_schedule_values():
    if np.isscalar(MAX_MODE_SCHEDULE):
        return (int(MAX_MODE_SCHEDULE),)
    return tuple(int(value) for value in MAX_MODE_SCHEDULE)


# --------------------------- objective functions ---------------------------
def positive_part(value):
    return jnp.maximum(value, 0.0)


mirror_penalization = opt.transformed_geometry_objective(
    opt.geometry.vmec_mirror_ratio,
    lambda mirror_ratio: positive_part(mirror_ratio - MIRROR_TARGET),
    label="mirror_penalization",
)

bootstrap_penalty = opt.transformed_transport_objective(
    opt.transport.bootstrap_current_softmax_abs_scaled,
    lambda bootstrap_softmax_abs_scaled: positive_part(bootstrap_softmax_abs_scaled - BOOTSTRAP_LIMIT_SCALED),
    label="bootstrap_current_penalty",
)

terms = [
    (opt.geometry.boozer_qi_objective, 0.0, QI_WEIGHT),
    (opt.geometry.boozer_maxj_objective, 0.0, MAXJ_WEIGHT),
    (mirror_penalization, 0.0, MIRROR_WEIGHT),
    (opt.geometry.vmec_aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    (opt.geometry.vmec_iota_mean, IOTA_TARGET, IOTA_WEIGHT),
    #(opt.transport.softmax_Er, MAX_ER_TARGET, MAX_ER_WEIGHT),
    #(opt.transport.Er_transition_left, ER_TRANSITION_LEFT_TARGET, ER_TRANSITION_LEFT_WEIGHT),
    #(opt.transport.Er_transition_right, ER_TRANSITION_RIGHT_TARGET, ER_TRANSITION_RIGHT_WEIGHT),
    #(bootstrap_penalty, 0.0, BOOTSTRAP_WEIGHT),
    (opt.transport.alpha_power_volume_average_mw_m3, ALPHA_POWER_TARGET_MW_M3, ALPHA_POWER_WEIGHT),
]


# --------------------------- reporting -------------------------------------
def iteration_diagnostics(evaluation):
    values = {
        label: float(np.asarray(jax.device_get(value), dtype=float))
        for label, value in evaluation.result.objective_values.items()
    }
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    residual_lookup = {
        label: float(residuals[i])
        for i, label in enumerate(evaluation.result.residual_labels)
    }

    def value(*labels):
        for label in labels:
            if label in values:
                return values[label]
        return np.nan

    def residual(*labels):
        for label in labels:
            if label in residual_lookup:
                return residual_lookup[label]
        return np.nan

    er_residual = residual("transport:softmax_Er", "softmax_Er")
    er_cost = 0.5 * er_residual * er_residual
    alpha_power_mw_m3 = value(
        "transport:alpha_power_volume_average_mw_m3",
        "alpha_power_volume_average_mw_m3",
    )
    alpha_power_mw = alpha_power_mw_m3 * ALPHA_REFERENCE_VOLUME_M3
    return (
        f"aspect_ratio={value('geometry:vmec_aspect_ratio', 'vmec_aspect_ratio'):.8e} "
        f"iota_mean={value('geometry:vmec_iota_mean', 'vmec_iota_mean'):.8e} "
        f"mirror_ratio={value('geometry:vmec_mirror_ratio', 'vmec_mirror_ratio'):.8e} "
        f"mirror_penalty={value('mirror_penalization'):.8e} "
        f"magnetic_well={value('geometry:vmec_magnetic_well', 'vmec_magnetic_well'):.8e} "
        f"qi_cost={value('geometry:boozer_qi_objective', 'boozer_qi_objective'):.8e} "
        f"maxJ_cost={value('geometry:boozer_maxj_objective', 'boozer_maxj_objective'):.8e} "
        f"softmax_Er={value('transport:softmax_Er', 'softmax_Er'):.8e} "
        f"Er_residual={er_residual:.8e} "
        f"Er_cost={er_cost:.8e} "
        f"Er_left={value('transport:Er_transition_left', 'Er_transition_left'):.8e} "
        f"Er_left_residual={residual('transport:Er_transition_left', 'Er_transition_left'):.8e} "
        f"Er_right={value('transport:Er_transition_right', 'Er_transition_right'):.8e} "
        f"Er_right_residual={residual('transport:Er_transition_right', 'Er_transition_right'):.8e} "
        f"bootstrap_penalty={value('bootstrap_current_penalty', 'transport:bootstrap_current_penalty'):.8e} "
        f"alpha_power_mw_m3={alpha_power_mw_m3:.8e} "
        f"P_alpha_MW={alpha_power_mw:.8e} "
        f"P_alpha_residual_MW={(alpha_power_mw - ALPHA_POWER_TARGET_MW):.8e}"
    )


def report(tag, problem, x):
    evaluation = problem.evaluate(x)
    residuals = np.asarray(jax.device_get(evaluation.residuals), dtype=float)
    jacobian = np.asarray(jax.device_get(evaluation.jacobian), dtype=float)
    values = {
        label: float(np.asarray(jax.device_get(value), dtype=float))
        for label, value in evaluation.result.objective_values.items()
    }
    print(f"[{tag}] elapsed_s={evaluation.elapsed_s:.3f} {iteration_diagnostics(evaluation)}")
    for label, value in values.items():
        print(f"  - {label}: value={value:.16e}")
    print(f"  residual_norm={float(np.linalg.norm(residuals)):.6e}")
    print(f"  jacobian_shape={jacobian.shape}")
    return evaluation


def write_geometry_artifacts(input_obj, label):
    artifact_dir = OUT_DIR / label
    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_path = artifact_dir / f"input.QI_neopax_geometry_full_transport_{label}"
    input_obj.to_indata(input_path)
    print(f"wrote {input_path}")

    eq = vmex_opt.solve_equilibrium(input_obj)
    wout_path = vj.write_wout(artifact_dir / f"wout_QI_neopax_geometry_full_transport_{label}.nc", eq.wout)
    print(f"wrote {wout_path}")
    if MAKE_WOUT_PLOTS:
        for _, path in vj.plot_wout(wout_path, artifact_dir).items():
            print(f"wrote {path}")
    return artifact_dir


def write_transport_report(input_obj, label):
    """Run the forward transport solver and save its usual plot/HDF5 outputs."""

    if not MAKE_TRANSPORT_REPORTS:
        return None
    artifact_dir = OUT_DIR / label / "transport"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    input_path = artifact_dir / f"input.QI_neopax_geometry_full_transport_{label}"
    input_obj.to_indata(input_path)
    config = copy.deepcopy(load_config(TRANSPORT_CONFIG))
    config.setdefault("geometry", {})["vmec_input_file"] = str(input_path)
    transport_output = config.setdefault("transport_output", {})
    transport_output["transport_plot"] = True
    transport_output["transport_write_hdf5"] = True
    transport_output["transport_output_dir"] = str(artifact_dir)
    print(f"[transport-report] running {label} forward transport output", flush=True)
    result = run_config(config)
    print(f"[transport-report] wrote usual transport outputs in {artifact_dir}", flush=True)
    return result


def write_outputs(optimized_input, initial_input):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_copy = OUT_DIR / SEED_INPUT.name
    optimized_input_path = OUT_DIR / "input.QI_neopax_geometry_full_transport_optimized"
    initial_input.to_indata(seed_copy)
    optimized_input.to_indata(optimized_input_path)
    print(f"wrote {seed_copy}")
    print(f"wrote {optimized_input_path}")
    if MAKE_INITIAL_PLOTS:
        write_geometry_artifacts(initial_input, "initial")
    write_transport_report(initial_input, "initial")
    write_geometry_artifacts(optimized_input, "optimized")
    write_transport_report(optimized_input, "optimized")


class GeometryInputSavingProblem:
    """Save the VMEC input corresponding to every successful optimizer evaluation."""

    def __init__(self, problem, artifact_dir):
        self.problem = problem
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._seen = {}

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def evaluate(self, scaled_parameter_values=None):
        evaluation = self.problem.evaluate(scaled_parameter_values)
        x = self.problem.x0 if scaled_parameter_values is None else jnp.asarray(
            scaled_parameter_values,
            dtype=jnp.float64,
        )
        key = tuple(np.asarray(jax.device_get(x), dtype=float).tolist())
        if key not in self._seen:
            index = len(self._seen)
            self._seen[key] = index
            input_path = self.artifact_dir / f"input.QI_neopax_geometry_full_transport_eval_{index:04d}"
            self.problem.input_from_scaled_parameters(x).to_indata(input_path)
            print(f"wrote {input_path}", flush=True)
        return evaluation


# --------------------------- continuation ladder ----------------------------
def main() -> int:
    active_terms = tuple(term for term in terms if float(term[2]) != 0.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x = None
    current_input = SEED_INPUT
    optimized_input = None
    initial_input = None
    last_problem = None
    last_result = None

    max_mode_schedule = max_mode_schedule_values()
    for max_mode in max_mode_schedule:
        print(f"\n===== NEOPAX geometry QI + full-transport stage, max_mode={max_mode} =====", flush=True)
        problem = opt.geometry_full_transport_least_squares_problem(
            TRANSPORT_CONFIG,
            active_terms,
            vmec_input=current_input,
            max_mode=max_mode,
            families=GEOMETRY_FAMILIES,
            scale_mode=SCALE_MODE,
            ess_alpha=ESS_ALPHA,
            mboz=QI_MBOZ,
            nboz=QI_NBOZ,
            surfaces=tuple(float(s) for s in SURFACES),
            geometry_max_iter=GEOMETRY_MAX_ITER,
            geometry_solver_device=SOLVER_DEVICE,
            device=SOLVER_DEVICE,
            accepted_step_limit=FULL_TRANSPORT_ACCEPTED_STEP_LIMIT,
            reverse_segment_length=REVERSE_SEGMENT_LENGTH,
            initial_er_root_ad="jax_selected_root",
            radau_jacobian_reuse_mode="legacy",
            reverse_stage_adjoint_solve_mode="bicgstab",
            reverse_rhs_transpose_mode="explicit_ntx_interpolated",
            reverse_step_bwd_mode="reduced_cotangent",
        )
        problem = GeometryInputSavingProblem(
            problem,
            OUT_DIR / f"geometry_inputs_m{max_mode}",
        )
        if x is None or len(x) != problem.parameter_count:
            x = np.asarray(jax.device_get(problem.x0), dtype=float)
        print(
            f"[setup] parameter_count={problem.parameter_count} "
            f"parameters={list(problem.parameter_labels)}",
            flush=True,
        )
        print(
            "[setup] full_transport "
            f"accepted_step_limit={FULL_TRANSPORT_ACCEPTED_STEP_LIMIT} "
            f"reverse_segment_length={REVERSE_SEGMENT_LENGTH}",
            flush=True,
        )
        if initial_input is None:
            initial_input = problem.input_from_scaled_parameters(x)
        report("initial", problem, x)
        last_result = opt.least_squares(
            problem,
            max_nfev=NFEV,
            ftol=FTOL,
            xtol=XTOL,
            verbose=1,
            iteration_reporter=iteration_diagnostics,
        )
        x = np.asarray(last_result.x, dtype=float)
        report(f"QI + full-transport stage {max_mode}", problem, x)
        optimized_input = problem.input_from_scaled_parameters(x)
        stage_input = OUT_DIR / f"input.QI_neopax_geometry_full_transport_stage_m{max_mode}"
        optimized_input.to_indata(stage_input)
        print(f"wrote {stage_input}")
        current_input = stage_input
        x = None
        last_problem = problem

    if optimized_input is None or initial_input is None or last_problem is None or last_result is None:
        raise RuntimeError("No optimization stage was executed.")

    summary = {
        "seed_input": str(SEED_INPUT),
        "transport_config": str(TRANSPORT_CONFIG),
        "max_mode_schedule": list(max_mode_schedule),
        "accepted_step_limit": FULL_TRANSPORT_ACCEPTED_STEP_LIMIT,
        "reverse_segment_length": REVERSE_SEGMENT_LENGTH,
        "parameter_labels": list(last_problem.parameter_labels),
        "x": np.asarray(last_result.x, dtype=float).tolist(),
        "cost": float(last_result.cost),
        "optimality": float(last_result.optimality),
        "status": int(last_result.status),
        "message": str(last_result.message),
    }
    summary_path = OUT_DIR / "optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    write_outputs(optimized_input, initial_input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
