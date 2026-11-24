"""Run the existing AD/FD benchmarks with generic or radial NTSS1D interpolation.

The parent process imports no JAX. Workers run sequentially on the GPU.
--plan-only prints the commands without importing JAX or starting workers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
MODE = "preprocessed_3d_radial_ntss1d"
CONFIG = ROOT/"examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_ntss1d.toml"
PARAMETERS = ("n0", "T0", "density_shape_power", "temperature_shape_power",
              "density_shape_alpha", "temperature_shape_alpha", "RBC:1:0", "ZBS:1:0")
AD_SCRIPT = "examples/benchmarks/benchmark_transport_reverse_ad_only.py"
FD_SCRIPT = "examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py"
AD_OPTIONS = (
    "--vmec-response-mode", "raw_block",
    "--reverse-parameter-mode", "profiles_plus_realtime_geometry",
    "--reverse-geometry-parameter", "RBC:1:0,ZBS:1:0",
    "--realtime-geometry-gradient-path", "reverse_payload",
    "--optimization-api-profile-dofs", "include", "--objective", "all",
    "--reverse-segment-length", "4", "--radau-jacobian-reuse-mode", "legacy",
    "--reverse-replay-jacobian-reuse-mode", "recorded_forward", "--timing-mode", "jit-warm",
    "--reverse-stage-adjoint-solve-mode", "block",
    "--reverse-rhs-transpose-mode", "explicit_database",
    "--reverse-step-bwd-mode", "reduced_cotangent_call_boundary",
    "--reverse-initial-cache-support-pullback-mode", "scalar",
    "--reverse-rebuild-support-pullback-mode", "separate",
    "--reverse-final-objective-cotangent-mode", "grouped_vjp",
    "--reverse-bootstrap-cotangent-mode", "joint_local_vjp_upar_only",
    "--reverse-database-interpolation-transpose-mode", "established",
    "--reverse-database-root-interpolation-transpose-mode", "established",
    "--reverse-database-bootstrap-interpolation-transpose-mode", "established",
    "--initial-Er-root-ad", "jax_selected_root", "--full-transport-shared-payload-smoke",
    "--reverse-schedule-artifact-mode", "reuse_static_probe",
)
FD_OPTIONS = (
    "--device", "auto",
    "--vmec-response-mode", "raw_block", "--radau-jacobian-reuse-mode", "legacy",
    "--geometry-fd-lane", "frozen_linearized", "--replay-mode", "accepted",
    "--fd-accepted-replay-contract", "reverse_frozen_metadata",
    "--initial-Er-root-ad", "jax_selected_root",
    "--initial-Er-root-fd-root-lane", "frozen_linearized", "--diagnose-fd-endpoint",
)
AD_REPORT = ROOT/"outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_full_transport_shared_payload_smoke_replay_jacobian_recorded_forward.json"
FD_REPORTS = ROOT/"outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd"


def commands(args):
    common = ["--config", str(args.config), "--accepted-step-limit", str(args.accepted_step_limit)]
    yield "ad", [sys.executable, AD_SCRIPT, *common, *AD_OPTIONS], AD_REPORT
    for parameter in PARAMETERS:
        safe = parameter.replace(":", "_")
        reference = ["--ad-baseline-reference", str(args.out_dir/"ad/baseline_reference.npz")]
        if ":" in parameter:
            reference += ["--ad-geometry-reference", str(args.out_dir/"ad/snapshots"),
                          "--geometry-fd-native-precision", "extended"]
        yield f"fd_{safe}", [sys.executable, FD_SCRIPT, *common, *FD_OPTIONS,
            *reference,
            "--parameter", parameter, "--fd-rel-step", str(args.fd_rel_step),
            "--fd-abs-step", str(args.fd_abs_step)], FD_REPORTS/f"{safe}_forward_fd_frozen_linearized_summary.json"


def prepare_ad_reference(out_dir):
    captures = list((out_dir/"ad/snapshots").glob("ad_terminal_*.npz"))
    if len(captures) != 1:
        raise ValueError(f"Expected one AD terminal capture; found {len(captures)}")
    reference = out_dir/"ad/baseline_reference.npz"
    if reference.exists() and reference.read_bytes() != captures[0].read_bytes():
        raise ValueError("Saved baseline reference differs from its AD capture")
    if not reference.exists():
        shutil.copy2(captures[0], reference)


def worker_environment():
    env = {k: v for k, v in os.environ.items() if k not in {
        "PYTHONPATH", "LD_LIBRARY_PATH", "JAX_PLATFORM_NAME", "JAX_COMPILATION_CACHE_DIR",
        "XLA_FLAGS", "XLA_PYTHON_CLIENT_ALLOCATOR", "NEOPAX_GRADIENT_AUDIT_DIR"}}
    env.update(JAX_PLATFORMS="cuda,cpu", JAX_ENABLE_X64="true", JAX_ENABLE_COMPILATION_CACHE="0",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", NTX_DISABLE_COMPILATION_CACHE="1",
        VMEX_COMPILATION_CACHE="disabled", NEOPAX_REVERSE_STEP_DIAGNOSTICS="1",
        NEOPAX_TEST_RUNTIME_NTSS1D="1")
    return env


def worker(command, directory, env):
    directory.mkdir(parents=True, exist_ok=True)
    local_env = dict(env, NEOPAX_GRADIENT_AUDIT_DIR=str(directory/"snapshots"))
    if Path("/usr/bin/time").exists():
        command = ["/usr/bin/time", "-v", *command]
    print("[ad-fd-comparison] " + shlex.join(command), flush=True)
    with (directory/"run.log").open("w") as stream:
        with subprocess.Popen(command, cwd=ROOT, env=local_env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            if process.wait():
                raise RuntimeError(f"Worker failed; see {directory/'run.log'}")


def compare(out_dir):
    from check_saved_transport_baseline_repeatability import differences, read_one
    ad = json.loads((out_dir/"ad/summary.json").read_text())
    columns = {label: i for i, label in enumerate(ad["parameter_order"])}
    rows_by_label = {label: i for i, label in enumerate(ad["residual_labels"])}
    data = read_one(out_dir/"ad/snapshots", "ad_terminal_*.npz")
    reference = {k: data[k] for k in ("initial_y", "final_y", "final_time")}
    prefixes = ("support['geometry'].", "support['database'].")
    support = {k: v for k, v in data.items() if k.startswith(prefixes)}
    rows = []
    for parameter in PARAMETERS:
        folder = out_dir/f"fd_{parameter.replace(':', '_')}"
        if not (folder/"summary.json").exists():
            continue
        fd = json.loads((folder/"summary.json").read_text())
        data = read_one(folder/"snapshots", "fd_baseline_*.npz")
        state_rows = differences(reference, {k: data[k] for k in reference if k in data})
        support_rows = differences(support, {k: v for k, v in data.items() if k.startswith(prefixes)})
        deltas = {k: 0. for k in reference}
        deltas.update({row["field"]: row.get("max_abs", float("inf")) for row in state_rows})
        support_difference = max((row.get("max_abs", float("inf")) for row in support_rows), default=0.)
        support_matched = bool(support) and not support_rows
        matched = not state_rows
        column = columns["vmec:"+parameter if ":" in parameter else parameter]
        for objective, value, gradient_fd in zip(fd["objective_labels"], fd["objective_values"], fd["gradient_fd"], strict=True):
            row = rows_by_label["transport:"+objective]
            gradient_ad = ad["jacobian"][row][column]
            objective_difference = abs(ad["residuals"][row]-value)
            objective_matches = objective_difference <= 1e-12*(1+abs(ad["residuals"][row]))
            difference = abs(gradient_ad-gradient_fd)
            relative = difference/abs(gradient_fd) if gradient_fd else (0. if not difference else float("inf"))
            rows.append(dict(database_interpolation_mode=ad.get("database_interpolation_mode", MODE),
                parameter=parameter, objective=objective, gradient_ad=gradient_ad,
                gradient_fd=gradient_fd, absolute_difference=difference, relative_difference_to_fd=relative,
                baseline_arrays_match=matched, initial_y_max_abs_difference=deltas["initial_y"],
                baseline_support_arrays_match=support_matched,
                baseline_support_max_abs_difference=support_difference,
                baseline_objective_matches=objective_matches,
                comparison_baseline_matches=matched and support_matched and objective_matches,
                final_y_max_abs_difference=deltas["final_y"],
                final_time_abs_difference=deltas["final_time"],
                baseline_objective_abs_difference=objective_difference))
        print(f"[ad-fd-comparison] {parameter}: baseline_arrays_match={matched} "
              f"support_arrays_match={support_matched} support_max_abs_difference={support_difference:.9e} {deltas}", flush=True)
    if rows:
        with (out_dir/"ad_vs_fd.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows({key: f"{value:.16e}" if isinstance(value, float) else value
                              for key, value in row.items()} for row in rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--accepted-step-limit", type=int, default=16)
    parser.add_argument("--fd-rel-step", type=float, default=3e-7)
    parser.add_argument("--fd-abs-step", type=float, default=1e-10)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.out_dir = args.out_dir.resolve()
    config = tomllib.loads(args.config.read_text())
    mode = config.get("neoclassical", {}).get("interpolation_mode", "generic")
    if mode not in ("generic", MODE):
        parser.error("This comparison requires generic or preprocessed_3d_radial_ntss1d interpolation.")
    if config["neoclassical"].get("flux_model") != "ntx_scan_runtime" or config.get("geometry", {}).get("vmec_lane") != "ad":
        parser.error("This comparison requires the live NTX scan and the common VMEC ad-lane equilibrium.")
    if args.accepted_step_limit <= 0 or args.fd_rel_step <= 0 or args.fd_abs_step <= 0:
        parser.error("Step limits and FD increments must be positive.")
    tasks = list(commands(args))
    if args.plan_only:
        for name, command, _ in tasks:
            print(name + ": " + shlex.join(command))
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # A resumed comparison must retain the same input, increments and source.
    digest = hashlib.sha256()
    sources = [path for path in (ROOT/"NEOPAX").glob("*.py") if path.name != "version.py"]
    helpers = [ROOT/"examples/benchmarks"/name for name in
               ("_fd_native_precision.py", "check_saved_transport_baseline_repeatability.py")]
    for path in sorted([*sources, *helpers, ROOT/AD_SCRIPT, ROOT/FD_SCRIPT, Path(__file__)]):
        digest.update(path.read_bytes())
    settings = args.config.read_text() + f"\n# comparison: steps={args.accepted_step_limit} rel={args.fd_rel_step} abs={args.fd_abs_step} source={digest.hexdigest()}\n"
    saved = args.out_dir/"config.toml"
    if saved.exists() and saved.read_text() != settings:
        parser.error("Saved run settings/source differ; choose a new output directory.")
    saved.write_text(settings)
    env = worker_environment()
    if mode == MODE:
        worker([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                "-p", "test_runtime_ntss1d_interpolation.py", "-v"], args.out_dir/"kernel_check", env)
    for name, command, report_path in tasks:
        folder = args.out_dir/name
        result = folder/"summary.json"
        if not result.exists():
            worker(command, folder, env)
            shutil.copy2(report_path, result)
        report = json.loads(result.read_text())
        if report.get("database_interpolation_mode") != mode:
            raise RuntimeError(f"Worker did not report {mode} interpolation: {result}")
        if report.get("accepted_step_limit") != args.accepted_step_limit or report.get("vmec_response_mode") != "raw_block":
            raise RuntimeError(f"Worker settings differ from the requested comparison: {result}")
        if name == "ad":
            expected = {"vmec:"+p if ":" in p else p for p in PARAMETERS}
            if (not report["all_finite"] or len(report["parameter_order"]) != len(PARAMETERS)
                    or set(report["parameter_order"]) != expected):
                raise RuntimeError("AD must provide finite gradients for all eight requested DOFs.")
            objectives = {label.removeprefix("transport:") for label in report["residual_labels"]
                          if label.startswith("transport:")}
            if len(objectives) != 13:
                raise RuntimeError("AD must provide all thirteen transport objectives.")
            prepare_ad_reference(args.out_dir)
        if name != "ad":
            if report["parameter_name"].replace(":", "_") != name[3:]:
                raise RuntimeError(f"FD report belongs to a different parameter: {result}")
            if not all(math.isfinite(value) for value in (*report["objective_values"], *report["gradient_fd"])):
                raise RuntimeError(f"FD values or gradients are nonfinite: {result}")
            if len(report["objective_labels"]) != 13 or set(report["objective_labels"]) != objectives:
                raise RuntimeError(f"FD objectives differ from the AD transport objectives: {result}")
            rows = compare(args.out_dir)
            if not all(row["comparison_baseline_matches"] for row in rows):
                raise RuntimeError("FD baseline differs from AD; comparison CSV retained, next worker not started.")
    print(f"[ad-fd-comparison] {mode}: saved all 104 pairs: {args.out_dir/'ad_vs_fd.csv'}", flush=True)


if __name__ == "__main__":
    main()
