"""Compare the two root losses from an original or saved optimized geometry.

Each candidate/group starts independently from the same saved full-time AD
point. No positive region or crossing is required to accept a trial. A trial
must lower its scalar cost, reach final time, and have matching finite AD.
Only physics workers build runtimes; the host comparison imports no JAX.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.optimization import optimize_profiles_er_root_candidate as pilot


def validate_creation_seed(report):
    metadata = report.get("optimization_start_case", {})
    saved_geometry = metadata.get("case_kind") == "saved_geometry_result"
    verified = report.get("baseline_case_verified") if saved_geometry else report.get("baseline_no_electron_region_verified")
    if not verified or not metadata:
        raise ValueError("Use the verified optimization-start AD report, not the earlier existing-root benchmark.")
    audit = report["root_loss_audit"]
    rho, er = np.asarray(audit["terminal_rho"]), np.asarray(audit["terminal_Er"])
    settings = audit["settings"]
    mask = (rho >= settings["inner_radius"]) & (rho <= settings["outer_radius"])
    if rho.shape != er.shape or not np.isfinite(er).all() or not np.any(mask):
        raise ValueError("Invalid observed baseline field/grid.")
    if not saved_geometry and (np.any(er[mask] > 0) or audit.get("nearest_downcrossing") is not None):
        raise ValueError("The comparison must start without a positive-Er region or downcrossing in the annulus.")


def geometry_result_ad_report(directory, out):
    """Prepare one new baseline AD for the accepted boundary, or reuse its report."""
    from examples.benchmarks import benchmark_er_root_candidates_optimization_start as baseline
    _, _, source = baseline.load_saved_geometry(directory)
    report_path = out / "baseline" / "ad_summary.json"
    if not report_path.exists():
        command = [sys.executable, str(Path(baseline.__file__).resolve()),
                   "--geometry-result-dir", str(Path(directory).resolve()),
                   "--out-dir", str(report_path.parent), "--ad-only"]
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            raise ValueError(f"Saved-geometry baseline AD failed; see {report_path.parent / 'ad.log'}.")
    report = json.loads(report_path.read_text())
    actual = report.get("optimization_start_case", {}).get("saved_geometry_result")
    if actual != source:
        raise ValueError("The saved geometry input/summary differs from this baseline AD; use a new output directory.")
    metadata = report.get("optimization_start_case", {})
    geometry = report.get("root_loss_audit", {}).get("experiment", {}).get("resolved_config", {}).get("geometry", {})
    if (metadata.get("saved_geometry_runtime_contract") != "optimization_boozer_resolution_v1"
            or geometry.get("mboz") != metadata.get("mboz") or geometry.get("nboz") != metadata.get("nboz")):
        raise ValueError("This cached AD predates the saved-geometry runtime resolution fix; use a new output directory.")
    return report_path


def save_comparison(out, seed_digest):
    rows = []
    for path in sorted(out.glob("*/summary.json")):
        result = json.loads(path.read_text())
        first, last = result["history"][0], result["history"][-1]
        rows.append(dict(candidate=result["candidate"], group=result["group"], status=result["status"],
                         converged=result["converged"], accepted_iterations=result["accepted_iterations"],
                         first_positive_inner_iteration=result["first_positive_inner_iteration"],
                         first_downcrossing_iteration=result["first_downcrossing_iteration"],
                         initial_has_downcrossing=first["has_downcrossing"], initial_root=first["root"],
                         initial_root_error=first["root_error"],
                         downcrossing_created=(not first["has_downcrossing"] and result["first_downcrossing_iteration"] is not None),
                         final_has_downcrossing=last["has_downcrossing"],
                         initial_cost=first["cost"], final_cost=last["cost"],
                         final_zero_cost=last["zero_cost"], final_sign_cost=last["sign_cost"],
                         final_root=last["root"], final_root_error=last["root_error"],
                         initial_inner_mean_Er=first["inner_mean_Er"], final_inner_mean_Er=last["inner_mean_Er"],
                         final_inner_positive_fraction=last["inner_positive_fraction"],
                         final_outer_negative_fraction=last["outer_negative_fraction"],
                         final_wrong_sign_fraction=last["wrong_sign_fraction"], summary=str(path)))
    pilot.write_json(out / "comparison.json", dict(seed_sha256=seed_digest, runs=rows,
        interpretation="Judge root creation when initially absent, or refinement when present; cost scales differ."))
    if rows:
        with (out / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def plot_comparison(out, seed):
    """Plot concrete accepted endpoints, with no extra physics evaluations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = sorted(out.glob("*/checkpoint.json"))
    if not paths:
        return
    fig, axes = plt.subplots(len(paths), 2, figsize=(11, 3.3 * len(paths)), squeeze=False,
                             constrained_layout=True)
    audit = seed["root_loss_audit"]
    for row, path in enumerate(paths):
        state = json.loads(path.read_text())["state"]
        endpoint, history = state["endpoint"], state["history"]
        axes[row, 0].plot(audit["terminal_rho"], audit["terminal_Er"], "k--", label="Initial")
        axes[row, 0].plot(endpoint["terminal_rho"], endpoint["terminal_Er"], label="Latest accepted")
        axes[row, 0].axhline(0, color="0.5", linewidth=.7)
        axes[row, 0].axvline(audit["target"], color="0.5", linestyle=":")
        axes[row, 0].set(title=path.parent.name, xlabel="rho", ylabel="Er [kV/m]")
        axes[row, 0].legend()
        iterations = [item["iteration"] for item in history]
        for key, label in (("inner_positive_fraction", "Positive inner fraction"),
                           ("outer_negative_fraction", "Negative outer fraction")):
            axes[row, 1].plot(iterations, [item[key] for item in history], "o-", label=label)
        axes[row, 1].set(xlabel="Accepted iteration", ylabel="Fraction", ylim=(-.03, 1.03))
        axes[row, 1].legend()
    fig.savefig(out / "root_creation_comparison.svg")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ad-report", type=Path)
    source.add_argument("--geometry-result-dir", type=Path,
                        help="Prepare baseline AD from the exported geometry-only result, then compare losses.")
    parser.add_argument("--descent-dir", type=Path, help="Reuse identical saved baseline/first forward endpoints.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/root_creation_comparison")
    parser.add_argument("--candidates", default="local,annular")
    parser.add_argument("--groups", default="profiles", help="profiles, geometry, or profiles,geometry")
    parser.add_argument("--resume", action="store_true", help="Resume existing runs and start any not yet created.")
    for name, kind, value in (("max-iterations", int, 10), ("initial-step", float, .001),
                              ("max-step", float, .05), ("min-step", float, 1e-7),
                              ("growth-factor", float, 2.), ("max-backtracks", int, 8),
                              ("root-tolerance", float, 1e-4)):
        parser.add_argument("--" + name, type=kind, default=value)
    args = parser.parse_args(argv)
    candidates, groups = args.candidates.split(","), args.groups.split(",")
    if (not set(candidates) <= {"local", "annular"} or len(candidates) != len(set(candidates))
            or not set(groups) <= {"profiles", "geometry"} or len(groups) != len(set(groups))):
        parser.error("Choose distinct local/annular candidates and profiles/geometry groups.")
    out = args.out_dir.resolve()
    try:
        options_list = []
        for candidate in candidates:
            for group in groups:
                options = pilot.PilotOptions(candidate=candidate, group=group, allow_root_creation=True,
                    **{key: getattr(args, key) for key in ("max_iterations", "initial_step", "max_step", "min_step",
                                                         "growth_factor", "max_backtracks", "root_tolerance")})
                options_list.append(options)
        report_path = (geometry_result_ad_report(args.geometry_result_dir, out)
                       if args.geometry_result_dir is not None else args.ad_report)
        seed = json.loads(report_path.read_text())
        validate_creation_seed(seed)
        for options in options_list:
            pilot.validate_seed(seed, options)
    except ValueError as exc:
        parser.error(str(exc))
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    manifest_path = out / "experiment.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["seed_sha256"] != digest:
        parser.error("This directory belongs to a different AD seed; use a new output directory.")
    pilot.write_json(manifest_path, dict(seed_sha256=digest, ad_report=str(report_path.resolve()),
        optimization_start_case=seed["optimization_start_case"],
        baseline_no_electron_region_verified=bool(seed.get("baseline_no_electron_region_verified")), root_weight=1.0))
    cache_directory = args.descent_dir
    for candidate in candidates:
        for group in groups:
            run_dir = out / f"{candidate}_{group}"
            command = ["--ad-report", str(report_path.resolve()), "--out-dir", str(run_dir),
                       "--candidate", candidate, "--group", group, "--allow-root-creation"]
            for name in ("max_iterations", "initial_step", "max_step", "min_step", "growth_factor",
                         "max_backtracks", "root_tolerance"):
                command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
            if cache_directory is not None:
                command.extend(("--descent-dir", str(cache_directory.resolve())))
            if args.resume and (run_dir / "checkpoint.json").exists():
                command.append("--resume")
            label = "saved optimized geometry" if args.geometry_result_dir is not None else "saved AD point"
            print(f"[creation-comparison] starting {candidate}/{group} from the same {label}", flush=True)
            pilot.main(command)
            if cache_directory is None:
                cache_directory = run_dir
            save_comparison(out, digest)
            plot_comparison(out, seed)
    print(f"[creation-comparison] saved {out / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
