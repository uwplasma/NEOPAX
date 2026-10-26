"""Forward-only radial convergence at two saved profile vectors.

The baseline and one accepted profile point are held fixed while only
geometry.n_radial changes. Matching saved forward endpoints are reused.
Each new full-time solve runs in a separate worker. No optimization or
reverse AD is run. Host planning, cache checks and summaries import no JAX.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmarks import analyze_er_root_saved_profiles as saved
from examples.benchmarks import check_er_root_candidate_descent as forward
from examples.optimization.optimize_profiles_er_root_candidate import (
    PHYSICS_KEYS, PROFILES, forward_key, write_json,
)


def make_plan(report, checkpoint, grids, iteration):
    audit = report["root_loss_audit"]
    experiment = audit["experiment"]
    config = experiment["resolved_config"]
    labels = report["parameter_order"]
    state = checkpoint["state"]
    other = state["report"]
    if (not report.get("all_finite") or report.get("accepted_step_limit") is not None
            or config["transport_solver"].get("stop_after_accepted_steps") is not None
            or experiment["initial_er_root_ad"] != "jax_selected_root"):
        raise ValueError("Use the full-time, finite selected-root AD seed from the completed baseline.")
    if (other["parameter_order"] != labels
            or other["root_loss_audit"]["experiment"]["resolved_config"] != config
            or other["root_loss_audit"]["settings"] != audit["settings"]):
        raise ValueError("The profile run and baseline report describe different physical cases.")
    if state.get("group") != "profiles":
        raise ValueError("Select a profiles-only optimization run.")
    baseline = np.asarray(experiment["parameter_values"], dtype=float)
    history = state["history"]
    if not history or not np.array_equal(history[0]["parameters"], baseline):
        raise ValueError("The checkpoint did not start from this baseline parameter vector.")
    accepted = next((row for row in history if row["iteration"] == iteration), None)
    if accepted is None or iteration <= 0:
        raise ValueError(f"Accepted iteration {iteration} is not in the saved checkpoint.")
    point = np.asarray(accepted["parameters"], dtype=float)
    if (point.shape != baseline.shape or baseline.shape != (len(labels),)
            or not np.isfinite([baseline, point]).all()
            or len(set(labels)) != len(labels)
            or {label for label in labels if ":" not in label} != set(PROFILES)):
        raise ValueError("Invalid profile vectors or parameter labels.")
    for index, label in enumerate(labels):
        if ":" in label:
            if baseline[index] != 0 or point[index] != baseline[index]:
                raise ValueError("Use the saved optimized boundary with zero, fixed geometry offsets.")
        elif baseline[index] <= 0 or point[index] <= 0:
            raise ValueError("Profile parameters must be positive.")
    grids = tuple(int(n) for n in grids)
    if not grids or len(set(grids)) != len(grids) or any(n < 3 for n in grids):
        raise ValueError("Choose distinct radial cell counts of at least three.")
    rho = np.asarray(audit["terminal_rho"], dtype=float)
    if rho.shape != (int(config["geometry"]["n_radial"]),):
        raise ValueError("The saved baseline grid differs from its recorded configuration.")
    target = audit["settings"]["target"]
    left, right = np.searchsorted(rho, target, side="left") - 1, np.searchsorted(rho, target, side="right")
    if not 0 <= left < right < len(rho):
        raise ValueError("The target must have original-grid neighbors on both sides.")
    seed = Path(config["geometry"]["vmec_input_file"])
    digest = hashlib.sha256(seed.read_bytes()).hexdigest()
    recorded = report.get("optimization_start_case", {}).get("seed_sha256")
    if recorded is None or recorded != digest:
        raise ValueError("The saved geometry input changed or has no recorded hash; select its matching baseline report.")
    base = dict(resolved_config=config, parameter_labels=labels, settings=audit["settings"],
                initial_er_root_ad=experiment["initial_er_root_ad"],
                profile_contract="ad_initial_profiles_fixed_config_selected_root",
                source_digest=forward.source_digest(config))
    tasks = []
    for n in grids:
        for name, values in (("baseline", baseline), (f"accepted_{iteration:04d}", point)):
            task = copy.deepcopy(base)
            task["resolved_config"]["geometry"]["n_radial"] = n
            task.update(name=f"n{n}_{name}", parameter_values=values.tolist())
            tasks.append(task)
    return dict(tasks=tasks, iteration=iteration, grids=list(grids), seed_sha256=digest,
                fixed_probe_radii=[float(rho[left]), float(target), float(rho[right])],
                baseline_reference=dict(n_radial=len(rho), terminal_rho=rho.tolist(),
                                        terminal_Er=audit["terminal_Er"], final_time=audit["terminal_time"]),
                interpretation="Only geometry.n_radial changes. Initial profiles/roots are rebuilt on each grid. "
                "Native local-cost sign probes move with the grid; compare actual crossings and fixed-radii Er.")


def validate_endpoint(task, endpoint):
    n = task["resolved_config"]["geometry"]["n_radial"]
    rho, er = np.asarray(endpoint["terminal_rho"]), np.asarray(endpoint["terminal_Er"])
    initial = np.asarray(endpoint.get("initial_state", {}).get("Er", []))
    expected = (np.arange(n) + .5) / n
    if (rho.shape != (n,) or er.shape != (n,) or initial.shape != (n,)
            or not np.isfinite([rho, er, initial]).all()
            or not np.allclose(rho, expected, rtol=1e-10, atol=1e-12)
            or not np.array_equal(endpoint["parameter_values"], task["parameter_values"])
            or not np.isclose(endpoint["final_time"], task["resolved_config"]["transport_solver"]["t_final"],
                              rtol=0, atol=1e-10)):
        raise ValueError(f"Invalid or incomplete forward endpoint for {task['name']}; inspect its worker log.")
    costs = endpoint["metrics"]["candidates"]
    if not all(np.isfinite(costs[name]["cost"]) for name in ("audit_root_local_cost", "audit_root_annular_cost")):
        raise ValueError(f"Nonfinite candidate cost in {task['name']}.")


def validate_baseline_reference(plan, task, endpoint):
    reference = plan["baseline_reference"]
    if (task["name"].split("_", 1)[1] == "baseline"
            and task["resolved_config"]["geometry"]["n_radial"] == reference["n_radial"]):
        if (not np.allclose(endpoint["terminal_Er"], reference["terminal_Er"], rtol=1e-6, atol=1e-7)
                or not np.isclose(endpoint["final_time"], reference["final_time"], rtol=0, atol=1e-10)):
            raise ValueError("The original-grid baseline does not reproduce its saved AD field; "
                             "inspect the endpoint before interpreting refinement.")


def load_cache(directories):
    cache = {}
    for directory in directories:
        for path in Path(directory).glob("*_task.json"):
            result_path = path.with_name(path.name.replace("_task.json", ".json"))
            if not result_path.exists():
                continue
            try:
                packet = saved.read_json(path)
                if packet.get("kind") == "ad":
                    continue
                task = packet.get("forward_task", packet)
                if not all(key in task for key in PHYSICS_KEYS):
                    continue
                endpoint = saved.read_json(result_path)
                if endpoint.get("task_digest") != forward.task_digest(task):
                    continue
                validate_endpoint(task, endpoint)
            except (ValueError, KeyError, TypeError):
                continue  # Partial/stale files cannot establish an identical completed solve.
            cache[forward_key(task)] = (endpoint, str(result_path.resolve()))
    return cache


def run_worker(task, task_path, result_path, log_path):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-task", str(task_path),
               "--worker-output", str(result_path)]
    print(f"[radial-resolution] forward solve {task['name']}; log={log_path}", flush=True)
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise ValueError(f"Forward worker failed for {task['name']}; see {log_path}.")
    return saved.read_json(result_path)


def level_width(er, rho, root):
    """Interpolated +/-2 kV/m width near the root; not proof of a resolved front."""
    if root is None:
        return dict(interpolated_width_pm2=None, centers_inside_pm2=None)
    points = []
    for level in (2., -2.):
        indices = np.flatnonzero((er[:-1] >= level) & (er[1:] < level))
        q = rho[indices] + np.diff(rho)[indices] * (er[indices] - level) / (er[indices] - er[indices+1])
        q = q[(q <= root) & (q >= root-.05)] if level > 0 else q[(q >= root) & (q <= root+.05)]
        if not len(q):
            return dict(interpolated_width_pm2=None, centers_inside_pm2=None)
        points.append(float(q[np.argmin(abs(q-root))]))
    return dict(interpolated_width_pm2=points[1]-points[0],
                centers_inside_pm2=int(np.sum((rho >= points[0]) & (rho <= points[1]))))


def endpoint_row(task, endpoint, probes, result_path, reused_from):
    settings = {k: task["settings"][k] for k in ("target", "inner_radius", "outer_radius")}
    rho, er = np.asarray(endpoint["terminal_rho"]), np.asarray(endpoint["terminal_Er"])
    initial = saved.crossing(endpoint["initial_state"]["Er"], rho, **settings)
    final = saved.crossing(er, rho, **settings)
    candidates = endpoint["metrics"]["candidates"]
    fields = np.interp(probes, rho, er)
    n = task["resolved_config"]["geometry"]["n_radial"]
    return dict(name=task["name"], n_radial=n, point=task["name"].split("_", 1)[1],
                spacing=1./n, initial_root=initial["root"], initial_bracket=initial["bracket"],
                initial_Er_at_target=initial["Er_at_target"], root=final["root"],
                root_error=abs(final["root"]-settings["target"]) if final["root"] is not None else None,
                bracket=final["bracket"], crossing_fraction=final["fraction"],
                Er_at_target=final["Er_at_target"],
                Er_left_fixed_probe=float(fields[0]), Er_right_fixed_probe=float(fields[2]),
                **level_width(er, rho, final["root"]),
                native_local_cost=candidates["audit_root_local_cost"]["cost"],
                native_annular_cost=candidates["audit_root_annular_cost"]["cost"],
                accepted_steps=endpoint["accepted_steps"], attempts=endpoint["attempts"],
                final_time=endpoint["final_time"], endpoint_file=str(result_path), reused_from=reused_from)


def differences(rows):
    pairs, convergence = [], []
    subtract = lambda a, b: a-b if a is not None and b is not None else None
    for n in sorted({row["n_radial"] for row in rows}):
        group = [r for r in rows if r["n_radial"] == n]
        if len(group) == 2:
            a = next(r for r in group if r["point"] == "baseline")
            b = next(r for r in group if r["point"] != "baseline")
            pairs.append(dict(n_radial=n, spacing=1./n,
                initial_root_change=subtract(b["initial_root"], a["initial_root"]),
                root_change=subtract(b["root"], a["root"]),
                root_change_in_cells=n*subtract(b["root"], a["root"]) if a["root"] is not None and b["root"] is not None else None,
                root_error_change=subtract(b["root_error"], a["root_error"]),
                Er_at_target_change=b["Er_at_target"]-a["Er_at_target"],
                native_local_cost_change=b["native_local_cost"]-a["native_local_cost"],
                native_annular_cost_change=b["native_annular_cost"]-a["native_annular_cost"]))
    for point in sorted({row["point"] for row in rows}):
        group = sorted((r for r in rows if r["point"] == point), key=lambda r: r["n_radial"])
        for a, b in zip(group, group[1:]):
            convergence.append(dict(point=point, coarse=a["n_radial"], fine=b["n_radial"],
                initial_root_difference=subtract(b["initial_root"], a["initial_root"]),
                root_difference=subtract(b["root"], a["root"]),
                Er_at_target_difference=b["Er_at_target"]-a["Er_at_target"]))
    return pairs, convergence


def save_tables(plan, rows, out):
    pairs, convergence = differences(rows)
    write_json(out / "summary.json", dict(rows=rows, profile_moves=pairs, convergence=convergence,
        fixed_probe_radii=plan["fixed_probe_radii"], interpretation=plan["interpretation"],
        width_note="Width uses interpolated +/-2 kV/m crossings within 0.05 of the root. "
                   "A sub-cell width with zero interior centers is not a resolved physical width."))
    if rows:
        saved.write_csv(out / "endpoints.csv", rows)
    if pairs:
        saved.write_csv(out / "profile_moves.csv", pairs)
    if convergence:
        saved.write_csv(out / "convergence.csv", convergence)


def plot_results(plan, endpoints, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for task, endpoint in endpoints:
        column = int(task["name"].split("_", 1)[1] != "baseline")
        n = task["resolved_config"]["geometry"]["n_radial"]
        for row, field in enumerate((endpoint["initial_state"]["Er"], endpoint["terminal_Er"])):
            axes[row, column].plot(endpoint["terminal_rho"], field, ".-", label=f"{n} cells", markersize=3)
    settings = plan["tasks"][0]["settings"]
    for row in range(2):
        for column in range(2):
            axis = axes[row, column]
            axis.axvline(settings["target"], color="black", linestyle=":")
            axis.axhline(0, color="0.5", linewidth=.7)
            axis.set(xlabel="rho", ylabel="Er [kV/m]",
                     xlim=(max(settings["inner_radius"], settings["target"]-.1),
                           min(settings["outer_radius"], settings["target"]+.1)),
                     title=f"{'Initial' if row == 0 else 'Final'}: {'baseline' if column == 0 else 'accepted profile point'}")
            axis.legend()
    fig.savefig(out / "radial_profiles.svg")
    plt.close(fig)


def run_plan(plan, out, reuse_dirs, *, runner=run_worker, plot=True, plan_only=False):
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "plan.json"
    if manifest.exists() and saved.read_json(manifest) != plan:
        raise ValueError("This output directory belongs to different inputs; choose a new directory.")
    write_json(manifest, plan)
    cache = load_cache([*reuse_dirs, out])
    rows, endpoints = [], []
    for task in plan["tasks"]:
        cached = cache.get(forward_key(task))
        print(f"[radial-resolution] {task['name']}: {'reuse saved forward endpoint' if cached else 'new forward solve required'}", flush=True)
        if plan_only:
            continue
        task_path, result_path = out / f"{task['name']}_task.json", out / f"{task['name']}.json"
        write_json(task_path, task)
        if cached:
            endpoint, reused_from = cached
            endpoint = copy.deepcopy(endpoint)
            endpoint.update(original_task_digest=endpoint["task_digest"],
                            task_digest=forward.task_digest(task), reused_from=reused_from)
        else:
            reused_from = None
            endpoint = runner(task, task_path, result_path, out / f"{task['name']}.log")
        if endpoint.get("task_digest") != forward.task_digest(task):
            raise ValueError(f"Worker returned a different task for {task['name']}.")
        validate_endpoint(task, endpoint)
        validate_baseline_reference(plan, task, endpoint)
        write_json(result_path, endpoint)
        row = endpoint_row(task, endpoint, plan["fixed_probe_radii"], result_path, reused_from)
        rows.append(row)
        endpoints.append((task, endpoint))
        save_tables(plan, rows, out)
        print("[radial-resolution-result] " + json.dumps(row, allow_nan=False), flush=True)
    if rows and plot:
        plot_results(plan, endpoints, out)
    if rows:
        print(f"[radial-resolution] saved {out / 'summary.json'}, profile_moves.csv and convergence.csv", flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ad-report", type=Path)
    parser.add_argument("--run-dir", type=Path, help="Saved profiles-only run with checkpoint.json and forward workers.")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--n-radial", default="51,101,201")
    parser.add_argument("--iteration", type=int, default=1, help="Fixed accepted profile point to compare to the baseline.")
    parser.add_argument("--reuse-dir", type=Path, action="append", default=[])
    parser.add_argument("--plan-only", action="store_true", help="Inspect cache matches and required solves; start no physics.")
    parser.add_argument("--worker-task", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker_task is not None:
            if args.worker_output is None:
                parser.error("--worker-output is required for a worker.")
            task = saved.read_json(args.worker_task)
            write_json(args.worker_output, forward.run_forward_task(task))
            return
        if any(value is None for value in (args.ad_report, args.run_dir, args.out_dir)):
            parser.error("--ad-report, --run-dir and --out-dir are required.")
        grids = [int(value) for value in args.n_radial.split(",")]
        plan = make_plan(saved.read_json(args.ad_report), saved.read_json(args.run_dir / "checkpoint.json"),
                         grids, args.iteration)
        run_plan(plan, args.out_dir.resolve(), [args.run_dir, *args.reuse_dir], plan_only=args.plan_only)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
