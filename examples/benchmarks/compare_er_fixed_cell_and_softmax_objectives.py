"""Compare fixed-cell Er and peak-Er targets with the candidate root losses.

Prescribed 51-cell arrays and already archived derivatives only. This script
does not construct a plasma runtime or run geometry, roots, or transport.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX import _reverse_ad_transport as reverse
from examples.benchmarks import compare_er_transition_loss_candidates as candidates

EXAMPLE = candidates.audit.example
REFERENCE_NAMES = ("Er_left_target", "Er_right_target", "Er_left_right_targets", "softmax_Er_target")
NAMES = (*candidates.NAMES, *REFERENCE_NAMES)
RAW_LABELS = ("Er_transition_left", "Er_transition_right", "softmax_Er")
BENCHMARK_CSV = ROOT / "docs/benchmarks/database_transition_objectives_ad_recorded_reuse_full_t20_left_right_2026-10-22.csv"
BENCHMARK_CONFIG = ROOT / "examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml"


def reference_observables(er):
    """Use the production dispatcher with the optimization cell indices."""
    state = SimpleNamespace(Er=er)
    return jnp.stack(tuple(reverse.objective_scalar_by_index(
        state, None, index, objective_labels=RAW_LABELS,
        er_transition_left_index=EXAMPLE.ER_TRANSITION_LEFT_INDEX,
        er_transition_right_index=EXAMPLE.ER_TRANSITION_RIGHT_INDEX,
    ) for index in range(len(RAW_LABELS))))


def reference_costs(er):
    left, right, peak = reference_observables(er)
    left_cost = 0.5 * EXAMPLE.ER_TRANSITION_LEFT_WEIGHT * (left - EXAMPLE.ER_TRANSITION_LEFT_TARGET)**2
    right_cost = 0.5 * EXAMPLE.ER_TRANSITION_RIGHT_WEIGHT * (right - EXAMPLE.ER_TRANSITION_RIGHT_TARGET)**2
    peak_cost = 0.5 * EXAMPLE.MAX_ER_WEIGHT * (peak - EXAMPLE.MAX_ER_TARGET)**2
    return jnp.stack((left_cost, right_cost, left_cost + right_cost, peak_cost))


def all_losses(er, rho):
    return jnp.concatenate((candidates.losses(er, rho), reference_costs(er)))


def reference_profile():
    rho = candidates.grid()
    return 10.0 * jnp.tanh((0.65 - rho) / 0.01)


def calibration():
    raw = np.asarray(all_losses(reference_profile(), candidates.grid()))
    return raw[0] / raw


def scenarios():
    # Keep fixed indices meaningful: the previous four-node and shifted-node
    # cases are outside this reference comparison's common 51-cell mesh.
    mesh = candidates.grid()
    cases = [case for case in candidates.scenarios()
             if case[3].shape == mesh.shape and np.array_equal(np.asarray(case[3]), np.asarray(mesh))]
    for q in (0.35, 0.45, candidates.TARGET, 0.58, 0.65):
        er = 40 * jnp.tanh((q - mesh) / 0.02)
        direction = jax.jacfwd(lambda value: 40 * jnp.tanh((value - mesh) / 0.02))(q)
        cases.append(("translate_amplitude_40", "q", q, mesh, er, direction))
    base = reference_profile()
    cases.append(("common_offset", "offset", 0.0, mesh, base, jnp.ones_like(base)))
    matched = base.at[25].set(EXAMPLE.ER_TRANSITION_LEFT_TARGET).at[26].set(EXAMPLE.ER_TRANSITION_RIGHT_TARGET)
    cases.append(("fixed_cell_targets_met", "amplitude", 1.0, mesh, matched, matched))
    for index, label in ((10, "inner_peak"), (36, "outer_peak")):
        er = base.at[index].set(40.0)
        cases.append((label, "peak_increment", 0.0, mesh, er, jnp.zeros_like(base).at[index].set(1.0)))
    return cases


def compare_rows(factors):
    evaluate = jax.jit(lambda er, rho: (all_losses(er, rho), jax.jacrev(all_losses)(er, rho)))
    rows, cotangents = [], []
    for sweep, coordinate, parameter, rho, er, direction in scenarios():
        raw, bars = evaluate(er, rho)
        raw, bars = np.asarray(raw), np.asarray(bars)
        gradients = bars @ np.asarray(direction)
        step = 1e-6 if np.any(np.asarray(er)) else 1e-8 / max(1.0, float(jnp.max(jnp.abs(direction))))
        fd = np.asarray((all_losses(er + step * direction, rho) - all_losses(er - step * direction, rho)) / (2 * step))
        left, right, peak = map(float, reference_observables(er))
        physical = candidates.profile_metrics(er, rho)
        for i, name in enumerate(NAMES):
            ad, numerical = float(factors[i] * gradients[i]), float(factors[i] * fd[i])
            rows.append(dict(
                sweep=sweep, coordinate=coordinate, parameter=parameter, candidate=name,
                raw_cost=float(raw[i]), raw_gradient=float(gradients[i]), calibration=float(factors[i]),
                cost=float(factors[i] * raw[i]), gradient=ad, directional_fd=numerical,
                fd_scaled_error=abs(ad - numerical) / max(1.0, abs(ad), abs(numerical)),
                Er_left=left, Er_right=right, softmax_Er=peak,
                left_index=EXAMPLE.ER_TRANSITION_LEFT_INDEX, right_index=EXAMPLE.ER_TRANSITION_RIGHT_INDEX,
                **physical,
            ))
        if sweep in ("inner_peak", "outer_peak") or (sweep == "build_target_bracket" and parameter in (0.0, 14.0)):
            for i, name in enumerate(NAMES):
                normalizer = max(float(np.linalg.norm(bars[i])), 1e-30)
                for index, (radius, value, bar) in enumerate(zip(rho, er, bars[i], strict=True)):
                    cotangents.append(dict(sweep=sweep, parameter=parameter, candidate=name,
                                           index=index, rho=float(radius), Er=float(value),
                                           raw_cost_Er_bar=float(bar), normalized_Er_bar=float(bar / normalizer)))
    return rows, cotangents


def landscape_rows(factors):
    rho, base = candidates.grid(), reference_profile()
    scans = (
        ("translate_amplitude_40", np.linspace(0.35, 0.7, 351), lambda q: 40 * jnp.tanh((q - rho) / 0.02)),
        ("common_offset", np.linspace(-30.0, 30.0, 301), lambda offset: base + offset),
        ("build_target_bracket", np.linspace(0.0, 24.0, 241), lambda p: base.at[26].add(-p)),
    )
    rows = []
    for sweep, parameters, profile in scans:
        matrix = np.asarray(jax.jit(jax.vmap(lambda p: all_losses(profile(p), rho)))(jnp.asarray(parameters)))
        for parameter, raw in zip(parameters, matrix, strict=True):
            for name, value, factor in zip(NAMES, raw, factors, strict=True):
                rows.append(dict(sweep=sweep, parameter=float(parameter), candidate=name,
                                 raw_cost=float(value), cost=float(factor * value)))
    return rows


def offset_only_fits():
    """Exact one-coordinate fits; no numerical optimization is performed."""
    rho, base = candidates.grid(), reference_profile()
    left, right, peak = map(float, reference_observables(base))
    weights = (EXAMPLE.ER_TRANSITION_LEFT_WEIGHT, EXAMPLE.ER_TRANSITION_RIGHT_WEIGHT)
    pair_offset = -(weights[0] * (left - EXAMPLE.ER_TRANSITION_LEFT_TARGET)
                    + weights[1] * (right - EXAMPLE.ER_TRANSITION_RIGHT_TARGET)) / sum(weights)
    fits = (("softmax_Er_target", EXAMPLE.MAX_ER_TARGET - peak),
            ("Er_left_target", EXAMPLE.ER_TRANSITION_LEFT_TARGET - left),
            ("Er_right_target", EXAMPLE.ER_TRANSITION_RIGHT_TARGET - right),
            ("Er_left_right_targets", pair_offset))
    rows = []
    for name, offset in fits:
        er = base + offset
        values = dict(zip(REFERENCE_NAMES, map(float, reference_costs(er)), strict=True))
        left, right, peak = map(float, reference_observables(er))
        rows.append(dict(objective=name, best_common_offset=offset, raw_objective_cost=values[name],
                         Er_left=left, Er_right=right, softmax_Er=peak, **candidates.profile_metrics(er, rho)))
    return rows


def archived_benchmark_analysis(path=BENCHMARK_CSV, config_path=BENCHMARK_CONFIG):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        data = {row["residual_label"].split(":", 1)[1]: row for row in csv.DictReader(stream)
                if row["residual_label"].startswith("transport:")}
    columns = [name for name in next(iter(data.values())) if name not in ("residual_label", "residual")]
    gradients = []
    left, right, peak = (float(data[name]["residual"]) for name in RAW_LABELS)
    for parameter in columns:
        jl, jr, jp = (float(data[name][parameter]) for name in RAW_LABELS)
        gl = EXAMPLE.ER_TRANSITION_LEFT_WEIGHT * (left - EXAMPLE.ER_TRANSITION_LEFT_TARGET) * jl
        gr = EXAMPLE.ER_TRANSITION_RIGHT_WEIGHT * (right - EXAMPLE.ER_TRANSITION_RIGHT_TARGET) * jr
        gp = EXAMPLE.MAX_ER_WEIGHT * (peak - EXAMPLE.MAX_ER_TARGET) * jp
        gradients.append(dict(parameter=parameter, dEr_left=jl, dEr_right=jr, dsoftmax_Er=jp,
                              dEr_left_minus_right=jl - jr, dEr_common_mode=0.5 * (jl + jr),
                              dleft_cost=gl, dright_cost=gr, dpair_cost=gl + gr, dsoftmax_cost=gp))
    with Path(config_path).open("rb") as stream:
        config = tomllib.load(stream)
    profile_names = tuple(EXAMPLE.opt.PROFILE_PARAMETER_ORDER)
    baseline = EXAMPLE.opt._profile_values_from_config(config, jnp.float64)
    nominal = np.asarray(EXAMPLE.opt._profile_scales_from_values(baseline, "nominal"))
    conditioning = []
    geometry_names = tuple(name for name in columns if name.startswith("vmec:"))
    for group, mode, names, scales in (
        ("profiles", "identity", profile_names, np.ones_like(nominal)),
        ("profiles", "nominal", profile_names, nominal),
        ("benchmark_geometry", "identity", geometry_names, np.ones(len(geometry_names))),
    ):
        matrix = np.array([[float(data[label][name]) for name in names] for label in RAW_LABELS[:2]]) * scales
        singular = np.linalg.svd(matrix, compute_uv=False)
        cosine = np.dot(matrix[0], matrix[1]) / (np.linalg.norm(matrix[0]) * np.linalg.norm(matrix[1]))
        conditioning.append(dict(parameter_group=group, mode=mode, parameters="|".join(names),
                                 column_scales="|".join(map(str, scales)), row_cosine=float(cosine),
                                 largest_singular_value=float(singular[0]), smallest_singular_value=float(singular[1]),
                                 condition_number=float(singular[0] / singular[1]),
                                 benchmark_left_index=20, benchmark_right_index=21))
    return gradients, conditioning


def joint_target_compatibility():
    """A lower bound when peak=25 and a particular cell=26 are both asked for."""
    left, peak = EXAMPLE.ER_TRANSITION_LEFT_TARGET, EXAMPLE.MAX_ER_TARGET
    wl, wp = EXAMPLE.ER_TRANSITION_LEFT_WEIGHT, EXAMPLE.MAX_ER_WEIGHT
    gap = max(left - peak, 0.0)
    return dict(left_target=left, peak_target=peak, left_weight=wl, peak_weight=wp,
                simultaneous_zero_possible=left <= peak,
                joint_cost_lower_bound=0.5 * wl * wp / (wl + wp) * gap**2,
                shared_peak_at_bound=(wl * left + wp * peak) / (wl + wp))


def plot_comparison(out_dir, rows, landscapes, cotangents):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = (
        ("current", "Current location + strength", "C0"),
        ("Er_left_right_targets", "Fixed Er-left/right targets", "C1"),
        ("softmax_Er_target", "Softmax Er target", "C2"),
        ("target_zero_local", "Target zero + local margins", "C3"),
        ("target_zero_regional", "Target zero + regional margins", "C4"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name, label, color in styles:
        for axis, sweep in ((axes[0, 0], "translate_amplitude_40"), (axes[0, 1], "common_offset"),
                            (axes[1, 0], "build_target_bracket")):
            selected = [r for r in landscapes if r["candidate"] == name and r["sweep"] == sweep]
            axis.plot([r["parameter"] for r in selected], [max(r["cost"], 1e-8) for r in selected], label=label, color=color)
        selected = [r for r in cotangents if r["candidate"] == name and r["sweep"] == "outer_peak"]
        axes[1, 1].plot([r["rho"] for r in selected], [r["normalized_Er_bar"] for r in selected],
                        label=label, color=color, marker=".", markersize=3)
    axes[0, 0].axvline(candidates.TARGET, color="grey", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Translate root, amplitude 40 kV/m", xlabel="True crossing q", ylabel="Calibrated cost", yscale="log")
    axes[0, 1].set(title="Change every Er cell by the same offset", xlabel="Common offset (kV/m)", ylabel="Calibrated cost", yscale="log")
    axes[1, 0].set(title="Create target crossing beside a stronger remote root", xlabel="Lower outer target cell by p (kV/m)", ylabel="Calibrated cost", yscale="log")
    axes[1, 1].axvline(candidates.TARGET, color="grey", linestyle="--", linewidth=1)
    axes[1, 1].set(title="Where gradients act when peak is at rho=0.716", xlabel="rho", ylabel="Er cost gradient / its L2 norm")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fixed-cell and peak-Er references; prescribed arrays only", fontsize=12)
    fig.savefig(out_dir / "reference_comparison.svg")
    fig.savefig(out_dir / "reference_comparison.png", dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/er_reference_loss_comparison")
    args = parser.parse_args(argv)
    jax.config.update("jax_enable_compilation_cache", False)
    factors = calibration()
    rows, cotangents = compare_rows(factors)
    landscapes = landscape_rows(factors)
    benchmark, conditioning = archived_benchmark_analysis()
    fits = offset_only_fits()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, output in (
        ("comparison.csv", rows), ("landscapes.csv", landscapes), ("Er_cotangents.csv", cotangents),
        ("offset_only_fits.csv", fits), ("archived_benchmark_gradients.csv", benchmark),
        ("archived_benchmark_conditioning.csv", conditioning),
        ("joint_target_compatibility.csv", [joint_target_compatibility()]),
        ("calibration.csv", [dict(candidate=name, factor=float(factor)) for name, factor in zip(NAMES, factors, strict=True)]),
    ):
        candidates.audit.write_csv(args.out_dir / filename, output)
    plot_comparison(args.out_dir, rows, landscapes, cotangents)
    print(json.dumps(dict(out_dir=str(args.out_dir), prescribed_cases=len(rows) // len(NAMES),
                         max_scaled_fd_error=max(r["fd_scaled_error"] for r in rows),
                         offset_only_fits=fits, archived_benchmark_conditioning=conditioning,
                         joint_target_compatibility=joint_target_compatibility()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
