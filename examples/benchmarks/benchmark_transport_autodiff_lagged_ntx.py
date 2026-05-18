"""AD-vs-FD benchmark for the lagged exact-runtime NTX transport solve.

This harness keeps the magnetic geometry fixed and compares JAX automatic
differentiation against central finite differences for smooth final-state
transport objectives while varying one initial-profile parameter.

Default parameter choices are aimed at the standard analytical profile model:

- ``n0``
- ``T0``
- ``density_shape_power``
- ``temperature_shape_power``

Outputs:

- JSON summary
- CSV sweep
- PNG/PDF figure
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import NEOPAX
from NEOPAX._orchestrator import build_runtime_context, run_transport
from NEOPAX._profiles import AnalyticalProfileModel
from NEOPAX._transport_flux_models import PRESSURE_SOURCE_STATE_TO_MW_M3


DEFAULT_CONFIG = Path(
    "examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_benchmark.toml"
)
ALLOWED_PARAMETERS = {"n0", "T0", "density_shape_power", "temperature_shape_power"}
OBJECTIVE_LABELS = [
    "softmax_Er",
    "smooth_root_proxy",
    "Er2_volume_average",
    "Er_volume_average",
    "electron_temperature_volume_average_keV",
    "total_pressure_volume_average",
    "alpha_power_volume_average_mw_m3",
]


def _prepare_benchmark_config(config_path: Path, *, device: str | None) -> dict[str, Any]:
    config = NEOPAX.prepare_config(config_path, device=device)
    config = copy.deepcopy(config)
    config.setdefault("general", {})["mode"] = "transport"
    transport_output = config.setdefault("transport_output", {})
    transport_output["transport_plot"] = False
    transport_output["transport_write_hdf5"] = False
    transport_output["transport_compare_ambipolarity_residual"] = False
    transport_output["transport_scan_ambipolarity_residual"] = False
    solver_cfg = config.setdefault("transport_solver", {})
    solver_cfg["debug_stage_markers"] = False
    solver_cfg["debug_disable_jit"] = False
    return config


def _apply_one_step_diagnostic_config(config: dict[str, Any]) -> dict[str, Any]:
    tuned = copy.deepcopy(config)
    solver_cfg = tuned.setdefault("transport_solver", {})
    solver_cfg["stop_after_accepted_steps"] = 1
    current_max_steps = int(solver_cfg.get("max_steps", 20000))
    solver_cfg["max_steps"] = max(current_max_steps, 20)
    return tuned


def _baseline_profile_cfg(config: dict[str, Any]) -> dict[str, Any]:
    profiles = copy.deepcopy(config.get("profiles", {}))
    profiles.setdefault("model", "standard_analytical")
    return profiles


def _parameterized_profile_set(
    profile_cfg: dict[str, Any],
    geometry,
    n_species: int,
    *,
    parameter_name: str,
    parameter_value,
):
    cfg = dict(profile_cfg)
    cfg[parameter_name] = parameter_value

    model = AnalyticalProfileModel(
        n0=cfg.get("n0", 4.21),
        n_edge=cfg.get("n_edge", 0.6),
        T0=cfg.get("T0", 17.8),
        T_edge=cfg.get("T_edge", 0.7),
        c_density=None if cfg.get("c_density") is None else tuple(cfg.get("c_density")),
        c_temperature=None if cfg.get("c_temperature") is None else tuple(cfg.get("c_temperature")),
        density_shape_power=cfg.get("density_shape_power", 2.0),
        temperature_shape_power=cfg.get("temperature_shape_power", 2.0),
        n_scale=cfg.get("n_scale", 1.0),
        T_scale=cfg.get("T_scale", 1.0),
        er0_scale=cfg.get("er0_scale", 100.0),
        er0_peak_rho=cfg.get("er0_peak_rho", 0.8),
        charge_qp=None if cfg.get("charge_qp") is None else tuple(cfg.get("charge_qp")),
    )
    return model.build(geometry, n_species)


def _parameterized_initial_state(
    *,
    baseline_state,
    profile_cfg: dict[str, Any],
    geometry,
    n_species: int,
    parameter_name: str,
    parameter_value,
):
    profile_set = _parameterized_profile_set(
        profile_cfg,
        geometry,
        n_species,
        parameter_name=parameter_name,
        parameter_value=parameter_value,
    )
    density_state = jnp.asarray(profile_set.density, dtype=baseline_state.density.dtype) / 1.0e20
    temperature_state = jnp.asarray(profile_set.temperature, dtype=baseline_state.pressure.dtype) / 1.0e3
    pressure_state = density_state * temperature_state
    return dataclasses.replace(
        baseline_state,
        density=density_state,
        pressure=pressure_state,
    )


def _softmax_objective(er_profile: jax.Array, *, beta: float = 16.0) -> jax.Array:
    beta_arr = jnp.asarray(beta, dtype=er_profile.dtype)
    return jax.scipy.special.logsumexp(beta_arr * er_profile) / beta_arr


def _smooth_root_proxy(er_profile: jax.Array, rho_grid: jax.Array, *, beta: float = 24.0, eps: float = 1.0e-4):
    beta_arr = jnp.asarray(beta, dtype=er_profile.dtype)
    eps_arr = jnp.asarray(eps, dtype=er_profile.dtype)
    smooth_abs = jnp.sqrt(er_profile * er_profile + eps_arr * eps_arr)
    weights = jnp.exp(-beta_arr * smooth_abs)
    return jnp.sum(rho_grid * weights) / jnp.maximum(jnp.sum(weights), jnp.asarray(1.0e-30, dtype=er_profile.dtype))


def _volume_average(profile: jax.Array, geometry) -> jax.Array:
    volume = jnp.trapezoid(jnp.asarray(geometry.Vprime), x=jnp.asarray(geometry.r_grid))
    integral = jnp.trapezoid(profile * jnp.asarray(geometry.Vprime), x=jnp.asarray(geometry.r_grid))
    return integral / jnp.maximum(volume, jnp.asarray(1.0e-30, dtype=integral.dtype))


def _alpha_power_volume_average(final_state, runtime) -> jax.Array:
    source_models = runtime.models.source or {}
    pressure_source_model = source_models.get("temperature") if isinstance(source_models, dict) else None
    if pressure_source_model is None:
        return jnp.asarray(0.0, dtype=final_state.pressure.dtype)
    raw_sources = pressure_source_model(final_state)
    alpha_power = raw_sources.get("AlphaPower") if isinstance(raw_sources, dict) else None
    if alpha_power is None:
        return jnp.asarray(0.0, dtype=final_state.pressure.dtype)
    alpha_mw_m3 = PRESSURE_SOURCE_STATE_TO_MW_M3 * jnp.asarray(alpha_power, dtype=final_state.pressure.dtype)
    return _volume_average(alpha_mw_m3, runtime.geometry)


def _electron_temperature_volume_average(final_state, runtime) -> jax.Array:
    species_idx = getattr(runtime.species, "species_idx", {})
    electron_idx = species_idx.get("e", 0)
    temperature = jnp.asarray(final_state.temperature[electron_idx], dtype=final_state.pressure.dtype)
    return _volume_average(temperature, runtime.geometry)


def _total_pressure_volume_average(final_state, runtime) -> jax.Array:
    total_pressure = jnp.sum(jnp.asarray(final_state.pressure, dtype=final_state.pressure.dtype), axis=0)
    return _volume_average(total_pressure, runtime.geometry)


def _objective_vector(final_state, runtime) -> jax.Array:
    er = jnp.asarray(final_state.Er)
    rho = jnp.asarray(runtime.geometry.rho_grid, dtype=er.dtype)
    er2_vol = _volume_average(er * er, runtime.geometry)
    er_vol = _volume_average(er, runtime.geometry)
    te_vol = _electron_temperature_volume_average(final_state, runtime)
    p_tot_vol = _total_pressure_volume_average(final_state, runtime)
    alpha_vol = _alpha_power_volume_average(final_state, runtime)
    return jnp.stack(
        [
            _softmax_objective(er),
            _smooth_root_proxy(er, rho),
            er2_vol,
            er_vol,
            te_vol,
            p_tot_vol,
            alpha_vol,
        ]
    )


def _transport_objectives_for_parameter(
    parameter_value,
    *,
    config: dict[str, Any],
    runtime,
    baseline_state,
    profile_cfg: dict[str, Any],
    parameter_name: str,
):
    state0 = _parameterized_initial_state(
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        geometry=runtime.geometry,
        n_species=runtime.species.number_species,
        parameter_name=parameter_name,
        parameter_value=parameter_value,
    )
    result = run_transport(config, runtime, state0)
    final_state = result["final_state"]
    return _objective_vector(final_state, runtime)


def _fd_step(baseline_value: float, *, rel_step: float, abs_step: float) -> float:
    return max(abs_step, rel_step * max(abs(baseline_value), 1.0))


def _accepted_count(mask) -> int | None:
    if mask is None:
        return None
    arr = np.asarray(jax.device_get(mask))
    return int(np.sum(arr))


def _result_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    accepted_mask = result.get("accepted_mask")
    failed_mask = result.get("failed_mask")
    fail_codes = result.get("fail_codes")
    diag = {
        "n_steps": None if result.get("n_steps") is None else int(np.asarray(jax.device_get(result["n_steps"]))),
        "accepted_count": _accepted_count(accepted_mask),
        "accepted_mask": None if accepted_mask is None else np.asarray(jax.device_get(accepted_mask), dtype=bool).tolist(),
        "failed_any": False if failed_mask is None else bool(np.any(np.asarray(jax.device_get(failed_mask), dtype=bool))),
        "fail_codes": None if fail_codes is None else np.asarray(jax.device_get(fail_codes)).tolist(),
    }
    return diag


def _write_sweep_csv(
    path: Path,
    *,
    parameter_name: str,
    sweep_values: np.ndarray,
    objective_values: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([parameter_name] + OBJECTIVE_LABELS)
        for value, row in zip(sweep_values, objective_values):
            writer.writerow([float(value)] + [float(v) for v in row])


def _write_figure(report: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    sweep_values = np.asarray(report["sweep_values"], dtype=float)
    sweep_objectives = np.asarray(report["sweep_objectives"], dtype=float)
    baseline_objectives = np.asarray(report["baseline_objectives"], dtype=float)
    grad_ad = np.asarray(report["gradient_autodiff"], dtype=float)
    grad_fd = np.asarray(report["gradient_fd"], dtype=float)
    rel_err = np.asarray(report["gradient_relative_error"], dtype=float)
    rho = np.asarray(report["rho_grid"], dtype=float)
    er_baseline = np.asarray(report["baseline_final_Er"], dtype=float)
    er_minus = np.asarray(report["fd_minus_final_Er"], dtype=float)
    er_plus = np.asarray(report["fd_plus_final_Er"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), constrained_layout=True)

    ax0 = axes[0, 0]
    for idx, label in enumerate(OBJECTIVE_LABELS):
        ax0.plot(sweep_values, sweep_objectives[:, idx], marker="o", linewidth=1.8, label=label)
    ax0.axvline(float(report["baseline_value"]), color="0.35", linestyle="--", linewidth=1.0)
    ax0.set_xlabel(report["parameter_name"])
    ax0.set_ylabel("objective value")
    ax0.set_title("Objective sweep")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8)

    ax1 = axes[0, 1]
    ax1.plot(rho, er_baseline, linewidth=2.1, label="baseline")
    ax1.plot(rho, er_minus, linestyle="--", linewidth=1.6, label="- fd step")
    ax1.plot(rho, er_plus, linestyle=":", linewidth=1.8, label="+ fd step")
    ax1.set_xlabel(r"$\rho$")
    ax1.set_ylabel(r"$E_r$")
    ax1.set_title("Final $E_r$ profile comparison")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = axes[1, 0]
    ax2.scatter(grad_fd, grad_ad, color="#111827", s=42)
    lo = float(min(np.min(grad_fd), np.min(grad_ad)))
    hi = float(max(np.max(grad_fd), np.max(grad_ad)))
    pad = 0.05 * max(hi - lo, 1.0e-12)
    ax2.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#dc2626", linestyle="--", linewidth=1.2)
    for x, y, label in zip(grad_fd, grad_ad, OBJECTIVE_LABELS):
        ax2.annotate(label, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax2.set_xlabel("central finite difference")
    ax2.set_ylabel("JAX autodiff")
    ax2.set_title("Derivative parity")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 1]
    x = np.arange(len(OBJECTIVE_LABELS))
    ax3.bar(x, rel_err, color="#2563eb")
    ax3.set_xticks(x, OBJECTIVE_LABELS, rotation=20, ha="right")
    ax3.set_yscale("log")
    ax3.set_ylabel("relative derivative error")
    ax3.set_title("AD vs FD relative error")
    ax3.grid(True, alpha=0.3)
    ax3.text(
        0.03,
        0.95,
        f"passed = {report['passed']}\n"
        f"fd_step = {float(report['fd_step']):.3e}\n"
        f"max rel. err = {float(report['max_relative_error']):.2e}",
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.92},
    )

    fig.suptitle(f"Lagged exact-runtime NTX AD-vs-FD gate: {report['parameter_name']}")
    fig.savefig(out, dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def _print_terminal_summary(report: dict[str, Any]) -> None:
    print(
        f"[autodiff-gate] mode={'one_step' if report.get('one_step_diagnostic') else 'full_solve'} "
        f"parameter={report['parameter_name']} "
        f"baseline_value={report['baseline_value']:.6e} "
        f"fd_step={report['fd_step']:.6e}"
    )
    path = report.get("solver_path", {})
    for key in ("baseline", "fd_minus", "fd_plus"):
        diag = path.get(key, {})
        print(
            f"[autodiff-gate] path {key}: "
            f"n_steps={diag.get('n_steps')} "
            f"accepted_count={diag.get('accepted_count')} "
            f"failed_any={diag.get('failed_any')}"
        )
    print(
        "[autodiff-gate] fd path parity:",
        f"accepted_mask_equal_minus_plus={path.get('accepted_mask_equal_minus_plus')}",
    )
    print("[autodiff-gate] objective errors:")
    for label, j_ad, j_fd, abs_err, rel_err in zip(
        report["objective_labels"],
        report["gradient_autodiff"],
        report["gradient_fd"],
        report["gradient_absolute_error"],
        report["gradient_relative_error"],
    ):
        print(
            f"  - {label}: "
            f"ad={float(j_ad):.6e} "
            f"fd={float(j_fd):.6e} "
            f"abs_err={float(abs_err):.6e} "
            f"rel_err={float(rel_err):.6e}"
        )


def build_report(
    *,
    config_path: Path,
    parameter_name: str,
    rel_fd_step: float,
    abs_fd_step: float,
    sweep_half_width_rel: float,
    sweep_points: int,
    with_sweep: bool,
    one_step_diagnostic: bool,
    device: str | None,
) -> dict[str, Any]:
    if parameter_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"parameter_name must be one of {sorted(ALLOWED_PARAMETERS)}")

    config = _prepare_benchmark_config(config_path, device=device)
    if one_step_diagnostic:
        config = _apply_one_step_diagnostic_config(config)
    runtime, baseline_state = build_runtime_context(config)
    profile_cfg = _baseline_profile_cfg(config)
    baseline_value = float(profile_cfg[parameter_name])

    objective_fn = lambda p: _transport_objectives_for_parameter(  # noqa: E731
        p,
        config=config,
        runtime=runtime,
        baseline_state=baseline_state,
        profile_cfg=profile_cfg,
        parameter_name=parameter_name,
    )

    fd_step = _fd_step(baseline_value, rel_step=rel_fd_step, abs_step=abs_fd_step)
    minus_value = baseline_value - fd_step
    plus_value = baseline_value + fd_step
    baseline_result = run_transport(
        config,
        runtime,
        _parameterized_initial_state(
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            geometry=runtime.geometry,
            n_species=runtime.species.number_species,
            parameter_name=parameter_name,
            parameter_value=jnp.asarray(baseline_value),
        ),
    )
    minus_result = run_transport(
        config,
        runtime,
        _parameterized_initial_state(
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            geometry=runtime.geometry,
            n_species=runtime.species.number_species,
            parameter_name=parameter_name,
            parameter_value=jnp.asarray(minus_value),
        ),
    )
    plus_result = run_transport(
        config,
        runtime,
        _parameterized_initial_state(
            baseline_state=baseline_state,
            profile_cfg=profile_cfg,
            geometry=runtime.geometry,
            n_species=runtime.species.number_species,
            parameter_name=parameter_name,
            parameter_value=jnp.asarray(plus_value),
        ),
    )

    baseline_objectives = _objective_vector(baseline_result["final_state"], runtime)
    objectives_minus = _objective_vector(minus_result["final_state"], runtime)
    objectives_plus = _objective_vector(plus_result["final_state"], runtime)
    gradient_ad = jax.jacfwd(objective_fn)(jnp.asarray(baseline_value))
    gradient_fd = (objectives_plus - objectives_minus) / (2.0 * fd_step)

    grad_ad_np = np.asarray(jax.device_get(gradient_ad), dtype=float)
    grad_fd_np = np.asarray(jax.device_get(gradient_fd), dtype=float)
    abs_err = np.abs(grad_ad_np - grad_fd_np)
    rel_err = abs_err / np.maximum(np.abs(grad_fd_np), 1.0e-10)

    if with_sweep:
        sweep_half_width = sweep_half_width_rel * max(abs(baseline_value), 1.0)
        sweep_values = np.linspace(
            baseline_value - sweep_half_width,
            baseline_value + sweep_half_width,
            int(sweep_points),
            dtype=float,
        )
        sweep_objectives = np.stack(
            [
                np.asarray(jax.device_get(objective_fn(jnp.asarray(value))), dtype=float)
                for value in sweep_values
            ],
            axis=0,
        )
    else:
        sweep_values = np.asarray([minus_value, baseline_value, plus_value], dtype=float)
        sweep_objectives = np.stack(
            [
                np.asarray(jax.device_get(objectives_minus), dtype=float),
                np.asarray(jax.device_get(baseline_objectives), dtype=float),
                np.asarray(jax.device_get(objectives_plus), dtype=float),
            ],
            axis=0,
        )

    baseline_diag = _result_diagnostics(baseline_result)
    minus_diag = _result_diagnostics(minus_result)
    plus_diag = _result_diagnostics(plus_result)

    report = {
        "config_path": str(config_path),
        "one_step_diagnostic": bool(one_step_diagnostic),
        "parameter_name": parameter_name,
        "baseline_value": baseline_value,
        "fd_step": float(fd_step),
        "baseline_objectives": np.asarray(jax.device_get(baseline_objectives), dtype=float).tolist(),
        "gradient_autodiff": grad_ad_np.tolist(),
        "gradient_fd": grad_fd_np.tolist(),
        "gradient_absolute_error": abs_err.tolist(),
        "gradient_relative_error": rel_err.tolist(),
        "max_relative_error": float(np.max(rel_err)),
        "passed": bool(np.all(np.isfinite(rel_err)) and np.max(rel_err) <= 5.0e-2),
        "objective_labels": OBJECTIVE_LABELS,
        "sweep_values": sweep_values.tolist(),
        "sweep_objectives": sweep_objectives.tolist(),
        "solver_path": {
            "baseline": baseline_diag,
            "fd_minus": minus_diag,
            "fd_plus": plus_diag,
            "accepted_mask_equal_minus_plus": (
                baseline_diag["accepted_mask"] is not None
                and minus_diag["accepted_mask"] is not None
                and plus_diag["accepted_mask"] is not None
                and minus_diag["accepted_mask"] == plus_diag["accepted_mask"]
            ),
        },
        "rho_grid": np.asarray(jax.device_get(runtime.geometry.rho_grid), dtype=float).tolist(),
        "baseline_final_Er": np.asarray(jax.device_get(baseline_result["final_state"].Er), dtype=float).tolist(),
        "fd_minus_final_Er": np.asarray(jax.device_get(minus_result["final_state"].Er), dtype=float).tolist(),
        "fd_plus_final_Er": np.asarray(jax.device_get(plus_result["final_state"].Er), dtype=float).tolist(),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--parameter",
        default="n0",
        choices=sorted(ALLOWED_PARAMETERS),
        help="Initial-profile parameter to differentiate against.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--fd-rel-step", type=float, default=1.0e-3)
    parser.add_argument("--fd-abs-step", type=float, default=1.0e-4)
    parser.add_argument("--sweep-half-width-rel", type=float, default=5.0e-2)
    parser.add_argument("--sweep-points", type=int, default=7)
    parser.add_argument("--with-sweep", action="store_true", help="Run extra sweep solves for objective curves.")
    parser.add_argument(
        "--one-step-diagnostic",
        action="store_true",
        help="Stop after one accepted transport step to isolate local AD-vs-FD behavior.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("outputs/autodiff_transport_lagged_ntx"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    report = build_report(
        config_path=args.config,
        parameter_name=args.parameter,
        rel_fd_step=args.fd_rel_step,
        abs_fd_step=args.fd_abs_step,
        sweep_half_width_rel=args.sweep_half_width_rel,
        sweep_points=args.sweep_points,
        with_sweep=args.with_sweep,
        one_step_diagnostic=args.one_step_diagnostic,
        device=args.device,
    )

    outdir = args.outdir / args.parameter
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"transport_autodiff_{args.parameter}_summary.json"
    csv_path = outdir / f"transport_autodiff_{args.parameter}_sweep.csv"
    fig_path = outdir / f"transport_autodiff_{args.parameter}.png"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_sweep_csv(
        csv_path,
        parameter_name=report["parameter_name"],
        sweep_values=np.asarray(report["sweep_values"], dtype=float),
        objective_values=np.asarray(report["sweep_objectives"], dtype=float),
    )
    _print_terminal_summary(report)
    if not args.no_plot:
        _write_figure(report, fig_path)
        print(f"Wrote {fig_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(
        f"parameter={report['parameter_name']} "
        f"passed={report['passed']} "
        f"max_rel_error={report['max_relative_error']:.3e}"
    )


if __name__ == "__main__":
    main()
