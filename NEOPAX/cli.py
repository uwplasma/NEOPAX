"""Command-line interface for NEOPAX."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from .main import load_config, run_config


def _coerce_cli_value(raw: str):
    value = str(raw).strip()
    lower = value.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"none", "null"}:
        return None
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _set_nested(config: dict, dotted_key: str, value) -> None:
    parts = [part for part in str(dotted_key).split(".") if part]
    if not parts:
        raise ValueError("Override key cannot be empty.")
    current = config
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    out = deepcopy(config)

    if getattr(args, "mode", None) is not None:
        out.setdefault("general", {})["mode"] = str(args.mode)

    geometry = out.setdefault("geometry", {})
    if getattr(args, "vmec_file", None) is not None:
        geometry["vmec_file"] = str(args.vmec_file)
    if getattr(args, "boozer_file", None) is not None:
        geometry["boozer_file"] = str(args.boozer_file)
    if getattr(args, "n_radial", None) is not None:
        geometry["n_radial"] = int(args.n_radial)

    energy_grid = out.setdefault("energy_grid", {})
    if getattr(args, "n_x", None) is not None:
        energy_grid["n_x"] = int(args.n_x)

    solver = out.setdefault("transport_solver", {})
    if getattr(args, "backend", None) is not None:
        solver["transport_solver_backend"] = str(args.backend)
        solver["integrator"] = str(args.backend)
    if getattr(args, "dt", None) is not None:
        solver["dt"] = float(args.dt)
    if getattr(args, "t_final", None) is not None:
        solver["t_final"] = float(args.t_final)
    if getattr(args, "output_dir", None) is not None:
        out.setdefault("transport_output", {})["transport_output_dir"] = str(args.output_dir)
        out.setdefault("fluxes", {})["fluxes_output_dir"] = str(args.output_dir)
        out.setdefault("sources", {})["sources_output_dir"] = str(args.output_dir)
        out.setdefault("ambipolarity", {})["er_ambipolar_output_dir"] = str(args.output_dir)

    for item in getattr(args, "set_values", []) or []:
        if "=" not in item:
            raise ValueError(f"Invalid --set override '{item}'. Expected section.key=value.")
        dotted_key, raw_value = item.split("=", 1)
        _set_nested(out, dotted_key.strip(), _coerce_cli_value(raw_value))

    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="NEOPAX",
        description=(
            "Run NEOPAX from a TOML configuration file with optional CLI overrides.\n\n"
            "The Python API remains the preferred direct path for JAX/autodiff workflows,\n"
            "while this CLI provides a lightweight user-facing override layer."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=str, help="Path to the NEOPAX TOML configuration file.")
    parser.add_argument("--mode", choices=["transport", "ambipolarity", "fluxes", "sources"], help="Override general.mode.")
    parser.add_argument("--vmec-file", type=str, help="Override geometry.vmec_file.")
    parser.add_argument("--boozer-file", type=str, help="Override geometry.boozer_file.")
    parser.add_argument("--n-radial", type=int, help="Override geometry.n_radial.")
    parser.add_argument("--n-x", type=int, help="Override energy_grid.n_x.")
    parser.add_argument(
        "--backend",
        type=str,
        help="Override transport_solver.transport_solver_backend (and integrator).",
    )
    parser.add_argument("--dt", type=float, help="Override transport_solver.dt.")
    parser.add_argument("--t-final", type=float, help="Override transport_solver.t_final.")
    parser.add_argument("--output-dir", type=str, help="Override the main output directory keys.")
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="section.key=value",
        help="Generic dotted config override. May be passed multiple times.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        parser.error(f"config file not found: {config_path}")

    config = load_config(config_path)
    try:
        config = apply_cli_overrides(config, args)
    except ValueError as exc:
        parser.error(str(exc))
    run_config(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
