"""Direct Python API helpers for NEOPAX."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

from .cli import apply_cli_overrides
from .main import load_config, run_config


@dataclasses.dataclass(frozen=True)
class RunResult:
    """Structured return object for direct API runs."""

    mode: str
    config: dict[str, Any]
    raw_result: Any
    final_state: Any = None
    saved_states: Any = None
    time_grid: Any = None
    saved_step_sizes: Any = None
    accepted_mask: Any = None
    failed_mask: Any = None
    fail_codes: Any = None
    n_steps: Any = None
    done: Any = None
    failed: Any = None
    fail_code: Any = None
    final_time: Any = None
    rho: Any = None
    output_dir: Path | None = None


def prepare_config(
    config_or_path: dict[str, Any] | str | Path,
    *,
    mode: str | None = None,
    vmec_file: str | None = None,
    boozer_file: str | None = None,
    n_radial: int | None = None,
    n_x: int | None = None,
    backend: str | None = None,
    dt: float | None = None,
    t_final: float | None = None,
    output_dir: str | None = None,
    set_values: list[str] | None = None,
) -> dict[str, Any]:
    """Load and override a NEOPAX config without executing it."""

    if isinstance(config_or_path, (str, Path)):
        config = load_config(config_or_path)
    else:
        config = dict(config_or_path)

    args = argparse.Namespace(
        config=str(config_or_path) if isinstance(config_or_path, (str, Path)) else "<in-memory>",
        mode=mode,
        vmec_file=vmec_file,
        boozer_file=boozer_file,
        n_radial=n_radial,
        n_x=n_x,
        backend=backend,
        dt=dt,
        t_final=t_final,
        output_dir=output_dir,
        set_values=list(set_values or []),
    )
    return apply_cli_overrides(config, args)


def _extract_final_state(raw_result: Any):
    if isinstance(raw_result, dict):
        return raw_result.get("final_state")
    return getattr(raw_result, "final_state", None)


def _extract_result_field(raw_result: Any, key: str, default=None):
    if isinstance(raw_result, dict):
        return raw_result.get(key, default)
    return getattr(raw_result, key, default)


def run(
    config_or_path: dict[str, Any] | str | Path,
    *,
    mode: str | None = None,
    vmec_file: str | None = None,
    boozer_file: str | None = None,
    n_radial: int | None = None,
    n_x: int | None = None,
    backend: str | None = None,
    dt: float | None = None,
    t_final: float | None = None,
    output_dir: str | None = None,
    set_values: list[str] | None = None,
) -> RunResult:
    """Convenience entry point for direct NEOPAX execution.

    This keeps the Python API explicit and usable from scripts or larger JAX
    workflows, while sharing the same common override mapping as the CLI.
    """

    config = prepare_config(
        config_or_path,
        mode=mode,
        vmec_file=vmec_file,
        boozer_file=boozer_file,
        n_radial=n_radial,
        n_x=n_x,
        backend=backend,
        dt=dt,
        t_final=t_final,
        output_dir=output_dir,
        set_values=set_values,
    )
    raw_result = run_config(config)
    resolved_mode = str(config.get("general", {}).get("mode", config.get("mode", "transport"))).strip().lower()
    rho = raw_result.get("rho") if isinstance(raw_result, dict) else None
    result_output_dir = raw_result.get("output_dir") if isinstance(raw_result, dict) else None
    if result_output_dir is not None and not isinstance(result_output_dir, Path):
        result_output_dir = Path(str(result_output_dir))
    return RunResult(
        mode=resolved_mode,
        config=config,
        raw_result=raw_result,
        final_state=_extract_final_state(raw_result),
        saved_states=_extract_result_field(raw_result, "ys"),
        time_grid=_extract_result_field(raw_result, "ts"),
        saved_step_sizes=_extract_result_field(raw_result, "dts"),
        accepted_mask=_extract_result_field(raw_result, "accepted_mask"),
        failed_mask=_extract_result_field(raw_result, "failed_mask"),
        fail_codes=_extract_result_field(raw_result, "fail_codes"),
        n_steps=_extract_result_field(raw_result, "n_steps"),
        done=_extract_result_field(raw_result, "done"),
        failed=_extract_result_field(raw_result, "failed"),
        fail_code=_extract_result_field(raw_result, "fail_code"),
        final_time=_extract_result_field(raw_result, "final_time"),
        rho=rho,
        output_dir=result_output_dir,
    )
