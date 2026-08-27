"""Optimization-only VMEX-like stage for geometry + initial-Er objectives.

This module intentionally does not alter the benchmark reverse path.  It owns
only persistent optimization callables; the callables themselves must invoke
the established root, payload, and VMEC reverse rules.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

from ._geometry_autodiff import GeometryRawBlockSolve, GeometryRawBlockStage


@dataclasses.dataclass(frozen=True, slots=True)
class InitialRootStageLayout:
    """Static structural key for one geometry + initial-root stage."""

    objective_names: tuple[str, ...]
    geometry_param_specs: tuple[tuple[str, int, int], ...]
    n_r: int
    n_theta: int
    n_zeta: int
    n_xi: int
    surface_backend: str
    flux_model: str


@dataclasses.dataclass(frozen=True, slots=True)
class GeometryInitialRootOptimizationStage:
    """Two bounded optimizer-level operators, analogous to VMEX rows/jac.

    `root_to_payload` and `payload_to_vmec` are intentionally separate.  They
    receive all trial-dependent values as arguments and must not retain an
    evaluation's state, payload cotangents, or DoF vector.
    """

    layout: InitialRootStageLayout
    root_to_payload: Callable[..., Any]
    payload_to_vmec: Callable[..., Any]


def build_geometry_initial_root_optimization_stage(
    *,
    layout: InitialRootStageLayout,
    raw_block_stage: GeometryRawBlockStage,
    root_to_payload_impl: Callable[..., Any],
    payload_to_vmec_impl: Callable[..., Any],
) -> GeometryInitialRootOptimizationStage:
    """Build persistent optimization callables without retaining trial data."""

    def root_to_payload(dynamic_raw_payload, /, **kwargs):
        return root_to_payload_impl(
            raw_block_solve=raw_block_solve_from_dynamic_payload(raw_block_stage, dynamic_raw_payload),
            **kwargs,
        )

    def payload_to_vmec(dynamic_raw_payload, /, **kwargs):
        return payload_to_vmec_impl(
            raw_block_solve=raw_block_solve_from_dynamic_payload(raw_block_stage, dynamic_raw_payload),
            **kwargs,
        )

    return GeometryInitialRootOptimizationStage(
        layout=layout,
        root_to_payload=root_to_payload,
        payload_to_vmec=payload_to_vmec,
    )


def raw_block_dynamic_payload(raw_block_solve: GeometryRawBlockSolve) -> tuple[Any, Any, Any]:
    """Return only the trial-dependent leaves of one shared raw-block solve."""

    return (
        raw_block_solve.implicit_params,
        raw_block_solve.state,
        raw_block_solve.dof_mask,
    )


def raw_block_solve_from_dynamic_payload(
    raw_block_stage: GeometryRawBlockStage,
    payload: tuple[Any, Any, Any],
) -> GeometryRawBlockSolve:
    """Rebuild the established raw-block container at an AD boundary.

    The stage owns only immutable VMEX setup; every payload element comes from
    the current trial and is therefore never retained across evaluations.
    """

    implicit_params, state, dof_mask = payload
    return GeometryRawBlockSolve(
        implicit=raw_block_stage.implicit,
        implicit_params=implicit_params,
        implicit_cfg=raw_block_stage.implicit_cfg,
        state=state,
        dof_mask=dof_mask,
        param_entries=raw_block_stage.param_entries,
    )


def initial_root_stage_layout(
    *,
    config: dict,
    objective_names: Sequence[str],
    geometry_param_specs: Sequence[tuple[str, int, int]],
    n_r: int,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    surface_backend: str,
) -> InitialRootStageLayout:
    """Validate the fixed exact-Lij configuration for a staged run."""

    flux_model = str(config.get("neoclassical", {}).get("flux_model", "ntx_database")).strip().lower()
    if flux_model != "ntx_exact_lij_runtime":
        raise NotImplementedError(
            "The geometry + initial-root staged optimization currently requires "
            "neoclassical.flux_model='ntx_exact_lij_runtime'; "
            f"got {flux_model!r}."
        )
    return InitialRootStageLayout(
        objective_names=tuple(str(name) for name in objective_names),
        geometry_param_specs=tuple(tuple(spec) for spec in geometry_param_specs),
        n_r=-1 if n_r is None else int(n_r),
        n_theta=-1 if n_theta is None else int(n_theta),
        n_zeta=-1 if n_zeta is None else int(n_zeta),
        n_xi=-1 if n_xi is None else int(n_xi),
        surface_backend=str(surface_backend),
        flux_model=flux_model,
    )
