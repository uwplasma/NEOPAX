"""Optimization-only VMEX-like stage for geometry + initial-Er objectives.

This module intentionally does not alter the benchmark reverse path.  It owns
only persistent optimization callables; the callables themselves must invoke
the established root, payload, and VMEC reverse rules.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from ._geometry_autodiff import (
    GeometryRawBlockSolve,
    GeometryRawBlockStage,
)


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
class FloatingPayloadLeafLayout:
    """Fixed payload structure with only inexact leaves supplied per trial.

    Geometry and NTX support dataclasses contain both numerical arrays and
    static metadata (mode indices, dimensions, and Python configuration).  A
    full-pytree JAX argument is therefore not a valid optimization boundary:
    JAX replaces static leaves while tracing reverse transforms.  This layout
    keeps that metadata stage-static and exposes only floating-point leaves as
    trial inputs.

    It is deliberately an optimization-only utility.  The benchmark reverse
    path continues to receive its existing geometry/support payloads.
    """

    treedef: Any
    floating_mask: tuple[bool, ...]
    static_leaves: tuple[Any, ...]

    @staticmethod
    def _is_floating_leaf(leaf: Any) -> bool:
        try:
            dtype = jnp.asarray(leaf).dtype
        except (TypeError, ValueError):
            return False
        return bool(jnp.issubdtype(dtype, jnp.inexact))

    @classmethod
    def from_template(cls, payload: Any) -> "FloatingPayloadLeafLayout":
        leaves, treedef = jax.tree_util.tree_flatten(payload)
        floating_mask = tuple(cls._is_floating_leaf(leaf) for leaf in leaves)
        static_leaves = tuple(
            None if is_floating else leaf
            for leaf, is_floating in zip(leaves, floating_mask, strict=True)
        )
        return cls(
            treedef=treedef,
            floating_mask=floating_mask,
            static_leaves=static_leaves,
        )

    def floating_leaves(self, payload: Any) -> tuple[Any, ...]:
        leaves, treedef = jax.tree_util.tree_flatten(payload)
        if treedef != self.treedef or len(leaves) != len(self.floating_mask):
            raise ValueError("Initial-Er optimization payload structure changed within a stage.")
        return tuple(
            jnp.asarray(leaf)
            for leaf, is_floating in zip(leaves, self.floating_mask, strict=True)
            if is_floating
        )

    def rebuild(self, floating_leaves: tuple[Any, ...]) -> Any:
        expected_count = sum(self.floating_mask)
        if len(floating_leaves) != expected_count:
            raise ValueError(
                "Initial-Er optimization floating payload leaf count changed: "
                f"got {len(floating_leaves)}, expected {expected_count}."
            )
        dynamic_leaves = iter(floating_leaves)
        leaves = tuple(
            next(dynamic_leaves) if is_floating else static_leaf
            for is_floating, static_leaf in zip(
                self.floating_mask, self.static_leaves, strict=True
            )
        )
        return self.treedef.unflatten(leaves)


@dataclasses.dataclass(frozen=True, slots=True)
class InitialErTransportPayloadAdapter:
    """Stage-owned, trial-data-free adapter for geometry and NTX support."""

    geometry_layout: FloatingPayloadLeafLayout
    support_layout: FloatingPayloadLeafLayout

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "InitialErTransportPayloadAdapter":
        try:
            geometry = payload["geometry"]
            support = payload["ntx_support"]
        except KeyError as exc:
            raise ValueError("Initial-Er transport payload requires geometry and ntx_support.") from exc
        return cls(
            geometry_layout=FloatingPayloadLeafLayout.from_template(geometry),
            support_layout=FloatingPayloadLeafLayout.from_template(support),
        )

    def dynamic_leaves(self, payload: dict[str, Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        return (
            self.geometry_layout.floating_leaves(payload["geometry"]),
            self.support_layout.floating_leaves(payload["ntx_support"]),
        )

    def rebuild(
        self,
        geometry_leaves: tuple[Any, ...],
        support_leaves: tuple[Any, ...],
    ) -> dict[str, Any]:
        return {
            "geometry": self.geometry_layout.rebuild(geometry_leaves),
            "ntx_support": self.support_layout.rebuild(support_leaves),
        }


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


@dataclasses.dataclass(frozen=True, slots=True)
class InitialRootReverseKernelSet:
    """Stable identities for the measured initial-root reverse boundaries.

    Every callable receives the current trial data explicitly.  In particular,
    it may not close over a trial geometry, state, support payload, root, or
    cotangent.  The final root-geometry callable preserves the benchmark's
    derivative partition: it returns geometry bars for the supplied residual
    bars, with geometry delta as its only differentiated numerical input.
    """

    corrected_bootstrap_fluxes: Callable[..., Any]
    bootstrap_state_pullback: Callable[..., Any]
    bootstrap_geometry_pullback: Callable[..., Any]
    bootstrap_support_pullback: Callable[..., Any]
    root_geometry_residual_pullback: Callable[..., Any]

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            if not callable(getattr(self, field.name)):
                raise TypeError(f"{field.name} must be callable.")


@dataclasses.dataclass(frozen=True, slots=True)
class InitialRootReverseOptimizationStage:
    """Optimization-only owner of bounded initial-root reverse kernels.

    ``layout`` and ``kernels`` are stage-static. Trial data is deliberately
    absent from this object and must be passed to a kernel at invocation.
    This is not an outer JIT boundary.
    """

    layout: InitialRootStageLayout
    kernels: InitialRootReverseKernelSet


@dataclasses.dataclass(frozen=True, slots=True)
class InitialRootReverseDependencies:
    """Existing benchmark operations used by optimization-only adapters."""

    root_geometry_residual_pullback: Callable[..., Any]

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            if not callable(getattr(self, field.name)):
                raise TypeError(f"{field.name} must be callable.")


def build_initial_root_reverse_kernels_optimization(
    *,
    neoclassical_model: Any,
    dependencies: InitialRootReverseDependencies,
) -> InitialRootReverseKernelSet:
    """Create stable optimization-only adapters for the measured boundaries.

    ``neoclassical_model`` is stage-static. Each returned callable receives
    the current trial's state, geometry, support, and cotangents explicitly.
    The root-geometry adapter delegates to the *narrow*, delta-only benchmark
    pullback supplied by the caller. It deliberately does not turn that into
    an all-input VJP with stopped arguments: such a rewrite would enlarge the
    transformed argument tree and has not been shown to improve cache reuse.
    """

    if not dataclasses.is_dataclass(neoclassical_model):
        raise TypeError(
            "The optimization-only initial-root adapters require a dataclass "
            "neoclassical model so current geometry/support can be supplied "
            "with dataclasses.replace."
        )
    required_methods = (
        "evaluate_momentum_corrected_fluxes",
        "pullback_momentum_corrected_upar_state_by_radius",
        "pullback_momentum_corrected_upar_geometry_by_radius",
        "pullback_momentum_corrected_upar_support_by_radius",
    )
    missing_methods = tuple(
        name for name in required_methods if not callable(getattr(neoclassical_model, name, None))
    )
    if missing_methods:
        raise TypeError(
            "The optimization-only initial-root adapters require corrected "
            f"bootstrap callbacks; missing {missing_methods!r}."
        )

    def _model_for_trial(geometry, support):
        return dataclasses.replace(
            neoclassical_model,
            geometry=geometry,
            support=support,
        )

    def corrected_bootstrap_fluxes_optimization(rooted_state, geometry, support):
        return _model_for_trial(geometry, support).evaluate_momentum_corrected_fluxes(
            rooted_state
        )

    def bootstrap_state_pullback_optimization(rooted_state, upar_bar, geometry, support):
        return _model_for_trial(geometry, support).pullback_momentum_corrected_upar_state_by_radius(
            rooted_state,
            upar_bar,
        )

    def bootstrap_geometry_pullback_optimization(rooted_state, upar_bar, geometry, support):
        return _model_for_trial(geometry, support).pullback_momentum_corrected_upar_geometry_by_radius(
            rooted_state,
            upar_bar,
            geometry,
            support,
        )

    def bootstrap_support_pullback_optimization(rooted_state, upar_bar, geometry, support):
        return _model_for_trial(geometry, support).pullback_momentum_corrected_upar_support_by_radius(
            rooted_state,
            upar_bar,
            support,
        )

    def root_geometry_residual_pullback_optimization(
        pre_root_state,
        er_profile,
        current_geometry,
        current_support,
        residual_bars,
        geometry_delta,
    ):
        """Return geometry bars through the benchmark's delta-only VJP."""

        return dependencies.root_geometry_residual_pullback(
            pre_root_state,
            er_profile,
            current_geometry,
            current_support,
            residual_bars,
            geometry_delta,
        )

    return InitialRootReverseKernelSet(
        corrected_bootstrap_fluxes=corrected_bootstrap_fluxes_optimization,
        bootstrap_state_pullback=bootstrap_state_pullback_optimization,
        bootstrap_geometry_pullback=bootstrap_geometry_pullback_optimization,
        bootstrap_support_pullback=bootstrap_support_pullback_optimization,
        root_geometry_residual_pullback=root_geometry_residual_pullback_optimization,
    )


def build_initial_root_reverse_optimization_stage(
    *,
    layout: InitialRootStageLayout,
    corrected_bootstrap_fluxes: Callable[..., Any],
    bootstrap_state_pullback: Callable[..., Any],
    bootstrap_geometry_pullback: Callable[..., Any],
    bootstrap_support_pullback: Callable[..., Any],
    root_geometry_residual_pullback: Callable[..., Any],
) -> InitialRootReverseOptimizationStage:
    """Build the explicit non-jitted stage for the measured cache owners."""

    return InitialRootReverseOptimizationStage(
        layout=layout,
        kernels=InitialRootReverseKernelSet(
            corrected_bootstrap_fluxes=corrected_bootstrap_fluxes,
            bootstrap_state_pullback=bootstrap_state_pullback,
            bootstrap_geometry_pullback=bootstrap_geometry_pullback,
            bootstrap_support_pullback=bootstrap_support_pullback,
            root_geometry_residual_pullback=root_geometry_residual_pullback,
        ),
    )


@dataclasses.dataclass(slots=True)
class LazyStageArtifacts:
    """One-time structural artifacts derived from the first trial raw state.

    This container must never retain the raw solve itself or any trial arrays.
    It may retain only caller-validated structural Boozer constants/index maps.
    """

    booz_constants_grids: Any = None
    booz_mode_indices: Any = None
    initialized: bool = False

    def initialize(self, *, booz_constants_grids: Any, booz_mode_indices: Any) -> None:
        if self.initialized:
            return
        self.booz_constants_grids = booz_constants_grids
        self.booz_mode_indices = booz_mode_indices
        self.initialized = True


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

    # These wrappers are intentionally left un-jitted here: their ``kwargs``
    # mix static Python configuration with dynamic numerical values.  The
    # compiled stage is installed only through the fixed-signature factory.
    return GeometryInitialRootOptimizationStage(
        layout=layout,
        root_to_payload=root_to_payload,
        payload_to_vmec=payload_to_vmec,
    )


def compiled_initial_root_stage_operators(
    *,
    raw_block_stage: GeometryRawBlockStage,
    root_impl: Callable[[tuple[Any, Any, Any], Any], Any],
    payload_impl: Callable[[tuple[Any, Any, Any], Any, Any, Any], Any],
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Create the two fixed-signature compiled optimization operators.

    ``raw_block_stage`` is captured once.  Every numerical value is supplied
    through the explicit dynamic payload arguments; no trial object is held by
    the compiled closures.
    """

    del raw_block_stage  # Static ownership is established by the bound impls.
    return jax.jit(root_impl), jax.jit(payload_impl)


def build_compiled_geometry_initial_root_stage(
    *,
    layout: InitialRootStageLayout,
    raw_block_stage: GeometryRawBlockStage,
    root_impl: Callable[[GeometryRawBlockSolve, Any], Any],
    payload_impl: Callable[[GeometryRawBlockSolve, Any, Any, Any, Any], Any],
) -> GeometryInitialRootOptimizationStage:
    """Build the VMEX-like fixed-signature compiled stage.

    ``root_impl`` and ``payload_impl`` close only over stage-static objects.
    The raw solve is reconstructed from fresh dynamic leaves for every call.
    """

    def root_operator(dynamic_raw_payload, profile_values):
        raw_block_solve = raw_block_solve_from_dynamic_payload(raw_block_stage, dynamic_raw_payload)
        return root_impl(raw_block_solve, profile_values)

    def payload_operator(
        dynamic_raw_payload, geometry_deltas, objective_values, profile_gradient_matrix, support_bars
    ):
        raw_block_solve = raw_block_solve_from_dynamic_payload(raw_block_stage, dynamic_raw_payload)
        return payload_impl(
            raw_block_solve, geometry_deltas, objective_values, profile_gradient_matrix, support_bars
        )

    root_jit, payload_jit = compiled_initial_root_stage_operators(
        raw_block_stage=raw_block_stage,
        root_impl=root_operator,
        payload_impl=payload_operator,
    )
    return GeometryInitialRootOptimizationStage(
        layout=layout,
        root_to_payload=root_jit,
        payload_to_vmec=payload_jit,
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
