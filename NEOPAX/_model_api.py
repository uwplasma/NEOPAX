from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class ModelCapabilities:
    jit_safe: bool = True
    autodiff_safe: bool = False
    vmap_safe: bool = False
    local_evaluator: bool = False
    face_fluxes: bool = False


@dataclasses.dataclass(frozen=True)
class ModelValidationContext:
    builder_kwargs: dict[str, Any]
    state: Any
    species: Any | None = None
    geometry: Any | None = None
    face_state: Any | None = None


def make_validation_context(
    *,
    builder_kwargs: dict[str, Any] | None = None,
    n_species: int = 2,
    n_radial: int = 8,
    density_value: float = 1.0,
    temperature_value: float = 2.0,
    er_value: float = 0.0,
    species: Any | None = None,
    geometry: Any | None = None,
    include_face_state: bool = True,
) -> ModelValidationContext:
    """Build a small default validation context for user model registration.

    This helper is intentionally lightweight and avoids depending on geometry
    builders or larger runtime objects. It is meant for registration-time smoke
    tests, not for physical validation.
    """
    from ._state import TransportState

    density = jnp.full((n_species, n_radial), density_value, dtype=jnp.float64)
    pressure = density * jnp.asarray(temperature_value, dtype=jnp.float64)
    er = jnp.full((n_radial,), er_value, dtype=jnp.float64)
    state = TransportState(density=density, pressure=pressure, Er=er)

    face_state = None
    if include_face_state:
        face_density = jnp.full((n_species, n_radial + 1), density_value, dtype=jnp.float64)
        face_pressure = face_density * jnp.asarray(temperature_value, dtype=jnp.float64)
        face_er = jnp.full((n_radial + 1,), er_value, dtype=jnp.float64)
        face_state = TransportState(density=face_density, pressure=face_pressure, Er=face_er)

    return ModelValidationContext(
        builder_kwargs={} if builder_kwargs is None else dict(builder_kwargs),
        state=state,
        species=species,
        geometry=geometry,
        face_state=face_state,
    )


def _as_jax_array(value: Any) -> jax.Array:
    return jnp.asarray(value)


def _assert_jax_compatible_pytree(value: Any, *, where: str) -> None:
    try:
        leaves = jax.tree_util.tree_leaves(value)
    except Exception as exc:
        raise TypeError(f"{where} is not a valid JAX pytree.") from exc
    for leaf in leaves:
        try:
            jnp.asarray(leaf)
        except Exception as exc:
            raise TypeError(f"{where} contains a non-array-compatible leaf of type {type(leaf)!r}.") from exc


def _assert_broadcastable(value: Any, target_shape: tuple[int, ...], *, where: str) -> None:
    arr = _as_jax_array(value)
    if arr.shape == target_shape:
        return
    if arr.ndim == 0:
        return
    if len(target_shape) >= 2 and arr.shape == target_shape[1:]:
        return
    if len(target_shape) >= 2 and arr.ndim == 1 and arr.shape[0] == target_shape[0]:
        return
    try:
        jnp.broadcast_to(arr, target_shape)
    except Exception as exc:
        raise ValueError(f"{where} has shape {arr.shape}, which is not compatible with target shape {target_shape}.") from exc


def validate_transport_flux_output(output: Any, state: Any, *, where: str = "transport model output") -> None:
    if not isinstance(output, dict):
        raise TypeError(f"{where} must be a dict with keys 'Gamma', 'Q', and 'Upar'.")
    for key in ("Gamma", "Q", "Upar"):
        if key not in output:
            raise KeyError(f"{where} is missing required key '{key}'.")
        _assert_broadcastable(output[key], tuple(state.density.shape), where=f"{where}[{key}]")
    _assert_jax_compatible_pytree(output, where=where)


def validate_source_output(output: Any, state: Any, *, where: str = "source model output") -> None:
    if output is None:
        return
    if not isinstance(output, dict):
        raise TypeError(f"{where} must be a dict mapping source names to arrays/scalars.")
    for key, value in output.items():
        try:
            _assert_broadcastable(value, tuple(state.pressure.shape), where=f"{where}[{key}]")
        except ValueError:
            _assert_broadcastable(value, tuple(state.density.shape), where=f"{where}[{key}]")
    _assert_jax_compatible_pytree(output, where=where)


def _jax_shape_smoke_test(callable_obj: Callable[..., Any], *args: Any, where: str) -> None:
    try:
        jax.eval_shape(callable_obj, *args)
    except Exception as exc:
        raise TypeError(f"{where} failed a JAX eval_shape smoke test.") from exc


def _jax_jit_smoke_test(callable_obj: Callable[..., Any], *args: Any, where: str) -> None:
    try:
        jax.jit(callable_obj)(*args)
    except Exception as exc:
        raise TypeError(f"{where} failed a JAX jit smoke test.") from exc


def _jax_autodiff_smoke_test(callable_obj: Callable[..., Any], state: Any, *, where: str) -> None:
    if getattr(state, "Er", None) is None:
        return

    def scalar_fn(er):
        new_state = dataclasses.replace(state, Er=er)
        output = callable_obj(new_state)
        leaves = jax.tree_util.tree_leaves(output)
        total = jnp.asarray(0.0, dtype=jnp.asarray(er).dtype)
        for leaf in leaves:
            total = total + jnp.sum(jnp.asarray(leaf))
        return total

    try:
        jax.grad(scalar_fn)(state.Er)
    except Exception as exc:
        raise TypeError(f"{where} failed a JAX autodiff smoke test.") from exc


def _jax_vmap_smoke_test(callable_obj: Callable[..., Any], state: Any, *, where: str) -> None:
    if getattr(state, "Er", None) is None:
        return
    er_stack = jnp.stack([state.Er, state.Er], axis=0)

    def mapped(er):
        new_state = dataclasses.replace(state, Er=er)
        return callable_obj(new_state)

    try:
        jax.vmap(mapped)(er_stack)
    except Exception as exc:
        raise TypeError(f"{where} failed a JAX vmap smoke test.") from exc


def validate_transport_flux_builder(
    builder: Callable[..., Any],
    context: ModelValidationContext,
    *,
    capabilities: ModelCapabilities | None = None,
    name: str = "transport model",
) -> None:
    model = builder(**context.builder_kwargs)
    output = model(context.state)
    validate_transport_flux_output(output, context.state, where=f"{name} output")
    _jax_shape_smoke_test(model, context.state, where=f"{name}.__call__")

    caps = capabilities or ModelCapabilities()
    if caps.jit_safe:
        _jax_jit_smoke_test(model, context.state, where=f"{name}.__call__")
    if caps.autodiff_safe:
        _jax_autodiff_smoke_test(model, context.state, where=f"{name}.__call__")
    if caps.vmap_safe:
        _jax_vmap_smoke_test(model, context.state, where=f"{name}.__call__")
    if caps.local_evaluator:
        evaluator = model.build_local_particle_flux_evaluator(context.state)
        if evaluator is None:
            raise ValueError(f"{name} declared local_evaluator support but returned None.")
        sample = evaluator(0, _as_jax_array(context.state.Er[0]))
        _assert_broadcastable(sample, (context.state.density.shape[0],), where=f"{name} local evaluator")
    if caps.face_fluxes and context.face_state is not None:
        face_output = model.evaluate_face_fluxes(context.state, context.face_state)
        if face_output is None:
            raise ValueError(f"{name} declared face_fluxes support but returned None.")
        validate_transport_flux_output(face_output, context.face_state, where=f"{name} face output")


def validate_source_model_builder(
    builder: Callable[..., Any],
    context: ModelValidationContext,
    *,
    name: str = "source model",
) -> None:
    model = builder(**context.builder_kwargs)
    output = model(context.state)
    validate_source_output(output, context.state, where=f"{name} output")
    _jax_shape_smoke_test(model, context.state, where=f"{name}.__call__")


def transport_model(name: str, registry_fn: Callable[..., None], **register_kwargs):
    def decorator(builder: Callable[..., Any]) -> Callable[..., Any]:
        registry_fn(name, builder, **register_kwargs)
        return builder

    return decorator


def source_model(name: str, registry_fn: Callable[..., None], **register_kwargs):
    def decorator(builder: Callable[..., Any]) -> Callable[..., Any]:
        registry_fn(name, builder, **register_kwargs)
        return builder

    return decorator
