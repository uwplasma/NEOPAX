"""Modular source-model registry and config builders for NEOPAX.

This mirrors the transport-flux registry style so source models can be selected
and composed from runtime configuration.
"""

from __future__ import annotations

from typing import Any, Callable
import dataclasses

import jax
import jax.numpy as jnp


SOURCE_MODEL_REGISTRY: dict[str, Callable[..., "SourceModelBase"]] = {}
_DEFAULTS_REGISTERED = False


def register_source_model(name: str, builder: Callable[..., "SourceModelBase"]) -> None:
    SOURCE_MODEL_REGISTRY[str(name).strip().lower()] = builder


# Backward-compatible alias used by existing examples/tests.
register_source = register_source_model


def _ensure_default_source_models_registered() -> None:
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return

    # Local import avoids hard import-cycle at module import time.
    from ._source_models import (
        FusionPowerFractionElectronsSource,
        DTReactionSource,
        PowerExchangeSource,
        BremsstrahlungRadiationSource,
    )

    register_source_model("fusion_power_fraction_electrons", FusionPowerFractionElectronsSource)
    register_source_model("dt_reaction", DTReactionSource)
    register_source_model("power_exchange", PowerExchangeSource)
    register_source_model("bremsstrahlung_radiation", BremsstrahlungRadiationSource)
    _DEFAULTS_REGISTERED = True


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class SourceModelBase:
    """Base class for non-conservative source models."""

    def __call__(self, state: Any):
        raise NotImplementedError


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class AnalyticSource(SourceModelBase):
    """Simple source wrapper: constant profile array or callable(state) -> array."""

    profile: Any

    def __call__(self, state: Any):
        if callable(self.profile):
            return self.profile(state)
        return jnp.asarray(self.profile)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class ExampleStateDrivenSource(SourceModelBase):
    scale: float = 1.0

    def __call__(self, state: Any):
        te = state["Te"]
        ne = state["ne"]
        return self.scale * ne * jnp.sqrt(te)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class CombinedSourceModel(SourceModelBase):
    sources: tuple[SourceModelBase, ...] = dataclasses.field(default_factory=tuple)

    def __call__(self, state: Any):
        out = self.sources[0](state)
        for src in self.sources[1:]:
            out = out + src(state)
        return out


def get_source_model(name: str, **kwargs) -> SourceModelBase:
    _ensure_default_source_models_registered()
    key = str(name).strip().lower()
    if key not in SOURCE_MODEL_REGISTRY:
        raise ValueError(f"Unknown source model '{name}'.")
    return SOURCE_MODEL_REGISTRY[key](**kwargs)


# Backward-compatible alias used by existing examples/tests.
def get_source(name: str, *args, **kwargs) -> SourceModelBase:
    if args:
        raise TypeError("Positional args are not supported for source builders; use keyword args.")
    return get_source_model(name, **kwargs)


def _as_name_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    v = str(value).strip()
    return [v] if len(v) > 0 else []


def _builder_kwargs_for(source_name: str, params_cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params_cfg, dict):
        return {}
    return dict(params_cfg.get(source_name, {})) if isinstance(params_cfg.get(source_name, {}), dict) else {}


def _compose_sources(names: list[str], params_cfg: dict[str, Any] | None = None) -> SourceModelBase | None:
    if len(names) == 0:
        return None
    params_cfg = params_cfg or {}
    models = tuple(
        get_source_model(name, **_builder_kwargs_for(name, params_cfg))
        for name in names
    )
    return CombinedSourceModel(models)


def build_source_models_from_config(cfg: dict[str, Any]) -> dict[str, SourceModelBase] | None:
    """Build density/temperature source callables from TOML-style config.

    Supported schema:

      [sources]
      density = ["name1", "name2"]
      temperature = ["name3"]

      [sources.parameters]
      name3 = {some_kw = 1.0}
    """
    src_cfg = cfg.get("sources", {}) if isinstance(cfg, dict) else {}
    if not isinstance(src_cfg, dict):
        return None

    params_cfg = src_cfg.get("parameters", {})
    density_src = _compose_sources(_as_name_list(src_cfg.get("density")), params_cfg)
    temp_src = _compose_sources(_as_name_list(src_cfg.get("temperature")), params_cfg)

    out: dict[str, SourceModelBase] = {}
    if density_src is not None:
        out["density"] = density_src
    if temp_src is not None:
        out["temperature"] = temp_src
    return out if len(out) > 0 else None


# Register simple built-ins that are defined in this module.
register_source_model("analytic", AnalyticSource)
register_source_model("example_state", ExampleStateDrivenSource)
