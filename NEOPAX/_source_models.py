import jax.numpy as jnp
import jax
from jax import jit
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
import interpax
from ._constants import proton_mass,elementary_charge
from ._species import coulomb_logarithm

#These should go to the physics_models.py 
#Get FusionPower Fraction to Electrons, using same model as NTSS - update in the future


import dataclasses
import jax
import jax.numpy as jnp
from typing import Any, Callable
from ._sources import fusion_power_fraction_electrons, dt_reaction, power_exchange, bremsstrahlung_radiation

# Registry for modular selection
SOURCE_MODEL_REGISTRY: dict[str, Callable[..., "SourceModelBase"]] = {}
_DEFAULTS_REGISTERED = False

def register_source_model(name: str, builder: Callable[..., "SourceModelBase"]) -> None:
    SOURCE_MODEL_REGISTRY[str(name).strip().lower()] = builder

def register_source(name: str, builder: Callable[..., "SourceModelBase"]) -> None:
    register_source_model(name, builder)

def _ensure_default_source_models_registered() -> None:
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    register_source_model("fusion_power_fraction_electrons", FusionPowerFractionElectronsSource)
    register_source_model("dt_reaction", DTReactionSource)
    register_source_model("power_exchange", PowerExchangeSource)
    register_source_model("bremsstrahlung_radiation", BremsstrahlungRadiationSource)
    register_source_model("analytic", AnalyticSource)
    register_source_model("example_state", ExampleStateDrivenSource)
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
        outs = [src(state) for src in self.sources]
        merged: dict[str, Any] = {}
        for out in outs:
            if out is None:
                continue
            if not isinstance(out, dict):
                raise TypeError("CombinedSourceModel expects each source to return a dict.")
            for key, value in out.items():
                if key in merged:
                    merged[key] = merged[key] + value
                else:
                    merged[key] = value
        return merged

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class FusionPowerFractionElectronsSource(SourceModelBase):
    species: Any = dataclasses.field(repr=False, default=None)
    def __call__(self, state, species=None):
        active_species = self.species if self.species is not None else species
        return {"fusion_power_fraction_electrons": fusion_power_fraction_electrons(state, active_species)}

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class DTReactionSource(SourceModelBase):
    species: Any = dataclasses.field(repr=False, default=None)
    def __call__(self, state, species=None):
        active_species = self.species if self.species is not None else species
        DTreactionRate, HeSource, AlphaPower = dt_reaction(state, active_species)
        return {
            "DTreactionRate": DTreactionRate,
            "HeSource": HeSource,
            "AlphaPower": AlphaPower,
        }

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PowerExchangeSource(SourceModelBase):
    species: Any = dataclasses.field(repr=False, default=None)
    idx_a: str = None
    idx_b: str = None
    def __call__(self, state, species=None):
        active_species = self.species if self.species is not None else species
        return {"power_exchange": power_exchange(state, species=active_species)}

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class BremsstrahlungRadiationSource(SourceModelBase):
    species: Any = dataclasses.field(repr=False, default=None)
    delta_zeff: float = 0.0
    def __call__(self, state, species=None):
        active_species = self.species if self.species is not None else species
        PBrems, Zeff = bremsstrahlung_radiation(
            state,
            species=active_species,
            delta_zeff=self.delta_zeff,
        )
        return {"PBrems": PBrems, "Zeff": Zeff}

def get_source_model(name: str, **kwargs) -> SourceModelBase:
    _ensure_default_source_models_registered()
    key = str(name).strip().lower()
    if key not in SOURCE_MODEL_REGISTRY:
        raise ValueError(f"Unknown source model '{name}'.")
    return SOURCE_MODEL_REGISTRY[key](**kwargs)

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

def _builder_kwargs_for(source_name: str, params_cfg: dict[str, Any], species: Any | None = None) -> dict[str, Any]:
    if not isinstance(params_cfg, dict):
        out = {}
    else:
        out = dict(params_cfg.get(source_name, {})) if isinstance(params_cfg.get(source_name, {}), dict) else {}
    if species is not None and "species" not in out:
        out["species"] = species
    return out

def _compose_sources(names: list[str], params_cfg: dict[str, Any] | None = None, species: Any | None = None) -> SourceModelBase | None:
    if len(names) == 0:
        return None
    params_cfg = params_cfg or {}
    models = tuple(
        get_source_model(name, **_builder_kwargs_for(name, params_cfg, species))
        for name in names
    )
    return CombinedSourceModel(models)

def _broadcast_profile(value: Any, template: jax.Array) -> jax.Array:
    arr = jnp.asarray(value, dtype=template.dtype)
    if arr.shape == template.shape:
        return arr
    if arr.ndim == template.ndim - 1:
        return jnp.broadcast_to(arr[None, :], template.shape)
    return jnp.broadcast_to(arr, template.shape)


def _species_component(template: jax.Array, species_index: int, profile: Any) -> jax.Array:
    arr = jnp.asarray(profile, dtype=template.dtype)
    out = jnp.zeros_like(template)
    return out.at[species_index].set(arr)


def sum_source_components(components: dict[str, jax.Array] | None, template: jax.Array) -> jax.Array:
    if not components:
        return jnp.zeros_like(template)
    total = jnp.zeros_like(template)
    for value in components.values():
        total = total + jnp.asarray(value, dtype=template.dtype)
    return total


def assemble_density_source_components(source_value: Any, state: Any, species: Any) -> dict[str, jax.Array]:
    template = state.density
    if source_value is None:
        return {}
    if not isinstance(source_value, dict):
        return {"configured": _broadcast_profile(source_value, template)}

    out: dict[str, jax.Array] = {}
    species_idx = species.species_idx if hasattr(species, "species_idx") else {}
    he_idx = species_idx.get("He")
    d_idx = species_idx.get("D")
    t_idx = species_idx.get("T")
    for key, value in source_value.items():
        if key == "HeSource":
            if he_idx is not None:
                out["HeSource"] = _species_component(template, he_idx, value)
                he_profile = jnp.asarray(value, dtype=template.dtype)
                if d_idx is not None:
                    out["fusion_sink_D"] = -_species_component(template, d_idx, he_profile)
                if t_idx is not None:
                    out["fusion_sink_T"] = -_species_component(template, t_idx, he_profile)
            continue
        if key in {"DTreactionRate", "AlphaPower", "PBrems", "Zeff", "fusion_power_fraction_electrons", "power_exchange"}:
            continue
        out[key] = _broadcast_profile(value, template)
    return out


def assemble_pressure_source_components(source_value: Any, state: Any, species: Any) -> dict[str, jax.Array]:
    template = state.pressure
    if source_value is None:
        return {}
    if not isinstance(source_value, dict):
        return {"configured": _broadcast_profile(source_value, template)}

    out: dict[str, jax.Array] = {}
    species_idx = species.species_idx if hasattr(species, "species_idx") else {}
    electron_idx = species_idx.get("e")
    alpha_power = None
    electron_fraction = None

    for key, value in source_value.items():
        if key == "power_exchange":
            out["power_exchange"] = _broadcast_profile(value, template)
            continue
        if key == "PBrems":
            if electron_idx is not None:
                out["bremsstrahlung"] = -_species_component(template, electron_idx, value)
            continue
        if key == "AlphaPower":
            alpha_power = jnp.asarray(value, dtype=template.dtype)
            continue
        if key == "fusion_power_fraction_electrons":
            electron_fraction = jnp.asarray(value, dtype=template.dtype)
            continue
        if key in {"DTreactionRate", "HeSource", "Zeff"}:
            continue
        out[key] = _broadcast_profile(value, template)

    if alpha_power is not None:
        if electron_fraction is None:
            electron_fraction = jnp.zeros_like(alpha_power)
        alpha_component = jnp.zeros_like(template)
        if electron_idx is not None:
            alpha_component = alpha_component.at[electron_idx].set(electron_fraction * alpha_power)
        ion_indices = [species_idx[name] for name in ("D", "T") if name in species_idx]
        if len(ion_indices) == 0 and hasattr(species, "ion_indices"):
            ion_indices = list(species.ion_indices)
        if len(ion_indices) > 0:
            ion_share = ((1.0 - electron_fraction) * alpha_power) / float(len(ion_indices))
            alpha_component = alpha_component.at[jnp.asarray(ion_indices)].add(
                jnp.broadcast_to(ion_share, (len(ion_indices),) + ion_share.shape)
            )
        out["alpha_power"] = alpha_component
    return out


def build_source_models_from_config(cfg: dict[str, Any], species: Any | None = None) -> dict[str, SourceModelBase] | None:
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
    density_src = _compose_sources(_as_name_list(src_cfg.get("density")), params_cfg, species)
    temp_src = _compose_sources(_as_name_list(src_cfg.get("temperature")), params_cfg, species)

    out: dict[str, SourceModelBase] = {}
    if density_src is not None:
        out["density"] = density_src
    if temp_src is not None:
        out["temperature"] = temp_src
    return out if len(out) > 0 else None

