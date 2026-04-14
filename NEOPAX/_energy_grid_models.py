"""
Modular energy grid models for NEOPAX.
- Registry-based selection (like transport_flux_models.py)
- JIT-compatible, differentiable (JAX-friendly)
- No species or radial grid info (energy grid only)
"""
import dataclasses
import functools
import jax
import jax.numpy as jnp
from typing import Callable, Any
from jaxtyping import Array, Float, Int  # https://github.com/google/jaxtyping
import orthax

ENERGY_GRID_MODEL_REGISTRY: dict[str, Callable[..., "EnergyGridModelBase"]] = {}

def register_energy_grid_model(name: str, builder: Callable[..., "EnergyGridModelBase"]):
    ENERGY_GRID_MODEL_REGISTRY[str(name).strip().lower()] = builder

def get_energy_grid_model(name: str, **kwargs) -> "EnergyGridModelBase":
    key = str(name).strip().lower()
    if key not in ENERGY_GRID_MODEL_REGISTRY:
        raise ValueError(f"Unknown energy grid model '{name}'.")
    return ENERGY_GRID_MODEL_REGISTRY[key](**kwargs)

@jax.tree_util.register_dataclass
@dataclasses.dataclass(eq=False)
class EnergyGridModelBase:
    """Abstract base class for energy grid models."""
    pass







@jax.tree_util.register_dataclass
@dataclasses.dataclass(eq=False)
class StandardLaguerreEnergyGrid(EnergyGridModelBase):
    #Defining radial and energy grids
    n_x: int
    n_order: int
    Sonine_expansion : Float[Array, "n_order"]   
    sonine_indeces : Int[Array, "n_order"] 
    x : Float[Array, "n_x"]      
    xWeights : Float[Array, "n_x"] 
    v_norm : Float[Array, "n_x"] 
    L11_weight : Float[Array, "n_x"] 
    L12_weight : Float[Array, "n_x"] 
    L22_weight : Float[Array, "n_x"] 
    L13_weight : Float[Array, "n_x"] 
    L23_weight : Float[Array, "n_x"] 
    L33_weight : Float[Array, "n_x"] 
    L24_weight : Float[Array, "n_x"] 
    L25_weight : Float[Array, "n_x"]                    
    L43_weight : Float[Array, "n_x"] 
    L44_weight : Float[Array, "n_x"] 
    L45_weight : Float[Array, "n_x"]   
    L55_weight : Float[Array, "n_x"]     
    def __init__(
        self,
        n_x: int,
        n_order: int = 3,
        **kwargs,
    ):
        # Ensure n_x is a Python int (not a JAX tracer)
        self.n_x = n_x
        self.n_order = n_order

        # JAX dataclass pytree reconstruction may call StandardLaguerreEnergyGrid(...) with all
        # fields as keyword arguments. If the full payload is present, load it
        # directly instead of regenerating derived arrays.
        grid_fields = [f.name for f in dataclasses.fields(type(self))]
        payload_fields = [name for name in grid_fields if name not in ("n_x", "n_order")]
        if all(name in kwargs for name in payload_fields):
            for name in payload_fields:
                setattr(self, name, kwargs[name])
            return

        self.Sonine_expansion = jnp.array([1.0, 0.4, 8.0/35.0]) if self.n_order == 3 else jnp.ones(self.n_order)
        self.sonine_indeces = jnp.arange(len(self.Sonine_expansion))
        xgrid = orthax.laguerre.laggauss(self.n_x)
        self.x = xgrid[0]
        self.xWeights = xgrid[1]
        self.v_norm = jnp.sqrt(self.x)
        self.L11_weight = jnp.power(self.x, 2.)
        self.L12_weight = jnp.power(self.x, 3.)
        self.L22_weight = jnp.power(self.x, 4.)
        self.L13_weight = jnp.power(self.x, 1.5)
        self.L23_weight = jnp.power(self.x, 2.5)
        self.L33_weight = jnp.power(self.x, 1.)
        self.L24_weight = jnp.power(self.x, 3.5)
        self.L25_weight = jnp.power(self.x, 4.5)
        self.L43_weight = jnp.power(self.x, 2.)
        self.L44_weight = jnp.power(self.x, 3.)
        self.L45_weight = jnp.power(self.x, 4.)
        self.L55_weight = jnp.power(self.x, 5.)
        self.L55_weight=jnp.power(self.x,5.)

register_energy_grid_model("standard_laguerre", StandardLaguerreEnergyGrid)
