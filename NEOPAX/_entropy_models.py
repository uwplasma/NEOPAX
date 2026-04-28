"""
Modular entropy model registry for NEOPAX ambipolar root-finding.

Each entropy model should be a callable that takes (state, fluxes, ...), and returns a scalar entropy.
"""

from typing import Callable, Any, Dict

ENTROPY_MODEL_REGISTRY: Dict[str, Callable[..., float]] = {}


def register_entropy_model(name: str, builder: Callable[..., float]) -> None:
    ENTROPY_MODEL_REGISTRY[str(name).strip().lower()] = builder

def get_entropy_model(name: str) -> Callable[..., float]:
    key = str(name).strip().lower()
    if key not in ENTROPY_MODEL_REGISTRY:
        raise ValueError(f"Unknown entropy model '{name}'.")
    return ENTROPY_MODEL_REGISTRY[key]

# Example: Monkes-style entropy (sum of |Gamma|)
def monkes_database_entropy(state, fluxes, **kwargs):
    # Assume fluxes is a dict with 'Gamma_total' (shape: [species, ...])
    # and state has 'species' with 'charge_qp' attribute
    Gamma = fluxes.get("Gamma_total")
    if Gamma is None:
        raise ValueError("Flux model did not return 'Gamma_total'.")
    # Default: sum over all species and space
    return float(jnp.sum(jnp.abs(Gamma)))

try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp

register_entropy_model("monkes_database", monkes_database_entropy)
# Flux-file transport currently uses the same simple ambipolar entropy proxy:
# sum over absolute species particle fluxes. Registering this alias keeps
# entropy-model resolution consistent when [neoclassical].flux_model is set
# to "fluxes_r_file".
register_entropy_model("fluxes_r_file", monkes_database_entropy)
