import dataclasses

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class TurbulenceState:
    Gamma_turb: jnp.ndarray
    Q_turb: jnp.ndarray
