import dataclasses

import jax.numpy as jnp

import NEOPAX


@dataclasses.dataclass(frozen=True, eq=False)
class ExampleFluxModel:
    amplitude: float = 1.0

    def __call__(self, state, geometry=None, params=None):
        del geometry, params
        base = self.amplitude * jnp.ones_like(state.density)
        return {
            "Gamma": base,
            "Q": 2.0 * base,
            "Upar": jnp.zeros_like(base),
        }

def register_example_flux_model():
    context = NEOPAX.make_validation_context(
        builder_kwargs={"amplitude": 1.0},
        n_species=2,
        n_radial=8,
    )
    NEOPAX.register_transport_flux_model(
        "example_flux_model",
        ExampleFluxModel,
        capabilities=NEOPAX.ModelCapabilities(),
        validate=True,
        validation_context=context,
    )


if __name__ == "__main__":
    register_example_flux_model()
    context = NEOPAX.make_validation_context(n_species=2, n_radial=8)
    model = NEOPAX.get_transport_flux_model("example_flux_model")(amplitude=3.0)
    out = model(context.state)
    print("Registered example_flux_model")
    print("Gamma shape:", out["Gamma"].shape)
