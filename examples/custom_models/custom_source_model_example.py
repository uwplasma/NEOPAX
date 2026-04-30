import dataclasses

import jax.numpy as jnp

import NEOPAX


@dataclasses.dataclass(frozen=True, eq=False)
class ExampleSourceModel:
    amplitude: float = 1.0

    def __call__(self, state):
        return {
            "pressure_source": self.amplitude * jnp.ones_like(state.pressure),
        }

def register_example_source_model():
    context = NEOPAX.make_validation_context(
        builder_kwargs={"amplitude": 1.0},
        n_species=2,
        n_radial=8,
    )
    NEOPAX.register_source_model(
        "example_source_model",
        ExampleSourceModel,
        validate=True,
        validation_context=context,
    )


if __name__ == "__main__":
    register_example_source_model()
    context = NEOPAX.make_validation_context(n_species=2, n_radial=8)
    model = NEOPAX.get_source_model("example_source_model", amplitude=3.0)
    out = model(context.state)
    print("Registered example_source_model")
    print("pressure_source shape:", out["pressure_source"].shape)
