Custom Flux And Source Models
=============================

NEOPAX supports user-defined transport flux models and source models through a
small registration API. The goal is to make custom models easier to add while
still checking that they remain compatible with:

- JAX arrays and pytrees
- JIT tracing
- expected output keys
- expected output shapes


What The Extension API Provides
-------------------------------

The main public pieces are:

- ``NEOPAX.register_transport_flux_model(...)``
- ``NEOPAX.register_source_model(...)``
- ``NEOPAX.transport_flux_model(...)``
- ``NEOPAX.source_model(...)``
- ``NEOPAX.ModelCapabilities``
- ``NEOPAX.ModelValidationContext``
- ``NEOPAX.make_validation_context(...)``

In practice, most users will use:

- direct registration with validation, or
- decorator-based registration

Once a model is registered in the Python process, it can be referenced by name
through the normal TOML configuration flow just like a built-in model.


Transport Flux Model Contract
-----------------------------

A transport flux model must return a dict containing:

- ``Gamma``
- ``Q``
- ``Upar``

Each of these must be compatible with the transport state density shape:

- usually ``(n_species, n_r)``

The model should be:

- side-effect free in its hot call path
- JAX-array compatible
- safe to evaluate under ``jax.eval_shape(...)``

Optional extra capabilities can be declared, such as:

- ``jit_safe=True``
- ``autodiff_safe=True``
- ``vmap_safe=True``
- ``local_evaluator=True``
- ``face_fluxes=True``


Source Model Contract
---------------------

A source model must return a dict mapping source names to arrays or scalars.

Those outputs should be broadcast-compatible with either:

- the density shape, or
- the pressure shape

depending on which kind of source is being represented.

Like flux models, source models should remain:

- JAX-friendly
- side-effect free in the main call path
- shape-stable


Validated Registration
----------------------

The simplest validated registration pattern is:

.. code-block:: python

    import dataclasses
    import jax.numpy as jnp
    import NEOPAX


    @dataclasses.dataclass(frozen=True, eq=False)
    class MyFluxModel:
        amplitude: float = 1.0

        def __call__(self, state, geometry=None, params=None):
            del geometry, params
            base = self.amplitude * jnp.ones_like(state.density)
            return {
                "Gamma": base,
                "Q": 2.0 * base,
                "Upar": jnp.zeros_like(base),
            }


    context = NEOPAX.make_validation_context(
        builder_kwargs={"amplitude": 1.0},
        n_species=2,
        n_radial=8,
    )

    NEOPAX.register_transport_flux_model(
        "my_flux_model",
        MyFluxModel,
        capabilities=NEOPAX.ModelCapabilities(local_evaluator=False),
        validate=True,
        validation_context=context,
    )

The validation step checks that:

- the builder works
- the output contains the required keys
- the output shapes are compatible
- the model survives a lightweight ``jax.eval_shape(...)`` pass
- and, when requested through capabilities, can also smoke-test:
  - ``jax.jit(...)``
  - ``jax.grad(...)``
  - ``jax.vmap(...)``
  - local particle-flux evaluator support
  - face-flux evaluation support


Decorator Registration
----------------------

For code you want to ship or reuse repeatedly, the decorator form is a little
cleaner.

Transport model:

.. code-block:: python

    import dataclasses
    import jax.numpy as jnp
    import NEOPAX


    @NEOPAX.transport_flux_model("my_decorated_flux_model")
    @dataclasses.dataclass(frozen=True, eq=False)
    class MyDecoratedFluxModel:
        scale: float = 1.0

        def __call__(self, state, geometry=None, params=None):
            del geometry, params
            base = self.scale * jnp.ones_like(state.density)
            return {"Gamma": base, "Q": base, "Upar": jnp.zeros_like(base)}

Source model:

.. code-block:: python

    import dataclasses
    import jax.numpy as jnp
    import NEOPAX


    @NEOPAX.source_model("my_decorated_source_model")
    @dataclasses.dataclass(frozen=True, eq=False)
    class MyDecoratedSourceModel:
        amplitude: float = 1.0

        def __call__(self, state):
            return {
                "pressure_source": self.amplitude * jnp.ones_like(state.pressure)
            }

If you want decorator-based registration plus validation, the recommended
pattern is still to keep an explicit registration block in test or setup code,
because validation requires a concrete example state.


Using ``make_validation_context(...)``
--------------------------------------

The easiest way to build a small validation context is:

.. code-block:: python

    import NEOPAX

    context = NEOPAX.make_validation_context(
        builder_kwargs={"amplitude": 1.0},
        n_species=2,
        n_radial=8,
    )

This helper creates:

- a default transport state
- a matching face state
- a ready-to-use ``builder_kwargs`` payload

It is intended for registration-time smoke tests, not for full physical model
verification.


Minimal Flux Example
--------------------

See:

- ``examples/custom_models/custom_flux_model_example.py``
- ``examples/custom_models/ntx_runtime_scan_direct_api_example.py``

This example shows:

- a simple custom flux model
- validated registration
- lookup through the standard model registry

The runtime NTX example is not a user-registered model example. Instead, it
shows how a built-in model can expose a cleaner direct-Python workflow by
separating static setup from later evaluation.


Minimal Source Example
----------------------

See:

- ``examples/custom_models/custom_source_model_example.py``

This example shows:

- a simple custom source model
- validated registration
- lookup through the standard source-model registry


Using Custom Models From TOML
-----------------------------

If a Python module registers custom models when it is imported, NEOPAX can load
that module before building models from the TOML file.

Example config fragment:

.. code-block:: toml

    [extensions]
    python_modules = ["my_project.neopax_models"]

    [neoclassical]
    flux_model = "my_flux_model"

    [sources]
    temperature = ["my_source_model"]

You can also load a local Python file directly:

.. code-block:: toml

    [extensions]
    python_files = ["./user_models.py"]

Relative ``python_files`` paths are resolved relative to the config file
directory when the case is launched from a TOML path.

The expected pattern is:

1. your module imports ``NEOPAX``
2. it registers one or more models
3. the TOML file references those registered names

This keeps the TOML-driven path and the direct Python registration path
compatible with each other.


Practical Guidance
------------------

To stay JAX/JIT/differentiation-friendly, user models should:

- return only JAX-compatible arrays/scalars/pytrees
- avoid Python-side mutation in ``__call__``
- avoid hidden file I/O inside the hot evaluation path
- avoid changing output rank/shape based on runtime values
- keep control flow compatible with JAX tracing

Good first use cases for custom models are:

- prescribed analytic flux profiles
- modified turbulence closures
- problem-specific source terms
- reduced-order surrogate closures


Current Limits
--------------

The current validation layer is a first pass. It already checks:

- key presence
- shape compatibility
- JAX pytree compatibility
- lightweight shape tracing
- optional JIT/autodiff/vmap smoke tests when the declared capabilities request them

Planned follow-up improvements include:

- clearer capability-based validation paths
- more polished user-facing example coverage


See Also
--------

- :doc:`methods_of_use`
- :doc:`worked_examples`
- :doc:`input_file_reference`
