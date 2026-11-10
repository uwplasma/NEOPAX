JAX GPU environment
===================

This guide records the environment that passed GPU initialization, a small
compiled matrix/AD check, and the required VMEX/NTX interface checks on
9 November 2026. It uses Python 3.12, JAX/jaxlib 0.10.0 and CUDA 12 wheels,
with editable source checkouts in ``$HOME`` and a virtual environment named
``jax_env``. These installation checks do not establish full transport or
reverse-AD benchmark parity; that validation remains separate.

Check the machine
-----------------

Run these commands in a Linux bash shell. They assume Git and Python 3.12
with the ``venv`` module are already available. For Windows, use a WSL2 Linux
shell with NVIDIA GPU access rather than native PowerShell; JAX lists WSL2
GPU support as experimental.

.. code-block:: bash

   python3.12 --version
   git --version
   nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv

CUDA 12 wheels require a Linux NVIDIA driver of at least version 525.
The wheels supply the CUDA user-space libraries, so a separate system CUDA
toolkit installation is unnecessary for this setup. See the
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html#nvidia-gpu>`_.

JAX 0.10.0 was selected for the GRID A100X virtual GPU. Certain fractional
vGPU profiles cannot initialize with JAX/jaxlib 0.10.1 through 0.11.2 because
those builds require CUDA virtual memory management; the
`upstream vGPU report <https://github.com/openxla/xla/issues/49252>`_
documents the regression. This pin reproduces the environment checked here;
it is not a claim that all later releases fail on every GPU. VMEX's main-based
checkout requires JAX/jaxlib >=0.9.2 and SciPy >=1.16 on Python >=3.11, so the
older JAX 0.7.2 environment does not satisfy its declared requirements.

Clone the source checkouts
-------------------------

For a fresh machine without these directories, clone:

.. code-block:: bash

   git clone --branch en/neopax-main-utilities \
     https://github.com/uwplasma/vmex.git "$HOME/VMEX"
   git clone --branch en/neopax-main-utilities \
     https://github.com/uwplasma/NTX.git "$HOME/NTX"
   git clone --branch en/v0_main \
     https://github.com/uwplasma/NEOPAX.git "$HOME/NEOPAX"

The dependency branches contain the NEOPAX utility ports on top of upstream
main. The GPU installation checks used VMEX ``32d96d34``, NTX ``4f7aa88`` and
NEOPAX ``dc3820b14``. Branch tips can advance; record the actual revisions
alongside benchmark results. Once the utility PRs are merged, use dependency
``main`` revisions containing those utilities. Existing checkouts can be
reused after checking their branches and local changes.

Create and install into jax_env
------------------------------

Create a fresh environment, or activate the existing one when it already
contains this setup:

.. code-block:: bash

   python3.12 -m venv "$HOME/jax_env"
   source "$HOME/jax_env/bin/activate"
   python -m pip install --upgrade pip setuptools wheel

Install the GPU build and application dependencies together so the resolver
keeps the explicit JAX pin:

.. code-block:: bash

   python -m pip install \
     "jax[cuda12]==0.10.0" "scipy>=1.16" \
     "booz_xform_jax>=0.5.1" "solvax>=0.35.1" \
     matplotlib equinox optimistix lineax diffrax \
     interpax orthax h5py netCDF4 toml tomli \
     -e "$HOME/VMEX" -e "$HOME/NTX"
   python -m pip install --no-deps -e "$HOME/NEOPAX"
   python -m pip check

``--no-deps`` on the NEOPAX installation preserves the selected editable
dependency checkouts: NEOPAX's package metadata names VMEX and NTX through
direct Git URLs at ``main``, which could otherwise replace the utility
branches. Its other requirements were installed explicitly above.
The source stays in ``$HOME/VMEX``, ``$HOME/NTX`` and ``$HOME/NEOPAX``;
packages and editable-install metadata live in ``$HOME/jax_env``.

Verify GPU compilation and interfaces
------------------------------------

Run this small check from a new shell after installation. Explicit environment
cleanup avoids importing another checkout through ``PYTHONPATH`` or overriding
the pip CUDA libraries through ``LD_LIBRARY_PATH``. The command also enables
64-bit arithmetic, disables the persistent compilation cache and uses the
default XLA compiler settings.

.. code-block:: bash

   cd "$HOME/NEOPAX" || exit 1
   env -u PYTHONPATH -u LD_LIBRARY_PATH -u JAX_PLATFORM_NAME -u XLA_FLAGS \
     -u JAX_COMPILATION_CACHE_DIR \
     JAX_PLATFORMS=cuda,cpu \
     JAX_ENABLE_X64=true \
     JAX_ENABLE_COMPILATION_CACHE=0 \
     "$HOME/jax_env/bin/python" - <<'PY'
   import sys
   import jax
   import jaxlib
   import jax.numpy as jnp
   import vmex
   import ntx
   import NEOPAX
   from vmex.core import implicit

   print("Python:", sys.executable)
   print("JAX / jaxlib:", jax.__version__, jaxlib.__version__)
   print("Devices:", jax.devices())
   print("VMEX:", vmex.__file__)
   print("NTX:", ntx.__file__)
   print("NEOPAX:", NEOPAX.__file__)
   assert jax.__version__ == jaxlib.__version__ == "0.10.0"
   assert jax.default_backend() == "gpu"
   assert jax.config.x64_enabled
   assert callable(implicit.implicit_state_tangent_raw_block)
   assert callable(implicit.implicit_state_pullback_multi_rhs_raw_block_transpose)
   assert callable(vmex.j_invariant_qi_maxj_residual_from_boozer)
   assert callable(ntx.solve_prepared_coefficient_vector_derivative_vjp)
   assert callable(ntx.pullback_neopax_scan_coefficient_blocks_from_primal_record)

   x = jnp.arange(16, dtype=jnp.float64).reshape(4, 4) / 16
   value, gradient = jax.jit(jax.value_and_grad(
       lambda matrix: jnp.sum((matrix @ matrix) ** 2)
   ))(x)
   gradient.block_until_ready()
   assert bool(jnp.isfinite(value))
   assert bool(jnp.all(jnp.isfinite(gradient)))
   print("PASS: GPU compilation, AD, and required interfaces.")
   PY

The printed paths should point to the three intended source directories.
``JAX_PLATFORMS=cuda,cpu`` keeps CUDA first and makes the CPU backend available
for VMEX host callbacks. The default-backend assertion verifies GPU execution.
Passing this check confirms installation and the small
compiled operation; benchmark memory requirements must be assessed separately.

Native monoenergetic interpolation
---------------------------------

The generic monoenergetic database path uses a native JAX local C1 cubic
interpolator. It preserves the previous cubic slope and extrapolation rules,
reads a local 4x4 stencil and shares its interval calculations across D11,
D13 and D33. Scalar radial queries also evaluate only their active radial
closure, sharing evaluation across its three or four surfaces. An exact
arithmetic cell lookup accelerates sufficiently uniform grids, with a search
fallback for other grids. No special XLA guard-padding flag is required.
Interpax remains a dependency for other interpolation paths.

GPU component checks using saved QI coefficients, JAX 0.10.0 and default XLA
settings reduced initial-root coordinate-pullback peak JAX allocation from
1,552 MiB to about 288 MiB. Radial-surface grouping contributes to this saving
and also benefits an Interpax control. Native interpolation improves query
timing. The coordinate pullback uses three reusable stages for flux-query
preparation, local stencil derivatives and objective contraction, reducing
the preceding 400 MiB peak by another 28%. Numerical inputs of the same
shapes and dtypes reuse compilation. Values and derivatives matched
within the comparison tolerance. These component measurements do not
establish the memory use or performance of the full transport benchmark.
See the `native interpolation comparison
<https://github.com/uwplasma/NEOPAX/blob/en/v0_main/docs/benchmarks/native_database_interpolation_2026-11-10.md>`_
for the reproducible check and its scope.

Reuse and record the environment
-------------------------------

Activate the same environment in later shells with
``source "$HOME/jax_env/bin/activate"``, or call
``"$HOME/jax_env/bin/python"`` explicitly. Editable installs make a separate
``PYTHONPATH`` unnecessary. Reuse the launch environment shown above when
running a benchmark, replacing the Python here-document with its script and
arguments.

Record the resolved package versions and source commits before a run:

.. code-block:: bash

   mkdir -p "$HOME/NEOPAX/outputs/environment"
   "$HOME/jax_env/bin/python" -m pip freeze \
     > "$HOME/NEOPAX/outputs/environment/pip-freeze.txt"
   git -C "$HOME/VMEX" rev-parse HEAD
   git -C "$HOME/NTX" rev-parse HEAD
   git -C "$HOME/NEOPAX" rev-parse HEAD

The package listing records transitive dependencies as well as JAX. Avoid
an unpinned JAX upgrade when reproducing this vGPU setup; recheck GPU
initialization and dependency compatibility after any environment update.
