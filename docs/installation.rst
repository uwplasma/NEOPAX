Installation
============

NEOPAX is normally used from a local checkout.

.. code-block:: console

   git clone https://github.com/uwplasma/NEOPAX.git
   cd NEOPAX
   pip install -e .

For GPU runs, install NEOPAX inside an environment with a JAX build matching
the available CUDA stack.  The CPU path is useful for small tests and
documentation examples, but production NTX and reverse-AD benchmarks are
intended for GPU execution.

After installation, check that the package imports:

.. code-block:: console

   python -c "import NEOPAX; print(NEOPAX.__name__)"

