CLI Reference
=============

NEOPAX can be run from the command line with either:

.. code-block:: console

   NEOPAX my_case.toml

or:

.. code-block:: console

   python -m NEOPAX my_case.toml

Common options include:

- ``--mode``
- ``--device``
- ``--vmec-file``
- ``--boozer-file``
- ``--n-radial``
- ``--n-x``
- ``--backend``
- ``--dt``
- ``--t-final``
- repeated ``--set section.key=value`` overrides

The CLI is a thin configuration layer.  It should not be treated as a separate
physics runtime from the Python API.

For examples and details, see :doc:`methods_of_use`.

