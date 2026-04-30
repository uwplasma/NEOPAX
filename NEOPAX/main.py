"""Compatibility shim for the legacy ``NEOPAX.main`` module path.

The orchestration implementation now lives in :mod:`NEOPAX._orchestrator`.
This module re-exports that surface so older imports keep working while the
internal architecture uses the clearer module name.
"""

from ._orchestrator import *  # noqa: F401,F403
