#!/usr/bin/env python
"""Finite-beta-seed variant of the geometry-only QI + max-J optimization.

This intentionally reuses the existing QI/max-J-only example unchanged except
for its VMEC input and output directory.
"""

from __future__ import annotations

import optimize_geometry_qi_only as _base


_base.SEED_INPUT = _base.ROOT / "examples" / "inputs" / "input.QI_nfp2_initial_finitebeta"
_base.OUT_DIR = _base.ROOT / "outputs" / "geometry_qi_only_finitebeta_optimization"


if __name__ == "__main__":
    raise SystemExit(_base.main())
