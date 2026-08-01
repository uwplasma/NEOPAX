#!/usr/bin/env python
"""Geometry QI + max-Er optimization with targeted initial-Er transition."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.optimization import optimize_geometry_qi_max_er_initial_root as base


# For the 51-point root grid, indices 25/26 straddle rho ~= 0.5.
ER_TRANSITION_LEFT_INDEX = 25
ER_TRANSITION_RIGHT_INDEX = 26
ER_TRANSITION_LEFT_TARGET = 26.0
ER_TRANSITION_RIGHT_TARGET = -10.0
ER_TRANSITION_LEFT_WEIGHT = 1.0
ER_TRANSITION_RIGHT_WEIGHT = 1.0

base.OUT_DIR = base.ROOT / "outputs" / "geometry_qi_max_er_transition_initial_root_optimization"
base.ROOT_OPTIONS = {
    "Er_transition_left_index": ER_TRANSITION_LEFT_INDEX,
    "Er_transition_right_index": ER_TRANSITION_RIGHT_INDEX,
}
base.terms = [
    *base.terms,
    (base.opt.transport.Er_transition_left, ER_TRANSITION_LEFT_TARGET, ER_TRANSITION_LEFT_WEIGHT),
    (base.opt.transport.Er_transition_right, ER_TRANSITION_RIGHT_TARGET, ER_TRANSITION_RIGHT_WEIGHT),
]


if __name__ == "__main__":
    raise SystemExit(base.main())
