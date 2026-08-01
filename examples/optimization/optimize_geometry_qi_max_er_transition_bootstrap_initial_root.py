#!/usr/bin/env python
"""Geometry QI + max-Er + transition optimization with bootstrap-current penalty."""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp

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

# The bootstrap objective is scaled so 0.1 corresponds to 10 kA/m^2.
BOOTSTRAP_LIMIT_SCALED = 0.1
BOOTSTRAP_WEIGHT = 1.0


def bootstrap_penalty_value(bootstrap_softmax_abs_scaled):
    return jnp.maximum(bootstrap_softmax_abs_scaled - BOOTSTRAP_LIMIT_SCALED, 0.0)


bootstrap_penalty = base.opt.transformed_transport_objective(
    base.opt.transport.bootstrap_current_softmax_abs_scaled,
    bootstrap_penalty_value,
    label="bootstrap_current_penalty",
)

base.OUT_DIR = (
    base.ROOT
    / "outputs"
    / "geometry_qi_max_er_transition_bootstrap_initial_root_optimization"
)
base.ROOT_OPTIONS = {
    "Er_transition_left_index": ER_TRANSITION_LEFT_INDEX,
    "Er_transition_right_index": ER_TRANSITION_RIGHT_INDEX,
}
base.terms = [
    *base.terms,
    (base.opt.transport.Er_transition_left, ER_TRANSITION_LEFT_TARGET, ER_TRANSITION_LEFT_WEIGHT),
    (base.opt.transport.Er_transition_right, ER_TRANSITION_RIGHT_TARGET, ER_TRANSITION_RIGHT_WEIGHT),
    (bootstrap_penalty, 0.0, BOOTSTRAP_WEIGHT),
]


if __name__ == "__main__":
    raise SystemExit(base.main())
