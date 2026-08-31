#!/usr/bin/env python
"""Finite-beta variant of the initial-Er-root/bootstrap geometry optimization.

This is intentionally a thin wrapper around
``optimize_geometry_qi_max_er_transition_bootstrap_initial_root.py``: it
inherits the exact same geometry, Er-root, bootstrap-current, reporting, and
staged reverse-AD setup, then changes only the seed, output directory, and
adds VMEX's traceable Mercier softmax objective.

The Mercier objective uses VMEX's defaults: margin=0, softplus
smoothing=1e-6, and softmax temperature=1e-3.  It penalizes an unstable
(negative-DMerc) interior surface, so its least-squares target is zero.
"""

from __future__ import annotations

from pathlib import Path

import optimize_geometry_qi_max_er_transition_bootstrap_initial_root as base


# Keep the same NFP=2 boundary and the same two-stage 24/51 radial solve as
# the vacuum seed.  This deck differs only by its finite Akima pressure
# profile.
SEED_INPUT = base.ROOT / "examples" / "inputs" / "input.QI_nfp2_initial_finitebeta"
OUT_DIR = base.ROOT / "outputs" / "geometry_qi_max_er_transition_bootstrap_initial_root_dmerc_optimization"

# These are VMEX's documented/default smooth-Mercier settings.  The public
# NEOPAX objective deliberately exposes that fixed default kernel, rather
# than a second, differently-parameterized DMerc definition.
DMERC_TARGET = 0.0
DMERC_WEIGHT = 0.05
DMERC_MARGIN = 0.0
DMERC_SOFTPLUS_SMOOTHING = 1.0e-6
DMERC_SOFTMAX_TEMPERATURE = 1.0e-3


def main() -> int:
    # ``base.main`` reads module globals, so configure the otherwise identical
    # campaign before executing it.  The original vacuum script is untouched.
    base.SEED_INPUT = Path(SEED_INPUT)
    base.OUT_DIR = Path(OUT_DIR)
    base.terms = [
        *base.terms,
        (base.opt.geometry.vmec_dmerc_stability_softmax, DMERC_TARGET, DMERC_WEIGHT),
    ]
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
