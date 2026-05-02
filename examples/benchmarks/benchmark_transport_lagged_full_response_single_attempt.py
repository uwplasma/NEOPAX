"""Run only the lagged-response benchmark for exactly one solver attempt
using the full radial exact-runtime NTX response (no coarse anchors).

This is a thin wrapper around ``benchmark_transport_rhs_modes.py`` that:

- forces ``rhs_mode = lagged_response``
- forces ``transport_solver.max_steps = 1``
- clears ``stop_after_accepted_steps``
- forces ``ntx_exact_response_anchor_count = 0``

Examples:

    python examples/benchmarks/benchmark_transport_lagged_full_response_single_attempt.py

    python examples/benchmarks/benchmark_transport_lagged_full_response_single_attempt.py \
        --backend radau \
        --device gpu
"""

from __future__ import annotations

import sys

from benchmark_transport_rhs_modes import main


if __name__ == "__main__":
    if "--rhs-modes" not in sys.argv:
        sys.argv.extend(["--rhs-modes", "lagged_response"])
    if "--single-attempt" not in sys.argv:
        sys.argv.append("--single-attempt")
    if "--ntx-response-anchor-count" not in sys.argv and "--ntx-response-anchor-counts" not in sys.argv:
        sys.argv.extend(["--ntx-response-anchor-count", "0"])
    main()
