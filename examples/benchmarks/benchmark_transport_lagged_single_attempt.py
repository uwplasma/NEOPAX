"""Run only the lagged-response benchmark for exactly one solver attempt.

This is a thin wrapper around ``benchmark_transport_rhs_modes.py`` that:

- forces ``rhs_mode = lagged_response``
- forces ``transport_solver.max_steps = 1``
- clears ``stop_after_accepted_steps``

Examples:

    python examples/benchmarks/benchmark_transport_lagged_single_attempt.py

    python examples/benchmarks/benchmark_transport_lagged_single_attempt.py \
        --backend radau \
        --device gpu \
        --ntx-response-anchor-counts 7
"""

from __future__ import annotations

import sys

from benchmark_transport_rhs_modes import main


if __name__ == "__main__":
    if "--rhs-modes" not in sys.argv:
        sys.argv.extend(["--rhs-modes", "lagged_response"])
    if "--single-attempt" not in sys.argv:
        sys.argv.append("--single-attempt")
    main()
