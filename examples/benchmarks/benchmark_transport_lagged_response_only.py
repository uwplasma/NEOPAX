"""Run only the lagged-response transport benchmark mode.

This is a thin convenience wrapper around ``benchmark_transport_rhs_modes.py``
for quickly testing the D1 path without first running ``black_box`` or
``lagged_linear_state``.

Examples:

    python examples/benchmarks/benchmark_transport_lagged_response_only.py

    python examples/benchmarks/benchmark_transport_lagged_response_only.py \
        --backend radau \
        --device gpu \
        --ntx-radial-batch-size 4
"""

from __future__ import annotations

import sys

from benchmark_transport_rhs_modes import main


if __name__ == "__main__":
    if "--rhs-modes" not in sys.argv:
        sys.argv.extend(["--rhs-modes", "lagged_response"])
    main()
