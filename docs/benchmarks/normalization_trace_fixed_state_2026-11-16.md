# Fixed-state normalization trace, 2026-11-16

Remote NEOPAX commit `821c5b8`; output directory
`outputs/normalization_trace_fixed_state_20261116`.

Three independent GPU processes reused the `implicit` fields in
`outputs/vmex_primal_consistency/states.npz`. All three pairwise comparisons
report **zero differing captured leaves**. For baseline 2 and baseline 3,
the console explicitly reports `first_changed_stage=none` against baseline 1.

The captured equilibrium state, volume inputs, Boozer inputs/constants/grids,
Boozer outputs, transport geometry, scan surfaces/channels and normalized
electric-field inputs match at every recorded boundary. Input/config/state,
package-version and source-hash checks also pass. The checker launches all
three repeats with the same baseline-building configuration.

This run performed no equilibrium, NTX coefficient, ambipolar root or transport
solve, and no AD or FD differentiation. It confirms that the normalization
variation was not reproduced in this fixed-state control. It does not prove
that the fresh VMEX solve is the cause: the saved state may differ from the
original fresh state, the compilation/call history changes when the solve is
skipped, and intermittent variation is not excluded by three matching runs.

The next existing mode omits `--reuse-states`. It performs three independent
fresh equilibrium solves, captures the same boundaries, and still stops
before NTX coefficient construction. A first difference in `equilibrium`
would identify a different fresh state before normalization. A later first
difference would instead locate the observed variation downstream of matching
states. All repeats matching would leave the earlier intermittent effect
unreproduced rather than establish a fix.

[Pairwise results](normalization_trace_fixed_state_2026-11-16.csv).
[Diagnostic and fresh-state command](normalization_input_reproducibility.md).
These scalar results were supplied in conversation. No local physical solve
ran and no production numerical code changed.
