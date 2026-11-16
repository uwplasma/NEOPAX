# Fresh-equilibrium normalization trace, 2026-11-16

The run in `outputs/normalization_trace_fresh_state_20261116` reproduces
the variation before any NTX coefficient, root, transport or derivative solve.
The supplied log does not print Git HEAD; this trace mode was introduced
at `821c5b8`.

| Comparison | Differing captured leaves | First differing numerical boundary |
| --- | ---: | --- |
| baseline 1 vs 2 | 0 | none |
| baseline 1 vs 3 | 63 | `03_boozer_0_inputs` |
| baseline 2 vs 3 | 63 | `03_boozer_0_inputs` |

The second and third comparisons contain the same 63 differing leaves and
maximum differences. No differences are reported for the returned equilibrium
state, the state passed to geometry construction, or the captured volume/vp.
Native scan surfaces also match. Variation is first visible in the conversion
of matching returned states to Boozer input tables, before the transform.

Representative input-table differences are:

| Field | Maximum absolute difference | Index |
| --- | ---: | --- |
| bmnc | `1.776356839e-15` | (3, 0) |
| bsubumnc | `8.881784197e-16` | (131, 14) |
| bsubvmnc | `2.842170943e-14` | (16, 0) |
| lmns | `2.775557562e-17` | (6, 1) |
| rmnc | `2.467162277e12` | (0, 105) |
| zmns | `6.167905692e11` | (0, 104) |

Later differences reproduce the earlier normalization gaps: a_b
`2.220446049e-16`, B00 `1.776356839e-15`, dr_tildeds `4.440892099e-16`.
These are absolute differences. In particular, the large axis-row R/Z
differences must not be described as ordinary roundoff without checking
the conversion.

## Concrete axis defect found in the adapter

NEOPAX's batched current-VMEX adapter divided projected odd-mode axis
coefficients by `max(sqrt(s), 1e-30)`. At s=0, Fourier projection noise
of order `1e-17` is amplified by `1e30` before half-mesh interpolation.
The current VMEX helper `vmex.core.boozer_tables.boozer_input_tables`
explicitly uses a denominator of one at the axis. NEOPAX now follows that
convention. Positive-radius denominators and even-mode interpolation are
unchanged.

Three NumPy regressions exercise the actual nested production interpolation
kernel, extracted without importing JAX: bounded noisy axis coefficients,
unchanged nonaxis/even interpolation, and finite linear coefficient sensitivity.
This is a concrete adapter correction, not a change to equilibrium solving,
NTX solving or transport differentiation.

It does **not** establish the origin of the smaller projection differences
or demonstrate that all normalization/coefficient discrepancies are fixed.
The enhanced trace can capture real-space preparation and exact Fourier
projection operands/results. Identical operands with differing contraction
outputs would isolate a different problem from differing prepared field arrays.
There is no validated compiler/backend diagnosis yet.

The old trace compares numerical values with matching NaNs permitted; it
does not separately certify signed-zero or NaN-payload byte identity.
The enhanced comparator records byte equality and the first byte-different
stage separately so those distinctions are explicit.

Saved: [all 126 printed mismatch rows](normalization_trace_fresh_state_2026-11-16_differences.csv),
[three comparison summaries](normalization_trace_fresh_state_2026-11-16_summary.csv),
[complete supplied log](normalization_trace_fresh_state_2026-11-16_remote_log.txt).
The last comparison's first differing stage is inferred from its identical
printed mismatch rows. Original attachment SHA256: `adc84186bbc231787b0f63edb73f9d829f4c8dbbd7b4b0db773f3a9dcbbefec9`.
Only parsing and lightweight NumPy/mock checks ran locally.
