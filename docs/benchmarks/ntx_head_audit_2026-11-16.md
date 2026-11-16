# NTX reduced-system conditioning and solve audit

The remote `6225cb9` run completes the reduced-system audit, after the
flat-input compilation-interface correction. All ten saved D13 replays
remain exact, and the extra audit's production-head D13 values also match
the original scan values exactly. The reported head pivot arrays do not
change between the two field inputs; eliminated-tail pivots are not recorded.

The compiled argument/temporary/output buffer estimate is `269461180` bytes.
This is not a measurement of total GPU or compiler peak memory. The run
uses the saved surfaces and builds no equilibrium, roots or transport solve.

## Conditioning and reduced-system residuals

| Radius index | Field case | Head condition | Scaled head condition |
| --- | --- | ---: | ---: |
| 5 | `ad` | `3.246927090e+09` | `8.113092652e+08` |
| 5 | `fd2` | `3.246926762e+09` | `8.113092030e+08` |
| 6 | `ad` | `1.290596541e+12` | `2.940226830e+11` |
| 6 | `fd2` | `1.289821827e+12` | `2.938462042e+11` |

The corresponding parallel-source RHS-relative residuals are:

| Radius index | Field case | Existing block head | Pivoted head | Scaled pivoted head |
| --- | --- | ---: | ---: | ---: |
| 5 | `ad` | `2.468405039e-03` | `2.705135944e-08` | `3.967458331e-09` |
| 5 | `fd2` | `2.374051260e-03` | `3.069272968e-08` | `2.944371029e-09` |
| 6 | `ad` | `1.188526486e-01` | `6.438222215e-06` | `1.928845619e-06` |
| 6 | `fd2` | `1.029430536e-01` | `5.908668893e-06` | `2.107230032e-06` |

At the outer point, pivoting/scaling substantially improves the residual
against this represented reduced matrix. However, raw D11 remains negative
for all three methods, and the Onsager discrepancy remains around `2.53e1`.
The alternate head solves are therefore not a validated physical/numerical
fix. Their matrices are still strongly ill-conditioned after scaling.

The field sensitivity also remains:

| Radius index | Method | Absolute D13 difference | Relative D13 difference to AD-field value |
| --- | --- | ---: | ---: |
| 5 | `production_head` | `3.664789592e-06` | `6.375041037e-05` |
| 5 | `pivoted_head` | `7.413740515e-07` | `1.289742618e-05` |
| 5 | `scaled_pivoted_head` | `7.417502216e-07` | `1.290397031e-05` |
| 6 | `production_head` | `9.517854824e-03` | `3.279306687e-03` |
| 6 | `pivoted_head` | `2.070598784e-03` | `7.143935766e-04` |
| 6 | `scaled_pivoted_head` | `2.070402343e-03` | `7.143246011e-04` |

These are sensitivities of the computed coefficient to the two saved field
values, not transport AD/FD gradient errors. The outer-point gap falls from
about `9.52e-3` to `2.07e-3`, but it does not disappear.

## Scope of the diagnosis

The represented three-block system has a condition number around `1.29e12`
at the outer point, so the earlier small LU-factor residual was not an
accuracy certificate. Pivoting/scaling improves the represented-system
solve, but does not establish accuracy of the original full DKE operator.
The last reduced diagonal is reconstructed from retained LU factors and
therefore does not independently validate tail elimination. Small backward
error, a small RHS-relative residual and a physically accurate coefficient
are different claims; full backward-error values are in the remote CSV,
but were not printed in this log.

The subsequent NumPy-only readout localizes the amplification:

| Radius index | Theta derivative mismatch | Zeta derivative mismatch | Original row 0 change | Original row 1 change | Tail-reduced row 2 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 | `4.927e-14` | `1.347e-14` | `7.372e-18` | `2.295e-18` | `6.539e-8` |
| 6 | `4.012e-14` | `1.360e-14` | `9.954e-18` | `3.100e-18` | `4.673e-8` |

These are relative norms, not transport gradient errors. No input field
harmonics exceed the angular Nyquist limits; weighted drift mean defects
are below `1e-16`. This finds no unresolved input B harmonics or inconsistency
between the saved spectral and analytic B derivatives. It does not certify
resolution of the kinetic solution or all variable-coefficient operators.

The outer head's primary/parallel normwise backward errors improve from
about `1e-12` to `2e-17` under scaling and pivoting. Yet its Onsager discrepancy
remains around `2.53e1`. Solving the represented head more accurately does
not repair the already-reduced equations. The amplification is confined to
the represented tail contribution, but genuine conditioning versus numerical
error in elimination still needs an independent comparison.

The diagnostic's `--audit-tail` comparison reuses these saved prepared arrays,
eliminates adjacent high Legendre modes in pairs, and checks every original
equation for both sources after recovering all modes. The angular grid,
65 Legendre modes, NTX block generator, gauge and sources are unchanged.
No production patch is made. This GPU comparison has not run yet; nine
local NumPy/mock tests pass, including agreement with independent full
small-system solves and detection of corruption confined to the eliminated
tail. No local JAX or physical solve ran.

Saved: [four conditioning rows](ntx_head_audit_2026-11-16_conditions.csv),
[all twelve solve comparisons](ntx_head_audit_2026-11-16_methods.csv),
[coefficient sensitivities](ntx_head_audit_2026-11-16_field_sensitivity.csv),
[saved-array checks](ntx_head_audit_2026-11-16_saved_array_checks.csv),
[outer-point backward errors](ntx_head_audit_2026-11-16_backward_errors.csv),
[complete supplied log](ntx_head_audit_2026-11-16_remote_log.txt).
The `_as_printed` CSV fields retain the log's limited printed precision;
the exact full diagnostics remain in the remote `head_comparison.csv`.
Raw attachment SHA256: `88b76a988496e802c0b45cd59c7510ad2abc0d6ef8d63bfccbda58c4b9a2f1da`.
Only standard-library log parsing ran locally.
