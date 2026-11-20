# Weighted derivative check of the saved NTX table

The NTSS1D ZBS FD16 run `ntss1d_zbs_support_chain_UfR8XG` has an exact
AD/FD baseline. Its full support-adjoint contraction agrees with FD within
`6.782123e-08` relative for all objectives. Reconstructing the final AD matrix
from its saved compact tangents and bars differs by at most `5.684342e-14`
in absolute value. These are different checks of different boundaries.

The next saved-array split gives these signed differences (`AD - FD`) in
the derivative contributions:

| Objective | Direct geometry gap | Folded table gap |
| --- | ---: | ---: |
| softmax_Er | `-6.623300e-07` | `-1.645806e-04` |
| Er_transition_left | `-1.107000e-07` | `-1.482469e-04` |
| Er_transition_right | `-9.997000e-08` | `-1.777844e-04` |
| bootstrap_current_softmax_abs_scaled | `1.490436e-07` | `-1.404399e-04` |
| Er_transition_rho | `3.315250e-10` | `-9.484590e-09` |
| electron_temperature_volume_average_keV | `-8.888009e-08` | `-5.294482e-09` |
| total_pressure_volume_average | `-5.439376e-07` | `-5.664050e-09` |

These are absolute signed contribution gaps, not relative errors. The table
contribution dominates the bootstrap and principal Er discrepancies, while
temperature and pressure have larger direct-geometry discrepancies. The
table contribution includes the coefficient solve, its normalization and
the surface/channel mapping. It has not yet been identified as an erroneous
custom rule; FD numerical sensitivity at that boundary is also possible.

`check_captured_ntx_directional_derivative.py --audit-table` compares the saved
production table pullback against a fresh raw-solver JVP at every table node
with a nonzero objective cotangent. It differentiates the ordinary
`solve_prepared_coefficient_vector`, rather than rerunning the recorded NTX
coefficient VJP. Both paths include the production table normalization.
Generic and radial NTSS1D runtime tables keep their configured mode.

With `--compare-saved-fd`, the same objective bars are also contracted with
the saved FD database direction. All three contractions therefore use the
same AD point and objective weights. Baseline mismatch or missing endpoint
support aborts before the per-node solves. Both `ad_snapshots` and
`ad/snapshots` layouts are supported. No equilibrium, root search or transport
integration is performed; this is a GPU coefficient-derivative calculation.

The key outputs in `table_comparison.csv` are:

- `captured_table_vjp`: the production contribution already recorded by AD.
- `raw_solver_table_jvp`: independently differentiated raw coefficient solves.
- `saved_table_fd`: the contribution inferred from existing FD table endpoints.
- `jvp_minus_recorded_table_vjp` and `jvp_minus_saved_table_fd`: the two gaps.

Agreement of the first two with a remaining FD gap supports an FD numerical
explanation; it does not establish convergence of the FD estimate. A gap
between the first two requires inspection of their primal offsets and the
reported radius/field contributions before attributing it to a derivative bug.
`table_fd_fields.csv` partitions the JVP/FD gap by database field; other CSVs
retain per-node raw coefficients, primal offsets and component/radius splits.

Six NumPy-only preparation tests pass, including an independent polynomial
direction check and rejection of missing/nonfinite FD inputs. The weighted
GPU result for this NTSS1D case is pending. Production methods are unchanged.
