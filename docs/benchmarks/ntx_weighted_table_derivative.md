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
direction check and rejection of missing/nonfinite FD inputs. Production
methods are unchanged.

## Remote result: all weighted ZBS nodes

The check completed at commit `d8d12ea` with JAX 0.10.0 on CUDA, using the
same saved AD and exact FD baseline. Output directory:
`outputs/ntss1d_zbs_support_chain_UfR8XG/ntx_table_ZBS_PXKHjE`.
It evaluated 617 active coefficient cases and completed all 13 objective
comparisons. Wall time was 7:52.30; maximum host RSS was 2,063,004 KiB.
That is a host-memory measurement, not a GPU-memory measurement.

| Objective | Recorded table VJP vs independent JVP | Independent JVP vs saved table FD |
| --- | ---: | ---: |
| softmax_Er | `2.316983e-10` | `6.047803e-06` |
| net_total_power_volume_average_mw_m3 | `4.425403e-09` | `3.399916e-05` |
| Er_transition_left | `3.709823e-09` | `1.023147e-05` |
| Er_transition_right | `3.171187e-09` | `1.154096e-05` |
| Er2_volume_average | `6.938096e-10` | `1.189970e-04` |
| Er_volume_average | `7.209422e-11` | `1.200586e-05` |
| electron_temperature_volume_average_keV | `2.907964e-10` | `7.053826e-06` |
| total_pressure_volume_average | `6.534272e-09` | `5.403356e-05` |
| alpha_power_volume_average_mw_m3 | `4.567144e-09` | `3.462381e-05` |
| bootstrap_current_softmax_abs_scaled | `3.973415e-09` | `3.606363e-05` |
| Er_transition_rho | `6.094683e-10` | `4.405473e-07` |
| Er_transition_location_moment | `6.094683e-10` | `4.405473e-07` |
| Er_transition_strength | `1.234056e-10` | `3.081355e-06` |

These are relative differences of the table contribution, not the full
parameter gradient, and not percentages. The first column uses the recorded
table VJP as denominator; the second uses saved table FD. The largest first
column difference is `6.534272e-09`. This is agreement of the objective-weighted
directional contractions after including every active case; it does not test
each node's VJP independently or cover other parameter directions.

For bootstrap, the independent table derivative is `-3.893936979605034`,
versus recorded VJP `-3.893936995077262` and saved FD `-3.893796555156176`.
The AD implementations differ by `1.547223e-08`, whereas the JVP/FD gap is
`-1.404244e-04`. The transport/support contraction had already matched the
same FD response. The large discrepancy therefore remains between the
parameter-to-table derivative and its FD secant, rather than between the
recorded pullback and independent differentiation of the coefficient solve.

This strongly supports the recorded rule for the tested direction. It does
not yet distinguish coefficient-solve roundoff/cancellation, finite-step
effects or differences in the upstream endpoint input directions. The saved
field breakdown and database directions can locate which coefficients and
nodes dominate that gap, without another GPU or transport run.

The full numerical comparisons and remote log remain in the output directory
above as `table_comparison.csv` and `run.log`.
Attachment SHA256:
`9754923f7e19ea6845424eb2492781d593d0ca1d3d053ab273cf0df083afb485`.

## Fields responsible for the remaining FD gap

The saved-array ranking gives the following signed weighted differences
(`independent table JVP - saved table FD`). These are contribution differences,
not relative errors:

| Objective | D11_log | D13 | D33 | Total over all database fields |
| --- | ---: | ---: | ---: | ---: |
| softmax_Er | `-1.646102e-04` | `0` | `0` | `-1.645743e-04` |
| Er2_volume_average | `6.946338e-03` | `0` | `0` | `6.946135e-03` |
| bootstrap_current_softmax_abs_scaled | `5.140155e-07` | `-1.471129e-04` | `6.176978e-06` | `-1.404244e-04` |
| Er_transition_rho | `-9.534860e-09` | `0` | `0` | `-9.497727e-09` |

The zero D13/D33 entries for Er rows follow the particle-flux/table support
contract and their absence from the nonzero field ranking. Small field sums
can conceal cancellation between larger node contributions; for example the
root-position D11_log nodes have contributions near `4e-08` of both signs.

The precision check targets six nodes selected from those measured gaps:

| Radius/nu/field indices | Contribution motivating the point |
| --- | --- |
| `3:4:3` | largest negative softmax-Er D11_log node |
| `3:5:7` | largest positive softmax-Er D11_log node |
| `5:6:7` | largest Er2 D11_log node |
| `4:2:6` | largest bootstrap D13 node; also affects root position |
| `5:4:2` | second largest bootstrap D13 node |
| `4:4:3` | largest root-position D11_log node |

Use `--precision-from-table-audit PATH --points 3:4:3,3:5:7,5:6:7,4:2:6,5:4:2,4:4:3`.
The reference must be the completed `ntx_table_ZBS_PXKHjE` directory. Its
objective/parameter labels, active points and database bytes are checked
before GPU initialization. The check reuses its saved coefficient and
normalized-table JVPs; it performs no new coefficient VJP/JVP.

The raw coefficient kernel receives all numerical surface fields from the
weighted audit, including the extra native-surface directions, and Es.
The same step multipliers `1,3,10,30` apply to every point and coefficient.
The drds dependence in D11_log/D13 normalization stays live. `precision.csv`
records both raw and normalized FD derivatives, reference JVPs, baseline
primal offsets, endpoint values and D11-floor flags.

This is FD along linearized numerical NTX inputs, not a rerun of the complete
geometry-to-transport endpoints. Decreasing error as the step grows supports
small-step coefficient cancellation. Accurate local FD with a persistent
saved-endpoint gap instead requires examining the upstream inputs or scan
evaluation context. A D11 floor switch is reported and must be assessed.

Eight local tests pass without importing real JAX or physics. The new checks
reject mismatched references and run the precision control flow against an
independent analytic coefficient model with a nonzero extra-surface derivative
and live normalization. This validates diagnostic wiring; numerical GPU
validation of the targeted precision sweep is pending.
