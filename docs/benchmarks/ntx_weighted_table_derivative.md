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
and live normalization. This validates diagnostic wiring separately from the
numerical GPU result below.

## Targeted precision sweep result

Run `ntx_precision_ZBS_vDo4al` completed at diagnostic commit `3874477`.
It used JAX 0.10.0 and the same geometry/NTX source hashes as the weighted
audit. All 72 rows completed in 43.91 s. Maximum host RSS was 1,970,512 KiB;
GPU peak memory was not measured.

Relative differences from the saved independent JVP at the principal fields:

| Node | Field | h | 3h | 10h | 30h |
| --- | --- | ---: | ---: | ---: | ---: |
| `3:4:3` | D11_log | `2.441525e-07` | `1.215899e-07` | `4.656897e-08` | `1.642096e-08` |
| `3:5:7` | D11_log | `1.433149e-07` | `1.131002e-07` | `7.623466e-09` | `4.161230e-10` |
| `5:6:7` | D11_log | `3.194172e-08` | `8.845846e-09` | `8.625480e-08` | `7.986945e-07` |
| `4:2:6` | D13 | `1.019053e-03` | `2.830533e-03` | `4.013351e-04` | `1.613987e-04` |
| `5:4:2` | D13 | `6.976994e-05` | `1.093623e-04` | `3.454970e-06` | `5.133718e-06` |
| `4:4:3` | D11_log | `1.051595e-06` | `2.258159e-07` | `7.830416e-08` | `5.822391e-08` |

These are coefficient-level relative differences, not full objective-gradient
errors. The printed log provides seven significant digits for these errors;
the complete endpoint values, signed derivatives and floor flags are in the
remote `precision.csv`.

The numerical-input FD is demonstrably step-sensitive at the nodes responsible
for the saved endpoint gaps. Larger steps help several entries, while the
Er2-dominant `5:6:7` D11_log entry is best at 3h and worsens approximately with
h squared at larger steps. The low-collisionality D13 node `4:2:6` remains
inaccurate even at 30h and varies non-monotonically. The patterns are consistent
with cancellation/precision limits combined with finite-step bias, rather than
a uniform recommendation to multiply every full-transport FD step by 30.
The log alone does not establish the D11-floor flags or a precise numerical
error budget for the saved full geometry FD endpoints.

The next readout contracts each local FD error with its actual objective bar
and compares it with the saved endpoint error at the same six nodes. This
uses existing CSV/NPZ files only. It is needed before asserting that local
coefficient precision explains the entire observed gradient gap; upstream
endpoint differences and unselected-node contributions remain distinguishable
possibilities.

Full result directory:
`outputs/ntss1d_zbs_support_chain_UfR8XG/ntx_precision_ZBS_vDo4al`.
Attachment SHA256:
`767ae14ad533877c20ad5a66d41e78ef137bdb91a84c385df9745a918b6aa9ce`.

## Weighted precision budget: local FD does not reproduce original endpoints

The NumPy-only budget readout found no D11 floor-active endpoints in the local
precision sweep. Comparing the same selected six nodes at the original step:

| Objective | Whole saved table gap | Selected nodes: original FD gap | Selected nodes: local numerical-input FD gap at h |
| --- | ---: | ---: | ---: |
| softmax_Er | `-1.645743e-04` | `1.733452e-05` | `-1.231777e-05` |
| Er2_volume_average | `6.946135e-03` | `6.166804e-03` | `-1.300859e-04` |
| bootstrap_current_softmax_abs_scaled | `-1.404244e-04` | `-1.901320e-04` | `6.619114e-06` |
| Er_transition_rho | `-9.497727e-09` | `6.852709e-09` | `-1.223632e-08` |

Every column uses signed JVP-minus-FD contributions and the actual saved
objective bars. The original and local selected-node gaps differ substantially
and have opposite signs. The local coefficient precision evidence therefore
does not establish an error budget for the original geometry FD endpoints.
It is premature to resolve the full mismatch by changing the FD step alone.
The original endpoint input direction and the full scan evaluation context
remain to be checked; a passing weighted JVP/VJP test still supports the AD
table derivative at its captured direction.

The original FD benchmark computes its VMEC state tangent in
`_manual_implicit_forward_state_tangent`. Production AD recorded its actual
state tangent inside the compact payload contraction. The next controlled
check compares those two at the saved AD state, parameter values and mask.
It uses no finite perturbation and performs no equilibrium, NTX coefficient,
root or transport solve. It does perform one VMEX linear-response calculation.

Run `check_saved_compact_geometry_gradient.py --fd-tangent-only --parameter ZBS:1:0`
with the archived `config.toml`, one saved `ad_compact_geometry_*.npz` as
`--capture`, and a fresh `--out-dir`. The script reads the adjacent saved
`ad_compact_tangents_*.npz`, uses the original FD configuration and tangent
entrypoint, and checks that the constructed parameters match the captured
values/dtypes/bytes. Constructor scalar/weak types are preserved. A different
VMEX module or parameter point aborts before the linear solve.

`state_tangents.csv` reports per-field maximum absolute and relative L2
differences; `state_tangents.npz` retains both directions. The original FD
run did not record its VMEC tangent, so this is a controlled repeat of that
entrypoint, not a claim to have recovered the exact old FD tangent. A passing
repeat would leave endpoint construction and scan context unresolved.

Five NumPy-only saved-geometry tests and eight coefficient-diagnostic tests
pass. The new bookkeeping test detects a wrong parameter column, preserves
nonzero tangent discrepancies, and rejects missing/invalid fields. The GPU
comparison has now completed; see the result below. Production physics and
transport methods are unchanged.

Add `--fd-report RUN/fd_ZBS_1_0/summary.json` to quantify the tangent difference
for all 13 costs in the same run. The diagnostic checks the original AD/FD
baseline arrays and objective values before using the FD report. It then
temporarily supplies each fixed state direction to the existing production
geometry payload contraction, retaining all recorded folded objective bars.
No additional implicit solve or NTX coefficient solve is performed for these
two projections. The temporary diagnostic substitution does not modify any
production implementation or run a transport rollout.

`gradient_directions.csv` records the saved AD gradient, the replay with the
saved AD direction (a control), the projection of the fresh FD direction,
and the original FD gradient for every cost. The control must reproduce the
saved AD result closely before attributing a projected change to the VMEC
direction. The printed `control_relative`, `original_relative` and
`projected_relative` columns expose all three comparisons. The new NumPy test
keeps control replay changes separate from effects of the direction and
rejects mismatched objective order or invalid projections.

## FD geometry direction result: not the main source of the gap

Run `fd_direction_audit_ZBS_XSdhvU` completed on JAX 0.10.0 using the same
geometry, VMEX implicit and Boozer source hashes as the earlier audit.
The replay with the saved AD direction reproduces all 13 saved AD gradients
within `1.982448e-15` relative. The new FD-entrypoint state tangent differs
from AD by `1.124061e-08` relative L2 in R_cos, `1.165407e-09` in Z_sin and
`8.700567e-09` in L_sin. The other three spectral fields match exactly.

Projecting that fresh FD tangent through the same production payload mapper
and recorded objective bars barely changes the observed discrepancies:

| Objective | Original AD vs saved FD | Fresh FD direction vs saved FD |
| --- | ---: | ---: |
| softmax_Er | `2.494047e-05` | `2.494011e-05` |
| net_total_power_volume_average_mw_m3 | `1.302500e-06` | `1.302154e-06` |
| Er_transition_left | `6.101322e-05` | `6.101360e-05` |
| Er_transition_right | `6.610191e-05` | `6.610244e-05` |
| Er2_volume_average | `2.112375e-05` | `2.112154e-05` |
| Er_volume_average | `1.602953e-05` | `1.602825e-05` |
| electron_temperature_volume_average_keV | `2.344183e-06` | `2.344325e-06` |
| total_pressure_volume_average | `2.345669e-06` | `2.345764e-06` |
| alpha_power_volume_average_mw_m3 | `1.408631e-06` | `1.408329e-06` |
| bootstrap_current_softmax_abs_scaled | `3.915224e-05` | `3.915179e-05` |
| Er_transition_rho | `6.965993e-07` | `6.963135e-07` |
| Er_transition_location_moment | `6.965993e-07` | `6.963135e-07` |
| Er_transition_strength | `1.744869e-06` | `1.745136e-06` |

These are relative differences, not percentages. This rules out the measured
VMEX tangent difference as the principal explanation for this direction.
It does not validate the entire geometry-to-table FD map. The exact original
FD tangent was not saved, and the nonlinear endpoint construction and the
coefficient scan execution context still require a controlled comparison.
Elapsed time was 1:46.43; maximum host RSS was 3,072,108 KiB. GPU peak memory
was not measured. Supplied attachment SHA256:
`558c38275ac8c4dda13c3dca41a2f324a024ba6c73f99918ff0b7773360a1275`.

## Repeat the actual geometry endpoints and separate the coefficient paths

`check_saved_ntx_geometry_endpoints.py` reuses this saved equilibrium and
`state_tangents.npz`, including its independently checked AD reference.
It calls the original FD runtime entrypoint at zero and the original plus/minus
step. A temporary observer stops that entrypoint immediately after it builds
the transport geometry, channels and native scan surfaces, before any NTX,
root or transport solve. All three numerical input sets are saved. The zero
geometry must match the captured AD geometry byte for byte.

For each selected radius, the checker then calls the same public
`ntx.solve_monoenergetic_scan` used by generic FD construction with the full
configured `(nu,Er)` grid. This retains the original scan shape instead of
replacing it with a six-point scan. The scalar comparison receives all floating
surface fields, the electric-field input and drds from exactly those endpoints.
The JVP includes live drds in D11_log/D13 normalization and the complete native
surface directions. It performs no VMEX equilibrium or implicit solve and
does not change any production implementation, controller or replay contract.

The default selects six previously identified nodes on three radii. It runs
nine full single-radius scans (three endpoints on each radius), plus scalar
and coefficient-JVP controls at those six nodes. It is not a transport run or
a new full-table derivative validation. Its GPU execution is still pending.

`endpoint_gaps.csv` gives a signed telescoping budget in derivative units:

1. `audit_control_gap`: cached reference JVP minus the current scalar-kernel
   JVP along the saved AD numerical-input direction. A large value prevents
   attributing that difference to the upstream geometry map.
2. `upstream_gap`: controlled AD-input JVP minus the JVP along the actual
   endpoint numerical-input secant. This measures the effect of upstream
   input changes through the local coefficient derivative.
3. `scalar_fd_gap`: input-secant JVP minus scalar FD of the actual endpoints.
   This combines finite-step coefficient bias and floating-point effects;
   the diagnostic does not claim to distinguish those two effects.
4. `scan_context_gap`: scalar FD minus production-scan FD with the same
   numerical endpoints.
5. `reproduction_gap`: fresh production-scan FD minus the original saved FD.
   A large value means the original endpoints have not been reproduced.

Their sum is `total_gap`, the cached JVP minus original saved FD. Baseline
scalar/scan primal offsets are recorded separately. `selected_node_budget.csv`
contracts these terms with all 13 original objective bars at the selected
nodes only; other nodes and the coordinate fields are excluded. The checker
never labels this partial budget as an explanation of the whole gradient.
`sources.txt` records numerical source hashes, including the scan module, and
the state-direction/config/equilibrium-input hashes.

Four NumPy-only tests pass: an independent nonlinear composition with injected
scan/reproduction errors, the interruption of the original constructor before
NTX/root work, exact baseline guards, and an end-to-end mocked run with live
normalization and a nonzero extra native-surface derivative. No real JAX or
physics was run locally.

After publishing the diagnostic, run on the large machine:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git switch en/v0_main || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  neopax_case=outputs/ntss1d_zbs_support_chain_UfR8XG
  neopax_out=$(mktemp -d "$neopax_case/geometry_endpoint_audit_XXXXXX") || exit 1
  set -o pipefail
  env JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
    JAX_ENABLE_COMPILATION_CACHE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    NTX_DISABLE_COMPILATION_CACHE=1 VMEX_COMPILATION_CACHE=disabled \
    "$HOME/jax_env/bin/python" examples/benchmarks/check_saved_ntx_geometry_endpoints.py \
    --run-dir "$neopax_case" \
    --state-tangents "$neopax_case/fd_direction_audit_ZBS_XSdhvU/state_tangents.npz" \
    --table-audit "$neopax_case/ntx_table_ZBS_PXKHjE" \
    --out-dir "$neopax_out" \
    2>&1 | tee "$neopax_out.log"
) || printf '\nEndpoint audit stopped; terminal kept open.\n'
```

The log stays outside the new output directory so it is still empty when the
script checks it. The six selected points are defaults; the complete endpoint
tables at their three radii are retained for further saved-array analysis.
