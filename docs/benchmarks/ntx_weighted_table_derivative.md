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
a new full-table derivative validation. GPU execution completed at commit
`35430f1`; the result is recorded below.

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

## Endpoint audit result: 2026-11-20

The supplied GPU log contains all 18 coefficient rows and all 13 weighted
objective budgets. It uses JAX 0.10.0, CUDA and
`h=3.39461852025036750e-07`. The saved AD/FD baseline checks and reconstructed
zero-step geometry guard passed. A standard-library readout verified that
the five signed components sum to every printed total within rounding error.
The log does not include the exact output subdirectory, timing or memory
measurements. The new `sources.txt` and NPZ/CSV files remain on the remote
machine; their additional contents have not yet been read.

The following are signed contributions to objective derivatives at the six
selected nodes, not relative errors or complete objective-gradient gaps:

| Objective | Cached/current AD control | Geometry input secant | Scalar coefficient FD | Scalar versus scan | Fresh versus archived FD | Selected-node total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| softmax_Er | `7.146570e-12` | `1.154426e-04` | `1.566293e-05` | `-1.800617e-08` | `-1.137530e-04` | `1.733452e-05` |
| Er2_volume_average | `4.884435e-11` | `3.016728e-03` | `2.085807e-04` | `5.532240e-06` | `2.935963e-03` | `6.166804e-03` |
| bootstrap_current_softmax_abs_scaled | `1.258304e-11` | `-2.905206e-06` | `1.107222e-05` | `-4.176763e-07` | `-1.978814e-04` | `-1.901320e-04` |

The controlled AD-input JVP still agrees with its cached reference. For these
weighted contributions, changing scalar execution to the full scan with
identical numerical endpoints is much smaller than the unresolved gap.
The Er2 contribution has substantial geometry-input-secant and endpoint
reproduction terms. The selected bootstrap discrepancy is dominated by the
failure of the fresh perturbed tables to reproduce the archived FD tables.
The softmax terms partially cancel, so the final small total must not be
mistaken for individually small errors at every boundary.

This does not establish nondeterminism for identical inputs: the original
FD run did not save its complete native scan surfaces/channels or VMEC state
tangent. The audit reuses the captured AD equilibrium and the checked repeat
of the FD tangent. Its exact zero-step transport geometry is a useful control,
but does not establish byte-identical perturbed native inputs. Likewise, the
`upstream_gap` combines finite-step geometry bias and floating-point effects;
the single step does not separate them. No faulty transport derivative or
NTX custom derivative follows from this result.

The already-written `baseline_inputs.npz`, `minus_inputs.npz`, `plus_inputs.npz`
and `endpoint_gaps.csv` permit the next boundary comparison without another
simulation: compare fresh versus archived transport geometry at zero/minus/
plus, then read the scalar/scan baseline offsets and both endpoint offsets.
This distinguishes changed known geometry inputs from coefficient differences
and exposes whether the new coefficient baseline itself reproduces the old
one. Matching transport geometry alone would still leave the unsaved native
scan inputs unresolved. That NumPy-only readout has now completed, as recorded
below.

Supplied attachment SHA256:
`261219941270256ead90a4c8d5bd7586d115b34066025f25480dba2adbfa9318`.

## Saved input readout: perturbed geometry already differs before NTX

The completed audit directory is
`outputs/ntss1d_zbs_support_chain_UfR8XG/geometry_endpoint_audit_6CAHcL`.
Its zero-step geometry matches the archived FD geometry in all 25 arrays.
All 18 sampled production-scan baseline coefficient values also match exactly.
At both the minus and plus endpoints, however, 20 of the 25 transport-geometry
arrays already differ from the archived FD values:

| Field | Minus: maximum absolute difference | Plus: maximum absolute difference |
| --- | ---: | ---: |
| a_b | `2.220446049e-16` | `2.220446049e-16` |
| B0 | `9.290346270e-13` | `8.935074902e-13` |
| Vprime | `1.090256774e-10` | `4.979483492e-11` |
| B0prime | `3.394096948e-10` | `8.558162790e-10` |
| enlogation | `4.019877764e-09` | `2.529997545e-09` |

These are input-array differences, not gradient relative errors. This locates
a changed boundary before NTX and prevents interpreting the archived/fresh
coefficient difference as a nondeterministic NTX solve on identical inputs.
The output does not establish whether the old equilibrium, old tangent or
their subsequent floating-point evaluation first differed: the original FD
did not record its VMEC linearization. The first known changed boundary is
the perturbed transport geometry, not a proven earlier spectral-state defect.

The corresponding coefficient offsets are small in value but amplified by
division by `2h`. For table D13 at node `4:2:6`, the fresh-minus-archived value
offsets are `6.149235079e-10` at minus and `-4.809385078e-10` at plus. Therefore
the change in its FD derivative is

```math
\frac{-4.809385078\times10^{-10}-6.149235079\times10^{-10}}
     {2(3.3946185202503675\times10^{-7})}
=-1.614116592428\times10^{-3}.
```

This independently reproduces the audit's `reproduction_gap` at that node.
It explains the arithmetic amplification of changed endpoint values, not
the entire origin of those changes or the complete objective gradient.
The scalar baseline differs from the scan by up to `1.196820421e-12` at the
sampled cells, while the production-scan baseline offsets are all zero.
The newly supplied source manifest matches the known geometry, VMEX and NTX
sources, including the sequential scan implementation hash
`f63cd55041f9d2441ca087d856faa0a0c71f5c3f4b17b8a03464b5248afd9a55`.
Attachment SHA256:
`42f6a303fea21c26bbaca4d69ae0a25842e11ebbdbbfb3e1a087082a02a9415e`.

## Reuse the recorded AD VMEC linearization in the original FD benchmark

The ordinary frozen-linearized FD constructor still independently solves the
baseline VMEC equilibrium and then its selected parameter tangent. The endpoint
audit instead used the saved AD equilibrium and a checked repeat of the FD
tangent. Matching final zero-step support does not prove that these underlying
linearizations are byte-identical at perturbed points.

The FD benchmark now has an optional `--ad-geometry-reference SNAPSHOT_DIR`.
It restores the actual AD spectral state, mask and the named tangent column
from one adjacent `ad_compact_geometry_*.npz` / `ad_compact_tangents_*.npz` pair.
It checks parameter labels, VMEC response mode, radial resolution, every
constructed VMEC parameter's shape/dtype/bytes, and finite compatible spectral
arrays. There is no equilibrium or implicit solve on this path. The original
`state_star +/- h*state_tangent` endpoint constructor, NTX scans, root contract,
Radau accepted-step map, and objective evaluation are retained. The default
path continues to make its existing solves.

When gradient capture is enabled, both paths now save
`fd_geometry_linearization_*.npz` with the actual state, tangent, mask, params
and reference provenance. Future comparisons need not reconstruct this missing
boundary after the fact. Full FD reports also record `ad_geometry_reference`.
Use the existing `--ad-baseline-reference` simultaneously to retain the strict
common initial-root/baseline guard.

Eighteen NumPy-only tests pass, including the actual benchmark helper and CLI
wiring with mocks, rejected mismatched references and unchanged default solve
calls. An additional NumPy-only check restored the state and both named tangent
columns byte-for-byte from a real archived AD capture. No JAX or physics ran
locally. GPU validation was pending when this option was added. The completed
FD16 result below confirms the saved-reference execution and common baseline,
but does not resolve the gradient discrepancies. This option removes
independent VMEC reconstruction from the comparison and is not a claimed cure
for all finite-difference roundoff or truncation errors.

After publishing, the following reuses the existing AD16 result and runs only
the original ZBS FD16 benchmark with the recorded AD geometry linearization.
It retains the existing root and accepted-step frozen-metadata contracts and
prints all 13 full parameter-gradient comparisons:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git switch en/v0_main || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  env -u PYTHONPATH -u LD_LIBRARY_PATH -u JAX_PLATFORM_NAME \
    -u XLA_FLAGS -u JAX_COMPILATION_CACHE_DIR -u XLA_PYTHON_CLIENT_ALLOCATOR \
    JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
    JAX_ENABLE_COMPILATION_CACHE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    NTX_DISABLE_COMPILATION_CACHE=1 VMEX_COMPILATION_CACHE=disabled \
    NEOPAX_TEST_RUNTIME_NTSS1D=1 \
    "$HOME/jax_env/bin/python" - <<'PY'
from pathlib import Path
import os, shutil, sys, tempfile
sys.path.insert(0, 'examples/benchmarks')
import benchmark_transport_ntss1d_ad_fd as check
base = (check.ROOT/'outputs/ntss1d_zbs_support_chain_UfR8XG').resolve()
snapshots = base/'ad/snapshots'
terminals = list(snapshots.glob('ad_terminal_*.npz'))
if len(terminals) != 1:
    raise SystemExit('Expected exactly one saved AD terminal capture')
out = Path(tempfile.mkdtemp(prefix='ntss1d_zbs_saved_ad_geometry_', dir=check.ROOT/'outputs'))
print('Results:', out, flush=True)
shutil.copy2(base/'config.toml', out/'config.toml')
(out/'ad').symlink_to((base/'ad').resolve(), target_is_directory=True)
command = [sys.executable, check.FD_SCRIPT, '--config', str(out/'config.toml'),
    '--parameter', 'ZBS:1:0', '--accepted-step-limit', '16', *check.FD_OPTIONS,
    '--ad-geometry-reference', str(snapshots),
    '--ad-baseline-reference', str(terminals[0])]
check.worker(command, out/'fd_ZBS_1_0', os.environ)
shutil.copy2(check.FD_REPORTS/'ZBS_1_0_forward_fd_frozen_linearized_summary.json',
             out/'fd_ZBS_1_0/summary.json')
rows = check.compare(out)
if not rows or not all(r['comparison_baseline_matches'] for r in rows):
    raise SystemExit('Baseline mismatch: gradient comparison is not validated')
for r in rows:
    print(f"{r['objective']}: AD={r['gradient_ad']:.12e} FD={r['gradient_fd']:.12e} "
          f"relative_error={r['relative_difference_to_fd']:.6e}")
PY
) || printf '\nSaved-geometry FD check stopped; terminal kept open.\n'
```

## Shared AD geometry FD16 result: discrepancy remains

Run `outputs/ntss1d_zbs_saved_ad_geometry_2_versy_` completed at benchmark commit
`db8da34`, using the existing AD16 result. The log explicitly confirms loading
the saved AD geometry linearization. It also records
`fd_geometry_linearization_hust3ha2.npz`, so the FD state and tangent are now
available for direct inspection. The fresh initial Er already agrees with AD
before applying the saved-reference override.

Baseline `initial_y`, `final_y`, `final_time`, geometry and database arrays
match AD exactly. The baseline has 16 accepted steps and 23 attempts at
`t=4.1195452419569767e-04`. Both perturbed accepted-step replays finish at this
same time, with 16 active steps, no nonconverged active steps and zero baseline
or endpoint Newton retries. Geometry and initial-root lanes remain
`frozen_linearized`, and the accepted replay contract remains
`reverse_frozen_metadata` with `recorded_forward` Jacobians.

The following relative errors use `abs(AD-FD)/abs(FD)` at the same
`h=3.3946185202503675e-07`. They are not percentages. The before column is the
original `ntss1d_zbs_support_chain_UfR8XG` FD reference; AD is unchanged in both
columns.

| Objective | Independent VMEC reconstruction | Recorded AD VMEC state/tangent |
| --- | ---: | ---: |
| softmax_Er | `2.494047e-05` | `6.241382e-06` |
| net_total_power_volume_average_mw_m3 | `1.302500e-06` | `8.953810e-07` |
| Er_transition_left | `6.101322e-05` | `1.300480e-05` |
| Er_transition_right | `6.610191e-05` | `1.282220e-05` |
| Er2_volume_average | `2.112375e-05` | `2.349415e-05` |
| Er_volume_average | `1.602953e-05` | `1.330842e-05` |
| electron_temperature_volume_average_keV | `2.344183e-06` | `2.278717e-06` |
| total_pressure_volume_average | `2.345669e-06` | `2.257583e-06` |
| alpha_power_volume_average_mw_m3 | `1.408631e-06` | `9.816440e-07` |
| bootstrap_current_softmax_abs_scaled | `3.915224e-05` | `4.598480e-05` |
| Er_transition_rho | `6.965993e-07` | `4.889824e-06` |
| Er_transition_location_moment | `6.965993e-07` | `4.889824e-06` |
| Er_transition_strength | `1.744869e-06` | `1.839022e-06` |

The printed AD/FD values independently reproduce all 13 reported relative
errors within their printed precision. Some Er gradients improve, but the
Er2 and bootstrap discrepancies persist and are slightly larger. The root
position discrepancy also increases. This is not a successful gradient fix;
independent VMEC reconstruction cannot explain the whole observed mismatch.
The result does not by itself identify a bad AD rule or establish FD roundoff
as the remaining cause. The exact-baseline and successful replay checks are
distinct from gradient correctness.

The run took 39:30.85 and reported maximum host RSS of 8,345,712 KiB. This is
host process memory, not GPU peak usage. No additional physics run is needed
for the next localization: apply the existing terminal/full-support saved-array
comparison to this new FD output and the same original AD bars. The earlier
support-chain agreement belongs to the earlier endpoint data and must not be
assumed for this new FD run. That NumPy-only readout has now completed; see
the new result below.

Supplied attachment SHA256:
`724c994184cc7993eb5272aa0438c259eaa7f0dee04faeed76284cf8556963db`.

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  neopax_case=outputs/ntss1d_zbs_saved_ad_geometry_2_versy_
  neopax_fold=("$neopax_case"/ad/snapshots/ad_before_scan_fold_*.npz)
  [ "${#neopax_fold[@]}" -eq 1 ] && [ -f "${neopax_fold[0]}" ] || exit 1
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$HOME/jax_env/bin/python" examples/benchmarks/compare_transport_gradient_audit.py \
    --ad-dir "$neopax_case/ad/snapshots" \
    --fd-dir "$neopax_case/fd_ZBS_1_0/snapshots" \
    --fd-report "$neopax_case/fd_ZBS_1_0/summary.json" \
    --ad-before-scan-fold "${neopax_fold[0]}" \
    --out "$neopax_case/fd_ZBS_1_0/support_chain.csv"
) || printf '\nSaved-array check stopped; terminal kept open.\n'
```

## Shared-geometry FD16 support chain: the transport boundary agrees

The completed saved-array check for `ntss1d_zbs_saved_ad_geometry_2_versy_`
matches baseline packed/physical states, time and objective values exactly.
The full terminal chain includes each objective's explicit geometry/table
contribution. The support chain contracts the pre-scan-fold adjoint with the
actual new FD geometry and table secants, including the initial-root and
transport contributions. Both comparisons use the new endpoints, rather
than assuming that the earlier FD run's result still applies.

All columns below are relative differences from this same full parameter FD:

| Objective | Full parameter AD | Full terminal chain | Full support chain |
| --- | ---: | ---: | ---: |
| softmax_Er | `6.241382e-06` | `1.809514e-10` | `8.638691e-09` |
| net_total_power_volume_average_mw_m3 | `8.953810e-07` | `5.536184e-09` | `1.255901e-08` |
| Er_transition_left | `1.300480e-05` | `0.000000e+00` | `5.759969e-08` |
| Er_transition_right | `1.282220e-05` | `0.000000e+00` | `2.222397e-08` |
| Er2_volume_average | `2.349415e-05` | `4.368774e-12` | `8.083489e-10` |
| Er_volume_average | `1.330842e-05` | `6.715926e-11` | `3.096932e-10` |
| electron_temperature_volume_average_keV | `2.278717e-06` | `9.170539e-09` | `1.333907e-08` |
| total_pressure_volume_average | `2.257583e-06` | `5.180900e-09` | `4.484248e-09` |
| alpha_power_volume_average_mw_m3 | `9.816440e-07` | `3.035909e-08` | `1.360227e-08` |
| bootstrap_current_softmax_abs_scaled | `4.598480e-05` | `1.948953e-09` | `6.687975e-11` |
| Er_transition_rho | `4.889824e-06` | `5.617453e-09` | `6.529878e-09` |
| Er_transition_location_moment | `4.889824e-06` | `5.617453e-09` | `6.529878e-09` |
| Er_transition_strength | `1.839022e-06` | `3.080962e-10` | `4.266008e-09` |

The largest support-chain discrepancy is `5.759969e-08`; bootstrap agrees to
`6.687975e-11`. Thus the fixed-support transport adjoint reproduces this FD
direction far more closely than the complete parameter gradient does. The
larger discrepancy is localized to the parameter-to-geometry/table mapping
or the difference between its derivative and its finite secant. This does
not yet distinguish an incorrect derivative from finite-step or numerical
precision effects at that mapping, and is not a validation of every parameter
or later time prefix.

The pressure terminal-state-only relative difference `5.282991e-04` concerns
a small partial state contribution. Adding the explicit geometry/table term
gives the full terminal-chain relative difference `5.180900e-09`. Those two
numbers must not be interpreted as errors of the same complete derivative.
Supplied attachment SHA256:
`b60987ab182c0bcaf0b6cf3db2d563b7de4c5171e3fb0e27ef3344797d63b9fb`.

## Check the input geometry map without coefficient or transport solves

`check_saved_geometry_payload_fd.py` uses all 13 actual AD16 objective rows and
their folded payload bars. It does not substitute arbitrary VMEC objectives.
With these bars held fixed, it contracts finite secants of the transport
geometry, normalization channels and native surfaces with exactly the bars
whose cached JVP reconstructs the full saved parameter gradient.

The checker reads the actual `fd_geometry_linearization_*.npz` from the new
FD16 run. Before GPU initialization it requires its state, mask, parameters
and named tangent to match the saved AD record byte-for-byte, and checks the
AD/FD baseline. It then uses the original FD constructor, interrupting it
immediately after native inputs are built and before NTX. It requires the
zero, minus-h and plus-h geometries to reproduce the saved FD geometry arrays
exactly. A mismatch aborts, rather than silently using a new reconstruction
as the explanation of the earlier result.

It applies the same step multipliers `1,3,10,30` to every input and objective.
There are nine forward geometry/input evaluations and no equilibrium,
implicit-linear-response, NTX coefficient, ambipolar-root or transport solve.
It also performs no new AD sweep. This tests the geometry input map separately
from the coefficient solve. Step sensitivity and a persistent derivative
mismatch can then be investigated at that boundary; increasing h is not
assumed to be uniformly better.

`geometry_map_fd.csv` contains the saved AD gradient, reconstruction control,
projected map FD, three input-group contributions and each relative difference
`abs(map_FD-AD)/abs(AD)`. The printed output labels this `relative_to_ad`.
The comparison against the previously saved full FD is defined only at the
original h; larger-h rows leave that column empty. The checker also writes
per-path contributions to `geometry_map_fields.csv`, the projected baseline/
original-step input arrays to NPZ, and source/input hashes to `sources.txt`.
The fixed bars are a local linearization: this check does not evaluate the
nonlinear NTX coefficient map or a changed transport endpoint.

Three new NumPy-only tests pass. An independent nonlinear input map checks
all cost rows and groups against analytic secants, checks nonzero curvature
at larger h, and rejects wrong cached derivatives or changed original-step
geometry. Other tests reject unsupported paths, invalid arrays and a changed
recorded spectral state before JAX is imported. The four existing pre-NTX
endpoint tests also pass. No JAX or physics ran locally. GPU execution of
this geometry-only check has now completed; see the result below.

After publishing these diagnostic files, run on the large machine:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git switch en/v0_main || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  neopax_case=outputs/ntss1d_zbs_saved_ad_geometry_2_versy_
  neopax_out=$(mktemp -d "$neopax_case/geometry_input_fd_XXXXXX") || exit 1
  set -o pipefail
  env -u PYTHONPATH -u LD_LIBRARY_PATH -u JAX_PLATFORM_NAME \
    -u XLA_FLAGS -u JAX_COMPILATION_CACHE_DIR -u XLA_PYTHON_CLIENT_ALLOCATOR \
    JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
    JAX_ENABLE_COMPILATION_CACHE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    NTX_DISABLE_COMPILATION_CACHE=1 VMEX_COMPILATION_CACHE=disabled \
    NEOPAX_TEST_RUNTIME_NTSS1D=1 \
    "$HOME/jax_env/bin/python" examples/benchmarks/check_saved_geometry_payload_fd.py \
    --run-dir "$neopax_case" --parameter ZBS:1:0 --out-dir "$neopax_out" \
    2>&1 | tee "$neopax_out.log"
) || printf '\nGeometry-input check stopped; terminal kept open.\n'
```

## Geometry-only sweep: native surface input FD dominates the remaining gap

Run `geometry_input_fd_bQ2Z1t` under
`outputs/ntss1d_zbs_saved_ad_geometry_2_versy_` completed at `a6afcc3` on JAX
0.10.0/CUDA. Its preflight checks passed, including byte-identical baseline
and original-step minus/plus geometry. There were no NTX coefficient, root,
transport, equilibrium or implicit-response solves. It reused the actual AD16
cost bars and checked all 13 objective rows. A standard-library readout
independently verified all 52 printed relative differences at their printed
precision and confirmed that the AD reference stayed unchanged across steps.

These relative differences are `abs(geometry_map_FD-AD)/abs(AD)`, with the same
multiplier for every objective and input, not percentages or full-transport
FD errors at the larger steps:

| Objective | h | 3h | 10h | 30h |
| --- | ---: | ---: | ---: | ---: |
| softmax_Er | `7.093370e-06` | `1.297093e-05` | `4.066287e-06` | `4.939542e-07` |
| net_total_power_volume_average_mw_m3 | `9.607474e-07` | `5.591904e-07` | `1.760418e-07` | `9.336428e-09` |
| Er_transition_left | `5.682992e-06` | `2.910831e-06` | `3.254192e-06` | `2.101201e-06` |
| Er_transition_right | `2.769199e-06` | `2.133162e-06` | `1.614098e-06` | `1.940635e-06` |
| Er2_volume_average | `2.529453e-05` | `4.496418e-06` | `3.597049e-07` | `8.464085e-08` |
| Er_volume_average | `1.203892e-05` | `7.324737e-07` | `2.192127e-07` | `1.208383e-07` |
| electron_temperature_volume_average_keV | `2.284889e-06` | `3.609252e-07` | `1.433240e-07` | `4.956710e-08` |
| total_pressure_volume_average | `2.261780e-06` | `3.601831e-07` | `1.644115e-07` | `5.204319e-08` |
| alpha_power_volume_average_mw_m3 | `1.068927e-06` | `5.469004e-07` | `1.769235e-07` | `1.297254e-08` |
| bootstrap_current_softmax_abs_scaled | `4.221015e-05` | `1.171753e-05` | `6.581062e-06` | `8.730645e-07` |
| Er_transition_rho | `2.271891e-06` | `6.275209e-06` | `7.090784e-07` | `1.720603e-07` |
| Er_transition_location_moment | `2.271891e-06` | `6.275209e-06` | `7.090784e-07` | `1.720603e-07` |
| Er_transition_strength | `1.276388e-06` | `2.698609e-06` | `6.834531e-07` | `1.610424e-07` |

At the original h, the geometry-input map alone reproduces most of the full
bootstrap and Er2 discrepancies. The signed derivative differences are:

| Objective | Full FD minus AD | Input-map FD minus AD | Full FD minus input-map FD |
| --- | ---: | ---: | ---: |
| bootstrap_current_softmax_abs_scaled | `1.647363283e-04` | `1.512209250e-04` | `1.351540330e-05` |
| Er2_volume_average | `-7.772153875e-03` | `-8.367543600e-03` | `5.953897247e-04` |

The strong reduction and fluctuating signs with larger h support small-step
finite-difference precision/cancellation in the upstream input construction
as the main numerical explanation for these observed gaps. The left/right
probe errors are less monotonic and are still about `2e-06` at 30h. This is
not proof of exact AD for every input or of a universal best FD step.

### Saved field contributions locate the native VMEC surfaces

The CSV-only readout supplied in chat separates the three input groups.
The entries below are signed FD-minus-AD contributions to the input-map
derivative, not relative errors:

| Objective | Input group | h | 30h |
| --- | --- | ---: | ---: |
| softmax_Er | geometry | `6.684934206e-07` | `-8.692424558e-10` |
| softmax_Er | channels | `-3.836174400e-08` | `-2.180318148e-10` |
| softmax_Er | surfaces | `-4.760686742e-05` | `-3.270186895e-06` |
| Er2_volume_average | geometry | `-4.156917521e-05` | `1.560798637e-07` |
| Er2_volume_average | channels | `1.299544792e-07` | `-4.117147512e-09` |
| Er2_volume_average | surfaces | `-8.326104439e-03` | `2.784761159e-05` |
| bootstrap_current_softmax_abs_scaled | geometry | `-1.557007305e-07` | `-5.013383375e-09` |
| bootstrap_current_softmax_abs_scaled | channels | `-7.582476313e-10` | `1.551536677e-14` |
| bootstrap_current_softmax_abs_scaled | surfaces | `1.513773836e-04` | `3.132830431e-06` |
| Er_transition_rho | geometry | `-3.319018480e-10` | `-2.902521973e-13` |
| Er_transition_rho | channels | `-3.920201316e-11` | `-9.558846770e-14` |
| Er_transition_rho | surfaces | `-2.982149879e-08` | `2.287004868e-09` |

The leading weighted errors are native-surface `b_cos` and
`b_sup_theta_cos`. For Er2 at h, `surfaces[6].b_cos` contributes
`-5.852596827e-03` and `surfaces[5].b_cos` contributes `-2.390727544e-03`.
For bootstrap, `surfaces[5].b_cos` contributes `1.205430118e-04` and
`surfaces[4].b_cos` contributes `5.072460803e-05`. These are sensitivity-weighted
errors; large contributions need not imply large errors in the raw field
values. Some large terms cancel, so a small total does not imply that every
field derivative has individually converged.

These arrays are constructed by
`_traceable_vmec_field_tables_from_state` (VMEX real-space fields followed by
Fourier projection) and `_traceable_vmec_surface_from_tables` (radial sampling
into native surfaces). The former builds `bmnc` from magnetic-field magnitude
and `bsupumnc` from the contravariant poloidal field; the latter exposes them
as `b_cos` and `b_sup_theta_cos`. This evidence locates the sensitive input
branch but does not isolate a single floating-point operation among field
reconstruction, projection and interpolation.

A source-only AST comparison of local `en/v0_new` at `117f797` and
`en/v0_main` at `a6afcc3` found identical definitions for
`_traceable_vmec_field_tables_from_state` and
`_traceable_vmec_surfaces_from_state`. In `_traceable_vmec_surface_from_tables`,
the only code difference is the already-recorded b0 correction from a maximum
reduction to gathering the maximizing coefficient. The extraction formulas
for the two dominant fields above are unchanged. This source check does not
compare VMEX/JAX internals or establish that a particular JAX upgrade caused
the changed FD behavior.

No production physics, forward solver, AD rule or default FD step was changed
in response. The main discrepancy is now reproducible before any NTX solve
and diminishes substantially with a larger step. The full FD16 confirmation
below also shows substantial improvement. Its results must be assessed
separately from the input-map sweep because it includes nonlinear coefficient
and transport effects. The earlier coefficient sweep also showed that
excessively large steps can increase truncation error.

Geometry-only sweep attachment SHA256:
`04728f9b40f72fa09a4074f70aeebf0561cfe3bc606204d122d20f55782f2143`.

### Full FD16 confirmation at the measured 30h step

The full FD16 confirmation completed in
`outputs/ntss1d_zbs_fd30h_1o6j_jep`, using the existing benchmark without
another code change or AD run. It reused the same saved AD16 point, VMEC state
and tangent, initial-root contract and frozen accepted-step replay contract.
One uniform `--fd-rel-step 9e-6` was used for all 13 ZBS objective rows. This
is 30 times the original relative increment; its absolute increment is
approximately `1.018385556e-05` for this coefficient. The interpolation is
`preprocessed_3d_radial_ntss1d` with 51 radial points and the runtime NTX grid
`(25,31,64)`. These results concern `ZBS:1:0` only.

The fresh initial Er matched saved AD before the reference override. The
baseline initial state, terminal state and time matched AD exactly, with
zero maximum support difference. The baseline had 16 accepted steps and 23
attempts, ending at `4.1195452419569767e-04`. Both perturbed endpoints replayed
16 accepted steps at those fixed timesteps, with no Newton retries or active
nonconverged steps. All reported state and support arrays were finite.

The table compares the original h and 30h full-transport FD errors against
the unchanged AD16 reference. Both columns use `abs(AD-FD)/abs(FD)`, not the
input-map sweep's `relative_to_ad` metric:

| Objective | Full FD16 at h | Full FD16 at 30h |
| --- | ---: | ---: |
| softmax_Er | `6.241382e-06` | `1.298666e-07` |
| net_total_power_volume_average_mw_m3 | `8.953810e-07` | `2.177318e-08` |
| Er_transition_left | `1.300480e-05` | `6.451990e-06` |
| Er_transition_right | `1.282220e-05` | `5.041166e-06` |
| Er2_volume_average | `2.349415e-05` | `2.161232e-07` |
| Er_volume_average | `1.330842e-05` | `1.870485e-06` |
| electron_temperature_volume_average_keV | `2.278717e-06` | `4.969649e-08` |
| total_pressure_volume_average | `2.257583e-06` | `4.145530e-08` |
| alpha_power_volume_average_mw_m3 | `9.816440e-07` | `1.459823e-08` |
| bootstrap_current_softmax_abs_scaled | `4.598480e-05` | `6.985355e-08` |
| Er_transition_rho | `4.889824e-06` | `1.896401e-07` |
| Er_transition_location_moment | `4.889824e-06` | `1.896401e-07` |
| Er_transition_strength | `1.839022e-06` | `1.210307e-07` |

The 13 printed relative errors were independently checked against the
printed AD/FD gradients within their rounding precision, using only a
standard-library readout. The run exited successfully after `40:01.37`.
Maximum host RSS was `8,348,248 KiB`; this is not a measurement of GPU peak
memory.

Together with the geometry-only sweep, the large reduction in bootstrap and
Er2 discrepancies supports small-step FD precision/cancellation in upstream
native-surface construction as a major source of the earlier errors. The AD
reference did not change. The left/right Er probes still differ by about
`5e-06` to `6e-06`, and mean Er by `1.87e-06`. This is not an exact match for
every cost or evidence of identical errors to the old branch. The full
support-chain contraction should be checked on these new 30h snapshots to
separate upstream secant errors from finite-step effects in the transport
map; the successful h support-chain check does not automatically establish
that result at 30h. No further simulation is needed for that readout.

This is a validation point, not a change to the default FD step or a claim
that 30h is the best step for every objective or transport prefix.

Full FD16 attachment SHA256:
`fedbc7e123036e5ed78b97abe3e5e9da1ea33e114d24720689f0caf1e73f1968`.

Reproduction command:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  env -u PYTHONPATH -u LD_LIBRARY_PATH -u JAX_PLATFORM_NAME \
    -u XLA_FLAGS -u JAX_COMPILATION_CACHE_DIR -u XLA_PYTHON_CLIENT_ALLOCATOR \
    JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
    JAX_ENABLE_COMPILATION_CACHE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    NTX_DISABLE_COMPILATION_CACHE=1 VMEX_COMPILATION_CACHE=disabled \
    NEOPAX_TEST_RUNTIME_NTSS1D=1 \
    "$HOME/jax_env/bin/python" - <<'PY'
from pathlib import Path
import os, shutil, sys, tempfile
sys.path.insert(0, 'examples/benchmarks')
import benchmark_transport_ntss1d_ad_fd as check
base = (check.ROOT/'outputs/ntss1d_zbs_saved_ad_geometry_2_versy_').resolve()
snapshots = base/'ad/snapshots'
terminals = list(snapshots.glob('ad_terminal_*.npz'))
if len(terminals) != 1:
    raise SystemExit('Expected exactly one saved AD terminal capture')
out = Path(tempfile.mkdtemp(prefix='ntss1d_zbs_fd30h_', dir=check.ROOT/'outputs'))
print('Results:', out, flush=True)
shutil.copy2(base/'config.toml', out/'config.toml')
(out/'ad').symlink_to((base/'ad').resolve(), target_is_directory=True)
command = [sys.executable, check.FD_SCRIPT, '--config', str(out/'config.toml'),
    '--parameter', 'ZBS:1:0', '--accepted-step-limit', '16', *check.FD_OPTIONS,
    '--fd-rel-step', '9e-6', '--fd-abs-step', '1e-10',
    '--ad-geometry-reference', str(snapshots),
    '--ad-baseline-reference', str(terminals[0])]
check.worker(command, out/'fd_ZBS_1_0', os.environ)
shutil.copy2(check.FD_REPORTS/'ZBS_1_0_forward_fd_frozen_linearized_summary.json',
             out/'fd_ZBS_1_0/summary.json')
rows = check.compare(out)
if not rows or not all(r['comparison_baseline_matches'] for r in rows):
    raise SystemExit('Baseline mismatch: gradient comparison is not validated')
for r in rows:
    print(f"{r['objective']}: AD={r['gradient_ad']:.12e} FD={r['gradient_fd']:.12e} "
          f"relative_error={r['relative_difference_to_fd']:.6e}")
PY
) || printf '\nLarger-step FD check stopped; terminal kept open.\n'
```

## Original-step arithmetic isolation on saved spectral data

A subsequent NumPy-only source-algebra replay kept the original
`h=3.3946185202503675e-07`. It used the existing local
`axis_fix_saved_arrays_20261116_RPTPxK/ad_snapshots` capture: the actual
201-surface, 72-mode equilibrium and its saved ZBS tangent and objective
bars. This is an earlier generic-interpolation AD16 capture, not a local
copy of the latest NTSS1D run. The numerical results below therefore must
not be presented as that run's complete gradient errors.

The scratch replay executes selected VMEX functions' source ASTs with a
NumPy array namespace and functional copies replacing JAX indexed updates.
It loads no JAX/NTX/VMEX modules and performs no equilibrium, implicit,
coefficient, root or transport solve. It retains the benchmark's six
poloidal modes, six toroidal modes, angular grid `(10,16)`, zero pressure,
zero-current constraint, flux and original cubic Hermite radial sampling.
The numerical experiment compares float64 arithmetic with NumPy
`longdouble` (63 mantissa bits on this host, not IEEE binary128), using the
same double-precision saved inputs and step. It is an independent algebra
check, not a reproduction of CUDA kernels or their reduction ordering.

### Perturbation addition already changes the realized direction

With the original double-precision update used by FD,

```
x_plus  = float64(x + h*v)
x_minus = float64(x - h*v)
```

the realized secant `(x_plus-x_minus)/(2*h)` is not exactly `v`, even when
the captured baseline and intended tangent match byte-for-byte. Examples
from this saved ZBS direction:

| State field | Largest absolute direction error | Location (radial row, mode) |
| --- | ---: | --- |
| R_cos | `2.610084343e-09` | `(48,0)` |
| Z_sin | `1.306524521e-09` | `(45,1)` |
| L_sin | `3.080474764e-11` | `(200,1)` |

At the first location the stored coordinate is `12.048933794228684`, the
intended direction is `-0.06278770474858161`, and the realized secant is
`-0.06278770213849727`. This is input rounding, not a different VMEC solve
or a changed perturbation label.

### Half-mesh radial differencing amplifies the small errors

`vmex.core.geometry.half_mesh_jacobian` forms radial differences after
real-space synthesis, including

```
rs_inner = ((R_even[j]-R_even[j-1])
            + sh*(R_odd[j]-R_odd[j-1])) / hs
```

The equilibrium has `hs=1/200`; transport's 51-point grid does not set this
spacing. On the seven sampled interior regions, the main-source float64
FD secant differs from the extended-precision secant by relative L2
`2.782592012e-08` in `R_even`, growing to `6.746020854e-06` in `dR_ds`.
The corresponding maximum absolute differences are `8.783184197e-09` and
`2.659368819e-06`. This then changes the Jacobian, contravariant fields and
the projected native-surface coefficients.

At the same original h, native-surface coefficient secants agree much more
closely with the actual cached AD tangent when both perturbation addition
and field algebra use extended precision:

| Main-source surface field | Float64: max abs(secant - cached AD) | Extended precision: max abs(secant - cached AD) |
| --- | ---: | ---: |
| surface[5].b_cos | `1.600030096e-05` | `2.236664580e-09` |
| surface[5].b_sup_theta_cos | `1.151732694e-06` | `1.927326333e-10` |
| surface[6].b_cos | `5.267079912e-06` | `2.573963673e-09` |

These are absolute coefficient-direction errors, not relative errors of
full cost gradients. The bootstrap-weighted sum over all captured native
surface fields changes from `1.255848171e-04` to `-5.045387896e-08` in
signed FD-minus-AD contribution units. If only the state additions are
rounded to float64 and the rest of the field calculation stays in extended
precision, a `5.151033244e-05` bootstrap contribution gap returns.
Rounding only `R_cos` gives `4.290833453e-05`, only `Z_sin` gives
`8.542887373e-06`, and only `L_sin` gives `-4.492635623e-08`.
Thus input quantization as well as subsequent arithmetic contributes.

### What the old/main source comparison does and does not establish

VMEX `en/local_test` uses a single phase-table contraction over signed
helical modes. `en/neopax-main-utilities` repacks those coefficients and
contracts toroidal and poloidal modes separately, including parity slices.
This changes floating-point reduction order before the radial subtraction.
On the *same captured main point*, NumPy execution of the old source also
shows a float64 FD gap and a strong extended-precision improvement: its
bootstrap native contribution gap is `1.741371605e-04` in float64 and
`-8.598171243e-08` in extended precision. Main is not systematically worse
in this independent calculation. This does not reproduce the old JAX
0.7.2/CUDA benchmark and does not identify a JAX-version regression.

A scratch variant that differences the spectral coefficients before
synthesis reduces the main Er2 native contribution gap from
`-4.536946143e-03` to `-8.772464021e-05`, but leaves a bootstrap gap of
`5.527101428e-05`, consistent with remaining state-addition rounding.
It is not a complete fix and was not applied to either production repository.

The concrete mechanism reproduced here is loss of precision while forming
and radially differentiating perturbed coordinates, before NTX or transport.
It explains why a byte-identical baseline/intended direction is insufficient
to certify a small-h FD reference. It does not establish that all remaining
NTSS1D/CUDA errors share that cause. Production physics, time integration,
AD rules and default FD increments remain unchanged.

## Applied correction: preserve native-field increments at the original h

The FD benchmark now accepts `--geometry-fd-native-precision extended`.
It addresses the demonstrated precision loss in the validation reference;
it is not a change to the production transport solver or reverse AD rule.
The ordinary `float64` mode remains available and is the default so archived
comparisons retain their original meaning. The chosen mode is saved in the
FD report. It requires frozen-linearized geometry with native VMEC scan
surfaces and a host whose `longdouble` is wider than float64; unsupported
precision or configurations raise an error rather than silently falling back.

`_fd_native_precision.py` executes the existing forward field functions in
copied namespaces with a NumPy extended-precision array backend. It does
not modify imported VMEX/NEOPAX modules or use a JVP/VJP to construct
endpoints. The only separately expressed numerical rule is the same local
cubic Hermite interpolation (arithmetic-mean adjacent slopes) used by
`interpax` at these fixed radial queries. Its linear, interior quadratic
and extrapolation behavior are independently checked.

Both state addition and field reconstruction stay in extended precision
until the native-surface finite increments are computed. For each field,

```
corrected_endpoint = ordinary_baseline
                   + (extended_forward(x + h*v) - extended_forward(x))
```

The subtraction and addition above are performed in extended precision
before the final result is cast back to the ordinary field's dtype. This
recentring preserves the exact common CUDA baseline and excludes a static
CPU/CUDA evaluation offset from the measured increment. It does not force
an endpoint, derivative or cost to match AD. At zero step the hook returns
the original surface arrays unchanged and remembers those exact objects
used by the baseline scan. It performs no extended-precision evaluation.
This avoids a second CUDA reconstruction when choosing the anchoring
baseline. Recorded solver/replay metadata follow their original code path.

The correction is deliberately limited to native VMEC surface inputs.
Transport geometry and normalization channels retain their ordinary
evaluation, and NTX coefficient construction, interpolation, initial-root
linearization and accepted-step replay are unchanged. Their residual
numerical errors are not removed by this option.

### Saved-state verification before any new CUDA solve

The helper itself was exercised on the actual saved equilibrium, tangent
and all 13 objective bars from the earlier local generic AD16 capture,
with the original `h=3.3946185202503675e-07`. The ordinary baseline remained
byte-identical. Signed discrepancies of the *native-surface contribution*
against the cached AD input directions were:

| Objective | Ordinary float64 | Corrected, same original h |
| --- | ---: | ---: |
| softmax_Er | `4.021459880e-05` | `3.747713753e-08` |
| net_total_power_volume_average_mw_m3 | `-4.379574555e-09` | `2.557398063e-12` |
| Er_transition_left | `8.505867673e-05` | `1.506046783e-09` |
| Er_transition_right | `7.920196296e-05` | `6.982253482e-09` |
| Er2_volume_average | `-4.536946143e-03` | `-9.150915515e-07` |
| Er_volume_average | `1.335733308e-04` | `2.534813189e-08` |
| electron_temperature_volume_average_keV | `4.148143274e-09` | `-3.388874557e-13` |
| total_pressure_volume_average | `5.937849814e-09` | `-1.653951438e-12` |
| alpha_power_volume_average_mw_m3 | `-4.325671847e-09` | `2.527700865e-12` |
| bootstrap_current_softmax_abs_scaled | `1.255848171e-04` | `-5.322614018e-08` |
| Er_transition_rho | `2.593124034e-10` | `7.079154961e-12` |
| Er_transition_location_moment | `2.593124036e-10` | `7.079154961e-12` |
| Er_transition_strength | `2.330074807e-07` | `3.625366766e-10` |

These are absolute signed contribution errors, not full AD16/FD16 relative
errors and not results from the latest NTSS1D CUDA run. The table was
recomputed after the nested-call traversal correction described below.
Five new NumPy-only
tests pass, covering the source-function backend, independent interpolation
and nonlinear derivative oracles, preservation of baseline/displacement,
and the actual FD runtime hook including its zero-step/default paths.
The existing 11 baseline/replay and 3 geometry-map tests also pass without
JAX imports. CUDA integration and the resulting full FD16 gradients remain
unverified locally.

After publishing the changes, the existing geometry-only checker can
verify this correction on the current NTSS1D capture at *only the original
step*, before committing time or memory to another transport run:

Publish only the five intended files from the local `en/v0_main` checkout
(the index may contain unrelated staged scratch files):

```bash
(
  cd /mnt/d/Github_Codes/NEOPAX-v0_main || exit 1
  git switch en/v0_main || exit 1
  neopax_files=(
    examples/benchmarks/_fd_native_precision.py
    examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py
    examples/benchmarks/check_saved_geometry_payload_fd.py
    tests/test_fd_native_precision.py
    docs/benchmarks/ntx_weighted_table_derivative.md
  )
  git add -- "${neopax_files[@]}" || exit 1
  git commit --only -m "Preserve native geometry increments in frozen FD checks" -- "${neopax_files[@]}" || exit 1
  git push origin en/v0_main
) || printf '\nPublish stopped; terminal kept open.\n'
```

On the large machine:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  neopax_case=outputs/ntss1d_zbs_saved_ad_geometry_2_versy_
  neopax_out=$(mktemp -d "$neopax_case/native_precision_fix_XXXXXX") || exit 1
  set -o pipefail
  env -u PYTHONPATH -u LD_LIBRARY_PATH -u JAX_PLATFORM_NAME \
    -u XLA_FLAGS -u JAX_COMPILATION_CACHE_DIR -u XLA_PYTHON_CLIENT_ALLOCATOR \
    JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
    JAX_ENABLE_COMPILATION_CACHE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    NTX_DISABLE_COMPILATION_CACHE=1 VMEX_COMPILATION_CACHE=disabled \
    NEOPAX_TEST_RUNTIME_NTSS1D=1 \
    "$HOME/jax_env/bin/python" examples/benchmarks/check_saved_geometry_payload_fd.py \
    --run-dir "$neopax_case" --parameter ZBS:1:0 \
    --native-precision extended --step-scales 1 --out-dir "$neopax_out" \
    2>&1 | tee "$neopax_out.log"
) || printf '\nNative precision verification stopped; terminal kept open.\n'
```

It uses the saved state and tangent, with no equilibrium, implicit, NTX,
root or transport solve. For the later full FD16 confirmation, the existing
FD command takes `--geometry-fd-native-precision extended` at the original
`--fd-rel-step 3e-7`; the same AD16 reference and all frozen contracts apply.

### First CUDA verification: missed generator dependency corrected

The run at `66492e3` passed preparation and began the original-step minus
input evaluation, but failed before any new comparison rows. Inside VMEX's
`fourier_to_real`, `_toroidal_stage` is referenced by a generator expression.
The namespace copier inspected only the outer function's `co_names`;
generator/nested-function code objects have separate name tables. The
uncopied `_toroidal_stage` therefore called JAX `concatenate` with host
extended-precision arrays, which JAX correctly rejected as `float128`.
This is a precision-helper bug, not a reported gradient mismatch.

The focused correction recursively collects names from nested `CodeType`
constants before copying dependencies. A new regression uses nested
functions and comprehensions with an original backend that explicitly
rejects extended inputs. It reproduces the reported failure before the
patch and passes after it, while asserting that the original namespace
was not mutated. All five precision regressions now pass without JAX.

The earlier local saved-state wrapper used NumPy for original functions,
which masked this backend leak; it was not a substitute for the CUDA
integration check. After the traversal fix, the actual saved-state check
again completes all 13 rows with an exact baseline; its updated results
are in the table above. CUDA verification still requires rerunning the
same original-step command in a fresh output directory.
