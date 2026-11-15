# Unresolved Er and bootstrap gradient investigation

The full sixteen-accepted-step geometry/bootstrap discrepancy is **not fixed**. The latest
matched-value RBC bootstrap derivative is `3.8531638809481883e-1` in AD and
`1.1847207619995317e-1` in FD, relative error `2.252381493e0`.
The matched T0 profile test below validates its Er and bootstrap derivatives;
the other five profile columns have not yet been repeated at a matching
point. The archived profile FD baseline differs from the latest AD baseline.
Agreement of physical replay endpoints does
not certify derivatives, and component tests do not bound error accumulation
through the full calculation.

## Original versus main execution path (source audit, 2026-11-14)

Restoring the raw-block response did **not** restore the complete original
equilibrium path. Main still selects the implicit primal, whose host solve,
state refinement and constrained-mode assembly differ from the old dependency.
This audit makes no solver/configuration changes and runs no physics tests.

The local source references are NEOPAX `en/v0_new` at `117f797`, VMEX
`en/local_test` at `2fd82e7b`, and NTX at `af223d7`. The main references are
NEOPAX `b95fad7` plus its diagnostic working-tree changes, VMEX utilities at
`e5190622`, and NTX utilities at `4f7aa88`. The archived October 17 log does
not record its exact source commits; its actual command is in the timing
footer, including `--full-transport-shared-payload-smoke`, sixteen accepted
steps, four-step segments and recorded-forward Jacobian reuse. The old
checkout is a source reference, not an imported production dependency.

| Boundary | Original v0_new path | Current main path |
| --- | --- | --- |
| Configured equilibrium for transport | TOML `vmec_lane="forward"`; ordinary `solve_multigrid(..., mode="cli")` | TOML `vmec_lane="ad"`; `solve_implicit` |
| Separate geometry auxiliary solve | Mixed evaluator calls `geometry_raw_block_solve_from_param_vector`, which calls `solve_implicit_with_aux` regardless of the TOML lane | Same NEOPAX function and call; different VMEX implementation underneath |
| Auxiliary primal returned by VMEX | Old `_host_solve_and_mask` returns the host solver's state without fixed-point refinement | Main implicit host solve holds the constrained m=1 family and then refines its state |
| Runtime chosen at zero geometry delta | `use_runtime_payload=True`: retain baseline runtime; do not replace it with the auxiliary state | Same shortcut: retain baseline runtime, now created with the configured implicit primal |
| Geometry and scan inputs | Build Boozer-derived transport geometry and VMEC scan surfaces from the selected runtime state | Same active NEOPAX geometry construction; environment imports main dependencies |
| Coefficient database | Live NTX scan converted to `Monoenergetic`; fixed table passed into transport | Same NEOPAX scan model, conversion and recorded-owner split; newer NTX/SOLVAX and native interpolation underneath |
| Initial Er | Configured two-stage root selection; selected-root implicit derivative | Same NEOPAX initialization/selected-root routing |
| Transport reverse | Production accepted-step map; recorded-forward Newton/Jacobian metadata, segmented support reverse | Same accepted-step map and support-reverse entrypoint |
| Final NTX/geometry chain | Fold accumulated database bars through the saved NTX scan, then map geometry/channels/surfaces bars through VMEC | Same fold/assembly routing and compact tangent contraction, with optional diagnostic recordings |
| VMEC raw response | Original raw block tangent and paired transpose, with the old state assembly/residual | Restored raw block inverse/tangent/transpose algorithm, still using main's assembly/residual |
| Frozen geometry FD | Forward equilibrium zero point; raw-block tangent there; build endpoints from `state_star +/- h*state_tangent` | Implicit equilibrium zero point from the shared raw helper; same endpoint formula |

### Runtime reuse is decisive

The full shared-payload benchmark starts with `build_runtime_context(config)`.
The mixed evaluator separately obtains `shared_raw_block_solve`, then sets
`use_runtime_payload` when all geometry deltas are close to zero. The internal
table builder starts with `table_context.baseline_runtime` and rebuilds from
the raw-solve state only when that flag is false. Consequently the original
baseline transport run really used its configured forward equilibrium; the
existence of an implicit auxiliary solve does not replace that runtime.

The final geometry contraction and geometry-only objectives consume the
separately supplied raw-solve state. Old VMEX returned its unrefined host
state for that auxiliary solve. This does not prove that two independent
old equilibrium solves were bitwise identical; the old FD source explicitly
preserves the forward zero point rather than assuming it can substitute an
implicit solve.

Main's TOML was changed to `ad` to align the transport primal definition
with main's implicit auxiliary definition. That setting was not reverted
when the raw-block response functions were restored. It is therefore
incorrect to describe the latest raw-block run as a restoration of the
complete v0_new path.

### The raw-block restoration also retains main's constrained-mode semantics

The inverse now performs the original unrefined raw block solve, rather than
delegating to main's refined/certified response API. However, both the tangent
and transpose call main's `residual_fn` and `_assemble`. Old `_assemble`
returns its assembled spectral state directly; main returns
`_held(SpectralState(...), rt_p)`. The latter derives the constrained m=1
family from the parameter-dependent initial guess. Main's implicit host solve
is also wrapped by `_holding_m1`; old `_host_solve` is not. These are
differences in the primal/parameter response, not transport timestep choices.
Removing response refinement alone does not remove either difference.

### Source comparison and limits

A read-only AST comparison finds identical NEOPAX definitions for the runtime
dispatcher, single-parameter solver routing, raw auxiliary helper, mixed
least-squares evaluator, transport geometry construction, full geometry
objective pullback, runtime scan model, scan conversion, root setup, recorded
scan fold and accepted-step map. The compact geometry bridge differs only by
the audit recording calls. The FD accepted replay and frozen-linearized
initial-root helper are also identical. The configured AD-lane geometry FD
bundle differs as described above. Module loading now uses installed
VMEX/NTX instead of inserting sibling checkouts; the alternative Boozer scan
loader changed, but this benchmark selects the VMEC surface backend.

This verifies the listed orchestration functions, not every transitive
dependency implementation. The interpolation implementation, NTX/SOLVAX
linear algebra and JAX version also changed and are not certified equivalent
by this source comparison. The archived run has eleven transport objectives;
the current run has thirteen after adding left/right Er. The current local
v0_new source already contains those added objectives.

Changing only the TOML back to `forward` would leave main's refined/held
auxiliary state in the derivative path, recreating the split between primal
and linearization. An exact comparison requires explicitly identifying and
aligning the equilibrium state, mask and constrained-mode response used by
both AD and FD. The saved main forward/implicit states already show maximum
absolute differences `1.526272e-3` (R_cos), `2.630243e-3` (Z_sin) and
`9.962389e-3` (L_sin).

These findings explain why a cross-branch gradient comparison was not at the
same geometry. They do **not** establish the cause or resolution of the
remaining bootstrap AD/FD disagreement on a matched main baseline. Both
current AD and geometry FD already select the implicit primal and raw-block
tangent, so that discrepancy still requires investigation through the full
geometry chain.

## Additional boundary checks (2026-11-14)

The actual NEOPAX setup functions were executed with only the expensive
equilibrium calls replaced by the supplied saved state. Transport-primal,
two-parameter AD auxiliary and one-parameter FD auxiliary setup produce
identical parameter arrays and the same canonical VMEX configuration object.
The parameter-byte SHA256 is
`730479114cb92de4f80d70375670a002e443731c79fc82c7627be102f4ac3765`.
All three select CLI multigrid, ns=201, ftol=1e-16, refine_tol=1e-10 and
device=None. This verifies setup/cache-key equality without executing or
comparing new equilibrium solves. Both real solver entrypoints then use
the same callback, with equilibrium refinement enabled.

A bounded GPU check selects twelve strongly weighted coefficient cases on
rho=0.625. It uses the actual thirteen accumulated AD16 cotangent rows and
the saved RBC geometry direction, with corrected main SOLVAX. The production
thirteen-row native scan transpose agrees with forward AD for the weighted
bootstrap contribution: `1.490699117010257` versus `1.490699119339745`,
absolute difference `2.329488e-9`. Original-step FD gives
`1.490872504538565`, an absolute difference `1.733852e-4` from forward AD.
This subset does not validate the other scan cases or explain the full
`2.668443e-1` bootstrap discrepancy. All thirteen rows are saved in
[actual_ad16_weighted_ntx_boundary_2026-11-14.csv](actual_ad16_weighted_ntx_boundary_2026-11-14.csv).

A further case at rho index 4, nu index 2 (`nu=3e-6`) and Er index 1
checks the low-collisionality region separately. Native and complete
prepared transposes agree with direct forward AD to `1.98e-8` relative
for D11, `1.12e-7` for D13 and `7.09e-8` for D33. Identical-prepared-input
FD at the original step differs by `1.75e-5`, `1.76e-4` and `2.04e-4`,
respectively. See
[actual_ad16_low_nu_ntx_boundary_2026-11-14.csv](actual_ad16_low_nu_ntx_boundary_2026-11-14.csv).
This is another component result, not validation of the full scan fold.

The saved terminal bootstrap map was then evaluated by FD without any NTX,
equilibrium, root or transport solves. It uses the original RBC step
`3.3131168969686005e-7` and the saved final plasma state:

| Support direction | AD / cotangent projection | Measured FD | Absolute difference |
| --- | ---: | ---: | ---: |
| Geometry only | `-5.940458672855726e-1` | `-5.940458170203275e-1` | `5.026525e-8` |
| Linear table direction from the twelve cases | `1.259679899451639e0` | `1.259679898206406e0` | `1.245233e-9` |
| Actual twelve-case NTX endpoints | `1.259853896459586e0` | `1.259853896487216e0` | `2.763034e-11` |
| Coupled database radius / electric-field knots | `-5.672417859065727e-1` | `-5.672417867283179e-1` | `8.217452e-10` |
| Geometry plus database coordinates | `-1.161287653192145e0` | `-1.161287599727454e0` | `5.346469e-8` |

The complete endpoints are saved in
[actual_ad16_bootstrap_endpoint_map_2026-11-14.csv](actual_ad16_bootstrap_endpoint_map_2026-11-14.csv).
The accumulated D13/D33 table cotangents match the independent terminal
cotangents within `1.02e-12` / `1.59e-13` absolute L2. D11 cotangents arise
through the transport/root response; their terminal bootstrap counterparts
are zero. No duplicated terminal D13/D33 contribution was found.

### Full saved-support FD projection completed

The completed GPU check evaluates both original-h endpoints for all 761
coefficient cases with a nonzero saved AD16 cotangent. Physical solves remain
sequential, with 101 MiB temporary executable workspace. It reuses the saved
equilibrium, RBC direction, final state and accumulated thirteen-objective
cotangents. It runs no new equilibrium, root, ODE or transport reverse sweep.

For bootstrap, the recorded AD16 gradient is `3.853163880946595e-1`, while
FD of the support map projected through those **same saved cotangents** gives
`1.208306510333128e-1`, a difference `-2.644857370613467e-1`. This reproduces
nearly the entire previously observed `2.668443e-1` discrepancy after the
transport/root cotangents have already been fixed. Thus that reproduced gap
does not require a different accepted-step reverse or another transport run.
It lies in the downstream coefficient/geometry chain or its FD evaluation;
this result alone does not distinguish a transpose bug from a coefficient
solver's numerical behavior.

The root-position projection differs from recorded AD by `5.750579e-9`
absolute; strength differs by `2.412424e-5`. All thirteen rows are in
[actual_ad16_full_support_fd_projection_2026-11-14.csv](actual_ad16_full_support_fd_projection_2026-11-14.csv).
This support projection is not an independent rerun of the full objective FD;
the prior remote full-objective bootstrap FD was `1.184720761999532e-1`.

### NEOPAX recorded conversion bookkeeping

The production `recorded_runtime_database_support_bar_batched` was tested
separately with the actual thirteen input table-bar rows. Only its NTX
transpose callback was replaced with zero outputs, exposing its direct
raw-scan/table conversion and channel terms without any coefficient solve.
The D13/D33 coefficient bars match the analytic conversion exactly; D11
matches within `1.17e-16` relative L2. The bootstrap channel projection is
`-5.862463772998396e-2` in the production path and
`-5.862463772998416e-2` in independent forward AD, absolute difference
`2.08e-16`. The other objective rows also agree to rounding.
See [actual_ad16_recorded_conversion_2026-11-14.csv](actual_ad16_recorded_conversion_2026-11-14.csv).
This clears the checked conversion bookkeeping, not the real NTX transpose,
whose replacement was explicit in this diagnostic. The remaining coefficient
cases and the folded geometry contraction still need isolation. No production
physics, interpolation, equilibrium or timestep method has been changed.

### Additional forward and reverse coefficient checks

Forward AD through the production coefficient solve was completed at every
active table case on the three outer radii, using sequential solves and
161 MiB executable temporary workspace. Each case's primal and tangent is
checkpointed separately from the earlier FD endpoints. The bootstrap table
contributions are:

| Scan radius index | Forward AD table contribution | FD table contribution |
| --- | ---: | ---: |
| 4 | `4.821459843341137e-1` | `4.807091499466434e-1` |
| 5 | `1.354430403328137e0` | `1.356068279082259e0` |
| 6 | `-3.147276148227701e-1` | `-3.139678610991104e-1` |

Adding the direct geometry and coordinate terms gives a forward-AD subtotal
`1.175260833942387e-1`. This subtotal excludes the inner-radius table
contributions: it must not be presented as the completed full gradient.
For scale, their combined contribution in the completed FD projection is
approximately `2.34e-3`. The forward calculation was interrupted after saving
all three outer radii and the first inner chunk, to focus the next check on
the reverse boundary without competing diagnostic processes.

Sixteen further cases at radius index 4 were then checked with the actual
thirteen-row production NTX primitive transpose. These include the
low-collisionality entries with the largest FD/forward differences and
strong cases not covered by the earlier twelve-case sample. Their largest
absolute bootstrap contribution difference from forward AD is `9.92e-8`.
The completed rows are in
[actual_ad16_additional_ntx_cases_2026-11-14.csv](actual_ad16_additional_ntx_cases_2026-11-14.csv).
The check stopped at its memory-margin guard before executing the next
radius while another GPU job was active. This is a partial coefficient
check, not a completed replay of the full scan transpose.

The saved bootstrap support tree was also inspected for omitted terms: it
contains only `geometry` and `database`, with no direct channel/surface or
native coefficient bars. The projection above therefore does not omit an
additional support branch. FD table-center checks find small reconstruction
differences concentrated at low collisionality; no large primal table
mismatch has been established as the cause.

The investigation has not identified a production line to fix. In
particular, these passing case checks cannot certify the complete recorded
scan fold or its final VMEC contraction. The supplied capture contains no
actual folded payload bars or compact-route VMEC state/parameters/mask.
The separate supplied implicit state reproduces the active runtime geometry
to the documented tolerances, but that does not prove equality of the
actual derivative contraction. The compact-route recording hook above
captures that missing boundary during a future AD16 run without adding a
physics solve or changing its derivative method. No further physics test
should be described as validating the full calculation until those actual
boundaries, or a complete reconstruction of the fold, have been compared.

### Latest RBC errors versus the archived v0_new table

Using `abs(AD-FD)/abs(FD)` in both tables gives the following comparison.
These runs have different equilibria, so this compares achieved derivative
agreement, not equal gradient values. The archived FD entries were printed
with six decimal places in scientific notation.

| Objective | Archived v0_new relative error | Latest main RBC relative error |
| --- | ---: | ---: |
| softmax Er | `5.325790e-5` | `2.231250e-5` |
| Er transition rho / location | `1.520333e-4` | `7.499709e-5` |
| Er transition strength | `5.968778e-4` | `5.153851e-4` |
| Bootstrap | `3.220080e-3` | `2.252381e0` |

The latest matched RBC Er rows are within the old table's error range;
bootstrap remains the clear large regression. This statement does not
validate every profile or ZBS column.

## Downloaded AD16 capture (2026-11-14)

`ad16_gradient_audit_HGNZFk3J.tar.gz` contains the actual thirteen-objective
terminal capture and the cotangents before the NTX scan fold. It uses the
benchmark's `7x16x11` database. Independently differentiating the terminal
objectives on these saved inputs reproduces the recorded packed-state
cotangents: relative L2 differences are `9.74e-14` for bootstrap,
`4.33e-19` for transition strength, and zero for softmax Er and the left/right
terms. This check does not run a transport, root, NTX scan or VMEX solve.
The complete rows are saved in
[actual_ad16_terminal_vjp_2026-11-14.csv](actual_ad16_terminal_vjp_2026-11-14.csv).

A separate generic terminal bootstrap VJP also reproduces the recorded
cotangents for D13, D33, the electric-field scan coordinates, database length
scale, transport radii and magnetic-field average, with relative L2
differences below `4.3e-13`. The scan's fixed rho/nu axes are held constant by
the production compact transpose; their unrestricted generic cotangents
are therefore not a comparison of active geometry derivatives. These checks
clear the terminal objective boundary at this point, not the earlier
transport sweep or the NTX/VMEX fold.
The unrestricted support rows, including the fixed-axis entries, are in
[actual_ad16_bootstrap_support_vjp_2026-11-14.csv](actual_ad16_bootstrap_support_vjp_2026-11-14.csv).
Derived radial-knot entries have absolute differences no larger than
`6.1e-14`; their relative errors against zero should not be read as errors of
that size in a final parameter gradient.

The six saved pre-fold component trees sum to the accumulated support bars
within `1.6e-16` relative L2. This verifies their bookkeeping only.

Reconstructing the fixed-table runtime locally gives sixteen accepted steps
and twenty-five attempts. Its final time differs from the downloaded run by
`5.67e-11 s`, and its maximum packed-state difference is `2.92e-7`.
The comparison guard refuses to treat these as identical baselines. A
profile AD/FD comparison must use one shared locally recorded schedule or
matching remote endpoint captures, rather than mix these two endpoints.

### Completed matched profile column: T0

The archived November 12 T0 FD prefix ends at
`3.6693474249138833e-4 s`; the downloaded November 14 AD terminal capture
ends at `3.6641282487344959e-4 s`. Both contain sixteen accepted steps,
but their endpoints differ in physical time by `5.2191761793874e-7 s`.
They therefore do not define the same frozen accepted-step map. The runs
constructed their baselines independently across intervening source and
dependency updates. The exact upstream cause of the changed schedule has
not been isolated; this observation alone is not proof of nondeterminism
within one executable or of which update caused the difference.

The completed local GPU test uses the production thirteen-objective support
reverse through all sixteen accepted steps, including its implicit selected
initial-root boundary. It restores the downloaded geometry/database and
initial Er; it performs no new VMEX equilibrium or NTX coefficient scan.
Its profile AD matrix differs from the remote matrix by at most `4.83e-7`
absolute across all 78 entries. The local endpoint is kept separate from the
remote endpoint rather than compared with the old FD reports.

FD runs in a fresh process using the original central step `h=5.34e-6`,
frozen geometry/tables, frozen-linearized initial roots, and the same
recorded accepted times and Newton/Jacobian-reuse decisions as this local
AD calculation. Reconstruction of the local AD schedule matches its saved
endpoint and time exactly. The unperturbed FD replay differs from that
endpoint and its objective values by at most `8.88e-16`; both end at
`3.6641288153859146e-4 s`. The final AD/FD relative errors are:

| Objective | abs(AD-FD)/abs(FD) for T0 |
| --- | ---: |
| softmax Er | `4.785058e-10` |
| transition rho / location moment | `1.089522e-9` |
| transition strength | `1.305372e-8` |
| bootstrap | `6.010122e-10` |
| net power | `1.345881e-10` |

All thirteen T0 rows are in
[matched_ad16_T0_fd_2026-11-14.csv](matched_ad16_T0_fd_2026-11-14.csv).
Thus the earlier T0 profile comparison with the mismatched archived FD
baseline is not evidence of a broken accepted-step derivative. This result
does not validate the other profile columns or the NTX/VMEX geometry chain.
The combined local process ended before FD produced results; the completed
AD matrix was retained, and FD was completed separately without repeating
the AD sweep. All physics execution used the GPU with no CPU fallback.

A direct source comparison against the unmodified local `en/v0_new`
checkout finds identical definitions of `_radau_apply_accepted_step_map`,
the minimal recorded segment replay call, the frozen-metadata FD accepted
replay and the frozen-linearized initial-Er helper. The benchmark TOMLs
differ in the configured VMEX primal lane (`forward` in the old checkout,
`ad` in the main-based checkout); the accepted-step algorithm was not
replaced for this comparison.

## Completed isolation checks

Measurements are saved in
[gradient_rule_isolation_2026-11-13.csv](gradient_rule_isolation_2026-11-13.csv).
These are component checks, **not additional full AD16 results**.

* On identical saved NTX prepared inputs, production implicit reverse case
  gradients agree within `3.0e-10` relative under JAX 0.7.2 and 0.10.0
  (they are not bitwise identical), with the same
  current main-based NTX and corrected main-based SOLVAX source. Forward AD
  through the selected-head factorization disagrees with those reverse
  gradients in both versions. For example, the D13 epsilon derivative has
  relative disagreement `1.442349e-3` under 0.10.0 and `5.340295e-4` under
  0.7.2. This inconsistency is not established as the new regression's cause.
* On the saved 51-point plasma state and saved QI database, the terminal
  bootstrap pullback agrees with forward AD: relative differences below
  `6.1e-14` for density, pressure and Er directions, and `2.9e-15` for D13
  and D33 table directions. The D11 direction is zero in both. This fixture
  has a `7x16x12` database, unlike the benchmark's `7x16x11` live database;
  it cannot clear the terminal rule for every point in the remote benchmark.

No equilibrium, initial-root, NTX scan or transport solve was performed for
these checks. The saved NTX single-case solves and terminal evaluations ran
on the local GPU, with bounded executable workspace and no CPU fallback.
No physics, interpolation rule, transport integrator or reverse method was
changed during this investigation.

## Capture the actual full-calculation boundaries

The old logs contain final Jacobian entries but not their intermediate
cotangents. Set this additional variable on the existing AD16 command:

```bash
export NEOPAX_GRADIENT_AUDIT_DIR=outputs/gradient_audit16/ad
```

The optional recording copies existing arrays to unique NPZ files:

* `ad_terminal_*`: actual final packed state, terminal objective cotangents,
  objective labels/values, final time, initial packed state and fixed support.
* `ad_before_scan_fold_*`: profile Jacobian, accumulated database/geometry
  bars and their already-computed components, before the one NTX scan fold.
* `ad_compact_geometry_*`: the active compact geometry route's VMEC state,
  parameters, mask, folded payload bars and geometry Jacobian. This route
  contracts payload bars with VMEC tangents and returns before the fallback
  state-cotangent/VMEC-transpose code.
* `ad_vmec_transpose_input_*` and `ad_vmec_transpose_output_*` apply only when
  that fallback runs. They contain its actual VMEC-state and parameter bars.

The first supplied capture, commit `b95fad7`, saved the terminal and pre-fold
files but bypassed the original transpose recording hooks through this compact
return. The compact-route recorder was added after that capture. The existing
two files remain useful; analyze them before requesting any further run.

The flag adds no forward solve, replay, VJP, component scan fold or tangent
solve. In particular, do not enable the wide component-pullback option just
to obtain these files. With the variable unset there is no array traversal
or transfer. Recording does synchronize transfers and uses host/disk space;
the complete remote benchmark with recording has not yet been run.

For FD, use a separate recording directory **for each DOF and run**, while
retaining the original FD step, frozen geometry, selected-root linearization
and accepted-step metadata settings. For example:

```bash
export NEOPAX_GRADIENT_AUDIT_DIR=outputs/gradient_audit16/fd_T0
```

This writes `fd_baseline_*`, `fd_T0_minus_*` and `fd_T0_plus_*`, including
their final packed and physical states. Prioritize T0 for the unresolved
profile root-strength discrepancy and RBC:1:0 for the matched-baseline
bootstrap failure. Existing scalar-only reports cannot supply these arrays
after the fact.

After those runs, this command reads saved files only:

```bash
python examples/benchmarks/compare_transport_gradient_audit.py \
  --ad-dir outputs/gradient_audit16/ad \
  --fd-dir outputs/gradient_audit16/fd_T0 \
  --fd-report outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/T0_forward_fd_frozen_linearized_summary.json \
  --out outputs/gradient_audit16/T0_terminal_chain.csv
```

It checks the baseline arrays and computes
`terminal_objective_bar @ ((final_y_plus - final_y_minus)/(2*h))`.
For geometry it compares this with the FD report's fixed-geometry final-state
contribution; for profiles it compares with the full objective FD. A profile
disagreement also requires checking any direct dependence of frozen
temperature projections on the profile parameters; it does not automatically
implicate the reverse sweep. The script refuses mismatched baselines instead
of presenting them as current derivative errors. It does not test the entire
NTX/VMEX geometry chain by itself.

Six small array/analytic tests pass for recording, repeated-call preservation,
integer cotangent placeholders, terminal contraction and baseline rejection.
No full local transport validation was run.
