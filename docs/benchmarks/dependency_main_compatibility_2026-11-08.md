# VMEX and NTX main-branch compatibility: 2026-11-08

## Full benchmark primal-anchor audit: 2026-11-10

The supplied successful GPU runs are finite, but not numerically equivalent.
`v0_new` used 290 accepted steps and `v0_main` (`84e47ba`) used 285. The
transport objective values differ by at most 0.000200%, while geometry-only
QI/max-J values differ by 2.63%/1.46%. Bootstrap derivatives with respect to
RBC(1,0)/ZBS(1,0) differ by 19.3%/55.9%. The T0 transport column is about
3.38% smaller in magnitude on main. Those derivative differences are not
validated merely by observing finite values or a small replay endpoint error.

The source trace identifies a concrete primal/linearization inconsistency:

1. The benchmark TOML selected `geometry.vmec_lane = "forward"`.
   `_run_realtime_geometry_reverse_mode` calls `build_runtime_context`, whose
   geometry builder uses the ordinary multigrid forward solve.
2. The mixed least-squares evaluator separately calls
   `geometry_raw_block_solve_from_param_vector`. That uses the implicit
   solve with auxiliary data. New VMEX refines its returned state onto the
   implicit residual root; old `en/local_test` returned the host-solver state
   without that refinement.
3. At zero geometry displacement the evaluator sets `use_runtime_payload`
   true. Transport therefore continues with the existing forward runtime,
   while its final support-to-geometry VJP and geometry-only objectives use
   the implicit state. The zero-displacement shortcut does not check that
   those two equilibrium states coincide.

The shared benchmark TOML now selects `vmec_lane = "ad"`: its primal runtime
and geometry pullback use the same implicit equilibrium definition. This also
applies to the FD benchmark when using that same TOML. The AD report records
the configured primal lane and pullback lane. The input equilibrium, profiles,
grid, objectives, weights and accepted-step replay algorithm are unchanged.
Older FD results must be regenerated before a comparison at this new primal
anchor. This change does not establish parity with the old unrefined VMEX
response, which also differs in held-mode parameter derivatives and block
adjoint refinement/certification.

The apparent adjoint-RHS change from an assembly transpose to `P(gbar)` is
not by itself a missing term: the old assembly's derivative with respect to
its free state is the symmetric projector P. Its parameter contribution
remains a separate direct assembly VJP in the new implementation. The
ported shared-J QI/max-J function and its ten local helpers are unchanged
apart from placement/comments in the local dependency checkouts.

The observed main replay mismatch is at most 3.20e-12 in absolute state and
2.19e-7 after dividing by the solver's `atol + rtol * max(abs(states))`.
That is a fraction of the allowed state tolerance, not a relative gradient
error. It is far below the endpoint rejection threshold of 1, but endpoint
agreement alone does not certify the derivative.

Use the following bounded diagnostic on the remote GPU machine before
another full transport reverse run:

```bash
cd "$HOME/NEOPAX" || exit 1
JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
  "$HOME/jax_env/bin/python" \
  examples/benchmarks/check_vmex_primal_consistency.py \
  --include-boozer \
  --out-dir outputs/vmex_primal_consistency
```

It runs the forward and implicit VMEX solves, compares their state arrays
and optional Boozer objectives, and writes CSV/NPZ artifacts. It builds no
NTX scan and runs no ambipolarity or transport solve. Its comparison/error
handling and end-to-end artifact writing passed three tiny tests using NumPy
arrays and mocked solvers. The equilibrium diagnostic and
the corrected full benchmark have not been run locally. The exact numerical
share of each geometry discrepancy and the temperature-column discrepancy
remain to be isolated; the two supplied runs alone cannot determine them.

### Received GPU primal-state comparison

The remote JAX 0.10.0 run confirmed a numerical difference between the two
VMEX equilibrium paths using the same input deck:

| State field | Maximum absolute difference | Relative L2 difference |
| --- | ---: | ---: |
| `R_cos` | 0.00152627 | 0.0190191% |
| `Z_sin` | 0.00263024 | 0.0821938% |
| `L_sin` | 0.00996239 | 6.34546% |

`L_sin` contains VMEC's lambda angle-correction coefficients, so its relative
change is not a percentage change in magnetic-field strength. The suppressed
symmetry families remain zero to numerical precision.

The implicit magnetic-well penalty is 0.027481918565822407. This agrees with
the magnitude of the earlier main benchmark's magnetic-well scalar to about
1.2e-15. The forward value, 0.027476134737736763, differs from the magnitude
of the earlier `v0_new` value by about 6.0e-9. This is direct evidence for
different primal anchors; it does not quantify their contribution to every
transport derivative.

The first diagnostic version's `--include-boozer` evaluated reduced Boozer
observables and inadvertently omitted QI/max-J. The option now uses the full
geometry objective helper, including both objectives. Existing state snapshots
can be evaluated without repeating either equilibrium solve:

```bash
JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
  "$HOME/jax_env/bin/python" \
  examples/benchmarks/check_vmex_primal_consistency.py \
  --reuse-states outputs/vmex_primal_consistency/states.npz \
  --include-boozer \
  --out-dir outputs/vmex_primal_consistency_full_objectives
```

Use the same input deck and harmonic point that produced the snapshot. The
three tiny tests now also verify that both QI/max-J are written, and that the
saved-state path calls neither equilibrium solver. The diagnostic deliberately
compares `forward` against `ad` even when the corrected benchmark configuration
selects `ad` for its runtime.

The follow-up saved-state evaluation reproduced the main benchmark's implicit
mean iota (-0.5935876720479774), magnetic well (-0.027481918565821245), and
Mercier softmax (3.22768911172778). Its QI values 0.07901798840939844 /
0.07836820772359807 and max-J values 172.64624537265053 / 169.796559543182
used three surfaces, however: the diagnostic had inherited the general
context builder's `(0.25, 0.5, 0.75)` default, whereas the AD benchmark
factory selects `(0.1, 0.28, 0.46, 0.64, 0.82, 1.0)`. These are valid
comparisons between the saved equilibria but are not the benchmark's QI/max-J
aggregates. The mirror-ratio aggregate also depends on that selection.

The diagnostic now calls the benchmark's context factory directly, preserving
its surface selection, config aliases and Boozer resolution. It prints the
resolved requested surfaces and Boozer resolution. Reuse the existing
`states.npz` to evaluate the six-surface objectives; changing the objective
surfaces does not require another equilibrium solve. The small end-to-end
test executes the actual benchmark factory with its heavy builder replaced,
checks the six defaults and explicit surface/resolution overrides, and still
asserts that saved-state evaluation invokes neither solver. A separate real
factory import/default check passed without building a geometry context or
running any solve. Three diagnostic tests passed.

### Matched six-surface result

The remote run after `10c3672` confirmed the benchmark selection
`surface_s=(0.1, 0.28, 0.46, 0.64, 0.82, 1.0)`, `mboz=36`, `nboz=11`.
It reused the same saved equilibrium states. The comparison now reproduces
the geometry values in both original benchmark logs:

| Objective | Earlier `v0_new` | Saved forward state | Earlier `v0_main` | Saved implicit state |
| --- | ---: | ---: | ---: | ---: |
| QI | 0.213175147543050 | 0.213175160462260 | 0.207574930396346 | 0.207574930396347 |
| max-J | 443.408693982044 | 443.408763107004 | 436.936910365491 | 436.936910365507 |
| Mean iota | -0.593652599661015 | -0.593652598527763 | -0.593587672047977 | -0.593587672047977 |
| Magnetic well | -0.027476128749680 | -0.027476134737736 | -0.027481918565821 | -0.027481918565821 |
| Mirror ratio | 0.211538034671637 | 0.211538030656737 | 0.211565418205833 | 0.211565418205833 |
| Mercier softmax | 3.232961223588242 | 3.232961281863436 | 3.227689111727780 | 3.227689111727780 |

The saved implicit QI/max-J values agree with the earlier main benchmark
within 6.94e-16 / 1.53e-11 absolute difference. The saved forward values are
within 6.1e-8 / 1.6e-7 relative difference of the old benchmark. This
numerically explains the geometry-objective discrepancy through the changed
equilibrium path; it is not an interpolation-loss definition change.

No further VMEX primal diagnostic is needed before the corrected full AD run.
The shared benchmark already selects `vmec_lane="ad"`. The corrected run is
still needed to measure the resulting transport values and derivatives:
these state/objective comparisons do not establish that all bootstrap or
temperature-profile gradient differences are resolved. The newer VMEX also
changes the held-mode parameter response and certified adjoint solve.

## NEOPAX migration update: 2026-11-09

The NEOPAX adapter changes are now in the isolated `NEOPAX-v0_main` checkout
on `en/v0_main`. The historical audit below describes the dependency main
revisions **before** the utility ports; its missing-function findings do not
describe the prepared PR branches.

The numerical utilities are supplied by these main-based dependency ports:

| Repository | Branch | Tested utility revision |
| --- | --- | --- |
| VMEX | `en/neopax-main-utilities` | `32d96d34` |
| NTX | `en/neopax-main-utilities` | `4f7aa88` |

NEOPAX now uses VMEX's official `params_from_input(..., device=...)`, preserving
the complete parameter schema, including `ac_aux_f`. Its existing default/JAX
placement aliases retain their previous meaning. Boundary updates leave the
other parameter leaves intact. NTX's public derivative export replaces the
old prepared-module import; three local adjoint paths use main's consolidated
`ntx._solver` module because they require low-level algebra with traced prepared
systems. Their arithmetic and custom-AD boundaries are unchanged.

The in-memory Boozer surface adapter calls `ntx.vmex_backend` and passes the
matching VMEX `SolverRuntime`. NEOPAX no longer prepends sibling dependencies
or installs the `vmec_jax.api` shim. The AD/FD entry points use the current
exports. The standalone J-contour utility imports the surrogate from its new
home in `vmex.core.omnigenity`.

The package metadata requires Python 3.11 and names upstream VMEX and NTX
`main` directly. Those branches must include the utility PRs before a clean
main-only installation can run the advanced derivative paths. Until then,
use the prepared dependency checkouts explicitly, as described in
`docs/getting_started.rst`.

Validation uses an isolated environment with JAX 0.11.2 and the two dependency
ports above. Source/import checks resolve production NTX imports and direct
calls, VMEX module aliases, the compact derivative availability gate and the
AD/FD benchmark entry points. Existing tiny regression cases cover the
current parameter schema, device aliases, runtime forwarding, local adjoint
rules and recorded-scan ownership. No new pytest cases were added; the two
existing adapter cases were strengthened. All 15 selected existing regression
cases passed. Changed files compile, the static comparison against `HEAD`
adds no syntax or undefined-name diagnostics, and `git diff --check` passes.
No equilibrium or transport solve,
full optimization or GPU benchmark was run locally. Numerical parity of the
full benchmark remains a GPU validation step.

## Original source audit (before the utility PRs)

NEOPAX cannot currently switch unchanged to both upstream `main` branches.
VMEX main provides newer equilibrium and block-adjoint interfaces that can
replace part of the old bridge, but the established QI/max-J surrogate is
absent. NTX main lacks the recorded-scan reverse interface used by the current
database transport benchmark, as well as several compact derivative helpers.

This is a source/API audit, not a numerical comparison of the two dependency
stacks. No equilibrium, NTX, transport, AD or FD solve was run. Neither
dependency checkout was switched, edited or installed. Fetching updated remote
references did not change their checked-out source files.

## Revisions and scope

| Repository | Current local reference | Fetched main |
| --- | --- | --- |
| NEOPAX | `en/v0_main`, `805aa5b` | Not changed by this audit |
| VMEX | `en/local_test`, `2fd82e7b` | `fa938e3dab974077c86527f83e29febbaae13b45` |
| NTX | `en/custom_rule_NEOPAX`, `af223d7` | `50445e6be679464fc84a354b56f237728f3dab59` |

Both origins were fetched and their live branch tips checked with
`git ls-remote --heads origin`. The critical-entry-point search covered all
74 live VMEX branches and 24 live NTX branches. Stale, deleted remote refs
were excluded. One VMEX experimental branch changed during the audit and was
refetched before the final search. The GPU machine's installed dependency
commits were not supplied; the local working branches above are the reference.

## VMEX findings

| Capability | Main status | NEOPAX implication |
| --- | --- | --- |
| Fixed-boundary equilibrium and implicit state solve | Present | Main retains `params_from_input`, `make_config`, `solve_implicit` and `solve_implicit_with_aux`; this is not an absence of equilibrium AD. |
| Parameter schema | Changed | `ImplicitParams` has a required `ac_aux_f` field. NEOPAX's manual explicit-device constructor does not supply it. |
| Raw block transpose | Present under a newer API | Use `implicit_state_pullback_multi_rhs(..., solver="block")` through an adapter; the old `implicit_state_pullback_multi_rhs_raw_block_transpose` name is absent. |
| Raw block state tangent | Present under a newer API | Main provides `implicit_state_tangent_multi_rhs`, which accepts a batch and returns `(state_batch, report)`; NEOPAX calls the old single-direction helper returning only a state. |
| Established QI/max-J surrogate | Absent | `j_invariant_qi_maxj_residual_from_boozer` is absent. This is the default NEOPAX backend. |
| Physical-J QI and maximum-J objectives | Present | `core.qi` and `core.maxj` provide the APIs used by NEOPAX's explicit `physical` backend. Their definitions and validity rules differ from the established surrogate. |
| Mercier diagnostics | Present; aggregation differs | `d_merc_state` and `mercier_stability_residual` exist, but `mercier_stability_softmax` is absent. Preserve the intended aggregate, sign, normalization and radial selection when adapting optional Mercier objectives. |

Relevant NEOPAX call sites are in `NEOPAX/_geometry_autodiff.py`:

- `_implicit_params_from_input`, line 702; manual `ImplicitParams` construction
  at line 718. The automatic-device path already calls `params_from_input`;
  the schema problem applies to the explicit-device constructor.
- `QImaxJBackendSettings`, line 20, defaults to `backend="surrogate"`.
- `_vmec_j_invariant_qi_maxj_objectives_from_boozer`, line 2545 onward,
  distinguishes the physical backend from the established surrogate.
- The compact tangent contraction calls `implicit_state_tangent_raw_block`
  at line 6573.
- The state-cotangent path calls
  `implicit_state_pullback_multi_rhs_raw_block_transpose` at line 6702.

Main's block pullback accepts the same leading state/parameter operands,
factors the raw radial block Jacobian, refines the solution and checks a
residual certificate. Its batched tangent also returns a convergence report.
An adapter must preserve and check those results. Availability of this newer
API does not prove numerical identity with the old branch: main also changed
fixed-point refinement and held-mode treatment.

Main source references:

- [Implicit differentiation and block response APIs](https://github.com/uwplasma/vmex/blob/fa938e3dab974077c86527f83e29febbaae13b45/vmex/core/implicit.py)
- [Physical-J QI definition](https://github.com/uwplasma/vmex/blob/fa938e3dab974077c86527f83e29febbaae13b45/vmex/core/qi.py)
- [Physical maximum-J definition](https://github.com/uwplasma/vmex/blob/fa938e3dab974077c86527f83e29febbaae13b45/vmex/core/maxj.py)

Silently selecting `physical` would change the loss being optimized. To retain
the current benchmark, preserve the existing surrogate explicitly or port it
upstream; evaluate a physical-backend migration as a separate objective change.

## NTX findings

The current NEOPAX runtime always calls:

```python
ntx.build_ntx_neopax_scan_from_surfaces(
    surfaces,
    # rho, nu_v, Es, Er, drds, grid and source_name omitted here
    coefficient_reverse_mode=mode,
    return_primal_record=record_scan_primal,
)
```

NTX main's signature accepts neither `coefficient_reverse_mode` nor
`return_primal_record`, and has no catch-all keyword argument. Source-level
signature binding therefore fails even before a recorded-scan adjoint can
be attempted. Both flags are supplied by NEOPAX even when recording is false.
The call is in `NEOPAX/_transport_flux_models.py`, line 7593.

The database reverse path also requires these functions, absent from main:

- `pullback_neopax_scan_coefficient_blocks_from_primal_record`
- `pullback_neopax_scan_coefficient_blocks_from_primal_record_batched_vmec_native`

NEOPAX calls them at lines 7679 and 7778. Main internally defines
`NeopaxScanCoefficientBlocks`, but does not export it at package level as
NEOPAX expects. Adding an export alone would not supply the missing recording
and transpose implementations.

Other NTX exact-Lij and experimental support paths require custom helpers
absent from main, including:

- `solve_prepared_coefficient_vector_derivative_vjp`
- `solve_prepared_coefficient_vector_hessian_factorized`
- `solve_prepared_coefficient_vector_two_directional_prepared_vjp`
- compact `lowdot_two_pullbacks` variants that reuse prepared systems and
  return geometry/support cotangents.

Some advanced NEOPAX paths also import private `_solver_*` modules. NTX main
consolidated these modules, so those imports need updating independently of
the missing numerical rules.

Main has basic prepared solves, their ordinary VJPs, and forward NEOPAX scans.
Those are not the recorded-primal, bounded-transpose contract used by our
current full-transport AD benchmark. Replacing the latter with a generic AD
trace would require separate correctness and memory validation.

Main source references:

- [Scan construction and its accepted keywords](https://github.com/uwplasma/NTX/blob/50445e6be679464fc84a354b56f237728f3dab59/src/ntx/_neopax_scan.py)
- [Package exports](https://github.com/uwplasma/NTX/blob/50445e6be679464fc84a354b56f237728f3dab59/src/ntx/__init__.py)
- [Prepared coefficient solvers](https://github.com/uwplasma/NTX/blob/50445e6be679464fc84a354b56f237728f3dab59/src/ntx/_solver.py)

## Other branches

The all-live-branch search tested the definitions of four critical functions
per dependency. This checks concrete interfaces, not all possible equivalent
algorithms under other names.

| Repository / branch | Result |
| --- | --- |
| VMEX `en/local_test` | Only live branch with all four old entry points: raw transpose, raw tangent, established QI/max-J surrogate, Mercier softmax. |
| VMEX other live branches | None defines those four old entry points. Main has newer block-response and physical-J APIs as described above. |
| NTX `en/custom_rule_NEOPAX` | Only live branch with both recorded-scan transpose entry points, the derivative VJP and the factorized Hessian helper. |
| NTX `rebase/en-custom-rule-neopax` (`753dcfb`) | Contains the earlier derivative VJP, but neither recorded-scan transpose entry point nor the factorized Hessian helper. It is not a complete rebasing of the currently used branch. |
| NTX `en/custom_ad_rules` (`44e5452`) | Also contains the earlier derivative VJP, but lacks the later recorded-scan routines and Hessian helper. |
| NTX `perf/adjoint-gradient-scan`, `feature/exact-window-adjoint` | Do not expose the four required entry points; their operator-adjoint work does not provide the current NEOPAX recorded-scan interface. |

A follow-up search also checked implementation markers rather than only the
old exported function names. The established constructed-J surrogate's
Goodman-transform / `jc_pow_maxj` implementation markers occur only in VMEX
`en/local_test`. NTX scan-primal-record types, recording flags and recorded-scan
implementation markers occur only in `en/custom_rule_NEOPAX`. Compact
directional-rule definitions occur in that branch plus the two earlier NTX
custom-rule branches listed above. All 74 live VMEX branches have batched
implicit-response definitions; that common capability is not exclusive to
`en/local_test` and is not evidence that they carry the old surrogate.

## Packaging and import selection

The current NEOPAX `pyproject.toml` does not declare VMEX, NTX or Boozer as
dependencies. Its geometry loader also prepends sibling VMEX/vmec_jax source
directories to `sys.path`. Installing main into an environment can therefore
still import the old sibling checkout. A main-only migration needs explicit
dependency declarations and predictable import selection.

At these main revisions:

- VMEX declares version 0.13.0; on Python >=3.11 it requires JAX/jaxlib
  >=0.9.2 and SciPy >=1.16, plus `booz_xform_jax>=0.5.1` and
  `solvax>=0.35.1`.
- NTX declares version 0.3.0, Python >=3.11 and `solvax>=0.28.2`; its VMEX
  extra requires `vmex>=0.11.10`.

Together the minimum Python version is 3.11 and the stronger VMEX dependency
floors apply. No packages or environment requirements were changed here.

## Migration order

1. Adapt NEOPAX's geometry bridge to VMEX main: use the official parameter
   constructor/device selection and the current batched block response APIs.
   Preserve the established QI/max-J surrogate explicitly if retaining the
   current benchmark's cost definition.
2. Port the required recorded-scan construction, primal-record types and
   bounded single/batched VMEC transposes onto NTX main. Include the compact
   derivative/Hessian rules needed by supported exact-Lij paths. This is a
   focused port, not a wholesale merge of the experimental branch.
3. Update NEOPAX's imports and dependency metadata; remove reliance on a
   particular sibling feature checkout. Verify resolved module paths.
4. Check constructor signatures, shapes, PyTree layouts and tiny derivative
   cases before any full physics run. Then compare a short AD/FD case and the
   full-time GPU benchmark against the reference now being collected, with
   the same objectives, inputs and solver settings.

The first work can proceed against VMEX main, but switching both dependencies
now would break the existing benchmark before it established equivalence.

## Validation performed

Parsed the dependency sources with Python's standard-library AST module;
checked imports, definitions and selected signatures across current/main and
relevant feature snapshots; searched the live branch refs for critical
definitions. Four source-contract assertions passed: the current NTX branch
accepts the two scan keywords and main rejects them, and the current VMEX
parameter schema matches NEOPAX's manual constructor while main additionally
requires `ac_aux_f`. These are static compatibility checks, not numerical
gradient tests. No physics libraries were imported by the audit scripts.

## Seven-stage GPU replay endpoint investigation (2026-11-11)

The corrected full-time main-dependency run completed with 325 accepted
steps, 20 residual rows and eight parameters; all reported derivatives were
finite. Its largest segment endpoint discrepancy was `2.186127e-10`
absolute and `1.692562e-4` after state-tolerance scaling. Those checks compare
reconstructed states against adaptive-forward checkpoints, not AD against
finite differences. Passing the state tolerance does not validate gradients.

A bounded JAX 0.10.0 GPU reproduction isolated numerical nonidentity in the
seven-stage Radau kernel. With identical input carry, stage history, Jacobian
and LU factors, the ordinary and minimal step kernels returned different
`y + h*(b @ stages)` endpoints (maximum `3.552714e-15`). Replaying the same
14 accepted steps gave nonzero endpoint differences in three of four
segments. The three-stage reproduction had matched exactly; it was
insufficient to rule out a seven-stage kernel issue.

Both step kernels now use a shared endpoint helper with optimization
barriers around its inputs, stage contraction and output. This isolates the
contraction/update from the forward error estimator's additional consumers.
It preserves the Radau weights, Newton solve, recorded schedule, controller
and accepted-step derivative rules. The barriers have identity derivatives;
they do not stop gradients. No global XLA setting is required.

In the bounded GPU reproduction the fix restored exact endpoint equality
for every segment and for the single-step forward/minimal comparison. Warm
step medians were 6.93 versus 6.91 ms (ordinary step) and 5.63 versus
5.51 ms (minimal step), within timing variation. These are component
measurements, not full transport timings. Regression tests check the
seven-stage endpoint equality and its state/time/weight/stage JVP and VJP.
The Hairer-settings GPU comparison retained exact replay equality; compiled
workspace increased by 256 bytes for both ordinary and minimal kernels.

Before this fix, the Radau/replay source matched `en/v0_new`. The old run
used JAX 0.7.2 and the old interpolation graph; the main-dependency run uses
JAX 0.10.0 and native interpolation. The reproduction demonstrates that the
new compiled context can expose different endpoint arithmetic. It does not
isolate a JAX-only versus interpolation-only trigger: at that point no controlled
old/new-stack comparison had been completed. The old run's exact matches
therefore cannot be treated as a structural guarantee for other compiled
contexts.

This fixes one reproducible component source of GPU replay nonidentity.
The remote full-time rerun after `f315bb5` completed its segmented sweep
with 317 accepted steps, but all seven segment endpoints still had nonzero
differences. The largest absolute discrepancy was `3.048228e-12`; the
largest state-tolerance-scaled discrepancy was `2.524059e-7`. The endpoint
fix is therefore incomplete for the full transport case. Different adaptive
trajectories also prevent attributing the change in discrepancy to that fix
alone. Run flags have not been identified as the cause.

With `NEOPAX_REVERSE_STEP_DIAGNOSTICS=1`, the existing endpoint check now also
reports differences in the saved stage history, Jacobian, LU factors and
associated cache metadata. These comparisons run on the host after replay,
using the original adaptive checkpoints and the existing replay result.
They introduce no new solver calls, device history or changes to either
compiled numerical kernel. Cache comparisons are diagnostic only, because
inactive cache contents need not match, and a later adaptive checkpoint can
include cache updates from intervening rejected attempts. These endpoint
cache comparisons therefore do not prove that the two kernels received the
same Newton setup. They can distinguish state-update differences from
differences in the saved stages; they do
not locate the first differing operation within a segment or establish
AD/FD agreement.

## Repeated GPU execution and the original JAX stack (2026-11-11)

A bounded saved-file fixture uses the production 51-cell, four-species,
seven-stage transport equations (408 packed variables), the existing optimized
wout/Boozer/database files and the NTSS initial electric field. It does not
solve a VMEX equilibrium, build an NTX database, search for ambipolar roots or
run the full-time simulation. A 50-accepted-step diagnostic first found stage
differences at step 9 and state differences at step 13. Checkpoint inputs were
saved for subsequent single-step tests.

Repeated calls to the **same compiled forward executable** gave:

| NEOPAX | VMEX / NTX | JAX | Input carry | Repeats | Runs with different stages | Largest stage difference | Largest endpoint difference |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `en/v0_main` | main-based utility branches | 0.10.0 | replay | 30 | 30 | 3.552713678800501e-15 | 0 |
| `en/v0_main` | main-based utility branches | 0.10.0 | forward | 30 | 30 | 3.552713678800501e-15 | 0 |
| `en/v0_main` | `en/local_test` / `en/custom_rule_NEOPAX` | 0.10.0 | replay | 30 | 27 | 3.552713678800501e-15 | 0 |
| `en/v0_main` | `en/local_test` / `en/custom_rule_NEOPAX` | 0.10.0 | forward | 30 | 21 | 3.552713678800501e-15 | 0 |
| `en/v0_new` | `en/local_test` / `en/custom_rule_NEOPAX` | 0.7.2 | replay | 100 | 0 | 0 | 0 |
| `en/v0_new` | `en/local_test` / `en/custom_rule_NEOPAX` | 0.7.2 | forward | 100 | 0 | 0 | 0 |

The proper old-stack test used a separate Python 3.12.16 environment with
JAX, jaxlib, `jax-cuda12-plugin` and `jax-cuda12-pjrt` all pinned to 0.7.2.
Module paths were checked against the three original checkouts; no branch
was switched or edited. The existing environment was retained. All numeric
tests ran on the local RTX 5060 GPU, without CPU fallback.

The two dependency-only comparisons on JAX 0.10.0 had identical numerical
support/initial-state hashes, so the newer VMEX/NTX branches are not required
to reproduce the variation. The original NEOPAX `en/v0_new` RHS on JAX 0.10.0
also varied in one of 30 batched-stage evaluations (`3.552713678800501e-15`).
The checked original stack on JAX 0.7.2 had exact forward/minimal equality in
states, stages, Jacobians and LU factors. Its final residual-norm diagnostic
differed by `1.551204080162494e-15` between those two executables; that is a
different comparison from repeated execution of one executable.

These observations support an execution-stack trigger, but by themselves do not
identify a single responsible operation or prove a JAX-only cause. The old
and new full-stack tests also differ in NEOPAX interpolation implementation,
Python version and some supporting package versions. In particular, Lineax
was 0.1.0 versus 0.1.1 and SOLVAX was 0.36.1 versus 0.35.1. The original
remote machine's complete package lock and GPU environment were not recreated.
Passing this checkpoint test is not a full-trajectory or gradient validation.

Raw repeat counts are in
`primal_replay_nondeterminism_2026-11-11.csv`. The isolated environment's full
package list is saved outside the repositories in
`D:/Github_Codes/.tmp/neopax-jax072-replay-20261111/installed_versions.txt`.

### Identified source: pairwise exchange-power scatter accumulation

Further tests on JAX 0.10.0 narrowed the repeated-executable variation to
the electron pressure RHS. The seven-stage RHS differed in 65 of 100 repeats,
by at most `3.552713678800501e-15`. Center neoclassical fluxes, density and
temperature face fluxes, and standalone temperature-source outputs were each
identical in 100 repeats. Holding all center/face flux values fixed still
reproduced variation in the assembled pressure equation.

The optimized GPU graph contained the two duplicate-index scatter additions
from `_sources._power_exchange_impl`. It folded another electron source into
the initial scatter operand, so even the active-three-species benchmark could
sum multiple nonzero terms in different orders. This explains why testing the
exchange source in isolation with helium excluded had not reproduced the
full-equation issue. JAX documents that conflicting `.at[].add` updates can
have an implementation-dependent, nondeterministic order; see the
[JAX scatter-update documentation](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html).

Replace only this accumulation with a small species/pair incidence contraction:

```text
incidence[s, pair(i,j)] = (s == i) - (s == j)
exchange_power[s, r] = sum_pair(incidence[s, pair] * pair_power[pair, r])
```

The pair powers, Coulomb logarithms, species masks, physical model and AD
paths are unchanged. There is no new mode, TOML option or required XLA flag.
The implementation uses an explicit small reduction, not a GPU matrix-library
call: the matrix version tested correctly but requested about 32 MiB of
workspace. The explicit reduction needs only 256 more workspace bytes than
the original source kernel (18,192 versus 17,936 bytes, seven stages and
51 cells). Warm source-call medians were 1.403 versus 1.409 ms with all
species active, and 1.473 versus 1.406 ms with helium excluded; these small
dispatch-inclusive measurements do not establish a full-run speedup.
Against the original source, excluding helium gave exactly identical values,
JVPs and VJPs in the isolated comparison. With all species active, maximum
differences were `5.551115123125783e-17` in values,
`1.734723475976807e-18` in JVPs and `2.5757174171303632e-14` in VJPs.
The existing endpoint barriers are retained because they fix a separate
forward/minimal arithmetic-context difference.

With this change on JAX 0.10.0:

* The ordered source was identical in 100 repeats, with and without helium
  exchange enabled. The preliminary matrix form also removed the saved-stage
  full-RHS variation in 100 repeats before the lower-workspace reduction was
  selected.
* The bounded 51-cell fixture took 50 accepted steps in 84 attempts. All 50
  replayed physical states and stage histories matched the original adaptive
  run exactly; previously the fixture first differed at stages in step 9 and
  states in step 13.
* Forward/minimal state, stages, Jacobian, LU factors, iteration count, theta
  and final Newton correction matched at the final checkpoint. Repeating the
  forward step 100 times for each input carry produced no state/stage changes.
* The separately computed final residual-norm diagnostic still differed
  between the forward/minimal executables by `2.992082511525034e-14` (norms
  about `1.5e-8`), although the states and stages were exactly identical.

These were bounded local GPU results; the remote full-time benchmark had not
yet been rerun at that point. No expensive CPU transport solve or
new VMEX equilibrium, NTX database or ambipolar-root solve was run. The old
checkouts and paused cleanup branch were left unchanged.

The GPU source suite finished with 18 passing tests, including the added
conservation/masking, finite-difference/JVP/VJP and fused-stage repeatability
checks. One existing test, `test_power_exchange_temperature_equilibration`,
still fails because the separate equilibration source broadcasts a six-pair
condition against a `(6, 10)` array. That function is unchanged from HEAD
and is not used by this benchmark; it was not modified as part of this fix.

### Full-time remote confirmation (2026-11-11)

The supplied log for
`outputs/ad_ordered_exchange_guarded_full_t20_20261111` confirms a successful
full-time run to the configured `t_final=20` with `recorded_forward` Jacobian
reuse, 50-step segments and the ordered exchange-power reduction. The process
exited with status 0. All 20 residual values and all 160 Jacobian entries
(six profile and two geometry parameters) were finite.

The adaptive run had 313 accepted steps: six 50-step segments and one 13-step
segment. At all seven segment boundaries, both the absolute and scaled
physical-state endpoint discrepancies were exactly zero. The corresponding
`prev_stages` arrays also had zero differing entries. These are the checked
segment endpoints and their saved stage histories, not a separate comparison
of every intermediate accepted state.

The cache diagnostics reported nonzero `dt`, Jacobian/LU and Newton metadata
at segments 2, 3 and 6, while physical state, previous accepted step size and
saved accepted stages remained identical. The diagnostic compares the replay
immediately after an accepted segment with the original input to the next
accepted attempt. The latter can include cache and step-size changes from
intervening rejected attempts at the same physical time. Therefore these
cache comparisons are not comparisons at the same attempt boundary and do
not establish a replay failure. They do not require differentiating rejected
attempts or changing the accepted-step derivative contract.

The preceding OOM was caused by an omitted existing command option,
`--max-reverse-accepted-steps 1000`. Without it, schedule discovery used the
TOML's `max_steps=20000` to reserve 400 full checkpoint carries. The single
real-block complex-LU checkpoint buffer required
`400 * 3 * 816 * 816 * 8 = 6,392,217,600` bytes, exactly the reported 5.95 GiB
allocation. With the 1000-accepted-step guard there are 20 reserved checkpoint
slots instead of 400; the corresponding buffer is 319,610,880 bytes. The
guard still requires reaching the full final time and raises an error if it
is exceeded. The earlier successful commands already included this option.

Recorded timings were 559.669 s for runtime construction, 182.329 s for the
segmented cotangent sweep, and 709.805 s for the final recorded NTX-scan
transpose. The API evaluation took 1774.869 s; process wall time was
39:12.67. Maximum host RSS was 10,567,256 KiB (about 10.08 GiB). The log does
not contain a peak GPU-memory measurement or a repeated-iteration memory
trace, so it does not establish optimizer memory stability.

The supplied values and derivatives are saved in
[the full gradient table](ordered_exchange_full_transport_gradients_2026-11-11.csv).
The seven endpoint checks and nonzero cache-field names are saved in
[the replay table](ordered_exchange_full_transport_replay_2026-11-11.csv).
This confirms exact accepted-primal segment replay for this full-time case;
finite-difference agreement and between-run reproducibility remain separate
checks. The number of accepted steps in a previous independent run is not a
comparison of the forward and replay counts within this run.

### Sixteen-step FD follow-up (2026-11-12)

The updated supplied log contains all six completed profile jobs: `n0`, `T0`,
`density_shape_power`, `temperature_shape_power`, `density_shape_alpha` and
`temperature_shape_alpha`. Each has 16
accepted steps in 25 attempts. Both perturbed endpoints converge at all 16
accepted slots, with zero baseline/endpoint Newton retries and finite packed
state, density, pressure, temperature and electric field. All 13 transport
objectives and their printed central differences are finite. The geometry
columns have not completed, and the new 16-step AD report has not yet been
supplied.

| Parameter | Baseline prefix final time | Process wall time |
| --- | ---: | ---: |
| n0 | 3.6692757033701093e-4 | 17:33.44 |
| T0 | 3.6693474249138833e-4 | 17:44.55 |
| density_shape_power | 3.6693474249138833e-4 | 17:34.57 |
| temperature_shape_power | 3.6693474249138833e-4 | 17:59.90 |
| density_shape_alpha | 3.6693474249138833e-4 | 17:48.77 |
| temperature_shape_alpha | 3.6693474249138833e-4 | 18:03.14 |

The T0 and density-shape-power baseline objective values match exactly as
printed. The n0 job uses a slightly different independently discovered
baseline: its prefix final time differs by about 7.17215e-9, mean Er by
6.52922e-5 and transition radius by 5.33874e-9. Therefore comparison against
the forthcoming AD run must check baseline observables and the realized
prefix, not just compare derivative columns. The logs do not identify the
cause of this between-process drift. It is a different comparison from the
already verified within-run forward/replay segment equality.

The diagnostic also reports a nonfinite `geometry.G_PS` leaf. The legacy
VmecBoozer constructor computes a fractional power of signed iota, which is
invalid for this negative-iota case; the differentiable geometry constructor
already uses absolute iota there. This is not evidence of a nonfinite
physical FD endpoint. G_PS enters the legacy NTSS radial particle/heat-flux
correction, while this benchmark's uncorrected NTX transport fluxes and the
bootstrap parallel-flow scalar do not consume that correction. The geometry
field issue is recorded without changing the model during this comparison.

The [profile FD table](ordered_exchange_16step_fd_2026-11-12.csv) stores all
78 supplied objective/parameter entries and their individual baselines.
Its derivative column is explicitly named `fd_gradient_printed`: these
numbers were printed with six digits after the decimal in scientific
notation. Full-precision error analysis should use the remote FD summary
JSON files. The earlier full-time failure at accepted step 81 is not
exercised by this successful prefix test.

#### Geometry FD setup failure and repair

The loop then stops at `RBC:1:0`, before a transport baseline or perturbed
endpoint is run. The direct call to `vmex.core.implicit._dof_mask` in the
frozen-linearized geometry bundle invokes the default force evaluator, which
reaches SOLVAX's Pallas tridiagonal backend and fails to lower on the installed
JAX 0.10.0 GPU stack. This is an unsupported code path, not evidence that CUDA
is absent: all six profile FD jobs and the full-time AD benchmark used the GPU.
`ZBS:1:0` was not reached because the loop stopped on the first failure.

For a configured `vmec_lane="ad"`, the FD bundle now obtains the equilibrium,
implicit parameters, configuration and DOF mask together from the same
`geometry_raw_block_solve_from_param_vector` helper as the AD benchmark. It
does not separately recompute the private mask with a different/default force
evaluator. The diagnostic banner now prints the actual configured lane.
The configured forward lane retains its original forward equilibrium and
mask construction. No VMEX, NTX, SOLVAX, environment requirement or transport
replay rule is changed.

The frozen geometry endpoints still use the existing implicit VMEX tangent;
their transport derivatives remain forward central differences. This does
not independently validate VMEX's implicit tangent. No completed profile FD
column needs repeating for this setup-only repair. Remote geometry FD results
remain pending.

Local validation passed 79 tests and 48 subtests in 28.03 s across the
new geometry-bundle routing regressions, frozen-metadata FD replay tests,
benchmark plumbing and objective-routing tests. The new tests cover both RBC
and ZBS, forbid the unsupported private mask evaluator in the AD lane, check
the reused primal/mask/configuration and unit parameter tangent, and preserve
the forward-reference lane. Two existing parser-test lookups were updated to
read declarations from the already-refactored `build_argument_parser` rather
than `main`. These checks use mocked solves and tiny scalar ODEs; no local
VMEX equilibrium, NTX database, plasma transport or FD benchmark was run.
