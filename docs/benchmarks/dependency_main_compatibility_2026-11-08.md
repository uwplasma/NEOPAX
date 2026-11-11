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
