# Current NTX derivative diagnosis

The original FD capture `outputs/original_fd_ntx_capture_lq73VM` was received
and inspected with NumPy and compiler-text comparisons only. No local JAX
import, equilibrium, coefficient, root or transport solve ran.

## What the original capture establishes

Its 191 saved upstream geometry/channel/native-surface fields match both the
successful and failed archived NTSS1D builds. All 44 geometry/database fields
match the successful archived build. Only the three coefficient tables differ
from the failed archive, with the previously reported gaps. The old failure
did not recur; this is not a correction or a gradient validation.

At surface 4, all 48 prepared leaves, all 880 raw coefficients, and their
types/weakness/device commitments match the standalone replay exactly. Shared
NTX/SOLVAX sources are byte-identical, including `_neopax_ad.py`, `_geometry.py`
and SOLVAX's `direct.py` and `_reproducible.py`.

The actual initialization compiled 70 NTX preparation programs and one scan
program; subsequent surfaces reused them. Of 69 shared StableHLO programs,
68 have identical arithmetic/backend text after removing debug call stacks.
The scan differs only in unroll factors 2 versus 4 for two transpose fusions.
Its values are identical. Those observed compiler differences do not explain
the failed table or the geometry-gradient gap.

## The remaining generic geometry-gradient gap

The current generic AD16/FD16 snapshots allow a finer split using their
actual objective cotangents and geometry/database directions:

| Cost / parameter | Direct geometry gap / abs(full FD) | Database gap / abs(full FD) | Full AD/FD relative difference |
| --- | ---: | ---: | ---: |
| softmax Er / RBC | `2.833763e-09` | `8.376744e-06` | `8.381627e-06` |
| bootstrap current / RBC | `1.636478e-07` | `4.647737e-03` | `4.647559e-03` |
| softmax Er / ZBS | `1.012569e-07` | `1.641751e-05` | `1.650857e-05` |
| bootstrap current / ZBS | `4.539078e-08` | `5.635590e-04` | `5.635140e-04` |

Each component gap is its absolute AD-minus-FD difference divided by the
absolute **full** FD derivative. These are contributions to the full error,
not relative errors using each component as its denominator; cancellation
and the support-chain remainder prevent exact addition. The
[26-row CSV](fixed_projection_generic_table_gradient_split_2026-11-25.csv)
retains signed contributions and both gradients.

For example, the RBC bootstrap direct-geometry gap is `-1.957052e-08`, while
the database gap is `5.558196e-04`. The recorded support adjoint applied to the
saved FD support direction still reproduces the full FD result to
`1.455171e-08` relative. This localizes the dominant discrepancy to the
native-input-to-database derivative or its FD secant. It does not by itself
decide whether the NTX custom derivative or the FD coefficient calculation
is inaccurate.

## Specific source fork to test

NTX's ordinary raw solve calls `_solve_modes_with_tail_residual`, retaining
the lowest three modes after full-tail elimination. Its production custom
pullback calls `_prepared_implicit_vjp_primal`, which recomputes a full
block-Thomas factorization and full fields. Both represent the same discrete
system. Their primal and derivative agreement at the relevant operating
points must be measured before changing either implementation.

The existing selected-node checker now has `--jvp-vjp-only`. It uses all 13
captured transport-objective RHS rows, converting their D11_log/D13/D33 bars
through the actual raw-coefficient normalization. It compares the raw-solve
JVP with the existing production multi-RHS VJP and records both primals.
It runs no FD endpoints or perturbation sweep. Default FD diagnostic scales
remain unchanged. The 10 checker tests and 22 baseline-wrapper tests pass
locally with NumPy/mocks only. GPU selected-node results are recorded below.

The selected nodes are `(3,4,3)`, a large D11 contribution to softmax Er, and
`(4,4,2)`, a large D13/D33 contribution to bootstrap current in the actual
saved results. They are checked for both RBC and ZBS directions. This tests
local contributions of the actual AD16 objective rows, not a full trajectory
or all database nodes. Passing it cannot establish full AD/FD agreement.

Publish from local WSL:

```bash
(
  cd /mnt/d/Github_Codes/NEOPAX-v0_main || exit 1
  git switch en/v0_main || exit 1
  neopax_files=(
    examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py
    examples/benchmarks/check_saved_transport_baseline_repeatability.py
    examples/benchmarks/check_captured_ntx_directional_derivative.py
    tests/test_saved_transport_baseline_repeatability.py
    tests/test_captured_ntx_directional_derivative.py
    docs/benchmarks/ntss1d_transport_ad_fd.md
    docs/benchmarks/current_ntx_derivative_diagnosis_2026-11-25.md
    docs/benchmarks/fixed_projection_generic_table_gradient_split_2026-11-25.csv
  )
  git add -- "${neopax_files[@]}" || exit 1
  git commit --only -m "Check NTX forward and reverse derivatives without FD perturbations" \
    -- "${neopax_files[@]}" || exit 1
  git push origin en/v0_main
) || printf '\nPublish stopped; terminal kept open.\n'
```

Then on the large machine:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git switch en/v0_main || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  neopax_case=outputs/fixed_projection_ad_fd16_gu6ZO2/black_box
  neopax_out=$(mktemp -d outputs/ntx_derivative_probe_XXXXXX) || exit 1
  set -o pipefail
  for neopax_parameter in RBC:1:0 ZBS:1:0; do
    neopax_safe=${neopax_parameter//:/_}
    env -u PYTHONPATH -u LD_LIBRARY_PATH -u JAX_PLATFORM_NAME \
      -u XLA_FLAGS -u JAX_COMPILATION_CACHE_DIR -u XLA_PYTHON_CLIENT_ALLOCATOR \
      -u NEOPAX_GRADIENT_AUDIT_DIR -u NEOPAX_ORIGINAL_FORWARD_DIR \
      -u NEOPAX_ORIGINAL_FORWARD_STOP \
      JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true JAX_ENABLE_COMPILATION_CACHE=0 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false NTX_DISABLE_COMPILATION_CACHE=1 \
      VMEX_COMPILATION_CACHE=disabled \
      "$HOME/jax_env/bin/python" \
      examples/benchmarks/check_captured_ntx_directional_derivative.py \
      --run-dir "$neopax_case" --parameter "$neopax_parameter" \
      --points 3:4:3,4:4:2 --jvp-vjp-only \
      --out-dir "$neopax_out/$neopax_safe" \
      2>&1 | tee "$neopax_out/$neopax_safe.log" || exit 1
  done
  tar -czf outputs/ntx_derivative_probe.tar.gz -C "$neopax_out" . || exit 1
  printf 'Saved: %s\n' "$neopax_out"
) || printf '\nNTX derivative probe stopped; terminal kept open.\n'
```

The NPZs retain all five ordinary/JVP/pullback primal coefficients, raw JVPs,
the 13 coefficient-bar rows and their reverse contractions. CSV `primal`
values in this mode are cotangent-weighted raw coefficients, not transport
objective values. `gradient_fd` and `fd_step` are empty because no perturbation
is evaluated. No numerical method, tolerance or production equality guard is
changed. The existing preflight rejects a reconstructed coefficient primal
that fails its saved-table comparison before comparing derivatives.

## Selected-node GPU result

`outputs/ntx_derivative_probe_Zs0EWC` at `83c2684` completed 52 comparisons:
two nodes, two geometry directions and all 13 actual objective RHS rows.
All printed weighted JVP-primal checks had zero difference from the ordinary
primal. JVP and recomputed production VJP agree as follows:

| Parameter | Node (radius,nu,Er) | Largest objective relative difference | Softmax Er relative difference | Bootstrap relative difference |
| --- | --- | ---: | ---: | ---: |
| RBC | `(3,4,3)` | `3.226815e-10` | `1.374727e-12` | `3.226815e-10` |
| RBC | `(4,4,2)` | `2.608602e-10` | `4.294757e-12` | `2.608602e-10` |
| ZBS | `(3,4,3)` | `2.978904e-11` | `5.563199e-12` | `2.978904e-11` |
| ZBS | `(4,4,2)` | `7.839271e-11` | `3.834935e-12` | `7.839271e-11` |

These differences are `abs(VJP-JVP)/abs(JVP)` for the local weighted
coefficient contribution. The [52-row CSV](captured_ntx_derivative_probe_2026-11-25.csv)
retains the full printed values. No FD perturbations were evaluated.

The measured local differences are too small to explain the full gradient
gap by themselves. They do not establish a large inconsistency between the
two elimination paths at these nodes, so no solver change is justified by
this result. Neither accumulation over other nodes nor the original recorded
scan fold has yet been checked independently. Recomputing a local VJP is not
the same as verifying the entire saved AD16 table transpose.

## Complete weighted table audit

The saved generic objective bars have 761 active `(radius,nu,Er)` nodes out
of 1,232. Their counts by radius are `93,100,103,105,124,121,115`. A NumPy-only
preflight confirms byte equality of all 47 required baseline fields for
both saved geometry FD columns: initial/final state, time and support.

The existing `--audit-table --compare-saved-fd` mode independently
differentiates the raw forward coefficient solves at every active node. It
then normalizes and sums their JVPs using the original AD16 objective bars,
comparing against both the **saved** recorded-scan VJP and the **saved** FD
table directions. It performs no new FD endpoints, equilibrium, root or
transport solve. It does not recompute the table VJP. This checks the full
weighted accumulation that the two-node probe cannot establish.

This mode already exists in the published checker; no new code push is
needed to run it. On the large machine:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  neopax_case=outputs/fixed_projection_ad_fd16_gu6ZO2/black_box
  neopax_out=$(mktemp -d outputs/ntx_full_table_jvp_XXXXXX) || exit 1
  set -o pipefail
  for neopax_parameter in RBC:1:0 ZBS:1:0; do
    neopax_safe=${neopax_parameter//:/_}
    env -u PYTHONPATH -u LD_LIBRARY_PATH -u JAX_PLATFORM_NAME \
      -u XLA_FLAGS -u JAX_COMPILATION_CACHE_DIR -u XLA_PYTHON_CLIENT_ALLOCATOR \
      -u NEOPAX_GRADIENT_AUDIT_DIR -u NEOPAX_ORIGINAL_FORWARD_DIR \
      -u NEOPAX_ORIGINAL_FORWARD_STOP \
      JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true JAX_ENABLE_COMPILATION_CACHE=0 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false NTX_DISABLE_COMPILATION_CACHE=1 \
      VMEX_COMPILATION_CACHE=disabled \
      "$HOME/jax_env/bin/python" \
      examples/benchmarks/check_captured_ntx_directional_derivative.py \
      --run-dir "$neopax_case" --parameter "$neopax_parameter" \
      --audit-table --compare-saved-fd \
      --out-dir "$neopax_out/$neopax_safe" \
      2>&1 | tee "$neopax_out/$neopax_safe.log" || exit 1
  done
  tar -czf outputs/ntx_full_table_jvp.tar.gz -C "$neopax_out" . || exit 1
  printf 'Saved: %s\n' "$neopax_out"
) || printf '\nFull-table derivative audit stopped; terminal kept open.\n'
```

Interpretation must include the saved ordinary/JVP primal offsets. If those
controls are consistent and the complete JVP agrees with the recorded VJP
while differing from the saved FD direction, the remaining discrepancy is
in that FD secant. If it disagrees with the recorded VJP, the radius and
component decompositions locate the missing table-transpose contribution.
No conclusion about all gradients follows before that full comparison.

## Complete weighted audit result

`outputs/ntx_full_table_jvp_ThvCQo` completed 761 active cases in each geometry
direction, producing 26 weighted objective comparisons. All reported saved
baseline checks passed. No coefficient VJP, new FD endpoint, root, equilibrium
or transport trajectory was run. The
[26-row full-table CSV](captured_ntx_full_table_jvp_2026-11-25.csv) retains all
printed values and the following common normalization:

| Cost / direction | Independent table JVP vs saved table VJP / abs(full FD) | Independent table JVP vs saved table FD / abs(full FD) |
| --- | ---: | ---: |
| softmax Er / RBC | `2.237724e-08` | `8.354367e-06` |
| bootstrap current / RBC | `4.138016e-06` | `4.643599e-03` |
| softmax Er / ZBS | `6.604238e-09` | `1.641091e-05` |
| bootstrap current / ZBS | `8.123035e-07` | `5.643713e-04` |

The small JVP/VJP residual is nonzero and larger than at the two sampled
nodes, but remains much smaller than the dominant FD gap. The full weighted
accumulation therefore does not reproduce the large reverse-mode error
previously suspected. The dominant mismatch follows the saved FD database
directions; this does not yet establish whether its cause is endpoint
construction, finite-step nonlinearity, or coefficient-solve roundoff.

The ordinary and JVP reconstructions have identical reported weighted primal
offsets from the captured tables. These are small but nonzero; the bootstrap
offset is `-5.962655479318e-10`, and the softmax-Er offset is
`-1.996967820352e-11`. They are not a strict per-node error bound. Dividing one
reconstruction offset by the FD step is only a sensitivity scale, not a
measurement of the actual plus/minus FD error.

A NumPy-only readout of the existing FD endpoints confirms no D11 floor-mask
switches for either sign or geometry parameter. Their weighted bootstrap
table slopes are:

| Direction | Positive one-sided slope | Negative one-sided slope | Central slope |
| --- | ---: | ---: | ---: |
| RBC | `6.848339278744e-01` | `6.732541542956e-01` | `6.790440410850e-01` |
| ZBS | `-3.580245310950e+00` | `-3.588535229586e+00` | `-3.584390270268e+00` |

This shows endpoint asymmetry at the original perturbations; it does not
distinguish smooth curvature from numerical error by itself. Saved FD metadata
specifies extended native inputs, frozen-linearized geometry and initial-Er
root lanes, and the reverse-frozen accepted-step replay contract. No step size
or numerical method has been changed during this analysis.

## Per-node localization from the complete archive

The complete archive was read with NumPy only. All 19 saved baseline database
fields agree exactly with the original AD capture. The 761 ordinary and
JVP primal coefficient vectors agree bit-for-bit, and the independently
reconstructed RBC and ZBS ordinary baselines also agree bit-for-bit.

The following signed gaps are **independent table JVP minus saved table FD**.
They are absolute derivative contributions, not relative errors:

| Direction / cost | D11_log contribution | D13 contribution | D33 contribution |
| --- | ---: | ---: | ---: |
| RBC / softmax Er | `2.185102e-04` | `0.000000e+00` | `0.000000e+00` |
| ZBS / softmax Er | `-9.644616e-05` | `0.000000e+00` | `0.000000e+00` |
| RBC / bootstrap | `4.621555e-06` | `5.675182e-04` | `-1.681753e-05` |
| ZBS / bootstrap | `9.038185e-06` | `-1.847848e-03` | `3.012888e-05` |

The Er discrepancy follows D11, whereas the bootstrap discrepancy is dominated
by D13. The largest nodes are now at `nu_index=1`, corresponding to
`nu_v=1.0e-6`, rather than the moderate-collisionality `nu_index=4` nodes
used in the earlier two-node derivative probe.

| Direction / cost | Field / node `(radius,nu,Er)` | Independent normalized JVP | Saved normalized FD | Weighted JVP minus FD |
| --- | --- | ---: | ---: | ---: |
| RBC / softmax Er | D11_log / `(3,1,6)` | `-1.123294e+00` | `-1.126429e+00` | `1.271928e-04` |
| ZBS / softmax Er | D11_log / `(3,1,6)` | `1.454973e+00` | `1.457701e+00` | `-1.106890e-04` |
| RBC / bootstrap | D13 / `(6,1,7)` | `1.478986e+02` | `1.566553e+02` | `1.596136e-03` |
| ZBS / bootstrap | D13 / `(6,1,7)` | `1.273567e+02` | `1.165331e+02` | `-1.972860e-03` |

Individual contributions can exceed the total because other nodes cancel
part of the difference. `(5,1,6)` is also a leading entry for both costs and
directions. The saved-array audit therefore identifies three focused control
points: `(3,1,6)`, `(5,1,6)` and `(6,1,7)`.

The ordinary raw reconstruction and the actual captured production scan are
not universally bit-identical. At the two leading nodes `(3,1,6)` and
`(6,1,7)` the corresponding raw D11 and D13 primals do match exactly. At
`(5,1,6)`, the relative raw D11 and D13 primal differences are
`-6.667919e-09` and `-2.388774e-06`, respectively. This does not measure the
error of either FD endpoint. It prevents interpreting the small summed
primal offset as proof of uniform per-node numerical agreement.

These results locate the dominant discrepancy in the perturbed coefficient
table evaluation, upstream of fixed-table transport differentiation. They do
not yet distinguish finite-step curvature, input rounding and coefficient
solve roundoff. No solver change is justified by this localization alone.

The existing numerical-input FD control can now run at **only the original
saved perturbation**, with `--precision-step-scales 1`. It evaluates the
baseline and two endpoints at these three numerical input points in each
direction, using the existing NTX forward solver and the completed JVP as
reference. It performs no new equilibrium, implicit response, root, transport
trajectory or full coefficient table. This is a control of the numerical
inputs after geometry conversion, not a replacement geometry FD benchmark.
The four-scale default is preserved for existing invocations.

The original-step-only control passed the existing ten NumPy/mocked API
tests locally. Its analytic test verifies identical scale-one results,
exactly three forward evaluations, and the original `h0` without any
perturbation-size adjustment. CUDA results for these newly selected points
remain pending.

Publish from the local WSL checkout (not the large machine):

```bash
(
  cd /mnt/d/Github_Codes/NEOPAX-v0_main || exit 1
  neopax_files=(
    examples/benchmarks/check_captured_ntx_directional_derivative.py
    tests/test_captured_ntx_directional_derivative.py
    docs/benchmarks/current_ntx_derivative_diagnosis_2026-11-25.md
    docs/benchmarks/captured_ntx_derivative_probe_2026-11-25.csv
    docs/benchmarks/captured_ntx_full_table_jvp_2026-11-25.csv
  )
  git add -- "${neopax_files[@]}" || exit 1
  git commit --only -m "Localize NTX table gaps at the original FD step" \
    -- "${neopax_files[@]}" || exit 1
  git push origin en/v0_main
) || printf '\nPublish stopped; terminal kept open.\n'
```

Then run only the six point/direction combinations on the large machine:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git switch en/v0_main || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  set -o pipefail
  neopax_out=$(mktemp -d outputs/ntx_original_step_nodes_XXXXXX) || exit 1
  for neopax_parameter in RBC:1:0 ZBS:1:0; do
    neopax_safe=${neopax_parameter//:/_}
    env -u PYTHONPATH -u LD_LIBRARY_PATH -u JAX_PLATFORM_NAME \
      -u XLA_FLAGS -u JAX_COMPILATION_CACHE_DIR -u XLA_PYTHON_CLIENT_ALLOCATOR \
      -u NEOPAX_GRADIENT_AUDIT_DIR -u NEOPAX_ORIGINAL_FORWARD_DIR \
      -u NEOPAX_ORIGINAL_FORWARD_STOP \
      JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true JAX_ENABLE_COMPILATION_CACHE=0 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false NTX_DISABLE_COMPILATION_CACHE=1 \
      VMEX_COMPILATION_CACHE=disabled \
      "$HOME/jax_env/bin/python" \
      examples/benchmarks/check_captured_ntx_directional_derivative.py \
      --run-dir outputs/fixed_projection_ad_fd16_gu6ZO2/black_box \
      --parameter "$neopax_parameter" \
      --points 3:1:6,5:1:6,6:1:7 \
      --precision-from-table-audit "outputs/ntx_full_table_jvp_ThvCQo/$neopax_safe" \
      --precision-step-scales 1 \
      --out-dir "$neopax_out/$neopax_safe" \
      2>&1 | tee "$neopax_out/$neopax_safe.log" || exit 1
  done
  tar -czf outputs/ntx_original_step_nodes.tar.gz -C "$neopax_out" . || exit 1
  printf 'Saved: %s\n' "$neopax_out"
) || printf '\nOriginal-step coefficient check stopped; terminal kept open.\n'
```
