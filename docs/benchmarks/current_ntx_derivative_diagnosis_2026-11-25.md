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

## Direct numerical-input FD result at the original perturbation

`outputs/ntx_original_step_nodes_smlYWZ` completed all six point/direction
combinations at `94008df`. The supplied archive was inspected with NumPy
only. Both FD steps equal their original saved values exactly; all 18 raw
baseline coefficient checks are bit-identical to the complete table audit.

The [18-row comparison](ntx_original_step_nodes_2026-11-25.csv) retains the
raw and normalized derivatives, the original geometry-FD direction and the
new affine numerical-input FD. The latter perturbs every numerical surface
input plus the electric-field and radial-normalization channels directly.
Geometry conversion, root selection and transport replay are outside this
coefficient calculation.

| Direction | Field / `(radius,nu,Er)` | Saved JVP | Direct numerical-input FD | Relative difference from JVP |
| --- | --- | ---: | ---: | ---: |
| RBC | D11_log / `(3,1,6)` | `-1.123294e+00` | `-1.127111e+00` | `3.397588e-03` |
| RBC | D11_log / `(5,1,6)` | `5.804422e-02` | `5.679761e-02` | `2.147676e-02` |
| RBC | D13 / `(6,1,7)` | `1.478986e+02` | `1.433183e+02` | `3.096931e-02` |
| ZBS | D11_log / `(3,1,6)` | `1.454973e+00` | `1.456197e+00` | `8.413315e-04` |
| ZBS | D11_log / `(5,1,6)` | `1.523587e+00` | `1.444534e+00` | `5.188624e-02` |
| ZBS | D13 / `(5,1,6)` | `-5.611849e-01` | `-4.482131e-01` | `2.013093e-01` |
| ZBS | D13 / `(6,1,7)` | `1.273567e+02` | `1.213744e+02` | `4.697278e-02` |

The raw D13 derivatives also disagree, before radial normalization: their
relative differences at `(6,1,7)` are `3.095690e-02` (RBC) and `4.695239e-02`
(ZBS). Normalization alone therefore cannot explain the discrepancy.
D11 at this node is floor-active at both endpoints; its zero normalized
derivative is a consequence of that floor, not an accuracy certificate.
No floor-mask change was observed at the selected points.

This result reproduces a coefficient-level FD/JVP discrepancy without the
original geometry-to-native conversion or a transport trajectory. It does
not establish which of finite-step curvature, rounded input displacement,
operator/solve arithmetic, or the derivative implementation explains it.
No numerical-method change follows from this test.

## Fixed-step endpoint-derivative control

The optional `--precision-endpoint-jvps` diagnostic keeps the original
coefficient solver and perturbation. At each existing point it records:

- JVPs at the center and the same two endpoints, along the same direction.
- A center JVP along the actual rounded input secant `(x_plus-x_minus)/(2h)`.
- A secant from the primal outputs of that JVP executable, compared with the
  separately compiled ordinary forward outputs already evaluated.
- The Simpson average `(JVP_minus+4*JVP_center+JVP_plus)/6`, as a smooth-curve
  diagnostic. This is not a rigorous quadrature-error bound.

These controls test different explanations at a single unchanged step. An
endpoint derivative variation can indicate curvature; a rounded-direction
response can indicate input-displacement sensitivity; a different primal
secant from the JVP executable can expose compilation-context sensitivity.
Agreement of those derivatives alongside a discrepant FD would support
numerical error in coefficient evaluation, while an inconsistent center JVP
would require examining the AD implementation. The values and numerical
inputs are saved in `curve_*.npz`; no full table or transport solve is run.

Eleven local NumPy/mocked API tests pass. They include a cubic with a known
finite-step curvature contribution, a sub-ULP input displacement and primal
output cancellation, plus the complete capture path on an analytic model.
No local JAX import or physical solve was used. The actual CUDA endpoint
derivative results remain pending.

Publish from local WSL:

```bash
(
  cd /mnt/d/Github_Codes/NEOPAX-v0_main || exit 1
  git switch en/v0_main || exit 1
  neopax_files=(
    examples/benchmarks/check_captured_ntx_directional_derivative.py
    tests/test_captured_ntx_directional_derivative.py
    docs/benchmarks/current_ntx_derivative_diagnosis_2026-11-25.md
    docs/benchmarks/ntx_original_step_nodes_2026-11-25.csv
  )
  git add -- "${neopax_files[@]}" || exit 1
  git commit --only -m "Audit NTX endpoint derivatives at the unchanged FD step" \
    -- "${neopax_files[@]}" || exit 1
  git push origin en/v0_main
) || printf '\nPublish stopped; terminal kept open.\n'
```

Then run the endpoint controls on the large machine:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git switch en/v0_main || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  set -o pipefail
  neopax_out=$(mktemp -d outputs/ntx_endpoint_jvps_XXXXXX) || exit 1
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
      --precision-step-scales 1 --precision-endpoint-jvps \
      --out-dir "$neopax_out/$neopax_safe" \
      2>&1 | tee "$neopax_out/$neopax_safe.log" || exit 1
  done
  tar -czf outputs/ntx_endpoint_jvps.tar.gz -C "$neopax_out" . || exit 1
  printf 'Saved: %s\n' "$neopax_out"
) || printf '\nEndpoint derivative check stopped; terminal kept open.\n'
```

## Partial endpoint-control result and strict baseline stop

At `d279e29`, the RBC worker completed `(3,1,6)` and then stopped because
its ordinary raw baseline at `(5,1,6)` failed the existing `rtol=1e-8`,
`atol=1e-12` reference check. ZBS was never started and no final archive
was created by the shell loop. The completed point's CSV and `curve_*.npz`
were saved before the exception. The old error did not report the failed
coefficient values or preserve that failed point's numerical inputs.

The [three printed rows](ntx_endpoint_jvps_partial_2026-11-25.csv) give these
relative differences, all normalized by the magnitude of the center JVP:

| Field at RBC `(3,1,6)` | Maximum sampled endpoint JVP variation | Rounded-input JVP minus center | Simpson JVP minus center | JVP-executable primal FD minus center |
| --- | ---: | ---: | ---: | ---: |
| D11_log | `4.290721e-05` | `4.401765e-06` | `7.809538e-07` | `8.768693e-03` |
| D13 | `8.184102e-05` | `2.239268e-05` | `1.847505e-07` | `2.942359e-02` |
| D33 | `3.099355e-05` | `5.331185e-07` | `4.983101e-07` | `4.697381e-04` |

The center JVPs reproduce the full-table reference to printed precision.
At D13, the endpoint derivatives and rounded-input response do not exhibit
changes comparable with the FD discrepancy. This strongly supports numerical
error in coefficient evaluation rather than an ordinary smooth-curvature
explanation at this point. Simpson's three-sample average is not a rigorous
error bound, and this result does not identify a particular operator-build,
factorization, substitution or coefficient-reduction operation as the cause.
It also does not validate the failed point or unrun ZBS controls.

The separately compiled ordinary FD reports the same absolute relative
discrepancy to the printed precision. The signed comparison and exact input
bytes require the already-saved partial NPZ/CSV. No rerun is required to
retrieve those files. From local WSL:

```bash
scp -r 'exouser@149.165.170.203:/home/exouser/NEOPAX/outputs/ntx_endpoint_jvps_*' \
  /mnt/d/Github_Codes/
```

The diagnostic now preserves the actual/reference raw coefficients, numerical
surface/electric-field values and direction, and collisionality in
`primal_mismatch_*.npz` before raising the same baseline error. Its tolerance
and stop behavior are unchanged. Twelve NumPy/mocked tests pass, including
an explicit test that a mismatched baseline is saved and still rejected.
This is a diagnostic capture correction, not a numerical-method correction;
no new CUDA run has been performed for it.

## Copied endpoint arrays and interpolation-path verification

The copied partial directory is `ntx_endpoint_jvps_LnXAeu`. It contains only
RBC `(3,1,6)`; no failed-point or ZBS arrays are present. NumPy-only checks
against the saved full benchmark establish the following; the
[array-derived CSV](ntx_endpoint_arrays_2026-11-25.csv) retains the numerical
comparisons independently of the earlier rounded stdout values:

- All 20 numerical baseline inputs match the original scan references
  byte-for-byte, including the electric field, seven active native fields,
  geometry metadata and radial normalization. The seven active surface
  directions and the radial-normalization direction match their saved AD
  directions byte-for-byte.
- Ordinary and JVP-executable primal outputs match byte-for-byte at both
  endpoints, for all three observed coefficients. Their FD derivatives
  therefore agree exactly within this execution.
- Extended-precision subtraction/division changes the D13 FD by only
  `2.530781e-17` relative to the center JVP. Recomputing normalization and
  differencing in extended precision changes it by `8.036726e-12`, versus
  the actual discrepancy `2.942359e-02`.
- The discrepancy is therefore present in the raw NTX coefficient values
  before final normalization/differencing. This comparison does not identify
  the geometry-preparation, elimination or coefficient-averaging operation
  responsible for their numerical sensitivity.

This is a **generic interpolation** reference: the checker uses
`fixed_projection_ad_fd16_gu6ZO2/black_box`. The generic full comparison
completed with bootstrap relative differences `4.647559e-03` (RBC) and
`5.635140e-04` (ZBS); completion and exact unperturbed baselines did not
mean that its geometry derivatives passed the desired accuracy target.

A separate NumPy/source verification confirms that the two saved TOMLs
differ only by `neoclassical.interpolation_mode`: absent/default `generic`
versus `preprocessed_3d_radial_ntss1d`. Their saved successful `n0` baselines
have 25 byte-identical geometry fields, 19 byte-identical database fields
and 166 byte-identical scan-input fields. These are corresponding baselines,
not a comparison of gradients belonging to different interpolators.

`NTXRuntimeScanTransportModel._build_runtime_database_and_record` calls
`ntx.build_ntx_neopax_scan_from_surfaces` before applying the interpolation
choice in `_ntx_runtime_scan_to_neopax_monoenergetic`. The raw scan and the
standalone coefficient test both reach
`ntx._solver._solve_prepared_coefficient_vector_raw`, then the same SOLVAX
selected-head solver. The four fixed-order Schur products therefore apply
to both interpolation cases. Recorded source hashes for `ntx._solver`,
`ntx._solver_scan`, `solvax.direct` and `solvax._reproducible` match the local
patched sources. No unpatched alternate NTX route was found for NTSS1D.

## Observation of the common forward stages

The optional `--capture-forward-stages` mode observes one point's ordinary
forward baseline and endpoints using the same NTX public forward solver.
It records the actual pre-LU head matrices, retained LU/pivots/eliminated
sources/lower blocks, retained modes and prepared geometry. It wraps the
existing solver calls for observation and restores them afterwards; it adds
no alternate solver, iteration, tolerance or physical approximation.

The ordinary coefficient results are retained as controls. Returning extra
data and adding callbacks can affect compilation layout, so observed versus
ordinary coefficient equality is printed and saved, rather than assumed.
Only the already-diagnosed point `(3,1,6)` is needed initially. These arrays
will allow coefficient averaging to be recomputed independently and the
actual head matrices/factors to be checked without an alternative solve.

Fifteen local tests pass: twelve coefficient-diagnostic tests and three new
NumPy/mocked observation tests covering exact captured arrays, restoration
on exceptions, phase labels, and explicit retention of any observed primal
drift. No local JAX import, GPU run or physical solve was performed.

Publish these diagnostic/evidence files from local WSL:

```bash
(
  cd /mnt/d/Github_Codes/NEOPAX-v0_main || exit 1
  git switch en/v0_main || exit 1
  neopax_files=(
    examples/benchmarks/check_captured_ntx_directional_derivative.py
    examples/benchmarks/_ntx_forward_stages.py
    tests/test_captured_ntx_directional_derivative.py
    tests/test_ntx_forward_stages.py
    docs/benchmarks/current_ntx_derivative_diagnosis_2026-11-25.md
    docs/benchmarks/ntx_endpoint_jvps_partial_2026-11-25.csv
    docs/benchmarks/ntx_endpoint_arrays_2026-11-25.csv
  )
  git add -- "${neopax_files[@]}" || exit 1
  git commit --only -m "Observe NTX forward stages at the unchanged FD step" \
    -- "${neopax_files[@]}" || exit 1
  git push origin en/v0_main
) || printf '\nPublish stopped; terminal kept open.\n'
```

Run only RBC `(3,1,6)` on the large machine. This captures the original-step
baseline and endpoints; it performs no equilibrium, root or transport solve.
The archive is produced even if the diagnostic stops, to preserve its partial
results and failure evidence:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git switch en/v0_main || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  set -o pipefail
  neopax_out=$(mktemp -d outputs/ntx_forward_stages_XXXXXX) || exit 1
  printf 'Results: %s\n' "$neopax_out"

  if env -u PYTHONPATH -u LD_LIBRARY_PATH -u JAX_PLATFORM_NAME \
    -u XLA_FLAGS -u JAX_COMPILATION_CACHE_DIR -u XLA_PYTHON_CLIENT_ALLOCATOR \
    -u NEOPAX_GRADIENT_AUDIT_DIR -u NEOPAX_ORIGINAL_FORWARD_DIR \
    -u NEOPAX_ORIGINAL_FORWARD_STOP \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true JAX_ENABLE_COMPILATION_CACHE=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false NTX_DISABLE_COMPILATION_CACHE=1 \
    VMEX_COMPILATION_CACHE=disabled \
    "$HOME/jax_env/bin/python" \
    examples/benchmarks/check_captured_ntx_directional_derivative.py \
    --run-dir outputs/fixed_projection_ad_fd16_gu6ZO2/black_box \
    --parameter RBC:1:0 --points 3:1:6 \
    --precision-from-table-audit outputs/ntx_full_table_jvp_ThvCQo/RBC_1_0 \
    --precision-step-scales 1 --capture-forward-stages \
    --out-dir "$neopax_out/data" \
    2>&1 | tee "$neopax_out/run.log"
  then
    neopax_status=0
  else
    neopax_status=$?
  fi

  tar -czf outputs/ntx_forward_stages.tar.gz -C "$neopax_out" . || exit 1
  printf 'Diagnostic exit status: %s\nArchive: %s\n' \
    "$neopax_status" "$HOME/NEOPAX/outputs/ntx_forward_stages.tar.gz"
  exit "$neopax_status"
) || printf '\nDiagnostic stopped; saved archive retained; terminal kept open.\n'
```

Retrieve the archive from local WSL:

```bash
scp exouser@149.165.170.203:/home/exouser/NEOPAX/outputs/ntx_forward_stages.tar.gz \
  /mnt/d/Github_Codes/
```

## Forward-stage capture result

`outputs/ntx_forward_stages_klS68a` at `526f8b7` completed successfully.
All five observed raw coefficients match ordinary execution bit-for-bit at
the baseline and both endpoints. The archive includes nine actual pre-LU
matrices and the retained factors, sources, lower blocks, modes and fields.
All four recorded NTX/SOLVAX source hashes match the patched local sources.
The archive was analyzed with NumPy only, without a new factorization,
equilibrium, NTX primal or transport calculation.

The captured native numerical inputs are byte-identical to the previous
endpoint-control run in every phase. Nevertheless the minus-endpoint D13
coefficient differs between those two compilations by `5.521608e-09`; their
baseline and plus coefficients agree. This is evidence of sensitivity across
compilations with identical recorded numerical inputs, not a demonstration
that one executable returns different results on repeated identical calls.

The [postprocessing table](ntx_forward_stages_2026-11-25_postprocessing.csv)
recomputes the unchanged NTX coefficient formula on the captured modes in
host extended precision. At D13, its difference from the GPU coefficient is
`4.031792e-17` at baseline, `9.021579e-17` at minus and `5.634802e-17` at plus.
Thus coefficient averaging cannot explain the main FD discrepancy:

| Raw coefficient | Forward JVP reference | Saved raw FD | FD minus JVP | Change from extended coefficient averaging |
| --- | ---: | ---: | ---: | ---: |
| D11 | `-6.747154e-06` | `-6.762969e-06` | `-1.581540e-08` | `1.006491e-14` |
| D13 | `4.075900e-01` | `4.110026e-01` | `3.412649e-03` | `5.111164e-11` |
| D33 | `4.428216e+05` | `4.424464e+05` | `-3.752254e+02` | `1.464786e-05` |

The [actual head matrices](ntx_forward_stages_2026-11-25_head_checks.csv)
have baseline condition numbers `1.705728e08`, `5.013085e08` and
`1.314891e08` for modes 0, 1 and 2. Across all phases, backward errors for
the reduced substitution equations are between `2.904348e-16` and
`9.483563e-16`. A small backward error does not certify coefficient accuracy
for these ill-conditioned matrices, nor does it test the eliminated tail.

Error-functional analysis applies the **already-recorded pivoted LU factors**
to transpose/error vectors, with wider host accumulation. It performs no new
factorization and replaces no physical primal or configured solver. The
estimated final head-substitution contribution to the D13 FD is
`9.482471e-06`, much smaller than the observed `3.412649e-03` gap. This is
an approximate error budget, not a rigorous error bound; the D13 adjoint
relative residual in that analysis is about `1.6e-12`. The
[FD budget table](ntx_forward_stages_2026-11-25_fd_budget.csv) retains the
averaging, substitution and observed-matrix sensitivity results.

The observed head-matrix/source/geometry secants, contracted with the saved
head adjoint, predict a D13 derivative `4.109945e-01`, close to the observed
FD `4.110026e-01` rather than the forward JVP `4.075900e-01`. Most of this
point's discrepancy therefore appears before the final substitution and
coefficient averaging, in the represented fields/eliminated matrices/sources.
The data do not yet identify which of field preparation, operator formation
and Schur elimination is responsible. No solver correction is claimed.

## Saved prepared-field comparison

The optional `--prepared-field-reference` diagnostic reuses those exact
baseline/minus/plus snapshots. It computes one baseline production NTX
multi-RHS pullback and a geometry-preparation JVP, then contracts the same
geometry bars with both the true preparation direction and the **saved**
prepared-field FD secant. Three unit raw-coefficient RHS rows cover D11,
D13 and D33. No new FD endpoints or perturbation sizes are introduced.

Input bytes, exact captured coefficient-baseline equality with the completed
audit, step, collisionality, observed/ordinary coefficient equality and
recorded solver source hashes are checked. This saved-data diagnostic uses
its captured baseline rather than another standalone forward replay; the
existing fresh-primal checks in all other modes are unchanged. Captured geometry arrays and
derivative matrices enter the production pullback as **dynamic** values.
Its implicit-primal coefficient offsets and any preparation-JVP primal byte
mismatches are reported; agreement is not assumed merely because the API is
shared. The prepared-field contraction can distinguish field-preparation
sensitivity from a discrepancy arising later in the raw solve, provided
these primal and forward/reverse controls are consistent.

Two analytic/mocked tests cover row contractions, an injected preparation
error and rejection of mismatched inputs/steps. The existing twelve
coefficient-diagnostic tests also pass. No production numerical method,
acceptance criterion or tolerance is changed.

Publish from local WSL:

```bash
(
  cd /mnt/d/Github_Codes/NEOPAX-v0_main || exit 1
  git switch en/v0_main || exit 1
  neopax_files=(
    examples/benchmarks/check_captured_ntx_directional_derivative.py
    examples/benchmarks/_ntx_prepared_field_check.py
    tests/test_ntx_prepared_field_check.py
    docs/benchmarks/current_ntx_derivative_diagnosis_2026-11-25.md
    docs/benchmarks/ntx_forward_stages_2026-11-25_postprocessing.csv
    docs/benchmarks/ntx_forward_stages_2026-11-25_head_checks.csv
    docs/benchmarks/ntx_forward_stages_2026-11-25_fd_budget.csv
  )
  git add -- "${neopax_files[@]}" || exit 1
  git commit --only -m "Localize NTX prepared-field sensitivity using saved endpoints" \
    -- "${neopax_files[@]}" || exit 1
  git push origin en/v0_main
) || printf '\nPublish stopped; terminal kept open.\n'
```

On the large machine, reuse the saved RBC point's fields and endpoints:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git switch en/v0_main || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  set -o pipefail
  neopax_out=$(mktemp -d outputs/ntx_prepared_field_check_XXXXXX) || exit 1
  printf 'Results: %s\n' "$neopax_out"

  if env -u PYTHONPATH -u LD_LIBRARY_PATH -u JAX_PLATFORM_NAME \
    -u XLA_FLAGS -u JAX_COMPILATION_CACHE_DIR -u XLA_PYTHON_CLIENT_ALLOCATOR \
    -u NEOPAX_GRADIENT_AUDIT_DIR -u NEOPAX_ORIGINAL_FORWARD_DIR \
    -u NEOPAX_ORIGINAL_FORWARD_STOP \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true JAX_ENABLE_COMPILATION_CACHE=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false NTX_DISABLE_COMPILATION_CACHE=1 \
    VMEX_COMPILATION_CACHE=disabled \
    "$HOME/jax_env/bin/python" \
    examples/benchmarks/check_captured_ntx_directional_derivative.py \
    --run-dir outputs/fixed_projection_ad_fd16_gu6ZO2/black_box \
    --parameter RBC:1:0 --points 3:1:6 \
    --precision-from-table-audit outputs/ntx_full_table_jvp_ThvCQo/RBC_1_0 \
    --prepared-field-reference outputs/ntx_forward_stages_klS68a/data/forward_stages_3_1_6_scale_1 \
    --out-dir "$neopax_out/data" \
    2>&1 | tee "$neopax_out/run.log"
  then
    neopax_status=0
  else
    neopax_status=$?
  fi

  tar -czf outputs/ntx_prepared_field_check.tar.gz -C "$neopax_out" . || exit 1
  printf 'Diagnostic exit status: %s\nArchive: %s\n' \
    "$neopax_status" "$HOME/NEOPAX/outputs/ntx_prepared_field_check.tar.gz"
  exit "$neopax_status"
) || printf '\nDiagnostic stopped; saved archive retained; terminal kept open.\n'
```

Retrieve from local WSL:

```bash
scp exouser@149.165.170.203:/home/exouser/NEOPAX/outputs/ntx_prepared_field_check.tar.gz \
  /mnt/d/Github_Codes/
```

## Prepared-field result and original-row error budget

`outputs/ntx_prepared_field_check_6cRGVm` at `f417325` completed. The
[saved comparisons](ntx_prepared_fields_2026-11-25.csv) use the original
RBC step and saved endpoints. Every geometry-preparation JVP primal field
matches the captured baseline byte-for-byte.

| Raw coefficient | Forward JVP | Production VJP | Saved raw FD | Effect of replacing the preparation JVP by saved field FD |
| --- | ---: | ---: | ---: | ---: |
| D11 | `-6.747153934427e-06` | `-6.747149177403e-06` | `-6.762969330465e-06` | `-1.763836e-15` |
| D13 | `4.075899759189e-01` | `4.075911457049e-01` | `4.110026248372e-01` | `1.492051e-09` |
| D33 | `4.428216132000e+05` | `4.428224202001e+05` | `4.424463877703e+05` | `-3.332454e-03` |

The D13 VJP/JVP relative difference is `2.870007e-06`. Replacing the true
preparation direction by the actual saved field secant changes that result
by only `3.660667e-09` relative to the JVP. The remaining raw FD difference
is `8.369876e-03` relative. Field preparation therefore does not explain
the dominant discrepancy at this point. The implicit-pullback primal
offsets are retained in the CSV; forward/reverse agreement is not exact.

NumPy-only analysis of the saved original operator equations adds a more
specific location. It uses the recorded modes and LU factors, and applies
those factors only to transpose error-functional vectors. It introduces
no new physical primal, factorization or solver. The original-row residual
is evaluated with extended accumulation and contracted with the transformed
head adjoint at each endpoint. Its estimated FD contributions are:

| Raw coefficient | Observed FD minus JVP | Original row 0 residual contribution | Original row 1 residual contribution |
| --- | ---: | ---: | ---: |
| D11 | `-1.581540e-08` | `-1.471636e-08` | `-2.126766e-10` |
| D13 | `3.412649e-03` | `8.923602e-05` | `2.971047e-03` |
| D33 | `-3.752254e+02` | `2.446679e+01` | `-5.416524e+01` |

These are approximate error-functional estimates, not rigorous bounds or
an exact decomposition of the entire derivative discrepancy. They locate
most of the D13 discrepancy at original Legendre row 1 and most of D11 at
row 0. They do not explain most of D33. The earlier reduced-equation
substitution errors and coefficient-averaging errors were too small to
explain D13. This points to arithmetic in the head elimination that builds
the reduced equations. Reproducible products alone do not establish their
accuracy in this low-collisionality system.

The existing `--capture-forward-stages` observation now also saves the
actual `d_j`, `u_j`, matrix-RHS solve `x`, source-RHS solve `solved_sigma`,
`b_j` and `sigma_next` at head row 1, plus `u_j` at row 0. The public
configured forward solver is called unchanged, and observed/ordinary
coefficient comparisons remain visible at all three phases. Only the
three retained rows are captured; no tail matrix tape is added.

`_ntx_head_roundoff.py` reads these operands and the existing retained
factors using NumPy. It prints a D13 FD error budget separating:

- Schur matrix-product and source-product rounding;
- final head substitutions;
- the next block's LU factorization error and LU-application error.

The decomposition residual and transpose-adjoint residual are reported,
because this error-functional analysis uses recorded factors rather than
an exact inverse. An observed/ordinary mismatch prevents attributing the
new capture's budget to the old ordinary result. Host extended precision
is required. Three manufactured tests independently inject product,
factorization and LU-application errors, verify budget closure, and check
pivot order; a saved-file report test keeps a mismatched observed primal
visible. Three mocked observation tests verify unchanged calls and
visible primal drift. All seven pass without JAX or a simulation.

The next run repeats only the original one-point, three-phase observation.
It retains the original step to determine which arithmetic correction is
justified before changing production code. It does not rerun either full
transport benchmark or change any numerical method.
