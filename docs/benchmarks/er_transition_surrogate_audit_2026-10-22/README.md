# Root-surrogate sensitivity and optimization-behavior audit

The goal is a sufficiently strong transition near the requested radius,
with useful gradients toward that goal. Exact reproduction of the earlier
Er-left/right amplitude targets is not required. This audit examines whether
the current location/strength surrogate provides that behavior, separately
from checking whether AD correctly differentiates its formula.

The next assessment is the [physical DoF audit](physical_dof_audit.md):
compare candidate gradients through the real NEOPAX reverse path and check
their predicted root-position progress separately for profiles and geometry.
The synthetic cases alone do not establish that any candidate meets that goal.

The follow-up [candidate comparison](candidates/README.md) tests four
alternatives on 38 prescribed profiles/directions, with plots and independent
derivative checks. It also exposes false mean roots and radial grid barriers
in the current strength definition. Production objectives remain unchanged.

The additional [Er-left/right and softmax comparison](left_right_softmax/README.md)
checks fixed-cell amplitude targets and the peak-Er objective against those
surrogates. It includes controlled directions, spatial gradient plots, exact
one-coordinate fits, and conditioning of the archived AD rows.

This audit uses only short vectors (at most 51 cells), the production transition formula,
and the current optimization weights. No VMEC, NTX, ambipolar root or transport
solve is involved. The same reasoning applies to the ambipolarity-only and
full-transport examples because both use the same local objective formulas.

## Refined criterion: useful gradients toward the desired transition

The user clarified that reproducing the old amplitude targets is not required.
The shared goal is a transition near the requested radius with useful profile
gradients toward it. The common-amplitude counterexample below does not by
itself establish a failure of that goal. Two added probes address actual
crossing displacement and creation of a candidate near the target.

### A moving crossing: resolved tracking versus grid locking

The [translation probe](translation.csv) samples
`Er(rho; q) = 40 * tanh((q - rho) / width)` on the production-style 51-cell
centre grid. The known continuous zero is q, so its derivative with respect
to q is exactly one. At the centre of a face, the chord-tracking derivative
is independently `x / sinh(x)`, with `x = cell_spacing / width`.

| Width / cell spacing | d(selected radius)/dq | True derivative |
| --- | ---: | ---: |
| 2.55 | 0.97482136 | 1 |
| 1.02 | 0.85611156 | 1 |
| 0.255 | 0.15543351 | 1 |
| 0.102 | 0.00108303 | 1 |
| 0.051 | 1.1964147e-7 | 1 |

The surrogate follows a resolved translating crossing. A transition much
narrower than the cell spacing becomes grid locked between nodes and gives
a very small response. The sampled Er values themselves also become locally
insensitive in this limit; this is not evidence that only the surrogate's AD
is wrong. The independent chord formula and the contraction of local Er
cotangents with `dEr/dq` match the directly differentiated response.

This demonstrates a resolution/conditioning limitation, without establishing
the width or degree of grid locking in the remote benchmark.

### A target-region direction can be ignored until its crossing wins

The [target-region probe](target_region.csv) starts from
`10 * tanh((0.65 - rho) / 0.01)`. A parameter p lowers only the Er value at
cell 26, adjacent to target 0.514; the existing crossing near 0.649 is
untouched. The comparison term is a unit-weight local sign-margin loss:
it asks for positive Er at cell 25 and negative Er at cell 26 with a
1.8 kV/m margin. It does not prescribe the old +26/-10 amplitudes and is
an audit diagnostic rather than a new production default.

| p | Er at target's outer cell | New target-region bracket | Selected radius | Current cost | d(current cost)/dp |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | +10 | absent | 0.64881220 | 9.08716506 | 4.33e-29 |
| 12 | -2 | 0.51633987 | 0.64881220 | 9.08716506 | -9.77e-11 |
| 14 | -4 | 0.51400560 | 0.64881138 | 9.08705471 | -8.84e-4 |
| 20 | -10 | 0.50980392 | 0.50980392 | 0.00880354 | +0.00205690 |

At p=0, the target-local sign-margin derivative is -11.8 while the current
cost derivative is about 4.33e-29. At p=14, a real bracket lies within
6e-6 of the target and has a raw crossing margin of 2.0, above 0.9. The
selector still reports the more distant, stronger crossing. At p=20 the
new candidate wins and the loss drops by about three orders of magnitude.

This illustrates loss-gradient starvation before an improving candidate
wins selection. The active logits multiply crossing strength by beta; the
radial preference does not ensure selection of the closest valid candidate.
The global strength condition is already satisfied by the remote crossing,
so it provides essentially no pressure to build one at the requested radius.

The intermediate synthetic profile has multiple crossings. It is not a proof
that a desired global, single-transition sign pattern has been achieved, nor
that a real profile parameter has this exact localized direction. It isolates
the selector/strength coupling that can make progress toward the target hard.

The archived table lacks the complete terminal Er profile and its spatial
profile-parameter derivatives. Thus it cannot establish how much of the
actual small `d(rho)/dn0` comes from local cancellation, resolution, or
selection concentration. Those quantities remain needed to identify the
mechanism in the remote case.

## The losses impose different conditions

The previous fixed-cell contribution is

\[
C_{\rm cells}=\tfrac12(0.09)\left[(E_L-26)^2+(E_R+10)^2\right].
\]

It separately targets positive Er of 26 kV/m and negative Er of -10 kV/m
at prescribed sample cells. The current coupled contribution is

\[
C_{\rm surrogate}=\tfrac12(1000)(\rho_{\rm selected}-0.514)^2
 +\tfrac12(100)\phi(0.90-S)^2,
\qquad
\phi(d)=\tfrac12(d+\sqrt{d^2+10^{-12}}).
\]

Here S is a radially biased soft maximum of the adjacent-face scores
`smooth_min(Er_inner / 2, -Er_outer / 2)`. For a single bracket with
`Er_inner = A > 0` and `Er_outer = -B < 0`, it is approximately
`min(A, B) / 2`. A 0.90 strength threshold therefore requires roughly
1.8 kV/m on each side in that simple limit, rather than the previous
26 and 10 kV/m amplitude targets. The multi-face normalization and radial
bias also enter S, so 1.8 kV/m is not a universal endpoint threshold.

The weaker endpoint principally controls a smooth-min score. Increasing an
already stronger endpoint need not significantly improve that score. Once
the strength requirement is satisfied, its penalty becomes almost inactive.

## Location has a common-amplitude null direction

For a single resolved bracket, the candidate position is approximately

\[
\rho_z=\rho_L+\Delta\rho\frac{A}{A+B},
\qquad
\frac{d\rho_z}{dp}=\Delta\rho\frac{B\,dA/dp-A\,dB/dp}{(A+B)^2}.
\]

This contains a radial-cell-width/amplitude scale factor. The two terms also
cancel when both amplitudes change proportionally. Nonzero cotangents on
both Er cells can therefore produce a tiny profile-parameter derivative.
For the 26/-10 bracket with radial spacing 0.02, the local derivatives with
respect to the two signed Er values are about 1.54321e-4 and 4.01235e-4.
Their contraction with the common-amplitude direction `(26, -10)` is zero
up to smoothing/roundoff effects.

The [amplitude probe](amplitude.csv) evaluates the same two cells at
rho = 0.50 and 0.52 using `Er = amplitude * (26, -10)`:

| Amplitude | Er-left | Er-right | Selected radius | Old cost | Surrogate cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 13 | -5 | 0.5144444444 | 8.73 | 9.87654e-5 |
| 1.0 | 26 | -10 | 0.5144444444 | 0 | 9.87654e-5 |
| 1.5 | 39 | -15 | 0.5144444444 | 8.73 | 9.87654e-5 |
| 2.0 | 52 | -20 | 0.5144444444 | 34.92 | 9.87654e-5 |

At amplitude 1.5, the old cost derivative is **34.92** while the surrogate
cost derivative is **1.8648277366750676e-17**. Strength already exceeds 0.90
throughout this sweep. This is a concrete counterexample to equivalent
costs, using identical sample cells. It does not depend on the separate
benchmark-versus-optimization index mismatch.

Changing weights cannot recover a direction the position formula cancels
and the satisfied strength penalty no longer constrains.

## The saved strength penalty is effectively inactive

The [full-time AD rerun](../database_transition_objectives_ad_recorded_reuse_full_t20_left_right_2026-10-22.md)
reports `S = 2.0149622433190042` and `dS/dn0 = 0.47338579231761335`.
The raw strength derivative is not near zero. Applying the current
strength-deficit transform and weight to that saved point gives the
[following chain](strength_penalty.csv):

| Quantity | Value |
| --- | ---: |
| Smooth deficit phi(0.90 - S) | 2.2426505097428162e-13 |
| d(deficit)/dS | -2.0114465648646274e-13 |
| d(weighted residual)/dn0 | -9.521902258129834e-13 |
| Strength contribution to cost | 2.5147406544248567e-24 |
| Strength contribution to d(cost)/dn0 | -2.135429895291614e-24 |

This reweights the archived benchmark at its own point, whose selection
prior is 0.5. It is not a newly evaluated optimizer Jacobian at prior 0.514.
The scalar transform still shows precisely why a satisfied margin provides
almost no additional optimization drive.

## Sharp selection can suppress movement between faces

With active logits `radial_bias + beta * crossing_score`, sufficiently large
score gaps concentrate the location on one candidate face. The derivative
through changes in the selection weights can then become tiny, leaving
mainly the within-face chord derivative.

The [selection probe](selection.csv) has genuine crossings at rho 0.46
and 0.56. It scales only the inner bracket:

| beta | Selected location | d(location)/d(inner bracket amplitude) |
| --- | ---: | ---: |
| 1 | 0.4901548584 | -0.1053103979 |
| 8 | 0.4600393514 | -0.0015734371 |
| 16 | 0.4600000132 | -1.0564897e-6 |

The production beta of 16 can therefore suppress this selection response
by about five orders of magnitude relative to beta 1 in this example.
This mechanism is demonstrated locally; the supplied full-time benchmark
does not contain its complete final Er profile or candidate weights, so its
actual degree of selection concentration and cancellation cannot be measured
from the archived objective table alone. Decreasing beta changes the selected
location and can average distinct crossings; it does not restore the old cost.

## Consequence for the optimization

Retaining the original amplitude targets is not necessary for the user's
goal. A better-conditioned replacement should provide pressure on an
incorrect sign pattern near the target before a new candidate becomes the
strongest crossing. A target-local existence/sign-margin term, or a radial
wrong-sign penalty, is one possible direction. Separating sufficient crossing
strength from the location preference would also avoid rewarding extra
strength at an already-valid but distant crossing in the selection itself.

The local sign-margin diagnostic demonstrates nonzero guidance in the toy
case; it has not been validated as a production optimizer replacement or as
a complete constraint on a global single-transition profile. Restoring the
old amplitudes or merely increasing a position weight is not established as
the solution. The existing Er derivatives already use the selected-root
differentiation contract, so this issue can concern the loss landscape even
when local AD correctly differentiates it.

No production objective definition was changed by this investigation. The
result establishes a difference in loss requirements and demonstrates
resolution and selection mechanisms for weak gradients. It does not
independently validate the full 20-second transport Jacobian, identify the
dominant mechanism in the real profile directions, or explain every failed
optimizer step.

## Reproduction and validation

From the repository, with the existing JAX environment:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_COMPILATION_CACHE=0 \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python examples/benchmarks/audit_er_transition_surrogate_objectives.py \
  --out-dir outputs/er_transition_surrogate_audit
```

This only evaluates scalar terminal formulas on arrays of at most 51 cells.
Twelve objective tests plus 21 subtests passed, including the independent
NumPy FD checks already in the objective suite, analytical chord cotangents,
the same-cell loss counterexample, margin activation, sharp selection,
crossing translation, and target-region gradient starvation. The expanded
target-region case was additionally checked after adding p=14. The local
suite emitted seven persistent-cache decompression warnings despite the
environment request to disable that cache; tests and probes completed
successfully. No expensive local physics test was run.
