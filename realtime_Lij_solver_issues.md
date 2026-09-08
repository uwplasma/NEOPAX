# Realtime Lij Radau edge-cache audit

## Verified rebuild at `t = 3.906000e-05 s`

This is an early, accepted Radau rebuild, not the later failure window.  It
records the private floating edge and the exact outer-face inputs passed to the
realtime NTX model.  Species order is the configured `[electron, deuterium,
tritium, helium]`; each column of `nu_over_v` and `Er_over_v` is an NTX energy
node.

### Cache versus live edge RHS

```text
edge_ref       = -3.237309120e+01 kV/m
edge_current   = -3.237396450e+01 kV/m
edge_delta     = -8.733006783e-04 kV/m
probe          =  8.733006783e-05 kV/m

old_at_ref     = -9.713487491e+00
live_at_ref    = -9.713487491e+00
old_at_current = -4.518094589e+01
new_at_current = -4.538461305e+01
live_at_current= -4.538461305e+01
wrapper_old_fd =  0.0
wrapper_live_fd=  0.0
```

The old response is therefore exact at its own anchor.  At the replacement
state it differs from rebuilt/live by `2.036167e-01`, about `0.45%` of the
edge RHS magnitude.  This particular rebuild is healthy and does not contain
the later large mismatch.

Important correction: the two `wrapper_*_fd` values above came from an early
diagnostic that differentiates the flattened Radau wrapper.  They conflict
with the nonzero analytic edge tangent printed by the same run and therefore
are **not valid evidence about the physical edge slope**.  They are retained
only as a record of the faulty probe.  The corrected diagnostic now evaluates
`ElectricFieldEquation.edge_rhs` directly through the cached face-flux primal
and compares its symmetric finite difference with the exact cached tangent.
The concrete wiring bug was an off-by-one private-edge index: `state_dim` is
the augmented vector length, but the final valid index is `state_dim - 1`.
JAX clamps the out-of-range read to the final element while silently dropping
the out-of-range indexed update, so the old probe read the edge but perturbed
nothing.

### Physical outer-face state

```text
drds = 5.889125072e-01

n_ref = n_current = [3.90e-01, 1.95e-01, 1.95e-01, 1.00e-06]

T_ref     = [0.56860679, 0.56857088, 0.56857043, 0.56769859]
T_current = [0.57218271, 0.57199618, 0.57199413, 0.56769859]

vth_ref     = [14143583.38677238, 233386.82581285, 190559.47059651, 164902.76627047]
vth_current = [14187987.47550295, 234088.77882201, 191132.34512131, 164902.76627047]

vnew_ref     = [8032605.21750635, 308367.60204232, 405879.16042405, 505449.73394337]
vnew_current = [8057823.75690064, 309295.07327136, 407099.34554750, 505449.73394337]
```

### Exact post-limit NTX inputs

`nu_over_v` is the collision input after applying `nu_v_min`.
`Er_over_v` is the electric input after applying the configured field limits:

`Er_over_v = Er * drds * 1e3 / vnew`.

```text
nu_over_v_ref =
[[1.69341687e-01, 7.19550761e-03, 1.15489119e-03, 2.77326915e-04],
 [5.39716275e-02, 3.28762904e-03, 5.71241542e-04, 1.41573098e-04],
 [4.51765194e-02, 2.95464447e-03, 5.41476709e-04, 1.37424829e-04],
 [1.58876895e-01, 1.08303262e-02, 2.06533250e-03, 5.36359165e-04]]

nu_over_v_current =
[[1.67297103e-01, 7.10863069e-03, 1.14094729e-03, 2.73978531e-04],
 [5.33479500e-02, 3.24963514e-03, 5.64639245e-04, 1.39936632e-04],
 [4.46546733e-02, 2.92051372e-03, 5.35221499e-04, 1.35837153e-04],
 [1.58498767e-01, 1.08129620e-02, 2.06370966e-03, 5.36219866e-04]]

Er_over_v_ref =
[[-0.00237344, -0.00102019, -0.00063286, -0.00043977],
 [-0.14383403, -0.06182530, -0.03835239, -0.02665070],
 [-0.17616006, -0.07572024, -0.04697191, -0.03264032],
 [-0.20356825, -0.08750132, -0.05428012, -0.03771872]]

Er_over_v_current =
[[-0.00236608, -0.00101703, -0.00063090, -0.00043841],
 [-0.14340659, -0.06164157, -0.03823841, -0.02657150],
 [-0.17563680, -0.07549533, -0.04683238, -0.03254336],
 [-0.20357374, -0.08750368, -0.05428158, -0.03771974]]
```

## Required evidence at the later failure rebuild

The same audit must be read at the rebuild with the previously observed large
old-cache/live discrepancy.  The key classification is:

1. `old_at_ref != live_at_ref`: incorrect response already at its anchor.
2. Anchor values agree but the corrected `old_ambipolar_fd` and
   `old_ambipolar_tangent` disagree: incorrect local Taylor derivative or
   coordinate conversion.
3. Slopes agree but `old_at_current != live_at_current`: incorrect quadratic
   displacement evaluation.
4. Old and live values/probes agree: another state component or solver
   assembly path is responsible.

## Failure-window result at `t = 1.997860e-03 s`

The edge cache value matches at the checked anchor and replacement state, but
the edge-slope conclusion remains pending the corrected primal-edge audit.

```text
edge_ref       = -3.311922756e+01 kV/m
edge_current   = -3.311936865e+01 kV/m
edge_delta     = -1.410938707e-04 kV/m

old_at_ref     = -4.980212800e+02
live_at_ref    = -4.980212800e+02
old_at_current = -4.980313118e+02
new_at_current = -4.980313118e+02
live_at_current= -4.980313118e+02
wrapper_old_fd = wrapper_live_fd = 0.0 in the faulty wrapper-level probe
```

That wrapper-level zero is not interpreted as a physical slope: it contradicts
the simultaneously reported nonzero edge tangent and has been superseded by
the direct cached-edge-primal comparison described above.

The physical outer-face density is unchanged and the outer-face temperatures,
`nu_over_v`, and `Er_over_v` differ only in their final displayed digits.
The assembled edge RHS and ambipolar-only edge RHS also agree exactly
(`rhs_extra = extra_drhs_dedge = 0`).

The failed stage is instead dominated by the public electric-field component
at radial index 29:

```text
stage Er[29] = -1.043689e+00 kV/m
accepted Er[29] = 8.033058e-01 kV/m
stage RHS Er[29] = -1.054203e+07
```

This is a statement about the numerically dominant equation component only;
it does not label the underlying physical feature as a root transition.

At the same stage, the current cached RHS JVP agrees with finite differences:

```text
current_jvp_vs_fd_rel = 2.06e-03
current_jvp_vs_fd_cos = 0.9999994
```

but the matrix action used by the stage update differs substantially from the
finite-difference stage direction (`relative_error = 6.21`).  The remaining
audit target is therefore the full-stage Newton matrix/action path for the
public `Er[29]` component, not outer-edge cache anchoring.

## Independent stable edge case at `t = 2.265902e-03 s`

This later accepted step rules out an inherent inability of the edge node to
move through a larger excursion:

```text
edge rebuild displacement = -6.076278e-02 kV/m
old_at_current            = -5.058410662e+02
new/live_at_current       = -5.058411742e+02

accepted stage edge displacement = -1.111661e-01 kV/m
Newton iterations                = 2
final residual norm              = 1.545529e-04
```

The outer-face `nu_over_v` and `Er_over_v` inputs vary smoothly, and the
matrix-direction error is only `2.69e-02` on iteration 1 (then `1.22e-01`
near convergence).  Thus neither an edge excursion of this size nor the
edge-cache quadratic evaluation is sufficient to cause the failure observed
near `t = 1.997860e-03 s`.

## Later failure pattern at `t = 6.033906e-03 s`

The outer edge remains well behaved after a substantial rebuild displacement:

```text
edge rebuild displacement = -7.358949e-02 kV/m
old_at_current            = -5.193305525e+02
new/live_at_current       = -5.193417599e+02
```

The difference is only `1.12e-02` on an RHS of about `-519`.  The failed large
trial (`h = 1.832662e-04`) is dominated by public `Er[28]`:

```text
accepted Er[28]  =  6.387160 kV/m
stage Er[28]     = -0.308568 kV/m
stage RHS Er[28] = -7.022689e+04
```

For this large trial, the finite-direction comparison is nonlocal to the
current stage state:

```text
current_jvp_vs_fd_rel = 2.87e+02
current_jvp_vs_fd_cos = -7.59e-01
```

After reducing the trial step to `3.665323e-05`, the same check becomes local
again (`2.77e-05`, cosine `1.0`) and the attempt converges.  This is evidence
of a large public-`Er[28]` stage excursion/nonlinear range at the rejected
trial size, not an edge-cache error.  It does not by itself identify the
physical origin of that interior electric-field response.
