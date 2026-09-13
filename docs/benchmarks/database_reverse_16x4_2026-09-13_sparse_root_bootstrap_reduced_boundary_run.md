# Database reverse 16x4 sparse root/bootstrap and reduced-boundary run (2026-09-13)

## Outcome

The full-transport, runtime-database reverse benchmark completed successfully
with 16 accepted Radau steps and a reverse segment length of 4.  The run used
the established exact block adjoint with the validated inline scalar support
path, plus the opt-in sparse initial-Er-root and terminal-bootstrap
interpolation transposes and the reduced-zero initial boundaries.

The process exited with status zero.  Every raw block parameter bar was finite,
there was no swap activity, and the complete residual/Jacobian table retained
numerical parity with the validated 2026-09-12 performance reference.

## Selected reverse modes

- stage-adjoint solve: `block`;
- stage cotangents: `full`;
- stage-adjoint memory: `default`;
- RHS transpose: `explicit_database`;
- RHS pullback: `separate`;
- step backward boundary: `reduced_cotangent_call_boundary`;
- database stage Jacobian: `independent`;
- objective support: `scalar`;
- segment support: `inline`;
- database transport interpolation: `legacy_sparse`;
- selected-root interpolation: `legacy_sparse`;
- terminal-bootstrap interpolation: `legacy_sparse`;
- initial direct-RHS support: `reduced_zero`;
- initial state: `reduced_zero_rhs`;
- support preparation: `shared`;
- centre geometry: `scalar_jvp`;
- ordinary terminal objectives: `grouped_joint_vjp`;
- bootstrap cotangent: `joint_local_vjp_upar_only`;
- schedule artifact: `reuse_static_probe`.

The log confirmed that every mode above was active.  Geometry/state diagnostics,
the JAX persistent compilation cache, and experimental deferred/batched segment
paths were disabled.

## Performance comparison

The immediate reference is the validated legacy-sparse run recorded in
`database_reverse_16x4_2026-09-13_legacy_sparse_run.md`.

| Quantity | Previous sparse run | Current run | Change |
|---|---:|---:|---:|
| Evaluation time | 2695.506 s | 2191.452 s | -504.054 s (-18.70%) |
| Wall time | 59:04.86 | 51:31.28 | -7:33.58 (-12.80%) |
| Peak RSS | 12,289,244 KiB | 11,685,164 KiB | -604,080 KiB (-4.92%) |
| Final-objective cotangents | 268.380 s | 134.243 s | -134.137 s (-49.98%) |
| Cold segment 4/4 | 476.076 s | 470.954 s | -5.122 s (-1.08%) |
| Segmented cotangent sweep | 478.486 s | 473.515 s | -4.971 s (-1.04%) |
| Initial direct-RHS support | 183.143 s | 0.387 s | -182.756 s (-99.79%) |
| Initial state pullback | 151.444 s | 0.379 s | -151.065 s (-99.75%) |
| Initial-Er root pullback | 240.367 s | 125.052 s | -115.315 s (-47.97%) |
| Final recorded-scan fold | 690.580 s | 762.053 s | +71.473 s (+10.35%) |

Relative to the older established performance run, evaluation time decreased
from 3776.534 s to 2191.452 s, a reduction of 1585.082 s (41.97%).  Peak RSS
decreased from 14,486,496 KiB to 11,685,164 KiB, a reduction of 2,801,332 KiB
(19.34%).

The three warm four-step reverse segments took 0.906 s, 0.858 s and 0.794 s,
for a total of 2.558 s.  Scaling that measured execution to all sixteen steps
gives approximately 3.41 s after compilation.  This satisfies the target of a
database reverse execution cost within a few times the roughly two-second
forward solve.  The 470.954 s first segment remains dominated by cold
compilation rather than reverse execution.

The runtime database build also increased from 602.463 s in the immediate
reference to 662.524 s in this run.  That approximately 10% increase is similar
to the final-scan-fold increase and is evidence of machine/run variability; the
recorded-scan fold should be separated into cold compilation and warm execution
before changing its mathematical contract.

## Phase timings

| Phase | Elapsed |
|---|---:|
| Runtime database build | 662.524 s |
| Solver components | 0.528 s |
| Profile-state VJP | 95.723 s |
| Reduced-zero-RHS initial carry | 0.901 s |
| Realized-schedule forward replay | 1.587 s |
| Final-objective cotangents | 134.243 s |
| Segment 4/4, cold compile plus execution | 470.954 s |
| Segment 3/4, warm execution | 0.906 s |
| Segment 2/4, warm execution | 0.858 s |
| Segment 1/4, warm execution | 0.794 s |
| Complete segmented sweep | 473.515 s |
| Reduced carry expansion | 0.700 s |
| Initial direct-RHS support pullback | 0.387 s |
| Initial state pullback | 0.379 s |
| Initial-Er root pullback | 125.052 s |
| Profile-parameter pullback | 1.385 s |
| Initial-profile scan-payload pullback | 2.132 s |
| Final recorded-scan fold | 762.053 s |

## Numerical parity

The full output was parsed against
`database_reverse_16x4_2026-09-12_performance_run.md`.  The intervening
legacy-sparse run had already been shown equivalent to that same reference.

- compared entries: 153 / 153;
- residuals: 17 / 17 bit-for-bit identical;
- profile derivative rows: 60 / 60 bit-for-bit identical;
- VMEC derivative rows: 34 / 34 finite and matching;
- maximum VMEC relative difference: `1.83861498453114e-10`;
- maximum absolute difference: `4.30591171607375e-08`.

The maximum absolute difference belongs to
`dgeometry:boozer_maxj_objective/dvmec:ZBS:1:0`, whose derivative has magnitude
approximately 1920.5.  Its relative difference is only
`2.24206881057465e-11`.  The maximum relative difference belongs to
`dtransport:Er_transition_right/dvmec:ZBS:1:0`, with an absolute difference of
`9.31018595551336e-11`.

The final raw-block parameter-bar norm was `4.453275e+02`; all entries were
finite and the first-nonfinite marker was `None`.  The new performance modes
therefore preserve the established derivatives and do not alter the existing
AD-versus-FD conclusions.

## Resource report

```text
User time (seconds): 3556.95
System time (seconds): 225.10
Percent of CPU this job got: 122%
Elapsed (wall clock) time: 51:31.28
Maximum resident set size (kbytes): 11685164
Major page faults: 0
Swaps: 0
File system inputs: 224
File system outputs: 392112
Exit status: 0
```

## Next performance target

The next candidate is the final recorded-scan fold.  It is already one batched
transpose over all ten transport-objective rows, so the next audit should first
measure cold compilation separately from warm execution and check whether the
compiled pullback is reused across optimization iterations.  The runtime
database build and cold segment compilation are the other remaining large
one-time phases; the warm segmented reverse execution itself is no longer a
bottleneck.
