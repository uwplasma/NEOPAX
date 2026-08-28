# Initial-root optimization memory handoff

## Reproduction

```bash
python ./examples/optimization/test_geometry_qi_max_er_transition_bootstrap_initial_root_memory.py --mode off --repeats 2 --diagnose-structure --diagnose-jax-dispatch-cache --diagnose-support-reverse-dispatch
```

Fixed exact-Lij objectives: `maxEr`, `Er_transition_left`,
`Er_transition_right`, and `bootstrap_current_softmax_abs_scaled`.

Measured on Linux:

- warm evaluation: about 153 seconds;
- RSS rises about +0.66 GiB on the second measured evaluation;
- `live_jax_arrays=286` is constant;
- exactly one raw VMEX solve and one selected-Er root per evaluation;
- JAX dispatch cache grows by 17 entries per evaluation.

`jax.clear_caches()` with `malloc_trim` eliminates the RSS slope, whereas
`malloc_trim` alone does not. This is retained executable/compiler cache, not
live JAX arrays.

## Exact cache breakdown

| Operation | Entries/evaluation |
| --- | ---: |
| Selected initial-Er root | 1 |
| Corrected bootstrap flux evaluation | 1 |
| Bootstrap state pullback | 1 |
| Bootstrap geometry pullback | 1 |
| Bootstrap NTX-support pullback | 1 |
| Initial-Er residual derivative | 1 |
| Compact root-state transpose | 1 |
| Root-geometry residual VJP construction | 4 |
| Batched root-geometry pullback | 1 |
| Root-NTX-support pullback | 0 |
| Payload compact tangent map | 1 |
| Final result assembly | 3 |
| **Total** | **17** |

The payload pullback is entered after the large transport-support/root cache
contribution already exists. Its own tangent map adds only one entry.

## Rejected causes

- No duplicated geometry calculation per objective or DoF.
- VMEX raw solve is not the source: it adds one entry and its `_block_lane`
  cache stays at 4.
- Pre-VMEX scaling, profile extraction, VMEC parameter extraction, and raw
  parameter updates do not grow the cache.
- Persistent raw callbacks/parameters/JIT and fixed Boozer/grid payload reuse
  did not reduce the slope. Payload reuse slightly worsened warm time.

## Required next implementation

Create an explicit optimization-only, fixed-layout stage that owns stable
bounded callable identities for corrected bootstrap fluxes, the three
bootstrap pullbacks, and the root-geometry residual VJP/batched application.

Rooted state, geometry, NTX support, residual bars, objective cotangents,
VMEX raw state, and optimizer values remain dynamic inputs; the stage retains
no trial arrays. Keep the `off` benchmark path mathematically and operationally
unchanged. Do not add a fused outer JIT or JIT inside VMEC/Boozer/NTX/flux/root
physics.

Acceptance: four-row residual/Jacobian parity versus `off`, one VMEX/root solve
per evaluation, no new entries at these boundaries after warmup, no RSS slope,
and no warm-execution regression.

## Implementation plan

1. **Remove the rejected payload-artifact route from the proposed solution.**
   It may remain available only as a diagnostic until it is deleted, but it
   must not be selected by an optimization example or treated as a memory fix.
   **Completed:** it is no longer an accepted builder or memory-harness mode.

2. **Define an optimization-only boundary API.**
   Add a fixed-layout stage, keyed by the existing TOML-selected flux model,
   objective layout, radial/grid layout, and VMEC parameter layout. It owns
   stable function identities, not trial values. Its dynamic input bundle is
   the current rooted state, pre-root state, geometry, NTX support, residual
   bars, bootstrap `Upar` bars, and geometry delta.

3. **Implement the five measured operators behind that API.**
   They reproduce the established operations exactly: corrected bootstrap
   fluxes; bootstrap state, geometry, and support pullbacks; and the
   root-geometry residual VJP plus its batched application. Reuse existing
   formulas/callbacks. The root-geometry boundary must preserve the existing
   derivative partition: geometry delta is the differentiated input and the
   current state/support inputs are dynamic but not differentiated. Do not
   replace it with a broad all-input VJP.

4. **Connect an explicit optimization mode only.**
   `off` continues to invoke the current benchmark route verbatim. The new
   stage is built once per optimization problem and used only by the geometry
   plus initial-root exact-Lij example. No outer fused JIT is introduced.

5. **Verify correctness and topology.**
   Compare all four residual rows and Jacobian rows to `off`; retain the
   existing floating-point tolerance. Verify one raw solve and one selected
   root per evaluation.

6. **Verify the intended memory behavior.**
   Run the existing two- and three-repeat diagnostic. Each measured bootstrap
   and root-geometry boundary must have a flat dispatch-cache count after
   warmup. Then compare warm elapsed time; reject the stage if it trades the
   RSS slope for a material warm-time regression.

7. **Only then consider full transport.**
   Its accepted-step path can vary, so it needs a separate fixed-topology
   stage design rather than being attached to this initial-root stage.

## Relevant files

- `NEOPAX/_reverse_ad_optimization.py`: mixed initial-root table and bootstrap
  cotangent.
- `NEOPAX/_reverse_ad_initial_er.py`: compact root-state transpose.
- `NEOPAX/_transport_flux_models.py`: corrected bootstrap flux and pullbacks.
- `examples/optimization/test_geometry_qi_max_er_transition_bootstrap_initial_root_memory.py`.
