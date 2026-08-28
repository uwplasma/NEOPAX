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

## Relevant files

- `NEOPAX/_reverse_ad_optimization.py`: mixed initial-root table and bootstrap
  cotangent.
- `NEOPAX/_reverse_ad_initial_er.py`: compact root-state transpose.
- `NEOPAX/_transport_flux_models.py`: corrected bootstrap flux and pullbacks.
- `examples/optimization/test_geometry_qi_max_er_transition_bootstrap_initial_root_memory.py`.
