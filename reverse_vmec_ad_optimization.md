# Future VMEC reverse-AD optimization: one-primal reuse

## Current behavior

The realtime transport setup and the VMEC raw-block reverse boundary currently
perform separate forward-equilibrium solves:

```text
runtime construction:  solve_multigrid(...) -> state A
reverse preparation:   implicit.solve_implicit_with_aux(...) -> state B + dof mask
                        raw-block transpose linearized at state B
```

The runtime geometry/database/recorded NTX scan is built from `state A`.  The
raw-block payload transpose rebuilds its geometry/payload from `state B`.
Both are intended to represent the same equilibrium, but they are produced by
different forward lanes.

## Why this is not a current database-derivative diagnosis

The VMEC geometry-only AD-vs-FD checks validate the implicit tangent/adjoint
operator.  Lij realtime uses the same retained-forward-payload/raw-block
payload-transpose pattern.  Therefore this item is a consistency and runtime
optimization investigation, not evidence of a missing database derivative.

The database root-only residual errors must continue to be audited at the
database-specific chain:

```text
geometry -> database interpolation/coordinates -> Gamma or corrected Upar
         -> selected Er root / bootstrap objective
```

## Candidate designs

### A. Reuse the configured forward state A

Retain the forward VMEC state, construct the implicit parameter tree and
structural DOF mask at that state, and pass a `GeometryRawBlockSolve` built
from those objects into the raw-block transpose.  This avoids the second
forward solve.

Before adopting it, validate that `state A` satisfies the implicit residual
formulation and that its structural mask is valid.  The forward lane is
`solve_multigrid(...)`; the reverse lane uses the implicit residual solver, so
equivalence must be measured rather than assumed.

### B. Make the implicit state B the sole runtime primal

At the beginning of a realtime-geometry reverse evaluation:

```text
implicit solve -> state B + dof mask
build runtime/database/scan from state B
evaluate transport/root objective and form cotangent
raw-block transpose at the retained state B
```

This has exactly one VMEC forward solve and an internally identical primal for
the runtime and transpose.  The frozen-linearized FD lane must then use that
same `state B` as `state_star`.

## Memory expectations

Replacing reverse state B with retained state A has approximately the same
reverse-phase VMEC-artifact footprint.  It may increase persistent memory
because A must remain alive between runtime construction and reverse.

Design B retains only one VMEC state, but it remains live while the runtime,
database and transport calculation execute.  Neither design should duplicate
the recorded NTX database table or scan record; measure peak host/GPU memory
before using either in the full 16-step/four-segment benchmark.

## Required validation

1. At zero parameter delta, compare the retained/rebuilt geometry and database
   scan channels/surfaces leafwise.
2. Check the implicit residual norm and structural mask for a retained forward
   state A.
3. Re-run VMEC geometry-only FD versus AD with the selected design.
4. Re-run database root-only FD versus AD, then the full 16/4 benchmark.
5. Keep Lij unchanged until the database-only experiment is validated.
