# VMEX-main reverse-AD migration

## Scope

Migrate NEOPAX realtime-geometry reverse AD to the `vmex` main-based
worktree.  Keep the protected `en/local_test` VMEX/NEOPAX worktrees unchanged.

## Implementation order

1. **Implicit bridge**
   - Use the main-based
     `implicit_state_pullback_multi_rhs_raw_block_transpose` compatibility
     entry point already added in `en/main_neopax_reverse_ad`.
   - Preserve the former full-RHS-batch default; response chunking remains
     opt-in.

2. **Freeze physical-pitch metadata per geometry stage**
   - After the baseline VMEX state is available, run the existing NEOPAX
     `booz_xform` once and reconstruct the sampled Boozer-angle `|B|` from
     those returned Fourier tables.  Compute common trapped pitches at depths
     `(0.35, 0.55, 0.75)` from that reconstruction.
   - Do **not** use main's `common_trapped_pitches_state` here: it calls
     `boozer_bmnc_state` and would create a second Boozer transform.
   - Store those values as non-differentiated stage metadata.  They must not
     be recomputed inside the differentiated objective, because well matching
     is discrete and the objective must compare the same particle population
     over a stage.

3. **Replace the old J-surrogate post-processing**
   - Keep the existing shared NEOPAX `booz_xform` and its state pullback
     structure; it is the sole Boozer transform for scalar, QI, and max-J
     rows.
   - Replace `j_invariant_qi_maxj_residual_from_boozer` with VMEX main's
     `j_invariant_qi_residual_from_boozer` and
     `maximum_j_residual_from_boozer` using the fixed pitches, `G_b`, `I_b`,
     and signed normalized `psi_b` / `psi_edge` data.
   - Preserve the existing scalar table names initially:
     `boozer_qi_objective`, `boozer_maxj_objective`.

4. **Add main's Mercier objective**
   - Rebuild the main VMEX runtime from the dynamic implicit parameters.
   - Add `mercier_stability_residual(state, runtime)` as the reverse-table
     DMerc rows, matching VMEX optimization examples.  Its interior profile
     (`[2:-1]`) is the residual; no WOUT or finite-difference path is used.

5. **Validation**
   - VMEX unit parity: compatibility multi-RHS pullback vs main scalar VJP.
   - NEOPAX CPU mock parity: QI/maxJ/DMerc values and state pullbacks.
   - Only then run a user-owned GPU reverse benchmark with
     `PYTHONPATH` selecting the two migration worktrees.

## Explicit non-goals

- Do not port `omnigenity_j.py` or old J-surrogate QI/maxJ code.
- Do not modify the protected active worktrees.
- Do not change the existing lagged NTX benchmark until the main-VMEX
  migration passes its isolated tests.

## Current state — 2026-08-29

- The main-VMEX compatibility bridge and the isolated NEOPAX migration live
  only in `vmex-rhs-block` / `NEOPAX-vmex-main-ad`; the normal
  `en/local_test` and `en/reverse_ad_improvement` worktrees remain untouched.
- QI and max-J now consume the Fourier data from NEOPAX's existing
  `booz_xform`.  Their common trapped pitches are reconstructed from that
  same output, so this route does not introduce a second Boozer transform.
- The physical VMEX-main Mercier residual is reduced to one
  `vmec_dmerc_stability_softmax` objective: a softmax-weighted maximum of the
  already-smooth instability residual on `DMerc[2:-1]`.  Radial DMerc rows
  are retained only as a low-level diagnostic helper, not as benchmark rows.
- CPU mock tests cover the QI/max-J adapter, DMerc state/runtime adapter, and
  the Mercier row contract.  They are structural tests only; they are not the
  final physics validation.

## Agreed next validation and integration

The authoritative validation must use the existing paired transport scripts,
not a separate solver-only harness:

1. Expose QI, max-J, and the scalar physical-DMerc softmax objective through
   the shared geometry objective selection used by the benchmarks.
2. Evaluate their values with
   `examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py`.
3. Evaluate the matching derivatives with
   `examples/benchmarks/benchmark_transport_reverse_ad_only.py` using
   `--realtime-geometry-gradient-path reverse_payload`.
4. Compare the same selected rows and parameters in the FD and reverse-payload
   reports.

This preserves the intended production chain: VMEC state -> one NEOPAX Boozer
transform -> objective table -> transport support payload -> reverse AD.

### Implemented benchmark objective surface

- `geometry_full_ad_objectives` now includes
  `vmec_dmerc_stability_softmax`, alongside the existing VMEX scalar, QI, and
  max-J rows.  It is a softmax-weighted maximum of VMEX main's physical,
  already-softplus-smoothed Mercier residual vector; it is **not** a set of
  radial output rows.
- The full-payload reverse benchmark adds this geometry term to the same mixed
  least-squares smoke path that already carries QI and max-J.
- The forward-FD benchmark exposes
  `--include-vmec-main-geometry-objectives` for a realtime geometry parameter.
  It appends the canonical full geometry table to the transport objective
  vector, so its QI/max-J/DMerc labels match the reverse path exactly.
- For an FD-vs-reverse comparison of those rows, use
  `--geometry-fd-lane nonlinear_resolve`; the frozen-linearized lane remains a
  useful local diagnostic, but is not the nonlinear finite-difference oracle.

## Integration plan: migrate the optimization lane to VMEX main

### Decision

Do not retain a dual old-VMEX / VMEX-main objective-table compatibility layer.
QI, max-J, and Mercier semantics are being changed for the optimization lane
anyway.  Instead, integrate the complete VMEX-main path on a new NEOPAX branch
based on the current protected reverse-AD branch.  The protected branch remains
the old-VMEX reference while this work is validated.

### Branch topology

```text
en/reverse_ad_improvement                 protected current NEOPAX lane
        \
         `-- en/vmex_main_reverse_ad_integration
                 +-- merge en/vmex_main_reverse_ad
                 +-- use VMEX en/main_neopax_reverse_ad via vmex-rhs-block
```

### Steps

1. **Snapshot the isolated work**
   - Commit the NEOPAX migration work on `en/vmex_main_reverse_ad`.
   - Commit the VMEX compatibility bridge on `en/main_neopax_reverse_ad`.
   - Do not alter or merge into the protected worktrees.

2. **Create the integration worktree**
   - Branch `en/vmex_main_reverse_ad_integration` from
     `en/reverse_ad_improvement` in a new NEOPAX worktree.
   - Merge the NEOPAX migration branch there.
   - Run that worktree with `PYTHONPATH` selecting `vmex-rhs-block` and NTX.

3. **Resolve the optimization-lane contract in the integration branch**
   - Make the normal geometry objective table use VMEX-main QI and max-J
     semantics from the existing NEOPAX `booz_xform` output.
   - Keep exactly one Boozer transform per geometry evaluation.
   - Include the scalar `vmec_dmerc_stability_softmax` term, not radial DMerc
     rows.
   - Update optimization-script labels/reporting and any expected objective
     ordering to the new table contract.

4. **Update the two authoritative benchmark lanes**
   - Forward FD: append the same canonical VMEX-main geometry objective table
     when `--include-vmec-main-geometry-objectives` is selected.
   - Reverse payload: use those rows through the mixed geometry/transport
     objective table and its raw-block VMEX transpose.
   - Keep all current lagged-NTX timing modes unchanged; this migration is
     orthogonal to their reverse-segment implementation.

5. **Validate in increasing scope**
   - CPU mock/unit checks for table ordering, one-Boozer ownership, and VMEX
     residual contracts.
   - User-owned nonlinear-resolve FD run for one RBC/ZBS parameter with the
     extended geometry rows selected.
   - User-owned reverse-payload run selecting the same rows and parameter.
   - Compare values and gradients row-by-row.  Only after parity is acceptable
     should the integration branch be considered for the active lane.
