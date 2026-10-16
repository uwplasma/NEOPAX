# Geometry FD command and context audit

The saved `origin/en/reverse_ad_improvement` reference is commit `59aadd4`.
Its geometry notes use `--geometry-fd-lane frozen_linearized`, sixteen accepted
steps, legacy Jacobian reuse, and accepted replay. Initial-Er tests enable
`jax_selected_root`; frozen-linearized root tests hold the selected branch
fixed. The existing frozen-metadata option is an independent replay selector.
There is no historical command flag that overrides the Boozer resolution in
the current FD runner.

The older notes describe two separate problems: an inconsistent forward versus
implicit VMEC linearization point, and omitted composite turbulent/classical
geometry contributions. The latter was corrected in the September 12 record;
its matching RBC net-power AD/FD values were about 5.748758e-05 and
5.760105e-05. Those historical records are in
[the database AD/FD history](../../reverse_ad_vs_fd_database.md).

The new RBC FD net-power value, 2.917066e-04, is also close to the old
explicitly stale FD value, 2.917166e-04, rather than the later matching
reference, 5.760105e-05. The new geometry baseline softmax Er,
21.362915224766219, similarly resembles that stale FD baseline,
21.36291598654. This resemblance reinforces the need to reconcile the primal
setup; it is not numerical proof of the cause by itself.

## Confirmed current configuration mismatch

The current database TOML intentionally omits `mboz` and `nboz` to request
input-dependent Boozer sizing. The production runtime passes unset values
through `build_geometry_autodiff_context`. Its resolver chooses
`mboz = 6 * MPOL` and `nboz = 2 * NTOR - 1` for this input. The input has
`MPOL = 6` and `NTOR = 6`, giving `(36, 11)`.

The benchmark helper `_geometry_context_from_config` instead hard-coded
`(18, 18)` when those settings were absent. The same fallback exists in the
saved improvement-branch source. Geometry FD calls this helper to construct
its full baseline runtime; profile FD and the AD transport primal call
`build_runtime_context`, which uses automatic sizing. Thus the two paths
have demonstrably different geometry configuration despite using the same
TOML. This is a concrete cause for concern before comparing derivatives.

The pasted geometry log already prints `baseline forward VMEC solve for
frozen-linearized geometry FD`, proving that the older correction retaining
the forward VMEC state was present in that run. Merely changing the replay
selector cannot repair the observed baseline difference: the baseline
objective is computed before endpoint replay and before the frozen-root FD
branch is prepared.

## Correction

The shared benchmark helper now uses the runtime's `_optional_config_int`
normalization and passes `None` through for unspecified Boozer sizes.
Explicit sizes and the `vmec_mboz`/`vmec_nboz` aliases still follow the runtime
configuration rules. This aligns both benchmark geometry contexts with the
production runtime's selected resolution. The geometry FD startup banner now
prints the resolved `mboz` and `nboz`; this case should report `(36, 11)`.

The regression in `tests/test_reverse_benchmark_root_objectives.py` compares
the benchmark and production runtime context arguments with omitted, explicit,
aliased, null, zero, and precedence cases. It mocks geometry construction and
does not run VMEC, NTX, or transport solves.

All five tests in that file passed locally in 6.425 seconds of test execution,
including the new configuration-parity regression and the prior synthetic
root-objective and mocked AD/FD routing checks. `git diff --check` also passed.

This is a source-level configuration correction, not a completed numerical
reproduction of the baseline or a proof that every geometry derivative is
correct. The previously documented forward/implicit state-consistency concern
must still be considered if discrepancies remain after matching resolution.

## Command selection and next comparison

The FD mode selection already used in the new run is appropriate for the
intended local derivative comparison:

```text
--accepted-step-limit 16
--radau-jacobian-reuse-mode legacy
--geometry-fd-lane frozen_linearized
--replay-mode accepted
--fd-accepted-replay-contract reverse_frozen_metadata
--initial-Er-root-ad jax_selected_root
--initial-Er-root-fd-root-lane frozen_linearized
```

Use the wHe runtime-database black-box TOML from the
[paired benchmark instructions](database_transition_objectives_ad_fd.md),
with `--parameter RBC:1:0` and `--parameter ZBS:1:0` for the two geometry
columns. Historical commands using the no-He exact-Lij TOML describe a
different test case and should not replace this configuration.

The shared helper also supplies the AD derivative context and geometry-only
objective context, so corrected geometry FD must be compared with a refreshed
AD geometry reference using the same code and resolution. First check that
their baseline observables match. The archived pre-correction AD/FD values
remain evidence of the mismatch, rather than validation data for the
corrected setup. No expensive local physics test was run for this audit.

## Subsequent remote confirmation

Attachment `b1db715e-cb91-4f81-a93c-9bc8cb093fc0/Pasted text.txt` contains the
complete eight-parameter FD rerun after the correction. Both geometry banners
print `(36, 11)`, all eight runs succeed, and all 88 baseline objective values
now match the archived AD run at printed precision. This numerically confirms
that the observed baseline mismatch has been removed by the corrected setup.

The [updated comparison](../../reverse_ad_vs_fd_database.md#2026-10-15-boozer-aligned-fd-rerun-all-eight-dofs)
shows large reductions in the geometry discrepancies. For example, RBC
net-power error versus the same archived AD column falls from 80.292974% to
0.049984%, and RBC strength error from 24.208012% to 0.059685%. Bootstrap
remains the largest geometry discrepancy at 0.322008% (RBC) and 0.755097%
(ZBS). No refreshed post-correction AD run is present in this attachment;
the comparison identifies its older AD reference accordingly. Raw output
and all 88 pairs are archived without changing the previous results.
