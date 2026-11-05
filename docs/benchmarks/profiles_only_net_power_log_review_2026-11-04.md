# Profiles-only net-power run: log review, 2026-11-04

The supplied log is from
`optimize_profiles_max_er_transition_bootstrap_net_power_initial_root_database_full_transport.py`.
It begins at evaluation 17 and ends after report generation fails. The original
attachment is `c0a33014-fb34-444a-8d63-a21b489b66e5/Pasted text.txt`.
No local transport, VMEC, NTX, optimization, AD or FD simulation was run for
this review.

## Optimization result

The four shared variables are `n0`, `T0`, `density_shape_power` and
`temperature_shape_power`; the reported Jacobian has shape `(4, 4)`. The active
residuals are the old left/right electric-field targets, corrected bootstrap
penalty, and net power. Maximum Er and the newer transition position/strength
terms are inactive. The reporter returns `np.nan` for missing objective labels;
the displayed NaNs are therefore not evidence of a nonfinite adjoint.

The cost decreases from 89.085 to 54.5359124 in 19 function evaluations. SciPy
stops on `ftol`, with first-order optimality 0.0195. Small cost change does not
establish that the requested physical targets have been achieved.

| Term | Target | Final value | Contribution to cost |
| --- | ---: | ---: | ---: |
| Er left [kV/m] | 26 | -7.93647666 | 51.8258002 |
| Er right [kV/m] | -10 | -10.1375726 | 0.000851680 |
| Bootstrap deficit, scaled | 0 | 1.59862704 | 2.55560841 |
| Net power [MW/m^3] | 0.906292944 | 0.351942720 | 0.153652085 |

The left-field term contributes 95.03% of final cost; net power contributes
0.282%. Net power is weighted by 1, bootstrap by 2, and each Er point by 0.09
in this script. The printed net power is 116.499656 MW, obtained by multiplying
the volume average by the fixed reference volume 331.01879699 m^3.

The final full-time monitor is `softmax_Er=-0.30386390834537719` kV/m. As this
log-sum-exp maximum is an upper bound on the sampled maximum, its negative
value indicates that the sampled profile remains entirely on the negative-Er
side. There is no electron-root region or positive-to-negative crossing in
that final sampled profile.

Three parameters are at or very near their allowed bounds:

| Parameter | Final value | Allowed interval |
| --- | ---: | ---: |
| n0 | 2.0000000000000036 | [2, 10] |
| T0 | 24.99999846285028 | [7, 25] |
| density_shape_power | 11.999897036517918 | [0.1, 12] |
| temperature_shape_power | 10.835566224074642 | [0.1, 12] |

This run stalls within its current bounds and feasible forward trials. The
excerpt does not prove global infeasibility or incorrect derivatives.

## Incomplete forward trials

Evaluations 17 and 18 stop at 0.1003038887013962 and 0.1164339336672702 s,
respectively, instead of reaching 2 s. The complete-final-time guard rejects
them before terminal objective cotangents and the segment adjoint sweep.
They are returned to SciPy as penalty residuals. Evaluation 19 completes its
reverse sweep with 436 accepted steps over nine segments (8*50 + 36).

The excerpt lacks attempted-step counts and solver failure status for the
incomplete trials, so it does not establish whether they exhausted the total
attempt budget or hit a numerical stopping condition. The checked-in script
sets 1000 total attempts and a 500 accepted-step guard. Do not diagnose a
specific limit or a nonfinite reverse failure from this message alone.

## Report-generation defect and fix

The optimizer writes both profile-parameter JSON files and
`optimization_summary.json` before trying to run the initial transport report.
That report fails on `general.device="default"`: optimization accepts this
selector, while `run_config` accepts `auto`, `cpu` or `gpu`.

`ProfileFullTransportLeastSquaresProblem.config_from_scaled_parameters` now
maps `default` to `auto`, as the geometry export already does. Explicit `cpu`
and `gpu` choices remain valid. The original optimization configuration is
not mutated. This addresses both profile report generation and the public
profile final-transport helper, which use this shared export method.

Validation: five small configuration-export cases and the existing mocked
profile-builder check pass (six checks total). They exercise only tiny arrays,
dictionary conversion and dummy runtimes; no physics simulation is invoked.
The saved remote optimizer parameters remain available despite the report
crash. No objective weights, bounds, defaults or solver behavior were changed.

## Requested follow-up: 51-surface wHe geometry and profile comparisons

The standalone profile example now defaults to the coupled transition
location/strength objective (`--transition-location-objective`), with the
older fixed-cell Er-left/right objectives disabled. Bootstrap and net power
are enabled by default for the requested combined profile-only case. The
four shared profile variables, their bounds, and objective weights are retained.

The fixed geometry input is `examples/inputs/input.QI_nfp2_newNT_opt_hires`,
with VMEC `NS_ARRAY=[24,51]` and 51 transport radial cells. The wHe realtime
benchmark's `_true` input has identical physical input lines; only its radial
solver staging continues to 101 and 201 surfaces. The 51-surface input is used
as explicitly requested. The transport endpoint remains 2 s and energy
quadrature remains four points.

The normal optimization command writes these comparisons as PNG/PDF:

- `initial_profiles_initial_vs_optimized`: input density and temperature for
  all four species, plus differences. This uses only the analytical profile
  map and is written before transport reporting.
- `final_profiles_initial_vs_optimized`: terminal density, temperature and Er
  from the two existing report HDF5 files, after both reports complete.

The input comparison also writes `initial_optimized_initial_profiles.csv`.
The summary records the active transition-location flag, radial target,
strength target, radial grid size, profile-plot configuration and plot paths.

To draw input profiles from an already saved optimization result without
VMEC, NTX or transport:

```bash
python examples/optimization/optimize_profiles_max_er_transition_bootstrap_net_power_initial_root_database_full_transport.py --plot-profiles-only
```

Use `--out-dir` to select a different saved run. Older results whose report
stage crashed are supported through their saved summary/parameter JSON files.

Validation: nine small checks passed. These cover physical-equilibrium input
identity and 51-surface staging, objective defaults, the plot-only path with
physics entrypoints forbidden, small saved-HDF5 plotting fixtures, ordering
of parameter plots before transport reports, and device export. No local
equilibrium, NTX, transport or optimization simulation was run.
