# Independent species profiles on a saved optimized geometry

`examples/optimization/optimize_species_profiles_from_geometry_result.py`
performs stage 2 with fixed optimized geometry and independent analytical
initial-profile parameters. The existing six shared parameters are expanded
to 15 independent variables for the current e/D/T/He configuration:

| Species | Density DOFs | Temperature DOFs |
| --- | --- | --- |
| e | Derived by quasineutrality | T0[e], temperature_shape_power[e], temperature_shape_alpha[e] |
| D | n0[D], density_shape_power[D], density_shape_alpha[D] | T0[D], temperature_shape_power[D], temperature_shape_alpha[D] |
| T | n0[T], density_shape_power[T], density_shape_alpha[T] | T0[T], temperature_shape_power[T], temperature_shape_alpha[T] |
| He | Zero initial density remains fixed | Prescribed frozen temperature profile remains fixed |

Labels identify independent variables, not different initial values: the
starting values reproduce the nominal profiles of the saved geometry run.
The existing c_density/n_scale/c_temperature/T_scale factors and edge values
are retained. A species n0 or T0 is the analytical-model amplitude before
these factors, as in the existing shared-parameter example. Electron density
is reconstructed from ion charge densities before forming initial pressure
and selecting the ambipolar root. This avoids an electron-density variable
that the transport quasineutrality projection would subsequently overwrite.
Floored helium pressure retains the configured finite temperature profile.
The density/temperature equation toggles and helium source/evolution are
unchanged; only the initial-profile optimization variables are expanded.

The new layout is opt-in through
`optimization.species_profile_parameters` and
`optimization.species_profile_quasineutral` in the resolved configuration.
The established shared-parameter order remains unchanged when these keys
are absent. The packed vector, selected-root profile pullback and direct
profile-only reverse table all carry the expanded column order. These are
AD columns of the individual species map, not copies of a shared gradient.

The example uses the existing persistent database full-transport reverse
stages with recorded forward Newton/Jacobian setup, and bounded SciPy TRF.
Profile coordinates are nominally scaled deltas by default, with trust
multiplier 1. The amplitude/shape bounds are the existing stage-2 bounds.
It loads the exported accepted VMEC input as the fixed boundary, uses the
saved transport time/database mesh/step budgets, and matches the 18/18
Boozer resolution used by the optimization. No geometry optimization is
run. If the saved optimized transport HDF5 exists, the first terminal field
is checked against it before the first reverse sweep proceeds.

## Run on the GPU machine

After committing/pushing the changes and pulling them on the remote machine:

```bash
cd /home/exouser/NEOPAX || exit 1
git switch en/v0_new || exit 1
git pull --ff-only origin en/v0_new || exit 1
mkdir -p outputs
set -o pipefail

env -u JAX_COMPILATION_CACHE_DIR \
JAX_ENABLE_COMPILATION_CACHE=0 \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src${PYTHONPATH:+:$PYTHONPATH}" \
/home/exouser/jax_env/bin/python \
  examples/optimization/optimize_species_profiles_from_geometry_result.py \
  --geometry-result-dir outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --transition-location-objective \
  --bootstrap-penalty \
  --net-power \
  --no-max-er \
  --no-root-objectives \
  --er-transition-rho-target 0.514 \
  --er-transition-strength-target 0.9 \
  --profile-trust-multiplier 1.0 \
  --max-nfev 30 \
  --out-dir outputs/species_profiles_root_bootstrap_net_power_20261027 \
  2>&1 | tee outputs/species_profiles_root_bootstrap_net_power_20261027.log
```

The production geometry example supplies the targets and weights directly:
root position 0.514 with weight 1000; strength at least 0.9 with weight 100;
bootstrap observable at most 0.1 with weight 4; and the 300 MW
fixed-reference-volume net-power target with weight 10000. The old Er-left
and Er-right terms are disabled by the command above. Frozen geometry-only
cost rows are omitted: they have zero profile derivatives, so total costs
exclude their constant offset and should not be directly compared to the
old geometry-plus-transport total. Their omission also changes the constant
offset in SciPy's relative cost stopping criterion; the physical transport
objectives and their weights are retained.

Use `--transport-max-steps` and `--max-reverse-accepted-steps` to explicitly
override the inherited attempted/accepted-step budgets. Neither is a request
to use a truncated transport objective. A trial must reach final time.
CUDA command buffers remain disabled by default and memory diagnostics are
enabled. A fresh output directory is required after a completed run.

To review the exact DOFs, initial parameter values, bounds and objective
weights without building a runtime, add `--plan-only` to the command.
For a smaller subset, for example independent temperatures only, use:

```bash
--species-profile-parameters 'T0[e],temperature_shape_power[e],temperature_shape_alpha[e],T0[D],temperature_shape_power[D],temperature_shape_alpha[D],T0[T],temperature_shape_power[T],temperature_shape_alpha[T]'
```

## Weight 1 for power, bootstrap and root

The species-profile example accepts `--root-weight`, `--bootstrap-weight`
and `--net-power-weight`. Omit these options to retain the production
weights above. `--root-weight` overrides both the location-moment and
strength-deficit weights; the raw location/strength monitor rows retain
zero weight. Targets, residual definitions and the fixed geometry are
unchanged, and effective weights are saved in `case_manifest.json` and
`optimization_summary.json`.

```bash
cd /home/exouser/NEOPAX || exit 1
mkdir -p outputs
set -o pipefail

env -u JAX_COMPILATION_CACHE_DIR \
JAX_ENABLE_COMPILATION_CACHE=0 \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src${PYTHONPATH:+:$PYTHONPATH}" \
/home/exouser/jax_env/bin/python \
  examples/optimization/optimize_species_profiles_from_geometry_result.py \
  --geometry-result-dir outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --transition-location-objective \
  --bootstrap-penalty \
  --net-power \
  --no-root-objectives \
  --no-max-er \
  --root-weight 1 \
  --bootstrap-weight 1 \
  --net-power-weight 1 \
  --er-transition-rho-target 0.514 \
  --er-transition-strength-target 0.9 \
  --profile-trust-multiplier 1.0 \
  --max-nfev 30 \
  --out-dir outputs/species_profiles_unit_weights_20261027 \
  2>&1 | tee outputs/species_profiles_unit_weights_20261027.log
```

This uses all 15 profile DOFs. Unit weights mean the cost is half the sum
of the squared existing residuals; no new normalization is introduced.
The power residual remains in MW/m^3, the position residual is a radial
location moment, and the bootstrap residual is the excess over 0.1.
Their magnitudes and derivatives can still differ substantially. At the
saved baseline (root about 0.51742, bootstrap about 0.53806, power about
299.89 MW), bootstrap dominates the unit-weight cost. This experiment
changes their trade-off and does not give them equal influence.

## Outputs and checks

The output includes a fixed-geometry input copy, case manifest, saved
parameter JSON for every evaluation, terminal Er/density/temperature NPZ
snapshots, optimization diagnostics with weighted residual/Jacobian tables,
optimized parameter/config JSON and initial-versus-optimized initial-profile
CSV/PNG plots. Saving and plotting those profiles require no extra transport
solve or selected-root solve. The optimized config retains the opt-in
quasineutral initial-state map, so exporting different D/T profiles does not
silently reconstruct a different electron density on a normal forward run.

Local verification passed 40 tests and two subtests using small arrays and
mocked workers. This covers AD versus centered FD for all 15 initial-profile
columns, species independence, quasineutrality, finite floor-temperature
handling, selected-root profile-column routing, direct table row/column
selection without a VMEC transpose, exported-state equivalence, same
production weights/accepted-boundary metadata, plan-only operation and the
existing shared-profile stage-2 compatibility checks. No local VMEC, NTX,
transport solve, full physical AD/FD benchmark or optimization was run.
Physical convergence and the effect on the observed root jumps remain to be
measured by the remote experiment.

Files needed for this change:

- NEOPAX/_orchestrator.py
- NEOPAX/_reverse_ad_parameters.py
- NEOPAX/_reverse_ad_transport.py
- NEOPAX/_reverse_ad_optimization.py
- NEOPAX/optimization.py
- NEOPAX/_species_profile_parameters.py
- examples/optimization/optimize_species_profiles_from_geometry_result.py
- tests/test_species_profile_stage2.py
- docs/benchmarks/species_profiles_stage2.md
