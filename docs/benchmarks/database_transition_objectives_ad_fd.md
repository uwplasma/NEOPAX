# Database transport transition-objective AD/FD comparison

The full-transport shared-payload AD benchmark and realtime-geometry FD runner
now use the same eleven transport objective rows. `Er_transition_left` and
`Er_transition_right` are replaced by `Er_transition_rho`,
`Er_transition_location_moment`, and `Er_transition_strength`. Bootstrap and
net power remain included.

The FD runner imports `FULL_TRANSPORT_OBJECTIVE_LABELS` from the AD benchmark
and evaluates the production scalar value functions. It computes derivatives
by central differences of perturbed endpoint solves, without using the
transport reverse sweep. This checks derivatives through transport and its
boundaries; sharing the primal objective formulas does not independently
validate their physical definitions.

Both benchmarks use the scalar objective defaults: transition radial window
0.25 to 0.75, target/prior 0.5, Er temperature 2.0 kV/m, smoothing epsilon
1e-6, radial softness 0.05, and softmax beta 16. These are benchmark defaults;
they differ from some optimization-script target/window settings.

## Matching 16-step FD command

On the GPU machine, from NEOPAX with `jax_env` active, use the same database
TOML as the AD run:

```bash
env -u JAX_COMPILATION_CACHE_DIR \
JAX_ENABLE_COMPILATION_CACHE=0 \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
/usr/bin/time -v \
python examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml \
  --parameter n0 \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --geometry-fd-lane frozen_linearized \
  --replay-mode accepted \
  --fd-accepted-replay-contract reverse_frozen_metadata \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-fd-root-lane frozen_linearized
```

The selected-root linearization freezes the baseline root branch, matching the
local implicit derivative used by AD. The accepted-replay contract freezes
solver predictor/controller/Jacobian/factorization metadata to the unperturbed
replay, matching the metadata held nondifferentiable by reverse AD. These
existing FD modes are selected explicitly by the command; their defaults have
not changed.

For all eight columns of the AD case, repeat the command with `--parameter`
set to each of:

```text
n0
T0
density_shape_power
temperature_shape_power
density_shape_alpha
temperature_shape_alpha
RBC:1:0
ZBS:1:0
```

Compare FD rows by name with the AD `transport:<name>` residual rows and the
matching parameter column. The AD benchmark uses unit weights and zero targets
for these raw observables, so its transport residual Jacobian equals their
objective Jacobian. The AD run also reports seven geometry-only residual rows;
this FD runner covers the eleven transport rows, not those additional rows.

The FD console and JSON contain eleven objective values and eleven derivative
entries per parameter. JSON reports are written under
`outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/`; the selected
command uses the `_frozen_linearized_summary.json` filename suffix. Keep both
runs' configuration, baseline values, step count, replay mode, selected-root
mode, FD step size, and commit when comparing results.

## Local checks and remaining validation

`tests/test_reverse_benchmark_root_objectives.py` checks the shared row order,
the AD reverse-builder request, FD replay endpoint routing, and synthetic
central differences of the new terminal root rows against local AD. It runs
without VMEC, NTX, or transport solves. It does not validate the complete GPU
Jacobian. The 16-step AD/FD comparison remains pending a remote run.

The updated 16-step AD run has now completed successfully: 18 residual rows,
eight parameter columns, and all 144 Jacobian entries finite. Its
[run note, original output, and CSV reference](database_transition_objectives_ad_16x4_2026-10-14.md)
are archived for comparison. A matching FD result remains pending.
