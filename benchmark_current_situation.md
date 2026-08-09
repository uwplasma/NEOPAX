# Current Benchmark Situation

This file records the current benchmark state for the AD/FD validation work around shared-payload optimization internals. The detailed numeric comparison tables are saved in:

- `optimization_ad_vs_fd.md`
- `shared_payload_fd_comparison.md`

## Protected Baseline

- The benchmark AD paths are treated as reference behavior. Do not change them while working on optimization plumbing unless explicitly agreed first.
- Optimization internals should reproduce the benchmark AD graph/numerics, especially the compact root/transport pullbacks and raw-block geometry pullback behavior.
- The fused/shared optimization path should reuse the same geometry state/payload where possible, but not change the differentiated physics path.

## Validated Geometry Objectives

Geometry-only frozen-linearized FD checks using `input.QI_nfp2_newNT_opt_hires_true` have been run for `RBC:1:0`.

| Objective | Value | Frozen Linearized FD | Forward JVP | Internal Reverse Grad | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `boozer_qi_objective` | `3.2109136309506803e-03` | `8.9996041589633161e-02` | `8.9989229074526111e-02` | `8.9989229072212851e-02` | AD/JVP match; FD close |
| `boozer_maxj_objective` | `1.3389843913753766e-01` | `-1.1591364898936340e+00` | `-1.1593167988198267e+00` | `-1.1593167987078914e+00` | AD/JVP match; FD close |

Commands:

```bash
python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py \
  --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true \
  --parameter RBC:1:0 \
  --objective boozer_qi_objective \
  --multigrid \
  --forward-linear-solve-mode raw_block \
  --forward-linear-maxiter 300 \
  --adjoint-maxiter 300
```

```bash
python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py \
  --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true \
  --parameter RBC:1:0 \
  --objective boozer_maxj_objective \
  --multigrid \
  --forward-linear-solve-mode raw_block \
  --forward-linear-maxiter 300 \
  --adjoint-maxiter 300
```

## Validated Initial-Er Root Only

Root-only shared payload AD/FD checks include:

- `softmax_Er`
- `smooth_root_proxy`
- `Er_transition_left`
- `Er_transition_right`
- `Er2_volume_average`
- `Er_volume_average`
- `bootstrap_current_softmax_abs_scaled`

The corrected frozen-linearized root FD lane is needed for stable comparisons of root-branch-sensitive volume averages.

Example FD command:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter n0 \
  --geometry-fd-lane frozen_linearized \
  --radau-jacobian-reuse-mode legacy \
  --initial-Er-root-ad jax_selected_root \
  --initial-Er-root-only-fd \
  --initial-Er-root-only-fd-root-lane frozen_linearized \
  --root-only-objective all
```

Known root-only FD examples:

| Parameter | Objective | FD |
| --- | --- | ---: |
| `RBC:1:0` | `bootstrap_current_softmax_abs_scaled` | `-1.705882e+00` |
| `T0` | `bootstrap_current_softmax_abs_scaled` | `2.162376e-01` |
| `density_shape_power` | `bootstrap_current_softmax_abs_scaled` | `-1.229510e-02` |
| `temperature_shape_power` | `bootstrap_current_softmax_abs_scaled` | `1.439570e+00` |
| `n0` | `bootstrap_current_softmax_abs_scaled` | `-2.147247e-03` |

## Full Transport Shared Payload

The 2-step shared-payload AD smoke ran successfully without OOM and used optimization internals.

Command:

```bash
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0 \
  --realtime-geometry-gradient-path reverse_payload \
  --ntx-exact-derivative-mode direct \
  --ntx-exact-derivative-field-pullback-mode generic_jvp \
  --objective all \
  --accepted-step-limit 2 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 1 \
  --reverse-stage-adjoint-solve-mode bicgstab \
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \
  --reverse-step-bwd-mode reduced_cotangent \
  --initial-Er-root-ad jax_selected_root \
  --full-transport-shared-payload-smoke
```

The 16-step shared-payload AD result has been compared against the corrected 16-step FD for `RBC:1:0`.

| Objective | AD d/d`RBC:1:0` | FD d/d`RBC:1:0` | Abs Diff | Rel Diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `softmax_Er` | `-6.2626550858639305e+01` | `-6.262732e+01` | `7.691414e-04` | `1.228124e-05` | good |
| `smooth_root_proxy` | `-3.6780535157069163e-04` | `-5.576429e-04` | `1.898375e-04` | `3.404285e-01` | small derivative |
| `Er_transition_left` | `-2.0130902762944103e+01` | `-2.013024e+01` | `6.627629e-04` | `3.292375e-05` | good |
| `Er_transition_right` | `-2.2349701278432580e+01` | `-2.234945e+01` | `2.512784e-04` | `1.124316e-05` | good |
| `Er2_volume_average` | `-2.7118433131494726e+02` | `-2.712342e+02` | `4.986869e-02` | `1.838584e-04` | good |
| `Er_volume_average` | `-2.3645736589729140e+01` | `-2.364403e+01` | `1.706590e-03` | `7.217846e-05` | good |
| `electron_temperature_volume_average_keV` | `-2.2448477914450340e-02` | `-2.244865e-02` | `1.720855e-07` | `7.665742e-06` | good |
| `total_pressure_volume_average` | `-7.7520438710911577e-02` | `-7.752102e-02` | `5.812891e-07` | `7.498471e-06` | good |
| `alpha_power_volume_average_mw_m3` | `1.2073914238870094e-03` | `1.207393e-03` | `1.576113e-09` | `1.305385e-06` | good |
| `bootstrap_current_softmax_abs_scaled` | `-1.7920897096814259e+00` | `-1.791772e+00` | `3.177097e-04` | `1.773159e-04` | good |

16-step FD command:

```bash
python ./examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter RBC:1:0 \
  --geometry-fd-lane frozen_linearized \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --replay-mode accepted \
  --initial-Er-root-ad jax_selected_root
```

## Current Open Items

- The latest attempted 16-step shared-payload reverse-AD rerun after the
  finite-volume boundary/evaluated-state changes failed before producing
  objective/Jacobian rows. It reached:
  `runtime build = 301.307 s`, `solver components = 119.625 s`,
  `profile-state VJP = 120.077 s`, `initial-carry VJP = 56.074 s`,
  `realized-schedule VJP forward = 775.634 s`, then failed at segmented
  cotangent sweep start with a JAX `float0` addition in
  `ComposedEquationSystem._shared_fluxes_add`.
- A narrow reverse-bookkeeping patch now sanitizes `float0` cotangent leaves
  to real zero bars in shared-flux cotangent addition and in
  `_sanitize_float_delta_bar_tree`. This does not change the forward physics,
  finite-volume equations, BCs, or NTX evaluation path. It should be retested
  with the same 16-step shared-payload reverse command before comparing
  profile-column AD against the fresh FD rows.
- New 8-parameter full-transport shared-payload reverse-AD snapshot is saved
  in `shared_payload_8param_benchmark_snapshot.md`. This run uses
  `density_shape_alpha`, `temperature_shape_alpha`, `RBC:1:0`, and `ZBS:1:0`,
  so it should be validated against newly rerun FD references instead of the
  older 5-parameter table.
- New current QI frozen-linearized FD for `RBC:1:0` is saved in the 8-parameter
  snapshot: FD `5.9397387449796542e+00`, forward JVP
  `5.9392891189187216e+00`, optimization-internal reverse
  `5.9392891187135319e+00`, shared-payload reverse
  `5.9392927523725234e+00`.
- New current QI frozen-linearized FD for `ZBS:1:0` is saved in the
  8-parameter snapshot: FD `-1.2367491295362752e-01`, forward JVP
  `-1.2365550023172692e-01`, optimization-internal reverse
  `-1.2365550092181365e-01`, shared-payload reverse
  `-1.2365499862085017e-01`.
- New current maxJ frozen-linearized FD for `RBC:1:0` is saved in the
  8-parameter snapshot: FD `-3.8425374640355458e+03`, forward JVP
  `-3.8431351880877260e+03`, optimization-internal reverse
  `-3.8431351877333364e+03`, shared-payload reverse
  `-3.8431400490597589e+03`.
- New current maxJ frozen-linearized FD for `ZBS:1:0` is saved in the
  8-parameter snapshot: FD `-1.9205101180950553e+03`, forward JVP
  `-1.9205082810004324e+03`, optimization-internal reverse
  `-1.9205082803893893e+03`, shared-payload reverse
  `-1.9205081921396923e+03`.
- Save matching 2-step FD references if we want a formal 2-step AD-vs-FD table.
- Add full-transport FD references for selected profile parameters if needed; current exact 16-step table is geometry `RBC:1:0`.
- Keep bootstrap-current objective in the root/full benchmark tables, but be careful not to reintroduce generic full flux VJPs that caused OOM.
- Continue optimization-fusion work only through internals, preserving benchmark AD behavior.
