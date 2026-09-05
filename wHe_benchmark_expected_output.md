# wHe reverse-AD benchmark: expected output

This is the accepted reference captured from the successful wHe exact-Lij
runtime benchmark.  It is the comparison target for the reverse-AD versus
forward-FD table, not a new performance target.

Source capture: `2026-09-02`, saved from the terminal transcript supplied with
this repository task.  The final pasted CLI line was mangled by terminal/chat
copying; the runtime banner below is the authoritative record of the modes
actually used.

## Invocation (as captured)

```bash
cd ~/NEOPAX
env -u JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE=0 \\
PYTHONPATH=~/VMEX:~/NTX/src \\
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \\
  --config ./examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_exact_lagged_runtime_vmec_realtime_geometry_benchmark.toml \\
  --reverse-parameter-mode profiles_plus_realtime_geometry \\
  --reverse-geometry-parameter RBC:1:0,ZBS:1:0 \\
  --realtime-geometry-gradient-path reverse_payload \\
  --objective all \\
  --accepted-step-limit 16 \\
  --radau-jacobian-reuse-mode legacy \\
  --timing-mode jit-warm \\
  --reverse-segment-length 4 \\
  --reverse-stage-adjoint-solve-mode block \\
  --reverse-rhs-transpose-mode explicit_ntx_interpolated \\
  --reverse-step-bwd-mode reduced_cotangent_call_boundary \\
  --reverse-initial-cache-support-pullback-mode ntx_batched_interpolated_faces \\
  --reverse-rebuild-support-pullback-mode ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_product_rule \\
  --reverse-segment-start-replay-mode minimal \\
  --reverse-schedule-artifact-mode reuse_static_probe
```

## Authoritative runtime modes and timing checkpoints

```
runtime build ready                                      306.571 s
solver components ready                                    38.364 s
profile-state VJP ready                                    39.935 s
initial carry VJP ready                                    31.868 s
realized-schedule VJP forward ready                       250.023 s
final-objective cotangents ready                           55.126 s
  ordinary_mode=scalar
  bootstrap_mode=joint_local_vjp_upar_only
segments 4/4, 3/4, 2/4, 1/4          549.932, 86.310, 170.907, 128.769 s
segmented cotangent sweep ready                           935.919 s
initial-cache support pullback ready                      138.932 s
initial state pullback ready                              139.773 s
initial-Er root compact pullback ready                     54.269 s
complete benchmark                                      2438.943 s
```

The log explicitly says: `reusing static schedule artifact (no second adaptive
rollout; no per-step carry tape)`.  At this reference point, it still performed
the fixed accepted-schedule replay, as demonstrated by the 250.023 s
realized-schedule VJP-forward phase.

## Expected transport objective values and geometry derivatives

| Objective | Residual | d/d RBC:1:0 | d/d ZBS:1:0 |
|---|---:|---:|---:|
| softmax_Er | 2.0480064446374552e+01 | -5.1618776959793159e+01 | 1.5440933652230978e+01 |
| net_total_power_volume_average_mw_m3 | 5.0826535340909273e-01 | -6.5270038884536223e-03 | -2.2816855843463312e-03 |
| Er_transition_left | 1.7728040030152442e+01 | -2.0297158863409784e+01 | 1.5384968704595270e-01 |
| Er_transition_right | 1.8374571899363186e+01 | -2.2487555292017525e+01 | 1.0202463119282812e+00 |
| Er2_volume_average | 2.3861197168995284e+02 | -1.3706976025078600e+02 | -1.7102483677617704e+02 |
| Er_volume_average | -3.4490154638731294e+00 | -4.2363292935243607e+01 | 3.0148208968125832e+01 |
| electron_temperature_volume_average_keV | 6.5641846139871820e+00 | -1.2971616088492266e-02 | -3.9976099971202272e-02 |
| total_pressure_volume_average | 3.4212729740002629e+01 | -7.3602390215506810e-02 | -2.3734304102205528e-01 |
| alpha_power_volume_average_mw_m3 | 5.8937225323063791e-01 | -6.6848179322357756e-03 | -2.7521862544646210e-03 |
| bootstrap_current_softmax_abs_scaled | 1.4479788885810612e+00 | -1.7603771655874361e+00 | -6.2377491761315520e+00 |

