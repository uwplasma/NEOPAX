# Database reverse 16/4 no-diagnostics baseline — 2026-09-12

Source: user attachment `79cfff38-9c8b-4b18-a1c4-26166b9c44e8/pasted-text.txt`.
Original text follows; the time footer records the executed command.

```text
cd ~/NEOPAX

env -u JAX_COMPILATION_CACHE_DIR \
JAX_ENABLE_COMPILATION_CACHE=0 \
PYTHONPATH=~/VMEX:~/NTX/src \
/usr/bin/time -v \
python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config ./examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0,ZBS:1:0 \
  --realtime-geometry-gradient-path reverse_payload \
  --objective all \
  --accepted-step-limit 16 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-segment-length 4 \
  --reverse-stage-adjoint-solve-mode block \
  --reverse-rhs-transpose-mode explicit_database \
  --reverse-step-bwd-mode reduced_cotangent_call_boundary \
  --reverse-initial-cache-support-pullback-mode scalar \
  --reverse-rebuild-support-pullback-mode separate \
  --reverse-final-objective-cotangent-mode grouped_vjp \
  --reverse-bootstrap-cotangent-mode joint_local_vjp_upar_only \
  --reverse-schedule-artifact-mode reuse_static_probe
[NEOPAX] built runtime NTX scan database: rho=7 nu_v=16 Er_tilde=11 grid=(25,31,64) backend=vmec
[autodiff-gate] progress: realtime geometry runtime build ready elapsed_s=646.782
[autodiff-gate] progress: realtime geometry solver components ready elapsed_s=0.566
[autodiff-gate] realtime geometry device: default_backend=gpu baseline_values_device=cuda:0 local_devices=['cuda:0']
[autodiff-gate] progress: running realtime geometry optimization API smoke
[autodiff-gate] progress: full-transport reverse stage-adjoint solve mode=block rhs_pullback_mode=separate initial_cache_support_pullback_mode=scalar rebuild_support_pullback_mode=separate segment_jit_diagnostics=False segment_input_diagnostics=False segment_start_replay_mode=minimal segment_primal_record_mode=reuse_segment_primal_record step_bwd_mode=reduced_cotangent_call_boundary
E0912 06:23:40.705004   53744 slow_operation_alarm.cc:73]
********************************
[Compiling module jit_scan for GPU] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
E0912 06:24:12.423666   53703 slow_operation_alarm.cc:140] The operation took 2m31.718771361s

********************************
[Compiling module jit_scan for GPU] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
[autodiff-gate] progress: support reverse profile-state vjp ready elapsed_s=116.223
[autodiff-gate] progress: support reverse initial carry vjp ready elapsed_s=0.918
[autodiff-gate] progress: support reverse reusing static schedule artifact (no second adaptive rollout; no per-step carry tape)
[autodiff-gate] progress: support reverse realized-schedule vjp forward ready elapsed_s=2.102
[autodiff-gate] progress: support reverse final-objective cotangents ready elapsed_s=320.216 ordinary_mode=grouped_vjp bootstrap_mode=joint_local_vjp_upar_only
[autodiff-gate] progress: support reverse segmented cotangent sweep start segments=4 segment_length=4 objectives=10 cotangent_mode=full
[autodiff-gate] progress: database segment reverse uses fixed-table table transpose plus compact local fixed-table geometry bars; only table bars cross the recorded scan (no scan owner in segments)
[autodiff-gate] progress: support reverse segment 4/4 ready elapsed_s=1067.933 active_steps=4 support_reuse=0 support_rebuild=4
[autodiff-gate] progress: support reverse segment 3/4 ready elapsed_s=22.907 active_steps=4 support_reuse=0 support_rebuild=4
[autodiff-gate] progress: support reverse segment 2/4 ready elapsed_s=23.099 active_steps=4 support_reuse=0 support_rebuild=4
[autodiff-gate] progress: support reverse segment 1/4 ready elapsed_s=22.433 active_steps=4 support_reuse=0 support_rebuild=4
[autodiff-gate] progress: support reverse segmented cotangent sweep ready elapsed_s=1136.374 support_reuse=0 support_rebuild=16
[autodiff-gate] progress: support reverse reduced carry bars expanded ready elapsed_s=0.704
E0912 06:54:22.383541   53744 slow_operation_alarm.cc:73]
********************************
[Compiling module jit__pullback for GPU] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
E0912 06:55:29.919719   53703 slow_operation_alarm.cc:140] The operation took 3m7.536269204s

********************************
[Compiling module jit__pullback for GPU] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
[autodiff-gate] progress: support reverse initial direct-RHS support pullback ready elapsed_s=296.198
[autodiff-gate] progress: support reverse initial state pullback ready elapsed_s=182.195
[autodiff-gate] progress: initial-Er root boundary compact pullback ready elapsed_s=280.169
[autodiff-gate] progress: support reverse profile parameter pullback ready elapsed_s=0.960
[autodiff-gate] progress: support reverse initial-profile scan payload pullback ready elapsed_s=1.468
[autodiff-gate] progress: database final recorded-scan fold ready elapsed_s=770.232 objective_rows=10 groups=1 contract=one_batched_scan_transpose
[autodiff-gate] full-transport shared payload: ntx_scan_runtime_active_float_leaves=63
[autodiff-gate] full-transport shared payload: compact_payload_tangent_contract=True native_vmec_coefficient_tangent_contract=False
[autodiff-gate] full-transport shared payload: raw_block_param_bar_l2=4.453275e+02 raw_block_param_bar_all_finite=True raw_block_param_bar_first_nonfinite=None
[geometry-fd-ad] progress: objective_table vmec implicit state/raw-block aux ready elapsed_s=0.012
[geometry-fd-ad] progress: objective_table booz input tables ready elapsed_s=2.348
[geometry-fd-ad] progress: objective_table booz_xform vjp ready elapsed_s=8.907
[geometry-fd-ad] progress: objective_table vmec objective cotangents ready elapsed_s=18.757
[geometry-fd-ad] progress: objective_table DMerc softmax cotangent ready elapsed_s=12.440
[geometry-fd-ad] progress: objective_table boozer light cotangents ready elapsed_s=3.271
[geometry-fd-ad] progress: objective_table aspect proxy cotangents ready elapsed_s=0.103
[geometry-fd-ad] progress: objective_table j-qi/maxj Boozer cotangents ready elapsed_s=34.504
[geometry-fd-ad] progress: objective_table booz cotangents pulled to state elapsed_s=9.721
[geometry-fd-ad] progress: objective_table final vmec parameter pullback ready elapsed_s=51.323
[autodiff-gate] mode=transport_reverse_ad_only_full_transport_shared_payload_smoke objective=all residual_count=17 parameter_count=8 elapsed_s=3805.329
[autodiff-gate] optimization API residuals/Jacobian rows:
  - transport:softmax_Er: residual=2.1347597245906741e+01
      dtransport:softmax_Er/dn0: jac=-2.8702292832128968e+00
      dtransport:softmax_Er/dT0: jac=2.7740118826951718e+00
      dtransport:softmax_Er/ddensity_shape_power: jac=-8.0970009092364181e-02
      dtransport:softmax_Er/dtemperature_shape_power: jac=1.8718548957145615e+00
      dtransport:softmax_Er/ddensity_shape_alpha: jac=1.5312954921315222e-01
      dtransport:softmax_Er/dtemperature_shape_alpha: jac=1.0768407297399117e+01
      dtransport:softmax_Er/dvmec:RBC:1:0: jac=-2.7683707281136662e+01
      dtransport:softmax_Er/dvmec:ZBS:1:0: jac=6.3556954390710434e+00
  - transport:net_total_power_volume_average_mw_m3: residual=5.0807530042924676e-01
      dtransport:net_total_power_volume_average_mw_m3/dn0: jac=2.4144994375404702e-01
      dtransport:net_total_power_volume_average_mw_m3/dT0: jac=8.1012117712766646e-02
      dtransport:net_total_power_volume_average_mw_m3/ddensity_shape_power: jac=1.0015251511699052e-03
      dtransport:net_total_power_volume_average_mw_m3/dtemperature_shape_power: jac=2.7450389881309428e-01
      dtransport:net_total_power_volume_average_mw_m3/ddensity_shape_alpha: jac=-2.4504760628385079e-04
      dtransport:net_total_power_volume_average_mw_m3/dtemperature_shape_alpha: jac=-3.9390935353161921e-01
      dtransport:net_total_power_volume_average_mw_m3/dvmec:RBC:1:0: jac=5.7487580005064010e-05
      dtransport:net_total_power_volume_average_mw_m3/dvmec:ZBS:1:0: jac=-6.4216000806514321e-03
  - transport:Er_transition_left: residual=1.7902257301111909e+01
      dtransport:Er_transition_left/dn0: jac=-9.8540288633937534e-01
      dtransport:Er_transition_left/dT0: jac=1.5889831820635036e+00
      dtransport:Er_transition_left/ddensity_shape_power: jac=-1.2265911027762052e-02
      dtransport:Er_transition_left/dtemperature_shape_power: jac=-7.1687550637460378e+00
      dtransport:Er_transition_left/ddensity_shape_alpha: jac=1.5260136870844827e-02
      dtransport:Er_transition_left/dtemperature_shape_alpha: jac=1.5568052949830751e+01
      dtransport:Er_transition_left/dvmec:RBC:1:0: jac=-1.3445891807357155e+01
      dtransport:Er_transition_left/dvmec:ZBS:1:0: jac=-8.6697187673722109e-01
  - transport:Er_transition_right: residual=1.8666769949428140e+01
      dtransport:Er_transition_right/dn0: jac=-1.0998955496995126e+00
      dtransport:Er_transition_right/dT0: jac=1.6914404449598357e+00
      dtransport:Er_transition_right/ddensity_shape_power: jac=-1.7086805199196833e-02
      dtransport:Er_transition_right/dtemperature_shape_power: jac=-6.2814820735972869e+00
      dtransport:Er_transition_right/ddensity_shape_alpha: jac=2.2559732570020176e-02
      dtransport:Er_transition_right/dtemperature_shape_alpha: jac=1.5577054914522920e+01
      dtransport:Er_transition_right/dvmec:RBC:1:0: jac=-1.5274580237777851e+01
      dtransport:Er_transition_right/dvmec:ZBS:1:0: jac=-5.0636952455207840e-01
  - transport:Er2_volume_average: residual=2.6970444513292227e+02
      dtransport:Er2_volume_average/dn0: jac=2.5842589258007074e+00
      dtransport:Er2_volume_average/dT0: jac=3.1320424054222428e+01
      dtransport:Er2_volume_average/ddensity_shape_power: jac=2.2824336263282099e+00
      dtransport:Er2_volume_average/dtemperature_shape_power: jac=-1.1938297555500746e+01
      dtransport:Er2_volume_average/ddensity_shape_alpha: jac=3.0861923809212257e+00
      dtransport:Er2_volume_average/dtemperature_shape_alpha: jac=1.0694492154833915e+02
      dtransport:Er2_volume_average/dvmec:RBC:1:0: jac=-2.9602462298898990e+02
      dtransport:Er2_volume_average/dvmec:ZBS:1:0: jac=-3.3046442994759906e+02
  - transport:Er_volume_average: residual=-2.8511644406025951e+00
      dtransport:Er_volume_average/dn0: jac=-1.9024031619531510e+00
      dtransport:Er_volume_average/dT0: jac=8.6605462867946792e-01
      dtransport:Er_volume_average/ddensity_shape_power: jac=-6.8688993949557423e-02
      dtransport:Er_volume_average/dtemperature_shape_power: jac=-6.9405878986901259e-01
      dtransport:Er_volume_average/ddensity_shape_alpha: jac=-1.6832079944450518e-01
      dtransport:Er_volume_average/dtemperature_shape_alpha: jac=3.0165805596904787e+00
      dtransport:Er_volume_average/dvmec:RBC:1:0: jac=-6.8360480192986559e+00
      dtransport:Er_volume_average/dvmec:ZBS:1:0: jac=1.4302240534246726e+01
  - transport:electron_temperature_volume_average_keV: residual=6.5667321685714866e+00
      dtransport:electron_temperature_volume_average_keV/dn0: jac=8.6352302266201608e-04
      dtransport:electron_temperature_volume_average_keV/dT0: jac=3.5577917899824130e-01
      dtransport:electron_temperature_volume_average_keV/ddensity_shape_power: jac=-7.1344857712301452e-05
      dtransport:electron_temperature_volume_average_keV/dtemperature_shape_power: jac=1.5248015861809205e+00
      dtransport:electron_temperature_volume_average_keV/ddensity_shape_alpha: jac=1.2218646390351195e-03
      dtransport:electron_temperature_volume_average_keV/dtemperature_shape_alpha: jac=-3.0449272577867226e+00
      dtransport:electron_temperature_volume_average_keV/dvmec:RBC:1:0: jac=-1.6115942109770211e-02
      dtransport:electron_temperature_volume_average_keV/dvmec:ZBS:1:0: jac=-3.9990631490992204e-02
  - transport:total_pressure_volume_average: residual=3.4217392413639516e+01
      dtransport:total_pressure_volume_average/dn0: jac=8.0624286614427199e+00
      dtransport:total_pressure_volume_average/dT0: jac=1.8654173307789694e+00
      dtransport:total_pressure_volume_average/ddensity_shape_power: jac=2.4426699648199540e-01
      dtransport:total_pressure_volume_average/dtemperature_shape_power: jac=7.7523359564773786e+00
      dtransport:total_pressure_volume_average/ddensity_shape_alpha: jac=-1.3265053153685458e+00
      dtransport:total_pressure_volume_average/dtemperature_shape_alpha: jac=-1.4516604356364713e+01
      dtransport:total_pressure_volume_average/dvmec:RBC:1:0: jac=-7.3305802256671312e-02
      dtransport:total_pressure_volume_average/dvmec:ZBS:1:0: jac=-2.3761691183351247e-01
  - transport:alpha_power_volume_average_mw_m3: residual=5.8919528044396130e-01
      dtransport:alpha_power_volume_average_mw_m3/dn0: jac=2.7962100211621660e-01
      dtransport:alpha_power_volume_average_mw_m3/dT0: jac=8.3216104055209142e-02
      dtransport:alpha_power_volume_average_mw_m3/ddensity_shape_power: jac=2.3609178836717423e-03
      dtransport:alpha_power_volume_average_mw_m3/dtemperature_shape_power: jac=2.8420414560668772e-01
      dtransport:alpha_power_volume_average_mw_m3/ddensity_shape_alpha: jac=-7.6383506297708842e-03
      dtransport:alpha_power_volume_average_mw_m3/dtemperature_shape_alpha: jac=-4.1289868849445077e-01
      dtransport:alpha_power_volume_average_mw_m3/dvmec:RBC:1:0: jac=-1.1720032048136180e-04
      dtransport:alpha_power_volume_average_mw_m3/dvmec:ZBS:1:0: jac=-6.8872114596785893e-03
  - transport:bootstrap_current_softmax_abs_scaled: residual=1.3717342708882792e+00
      dtransport:bootstrap_current_softmax_abs_scaled/dn0: jac=1.0697658604014659e-01
      dtransport:bootstrap_current_softmax_abs_scaled/dT0: jac=1.7479404655582315e-01
      dtransport:bootstrap_current_softmax_abs_scaled/ddensity_shape_power: jac=-1.1999309293285526e-02
      dtransport:bootstrap_current_softmax_abs_scaled/dtemperature_shape_power: jac=7.0788983350295842e-01
      dtransport:bootstrap_current_softmax_abs_scaled/ddensity_shape_alpha: jac=4.1575985810500586e-02
      dtransport:bootstrap_current_softmax_abs_scaled/dtemperature_shape_alpha: jac=-6.1967238271409819e-01
      dtransport:bootstrap_current_softmax_abs_scaled/dvmec:RBC:1:0: jac=-2.1363564988892496e+00
      dtransport:bootstrap_current_softmax_abs_scaled/dvmec:ZBS:1:0: jac=-1.2135493890455427e+00
  - geometry:boozer_qi_objective: residual=2.1192029964274445e-01
      dgeometry:boozer_qi_objective/dn0: jac=0.0000000000000000e+00
      dgeometry:boozer_qi_objective/dT0: jac=0.0000000000000000e+00
      dgeometry:boozer_qi_objective/ddensity_shape_power: jac=0.0000000000000000e+00
      dgeometry:boozer_qi_objective/dtemperature_shape_power: jac=0.0000000000000000e+00
      dgeometry:boozer_qi_objective/ddensity_shape_alpha: jac=0.0000000000000000e+00
      dgeometry:boozer_qi_objective/dtemperature_shape_alpha: jac=0.0000000000000000e+00
      dgeometry:boozer_qi_objective/dvmec:RBC:1:0: jac=5.9392891187203531e+00
      dgeometry:boozer_qi_objective/dvmec:ZBS:1:0: jac=-1.2365550092340527e-01
  - geometry:boozer_maxj_objective: residual=4.4387332574094023e+02
      dgeometry:boozer_maxj_objective/dn0: jac=0.0000000000000000e+00
      dgeometry:boozer_maxj_objective/dT0: jac=0.0000000000000000e+00
      dgeometry:boozer_maxj_objective/ddensity_shape_power: jac=0.0000000000000000e+00
      dgeometry:boozer_maxj_objective/dtemperature_shape_power: jac=0.0000000000000000e+00
      dgeometry:boozer_maxj_objective/ddensity_shape_alpha: jac=0.0000000000000000e+00
      dgeometry:boozer_maxj_objective/dtemperature_shape_alpha: jac=0.0000000000000000e+00
      dgeometry:boozer_maxj_objective/dvmec:RBC:1:0: jac=-3.8431351877821144e+03
      dgeometry:boozer_maxj_objective/dvmec:ZBS:1:0: jac=-1.9205082804683188e+03
  - geometry:vmec_aspect_ratio: residual=1.0015330918957178e+01
      dgeometry:vmec_aspect_ratio/dn0: jac=0.0000000000000000e+00
      dgeometry:vmec_aspect_ratio/dT0: jac=0.0000000000000000e+00
      dgeometry:vmec_aspect_ratio/ddensity_shape_power: jac=0.0000000000000000e+00
      dgeometry:vmec_aspect_ratio/dtemperature_shape_power: jac=0.0000000000000000e+00
      dgeometry:vmec_aspect_ratio/ddensity_shape_alpha: jac=0.0000000000000000e+00
      dgeometry:vmec_aspect_ratio/dtemperature_shape_alpha: jac=0.0000000000000000e+00
      dgeometry:vmec_aspect_ratio/dvmec:RBC:1:0: jac=-5.4006784187006147e+00
      dgeometry:vmec_aspect_ratio/dvmec:ZBS:1:0: jac=-5.5226885751318529e+00
  - geometry:vmec_iota_mean: residual=-5.9365259966101458e-01
      dgeometry:vmec_iota_mean/dn0: jac=0.0000000000000000e+00
      dgeometry:vmec_iota_mean/dT0: jac=0.0000000000000000e+00
      dgeometry:vmec_iota_mean/ddensity_shape_power: jac=0.0000000000000000e+00
      dgeometry:vmec_iota_mean/dtemperature_shape_power: jac=0.0000000000000000e+00
      dgeometry:vmec_iota_mean/ddensity_shape_alpha: jac=0.0000000000000000e+00
      dgeometry:vmec_iota_mean/dtemperature_shape_alpha: jac=0.0000000000000000e+00
      dgeometry:vmec_iota_mean/dvmec:RBC:1:0: jac=2.4405140609263865e-01
      dgeometry:vmec_iota_mean/dvmec:ZBS:1:0: jac=1.4567526019820762e-01
  - geometry:vmec_magnetic_well: residual=-2.7476128749679612e-02
      dgeometry:vmec_magnetic_well/dn0: jac=0.0000000000000000e+00
      dgeometry:vmec_magnetic_well/dT0: jac=0.0000000000000000e+00
      dgeometry:vmec_magnetic_well/ddensity_shape_power: jac=0.0000000000000000e+00
      dgeometry:vmec_magnetic_well/dtemperature_shape_power: jac=0.0000000000000000e+00
      dgeometry:vmec_magnetic_well/ddensity_shape_alpha: jac=0.0000000000000000e+00
      dgeometry:vmec_magnetic_well/dtemperature_shape_alpha: jac=0.0000000000000000e+00
      dgeometry:vmec_magnetic_well/dvmec:RBC:1:0: jac=-1.1090116065531674e-02
      dgeometry:vmec_magnetic_well/dvmec:ZBS:1:0: jac=-4.1682027482238482e-02
  - geometry:vmec_mirror_ratio: residual=2.1153803467163693e-01
      dgeometry:vmec_mirror_ratio/dn0: jac=0.0000000000000000e+00
      dgeometry:vmec_mirror_ratio/dT0: jac=0.0000000000000000e+00
      dgeometry:vmec_mirror_ratio/ddensity_shape_power: jac=0.0000000000000000e+00
      dgeometry:vmec_mirror_ratio/dtemperature_shape_power: jac=0.0000000000000000e+00
      dgeometry:vmec_mirror_ratio/ddensity_shape_alpha: jac=0.0000000000000000e+00
      dgeometry:vmec_mirror_ratio/dtemperature_shape_alpha: jac=0.0000000000000000e+00
      dgeometry:vmec_mirror_ratio/dvmec:RBC:1:0: jac=-5.9359094714046601e-01
      dgeometry:vmec_mirror_ratio/dvmec:ZBS:1:0: jac=4.1437006125401626e-01
  - geometry:vmec_dmerc_stability_softmax: residual=3.2329612235882421e+00
      dgeometry:vmec_dmerc_stability_softmax/dn0: jac=0.0000000000000000e+00
      dgeometry:vmec_dmerc_stability_softmax/dT0: jac=0.0000000000000000e+00
      dgeometry:vmec_dmerc_stability_softmax/ddensity_shape_power: jac=0.0000000000000000e+00
      dgeometry:vmec_dmerc_stability_softmax/dtemperature_shape_power: jac=0.0000000000000000e+00
      dgeometry:vmec_dmerc_stability_softmax/ddensity_shape_alpha: jac=0.0000000000000000e+00
      dgeometry:vmec_dmerc_stability_softmax/dtemperature_shape_alpha: jac=0.0000000000000000e+00
      dgeometry:vmec_dmerc_stability_softmax/dvmec:RBC:1:0: jac=-7.2859943466255244e+00
      dgeometry:vmec_dmerc_stability_softmax/dvmec:ZBS:1:0: jac=-1.5178996489713270e+00
Wrote outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_full_transport_shared_payload_smoke.json
	Command being timed: "python ./examples/benchmarks/benchmark_transport_reverse_ad_only.py --config ./examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml --reverse-parameter-mode profiles_plus_realtime_geometry --reverse-geometry-parameter RBC:1:0,ZBS:1:0 --realtime-geometry-gradient-path reverse_payload --objective all --accepted-step-limit 16 --radau-jacobian-reuse-mode legacy --timing-mode jit-warm --reverse-segment-length 4 --reverse-stage-adjoint-solve-mode block --reverse-rhs-transpose-mode explicit_database --reverse-step-bwd-mode reduced_cotangent_call_boundary --reverse-initial-cache-support-pullback-mode scalar --reverse-rebuild-support-pullback-mode separate --reverse-final-objective-cotangent-mode grouped_vjp --reverse-bootstrap-cotangent-mode joint_local_vjp_upar_only --initial-Er-root-ad jax_selected_root --full-transport-shared-payload-smoke --reverse-schedule-artifact-mode reuse_static_probe"
	User time (seconds): 5523.60
	System time (seconds): 274.81
	Percent of CPU this job got: 121%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 1:19:27
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 15051068
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 0
	Minor (reclaiming a frame) page faults: 19206145
	Voluntary context switches: 7275540
	Involuntary context switches: 512146
	Swaps: 0
	File system inputs: 0
	File system outputs: 561584
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
```
