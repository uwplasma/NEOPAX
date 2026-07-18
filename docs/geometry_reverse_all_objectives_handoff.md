# Geometry Reverse All-Objectives Handoff

Goal for next session: restore the `geometry_full_ad_objectives` all-objective reverse table so it again gives the correct compact reverse derivatives.

Hard requirements:
- Use a compact grouped reverse rule.
- Do not replace the rule with forward JVP/tangent columns.
- Do not degroup into scalar objective sweeps as the final implementation.
- Do not use tolerance or finite-difference workarounds.
- Do not contaminate the forward solver realtime/frozen geometry paths.

Trusted diagnostic:
- Script: `examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py`
- Command:
  `python ./examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true --parameter RBC:1:0 --objective boozer_qi_objective --reference-fd=5.909044e-01 --multigrid`
- Important output:
  `forward_jvp=6.9347871114079501e-03`
  `reverse_state_dot_tangent=6.9347871114111004e-03`
  `implicit_reverse_param_grad=7.0168957131144793e-03`

Interpretation:
- `reverse_state_dot_tangent` is the scalar contraction `<dQI/dVMEC_state, dVMEC_state/dRBC:1:0>`.
- It is not the state cotangent itself.
- `reverse_state_dot_tangent` and `implicit_reverse_param_grad` are both intended to represent `dQI/dRBC:1:0`.
- They should match; the remaining gap means the reverse parameter pullback/table reconstruction still has a bug.

Layering:
- Boundary harmonic `p` feeds the VMEC implicit solve to produce state `x(p)`.
- Boozer is explicit: `booz(x)`.
- QI is explicit: `QI(booz(x))`.
- Correct reverse chain:
  `dQI/dbooz -> Boozer VJP -> dQI/dVMEC_state -> VMEC-JAX implicit reverse -> dQI/dharmonic`.
- Only the final VMEC state-cotangent-to-parameter step is the VMEC implicit reverse rule.
- The problem is not that implicit is applied to Boozer; the problem is the compact/grouped propagation into the VMEC implicit parameter pullback.

Current suspect path:
- File: `NEOPAX/_geometry_autodiff.py`
- Function: `geometry_full_ad_objective_table_pullback_from_param_vector`
- Compare against scalar weighted path: `geometry_observable_weighted_sum_from_param_vector`.
- The grouped table currently constructs a reduced Boozer output tree, combines light-Boozer and QI cotangents, then calls one `booz_state_pullback` before the VMEC implicit pullback.
- The next fix should make this grouped Boozer cotangent path match the scalar objective path structurally, while preserving compact grouped reverse behavior.

Benchmark command:
`python ./examples/benchmarks/benchmark_geometry_vmec_booz_fd_vs_ad.py --mode geometry_full_ad_objectives --vmec-input ./examples/inputs/input.QI_nfp2_newNT_opt_hires_true --param-specs RBC:1:0,ZBS:1:0 --fd-rel-step 3e-7 --fd-abs-step 1e-10 --ad-backend implicit --fd-lane ad --reverse-derivative-mode objective_table --skip-fd-check`

Recent bad/suspect outputs:
- Worse contaminated QI row: `dboozer_qi_objective/dRBC:1:0 ad=7.307171e-03`.
- Earlier closer table values were around `6.78e-03`.
- Target scalar diagnostic for `RBC:1:0` is `6.9347871114111004e-03`.

Before editing next session:
- Run `git diff -- NEOPAX/_geometry_autodiff.py examples/benchmarks/benchmark_geometry_vmec_booz_fd_vs_ad.py`.
- Preserve the removal of the OOM-prone compact diagnostic helper in `examples/benchmarks/compare_geometry_qi_frozen_linearized_fd.py` unless explicitly revisiting it.
