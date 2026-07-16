# Ambipolarity Initialization AD Plan

## Goal

Add differentiable support for the initial ambipolar `Er` solve after the transport reverse/profile lane and the realtime-geometry derivative bridge are working.

The intended semantics match the current benchmark staging:

- Current frozen/profile reverse AD obtains the baseline initial `Er` from the normal ambipolar solve.
- Current profile derivatives then treat that initial `Er` as fixed.
- Realtime geometry should first match that behavior: baseline ambipolar `Er` is computed once and copied into the differentiated transport rollout.
- Ambipolar AD is a later layer that differentiates this initial `Er` with respect to profiles and, once the realtime bridge is ready, geometry parameters.

## Mathematical Form

For each radial point, treat the selected ambipolar root as an implicit scalar equation:

```text
F_i(Er_i, p) = 0
```

where `p` may be profile parameters, geometry parameters, or both.

Use implicit differentiation:

```text
dEr_i/dp = - (dF_i/dp) / (dF_i/dEr_i)
```

This should not differentiate through the two-stage scan/root-search control flow. The root search selects the branch; the custom rule differentiates the selected branch.

## Staging

1. Preserve current benchmark behavior.

   Frozen/profile reverse AD and realtime-geometry reverse AD should continue to support a fixed-initial-`Er` mode. This remains the baseline for comparing transport rollout derivatives without ambipolar initialization derivatives.

2. Add a frozen-profile ambipolar derivative prototype.

   Implement a traceable local ambipolar residual `F_i(Er_i, state, geometry, support)` for the existing selected roots. Compute `dF/dEr` and `dF/dprofile` using the same NTX/local flux path used by the initialized transport state. Compare against finite differences that freeze the selected root branch.

3. Add diagnostics.

   Report:

   - selected root value,
   - residual at selected root,
   - `dF/dEr`,
   - small-denominator flags,
   - `dEr/dp`,
   - finite-difference comparison for the same branch.

4. Integrate with profile reverse AD.

   Add an optional mode where `_initial_state_for_parameter_vector(...)` includes the implicit `dEr/dprofile` contribution instead of preserving fixed `Er`. Keep fixed `Er` as the default until validated.

5. Extend to realtime geometry.

   Once the geometry bridge provides compact derivatives through:

   ```text
   geometry parameters -> VMEC/Boozer/NTX payload -> local flux residual
   ```

   reuse the same implicit ambipolar wrapper with `p = profiles + geometry`.

## Implementation Notes

- Freeze the selected root branch from the baseline solve. Do not AD through root enumeration, entropy ranking, or Python/NumPy block orchestration.
- Use the local residual evaluated at the baseline selected root.
- Boundary `Er` entries need explicit conventions:
  - Dirichlet entries have zero derivative unless the boundary value itself is a parameter.
  - Floating/ambipolar edge entries should use the same implicit residual path if they are selected by the initializer.
- Add a denominator floor or diagnostic-only warning for small `abs(dF/dEr)`. Do not silently clip in validation mode.
- Keep the derivative lane separate from the existing initializer until finite-difference checks pass.

## Validation Commands To Add

Start with frozen geometry and one profile parameter:

```bash
python ./examples/benchmarks/benchmark_ambipolar_initial_er_ad.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_benchmark.toml \
  --parameter n0 \
  --radial-index 25 \
  --fd-rel-step 3e-7 \
  --fd-abs-step 1e-10
```

Then test all radial points for one parameter:

```bash
python ./examples/benchmarks/benchmark_ambipolar_initial_er_ad.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_benchmark.toml \
  --parameter n0 \
  --all-radial \
  --fd-rel-step 3e-7 \
  --fd-abs-step 1e-10
```

After the realtime geometry bridge is validated, add geometry parameters:

```bash
python ./examples/benchmarks/benchmark_ambipolar_initial_er_ad.py \
  --config ./examples/benchmarks/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml \
  --parameter RBC:1:0 \
  --all-radial \
  --fd-rel-step 3e-7 \
  --fd-abs-step 1e-10
```

## Success Criteria

- Fixed-`Er` reverse benchmarks remain unchanged.
- Frozen-profile implicit `dEr/dprofile` matches finite differences for the selected root branch.
- Diagnostics identify ill-conditioned roots instead of hiding them.
- Realtime geometry ambipolar AD is only enabled after the compact geometry-to-NTX derivative bridge is validated.
- Final optimization mode can choose either:
  - fixed baseline ambipolar `Er`,
  - differentiable ambipolar initial `Er`.
