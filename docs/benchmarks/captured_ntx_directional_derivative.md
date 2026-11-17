# NTX derivative check along the captured AD16 direction

The transferred archive of `outputs/axis_fix_ad_fd16_20261116_RPTPxK`
has SHA256 `f153576ee2d9775239aef754833909871278e891c2de6e7b3eccc1e93fde2e3f`.
It records NEOPAX `f45cdec536a1c06b64f8e61d66c7746e61514d36`,
VMEX `e5190622aa7bbbb1ab5aca9767ade122eac63e2e`,
NTX `4f7aa88044ea608bc2167603a4eb2302d0af7b32`,
and SOLVAX `01e65eb9f42752f36c2b2b21647349398079541b`.

Offline NumPy inspection confirms byte-identical initial packed state, geometry
and coefficient tables for both FD baselines versus the AD snapshot. RBC/ZBS
full saved initial density, pressure and Er also match each other byte for byte.
Their accept/reject masks agree. The first saved error-estimate difference is
at attempt 6 (`4.0657581468206416e-20`); the first next-dt difference is after
attempt 11 (`-8.179797171634778e-17`). The resulting timestep drift explains
why the two 16-step baselines do not terminate at the same time. The saved
arrays do not identify the first underlying floating-point operation.

All 26 saved compact geometry contractions reconstruct the reported transport
geometry gradients to numerical roundoff. The six support component groups sum
to the saved pre-scan support bars to roundoff. This checks assembly and the
final stored contraction, not the correctness of the NTX or transport transpose.

The largest explicit-bootstrap table cotangents occur near radius index 4
(`rho=0.625`), nu index 4 (`nu/v=3e-5`). Both D13 and D33 bars are nonzero:
their norms are `1.373903236e1` and `2.223635751e0`. The largest D13 entry is
`6.347442170` at `(4,4,2)`. Initial-root and transport contributions to the
transition metrics have significant D11-log bars near `(4,4,3)` and `(4,4,5)`.
These nodes differ from the outer, lowest-collision points used by the earlier
head/tail audit. Large cotangents choose relevant checks; they do not establish
that those nodes are inaccurate.

There is a concrete numerical distinction to check in NTX. The forward
`solve_prepared_coefficient_vector` uses `_solve_modes_with_tail_residual`,
eliminating the high modes while retaining three. The production coefficient
pullback reconstructs its primal with `_prepared_implicit_vjp_primal`, which
factors and solves all modes. Both solve the same equations; their finite
precision primal and directional derivative agreement at the important nodes
has not yet been established by this offline inspection.

`check_captured_ntx_directional_derivative.py` tests that distinction on the GPU:

1. Reconstruct numerical NTX surfaces from the captured equilibrium state;
   restore the saved seven surface-field directions and electric-field direction.
   No VMEC equilibrium is solved.
2. At `(4,4,2)`, `(4,4,3)` and `(4,4,5)`, evaluate the ordinary raw coefficient
   solver and its JVP for D11, D13 and D33.
3. Verify the primal coefficients against the captured database, undoing its
   standard coefficient normalizations. Stop if the reconstructed baseline differs.
4. Evaluate the production native database VJP and expose the full-factorization
   primal it uses. Compare its contraction with the raw-solver JVP.
5. Check local coefficient FD slopes at `10h`, `3h` and `h`, with `h` from the
   saved RBC/ZBS FD report. Surface coefficients and the field change along
   the captured tangent. This isolates NTX, rather than redoing VMEC differentiation.

Only these three monoenergetic cases run; there is no full coefficient scan,
ambipolar root search, transport integration or optimization. The diagnostic
does not change any production solver, AD rule or interpolation method.
It compiles JAX code and performs small NTX solves on the GPU; it is not a
NumPy-only saved-array readout. It refuses a CPU backend and requires float64.
The local checks cover syntax and saved-array preparation only. No local JAX
or physics run was made; numerical GPU results remain pending.

After pushing/pulling the diagnostic on `en/v0_main`, run on the large machine:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  neopax_case=outputs/axis_fix_ad_fd16_20261116_RPTPxK
  neopax_check=$(mktemp -d "$neopax_case/ntx_direction_RBC_XXXXXX") || exit 1
  set -o pipefail
  env JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=0 \
    NTX_DISABLE_COMPILATION_CACHE=1 VMEX_COMPILATION_CACHE=disabled \
    /usr/bin/time -v "$HOME/jax_env/bin/python" \
    examples/benchmarks/check_captured_ntx_directional_derivative.py \
    --run-dir "$neopax_case" --out-dir "$neopax_check" \
    --parameter RBC:1:0 \
    2>&1 | tee "$neopax_check/run.log"
) || printf '\nFocused NTX check stopped; terminal kept open.\n'
```

The CSV distinguishes forward-primal versus pullback-primal differences,
raw-solver-JVP versus production-VJP differences, and FD versus VJP differences.
None should be substituted for a full transport-gradient validation. ZBS's
independent transport-baseline drift remains a separate question.

Offline artifacts: [all 26 stored chain contractions](axis_fix_ad_fd16_2026-11-16_offline_chain.csv),
[all 25 attempt comparisons](axis_fix_ad_fd16_2026-11-16_offline_schedule.csv),
[NumPy readout](axis_fix_ad_fd16_2026-11-16_offline_readout.txt).
The largest absolute difference between the six summed support components and
their saved total is `2.842170943e-14`. Two focused NumPy preparation tests pass.
The GPU coefficient derivative check has not been run locally.
