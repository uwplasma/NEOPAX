# Saved transport baseline repeatability

This isolates the unperturbed ZBS baseline drift in
`outputs/axis_fix_ad_fd16_20261116_RPTPxK`. It does not test the separate RBC
bootstrap derivative discrepancy.

The archived AD, RBC and ZBS runs have identical packed initial states,
transport geometry and coefficient tables. RBC reaches the same endpoint as
AD. The independently generated ZBS baseline differs in its endpoint by
`7.126189274e-7` and its final time by `1.385163667e-10`.
Its first **saved scalar** difference is the error estimate at attempt 6,
`4.065758147e-20`; the next timestep first differs after attempt 11.
These are baseline differences before the ZBS perturbations. The saved data
do not identify the first differing arithmetic operation or prove that ZBS
itself causes the difference.

`check_saved_transport_baseline_repeatability.py` restores the common saved
geometry, database and full initial plasma state. It retains the
`NTXRuntimeScanTransportModel` wrapper with the existing tables populated;
channels and scan surfaces are not reconstructed because no scan is needed.
Solver preparation, fixed-temperature reference profiles and the native
adaptive helper use the ordinary benchmark path. The saved FD report supplies
the accepted-step limit and Jacobian reuse setting. At 16 accepted steps the
native helper retains its original 256 attempt slots and Newton-reuse capture.

Two child processes run **sequentially**. Each evaluates the native helper
once, clears compilation caches, and evaluates one additional jitted callable
twice from the identical initial carry. This is six short forward integrations
in total. No equilibrium solve, NTX coefficient scan, ambipolar-root solve,
objective evaluation, finite difference or AD sweep is requested.

The additional jit is a diagnostic control and can change compilation and
rounding. It is explicitly compared with the native result and is never used
to replace production execution. Its two identical calls share one callable
and argument structure. Cache clearing occurs before those two calls, not
between them.

| Comparison | What a difference would establish |
| --- | --- |
| Initial carry, numeric execution settings, frozen temperature profile across processes | Solver setup is already different before integration. |
| Same jitted callable twice | Repeated execution differs despite identical explicit inputs. |
| Native helper across processes | The restored unperturbed native baseline is not repeatable across processes. |
| Native versus additional jit | Compilation context affects this calculation. |
| Native versus archived RBC/ZBS schedules and endpoints | Whether the reduced reproduction reaches either archived trajectory. |

The setup capture does not serialize callable closures. Matching numeric
setup arrays alone therefore does not prove every hidden constant is equal.
If every repeat matches, the full benchmark's preparation/compilation context
still needs investigation; this result would not prove the original drift is
fixed. A failing or nonfinite forward solve stops the diagnostic.

Run on the GPU machine after transferring these three diagnostic files:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  neopax_run=outputs/axis_fix_ad_fd16_20261116_RPTPxK
  neopax_out=$(mktemp -d outputs/zbs_baseline_repeat_XXXXXX) || exit 1
  set -o pipefail
  env JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$HOME/jax_env/bin/python" \
    examples/benchmarks/check_saved_transport_baseline_repeatability.py \
    --run-dir "$neopax_run" --out-dir "$neopax_out" \
    2>&1 | tee "$neopax_out/comparison.log"
) || printf '\nBaseline repeatability check stopped; terminal kept open.\n'
```

`comparison.json` records exact differences and the first differing attempt
for each saved schedule field. Each process folder has `run.log`,
`metadata.json`, `setup.npz`, and the three endpoint/schedule files.
Host-only comparison tests are in
`tests/test_saved_transport_baseline_repeatability.py`.
