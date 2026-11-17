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

## Remote result, 2026-11-17

The user ran commit `20fceaa` on `neopax-31`; results are in
`outputs/zbs_baseline_repeat_PgJe3r`.

| Comparison | Observed differences |
| --- | --- |
| One jitted callable run twice, in each process | None |
| Native versus additional jit, in each process | None |
| Independent process setups | None in captured fields |
| Independent native or jitted trajectories | None |
| Native versus archived RBC | Eight error estimates differ, starting at attempt 6; maximum `2.168404345e-19`. All timesteps and endpoint arrays match exactly. |
| Native versus archived ZBS | The existing timestep and endpoint differences remain: maximum final-state difference `7.126189274e-7`, time difference `1.385163667e-10`. |

Thus the restored calculation is repeatable in this test and reaches the
archived RBC trajectory. This does not reproduce the original ZBS drift or
prove its cause. In particular, the result does not demonstrate general GPU
nondeterminism or a failure in accepted-step replay.

The original FD benchmark computes the selected VMEC tangent and builds its
runtime/roots before preparing the same adaptive helper. The original
snapshots did not include its full initial carry, RHS/cache or the complete
model constants. These are the remaining preparation boundaries to compare.

`benchmark_transport_realtime_geometry_forward_fd.py --baseline-only-dir DIR`
now stops immediately after that benchmark's ordinary `_schedule_rollout`
call. It runs the original preparation for the selected parameter and saves
the setup, physical model data and native schedule/endpoint. Capture happens
**after** the forward solve and adds no callbacks, scan outputs, jit wrappers,
objective evaluations or FD endpoints. No production solver is changed.
The setup capture includes JAX weak-type markers as well as numerical values.
Older captures lack those markers; compare their common fields first.

One original ZBS preparation/baseline, using the saved configuration:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  neopax_case=outputs/axis_fix_ad_fd16_20261116_RPTPxK
  neopax_out=$(mktemp -d outputs/zbs_original_baseline_XXXXXX) || exit 1
  set -o pipefail
  env -u PYTHONPATH -u LD_LIBRARY_PATH -u JAX_PLATFORM_NAME \
    -u JAX_COMPILATION_CACHE_DIR -u XLA_FLAGS -u XLA_PYTHON_CLIENT_ALLOCATOR \
    -u NEOPAX_GRADIENT_AUDIT_DIR \
    JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true JAX_ENABLE_COMPILATION_CACHE=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false NTX_DISABLE_COMPILATION_CACHE=1 \
    VMEX_COMPILATION_CACHE=disabled NEOPAX_REVERSE_STEP_DIAGNOSTICS=1 \
    "$HOME/jax_env/bin/python" \
    examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
    --config "$neopax_case/config.toml" \
    --parameter ZBS:1:0 --accepted-step-limit 16 \
    --radau-jacobian-reuse-mode legacy \
    --geometry-fd-lane frozen_linearized --replay-mode accepted \
    --fd-accepted-replay-contract reverse_frozen_metadata \
    --initial-Er-root-ad jax_selected_root \
    --initial-Er-root-fd-root-lane frozen_linearized \
    --baseline-only-dir "$neopax_out" \
    2>&1 | tee "$neopax_out/run.log" || exit 1

  "$HOME/jax_env/bin/python" - "$neopax_out" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, 'examples/benchmarks')
from check_saved_transport_baseline_repeatability import differences, read_one
fresh = Path(sys.argv[1])
control = Path('outputs/zbs_baseline_repeat_PgJe3r/process_1')
for name in ('setup', 'native'):
    a, b = read_one(control, name+'.npz'), read_one(fresh, name+'.npz')
    common = a.keys() & b.keys()
    rows = differences({k: a[k] for k in common}, {k: b[k] for k in common})
    print(name, 'common fields:', len(common), 'differing fields:', len(rows))
    for row in rows:
        print(row)
PY
) || printf '\nOriginal baseline check stopped; terminal kept open.\n'
```

This run rebuilds VMEC, its selected tangent, NTX tables and initial roots;
unlike the saved-input control, it preserves those preparation steps.
Use the environment shown in the original archive command when reproducing
that preparation, including its disabled compilation caches. This preserves
the comparison conditions and is not evidence that caching caused the drift.
