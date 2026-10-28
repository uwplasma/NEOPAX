# Original position/strength and bootstrap controls on the saved geometry

This repeats the same 51-cell full-time experiment as the new root-loss
candidate comparison, using the original production objectives instead.
Four independent runs start at the same saved optimized boundary and nominal
profiles:

- Original transition position + strength deficit, profiles only.
- Original transition position + strength deficit, geometry only.
- Bootstrap-current penalty, profiles only.
- Bootstrap-current penalty, geometry only.

No net-power, QI, mirror, aspect or iota penalty is included in these scalar
test objectives. Each selected DOF block is varied independently; the other
block remains fixed. The forward/reverse solver, recorded-forward Jacobian
reuse, full final time, parameter scales and descent/backtracking algorithm
are the same as the candidate comparison. Radial refinement is not part of
this test.

The original root objective is

`0.5 * 1000 * Er_transition_location_moment**2
 + 0.5 * 100 * smooth_positive_part(0.9 - Er_transition_strength)**2`.

The bootstrap objective is

`0.5 * 4 * max(bootstrap_current_softmax_abs_scaled - 0.1, 0)**2`.

The code reads these constants and the original smooth-positive-part
implementation from the production geometry example. It calls the production
transition selector with the same target-centered prior, temperature,
softness and selection sharpness. Costs are minimized directly, without
squaring a scalar cost again.

The AD table retains the native raw observable rows
`Er_transition_location_moment`, `Er_transition_strength` and
`bootstrap_current_softmax_abs_scaled`. The weighted cost gradients are formed
from those rows by the chain rule. Retaining the native bootstrap name is
essential: it keeps the existing compact corrected-Upar bootstrap transpose
rather than sending a custom cost through an ordinary terminal VJP.

Use a fresh output directory on the remote GPU machine:

```bash
cd /home/exouser/NEOPAX || exit 1
git pull --ff-only origin en/v0_new || exit 1
mkdir -p outputs/new_root_candidates
set -o pipefail

env -u JAX_COMPILATION_CACHE_DIR \
JAX_ENABLE_COMPILATION_CACHE=0 \
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src${PYTHONPATH:+:$PYTHONPATH}" \
/home/exouser/jax_env/bin/python examples/optimization/optimize_er_root_candidates.py \
  --geometry-result-dir outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --out-dir outputs/new_root_candidates/original_root_bootstrap_20261026 \
  --candidates original,bootstrap \
  --groups profiles,geometry \
  --max-iterations 10 \
  --initial-step 0.001 \
  --max-step 0.05 \
  --growth-factor 2 \
  2>&1 | tee outputs/new_root_candidates/original_root_bootstrap_20261026.log
```

The driver automatically prepares one baseline report containing the original
observable rows and shares it between all four cases. It must reproduce the
saved optimized transport field before its reverse sweep proceeds. The old
five-row candidate-only report cannot supply bootstrap or original-selector
derivatives and is not silently reused. Original-objective forward tasks have
their own cache identity and record the objective constants; Er-only cached
endpoints cannot establish their missing bootstrap observables.

To inspect only the original baseline gradients and finite downhill moves
before multi-iteration optimization, use the existing benchmark with
`--objective-suite original`:

```bash
/home/exouser/jax_env/bin/python examples/benchmarks/benchmark_er_root_candidates_optimization_start.py \
  --geometry-result-dir outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --objective-suite original \
  --step-sizes 0.001 \
  --out-dir outputs/new_root_candidates/original_root_bootstrap_descent_20261026
```

Use the same environment settings as the full command. That shorter benchmark
also prepares new original-objective AD and then checks both DOF blocks using
independent full-time forward moves; it is not a local CPU test.

Logs and `comparison.csv` report the original position moment, strength,
strength deficit, bootstrap observable/excess and separate weighted costs,
alongside the actual interpolated root, field signs and parameter vectors.
The bootstrap run's success condition is its bootstrap limit; a root already
at its target must not terminate a bootstrap run with a nonzero penalty.
An original-root run also checks the actual root/sign pattern and strength.
Convergence is not inferred from a scalar cost drop alone.

Local verification passed 86 tests and 43 subtests across the objective and
workflow suites, followed by 26 targeted checks after adding saved-gradient
analysis and accepted-point row-routing coverage. These use small analytic
arrays, finite differences of local objective formulas and mocked workers.
They check the production selector/native bootstrap values, weighted cost
chain rule, independent starts and frozen DOF blocks, cache separation and
bootstrap-specific stopping. They do not validate the full physical
transport Jacobian or establish a physical optimization result; those runs
remain remote. No local VMEC, NTX or transport solve was performed.

The supplied 2026-10-26 run exhausted eight backtracks in each original-root
DOF block without accepting a move. Even the smallest step (0.0000078125)
changed the final root from approximately 0.51742 to 0.49781. The original
cost correctly increased and rejected those points. Bootstrap profiles have
three accepted steps, reducing its cost from 0.383795 to 0.258947; a fourth
forward endpoint is awaiting AD in the supplied log. This does not establish
a physical cause for the root jump or a complete physical Jacobian check.

Inspect the files already saved by this run, without another transport solve:

```bash
cd /home/exouser/NEOPAX || exit 1
neopax_case=outputs/new_root_candidates/original_root_bootstrap_20261026
set -o pipefail
for neopax_run in original_profiles original_geometry bootstrap_profiles; do
  /home/exouser/jax_env/bin/python examples/benchmarks/analyze_er_root_saved_profiles.py \
    --ad-report "$neopax_case/baseline/ad_summary.json" \
    --run-dir "$neopax_case/$neopax_run" \
    --descent-dir "$neopax_case/original_profiles" \
    --out-dir "$neopax_case/${neopax_run}_saved_analysis" \
    2>&1 | tee "$neopax_case/${neopax_run}_saved_analysis.log" || break
done
```

The analyzer imports NumPy only for its numerical work. It reads initial
Er/density/pressure arrays, terminal fields and stored solver counts, and
writes `initial_state_summary.csv`, crossing tables and a JSON summary.
`saved-initial-state` console records expose initial root/bracket changes
alongside the terminal changes. The shared baseline is loaded from the
input-matched forward cache in `original_profiles`; no solver is launched.
This distinguishes changes already present in the initializer from changes
that appear during transport, without assuming either is responsible.
