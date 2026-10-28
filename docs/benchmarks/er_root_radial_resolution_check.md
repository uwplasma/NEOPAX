# Forward-only radial-resolution check for the new root losses

The 51-cell saved-geometry experiment moved its final electric-field crossing
by approximately one radial cell while reducing the local root loss. This
check tests spatial resolution before attributing that behavior solely to
the loss. It holds the optimized VMEC boundary and two physical profile
vectors fixed: the baseline and a selected accepted point, iteration 1 by
default. It varies only `geometry.n_radial`, by default 51, 101 and 201.

The script reads the saved baseline AD report for its input configuration
and the profiles-only checkpoint for the accepted vector. It performs no
optimization and no reverse AD. New forward solves use the same existing
forward worker as the candidate experiment, including analytical profile
construction under the fixed nominal config and the JAX selected-root
initializer. Initial profiles and roots are rebuilt on each radial grid.
The old field is never interpolated to initialize the finer simulation.
Each new solve runs in a separate process and releases its GPU allocations
on exit.

Run on the remote GPU machine after the current comparison finishes on
that GPU:

```bash
cd /home/exouser/NEOPAX || exit 1
git pull --ff-only origin en/v0_new || exit 1
mkdir -p outputs/new_root_candidates
set -o pipefail
neopax_case=outputs/new_root_candidates/optimized_geometry_rerun_20261026

env -u JAX_COMPILATION_CACHE_DIR \
JAX_ENABLE_COMPILATION_CACHE=0 \
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src${PYTHONPATH:+:$PYTHONPATH}" \
/home/exouser/jax_env/bin/python \
  examples/benchmarks/check_er_root_radial_resolution.py \
  --ad-report "$neopax_case/baseline/ad_summary.json" \
  --run-dir "$neopax_case/local_profiles" \
  --iteration 1 \
  --n-radial 51,101,201 \
  --transport-max-steps 20000 \
  --out-dir outputs/new_root_candidates/radial_resolution_20261026_steps20000 \
  2>&1 | tee outputs/new_root_candidates/radial_resolution_20261026_steps20000.log
```

Appending `--plan-only` inspects the saved inputs and cache matches without
starting physics. Completed matching endpoints are also reused when the
same command is repeated after interruption. Output directories with a
different saved plan are rejected.

The script automatically scans the supplied profiles-only run for reusable
forward endpoints. Reuse requires a valid original task digest, the same
source digest, complete numerical config, exact physical parameter vector,
root initializer, full final time and radial grid. Matching 51-cell baseline
and iteration-1 endpoints normally save those two solves, leaving four new
solves at 101/201 cells. Changed physical sources or inputs invalidate the
cache; the script prints which solves are required. It rejects a changed
optimized input relative to the baseline's recorded hash.
If the original radial resolution is included, its baseline must also
reproduce the saved AD terminal field before refinement is interpreted.
This reference-field check is intentionally limited to the original grid;
different fine-grid fields are the quantity the experiment is measuring.

The experiment preserves `DEr`, final time, tolerances, initial profile
configuration, VMEC/Boozer settings and the NTX database radii and
angular/velocity grids. `--transport-max-steps` optionally increases the
attempt budget on grids different from the original grid. Original-grid
tasks retain their saved budget and cache keys. The extra budget allows more
work without changing the physical equation or loosening tolerances. Without
this option the saved attempt budget is preserved on every grid. This check varies the transport radial
grid; it is not a convergence study of the seven NTX database radii or the
angular/velocity grid. A solve that fails to reach final time is not a valid
endpoint and stops the check with its worker log retained.

Outputs:

- `endpoints.csv`: initial/final crossings, errors, target Er, native costs,
  accepted steps, attempts and reuse provenance for each point/grid.
- `profile_moves.csv`: the root and target-field changes between the two
  fixed profile vectors at each resolution, including shifts in cell units.
- `convergence.csv`: differences between adjacent resolutions at each fixed
  physical parameter point.
- `radial_profiles.svg`: native initial/final radial fields for both points.
- `summary.json`, `plan.json`, per-point task/result JSON and new-worker logs.

The original grid's two neighbors beside the target define fixed physical
probe radii used at every resolution. The local loss's native neighboring
nodes themselves move under refinement, so raw native costs across grids
are not a pure measure of solver convergence. Compare actual crossings,
target Er, fixed-probe Er and the profile-move response first.

`interpolated_width_pm2` measures the distance between interpolated +2 and
-2 kV/m downcrossings within 0.05 of the selected root. It is unavailable
when those levels are not bracketed locally. `centers_inside_pm2` records
how many grid centers resolve that interval. A sub-cell interpolated width
with zero interior centers is not evidence of a resolved physical front.

Refinement that changes the crossing and reduces the cell-sized parameter
response supports radial under-resolution. Stable fine-grid crossings with
continued cost/position disagreement support a persistent loss limitation.
If 101/201-cell results do not converge, more refinement may be needed;
this script does not declare convergence from two differences alone.

Local validation: 28 tests and 17 subtests passed across the new checker,
saved-profile analyzer and existing forward-descent helper. The checks use
small arrays, stored JSON and mocked workers. They cover fixed physical
inputs under refinement, cache identity and invalidation, partial endpoints,
baseline reproduction, resume without new solves, output tables/plots and
host imports without physics packages. No local physical forward or AD
solve was run; physical radial convergence remains to be measured remotely.

The first remote attempt reused both 51-cell endpoints, then failed in the
101-cell baseline worker. The supplied parent output does not include that
worker's exception, so no cause or fine-grid physical result is established.
Read the existing n101_baseline.log before rerunning any physics. The launcher
now preserves upstream worker tracebacks and prints the failed worker's exit
code and bounded last 100 log lines into the parent console/tee log. Completed
endpoints remain reusable. This changes error reporting only.

The reused data show initial crossings 0.5742164146563765 and
0.5741455425450299, both in cell 28, versus final crossings 0.5174235897859059
and 0.49780399703165884. The large final crossing-cell difference is not
already present in the initial nearest-crossing cell. These values alone
do not exclude changes elsewhere in the initial fields or identify the
evolution's sensitivity mechanism.

The subsequent worker traceback identifies an incomplete adaptive solve:
`RuntimeError: FD adaptive baseline did not complete:
final_time=3.1984666027806051e-02, fail_code=0.` The saved maximum is 1000
attempts. The existing schedule loop has this fixed attempt capacity and
can return `completed=False` without setting a failure code on exhaustion.
Thus `fail_code=0` is not a successful full-time result. The new command
above permits 20000 attempts on the refined grids; it does not guarantee
that an arbitrarily stiff solve will finish within that allowance.

The worker now saves incomplete-rollout counters in
`n101_baseline_failure.json` (or the corresponding point name), retaining
final/target time, attempts, accepted steps, budget, completion/failure
status and the next timestep when available. These diagnostics are read
from the raised helper's traceback context after failure; the solver and
its attempted steps are unchanged, and the worker never automatically
retries. The existing source digest/cache protocol is unchanged. A fresh
output directory is required when the planned refined-grid budget changes;
completed original-grid forward endpoints can still be reused from the
profile optimization run.
