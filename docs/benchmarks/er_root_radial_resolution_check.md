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
  --out-dir outputs/new_root_candidates/radial_resolution_20261026 \
  2>&1 | tee outputs/new_root_candidates/radial_resolution_20261026.log
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

The experiment preserves `DEr`, final time, tolerances, attempted-step budget,
initial profile configuration, VMEC/Boozer settings and the NTX database
radii and angular/velocity grids. This check varies the transport radial
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
