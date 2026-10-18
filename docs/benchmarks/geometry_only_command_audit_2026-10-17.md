# Geometry-only command audit

The user reports that invoking the geometry-only script without arguments now
produces the intended optimization behavior, unlike the earlier diagnostic
command. The subsequently supplied log excerpt is archived and assessed
below; it contains optimized-geometry postprocessing rather than the
optimization evaluations needed to compare trajectories.

The exact earlier assistant command was recovered from the same task's local
session record, timestamp `2026-10-17T02:47:36.317Z`:

```bash
cd ~/NEOPAX
git switch en/v0_new
git pull --ff-only origin en/v0_new

mkdir -p outputs
set -o pipefail

XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
NEOPAX_REVERSE_STEP_DIAGNOSTICS=1 \
NEOPAX_REVERSE_STEP_SNAPSHOT_DIR=outputs/geometry_only_net_power_root_bootstrap_replay_fix/bad_steps \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
python examples/optimization/optimize_geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport.py \
  --no-max-er \
  --no-root-objectives \
  --transition-location-objective \
  --bootstrap-penalty \
  --net-power \
  --max-nfev 30 \
  --out-dir outputs/geometry_only_net_power_root_bootstrap_replay_fix \
  2>&1 | tee outputs/geometry_only_net_power_root_bootstrap_replay_fix.log
```

The current user invocation is:

```bash
python examples/optimization/optimize_geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport.py
```

## Argument comparison

The [standard-library-only audit](audit_geometry_optimization_command_2026-10-17.py)
extracts the actual script parser and its constants without importing NEOPAX
or any physics dependency. It asserts that the only argparse difference is
the output directory. The audit passes:

| Setting | Earlier command | Bare invocation |
| --- | --- | --- |
| Maximum-Er objective | off | off |
| Old left/right root objectives | off | off |
| New transition location/strength | on | on |
| Bootstrap penalty | on | on |
| Net-power objective | on | on |
| Function-evaluation limit | 30 | 30 |
| Database grid | 25,25,33 | 25,25,33 |
| Seed input | `examples/inputs/input.QI_nfp2_initial` | same |

`git diff c6a1cca --` for the geometry script, transport TOML, and tracked
seed input is empty in the inspected checkout (`a8410b5` plus documentation
work). The earlier remote log identifies `c6a1cca`. This establishes parity
of these tracked local files across those commits, not the absence of any
remote uncommitted edits or changes in external VMEX/NTX dependencies.
The script initializes from `--seed-input`, not from an existing output
directory, so changing `--out-dir` does not request a different starting point.

## Environment differences

The earlier command adds settings absent from the bare invocation's text:

- `NEOPAX_REVERSE_STEP_DIAGNOSTICS=1` and a snapshot destination. Diagnostics
  select the matrix-summary variant of the block adjoint, retaining primal
  RHS values through `jacfwd(..., has_aux=True)` and adding finite checks and
  callbacks. The intended derivative is unchanged, but the traced computation
  differs. Instrumented nonfinite segment adjoints raise before further
  propagation. Diagnostic parity at the remote failing point is unverified.
- `XLA_FLAGS` appends an empty GPU command-buffer selection, disabling CUDA
  command buffers. This is intended to change execution/memory behavior, not
  the numerical optimization case.
- `PYTHONPATH` is replaced by `$HOME/VMEX:$HOME/NTX/src`. Depending on the
  active environment, this may choose different dependency source trees from
  a bare invocation. The actual remote imported modules have not been
  inspected; a dependency difference is not established.

The bare command inherits any already-exported environment values. Its text
alone does not establish whether these variables are absent in that process.
`tee`, the snapshot/output paths, and `pipefail` do not change objective
definitions. The earlier command's `git pull` also selects the code available
at that time; it is not an algorithm option.

No CLI objective, target, grid, seed-default, or evaluation-budget mismatch
was found. Which runtime difference caused the observed behavior remains
unidentified. In particular, the earlier bad optimization must not be treated
as a controlled before/after replay regression on this evidence alone. The
bare invocation is the intended baseline; no physics optimization was run
locally during this audit.

## Supplied successful-run excerpt and combined continuation

Attachment `3d560fd0-0b1e-4952-a5b8-20cf184e88ce/Pasted text.txt` is archived
byte-for-byte as [postprocessing output](geometry_only_bare_run_postprocessing_2026-10-17.txt):
193012 bytes, SHA256
`b308acc72f882fa08cb3d5dff48c37ba5214c50ce00aaa01c7be4ee92d6ae7aa`.
Its 1023 lines begin within a Radau Newton solve and finish with:

```text
[NEOPAX] solver step summary: accepted_saved=10/10 n_steps=217 failed_any=False done=True failed=False fail_code=0 final_time=2.000000e+00
```

The optimized transport reports, including bootstrap CSV/PNG, are written to
the default geometry-result directory, and the shell prompt returns normally.
This confirms completion of that optimized-geometry forward report. There
are no `[initial]`, `[NEOPAX least_squares]`, final objective-cost lines or
setup/command lines in the excerpt. It therefore cannot identify the first
optimization divergence from the earlier run or isolate a runtime override
as its cause. No numerical optimization optimum is inferred from this tail.
The direction-audit finite differences printed inside Newton are not the
objective AD-FD comparison table.

Continue from that saved result with geometry and profiles free, without
rerunning stage 1:

```bash
cd ~/NEOPAX
python examples/optimization/optimize_geometry_profiles_from_geometry_result.py \
  --geometry-result-dir outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --out-dir outputs/combined_from_successful_geometry \
  --max-nfev 30 \
  --no-diagnose-optimization \
  --no-diagnose-memory
```

This uses the current working Python environment without replacing
`PYTHONPATH` or enabling reverse-step diagnostics. The stage-2 runner itself
disables CUDA command buffers by default because the earlier combined run
exhausted memory while instantiating CUDA graphs; that behavior is retained
and must not be described as an unchanged CUDA execution setting. It restores
the saved geometry coordinates, keeps nominal initial profiles, frees 24
geometry and six profile coordinates for the supplied mode-2 result, and
inherits the objective switches and database grid. Initial/optimized
analytical-profile comparison plots remain enabled. Command options were
verified with `--help` only; no transport or optimization was run locally.

## Optimizer versus plotted forward solver: confirmed mismatch

The subsequent source audit found a separate, concrete inconsistency in the
two numerical runs whose outputs had been compared:

- `geometry_full_transport_least_squares_problem` forces
  `radau_jacobian_reuse_mode="legacy"`, which normalizes to `dt_close`.
- `write_transport_report` reruns `run_config` using the original example
  configuration. That TOML omits `radau_jacobian_reuse_mode`, so the forward
  solver factory selected its `retry_only` default.

Thus the optimization monitor and plotted final profiles were not generated
with identical forward solver settings. This mismatch exists independently of
the shell diagnostic environment. It establishes a setup difference; it does
not establish that this reuse policy caused an electron/ion branch change or
that the two shell commands found different optimized geometries.

The example's `transport_config` now explicitly writes `legacy` into the
common solver configuration. Optimization already overrode this value, so
its selected policy and AD equations are unchanged. Fresh initial/optimized
reports now use the same policy. This also reaches the single-stage combined
and warm-start examples, which reuse this configuration helper. No global
Radau default or benchmark configuration was changed. Existing output files
retain the settings with which they were originally produced.

The new configuration regression test captures the actual report config,
constructs both solver objects without solving, and compares their normalized
reuse modes. Before the fix it failed with `retry_only != dt_close`; after
the fix both are `dt_close`. All 17 checks in
`tests/test_simple_combined_least_squares.py` and
`tests/test_geometry_warm_start_combined_least_squares.py` pass. These are
configuration, source, and mocked optimizer checks; no expensive local physics
evaluation was performed.

Enabling profile DoFs at their nominal coordinates does not select a new
initial-root policy: both builders request `initial_er_root_ad=
"jax_selected_root"`, use the same configured canonical profile values, and
apply the same geometry/root/bootstrap/power terms. Source and existing
parameterization tests confirm this wiring, not numerical branch parity at a
particular optimized geometry.

## Full combined optimization from the usual initial condition

The user clarified that the requested run must start from the original input,
with geometry and profiles free immediately. It must not load a saved
geometry result or execute a geometry-only first stage. The single-stage
simple-least-squares runner does that:

```bash
cd ~/NEOPAX
python examples/optimization/optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_simple_least_squares.py \
  --seed-input examples/inputs/input.QI_nfp2_initial \
  --out-dir outputs/combined_from_usual_initial \
  --no-diagnose-optimization
```

The script starts at zero ESS geometry deltas and zero nominal profile deltas
against the usual input/configuration. At mode 2, 24 geometry coordinates and
all six profile coordinates are active from the first evaluation. Existing
defaults use 30 evaluations and a profile trust multiplier of 3.7. The command
does not replace `PYTHONPATH`, set reverse-step diagnostic environment flags,
or add a saved-result argument. It inherits the working shell environment.
Unlike the saved-geometry stage-2 runner, this script does not automatically
change CUDA command-buffer settings. The local report-mode fix must be synced
to the remote checkout to affect that run's later plots; the combined runner
and its initial-condition command already exist in the prior checkout.

To distinguish different optimizer results from different plotted forward
trajectories, the two existing summaries can be compared without another
optimization or transport solve:

```bash
python - <<'PY'
import json
from pathlib import Path

base = Path("outputs")
old = json.loads((base / "geometry_only_net_power_root_bootstrap_replay_fix" / "optimization_summary.json").read_text())
new = json.loads((base / "geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization" / "optimization_summary.json").read_text())
assert old["parameter_labels"] == new["parameter_labels"]
print("cost old/new:", old["cost"], new["cost"])
print("identical optimized coordinates:", old["x"] == new["x"])
print("max coordinate difference:", max(abs(a-b) for a, b in zip(old["x"], new["x"], strict=True)))
PY
```

Those remote summaries have not been supplied or read here. Do not claim
their coordinates agree, or that the report-mode fix resolves the original
command comparison, without that evidence.
