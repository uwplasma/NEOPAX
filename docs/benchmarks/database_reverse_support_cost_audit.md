# Database support cost audit — 2026-09-12

## Preserve the forward interpolation

The full-transport benchmark uses
`Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml`.
Its neoclassical model is `ntx_scan_runtime`; it does not specify a database
`interpolation_mode`. `Er_permittivity_mode = "ntss_like_midpoint"` controls
the electric-field equation, not the database interpolation.

The runtime scan conversion constructs `Monoenergetic`, which dispatches to
the generic `get_Dij` interpolation. This conversion predates this performance
work (git blame identifies commit `3835c825`, 2026-08-13). Both forward queries
and the current reverse coefficient transpose use that representation. The
user confirmed that retaining the forward/TOML behaviour is the requirement.
No switch to a preprocessed interpolation is part of this performance task.

## What the source audit establishes

- `_neoclassical.py` calls `monoenergetic_interpolation_table_bar` separately
  for D11-log, D13 and D33. Each call constructs the small-, middle- and
  large-radius expressions before selecting the result. There are ten
  radial surface-transpose bodies in each scalar call, not the 16-point
  preprocessed interpolation stencil. Source-level duplication alone does
  not establish how much work survives compiler optimisation.
- The scalar support hook is vmapped over objective rows inside the stage
  scan. Keep that established finite path: the older specialised rank-3
  entrypoint must not be reactivated as an untested performance shortcut.
- Coefficient-table bars and scan-coordinate bars are distinct contributions.
  Both are required. Timing only the coefficient scatter does not measure the
  full table-support boundary, which also includes the coordinate pullback.
- Shared support preparation retains primal flux values and equation-to-flux
  bars. It does not explicitly share every reconstructed equation owner or
  face-state preparation between the table and local-geometry branches.
  Density and temperature closures must remain separate.
- The centre scan-coordinate pullback constructs its local evaluator inside
  a radius scan; the builder includes state/physical-mesh work independent
  of the differentiated database coordinates. The face-coordinate rule
  already hoists analogous work. Compiler hoisting must be checked before
  claiming that moving this source code saves warm execution time.

With both native face closures, one stage has coordinate scans over 51
centres, 52 density faces and 52 temperature faces. Seven stages and four
steps therefore give 84 coordinate scan invocations and 4,340 local coordinate
VJP iterations per segment. These are compiled device loops, not 4,340 Python
calls or necessarily 4,340 host kernel launches. Objectives are vmapped rather
than another sequential factor of ten.

A subsequent isolated candidate could batch four/eight radius-local coordinate
pullbacks while retaining the old ordered addition of their small `(a_b,
Er_list)` results. That would preserve the local derivative and accumulation
order. It must not batch a full database cotangent tree or merge the two face
closures. The small coordinate outputs themselves are approximately 0.31 MiB
for 52 faces, ten objectives and 7 x 11 coordinates; interpolation AD
intermediates, not those outputs, are the memory risk. No such candidate has
been installed in production or timed yet.

## Measurement limitations and next check

The standalone stage fixture uses stored equilibrium/database files and the
same production equation implementations. It does not run VMEC, an NTX scan,
an initial ambipolar root, or a transport rollout. Its constructed stage states
are not the saved converged states of the full benchmark.

Read-only inspection of the stored HDF5 file found coefficient arrays of shape
`(7, 16, 12)`. The runtime scan in the user's full benchmark has shape
`(7, 16, 11)`. The fixture is not a giant dense database, but it is also not
the identical database or initial state. Reports must include the loaded
database kind, leaf shapes and sizes rather than calling it an identical
full-benchmark reproduction.

Two previous attempts to compile support on the local roughly 8-GiB WSL host
did not yield a support timing. One was OOM-killed after other probe kernels
had been compiled; the fresh small support-only attempt was safely stopped
near the host limit. Neither measures the full benchmark's host RSS or proves
which support component dominates its 22.264-s warm four-step segment.

The revised `tests/benchmark_database_reverse_stage_cost.py` selects one
component per fresh process and exports lowered IR before compilation. It
reports lowering, compilation, synchronized execution and process/device
memory separately. The next support measurements should distinguish:

1. The complete existing table-plus-local-geometry stage support boundary.
2. Its table/coordinate contribution.
3. Direct flux geometry and equation geometry separately, if needed.

Independent partials prepare their own primal inputs; their timings are not
additive measurements of the shared-preparation combined kernel. Saved stage
component outputs can be compared across fresh processes without rerunning
VMEC or compiling both variants together.

The two production candidates already available remain explicit opt-ins:
`--reverse-database-stage-jacobian-mode shared` and
`--reverse-database-initial-support-mode reduced_zero`. Neither changes the
default, interpolation, root lane, accepted schedule, or compilation caches.
See [mode contracts and the full 16/4 command](database_reverse_performance_modes.md).
The roughly 3x full-reverse/forward target and full-run peak RSS have not yet
been demonstrated for these candidates.

## Bounded coefficient-batching experiment (not installed in production)

`tests/benchmark_database_table_scatter_cost.py` compares the existing three
coefficient calls with a coefficient-axis `vmap` of the **unchanged** scalar
`monoenergetic_interpolation_table_bar` function. Its synthetic database is
runtime-shaped `(7, 16, 11)`; it isolates one radius, four energy queries and
ten objective rows, not the full support graph or transport benchmark.

On the local RTX 3060 Laptop / JAX 0.5.0 probe:

| Measurement | Three calls | Coefficient vmap |
| --- | ---: | ---: |
| Optimized HLO | 6.828 MB | 2.420 MB |
| Compilation | 26.408 s | 8.603 s |
| Warm median, seven samples | 18.163 ms | 14.604 ms |
| Compiled temporary device storage | 9.358 MB | 10.376 MB |

Nine axis/radial-branch/tie/extrapolation cases were finite with maximum
absolute output difference zero, including mixed signed seeds and a zero
objective row. A small CPU check also passed. Warm samples are noisy; this is
not a full-run speedup. Importantly, temporary device storage **increased** in
this experiment. It therefore remains a diagnostic-only option: no claim of
simultaneously improved full-run memory and time, and no default or production
support implementation change. Raw local artifacts are in
`/tmp/database-table-scatter-gpu-10x4` and
`/tmp/database-table-scatter-cpu-small`.

## Larger-machine support measurement

The next requested measurement is the existing combined support boundary,
isolated in one fresh process. This does not require another FD run or the
VMEC/NTX-scan/root setup of the full 16/4 benchmark. It still compiles a large
support graph and should run on the larger machine, not the small local WSL
host. Use a fresh output directory so earlier measurements are preserved:

```bash
# Run inside the NEOPAX repository with the existing JAX environment active.
probe_dir=$(mktemp -d /tmp/neopax-db-support.XXXXXX)
env -u JAX_COMPILATION_CACHE_DIR \
  -u NEOPAX_DATABASE_GEOMETRY_VJP_DIAGNOSTICS \
  -u NEOPAX_DATABASE_STATE_VJP_DIAGNOSTICS \
  -u NEOPAX_DATABASE_GEOMETRY_VJP_DIAGNOSTIC_FACE_INDEX \
  JAX_ENABLE_COMPILATION_CACHE=0 \
  PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
  /usr/bin/time -v \
  python tests/benchmark_database_reverse_stage_cost.py \
  --component support_table_and_geometry \
  --n-radial 51 --objectives 10 --device gpu \
  --warmups 3 --repeats 7 --dump-dir "$probe_dir" \
  2>&1 | tee "$probe_dir/run.log"
```

Keep `run.log`, `run.json` and `support_table_and_geometry.json`. The last
contains lowering, compilation and synchronized warm samples; the time footer
contains whole-process peak RSS. The run metadata explicitly reports the
stored fixture's actual loaded table dimensions and input fingerprints.
StableHLO is saved before compilation, so an interrupted compile no longer
discards all graph evidence. These numbers refer to one seven-stage support
call, not a four-step segment or the full reverse run.

If the combined support cost is significant, use fresh processes with
`--component support_table`, `support_flux_geometry`, and
`support_equation_geometry` to isolate it further. Do not run several GPU
timing probes concurrently. Adding `--lower-only` with a dump directory
exports the selected kernel's graph without compiling it; fixture preparation
still runs and is reported separately.
