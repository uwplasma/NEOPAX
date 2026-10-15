# Completed transition-objective database reverse AD run: 16 steps / 4-step segments

Source: user attachment `4af68492-2044-43a7-9c95-0a152a88c3ea/Pasted text.txt`.

The run completed on `neopax-22`, branch `en/v0_new`, with the GPU backend
(`cuda:0`) and `jax_env` active. The remote Git commit and JAX/NTX/VMEX versions
are not printed in the attachment. `git pull` reporting up to date does not
identify the commit.

## Saved evidence

- [Complete pasted command and output, preserved byte for byte](database_transition_objectives_ad_16x4_2026-10-14.txt).
- [Full residual/Jacobian CSV](database_transition_objectives_ad_16x4_2026-10-14.csv):
  18 rows, each containing the residual label, residual value, and eight AD
  derivatives. Original printed numeric precision is retained.
- Raw attachment SHA-256:
  `1ae8d7149a0b9c2bfb30d10c423e913664d24a0ed1e8befbfea56b8462e2f970`.

The shell-command header has a mangled final flag ending in
`reuse_static_probeupar_only`. The `/usr/bin/time -v` footer records the
successfully executed command and is authoritative for the actual flags.

## Case and outcome

The runner is `examples/benchmarks/benchmark_transport_reverse_ad_only.py` with
`Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml`,
`--full-transport-shared-payload-smoke`, `--objective all`, sixteen accepted
steps, four-step reverse segments, six profile parameters, and two geometry
parameters (`RBC:1:0`, `ZBS:1:0`).

| Quantity | Observed result |
| --- | ---: |
| Exit status | 0 |
| Transport objectives | 11 |
| Geometry objectives | 7 |
| Residual/Jacobian shape | 18 x 8 |
| Finite residuals | 18 / 18 |
| Finite Jacobian entries | 144 / 144 |
| Internal benchmark elapsed time | 2735.602 s |
| Whole-process wall time | 59:42.65 |
| Maximum host RSS | 12,326,472 KiB |
| Swaps | 0 |

The recorded-scan fold reported `objective_rows=11`, `groups=1`, and
`contract=one_batched_scan_transpose`. Its raw parameter bars were all finite.
The host RSS is a process peak, not a per-iteration trace or a GPU-memory
measurement; this single-evaluation run does not establish whether the combined
optimizer's memory-growth issue is resolved.

## New root rows

| Objective | Value |
| --- | ---: |
| `Er_transition_rho` | 6.1293923577522791e-01 |
| `Er_transition_location_moment` | 1.1293923577522791e-01 |
| `Er_transition_strength` | 2.2261594486928726e+00 |

All eight derivatives in each new row are finite and nonzero in the pasted
output. Radius and location-moment derivatives agree to floating-point
precision, consistent with a location moment equal to radius minus a fixed
target of 0.5. This establishes that the new rows participate in the reverse
calculation; it does not establish agreement with finite differences.

CSV parameter columns, in order:

```text
n0
T0
density_shape_power
temperature_shape_power
density_shape_alpha
temperature_shape_alpha
vmec:RBC:1:0
vmec:ZBS:1:0
```

## Selected modes and timings

The progress banner records `block`, `explicit_database`, separate RHS and
rebuild support pullbacks, `grouped_vjp` ordinary objective cotangents,
`joint_local_vjp_upar_only` bootstrap cotangents, and
`reduced_cotangent_call_boundary`. Transport interpolation uses `legacy_sparse`;
root and bootstrap interpolation use `established`. Initial support is `split`
and initial state is `generic`. Stage Jacobians are `independent`, objective
support is `scalar`, segment support is `inline`, and the static schedule is
reused. Segment diagnostics are disabled.

| Phase | Elapsed seconds |
| --- | ---: |
| Runtime database build | 613.455 |
| Final-objective cotangents | 272.005 |
| First reverse segment (4/4) | 476.603 |
| Warm reverse segment 3/4 | 0.812 |
| Warm reverse segment 2/4 | 0.780 |
| Warm reverse segment 1/4 | 0.722 |
| Segmented reverse sweep | 478.918 |
| Initial direct-RHS support pullback | 169.555 |
| Initial state pullback | 140.506 |
| Initial-Er root boundary pullback | 262.581 |
| Final recorded-scan fold | 714.093 |

These are printed phase timings, not an exhaustive decomposition of wall time.
The first segment includes cold work; the three later segments total 2.314 s.

## Remaining FD validation

No matching FD result is present in this attachment. Use the
[paired FD instructions](database_transition_objectives_ad_fd.md) and compare
the eleven transport rows by name and parameter column against the archived
CSV. The seven geometry-only AD rows require their separate geometry checks.
The complete AD/FD comparison and the original optimization issues remain open.
