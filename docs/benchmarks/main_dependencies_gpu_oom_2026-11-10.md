# Main-based dependencies: GPU autotuning memory investigation

Historical compiler-allocation experiment. The subsequent
[native interpolation implementation](native_database_interpolation_2026-11-10.md)
replaces the generic database query path and was tested with default XLA
settings. The red-zone flag below is not an installation requirement.

This investigation concerns `en/v0_main`, VMEX and NTX
`en/neopax-main-utilities`, and JAX/jaxlib 0.10.0. The old JAX 0.7.2
environment and the cleanup checkout were not modified.

## Failure and preserved benchmark settings

The supplied main-environment log completed the six reverse segments (289
accepted steps) before failing in
`NTXDatabaseTransportModel._pullback_local_particle_flux_database_coordinates`.
XLA reported an autotuning failure for a fusion with 74 operands and 21 small
`f64[10,13]` outputs. All three configurations failed with CUDA out of memory.
An earlier 3.49 GiB allocation failure was also logged; the process continued
beyond that message. Host maximum RSS does not measure GPU memory.

The archived successful full-time command and the supplied command use the
same benchmark arguments. The two additional explicit segment options in the
main run select existing defaults. Before the native interpolation change,
the benchmark TOML and established coordinate-pullback implementation were
unchanged between these checkouts.
The proposed launch change retains the full 20 s simulation, all objectives,
13 transport cotangent rows, 50-step segments, recorded-forward Jacobian
reuse, `grouped_vjp`, and the established root/bootstrap transpose paths.

## Controlled component measurements

`examples/benchmarks/benchmark_initial_er_coordinate_pullback.py` calls the
actual established coordinate pullback on smooth synthetic coefficient tables.
It performs no transport, VMEX or NTX solve. It refuses a CPU default device.
Each configuration runs in a fresh GPU process with x64, persistent compilation
cache disabled, CUDA command buffers disabled, and JAX preallocation disabled.
This last allocator setting is shared by both sides of the comparison, not
proposed as an additional full-benchmark change.

Local hardware: RTX 5060 Laptop GPU, 8 GiB, driver 592.25, WSL. The isolated
test environment uses JAX/jaxlib and both CUDA plugin packages at 0.10.0.
These are component measurements, not a successful full run on the 10 GiB
GRID A100X vGPU.

| Probe: 2 species, 4 radii, 3 cotangent rows | Interpax | Peak JAX allocation (bytes) | First call (s) |
| --- | --- | ---: | ---: |
| Default compiler guards | 0.3.14 | 1,342,208,000 | 47.661 |
| Default compiler guards | 0.3.16 | 1,476,429,824 | 34.949 |
| 64 KiB compiler guards | 0.3.16 | 67,531,008 | 36.683 |

Reducing the guard padding lowered this probe's peak by 95.4%, from
1,408.0 MiB to 64.4 MiB. Both returned derivative arrays (`a_b` and `Er_list`)
were bit-for-bit identical with default versus smaller guards. Live arrays
after the operation were 22,784 bytes in both runs. First-call time includes
compilation and execution; these single observations do not establish a
runtime speedup or regression.

The same check was repeated at the benchmark's dimensions: 51 state radii,
four species, four energy nodes, 13 cotangent rows, and a 7x16x11 database.

| Probe: benchmark dimensions, synthetic coefficients | Peak JAX allocation (bytes) | First call (s) |
| --- | ---: | ---: |
| Default compiler guards | 1,459,785,728 | 51.723 |
| 64 KiB compiler guards | 69,205,504 | 65.412 |

This is a 95.3% peak reduction, from 1,392.2 MiB to 66.0 MiB. The derivative
arrays passed the same tolerance check. `a_b` was bitwise identical;
`Er_list` had maximum scaled difference 1.03e-13, using
`abs(new-old) / max(1, abs(old))`. The synthetic inputs can
produce large derivative magnitudes, so absolute differences alone are not a
useful accuracy measure. This single first-call observation was 26% slower;
it includes compilation and does not measure steady-state execution. No
full-benchmark timing improvement is claimed.

Interpax 0.3.16 changes the interpolation implementation to a packed
`_stack_fields` representation, which appears in the supplied failing HLO.
The older 0.3.14 implementation still has a large peak under JAX 0.10.0.
Therefore changing Interpax alone does not eliminate this overhead. The two
versions' derivative outputs agreed within `rtol=atol=3e-10` in the small
probe; this does not validate every interpolation mode.

Disabling `multi_output_fusion` did not reduce the small probe's peak in a
separate JAX 0.11.2 check. That compiler-pass workaround is not part of the
proposed launch. No package downgrade or numerical algorithm change is
required for the guard-padding experiment.

## Why the guard setting matters

XLA allocates guard regions around buffers while profiling candidate GPU
kernels. Its [default setting](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc)
is 8 MiB on each side of each buffer: 16 MiB of padding per buffer before
counting its data. Many-input/output fusions can therefore require substantial
temporary memory even when their arrays are small. The
[option description](https://github.com/openxla/xla/blob/main/xla/xla.proto)
explicitly identifies smaller padding as a way to reduce autotuning memory.

Use `--xla_gpu_redzone_padding_bytes=65536` to retain 64 KiB guards on each
side. This changes the compiler's out-of-bounds detection coverage during
autotuning. It does not change the interpolation formula, root selection,
time-step schedule, solver tolerances or AD contract. Autotuning remains
enabled. Different compiler choices can still cause floating-point rounding
differences in other computations; the comparison above checks this component.

The measurements establish a large avoidable allocation in the implicated
component. They do not yet prove the full remote benchmark fits, nor rule out
other live-array or executable memory pressure in that process.

## Same full benchmark with the compiler allocation change

Run in the new main-based environment. Define the log directory before `tee`:

```bash
cd "$HOME/NEOPAX" || exit 1
neopax_main_run=outputs/dependency_main_redzone64k
mkdir -p "$neopax_main_run"
set -o pipefail

env -u PYTHONPATH -u LD_LIBRARY_PATH -u JAX_PLATFORM_NAME \
  -u JAX_COMPILATION_CACHE_DIR \
  JAX_PLATFORMS=cuda,cpu \
  JAX_ENABLE_X64=true \
  JAX_ENABLE_COMPILATION_CACHE=0 \
  XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_redzone_padding_bytes=65536" \
  NEOPAX_REVERSE_STEP_DIAGNOSTICS=1 \
  /usr/bin/time -v "$HOME/jax_env/bin/python" \
  examples/benchmarks/benchmark_transport_reverse_ad_only.py \
  --config examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml \
  --reverse-parameter-mode profiles_plus_realtime_geometry \
  --reverse-geometry-parameter RBC:1:0,ZBS:1:0 \
  --realtime-geometry-gradient-path reverse_payload \
  --optimization-api-profile-dofs include \
  --objective all \
  --max-reverse-accepted-steps 1000 \
  --reverse-segment-length 50 \
  --radau-jacobian-reuse-mode legacy \
  --timing-mode jit-warm \
  --reverse-stage-adjoint-solve-mode block \
  --reverse-rhs-transpose-mode explicit_database \
  --reverse-step-bwd-mode reduced_cotangent_call_boundary \
  --reverse-initial-cache-support-pullback-mode scalar \
  --reverse-rebuild-support-pullback-mode separate \
  --reverse-final-objective-cotangent-mode grouped_vjp \
  --reverse-bootstrap-cotangent-mode joint_local_vjp_upar_only \
  --initial-Er-root-ad jax_selected_root \
  --full-transport-shared-payload-smoke \
  --reverse-segment-start-replay-mode minimal \
  --reverse-segment-primal-record-mode reuse_segment_primal_record \
  --reverse-schedule-artifact-mode reuse_static_probe \
  --reverse-replay-jacobian-reuse-mode recorded_forward \
  2>&1 | tee "$neopax_main_run/ad.log"
```

`cuda,cpu` keeps CUDA as the default and allows the VMEX host callback to use
the CPU backend. It is not a request to solve transport on the CPU. Benchmark
reports retain their existing output paths; this directory separates the log.
