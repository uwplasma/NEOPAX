# Main b0 correction: matched AD and FD16 rerun

Use `en/v0_main`, the main-based installed VMEX/NTX/SOLVAX dependencies,
and `$HOME/jax_env/bin/python` (JAX 0.10.0), after publishing the b0
argmax/gather correction and its test. The old-stack comparison stays in
its separate checkout/environment.

Both runs stop after sixteen accepted steps on the unchanged 51-point
benchmark TOML. AD uses the same full shared-payload command as the previous
AD16 capture, including recorded-forward Jacobian reuse and four-step
segments. FD uses the original relative/absolute steps, frozen-linearized
geometry and initial root, and accepted reverse-frozen metadata. Its report
records that contract as `fd_replay_jacobian_reuse_mode=recorded_forward`.

The loop covers both geometry columns and all six profile columns. Both
benchmarks include softmax Er, left/right Er, net power, bootstrap,
transition radius, location moment and strength, alongside the remaining
transport diagnostics. The FD reports cover the thirteen transport rows;
AD also reports its geometry-only rows.

Run sequentially on the large machine. No local transport/solver/GPU runs
were performed to prepare these instructions. The captured state arrays are
needed to verify common AD/FD baselines; a successful run alone does not
establish gradient accuracy. The patch changes how the same maximum value
is differentiated, with first-winner subgradients at exact ties.

## Large-machine command

The block runs in a subshell, initializes all its variables, and catches
errors without closing the interactive terminal. It creates a unique
results directory and copies summaries immediately after each run because
the benchmark scripts themselves write fixed default summary paths.

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git switch en/v0_main || exit 1
  git pull --ff-only origin en/v0_main || exit 1

  unset PYTHONPATH LD_LIBRARY_PATH JAX_PLATFORM_NAME JAX_COMPILATION_CACHE_DIR
  unset XLA_FLAGS XLA_PYTHON_CLIENT_ALLOCATOR NEOPAX_GRADIENT_AUDIT_DIR
  export JAX_PLATFORMS=cuda,cpu
  export JAX_ENABLE_X64=true JAX_ENABLE_COMPILATION_CACHE=0
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export NTX_DISABLE_COMPILATION_CACHE=1 VMEX_COMPILATION_CACHE=disabled
  export NEOPAX_REVERSE_STEP_DIAGNOSTICS=1

  neopax_python="$HOME/jax_env/bin/python"
  neopax_config=examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml
  mkdir -p outputs || exit 1
  neopax_run=$(mktemp -d outputs/b0_fix_ad_fd16_20261115_XXXXXX) || exit 1
  printf 'RESULTS DIRECTORY: %s\n' "$HOME/NEOPAX/$neopax_run"
  cp "$neopax_config" "$neopax_run/config.toml" || exit 1
  git rev-parse HEAD > "$neopax_run/neopax_commit.txt" || exit 1
  git -C "$HOME/VMEX" rev-parse HEAD > "$neopax_run/vmex_commit.txt" || exit 1
  git -C "$HOME/NTX" rev-parse HEAD > "$neopax_run/ntx_commit.txt" || exit 1
  "$neopax_python" -m pip freeze > "$neopax_run/packages.txt" || exit 1
  set -o pipefail

  "$neopax_python" tests/test_vmec_surface_b0.py \
    2>&1 | tee "$neopax_run/b0_test.log" || exit 1

  NEOPAX_GRADIENT_AUDIT_DIR="$neopax_run/ad_snapshots" \
  /usr/bin/time -v "$neopax_python" \
    examples/benchmarks/benchmark_transport_reverse_ad_only.py \
    --config "$neopax_config" \
    --reverse-parameter-mode profiles_plus_realtime_geometry \
    --reverse-geometry-parameter RBC:1:0,ZBS:1:0 \
    --realtime-geometry-gradient-path reverse_payload \
    --optimization-api-profile-dofs include \
    --objective all \
    --accepted-step-limit 16 \
    --reverse-segment-length 4 \
    --radau-jacobian-reuse-mode legacy \
    --reverse-replay-jacobian-reuse-mode recorded_forward \
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
    --reverse-schedule-artifact-mode reuse_static_probe \
    2>&1 | tee "$neopax_run/ad.log" || exit 1

  cp outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_full_transport_shared_payload_smoke_replay_jacobian_recorded_forward.json \
    "$neopax_run/ad_summary.json" || exit 1

  for neopax_parameter in \
    RBC:1:0 ZBS:1:0 n0 T0 \
    density_shape_power temperature_shape_power \
    density_shape_alpha temperature_shape_alpha
  do
    neopax_safe_parameter="${neopax_parameter//:/_}"
    neopax_fd_dir="$neopax_run/fd_$neopax_safe_parameter"
    mkdir -p "$neopax_fd_dir" || exit 1

    NEOPAX_GRADIENT_AUDIT_DIR="$neopax_fd_dir/snapshots" \
    /usr/bin/time -v "$neopax_python" \
      examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
      --config "$neopax_config" \
      --parameter "$neopax_parameter" \
      --accepted-step-limit 16 \
      --fd-rel-step 3e-7 \
      --fd-abs-step 1e-10 \
      --radau-jacobian-reuse-mode legacy \
      --geometry-fd-lane frozen_linearized \
      --replay-mode accepted \
      --fd-accepted-replay-contract reverse_frozen_metadata \
      --initial-Er-root-ad jax_selected_root \
      --initial-Er-root-fd-root-lane frozen_linearized \
      --diagnose-fd-endpoint \
      2>&1 | tee "$neopax_fd_dir/run.log" || exit 1

    cp "outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/${neopax_safe_parameter}_forward_fd_frozen_linearized_summary.json" \
      "$neopax_fd_dir/summary.json" || exit 1
  done

  tar -czf "$neopax_run.tar.gz" -C "$neopax_run" . || exit 1
  printf 'SAVED RESULTS: %s\n' "$HOME/NEOPAX/$neopax_run.tar.gz"
) || printf '\nRun stopped after an error; terminal kept open.\n'
```

This command has been checked against the scripts' argument declarations
and output-path functions using source inspection and Bash syntax checking.
The full runs and the JAX test above still require execution on the GPU
machine. Neither the old-stack result nor the saved-array correction
substitutes for that verification. The remaining RBC AD/FD discrepancy
after the dominant b0 correction is unresolved.
