# Full transport AD with the VMEX main response

`--vmec-response-mode main` selects VMEX's public, certified multi-RHS
linear-response APIs throughout the existing transport AD benchmark:

- Transport payload contractions use `implicit_state_tangent_multi_rhs`,
  sharing one factorization across the requested geometry directions.
- Pure geometry objectives and payload transpose fallbacks use
  `implicit_state_pullback_multi_rhs(solver="block")`.
- The FD benchmark accepts the same option and uses the same public tangent
  for its frozen-linearized geometry endpoints.

Both probe and response chunk sizes are one. This bounds the workspace of
the VMEX response without splitting the transport replay or its objectives.
A failed tangent convergence certificate raises, or produces nonfinite
tangents inside a traced caller; it cannot silently become a valid gradient.

The setting can also be supplied as `geometry.vmec_response_mode` in a TOML.
The existing `raw_block` response remains the default and the comparison
control. The selected response is saved in AD/FD summaries and compact
geometry captures, and restored when replaying those captures.

This changes the VMEX derivative response only. The configured VMEC `ad`
primal, NTX rules, physics, accepted-step transport map, recorded-forward
Newton/Jacobian setup and segmented reverse method are unchanged. This patch
does not include the separate pending two-stage root convergence guard.

## Verification and interpretation

The local tests use tiny NumPy arrays and mocked VMEX APIs, executing the
production routing functions extracted by AST. They cover all-direction
sharing, convergence failures, transport payload contractions, pure geometry
transpose, matching FD routing, the default raw route and CLI/config wiring.
They run no JAX or physical solvers. Numerical results and performance for
this response choice remain to be measured on the GPU machine.

The command below runs all objective rows against the six profile DOFs plus
`RBC:1:0` and `ZBS:1:0`, for 16 accepted transport steps. Its snapshots allow
the initial state, terminal state, time and objective baselines to be checked
against the existing raw-response result before comparing gradients. A
different initial root is a different comparison point; response agreement
cannot be inferred from such a pair. For a new FD comparison, also select
`--vmec-response-mode main` in the FD benchmark.

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git switch en/v0_main || exit 1
  git pull --ff-only origin en/v0_main || exit 1

  mkdir -p outputs || exit 1
  neopax_run=$(mktemp -d outputs/ad16_vmex_main_response_XXXXXX) || exit 1
  neopax_config=examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml
  cp "$neopax_config" "$neopax_run/config.toml" || exit 1
  git rev-parse HEAD > "$neopax_run/neopax_commit.txt" || exit 1
  git -C "$HOME/VMEX" rev-parse HEAD > "$neopax_run/vmex_commit.txt" || exit 1
  git -C "$HOME/NTX" rev-parse HEAD > "$neopax_run/ntx_commit.txt" || exit 1
  "$HOME/jax_env/bin/python" -m pip freeze > "$neopax_run/packages.txt" || exit 1
  set -o pipefail

  env -u PYTHONPATH -u LD_LIBRARY_PATH -u JAX_PLATFORM_NAME \
    -u JAX_COMPILATION_CACHE_DIR -u XLA_FLAGS -u XLA_PYTHON_CLIENT_ALLOCATOR \
    JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
    JAX_ENABLE_COMPILATION_CACHE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    NTX_DISABLE_COMPILATION_CACHE=1 VMEX_COMPILATION_CACHE=disabled \
    NEOPAX_REVERSE_STEP_DIAGNOSTICS=1 \
    NEOPAX_GRADIENT_AUDIT_DIR="$neopax_run/ad_snapshots" \
    /usr/bin/time -v "$HOME/jax_env/bin/python" \
    examples/benchmarks/benchmark_transport_reverse_ad_only.py \
    --config "$neopax_config" \
    --vmec-response-mode main \
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
  tar -czf "$neopax_run.tar.gz" -C "$neopax_run" . || exit 1
  printf '\nSAVED RESULTS: %s/NEOPAX/%s.tar.gz\n' "$HOME" "$neopax_run"
) || printf '\nRun stopped after an error; terminal kept open.\n'
```
