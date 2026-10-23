#!/usr/bin/env bash
# Full-time AD of NEW root candidates, followed by actual downhill moves.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

neopax_candidate_out="${1:-outputs/new_root_candidates}"
mkdir -p "$neopax_candidate_out"

env -u JAX_COMPILATION_CACHE_DIR \
JAX_ENABLE_COMPILATION_CACHE=0 \
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src${PYTHONPATH:+:$PYTHONPATH}" \
/usr/bin/time -v \
python examples/benchmarks/benchmark_transport_reverse_ad_only.py \
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
  --root-loss-audit \
  --root-loss-audit-target 0.514 \
  --root-loss-audit-inner-radius 0.30 \
  --root-loss-audit-outer-radius 0.90 \
  --root-loss-audit-profile-trust-multiplier 3.7 \
  --root-loss-audit-ess-alpha 1.2 \
  2>&1 | tee "$neopax_candidate_out/ad.log"

cp outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_full_transport_shared_payload_smoke_root_loss_audit_replay_jacobian_recorded_forward.json \
  "$neopax_candidate_out/ad_summary.json"

env -u JAX_COMPILATION_CACHE_DIR \
JAX_ENABLE_COMPILATION_CACHE=0 \
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src${PYTHONPATH:+:$PYTHONPATH}" \
/usr/bin/time -v \
python examples/benchmarks/check_er_root_candidate_descent.py \
  --ad-report "$neopax_candidate_out/ad_summary.json" \
  --out-dir "$neopax_candidate_out/descent" \
  --step-sizes "${NEOPAX_ROOT_DESCENT_STEPS:-0.01}" \
  --candidates "${NEOPAX_ROOT_CANDIDATES:-local,annular}" \
  --groups "${NEOPAX_ROOT_DOF_GROUPS:-profiles,geometry}" \
  2>&1 | tee "$neopax_candidate_out/descent.log"
