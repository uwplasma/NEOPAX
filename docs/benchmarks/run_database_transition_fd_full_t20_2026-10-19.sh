#!/usr/bin/env bash
# Full-time FD comparison for the 2026-10-19 recorded-forward AD reference.
# Run on the GPU machine with jax_env active; no accepted-step prefix limit.
set -euo pipefail

neopax_repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$neopax_repo_dir"

neopax_fd_dir=outputs/database_transition_fd_full_t20_2026-10-19
mkdir -p "$neopax_fd_dir"

# Preserve the completed AD JSON before another AD benchmark overwrites it.
neopax_ad_json=outputs/autodiff_transport_lagged_ntx/reverse_ad/transport_reverse_ad_only_full_transport_shared_payload_smoke_replay_jacobian_recorded_forward.json
cp -- "$neopax_ad_json" "$neopax_fd_dir/ad_reference.json"

for parameter in \
  n0 T0 \
  density_shape_power temperature_shape_power \
  density_shape_alpha temperature_shape_alpha \
  RBC:1:0 ZBS:1:0
do
  neopax_parameter_file=${parameter//:/_}

  env -u JAX_COMPILATION_CACHE_DIR \
    JAX_ENABLE_COMPILATION_CACHE=0 \
    XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
    PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
    /usr/bin/time -v \
    python examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
      --config examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml \
      --parameter "$parameter" \
      --radau-jacobian-reuse-mode legacy \
      --geometry-fd-lane frozen_linearized \
      --replay-mode accepted \
      --fd-accepted-replay-contract reverse_frozen_metadata \
      --initial-Er-root-ad jax_selected_root \
      --initial-Er-root-fd-root-lane frozen_linearized \
      --diagnose-fd-endpoint \
      2>&1 | tee "$neopax_fd_dir/${neopax_parameter_file}.log"

  # The runner writes its usual per-parameter path. Keep full-time copies
  # together so a later 16-step run cannot overwrite this comparison set.
  cp -- \
    "outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/${neopax_parameter_file}_forward_fd_frozen_linearized_summary.json" \
    "$neopax_fd_dir/${neopax_parameter_file}_summary.json"
done
