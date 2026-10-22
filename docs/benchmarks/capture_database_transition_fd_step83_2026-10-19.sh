#!/usr/bin/env bash
# Diagnostic capture only: this prefix is not the 20-second AD/FD comparison.
# An expected replay failure writes an NPZ and exits nonzero.
set -euo pipefail

neopax_repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$neopax_repo_dir"
mkdir -p outputs/database_transition_fd_full_t20_2026-10-19

env -u JAX_COMPILATION_CACHE_DIR \
  JAX_ENABLE_COMPILATION_CACHE=0 \
  XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=" \
  PYTHONPATH="$HOME/VMEX:$HOME/NTX/src" \
  /usr/bin/time -v \
  python examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py \
    --config examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml \
    --parameter n0 \
    --accepted-step-limit 83 \
    --radau-jacobian-reuse-mode legacy \
    --geometry-fd-lane frozen_linearized \
    --replay-mode accepted \
    --fd-accepted-replay-contract reverse_frozen_metadata \
    --initial-Er-root-ad jax_selected_root \
    --initial-Er-root-fd-root-lane frozen_linearized \
    --diagnose-fd-endpoint \
    2>&1 | tee outputs/database_transition_fd_full_t20_2026-10-19/n0_capture_step83.log
