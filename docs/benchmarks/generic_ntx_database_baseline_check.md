# Generic NTX database baseline comparison

This check asks whether the generic-interpolation AD and FD jobs generate
the same unperturbed NTX coefficient tables. It compares no NTSS1D results
and calculates no objective gradients.

The old generic run in `outputs/b0_fix_ad_fd16_20261115_ZV2sFV` retains its
AD baseline tables but not its FD baseline tables. The user's file inventory
confirms this; `explicit_support_bars` are cotangents, not coefficient tables.
Consequently a direct table comparison of those historical jobs is impossible
from their saved snapshots alone. Comparing different interpolators' AD
reports would not answer that question.

`check_generic_ntx_database_baselines.py` launches three sequential, independent
GPU processes. Each uses the original black-box generic benchmark TOML and
`build_runtime_context`, the entrypoint shared by the production AD baseline
and profile-parameter FD baseline. The first applies the production AD
benchmark's `ntx_scan_coefficient_reverse_mode="structured"` and
`ntx_scan_record_primal=True` overrides; the other two retain the FD config.
All three use generic interpolation. The repeated FD build tests variation
between independent processes using the same route.

The sole common override is `general.mode="ambipolarity"`, which skips
transport-mode root initialization after database construction. The checker
never calls the ambipolarity solver, transport solver or derivative routines.
VMEX equilibrium and full NTX coefficient solves are required, on the large
GPU machine only; the worker rejects a CPU backend.

It saves each geometry, database, scan surface and channel payload in NPZ,
plus configuration, source hashes and package-version metadata. The CSV
separates individual coefficient-table differences from differences in
geometry, axes and scan inputs, and handles matching NaNs explicitly.
All numerical differences use scientific notation; they are coefficient
differences, not AD/FD gradient errors. The archived generic AD comparison
is additional evidence and may involve older source/dependency revisions.
Matching fresh builds would not prove that every historical FD build matched.

After pushing/pulling the diagnostic, run on the large machine:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  neopax_check=outputs/generic_ntx_baselines_20261116
  mkdir -p outputs || exit 1
  set -o pipefail
  "$HOME/jax_env/bin/python" \
    examples/benchmarks/check_generic_ntx_database_baselines.py \
    --reference outputs/b0_fix_ad_fd16_20261115_ZV2sFV/ad_snapshots/ad_terminal_tqq6x2mm.npz \
    --out-dir "$neopax_check" \
    2>&1 | tee outputs/generic_ntx_baselines_20261116.log
) || printf '\nBaseline check stopped; terminal kept open.\n'
```

Use a new, empty output directory. Worker progress is in `ad/run.log`,
`fd_1/run.log` and `fd_2/run.log`. The final `comparison.csv` reports fresh
AD versus each FD, the two FD builds against one another, and archived AD
versus fresh AD. This is not a sixteen-step transport benchmark rerun.

Local verification is limited to NumPy comparisons, configuration routing
and a mocked CPU-refusal check. No local JAX or physical solve is run; the
actual GPU database-build comparison remains pending.
