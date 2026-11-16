# Saved NTX electric-field sensitivity check

The generic baseline check has reproduced coefficient differences between
two independent FD builds. AD, FD_1 and archived generic AD agree; FD_2
differs. The user's subsequent saved-array readout confirms the same package
versions and source hashes in all three jobs: JAX/jaxlib 0.10.0, NTX 0.3.0,
VMEX 0.13.0 and SOLVAX 0.36.2.

The reconstructed field inputs and unscaled D13 values are:

| Scan indices (radius, nu, field) | AD and FD_1 Es | FD_2 Es | AD/FD_1 raw D13 | FD_2 raw D13 |
| --- | --- | --- | ---: | ---: |
| 5,0,6 | `0x1.5c9b91249ff79p-8` | `0x1.5c9b91249ff77p-8` | `-5.74865255079939907e-2` | `-5.74828607184022511e-2` |
| 6,0,7 | `0x1.c035ba9ccdac0p-7` | `0x1.c035ba9ccdabfp-7` | `2.90239850404327582` | `2.89288064921974719` |

These field differences are two ULPs and one ULP, respectively; both have
absolute magnitude `1.734723475976807e-18`. Because the raw values have
already divided out drds, the discrepancy is not explained by the final
database D13 rescaling alone. The fields were reconstructed with NumPy from
saved channels, not captured as realized GPU operator inputs. The identical
source files rule out a changed installed source between these jobs; they
do not identify the numerical cause.

`check_saved_ntx_field_sensitivity.py` isolates the two cases on the large
GPU. It checks that the saved dynamic surface fields agree, reconstructs
one surface per radius, and prepares it once. Reading the original VMEC
input supplies static metadata; no equilibrium is solved. It reuses the
production single-point scan body for the saved field pair and repeats the
AD-side input exactly. It also evaluates the production field-channel
formula on the GPU and tests those field values, to distinguish the NumPy
reconstruction from that calculation.

The scan body runs one point at a time rather than the original 176-point
radial batch. The CSV therefore reports each replay's difference from the
saved coefficient, rather than assuming the compiled context reproduces it.
An additional scalar solve reports the primary-source retained Schur residual and Onsager
residual. Its coefficient difference from the scan body is recorded because
returning extra diagnostics may change compilation. The Schur residual is
not a check of the parallel-source residual, a full-system residual or a
condition-number estimate.

The check saves both prepared systems as arrays for later numerical
inspection, without storing a full Legendre factorization or dense global
matrix. It refuses a CPU backend and rejects changed recorded source hashes
or package versions. No solver/interpolation formula or AD method changes.
Local checks use NumPy and a mocked CPU backend only; the GPU replay has not
been run locally.

On the large machine after pushing/pulling the script:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  set -o pipefail
  env JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$HOME/jax_env/bin/python" \
    examples/benchmarks/check_saved_ntx_field_sensitivity.py \
    --run-dir outputs/generic_ntx_baselines_20261116 \
    --out-dir outputs/generic_ntx_field_sensitivity_20261116 \
    2>&1 | tee outputs/generic_ntx_field_sensitivity_20261116.log
) || printf '\nSaved-point replay stopped; terminal kept open.\n'
```

Use a new empty output directory. This runs coefficient solves for the two
saved cases, not another database scan or AD/FD transport benchmark.
