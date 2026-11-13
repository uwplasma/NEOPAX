# Isolated bootstrap geometry derivative check

The sixteen-step AD/FD comparison has a large bootstrap geometry discrepancy,
including at a matching ZBS baseline. Small component tests pass on the old
and new stacks, but do not validate that actual plasma state. This diagnostic
checks the fixed-final-state geometry term on the benchmark equilibrium.

It performs one baseline equilibrium/runtime build and one adaptive transport
prefix with 16 accepted steps. That final density, pressure and electric field
are held fixed for every perturbed geometry. The equilibrium endpoints use the
same frozen implicit VMEX tangent as the existing FD benchmark. Geometry and
NTX tables are rebuilt; no perturbed transport trajectory or initial-root
derivative enters this isolated comparison.

The AD side uses the production corrected-Upar bootstrap state/geometry VJP,
the established database table/coordinate transpose, the recorded NTX scan
transpose and the geometry-payload-to-VMEX-state transpose. It contracts the
last result with the same VMEX tangent used by FD. Two component rows separate
direct geometry from the complete database contribution. No reverse transport
sweep or additional equilibrium solve is performed. This does not independently
validate the implicit VMEX tangent or its raw-block transpose.

Run on the main-utility dependency stack, after pulling the diagnostic changes:

```bash
cd "$HOME/NEOPAX" || exit 1
mkdir -p outputs/bootstrap_geometry_gradient_RBC
set -o pipefail

env JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$HOME/jax_env/bin/python" \
  examples/benchmarks/check_bootstrap_geometry_gradient.py \
  --parameter RBC:1:0 \
  --accepted-step-limit 16 \
  --out-dir outputs/bootstrap_geometry_gradient_RBC \
  2>&1 | tee outputs/bootstrap_geometry_gradient_RBC/run.log
```

The default is the same 51-point realtime-geometry benchmark TOML. It requires
`vmec_lane="ad"`, `ntx_scan_runtime`, GPU execution and float64. The script
selects the same structured recorded scan as the AD benchmark; it leaves the
TOML and all physics settings unchanged. There are no required XLA flags.
For ZBS, replace `RBC:1:0` with `ZBS:1:0` and use a different output directory.

Three central FD steps are measured: `3h`, `h`, and `h/3`, where `h` is the
existing benchmark's coefficient-scaled FD step. `--step-scales` can change
that list. One run entails six perturbed NTX runtime builds, so it belongs on
the large machine. It is not a quick scalar unit test.

`summary.json` saves baseline metadata, the accepted/attempted trace, both
bootstrap primal values (full flux and Upar-only), the AD components, both
endpoint objectives and every FD/error value. It is updated after each
completed pair. `comparison.csv` prints numbers in scientific notation;
relative errors use `abs(AD-FD)/abs(FD)` and are not percentages. Relative error
is undefined if FD is zero. `baseline_final_state.npz` saves the common plasma
profiles. These are generated output artifacts, not source data files.

Interpretation:

- Check the full-flux and Upar-only baseline values first. A discrepancy there
  means the two terminal-bootstrap primal implementations need investigation
  before their derivatives can be compared.
- If FD stabilizes across steps and disagrees with AD, the problem is already
  present in the direct bootstrap geometry/database chain, without transport
  replay. The separate AD components identify the next boundary to inspect.
- If FD changes substantially with step, that FD column has not converged.
  Smaller steps alone are not evidence of greater accuracy.
- If the fixed-state term agrees, the remaining full-gradient investigation
  moves to the plasma-state response, initial-root boundary and transport
  support contributions. Passing this isolated check does not certify them.

The older full-run FD `fixed_final_state_geometry_fd` is the corresponding
term, not the total derivative. For RBC the supplied value was approximately
`3.015968e-2`, while its total bootstrap FD was `1.209009e-1`. Compare to that
saved value only when the baseline objective and transport prefix also match.
