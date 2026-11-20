# Saved full-support derivative comparison

The NTSS1D ZBS case `ntss1d_zbs_shared_ad_root_psoBBD` has matching AD/FD
initial states, final states, final time and support data. Its terminal-state
derivative checks agree closely for Er and for the bootstrap state contribution,
but the full parameter derivatives still differ.

The remote saved-array preflight found exactly one terminal capture
(`ad_terminal_8wkp5r2c.npz`) and one pre-scan-fold capture
(`ad_before_scan_fold_5j3jig0x.npz`). All 13 objective labels match, all
42 floating support fields are finite, every required cotangent is present
and finite, and there are no native face-coefficient bars. The existing AD
capture therefore has the data needed for this check; no AD rerun is needed.

Let `S` be the transport geometry and monoenergetic database, `y` the final
packed state, and `p` the geometry parameter. The new NumPy-only comparison
checks two chain-rule contractions along the actual FD support direction:

1. `terminal_full_chain = L_y · (y_plus-y_minus)/(2h) + L_S · (S_plus-S_minus)/(2h)`.
   Both bars come from `ad_terminal_*.npz`. This includes the terminal cost's
   explicit geometry/table dependence, which matters for bootstrap.
2. `support_chain = total_support_bar · (S_plus-S_minus)/(2h)`.
   The total bar comes from `ad_before_scan_fold_*.npz`, after the transport
   reverse sweep and initial-state/root pullbacks, before the NTX scan fold
   and VMEX geometry contraction.

Both are compared with the actual full FD gradient for `p`. The support bar
treats geometry and database fields as independent inputs; the FD direction
includes their actual coupled variations, including derived database coordinates.

If the terminal chain agrees but the support chain does not, the fixed-support
initialization/transport derivative, runtime reconstruction or its FD
approximation needs investigation.
If both agree with FD while the final parameter AD gradient does not, the next
boundary is the NTX/VMEX pullback or the difference between its derivative and
the FD support secant. This is a directional check at one point, not a proof
that either solver or every derivative is correct. Finite perturbations,
conditioning and solver tolerances still need assessment before calling a
remaining difference a code bug.

The previous FD endpoint snapshots contain states but not perturbed support.
Those input arrays cannot be reconstructed from endpoint states. The focused
FD recorder change saves the existing `runtime.geometry` and cached database
in each endpoint snapshot. It performs no additional model evaluation or solve,
and only writes arrays when `NEOPAX_GRADIENT_AUDIT_DIR` is enabled. The AD
captures can be reused; only FD needs another run with the added recording.
Species and energy-grid arrays are also recorded at the FD baseline and both
endpoints, and must stay byte-identical across FD. The older AD terminal file
does not contain those fixed inputs, so this cannot independently establish
their numerical equality with AD.

After the new FD capture, use the existing comparison with one additional file:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  # Set neopax_case to the new FD output directory containing an ad symlink.
  neopax_fold=("$neopax_case"/ad/snapshots/ad_before_scan_fold_*.npz)
  [ "${#neopax_fold[@]}" -eq 1 ] && [ -f "${neopax_fold[0]}" ] || exit 1
  "$HOME/jax_env/bin/python" examples/benchmarks/compare_transport_gradient_audit.py \
    --ad-dir "$neopax_case/ad/snapshots" \
    --fd-dir "$neopax_case/fd_ZBS_1_0/snapshots" \
    --fd-report "$neopax_case/fd_ZBS_1_0/summary.json" \
    --ad-before-scan-fold "${neopax_fold[0]}" \
    --out "$neopax_case/fd_ZBS_1_0/support_chain.csv"
) || printf '\nSupport-chain comparison stopped; terminal kept open.\n'
```

The option is restricted to geometry parameters with the geometry/database-only
support contract. Profile parameters have additional direct profile terms;
native VMEC face-coefficient bars are also outside this comparison. Missing
endpoint support, missing bars, changed baseline support, label mismatches,
nonfinite arrays and incompatible shapes/dtypes cause a failure rather than
silently dropping a contribution. Host support contractions use `math.fsum`.
The original terminal-state-only command remains available without the new option.

Four NumPy-only tests exercise an independently differentiated nonlinear
composition, distinguish incorrect terminal bars from incorrect full support
bars, check invalid/missing captures and unsupported contracts, and run both
the legacy and extended CLI. The existing baseline diagnostics also pass.
The remote preflight checks data availability, not derivative correctness.
GPU/FD validation of the support-chain mode is pending.
