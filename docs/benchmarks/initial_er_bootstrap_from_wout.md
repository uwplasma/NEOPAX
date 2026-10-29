# Ambipolar electric field and momentum-corrected bootstrap current from a wout

`examples/optimization/plot_initial_er_root_from_wout.py` calculates the selected
ambipolar electric field and the momentum-corrected bootstrap current for a fixed
VMEC `wout` and prescribed profiles. It builds one runtime, solves the radial
ambipolarity problem once, and evaluates the corrected parallel flow at that
selected field. It performs no equilibrium solve, optimization, transport time
evolution or reverse AD. A missing Boozer file is generated from the supplied
`wout` by default.

The `wout` supplies geometry. The TOML supplies the species, density and
temperature profiles, neoclassical model, resolution and root-selection settings.
The default remains the original three-species, no-He exact-runtime TOML:
`examples/Solve_Transport_Equations/Solve_Transport_equations_noHe_radau_ntx_exact_lagged_runtime.toml`.
Use `--config` to select different profiles or a runtime-scan database model.
A tabulated database must belong to the supplied geometry; an existing database
for another equilibrium cannot be reused simply by changing the `wout` path.

For example, on the remote NEOPAX machine:

```bash
cd "$HOME/NEOPAX" || exit 1

env PYTHONPATH="$HOME/VMEX:$HOME/NTX/src${PYTHONPATH:+:$PYTHONPATH}" \
"$HOME/jax_env/bin/python" \
  examples/optimization/plot_initial_er_root_from_wout.py \
  outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization/optimized/wout_QI_neopax_geometry_full_transport_optimized.nc \
  --out outputs/wout_er_bootstrap/optimized_initial_er.png
```

This example uses the default prescribed profiles. To use the four-species
initial profiles and runtime database settings from the geometry-only
full-transport optimization instead, add:

```bash
  --config examples/optimization/full_transport_database_vacuum_t2.toml
```

That configuration specifies initial profiles. It does not load the evolved
terminal profiles from the geometry optimization's transport solution, so the
result need not reproduce its terminal electric field or bootstrap current.
Also retain any profile or numerical overrides used by the intended comparison.

Use `--boozer-file path/to/boozmn.nc` to select the matching Boozer file explicitly.
If generation is needed, `--mboz` and `--nboz` set its resolution (both default to
18); `--jit-boozer` is optional. Geometry in the TOML is replaced by the supplied
`wout`/Boozer files, so a realtime-VMEC TOML does not cause a new equilibrium solve.

With the output path in the example, the script writes:

- `optimized_initial_er.png`: electric field in kV/m.
- `optimized_initial_er.csv`: existing `rho,Er,finite_mask` columns.
- `optimized_initial_er_bootstrap_current.png`: bootstrap current in kA/m².
- `optimized_initial_er_bootstrap_current.csv`: `rho,Er_kV_m,Jboot_A_m2,Jboot_kA_m2,Jboot_scaled_1e5_A_m2,finite_mask`.

`--bootstrap-out path/to/bootstrap.png` overrides the bootstrap output path.
Both output directories are created automatically. `--no-bootstrap` retains the
original Er-only workflow and avoids the corrected-flux evaluation. The original
Python helper `solve_initial_er_root_from_wout` also keeps its three-value return.
The new `solve_initial_er_and_bootstrap_from_wout` returns rho, Er, bootstrap
current in units of 1e5 A/m², and the finite-root mask.

Bootstrap current uses the existing optimization helper's
`evaluate_momentum_corrected_fluxes`, corrected `Upar_neo` (or corrected `Upar`),
species weighting and normalization. It never substitutes an uncorrected flux
evaluation. One optimizer current unit is 1e5 A/m², or 100 kA/m².

Where a root is unavailable, `finite_mask` is zero and bootstrap current is NaN;
plots and the combined bootstrap CSV also mask the field there. The legacy Er
CSV retains the initializer's fallback value with `finite_mask=0`. No finite
roots, a missing corrected evaluator, or nonfinite current at a valid root stops
the combined calculation with an error.

Validation uses mocked runtimes/root solvers, small analytic flow arrays and
temporary CSV/PNG files in `tests/test_initial_er_bootstrap_from_wout.py`.
No physical VMEC, Boozer, NTX, ambipolarity or transport solve is run locally.
