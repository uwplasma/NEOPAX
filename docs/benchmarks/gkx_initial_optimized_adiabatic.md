# Adiabatic-electron GKX comparison of saved geometry equilibria

`examples/optimization/run_gkx_initial_optimized_adiabatic.py` compares
nonlinear ion heat flux on the initial and optimized equilibria exported
by the geometry-only NEOPAX optimization. It reads the existing files:

```text
<optimization-folder>/initial/wout_QI_neopax_geometry_full_transport_initial.nc
<optimization-folder>/optimized/wout_QI_neopax_geometry_full_transport_optimized.nc
```

Other unambiguous `wout*.nc` files in those two directories are supported;
use `--initial-wout` and `--optimized-wout` to select explicit files. The
script never substitutes the optimized equilibrium for a missing initial
one and does not solve VMEC or NEOPAX transport again.

Both equilibria use the same GKX numerical template, one kinetic ion species,
electrostatic nonlinear physics, and adiabatic electrons. Defaults are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `--s` | 0.5 | Normalized toroidal flux passed directly to `geometry.torflux`; rho is sqrt(s) |
| `--LT` / `--tprim` | 3.0 | Normalized inverse temperature gradient a/L_T |
| `--Ln` / `--fprim` | 0.0 | Normalized inverse density gradient a/L_n |
| `--alpha` | 0.0 | Field-line label |
| `--t-max` | 700 | Normalized GKX time horizon |
| `--average-fraction` | 0.2 | Final fraction of the saved time interval used for statistics |

Thus `LT=3` means **a/L_T=3**, as in the existing GKX flux-scan example,
rather than a dimensional temperature-gradient length. The imposed local
profiles are identical for both runs; they are not loaded from NEOPAX's
evolving transport profiles. In particular this comparison does not import
the optimized transport Er profile or its shear.

## Separate Python environments

Use a dedicated GKX environment. GKX 2.7.0 requires JAX >= 0.10.1, whereas
the NEOPAX runs discussed here used JAX/JAXLIB 0.7.2. Installing GKX into
that NEOPAX environment can replace its JAX, Equinox, Boozer and Solvax
versions. In the supplied installation log, pip selected JAX 0.12.0.1,
which also conflicts with the installed orthax/interpax requirement JAX < 0.11.

For the specific upgrade shown in that log, remove GKX from `jax_env` and
restore the five prior package versions listed there:

```bash
"$HOME/jax_env/bin/python" -m pip uninstall -y gkx &&
"$HOME/jax_env/bin/python" -m pip install \
  'jax==0.7.2' 'jaxlib==0.7.2' 'equinox==0.13.2' \
  'booz_xform_jax==0.1.1' 'solvax==0.11.2' &&
"$HOME/jax_env/bin/python" -m pip check
```

This restores the package versions shown in the log, rather than
reconstructing a full environment lockfile. The log does not show the
CUDA plugin packages being upgraded. Check GPU availability after repair.

Create a separate environment for GKX. The following uses CUDA 12 wheels
and keeps GKX on the JAX 0.10 series that meets its declared minimum:

```bash
python3 -m venv "$HOME/gkx_env" &&
"$HOME/gkx_env/bin/python" -m pip install --upgrade pip &&
"$HOME/gkx_env/bin/python" -m pip install \
  'jax[cuda12]>=0.10.1,<0.11' -e "$HOME/GKX" &&
"$HOME/gkx_env/bin/python" -m pip check &&
"$HOME/gkx_env/bin/python" -c \
  'import jax; print(jax.__version__, jax.devices()); assert jax.default_backend() == "gpu", "GKX requires a working GPU setup for this run"'
```

Use the matching accelerator extra if the machine requires a different
CUDA setup; see the [official JAX installation instructions](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).
Creating the environment or listing devices does not run a simulation.
There is no need to install NEOPAX in `gkx_env`: the comparison controller
uses standard Python, NumPy, NetCDF4 and plotting, and starts GKX separately.

## Run on the simulation machine

Commit/push the script, its test, and this note locally; then pull the branch
on the remote machine. GKX must be available there, normally at `$HOME/GKX`.
The script supplies its `src` directory to the worker's PYTHONPATH.

```bash
cd /home/exouser/NEOPAX || exit 1
git pull --ff-only origin en/v0_new || exit 1
set -o pipefail
mkdir -p outputs

/home/exouser/gkx_env/bin/python \
  examples/optimization/run_gkx_initial_optimized_adiabatic.py \
  outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --gkx-root "$HOME/GKX" \
  --s 0.5 --LT 3.0 --Ln 0.0 --alpha 0.0 \
  --t-max 700 \
  2>&1 | tee outputs/gkx_initial_optimized_adiabatic_s0p5.log
```

By default the worker uses the controller's interpreter, so both run in
`gkx_env` with this command. `--gkx-python "$HOME/gkx_env/bin/python"`
can also select the worker interpreter explicitly when invoking the
controller from another compatible environment.

The default template is `GKX/examples/common_input.toml`. Its grid,
velocity-space resolution, collision settings, field-line length and time
integrator are shared by the two runs. These settings are a starting point
for a comparison, not evidence of convergence for these equilibria. Supply
`--template path/to/custom.toml` for a different numerical setup; it must
contain exactly one positive-ion `[[species]]` block. The script overrides
the geometry, imposed gradients, adiabatic/electrostatic/nonlinear switches,
time horizon and output paths, and removes template restart input and fixed
`run.steps` overrides. The existing optimized-only radial scan remains
available as `run_gkx_optimized_flux_scan.py`.

## Outputs and reuse

By default outputs go below the optimization folder:

```text
gkx_adiabatic_comparison/s_0p5_LT_3_Ln_0_alpha_0/
  case_manifest.json
  initial/gkx.toml, gkx.out.nc, gkx.log, ...
  optimized/gkx.toml, gkx.out.nc, gkx.log, ...
  comparison.json
  heat_flux_history.csv
  heat_flux_summary.csv
  heat_flux_comparison.png
```

The runs execute sequentially in separate processes. GKX restart behavior
is inherited from the existing flux-scan runner: completed histories are
skipped, and unfinished cases can resume when a restart artifact exists.
The manifest records both equilibrium hashes and the template hash; a
different input or time horizon requires a fresh `--out-dir` so saved
outputs from a different case cannot silently be used in the comparison.

Add `--prepare-only` to write both input decks and the manifest without
starting GKX. Add `--plot-only` to the same command to regenerate comparison
outputs from existing histories without starting either run. Change
`--average-fraction` to change the averaging window without rerunning GKX.

Heat-flux statistics use elapsed-time weighting, including an interpolated
window boundary, because adaptive output samples may not be equally spaced.
The plotted standard deviation measures temporal variability; it is not an
uncertainty estimate or a saturation/convergence test. Fluxes retain GKX's
normalization, rather than being converted to physical heat power.

Local validation uses synthetic decks, mock workers and synthetic flux
histories only. It checks direct s (not s squared), equilibrium selection,
matching numerical settings and requested physics, stale-output rejection,
time-weighted statistics, and labeled CSV/plot generation. No GKX, VMEC,
NTX or NEOPAX transport simulation is run locally.
