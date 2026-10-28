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
| `--until-saturated` | Off | Enable GKX's native saturation assessment and early stopping; t_max remains the cap |

Thus `LT=3` means **a/L_T=3**, as in the existing GKX flux-scan example,
rather than a dimensional temperature-gradient length. The imposed local
profiles are identical for both runs; they are not loaded from NEOPAX's
evolving transport profiles. In particular this comparison does not import
the optimized transport Er profile or its shear.

## Separate Python environments

Use a dedicated GKX environment and retain GKX's JAX/JAXLIB >= 0.10.1
requirements. The NEOPAX runs discussed here used JAX/JAXLIB 0.7.2. Installing GKX into
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

Create a separate environment for GKX and keep the package's declared
requirements intact. Use a recent CUDA 12 JAX build containing the upstream
non-VMM fallback and enable that fallback for the supplied vGPU. The
device check verifies that the selected runtime accepts the flag and can
initialize CUDA before any long calculation:

```bash
python3 -m venv "$HOME/gkx_env" &&
"$HOME/gkx_env/bin/python" -m pip install --upgrade pip &&
"$HOME/gkx_env/bin/python" -m pip install --upgrade \
  'jax[cuda12]' -e "$HOME/GKX" &&
"$HOME/gkx_env/bin/python" -m pip check &&
env JAX_PLATFORMS=cuda \
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_experimental_vmm_disabled=true" \
"$HOME/gkx_env/bin/python" -c \
  'import jax; print(jax.__version__, jax.devices()); assert jax.default_backend() == "gpu", "GKX requires a working GPU setup for this run"'
```

Use the matching accelerator extra if the machine requires a different
CUDA setup; see the [official JAX installation instructions](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu).
Creating the environment or listing devices does not run a simulation.
There is no need to install NEOPAX in `gkx_env`: the comparison controller
uses standard Python, NumPy, NetCDF4 and plotting, and starts GKX separately.

### Device initialization fails with a VMM error

The supplied follow-up log installed GKX successfully in `gkx_env` and
passed `pip check`, but JAX 0.10.2 then failed with:

```text
Device 0 does not support CUDA Virtual Memory Management (VMM).
VMM is required for device memory allocation in XLA.
```

This is a device/runtime compatibility failure, before a GKX simulation
starts. The older JAX in `jax_env` listing a CUDA device does not establish
that the newer runtime can use it. An [upstream XLA report](https://github.com/openxla/xla/issues/49252)
reproduces this on fractional vGPU configurations: 0.10.0 and earlier work
there, whereas 0.10.1 and newer reject the device before allocator selection.
The follow-up hardware diagnostic identified GRID A100X-10C, driver
535.288.01 and `Virtualization Mode: VGPU`, consistent with the reported
restriction. A small diagnostic is:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv
```

The issue's later maintainer response points to the merged [XLA non-VMM
fallback](https://github.com/openxla/xla/pull/49760), enabled with
`--xla_gpu_experimental_vmm_disabled=true`. The issue's original description
predates this fix. Use a build containing the fix; an older build may reject
the flag as unknown. Repeating the original JAX 0.10.2 installation is not
the intended workaround.

Keep GKX's existing requirements: the proposed requirement-lowering patch
was withdrawn at the user's request, along with its tests and GKX
documentation changes. No changes remain in GKX. The NEOPAX interpreter
fix below addresses the separate missing-NumPy failure. Apply the XLA flag
only to GKX processes; older NEOPAX JAX builds may not recognize it.

## Run on the simulation machine

Commit/push the script, its test, and this note locally; then pull the branch
on the remote machine. GKX must be available there, normally at `$HOME/GKX`.
The script supplies its `src` directory to the worker's PYTHONPATH.

```bash
cd /home/exouser/NEOPAX || exit 1
git pull --ff-only origin en/v0_new || exit 1
set -o pipefail
mkdir -p outputs

env JAX_PLATFORMS=cuda \
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_experimental_vmm_disabled=true" \
/home/exouser/gkx_env/bin/python \
  examples/optimization/run_gkx_initial_optimized_adiabatic.py \
  outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization \
  --gkx-root "$HOME/GKX" \
  --s 0.5 --LT 3.0 --Ln 0.0 --alpha 0.0 \
  --t-max 700 \
  2>&1 | tee outputs/gkx_initial_optimized_adiabatic_s0p5.log
```

The generated decks are ordinary GKX inputs. Once the device check above
passes, they can also be run directly from the GKX checkout without using
the NEOPAX controller or waiting for its interpreter fix:

```bash
cd "$HOME/GKX" || exit 1
neopax_gkx_case="$HOME/NEOPAX/outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization/gkx_adiabatic_comparison/s_0p5_LT_3_Ln_0_alpha_0"
for equilibrium in initial optimized; do
  env JAX_PLATFORMS=cuda \
    XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_experimental_vmm_disabled=true" \
    "$HOME/gkx_env/bin/python" -m gkx.cli "$neopax_gkx_case/$equilibrium/gkx.toml" || break
done
```

By default the worker uses the controller's interpreter, so both run in
`gkx_env` with this command. `--gkx-python "$HOME/gkx_env/bin/python"`
can also select the worker interpreter explicitly when invoking the
controller from another compatible environment.

The launchers preserve the virtual-environment Python path when starting
workers. They make it absolute without resolving its symlinks: resolving
`gkx_env/bin/python` to `/usr/bin/python3.12` bypasses the environment and
can cause `ModuleNotFoundError: No module named 'numpy'`. Both this
comparison and the existing optimized-only scan use the corrected behavior.

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

## Boundary comparison, until saturation

The wrapper accepts `0 < s <= 1`, including exactly `--s 1.0`. GKX receives
`geometry.torflux = 1.0`; the wrapper does not clip it to a nearby interior
surface. GKX's existing radial spline path evaluates the boundary, which
entails extrapolation of coefficients stored on the VMEC half mesh beyond
their last half-mesh point. No geometry solve is performed by this wrapper.

After pulling the updated NEOPAX script, repeat the comparison with:

```bash
cd "$HOME/GKX" || exit 1
env JAX_PLATFORMS=cuda \
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_experimental_vmm_disabled=true" \
"$HOME/gkx_env/bin/python" \
  "$HOME/NEOPAX/examples/optimization/run_gkx_initial_optimized_adiabatic.py" \
  "$HOME/NEOPAX/outputs/geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_optimization" \
  --gkx-root "$HOME/GKX" \
  --s 1.0 --LT 3.0 --Ln 0.0 --alpha 0.0 \
  --until-saturated --t-max 700
```

Both generated decks explicitly set `[time] run_to = "saturation"`.
GKX checks its heat-flux statistics and Wphi/Wg stationarity during the
run; 700 is the maximum time, and reaching that cap alone does not imply
saturation. The case uses a separate `s_1_LT_3_Ln_0_alpha_0` directory.
The completed `s_0p5_LT_3_Ln_0_alpha_0` inputs and outputs are retained.

To run directly through the GKX CLI, add `--prepare-only` above, then use
the generated `initial/gkx.toml` and `optimized/gkx.toml` as before with
`--until-saturated` on each CLI invocation. Add `--plot-only` to the
boundary comparison command to regenerate its plots without running GKX.

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
The manifest records both equilibrium hashes, the template hash and the
run-to policy; a different input, time horizon or policy requires a fresh `--out-dir` so saved
outputs from a different case cannot silently be used in the comparison.
An early endpoint whose GKX summary records `saturated: true` is also
recognized as complete for `--until-saturated`. A partial unsaturated
checkpoint is resumed; an unsaturated run at the time cap is complete but
retains its unsaturated verdict. Older manifests without a run-to field
remain readable as fixed-horizon cases.

Add `--prepare-only` to write both input decks and the manifest without
starting GKX. Add `--plot-only` to the same command to regenerate comparison
outputs from existing histories without starting either run. Change
`--average-fraction` to change the averaging window without rerunning GKX.

Heat-flux statistics use elapsed-time weighting, including an interpolated
window boundary, because adaptive output samples may not be equally spaced.
The plotted standard deviation measures temporal variability; it is not an
uncertainty estimate or a saturation/convergence test. Fluxes retain GKX's
normalization, rather than being converted to physical heat power.
Comparison JSON/CSV also record `gkx_saturated`, `gkx_saturation_mean` and
`gkx_saturation_sem` from each native GKX summary, and the plot legend
labels assessed saturated/unsaturated cases. These native statistics are
kept separate from the diagnostic last-fraction mean; no saturation
verdict is inferred for old fixed-horizon outputs without one.

Local validation uses synthetic decks, mock workers and synthetic flux
histories only. It checks direct s (not s squared), equilibrium selection,
matching numerical settings and requested physics, stale-output rejection,
time-weighted statistics, exact boundary decks, saturation-policy dispatch,
early-saturation reuse and labeled CSV/plot generation. No GKX, VMEC,
NTX or NEOPAX transport simulation is run locally.
