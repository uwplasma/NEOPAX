# NTSSfusion / NEOPAX wHe model audit

Audited on 2026-10-31 on branch `en/v0_new`. This is a source and saved-data
comparison. No VMEC, Boozer, NTX, transport, AD, FD, or GKX simulation was run.
No physics or configuration defaults were changed by this audit.

Follow-up implementation: the opt-in `[equations] match_THe = true` now ties
helium temperature and gradient to evolving deuterium. The default is `false`,
and the original wHe example files explicitly retain `false`, preserving their original
behavior. Keep helium's temperature toggle disabled. This option addresses
the temperature closure only. A second opt-in setting, `match_workHe = true`,
now reproduces the NTSS local neoclassical work source and assigns half of
helium's signed electric work to each of D and T. Its default is also `false`.
Neither switch changes the radiation coefficient. The original snapshots
audited here predate these options.

The separate comparison case is
`examples/Solve_Transport_Equations/Solve_Transport_equations_wHe_radau_match_THe.toml`.
At the user's request it now enables both `match_THe` and `match_workHe`, selects
`[transport_solver] Er_diffusion_mode = "ntssfusion"`, and
sets `[sources.parameters.bremsstrahlung_radiation] coefficient_mode = "ntssfusion"`
to use 0.334 rather than 0.316. Its parsed input differs from the original
Radau wHe TOML only in those two flags, the Er diffusion and convection
temperature modes, the radiation
coefficient mode, and the two
output directories. It retains the same NTSS initial/final
reference files, source list, grids, tolerances and
20-second endpoint. It writes the existing Er, alpha-power, species-density
and temperature comparison plots to
`outputs/transport_solver_wHe_radau_ntss_match_THe_workHe_brems_er_ntss_face_T`.
The current case also selects
`temperature_convection_temperature_mode = "face"`, sharing the model's
reconstructed face temperature in Gamma*T. The earlier `_er_ntss` run below
still used the default upwind temperature. The energy grid stays at four
points, as requested by the user; it is not being varied in this experiment.
The earlier temperature-only result remains in
`outputs/transport_solver_wHe_radau_ntss_match_THe`; that simulation had
`match_workHe = false`. The user's subsequent work-matched run at
`outputs/transport_solver_wHe_radau_ntss_match_THe_workHe` still used 0.316:
its terminal saved net source power is 309.9822 MW, compared with 305.4375 MW
in the temperature-only run. Neither saved result uses the newly selected
0.334 coefficient. The subsequent work+radiation result discussed below uses
0.334 but still predates Er diffusion matching. The user has subsequently
run the Er-matched case too; its measured effect is recorded below. The
agent inspected saved results and did not launch those simulations.

On the GPU machine, from the repository root:

```bash
PYTHONPATH="$HOME/VMEX:$HOME/NTX/src${PYTHONPATH:+:$PYTHONPATH}" \
"$HOME/jax_env/bin/python" -m NEOPAX \
  examples/Solve_Transport_Equations/Solve_Transport_equations_wHe_radau_match_THe.toml
```

The agent did not launch this full transport case. The implementation checks
use small synthetic arrays for state packing, outputs, boundary gradients,
the reverse preparation pullback and a checkpoint-transform directional
derivative; they do not establish the resulting 20-second helium profile.

## Matched-helium result and power-plot follow-up

The user subsequently ran the matched case and supplied its plots. Its saved
file is now available at
`outputs/transport_solver_wHe_radau_ntss_match_THe/transport_solution.h5`.
Only saved-data analysis and plot regeneration were performed for this follow-up.
The full transport simulation was not repeated during the investigation.

The saved matched temperatures satisfy `THe = TD` to `3.56e-15 keV`.
Axis helium density changed from `0.21015017` in the old saved run to
`0.25368471`, compared with NTSS `0.25402` (units `1e20 m^-3`).
At common radii, terminal helium relative differences are approximately
`-0.10%` at rho 0.1, `-0.72%` at 0.3, `+4.88%` at 0.5,
`+6.22%` at 0.7, and `+8.89%` at 0.9. Helium is still increasing over
the final saved interval, from 17.7778 to 20 s; at rho 0.7 the increase is
`0.0128733` in density-state units. NTSS's last recorded time is also 20 s.
These are finite-time comparisons, not established steady states.

A separate plotting defect was confirmed: the colored NEOPAX source curves
include negative bremsstrahlung, but the NTSS `power_sources_total` overlay
summed alpha and external heating without subtracting radiation. Both the
standalone reference importer and the transport plot's private importer had
this discrepancy. Their radiation reconstruction also used 0.316, whereas the
active NTSSfusion source coefficient is 0.334.

Both reference loaders now subtract radiation, reconstruct missing NTSS
radiation with 0.334, and the source plot is titled **Net Power Sources**.
This is a postprocessing correction; it changes no transport source model,
solver setting, saved HDF5 profile, or `match_THe=false` evolution.

| Quantity at 20 s | Original saved NEOPAX | Matched THe NEOPAX | NTSS reference |
|---|---:|---:|---:|
| Integrated alpha heating [MW] | 372.1770 | 354.0790 | 349.0630 |
| Integrated net source power [MW] | 320.2011 | 305.4375 | 298.4862 |
| Net source power relative difference | +7.27% | +2.33% | reference |

The NEOPAX net integrals reproduce its saved `P_total_mw`. NTSS net power is
reconstructed as alpha heating minus
`0.334 * Zeff * ne^2 * sqrt(Te) / 62.422`, with no nonzero external heating
in this case. Integrals here use each file's native radius and volume-derivative
arrays, so the percentages are comparisons of saved numerical integrals,
not a proof that quadrature/grid effects have been removed.

The temperature-only matched NEOPAX run still uses its configured 0.316 radiation coefficient.
Changing *only the radiation evaluation at the same saved terminal state* to
0.334 would add `2.77072 MW` of radiation and give `302.6668 MW` net power.
This is an algebraic snapshot calculation, not a prediction of a new evolved
temperature or helium solution. The electric-work conventions identified
below also remained different in that saved run. Their contributions to the residual profile
errors have not been separated by new transport simulations.

Four small reference-accounting/plot-overlay regression checks passed. A
postprocessing-only command regenerates the state/source plots while avoiding
any geometry runtime, NTX database, or flux-model evaluation:

```bash
python examples/benchmarks/replot_saved_transport.py \
  --config examples/Solve_Transport_Equations/Solve_Transport_equations_wHe_radau_match_THe.toml \
  --solution outputs/transport_solver_wHe_radau_ntss_match_THe/transport_solution.h5 \
  --out-dir outputs/transport_solver_wHe_radau_ntss_match_THe/replotted
```

This command uses the source settings in the current TOML. To reproduce the
historical temperature-only plots below, use a copy with
`coefficient_mode = "neopax"`; the current `ntssfusion` setting would instead
recalculate radiation at the same saved profiles with 0.334.

The corrected net-power figure is
`outputs/transport_solver_wHe_radau_ntss_match_THe/replotted/transport_power_sources_total.png`.

## Follow-up: temperature, work and radiation matched

The user ran the updated TOML locally and produced
`outputs/transport_solver_wHe_radau_ntss_match_THe_workHe_brems/transport_solution.h5`.
Its log reports completion at 20 seconds, 738 solver steps and no terminal failure.
All saved density, pressure, temperature and Er profiles are finite. Saved
`THe - TD` stays below `3.56e-15 keV`. The agent inspected saved arrays and
evaluated algebraic sources on 51 cells; no geometry, NTX, flux-model or
transport solve was run during this follow-up.

| Terminal power [MW] | Matched THe | THe + work | THe + work + radiation | NTSS |
|---|---:|---:|---:|---:|
| Alpha heating | 354.0790 | 358.8207 | 357.9624 | 349.0630 |
| Net source | 305.4375 | 309.9822 | 306.3448 | 298.4861 |

These are native-grid trapezoidal integrals, consistent with NEOPAX's saved
`P_total_mw`. The latest net excess is 7.8586 MW (+2.633%): alpha heating is
8.8994 MW higher and radiation is 1.0408 MW higher. Replacing only the latest
saved tritium temperature in the fusion formula with interpolated NTSS TT,
while retaining NEOPAX's densities and radiation, gives 297.2040 MW net.
This is a saved-state decomposition, not an evolved prediction or an assertion
about which transport difference produced the hotter core.

| rho | Latest nHe difference from NTSS | Latest TD difference from NTSS |
|---:|---:|---:|
| 0.1 | +0.565% | +3.836% |
| 0.3 | -0.170% | +2.704% |
| 0.5 | +5.525% | +0.548% |
| 0.7 | +6.915% | +0.176% |
| 0.9 | +9.825% | -4.157% |

At rho 0.7 nHe still increases by 0.0129942 over the last saved interval,
17.7778 to 20 s. This is about 5.50% of its terminal value. Both codes are
being compared at 20 s; this observation does not by itself explain their
different transient solutions. The interpolated terminal downcrossings are
0.51695016 (NEOPAX) and 0.51727687 (NTSS), without a grid-convergence claim.

Same-state source evaluation agrees with the NTSS formulas at approximately
`1e-15` relative for D-T reaction rate, helium production, alpha energy,
electron deposition fraction and radiation. The active e-D/e-T/D-T exchange
sum is zero to rounding. Comparing exchange formulas with the actual
slightly different electron masses gives a relative maximum difference
`4.13e-5`; it is not a missing helium exchange term (NTSS disables those pairs).
No further missing fusion/helium source was identified in these checks.

### Confirmed remaining Er equation difference

In the saved runs above, NTSS `CPlasma_Fluxes.cpp:349` sets `GEr = -dEr + Erk/rk` and at `:460`
uses `FE = DEcoeff * GEr`. These fluxes enter the conservative divergence
in `CPlasma_RHS.cpp:236`. NEOPAX's `build_electric_field_equation` at
`NEOPAX/_transport_equations.py:1649` returns only `-er_cell_var.face_grad()`.
Its `ntss_like_midpoint` option changes the ambipolar relaxation prefactor,
not the diffusion flux. Thus the active matched TOML does not yet reproduce
the NTSS Er equation: the `+Er/r` contribution to the diffusive flux is absent
from its default `"neopax"` mode. The new opt-in mode below supplies it.
With volume derivative Vp, the extra NTSS RHS term is
`-DEr / Vp * d(Vp * Er/r)/dr`, times the configured Er relaxation factor.

This difference survives grid refinement. For example, away from the axis,
if `Er = C*r` and `Vp` is proportional to `r`, NTSS's diffusive flux is zero,
whereas NEOPAX's current scalar diffusion gives `DEr*C/r`. It is a difference
in the chosen regularization equation, not a missing fusion power source.
Its contribution to the terminal discrepancy has not been isolated by a
controlled transport run. No Er operator was changed in this audit.

A five-part verification before implementing the opt-in mode confirmed: (1) NTSS stores physical Er
and differentiates it without an Er/r change of variables; (2) its selected
build, source and divergence retain the Er/r flux term; (3) NEOPAX's actual
factory and `CellVariable.face_grad` contain no compensating term; (4) the
current TOML and complete Er RHS retain the scalar diffusion; and (5) an
eight-cell manufactured-field check distinguishes the operators. The latter
uses production boundary normalization, zero charge flux and Vp proportional
to r, comparing interior cells before multiplying by DEr and Er_relax.
For Er=2r, NEOPAX gives 2/r and the NTSS flux expression gives zero; for
Er=r^2 they give 4 and 2 respectively. All five checks passed. Results are
saved in `outputs/whe_ntss_audit_20261031/er_diffusion_five_checks.json`.
This verifies the current source and NEOPAX operator, not the provenance of
the executable that generated the historical NTSS reference. No full NTSS
or NEOPAX simulation was run, and the terminal discrepancy's cause remains
unisolated.

There is also a smaller prefactor difference: NTSS uses central D+T density
(`u[1]`) with midpoint magnetic geometry; NEOPAX uses midpoint D+T density.
The frozen D/T profiles are nearly flat in the inner region of this case.

### Opt-in Er diffusion implementation

`[transport_solver] Er_diffusion_mode = "ntssfusion"` now adds `Er/r` to
the existing reconstructed face-gradient flux before its conservative
divergence. `"neopax"` remains the default and retains the previous
arithmetic. Unknown mode names are rejected during configuration normalization
and by the direct equation factory. The choice is independent of the
permittivity, work and helium-temperature settings.

The correction uses the same reconstructed face field and boundary values as
the gradient. An explicit outer-face Er node is used when supplied. At an
axis face the regular flux is zero, with a safe denominator in both the
forward calculation and its reverse pullback. `DEr = 0` still bypasses the
diffusive flux, and the floating outer-edge ambipolar RHS is retained.
Forward, checkpoint/replay and reverse assembly share the selected flux builder.
This change does not address the midpoint-versus-central density prefactor or
the separate convection/energy-integration differences described here.

The focused regression checks passed 73 tests and 16 subtests across the
operator, boundary, solver-configuration, helium and floor checks. They include
bitwise default-operator comparisons, manufactured fields, conservative volume
balance, zero-DEr bypass, signed-field/outer-node VJP versus FD, and the inverse
square geometry-scaling derivative. All use small synthetic arrays. Two stale
helper-test issues were corrected (a missing JAX import and an old expectation
that face Dirichlet conditions overwrite cell-centred state values); the
production state-boundary behavior was not changed.

The matched TOML now enables this mode and writes to the `_er_ntss` output
folder listed above, preserving all three previously saved results. No full
transport solve was launched as part of implementing this option.

### Measured effect of matching Er diffusion

The user's new run completed to 20 s, with 784 reported solver steps,
`done=True`, `failed=False`, and finite saved profiles. Its initial density,
pressure, temperature and Er arrays are bitwise identical to the previous
work+radiation-matched run. Grids and saved times are also identical.
The startup logs have identical ambipolar-source ranges but different
diffusive flux and diffusion RHS ranges, confirming that the new operator
was active rather than silently ignored.

| At 20 s | Previous diffusion | NTSS diffusion | NTSS reference |
|---|---:|---:|---:|
| Alpha heating [MW] | 357.9624 | 357.9807 | 349.0630 |
| Net source [MW] | 306.3448 | 306.3600 | 298.4861 |
| Interpolated downcrossing rho | 0.51695016 | 0.51695554 | 0.51727687 |

The net change is only +0.015198 MW (+0.00496%), leaving a +2.638%
difference from the reference. Maximum terminal changes are 0.01382 keV
in temperature and 0.14235 kV/m in Er. Helium relative errors are now
+0.735%, -0.149%, +5.541%, +6.916% and +9.829% at rho
0.1, 0.3, 0.5, 0.7 and 0.9 respectively, essentially the previous pattern.
Helium temperature still follows deuterium to 3.56e-15 keV.

Thus the source-level Er diffusion mismatch is real, but this controlled
comparison shows it is not the dominant explanation for the remaining
helium/power discrepancy in this case. It does not identify the relative
contributions of energy quadrature, heat-convection reconstruction or other
transport differences. Those require matched-state flux comparisons.

New saved-array artifacts are in
`outputs/whe_ntss_audit_20261031/er_diffusion_comparison/`:
`comparison.png`, `summary.json`, and `common_radius.csv`.

### Opt-in shared face temperature for convection

`[transport_solver] temperature_convection_temperature_mode = "face"` now
uses the temperature from `build_face_transport_state` for every enabled
Gamma*T convection term, with the same reconstruction, floors and boundary
conditions as the Q/Gamma model evaluation. The default `"upwind"` preserves
the previous sign-dependent temperature selection. The choice is separate
from `temperature_convection_reconstruction`, which still determines how the
particle fluxes are obtained on faces.

This change reconstructs only the algebraic face state; it adds no flux-model
call, database query, runtime geometry construction, or extra transport solve.
Diagnostic components and the pressure RHS use the same selection. The
shared-flux reverse path retains both Gamma-face cotangents and the direct
pressure/density dependence of the reconstructed temperature. Equation geometry
rebuilding retains the same option. It does not replace NEOPAX's face-state
construction with midpoint pressure divided by midpoint density; the option
specifically ensures that Q, Gamma and the convection multiplier use a common
face temperature within NEOPAX.

The focused run passed 89 tests and 16 subtests, including bitwise default
upwind comparisons, both particle-flow signs, all three convection channels,
the same face state observed by a stub transport model with active floors and
thermal boundary conditions, direct pressure/flux VJP against FD, shared-flux
and direct-profile reverse assembly, and nonuniform-grid geometry sensitivity.
No transport model, geometry runtime or simulation was evaluated by these
checks beyond the small algebraic stub used to inspect the face state.

The current matched TOML enables this option and writes to `_er_ntss_face_T`,
preserving the preceding `_er_ntss` result for comparison. No full simulation
was launched during implementation.

### Measured effect of shared face temperature

The user subsequently ran the `_er_ntss_face_T` case. It completed to 20 s,
with 764 reported solver steps, `done=True`, `failed=False`, and finite saved
profiles. Its initial density, pressure, temperature and Er arrays are
bitwise identical to the upwind `_er_ntss` result; grids and save times also
match. The startup conductive Q-face window and Q-divergence ranges are
identical, while the neo-convective divergence changes, confirming that the
new temperature selection was active.

| At 20 s | Upwind temperature | Shared face temperature | NTSS reference |
|---|---:|---:|---:|
| Alpha heating [MW] | 357.9807 | 359.7835 | 349.0630 |
| Net source [MW] | 306.3600 | 308.0936 | 298.4861 |
| Interpolated downcrossing rho | 0.51695554 | 0.51695812 | 0.51727687 |

Net source increases by 1.73362 MW (+0.5659%), changing its relative excess
over NTSS from 2.638% to 3.219%. Maximum terminal changes are 0.05862 keV in
temperature and 0.12869 kV/m in Er. Helium errors at rho 0.1/0.3/0.5/0.7/0.9
are +0.973%/+0.002%/+5.537%/+6.864%/+9.956%, respectively. The outer-helium
pattern therefore persists, while core D temperature at rho 0.1 is now
4.130% above NTSS (previously 3.834%).

This controlled comparison does not improve overall agreement and does not
identify the main remaining cause. Sharing the face temperature aligns this
part of NEOPAX's flux construction; it is not evidence that its reconstructed
face states, gradients and neoclassical transport matrices match NTSS
pointwise. Energy resolution remains fixed at four as requested. A useful
next check is Gamma and Q at identical prescribed profiles and gradients,
separating transport coefficients from evolving-state/reconstruction effects.

Saved-data artifacts are under
`outputs/whe_ntss_audit_20261031/face_temperature_comparison/`:
`comparison.png`, `summary.json`, and `common_radius.csv`. The agent only
read saved files and regenerated comparison artifacts; it did not repeat the
simulation or evaluate the flux model.

### Other numerical comparisons and scope

- The TOML uses `energy_grid.n_x = 4`. NEOPAX's active database flux model
  passes those four Gauss-Laguerre nodes into the Lij integrals. The active
  NTSS Taguchi transport path uses a truncated order-64 rule (30 retained
  base nodes), with additional resonance resolution; see
  `Neo-DKES/taguchi/tc_dkes_for_tag.f90:153` and `:210`. It is not the unused
  CNeo C++ integration path. Their same-state flux difference is unmeasured.
  The user considers four energy points sufficient, so energy resolution is
  held fixed and is not the next proposed experiment.
- For neo convection, NEOPAX uses `Gamma_neo * T_upwind` even with
  `temperature_convection_reconstruction = "closure_face_flux"`
  (`_transport_equations.py:908`). NTSS uses `Gamma_neo * T` calculated from
  midpoint pressure and density (`CPlasma_Fluxes.cpp:240`, `:338`). The common
  multiplier is one in both. In the latest saved profiles, the upstream
  temperature differs from the arithmetic midpoint by up to 3.6% for D
  inside rho 0.8, and 16.1% near the edge; this is not an estimate of the
  resulting total heat-flux error. This describes the saved runs before the
  opt-in face-temperature change above.
- Density floors still differ (NEOPAX 1e-3, NTSS CorrectData 1e-6), and the
  radial boundary nodes/control volumes differ. Do not change floor AD
  treatment just to tune the comparison.

The active NTSS Fortran collision path uses the same common
`32.2 + 1.15*log10(Te_eV^2/ne_m3)` logarithm as NEOPAX's default model.
The unused CNeo C++ pairwise logarithm is not evidence to change the
neoclassical collisionality mode. The heat-exchange source logarithm is a
separate setting.

Geometry volume derivatives agree within 0.021% on common radii. The apparent
volume difference from centre-only trapezoids is an integration-domain effect:
NEOPAX's finite-volume sum gives 331.0157 m3, compared with NTSS 331.0055 m3,
while the centre-only trapezoid covers 324.4389 m3. Reintegrating the NTSS net
source on the NEOPAX grid gives 298.4201 MW, just 0.0661 MW below its native
integral; this alone cannot explain the 7.8586-MW net excess.

Saved-data artifacts are under `outputs/whe_ntss_audit_20261031/`:
`comparison.png`, `summary.json` and `common_radius.csv`. The next controlled
check is a matched-state Gamma/Q comparison at the existing four energy
points. The controlled Er diffusion and face-temperature reruns above have
not resolved the benchmark discrepancy.

## Cases and provenance

NEOPAX configurations inspected:

- `examples/Solve_Transport_Equations/Solve_Transport_equations_wHe_radau.toml`
- `examples/Solve_Transport_Equations/Solve_Transport_equations_wHe_theta.toml`
- `examples/Solve_Transport_Equations/Solve_Transport_equations_wHe_radau_ntx_exact_lagged_runtime.toml`

These use four species e/D/T/He, frozen D/T densities, an evolving helium
density, evolving e/D/T pressures, and a disabled helium temperature equation.
Their source choices and initial analytical profiles agree with each other.
They do not all use the same neoclassical database/runtime.

The NTSS reference copied into
`examples/inputs/NTSS_Test_Final_flux_scaled_wHe.h5` is byte-for-byte identical
to the file in the sibling repository:

```
NTSS-software-master/NTSSfusion/bin/
  QI_npf2_newNT_opt_NTSCAN_Test_MONKES_S_2/198m3/
  QI_OPT_nonscaled_newNT_N1_T1_NEW_3/
  NTSS_Test_Final_flux_scaled_wHe.h5
```

SHA256: `8192b51ab1215d20908052e08b9622167d54f8b8f112fee3392c5bb9938df7df`.
The corresponding input is `Test_Final_flux_scaled_wHe_ntss.inp`; its terminal
text output is `Test_Final_flux_scaled_wHe` in the same directory.

Saved NEOPAX data quantified below:
`docs/benchmark/figures/transport_solver_wHe_radau_ntss/transport_solution.h5`.
SHA256: `07e7e4856775d4240ef37780f2f76471332894216b26e58e8bc460d09011ee7b`.
Its final time is 20 s. The saved Radau and theta snapshots in the adjacent
wHe figure directories exhibit the same frozen-helium-temperature behavior.
The pasted images have slightly different output times, so these numerical
values describe the saved repository snapshots, not an identified remote run.

## Confirmed mismatch 1: helium temperature closure

NTSSfusion explicitly sets both:

```
THe = TD;
dlnTHe = dlnTD;
```

See `NTSSfusion/CPlasma/CNeo.cpp:60` and `:70`, and
`NTSSfusion/CPlasma/CPlasma_Fluxes.cpp:277` and `:287`. These are live values
calculated from the current deuterium pressure and density. NTSS has no
independent helium pressure equation in this case.

NEOPAX instead has:

```toml
[equations]
toggle_temperature = [true, true, true, false]
```

`NEOPAX/_orchestrator.py:1171` constructs `fixed_temperature_profile` from the
initial configured profiles. `project_fixed_temperature_species` in
`NEOPAX/_transport_equations.py:195` projects disabled rows to
`pressure = safe_density * T_fixed`. The working-state preparation at `:3006`
applies this before flux and source evaluation. Consequently the helium
closure uses its initial temperature and its initial temperature gradient,
rather than following deuterium as NTSS does.

In the saved NEOPAX run, helium temperature changes by at most
`1.78e-15 keV` over the saved time history. At the first radial cell, the final
deuterium temperature is `21.14761517 keV`, but helium remains at
`12.45884948 keV`. The NTSS axis deuterium/helium temperature is
`19.941 keV`.

This changes helium thermal speed, collisionality, the electric-field drive
proportional to `Z_He Er / T_He`, and the temperature-gradient drive. It can
change the *shape* of helium particle transport even with a correct helium
birth source. It is the clearest helium-specific model mismatch found here.
Its contribution to the final density error has not been isolated by a run.

Simply enabling an independent helium temperature equation would not reproduce
the NTSS closure. An equivalent implementation must tie helium temperature and
its reconstructed gradients to the evolving deuterium temperature, including
face flux evaluations. The existing exclusion of helium heat-exchange pairs
should remain for this particular NTSS comparison.

## Confirmed mismatch 2: electric work in the energy equations

In `NTSSfusion/CPlasma/CPlasma_Fluxes.cpp:490`, the source uses only
neoclassical particle fluxes and adds:

```
GDS += ZHe / ZD / 2 * GHe;
GTS += ZHe / ZT / 2 * GHe;
```

For D/T charges of one and helium charge of two, the corresponding work terms
before the pressure-equation factor `2/3` are:

```
W_e = -Er * Gamma_e_neo
W_D =  Er * (Gamma_D_neo + Gamma_He_neo)
W_T =  Er * (Gamma_T_neo + Gamma_He_neo)
```

The comment in NTSS explains that these extra terms retain the helium
contribution in the evolved-species power balance. Summing the three gives
`Er * (-Gamma_e + Gamma_D + Gamma_T + 2 Gamma_He)`.

NEOPAX `TemperatureEquation._work_rhs` at
`NEOPAX/_transport_equations.py:694` uses `Z_a * Er * Gamma_a` for each species.
There is no helium-to-D/T redistribution, and the helium row is subsequently
masked out at `:902`. The flux key is total `Gamma`, which includes turbulent
particle flux by default, whereas the active NTSS source uses neoclassical
flux only. The total-flux assembly is in
`NEOPAX/_transport_flux_models.py:1633`.

Therefore these energy equations differ even after matching helium temperature.
This is an established difference in the models; its numerical importance in
this case has not been measured. It is not evidence for an incorrect D-T
particle source.

The new `[equations] match_workHe = true` implements the NTSS source at centres:
only `Gamma_neo` is used, half of `ZHe Er Gamma_He` goes to each of D and T,
and the helium work row becomes zero. It retains the existing `2/3` pressure
factor and does not change the particle fluxes or the ambipolarity equation.
`match_workHe = false` or absent preserves the original work calculation,
including the configured centre/face reconstruction. This switch is independent
of `match_THe`; helium's independent temperature equation must remain disabled.
The small work/temperature/floor regression run passed 38 tests and 16
subtests, including bitwise default-work comparisons, signed NTSS work values,
the ambipolar work sum, the pressure factor, and the shared-flux AD pullback.
No VMEC, NTX, or transport solve was used for those checks.

## Confirmed mismatch 3: bremsstrahlung coefficient

Both models use the ion-only effective charge
`Zeff = (nD + nT + 4 nHe) / ne + DeltaZeff`, with `DeltaZeff = 0` in this case.
However:

| Model | Coefficient in `P_brems = C Zeff ne^2 sqrt(Te)` |
|---|---:|
| NTSSfusion active first return | 0.334 |
| NEOPAX wHe `coefficient_mode = "neopax"` | 0.316 |

See `NTSSfusion/CPlasma/CPlasma_Fluxes.cpp:702` and
`NEOPAX/_sources.py:293`. The second NTSS return with coefficient 0.316 is
unreachable. At an identical plasma state, NEOPAX radiates 5.389% less in this
term. The existing `coefficient_mode = "ntssfusion"` selects 0.334 in NEOPAX;
no new radiation formula is necessary.

The difference feeds back through temperature, fusion production, and the
anomalous transport coefficient, whose total-power input includes radiation.
It does not imply a 5.389% error in the total power or helium source.

## Source terms and choices that agree

- **D-T reactivity:** same Putvinskiy fit evaluated at tritium temperature
  (`NEOPAX/_sources.py:19`; NTSS `CPlasma_Fluxes.cpp:768`).
- **Helium production:** `SHe = 1e20 <sigma v> nD nT` in density-state units of
  `1e20 m^-3 s^-1`, with one helium per reaction in both codes. NEOPAX adds
  equal D/T fusion sinks, but both D/T density equations are frozen here.
- **Alpha energy:** `Palpha = 3520 SHe` in the common energy-source units, or
  `Palpha_MW_m3 = 3520 SHe / 62.422`.
- **Alpha deposition:** same electron heating fraction; the remaining heating
  is split equally between D and T. Neither case deposits alpha power in an
  evolved helium pressure equation.
- **Heat exchange:** NEOPAX uses `exclude_frozen` with `coulomb_log_mode =
  "ntssfusion"`, so its active pairs are e-D, e-T and D-T. NTSS also excludes
  the helium exchange pairs in this source implementation. The hot-core
  pair formulas agree; arbitrary-state piecewise branch equivalence was not
  exhaustively tested.
- **Anomalous coefficient:** both specify particle coefficient `6.5e-5` and
  electron/ion heat coefficient `0.0065`, multiplying `P_MW^0.75 / ne`.
  Anomalous heat convection is disabled and neoclassical heat convection is
  enabled in both. Different source/temperature solutions can nevertheless
  produce different numerical transport coefficients.
- **External heating/fueling:** no configured nonzero NBI, ECRH, or particle
  fueling in the NTSS input; NEOPAX has no such active source in the wHe list.
- **Macroscopic density/temperature boundary conditions:** zero helium
  boundary density, frozen D/T boundary densities of 0.195, and thermal decay
  length 0.05 m. NTSS uses boundary nodes and NEOPAX uses finite-volume faces,
  so equality of the physical boundary prescription does not imply identical
  discrete stencils. NEOPAX's Robin `value = 1` is not an imposed edge
  temperature in the profile-aware decay-length closure.

As a saved-data check, evaluating the common alpha formula on NEOPAX's saved
densities and tritium temperature reproduces its saved alpha power with maximum
relative discrepancy `1.78e-15`. Evaluating it on the rounded NTSS text output
agrees within `1.09e-4` relative where alpha power exceeds `1e-6 MW/m^3`,
consistent with the limited printed precision. There is no evidence here for
a reaction-rate, helium-yield, or alpha-energy normalization mismatch.

## Saved terminal profiles at common radii

NTSS radius is converted from meters to `rho = r/r_edge`. Values below use
linear interpolation onto the same requested radius, rather than comparing
node and cell indices directly. Density units are `1e20 m^-3`; temperatures
are keV.

| rho | NEOPAX nHe | NTSS nHe | NEOPAX THe | NEOPAX TD | NTSS THe = TD |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.211318 | 0.252210 | 12.339196 | 20.570369 | 19.416000 |
| 0.3 | 0.187157 | 0.215442 | 11.381964 | 16.666444 | 15.910198 |
| 0.5 | 0.239634 | 0.241719 | 9.467500 | 10.198244 | 9.903605 |
| 0.7 | 0.314791 | 0.221014 | 6.593964 | 8.760589 | 8.600238 |
| 0.9 | 0.196197 | 0.125382 | 2.763196 | 5.231580 | 5.419525 |

At rho 0.1/0.3/0.5/0.7, the saved NEOPAX alpha power is approximately
8.0%/7.9%/7.8%/5.2% larger than the NTSS result. These source differences
reflect different evolved plasma states even though the source formula is
the same. They alone do not explain why helium is deficient centrally but
excessive outside the transition.

## Remaining limits and next comparison

The current inputs also use a different density-floor treatment:
NEOPAX configures `density_floor = 1e-3`, whereas NTSS's `CorrectData` contains
a `1e-6` density clamp (`CPlasma_RHS.cpp:130`). NTSS's input token
`nTminValue = 0.001` is bound but no use beyond that binding was found in this
source tree. Floor behavior matters near zero initial/edge helium density;
it is not established as the explanation for the terminal profile shape.
Do not undo the previous AD floor fixes as part of matching the temperature
closure.

No claim is made that the grids, numerical tolerances, monoenergetic tables,
interpolation, Er boundary implementation, or every transport coefficient have
been matched pointwise. Those are additional numerical comparisons, not
reasons to ignore the confirmed physical-model differences above.

The first controlled correction should make helium temperature follow the
evolving deuterium temperature and gradient while retaining the disabled
helium pressure equation. Then compare the electric-work convention and use
the existing NTSSfusion bremsstrahlung option. Isolating those changes avoids
attributing their effects to timestep tolerances or radial resolution. A
matched-state helium particle-flux comparison can test the temperature-closure
effect before committing to a complete 20-second transport rerun.
