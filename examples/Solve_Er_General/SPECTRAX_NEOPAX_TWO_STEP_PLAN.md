# SPECTRAX-NEOPAX Two-Step Plan

This note is the handoff for the staged NEOPAX <-> SPECTRAX-GK workflow.

## Step 1

Build and stabilize the external workflow:

1. Run a NEOPAX transport case and write `transport_solution.h5`.
2. Read the final NEOPAX profiles `n_s(rho), T_s(rho), Er(rho)`.
3. Select several radial positions.
4. Convert each radius into one local SPECTRAX-GK nonlinear runtime case.
5. Run those cases on CPUs or GPUs.
6. Collect the resulting heat / particle fluxes into one HDF5 summary for later use.

### Current status

Implemented files:

- [neopax_spectrax_flux_bridge.py](/d:/PostDocsProxima/Github_5/NEOPAX/examples/Solve_Er_General/neopax_spectrax_flux_bridge.py)
- [run_spectrax_flux_scan_from_transport.py](/d:/PostDocsProxima/Github_5/NEOPAX/examples/Solve_Er_General/run_spectrax_flux_scan_from_transport.py)

Current behavior:

- reads NEOPAX `transport_solution.h5`
- supports adiabatic or kinetic electrons
- writes one SPECTRAX local case per selected radius
- supports process-level spreading across CPU workers or round-robin GPU ids
- collects final SPECTRAX fluxes into a single HDF5 file

Current physics / implementation assumptions to revisit:

- `torflux = rho^2`
- local gradients are estimated as `-d ln X / d rho`
- adiabatic-electron runs do not produce electron kinetic fluxes
- scheduler is process-level only, not explicit CPU affinity binding
- no direct reinjection into NEOPAX yet

### Practical requirements

To use Step 1 from a NEOPAX transport TOML:

1. set `transport_write_hdf5 = true`
2. keep `transport_output_dir` pointing to a stable output directory
3. run the NEOPAX transport solve first
4. then run the SPECTRAX wrapper on the same TOML

Example:

```powershell
python NEOPAX\examples\Solve_Er_General\run_spectrax_flux_scan_from_transport.py ^
  --neopax-config NEOPAX\examples\Solve_Er_General\transport_pressure_Er_debug_radau_temp_Er.toml ^
  --electron-model adiabatic ^
  --num-radii 5 ^
  --backend gpu ^
  --gpu-ids 0,1 ^
  --max-parallel 2
```

## Step 2

Embed SPECTRAX-GK as an internal NEOPAX transport/turbulence workflow.

Target architecture:

1. Add a NEOPAX turbulence / transport-flux model that calls the SPECTRAX local runner.
2. Build the local runtime input directly from the evolving `TransportState`.
3. Cache geometry generation and any reusable runtime metadata.
4. Return turbulent particle / heat fluxes back to the NEOPAX transport equations.
5. Add batching and scheduling policies so multiple radii can be launched efficiently inside the solve.

### Likely work items

- define a clean NEOPAX-side turbulence model API for external GK calls
- formalize units and normalization mapping between NEOPAX and SPECTRAX
- decide whether the SPECTRAX call is:
  - synchronous per radius
  - batched per time slice
  - database / surrogate backed
- add restart / cache logic so repeated radii do not regenerate identical geometry
- make the scheduler reusable from inside NEOPAX, not just from the CLI

### Recommended next-session order

1. tighten Step 1 normalization and geometry mapping
2. decide the exact turbulent flux contract expected by NEOPAX
3. move bridge logic into a callable NEOPAX-side model object
4. only then start coupling it into the transport solve loop
