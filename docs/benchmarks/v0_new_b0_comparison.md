# v0_new reference-field tangent comparison

Prepared 2026-11-15 on `en/v0_new_b0_audit`, based on NEOPAX `en/v0_new`
at `117f79727c4002b6aff19e99c87fe349f671d9a2`. The original branch is
unchanged. The original `b0=jnp.max(jnp.abs(bmnc))` expression is retained;
the main-branch argmax/gather patch is deliberately absent.

Use JAX **0.7.2**, VMEX `en/local_test` at
`2fd82e7bccd951f612740d225c5c64b12d747689`, and NTX
`en/custom_rule_NEOPAX` at `af223d70c2ac151ce1c7b03637abc2fd2e9cf689`.
The script rejects other JAX versions before requesting a GPU.

The check reuses the main run's captured state, parameters, mask and thirteen
folded objective cotangents. It runs the old raw geometry response and old
payload JVP, recording the state/payload tangents and all field contributions.
There is no equilibrium solve, NTX coefficient solve, root selection or
transport run. The new main-only `ac_aux_f` parameter is ignored only when
empty; a nonempty unsupported parameter is rejected.

`surface_b0_tangents.csv` and `[compact-b0]` log lines report both geometry
columns at all seven scan radii. Compare the `b0` and coefficient-zero
tangents at `rho=0.5`. Coefficient zero is the unique positive maximum in the
supplied reference case. Main recorded `db0=0` despite nonzero coefficient
tangents there. Inspect the saved tangents if the old stack differs.

This is a comparison of **stacks at the captured numerical point**, not a
reproduction of a historical v0_new rollout or a new AD/FD comparison. Old
VMEX's response/geometry kernels differ, so its total gradients need not
equal main's or the archived old benchmark. A result here alone does not
prove a JAX-only regression. The b0 correction is not applied in this branch.

Six NumPy/source tests and actual-capture schema checks pass locally. No
local JAX, VMEX, NTX or physics computation was run. GPU execution remains
to be performed on the large machine.

## Isolated large-machine checkouts

After publishing this diagnostic branch, use a separate parent directory.
Old NEOPAX automatically discovers sibling VMEX checkouts, so this layout
prevents it from importing the current main dependency by accident.

```bash
neopax_old_root="$HOME/neopax_v0_new_b0_audit"
mkdir -p "$neopax_old_root"

git -C "$HOME/NEOPAX" fetch origin en/v0_new_b0_audit || exit 1
git -C "$HOME/NEOPAX" worktree add --detach "$neopax_old_root/NEOPAX" FETCH_HEAD || exit 1

git -C "$HOME/VMEX" fetch origin 2fd82e7bccd951f612740d225c5c64b12d747689 || exit 1
git -C "$HOME/VMEX" worktree add --detach "$neopax_old_root/VMEX" FETCH_HEAD || exit 1

git -C "$HOME/NTX" fetch origin af223d70c2ac151ce1c7b03637abc2fd2e9cf689 || exit 1
git -C "$HOME/NTX" worktree add --detach "$neopax_old_root/NTX" FETCH_HEAD || exit 1
```

If these directories already exist, inspect/reuse them rather than creating
another worktree at the same path. This setup changes neither the current
main checkouts nor their installed environment.

## JAX 0.7.2 environment

If an existing old GPU environment is available, set `NEOPAX_OLD_PYTHON` to
its Python executable and retain that environment's installed dependencies.
Otherwise create a separate environment after the checkouts above:

```bash
"$HOME/jax_env/bin/python" -m venv "$HOME/jax072_env" || exit 1
export NEOPAX_OLD_PYTHON="$HOME/jax072_env/bin/python"
"$NEOPAX_OLD_PYTHON" -m pip install --upgrade pip || exit 1
"$NEOPAX_OLD_PYTHON" -m pip install \
  "jax[cuda12]==0.7.2" "equinox==0.13.8" "lineax==0.1.0" \
  "diffrax==0.7.2" "optimistix==0.1.0" "interpax==0.3.16" \
  "orthax==0.2.11" "booz-xform-jax==0.5.1" "solvax==0.36.1" f90nml \
  -e "$neopax_old_root/NEOPAX" \
  -e "$neopax_old_root/VMEX" \
  -e "$neopax_old_root/NTX" || exit 1
"$NEOPAX_OLD_PYTHON" -m pip check || exit 1
```

These package pins come from the existing local JAX 0.7.2 comparison
environment. They are not asserted to reproduce every dependency version
in the historical v0_new run. Installed distribution metadata confirms that
the pinned Equinox, Lineax, Diffrax and SOLVAX accept JAX 0.7.2. This setup
has not been installed or GPU-tested afresh during this task. The current
`jax_env` is only used as the base Python executable for creating the new
venv; installations target `jax072_env`.

## Run with the JAX 0.7.2 interpreter

Set `NEOPAX_OLD_PYTHON` to the absolute path of the old JAX 0.7.2 Python.
Do not guess its location or replace the current JAX 0.10 environment. The
capture path below assumes the original archive is on this machine; copy
that existing capture if using another machine.

```bash
: "${NEOPAX_OLD_PYTHON:?Set this to the existing JAX 0.7.2 Python executable}"
neopax_old_root="$HOME/neopax_v0_new_b0_audit"
neopax_old_out="$neopax_old_root/results"
mkdir -p "$neopax_old_out"
cd "$neopax_old_root/NEOPAX" || exit 1
set -o pipefail

env -u LD_LIBRARY_PATH -u JAX_PLATFORM_NAME -u JAX_COMPILATION_CACHE_DIR \
  -u XLA_FLAGS -u XLA_PYTHON_CLIENT_ALLOCATOR \
  PYTHONPATH="$neopax_old_root/VMEX:$neopax_old_root/NTX/src:$neopax_old_root/NEOPAX" \
  JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true \
  JAX_ENABLE_COMPILATION_CACHE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  NTX_DISABLE_COMPILATION_CACHE=1 VMEX_COMPILATION_CACHE=disabled \
  /usr/bin/time -v "$NEOPAX_OLD_PYTHON" \
  examples/benchmarks/check_saved_compact_geometry_gradient.py \
  --capture "$HOME/NEOPAX/outputs/bootstrap_ad16_audit_6Wd5Gihp/snapshots/ad_compact_geometry_7wcaqkvs.npz" \
  --out-dir "$neopax_old_out" \
  2>&1 | tee "$neopax_old_out/run.log"

tar -czf "$neopax_old_root/v0_new_b0_replay.tar.gz" -C "$neopax_old_out" .
```

The log identifies the loaded geometry, VMEX, NTX geometry and Boozer API
source files and hashes. Keep those with the output arrays when comparing.
