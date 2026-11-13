"""Small scalar Kvaerno5 AD comparison; no NEOPAX/VMEC/NTX physics.

Compare Diffrax's adaptive attempt-loop AD with replay of its accepted times.
Both use Kvaerno5, the same strict nonlinear tolerances and a frozen controller
derivative. Finite differences use the baseline accepted schedule, not fresh
adaptive timesteps. This does not reproduce the historical NEOPAX cache bug.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import diffrax as dfx


def compare(*, rate=.2, final_time=10., first_dt=8., rtol=1e-6, atol=1e-9):
    # Work with the excess temperature; actual T is 20 + y, y(0)=60 deg C.
    term = dfx.ODETerm(lambda t, y, k: -k * y)
    solver = dfx.Kvaerno5(root_finder=dfx.VeryChord(rtol=1e-10, atol=1e-12))
    max_steps = 128
    initial = jnp.asarray(60., dtype=jnp.float64)
    rate_array = jnp.asarray(rate, dtype=jnp.float64)

    def adaptive(k, *, save_steps=False):
        return dfx.diffeqsolve(
            term, solver, t0=0., t1=final_time, dt0=first_dt, y0=initial, args=k,
            stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
            saveat=dfx.SaveAt(steps=True) if save_steps else dfx.SaveAt(t1=True),
            max_steps=max_steps,
        )

    print("[kvaerno5] baseline scalar adaptive solve; at most 128 attempts", flush=True)
    baseline = adaptive(rate_array, save_steps=True)
    jax.block_until_ready(baseline.ys)
    accepted_times = np.asarray(baseline.ts)
    accepted_times = accepted_times[np.isfinite(accepted_times)]
    grid = jnp.asarray(np.concatenate(([0.], accepted_times)))
    accepted_count = int(baseline.stats["num_accepted_steps"])
    rejected_count = int(baseline.stats["num_rejected_steps"])
    if rejected_count == 0:
        raise RuntimeError("This comparison needs rejected attempts; increase --first-dt.")
    if len(accepted_times) != accepted_count:
        raise RuntimeError("Saved accepted times do not match the baseline step count.")

    def accepted_replay(k):
        return dfx.diffeqsolve(
            term, solver, t0=0., t1=final_time, dt0=None, y0=initial, args=k,
            stepsize_controller=dfx.StepTo(ts=grid),
            adjoint=dfx.RecursiveCheckpointAdjoint(), saveat=dfx.SaveAt(t1=True),
            max_steps=max_steps,
        ).ys[0]

    print("[kvaerno5] differentiating adaptive attempt loop", flush=True)
    adaptive_value, adaptive_gradient = jax.jit(jax.value_and_grad(lambda k: adaptive(k).ys[0]))(rate_array)
    jax.block_until_ready(adaptive_gradient)
    print("[kvaerno5] differentiating fixed accepted-time replay", flush=True)
    replay_value, replay_gradient = jax.jit(jax.value_and_grad(accepted_replay))(rate_array)
    jax.block_until_ready(replay_gradient)
    print("[kvaerno5] centered FD on the same accepted-time schedule", flush=True)
    replay_value_fn = jax.jit(accepted_replay)
    fd_steps = [1e-4, 1e-5, 1e-6]
    fd = {}
    for step in fd_steps:
        plus, minus = replay_value_fn(rate_array + step), replay_value_fn(rate_array - step)
        fd[str(step)] = float((plus - minus) / (2 * step))
    av, ag, rv, rg = map(float, (adaptive_value, adaptive_gradient, replay_value, replay_gradient))
    report = dict(
        versions={name: importlib.metadata.version(name) for name in ("jax", "diffrax", "optimistix", "equinox", "lineax")},
        device=str(jax.devices()[0]),
        problem="T'=-k*(T-20), T(0)=80 deg C; differentiate final temperature with respect to k",
        settings=dict(k=rate, final_time=final_time, first_dt=first_dt, rtol=rtol, atol=atol,
                      nonlinear_rtol=1e-10, nonlinear_atol=1e-12, max_steps=max_steps),
        accepted_steps=accepted_count, rejected_steps=rejected_count,
        accepted_times=accepted_times.tolist(),
        adaptive_final_temperature=20.+av, replay_final_temperature=20.+rv,
        adaptive_attempt_ad=ag, accepted_replay_ad=rg, frozen_schedule_fd=fd,
        continuous_exact_temperature=20.+60.*math.exp(-rate*final_time),
        continuous_exact_gradient=-60.*final_time*math.exp(-rate*final_time),
        endpoint_absolute_difference=abs(av-rv), gradient_absolute_difference=abs(ag-rg),
        scope="Scalar Kvaerno5 only; not a reproduction of NEOPAX's old attempt-map cache behavior.",
    )
    np.testing.assert_allclose(av, rv, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(ag, rg, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(rg, fd["1e-05"], rtol=1e-6, atol=1e-8)
    report["checks_passed"] = True
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=.2)
    parser.add_argument("--final-time", type=float, default=10.)
    parser.add_argument("--first-dt", type=float, default=8.)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--output", type=Path, default=Path("outputs/adjoint_visualization/kvaerno5_rejected_steps.json"))
    args = parser.parse_args(argv)
    if not all(math.isfinite(v) and v > 0. for v in (args.rate, args.final_time, args.first_dt, args.rtol, args.atol)):
        parser.error("Require finite positive parameters and tolerances.")
    report = compare(rate=args.rate, final_time=args.final_time, first_dt=args.first_dt,
                     rtol=args.rtol, atol=args.atol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
