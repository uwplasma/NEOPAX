"""Plot comparison for Theta Newton solver with and without AP preconditioner.

This script runs a stiff toy problem and compares:
- state trajectories (n, T, Er)
- Er absolute error versus a RADAU reference
- accepted step-size history

Usage:
    python examples/plot_theta_ap_comparison.py
"""

from __future__ import annotations

import time

import jax.numpy as jnp

from NEOPAX._transport_solvers import RADAUSolver, ThetaNewtonSolver


class SharpTransitionErToyModel:
    """Fast-slow toy model that mimics sharp electric-field transitions."""

    def __init__(
        self,
        a=0.6,
        b=0.9,
        c=0.4,
        d=0.35,
        n_star=1.0,
        t_star=1.0,
        epsilon=0.01,
        lam=1.1,
        s=1.0,
        k=12.0,
        eta=0.85,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.n_star = n_star
        self.t_star = t_star
        self.epsilon = epsilon
        self.lam = lam
        self.s = s
        self.k = k
        self.eta = eta

    def __call__(self, t, y):
        del t
        n, temperature, electric_field = y
        dn = -self.a * (n - self.n_star) + self.b * electric_field
        dtemperature = -self.c * (temperature - self.t_star) - self.d * electric_field
        drive = self.s * jnp.tanh(self.k * (n - self.eta * temperature))
        delectric_field = (-(electric_field**3 - self.lam * electric_field) + drive) / self.epsilon
        return jnp.array([dn, dtemperature, delectric_field])


def ap_diag_for_toy(problem: SharpTransitionErToyModel):
    """Diagonal AP-style stabilization focusing on the stiff Er component."""

    def _diag(t, y):
        del t
        er = y[2]
        # de/dt = g(e)/epsilon -> local stiff Jacobian scale ~ |g'(e)|/epsilon
        gprime = jnp.abs((-3.0 * er * er + problem.lam) / problem.epsilon)
        return jnp.array([0.0, 0.0, gprime])

    return _diag


def _accepted_series(theta_result):
    ys = theta_result["ys"]
    ts = theta_result.get("time", theta_result.get("ts"))
    dts = theta_result["dts"]
    accepted = theta_result["accepted_mask"]

    idx = jnp.where(accepted)[0]
    if idx.size == 0:
        return ts[:1], ys[:1], dts[:1]

    return ts[idx], ys[idx], dts[idx]


def _result_time_series(result):
    ts = result.get("time", result.get("ts"))
    if ts is None:
        raise KeyError("Solver result does not contain a supported time array key ('time' or 'ts').")
    return ts


def main():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting this comparison.") from exc

    problem = SharpTransitionErToyModel(epsilon=0.01)
    y0 = jnp.array([1.0, 1.0, -0.8])
    t0, tf = 0.0, 8.0

    theta_kwargs = dict(
        t0=t0,
        t1=tf,
        dt=2.5e-2,
        theta_implicit=1.0,
        tol=1.0e-9,
        maxiter=25,
        ptc_enabled=True,
        line_search_enabled=True,
        max_step_retries=12,
        differentiable_mode=False,
    )

    theta_off = ThetaNewtonSolver(**theta_kwargs)
    t_start = time.perf_counter()
    out_off = theta_off.solve(y0, problem)
    runtime_off = time.perf_counter() - t_start

    theta_on = ThetaNewtonSolver(**theta_kwargs)
    t_start = time.perf_counter()
    out_on = theta_on.solve(y0, problem, ap_preconditioner=ap_diag_for_toy(problem))
    runtime_on = time.perf_counter() - t_start

    # Reference trajectory for error comparison.
    radau = RADAUSolver(t0=t0, t1=tf, dt=2.0e-1, rtol=1.0e-7, atol=1.0e-9)
    out_ref = radau.solve(y0, problem)

    t_off, y_off, dt_off = _accepted_series(out_off)
    t_on, y_on, dt_on = _accepted_series(out_on)
    t_ref = _result_time_series(out_ref)
    y_ref = out_ref["ys"]

    # Interpolate reference Er to theta times.
    er_ref_off = jnp.interp(t_off, t_ref, y_ref[:, 2])
    er_ref_on = jnp.interp(t_on, t_ref, y_ref[:, 2])
    er_err_off = jnp.abs(y_off[:, 2] - er_ref_off)
    er_err_on = jnp.abs(y_on[:, 2] - er_ref_on)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(t_off, y_off[:, 0], label="n (AP off)", color="#c2410c")
    ax.plot(t_on, y_on[:, 0], "--", label="n (AP on)", color="#1d4ed8")
    ax.plot(t_ref, y_ref[:, 0], ":", label="n (RADAU ref)", color="#374151")
    ax.set_title("Density trajectory")
    ax.set_xlabel("t")
    ax.set_ylabel("n")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    ax.plot(t_off, y_off[:, 2], label="Er (AP off)", color="#c2410c")
    ax.plot(t_on, y_on[:, 2], "--", label="Er (AP on)", color="#1d4ed8")
    ax.plot(t_ref, y_ref[:, 2], ":", label="Er (RADAU ref)", color="#374151")
    ax.set_title("Electric field trajectory")
    ax.set_xlabel("t")
    ax.set_ylabel("Er")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 0]
    ax.semilogy(t_off, er_err_off + 1.0e-16, label="|Er-Er_ref| (AP off)", color="#c2410c")
    ax.semilogy(t_on, er_err_on + 1.0e-16, "--", label="|Er-Er_ref| (AP on)", color="#1d4ed8")
    ax.set_title("Er absolute error vs RADAU")
    ax.set_xlabel("t")
    ax.set_ylabel("abs error")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    ax.plot(t_off, dt_off, label="accepted dt (AP off)", color="#c2410c")
    ax.plot(t_on, dt_on, "--", label="accepted dt (AP on)", color="#1d4ed8")
    ax.set_title("Accepted step sizes")
    ax.set_xlabel("t")
    ax.set_ylabel("dt")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    title = (
        "Theta AP preconditioner comparison"
        f"\nRuntime AP off={runtime_off:.3f}s, AP on={runtime_on:.3f}s"
        f" | accepted steps off={int(jnp.sum(out_off['accepted_mask']))}, on={int(jnp.sum(out_on['accepted_mask']))}"
    )
    fig.suptitle(title, fontsize=11)

    out_png = "theta_ap_comparison.png"
    fig.savefig(out_png, dpi=160)
    print(f"Saved plot: {out_png}")

    plt.show()


if __name__ == "__main__":
    main()
