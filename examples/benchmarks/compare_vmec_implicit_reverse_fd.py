from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


NEOPAX_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = NEOPAX_ROOT.parent
for path in (NEOPAX_ROOT, WORKSPACE_ROOT / "vmec_jax"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vmec_jax.core import implicit as im  # noqa: E402
from vmec_jax.core import optimize as opt  # noqa: E402
from vmec_jax.core.input import VmecInput  # noqa: E402


DEFAULT_VMEC_INPUT = NEOPAX_ROOT / "examples" / "inputs" / "input.QI_nfp2_newNT_opt_hires_true"


OBJECTIVE_FUNCS = {
    "aspect_ratio": opt.aspect_ratio,
    "volume": opt.volume,
    "mean_iota": opt.mean_iota,
    "edge_iota": opt.edge_iota,
    "magnetic_well": opt.magnetic_well,
    "mirror_ratio": opt.mirror_ratio,
}


def _parse_param_specs(text: str) -> tuple[tuple[str, int, int], ...]:
    specs: list[tuple[str, int, int]] = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = [part.strip() for part in raw.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Parameter spec {raw!r} must be FAMILY:m:n, e.g. RBC:1:0.")
        family = parts[0].lower()
        if family not in {"rbc", "rbs", "zbc", "zbs"}:
            raise ValueError(f"Unsupported family {parts[0]!r}; use RBC, RBS, ZBC, or ZBS.")
        specs.append((family, int(parts[1]), int(parts[2])))
    if not specs:
        raise ValueError("--param-specs did not contain any valid specs.")
    return tuple(specs)


def _parse_objectives(text: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in str(text).split(",") if name.strip())
    if not names:
        raise ValueError("--objectives did not contain any valid objective names.")
    unknown = [name for name in names if name not in OBJECTIVE_FUNCS]
    if unknown:
        raise ValueError(
            f"Unknown objectives {unknown}; choices are {', '.join(OBJECTIVE_FUNCS)}."
        )
    return names


def _param_index(inp: VmecInput, family: str, m: int, n: int) -> tuple[int, int]:
    row = int(n) + int(inp.ntor)
    col = int(m)
    arr = getattr(im.params_from_input(inp), family)
    if row < 0 or row >= arr.shape[0] or col < 0 or col >= arr.shape[1]:
        raise ValueError(
            f"{family.upper()}:{m}:{n} maps to index {(row, col)}, outside shape {arr.shape}."
        )
    return row, col


def _param_value(params: im.ImplicitParams, family: str, row: int, col: int) -> float:
    return float(np.asarray(getattr(params, family))[row, col])


def _with_param_delta(
    params: im.ImplicitParams,
    family: str,
    row: int,
    col: int,
    delta,
) -> im.ImplicitParams:
    arr = getattr(params, family)
    return dataclasses.replace(params, **{family: arr.at[row, col].add(delta)})


def _fd_step(base_value: float, *, rel_step: float, abs_step: float) -> float:
    return max(abs(float(base_value)) * float(rel_step), float(abs_step))


def _relative_error(ad: float, fd: float) -> float:
    denom = max(abs(fd), 1.0e-300)
    return abs(ad - fd) / denom


def _make_objective(inp: VmecInput, cfg: im.ImplicitConfig, name: str):
    objective = OBJECTIVE_FUNCS[name]

    def func(params: im.ImplicitParams):
        state = im.solve_implicit(params, cfg)
        rt = im.runtime_from_params(params, cfg)
        return jnp.asarray(objective(state, rt), dtype=jnp.float64).reshape(())

    return func


def _run_one(args, inp: VmecInput, cfg: im.ImplicitConfig, params0: im.ImplicitParams, objective_name: str, spec):
    family, m, n = spec
    row, col = _param_index(inp, family, m, n)
    base_value = _param_value(params0, family, row, col)
    h = _fd_step(base_value, rel_step=args.fd_rel_step, abs_step=args.fd_abs_step)
    objective = _make_objective(inp, cfg, objective_name)

    def scalar(delta):
        return objective(_with_param_delta(params0, family, row, col, delta))

    print(
        f"[vmec-implicit-fd] objective={objective_name} parameter={family.upper()}:{m}:{n} "
        f"base={base_value:.16e} fd_step={h:.6e}",
        flush=True,
    )
    baseline = float(jax.device_get(scalar(jnp.asarray(0.0, dtype=jnp.float64))))
    ad = float(jax.device_get(jax.grad(scalar)(jnp.asarray(0.0, dtype=jnp.float64))))
    minus = float(jax.device_get(scalar(jnp.asarray(-h, dtype=jnp.float64))))
    plus = float(jax.device_get(scalar(jnp.asarray(h, dtype=jnp.float64))))
    fd = (plus - minus) / (2.0 * h)
    print(
        f"  value={baseline:.16e} fd={fd:.16e} reverse_ad={ad:.16e} "
        f"abs_delta={abs(ad - fd):.6e} rel_err={_relative_error(ad, fd):.6e}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "VMEC-only implicit reverse-AD vs centered-FD diagnostic. "
            "This bypasses NEOPAX geometry/Boozer/QI wrappers."
        )
    )
    parser.add_argument("--vmec-input", default=str(DEFAULT_VMEC_INPUT))
    parser.add_argument("--param-specs", default="RBC:1:0,ZBS:1:0")
    parser.add_argument(
        "--objectives",
        default="aspect_ratio,volume,mean_iota,magnetic_well,mirror_ratio",
        help=f"Comma-separated objective names. Choices: {', '.join(OBJECTIVE_FUNCS)}.",
    )
    parser.add_argument("--fd-rel-step", type=float, default=3e-7)
    parser.add_argument("--fd-abs-step", type=float, default=1e-10)
    parser.add_argument("--ns", type=int, default=None)
    parser.add_argument("--ftol", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--mode", default="cli")
    parser.add_argument("--multigrid", action="store_true")
    parser.add_argument("--no-lconm1", action="store_true")
    parser.add_argument("--adjoint-tol", type=float, default=1e-11)
    parser.add_argument("--adjoint-restart", type=int, default=30)
    parser.add_argument("--adjoint-maxiter", type=int, default=300)
    args = parser.parse_args()

    vmec_input = Path(args.vmec_input).resolve()
    inp = VmecInput.from_file(str(vmec_input))
    cfg = im.make_config(
        inp,
        ns=args.ns,
        ftol=args.ftol,
        max_iterations=args.max_iterations,
        mode=args.mode,
        multigrid=bool(args.multigrid),
        lconm1=not bool(args.no_lconm1),
        adjoint_tol=args.adjoint_tol,
        adjoint_restart=args.adjoint_restart,
        adjoint_maxiter=args.adjoint_maxiter,
    )
    params0 = im.params_from_input(inp)
    specs = _parse_param_specs(args.param_specs)
    objectives = _parse_objectives(args.objectives)

    print(
        "[vmec-implicit-fd] "
        f"input={vmec_input} ns={cfg.resolution.ns} ftol={cfg.ftol:.6e} "
        f"max_iterations={cfg.max_iterations} mode={cfg.mode} multigrid={cfg.multigrid} "
        f"lconm1={cfg.lconm1}",
        flush=True,
    )
    print(
        f"[vmec-implicit-fd] jax_backend={jax.default_backend()} devices={jax.devices()}",
        flush=True,
    )

    for objective_name in objectives:
        for spec in specs:
            _run_one(args, inp, cfg, params0, objective_name, spec)


if __name__ == "__main__":
    main()
