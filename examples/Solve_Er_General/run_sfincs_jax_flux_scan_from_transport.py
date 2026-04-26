#!/usr/bin/env python3
"""Run a radial sfincs_jax flux scan from a NEOPAX transport snapshot.

This script is a first-step bridge between NEOPAX transport outputs and
sfincs_jax. It:

1. reads a NEOPAX ``transport_solution.h5`` file,
2. extracts one saved time slice, defaulting to the final one,
3. builds one local sfincs_jax run per selected radial point,
4. launches those runs in parallel on CPUs or pinned GPUs,
5. collects particle flux, heat flux, and parallel-flow diagnostics,
6. writes an HDF5 profile file with datasets ``r``, ``Gamma``, ``Q``, and
   ``Upar`` that can be read by NEOPAX's ``FluxesRFileTransportModel``.
7. optionally writes PNG summary plots for ``Gamma``, ``Q``, and ``Upar``.

Default sfincs_jax resolution overrides used by this bridge:
- ``Ntheta = 25``
- ``Nzeta = 51``
- ``Nxi = 100``
- ``NL = 3``
- ``Nx = 5``
- ``solverTolerance = 1e-6``

Important note on normalization:
The written ``Gamma`` and ``Q`` values are taken from sfincs_jax's
SFINCS-style output fields, preferring the ``*_rHat`` variants when available.
This makes the bridge practical for workflow prototyping, but exact
normalization against NEOPAX's native flux units should still be verified
carefully before treating the result as a strict physical replacement.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any

import h5py
import numpy as np
from scipy.constants import elementary_charge, proton_mass

NEOPAX_DENSITY_REFERENCE_M3 = 1.0e20
NEOPAX_TEMPERATURE_REFERENCE_EV = 1.0e3
SFINCS_REFERENCE_R_M = 1.0
SFINCS_REFERENCE_MASS_KG = proton_mass
SFINCS_REFERENCE_T_J = NEOPAX_TEMPERATURE_REFERENCE_EV * elementary_charge
SFINCS_REFERENCE_V_MS = float(np.sqrt(2.0 * SFINCS_REFERENCE_T_J / SFINCS_REFERENCE_MASS_KG))
SFINCS_GAMMA_TO_NEOPAX = NEOPAX_DENSITY_REFERENCE_M3 * SFINCS_REFERENCE_V_MS / SFINCS_REFERENCE_R_M
SFINCS_Q_TO_NEOPAX = (
    NEOPAX_DENSITY_REFERENCE_M3
    * SFINCS_REFERENCE_MASS_KG
    * SFINCS_REFERENCE_V_MS**3
    / SFINCS_REFERENCE_R_M
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "sfincs_jax_flux_scan"


@dataclass(frozen=True)
class SpeciesMeta:
    name: str
    charge: float
    mass_mp: float


@dataclass(frozen=True)
class TransportSnapshot:
    rho: np.ndarray
    density: np.ndarray
    temperature: np.ndarray
    er: np.ndarray
    time_value: float | None


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _infer_neopax_root(config_path: Path) -> Path:
    parent = config_path.resolve().parent
    if parent.parent.name == "examples":
        return parent.parent.parent
    return parent


def _resolve_relative(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _default_transport_solution_path(config_path: Path, cfg: dict[str, Any]) -> Path:
    output_cfg = cfg.get("transport_output", {})
    output_dir = _resolve_relative(_infer_neopax_root(config_path), output_cfg.get("transport_output_dir"))
    if output_dir is None:
        raise ValueError("Could not resolve [transport_output].transport_output_dir from the NEOPAX config.")
    return output_dir / "transport_solution.h5"


def _parse_species_from_config(cfg: dict[str, Any]) -> list[SpeciesMeta]:
    species_cfg = cfg.get("species", {})
    names = list(species_cfg.get("names", []))
    charges = list(species_cfg.get("charge_qp", []))
    masses = list(species_cfg.get("mass_mp", []))
    if not names or len(names) != len(charges) or len(names) != len(masses):
        raise ValueError("NEOPAX [species] must define matching names, charge_qp, and mass_mp arrays.")
    return [
        SpeciesMeta(name=str(name), charge=float(charge), mass_mp=float(mass))
        for name, charge, mass in zip(names, charges, masses)
    ]


def _load_transport_snapshot(h5_path: Path, *, time_index: int) -> TransportSnapshot:
    with h5py.File(h5_path, "r") as f:
        keys = set(f.keys())
        if "rho" not in keys or "density" not in keys or "Er" not in keys:
            raise KeyError(f"{h5_path} is missing required NEOPAX transport datasets.")

        rho = np.asarray(f["rho"][()], dtype=np.float64)
        density_all = np.asarray(f["density"][()], dtype=np.float64)
        er_all = np.asarray(f["Er"][()], dtype=np.float64)

        if "temperature" in keys:
            temperature_all = np.asarray(f["temperature"][()], dtype=np.float64)
        elif "pressure" in keys:
            pressure_all = np.asarray(f["pressure"][()], dtype=np.float64)
            density_safe = np.maximum(density_all, 1.0e-30)
            temperature_all = pressure_all / density_safe
        else:
            raise KeyError(f"{h5_path} must contain either temperature or pressure.")

        ts = np.asarray(f["ts"][()], dtype=np.float64) if "ts" in keys else None

    if density_all.ndim == 2:
        idx = 0
        density = density_all
        temperature = temperature_all
        er = er_all
    else:
        idx = int(time_index)
        if idx < 0:
            idx = density_all.shape[0] + idx
        if idx < 0 or idx >= density_all.shape[0]:
            raise IndexError(f"time index {time_index} out of range for {h5_path}")
        density = np.asarray(density_all[idx], dtype=np.float64)
        temperature = np.asarray(temperature_all[idx], dtype=np.float64)
        er = np.asarray(er_all[idx], dtype=np.float64)

    time_value = None if ts is None else float(ts[idx])
    return TransportSnapshot(rho=rho, density=density, temperature=temperature, er=er, time_value=time_value)


def _parse_index_list(text: str | None) -> list[int] | None:
    if text is None or not text.strip():
        return None
    out: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(int(chunk))
    return out


def _choose_radius_indices(
    rho: np.ndarray,
    *,
    explicit: list[int] | None,
    rho_min: float | None,
    rho_max: float | None,
    num_radii: int | None,
) -> list[int]:
    if explicit is not None:
        idxs = sorted(set(int(v) for v in explicit))
        for idx in idxs:
            if idx < 0 or idx >= rho.size:
                raise IndexError(f"rho index {idx} out of range [0, {rho.size - 1}]")
        return idxs

    # By default, skip the magnetic axis point. A local sfincs_jax solve at
    # rho=0 is usually not the most useful first-pass transport postprocessing
    # target, and users can still include it explicitly via --rho-indices.
    mask = np.ones_like(rho, dtype=bool)
    mask &= ~np.isclose(rho, 0.0)
    if rho_min is not None:
        mask &= rho >= float(rho_min)
    if rho_max is not None:
        mask &= rho <= float(rho_max)
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        raise ValueError("No radii satisfy the requested rho filter.")
    if num_radii is None or int(num_radii) <= 0 or int(num_radii) >= candidates.size:
        return [int(v) for v in candidates]
    picks = np.linspace(0, candidates.size - 1, int(num_radii))
    return sorted(set(int(candidates[int(round(p))]) for p in picks))


def _gradient_profile(values: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return np.gradient(np.asarray(values, dtype=np.float64), np.asarray(rho, dtype=np.float64), edge_order=2)


def _bool_literal(value: bool) -> str:
    return ".true." if bool(value) else ".false."


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return _bool_literal(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.16g}"


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(_format_scalar(v) for v in np.asarray(value).ravel())
    return _format_scalar(value)


def _patch_group_value(*, text: str, group: str, key: str, value: Any) -> str:
    import re

    start = re.search(rf"(?im)^\s*&{re.escape(group)}\s*$", text)
    if start is None:
        raise ValueError(f"Missing namelist group &{group}")
    end = re.search(r"(?m)^\s*/\s*$", text[start.end() :])
    if end is None:
        raise ValueError(f"Missing '/' terminator for &{group}")
    end_pos = start.end() + end.start()
    group_txt = text[start.end() : end_pos]

    pat = re.compile(rf"(?im)^[ \t]*{re.escape(key)}[ \t]*=[ \t]*([^!\n\r]+)[ \t]*$")
    line = f"  {key} = {_format_value(value)}"
    match = pat.search(group_txt)
    if match is not None:
        group_txt2 = group_txt.replace(match.group(0), line)
    else:
        if not group_txt.endswith("\n"):
            group_txt = group_txt + "\n"
        group_txt2 = group_txt + line + "\n"
    return text[: start.end()] + group_txt2 + text[end_pos:]


def _drop_group_key(*, text: str, group: str, key: str) -> str:
    import re

    start = re.search(rf"(?im)^\s*&{re.escape(group)}\s*$", text)
    if start is None:
        raise ValueError(f"Missing namelist group &{group}")
    end = re.search(r"(?m)^\s*/\s*$", text[start.end() :])
    if end is None:
        raise ValueError(f"Missing '/' terminator for &{group}")
    end_pos = start.end() + end.start()
    group_txt = text[start.end() : end_pos]

    pat = re.compile(rf"(?im)^[ \t]*{re.escape(key)}[ \t]*=[^!\n\r]*(?:![^\n\r]*)?[\r]?\n?")
    group_txt2 = pat.sub("", group_txt)
    return text[: start.end()] + group_txt2 + text[end_pos:]


def _prepare_input_text(
    *,
    template_text: str,
    species: list[SpeciesMeta],
    snapshot: TransportSnapshot,
    radius_index: int,
    include_phi1: bool | None,
    resolution_overrides: dict[str, int | None],
    solver_tolerance: float | None,
) -> str:
    rho = snapshot.rho
    density = snapshot.density
    temperature = snapshot.temperature
    er = snapshot.er

    dndr = np.vstack([_gradient_profile(density[i], rho) for i in range(density.shape[0])])
    dtdr = np.vstack([_gradient_profile(temperature[i], rho) for i in range(temperature.shape[0])])

    text = template_text
    text = _patch_group_value(text=text, group="general", key="RHSMode", value=1)
    text = _patch_group_value(text=text, group="geometryParameters", key="inputRadialCoordinate", value=3)
    # Let sfincs_jax infer gradient coordinates separately:
    # species from dNHatdrNs/dTHatdrNs -> mode 3, Phi from Er -> mode 4.
    text = _drop_group_key(text=text, group="geometryParameters", key="inputRadialCoordinateForGradients")
    text = _patch_group_value(text=text, group="geometryParameters", key="rN_wish", value=float(rho[radius_index]))

    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="Zs",
        value=[sp.charge for sp in species],
    )
    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="mHats",
        value=[sp.mass_mp for sp in species],
    )
    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="nHats",
        value=density[:, radius_index],
    )
    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="THats",
        value=temperature[:, radius_index],
    )
    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="dNHatdrNs",
        value=dndr[:, radius_index],
    )
    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="dTHatdrNs",
        value=dtdr[:, radius_index],
    )
    text = _patch_group_value(
        text=text,
        group="physicsParameters",
        key="Er",
        value=float(er[radius_index]),
    )

    if include_phi1 is not None:
        text = _patch_group_value(
            text=text,
            group="physicsParameters",
            key="includePhi1",
            value=bool(include_phi1),
        )

    for key, value in resolution_overrides.items():
        if value is not None:
            text = _patch_group_value(text=text, group="resolutionParameters", key=key, value=int(value))

    if solver_tolerance is not None:
        text = _patch_group_value(
            text=text,
            group="resolutionParameters",
            key="solverTolerance",
            value=float(solver_tolerance),
        )

    return text


def _infer_wout_path(config_path: Path, cfg: dict[str, Any], explicit: str | None) -> Path | None:
    if explicit:
        return _resolve_relative(_infer_neopax_root(config_path), explicit)
    geometry_cfg = cfg.get("geometry", {})
    return _resolve_relative(_infer_neopax_root(config_path), geometry_cfg.get("vmec_file"))


def _last_species_vector(arr: np.ndarray, n_species: int) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 1:
        if data.shape[0] != n_species:
            raise ValueError(f"Expected {n_species} species values, got shape {data.shape}")
        return data
    if data.ndim == 2:
        if data.shape[0] == n_species:
            return data[:, -1]
        if data.shape[1] == n_species:
            return data[-1, :]
    raise ValueError(f"Unsupported diagnostic shape {data.shape} for n_species={n_species}")


def _last_scalar(arr: np.ndarray) -> float:
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 0:
        return float(data)
    return float(np.ravel(data)[-1])


def _extract_flux_triplet(
    results: dict[str, Any],
    n_species: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None, dict[str, str]]:
    gamma_key = None
    q_key = None
    for key in ("particleFlux_vm_rHat", "particleFlux_vm_rN", "particleFlux_vm_psiHat"):
        if key in results:
            gamma_key = key
            break
    for key in ("heatFlux_vm_rHat", "heatFlux_vm_rN", "heatFlux_vm_psiHat"):
        if key in results:
            q_key = key
            break
    if gamma_key is None or q_key is None or "FSABFlow" not in results:
        raise KeyError("sfincs_jax output is missing required flux/flow diagnostics.")

    gamma_hat = _last_species_vector(np.asarray(results[gamma_key]), n_species)
    q_hat = _last_species_vector(np.asarray(results[q_key]), n_species)
    gamma = SFINCS_GAMMA_TO_NEOPAX * gamma_hat
    q = SFINCS_Q_TO_NEOPAX * q_hat

    fsab_flow = _last_species_vector(np.asarray(results["FSABFlow"]), n_species)
    b0_over_bbar = _last_scalar(np.asarray(results.get("B0OverBBar", 1.0), dtype=np.float64))
    # NTX fixed-field parallel-flow audit bridge:
    # NEOPAX's physical Upar closure matches the SFINCS hat-normalized FSABFlow
    # after restoring the historical factor 2 * B0OverBBar / sqrt(pi).
    flow_bridge = 2.0 * float(b0_over_bbar) / float(np.sqrt(np.pi))
    upar = flow_bridge * fsab_flow
    meta = {
        "Gamma_key": gamma_key,
        "Q_key": q_key,
        "Upar_key": "FSABFlow",
        "Gamma_scale_to_neopax": f"{SFINCS_GAMMA_TO_NEOPAX:.16g}",
        "Q_scale_to_neopax": f"{SFINCS_Q_TO_NEOPAX:.16g}",
        "Upar_bridge": "2*B0OverBBar/sqrt(pi)",
        "B0OverBBar": f"{b0_over_bbar:.16g}",
    }
    r_hat = None
    if "rHat" in results:
        r_hat = _last_scalar(np.asarray(results["rHat"], dtype=np.float64))
        meta["rHat"] = f"{r_hat:.16g}"
    return gamma, q, upar, r_hat, meta


def _build_worker_env(args: argparse.Namespace, *, gpu_id: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if str(args.backend).lower() == "gpu":
        # On NVIDIA-backed JAX installs, using the generic "gpu" alias can
        # still let JAX probe other accelerator backends such as ROCm. Force
        # CUDA explicitly for these worker processes.
        env["JAX_PLATFORMS"] = "cuda"
        env["JAX_PLATFORM_NAME"] = "cuda"
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Drop accelerator-selection variables inherited from the parent shell
        # that can redirect JAX toward a different backend family.
        env.pop("PJRT_DEVICE", None)
        env.pop("JAX_BACKEND_TARGET", None)
        env.pop("ROCM_VISIBLE_DEVICES", None)
        env.pop("HIP_VISIBLE_DEVICES", None)
        env["SFINCS_JAX_SHARD"] = "0"
        env["SFINCS_JAX_AUTO_SHARD"] = "0"
        env["SFINCS_JAX_MATVEC_SHARD_AXIS"] = "off"
    else:
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_PLATFORM_NAME"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
        if int(args.cores_per_run) > 0:
            env["SFINCS_JAX_CORES"] = str(int(args.cores_per_run))
            env["SFINCS_JAX_SHARD"] = "0"
            env["SFINCS_JAX_AUTO_SHARD"] = "0"
            env["SFINCS_JAX_MATVEC_SHARD_AXIS"] = "off"
    return env


def _run_single_worker_from_payload(payload_path: Path) -> int:
    with payload_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    input_path = Path(payload["input_path"])
    output_path = Path(payload["output_path"])
    result_json = Path(payload["result_json"])
    wout_path = payload.get("wout_path")
    n_species = int(payload["n_species"])

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5

    write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=output_path,
        wout_path=None if wout_path in (None, "") else Path(wout_path),
        compute_transport_matrix=False,
        compute_solution=True,
        overwrite=True,
        verbose=bool(payload.get("verbose", False)),
    )
    results = read_sfincs_h5(output_path)
    gamma, q, upar, r_hat, meta = _extract_flux_triplet(results, n_species)
    summary = {
        "radius_index": int(payload["radius_index"]),
        "rho": float(payload["rho"]),
        "rHat": None if r_hat is None else float(r_hat),
        "Gamma": gamma.tolist(),
        "Q": q.tolist(),
        "Upar": upar.tolist(),
        "meta": meta,
    }
    with result_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return 0


def _write_summary_plots(
    *,
    output_dir: Path,
    rho: np.ndarray,
    gamma: np.ndarray,
    q: np.ndarray,
    upar: np.ndarray,
    species: list[SpeciesMeta],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Plotting was requested but matplotlib is not installed."
        ) from exc

    traces = (
        ("Gamma", np.asarray(gamma, dtype=np.float64), output_dir / "Gamma_vs_rho.png"),
        ("Q", np.asarray(q, dtype=np.float64), output_dir / "Q_vs_rho.png"),
        ("Upar", np.asarray(upar, dtype=np.float64), output_dir / "Upar_vs_rho.png"),
    )

    for label, values, path in traces:
        fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
        for i, sp in enumerate(species):
            ax.plot(rho, values[i], marker="o", linewidth=1.5, markersize=4.0, label=sp.name)
        ax.set_xlabel("rho")
        ax.set_ylabel(label)
        ax.set_title(f"{label} from sfincs_jax radial scan")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def _suppress_benign_worker_stderr(stderr: str, args: argparse.Namespace) -> str:
    if str(args.backend).lower() != "cpu":
        return stderr
    text = (stderr or "").strip()
    if not text:
        return ""
    benign_markers = (
        "Jax plugin configuration error",
        "jax_plugins.xla_cuda12.initialize()",
        "cuda_device_count()",
        "operation cuInit(0) failed: CUDA_ERROR_NO_DEVICE",
    )
    if all(marker in text for marker in benign_markers):
        return ""
    return stderr


def _launch_one_subprocess(
    *,
    script_path: Path,
    payload_path: Path,
    env: dict[str, str],
    stream_output: bool,
) -> tuple[int, str, str]:
    cmd = [sys.executable, str(script_path), "--worker-payload", str(payload_path)]
    if stream_output:
        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            check=False,
        )
        return int(proc.returncode), "", ""
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return int(proc.returncode), proc.stdout, proc.stderr


def _run_tasks_in_parallel(
    *,
    task_payloads: list[Path],
    args: argparse.Namespace,
    gpu_ids: list[str],
) -> None:
    max_parallel = max(1, int(args.max_parallel))
    script_path = Path(__file__).resolve()
    total = len(task_payloads)
    completed = 0

    def _submit(executor: ThreadPoolExecutor, payload_path: Path, slot: int):
        gpu_id = None
        if str(args.backend).lower() == "gpu":
            gpu_id = gpu_ids[slot % len(gpu_ids)]
        env = _build_worker_env(args, gpu_id=gpu_id)
        stream_output = bool(args.verbose_workers) and max_parallel == 1
        future = executor.submit(
            _launch_one_subprocess,
            script_path=script_path,
            payload_path=payload_path,
            env=env,
            stream_output=stream_output,
        )
        future._codex_payload = payload_path  # type: ignore[attr-defined]
        future._codex_gpu_id = gpu_id  # type: ignore[attr-defined]
        future._codex_stream_output = stream_output  # type: ignore[attr-defined]
        return future

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        running = set()
        pending = list(task_payloads)
        slot = 0
        while pending or running:
            while pending and len(running) < max_parallel:
                payload_path = pending.pop(0)
                running.add(_submit(executor, payload_path, slot))
                slot += 1
            done, running = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                code, stdout, stderr = fut.result()
                payload_path = getattr(fut, "_codex_payload", None)
                gpu_id = getattr(fut, "_codex_gpu_id", None)
                stream_output = getattr(fut, "_codex_stream_output", False)
                label = str(payload_path) if payload_path is not None else "<payload>"
                if code != 0:
                    msg = [
                        f"sfincs_jax worker failed for {label}",
                    ]
                    if gpu_id is not None:
                        msg.append(f"gpu={gpu_id}")
                    if stdout.strip():
                        msg.append("stdout:\n" + stdout.strip())
                    if stderr.strip():
                        msg.append("stderr:\n" + stderr.strip())
                    raise RuntimeError("\n".join(msg))
                if bool(args.verbose_workers) and not stream_output:
                    if stdout.strip():
                        print(f"[sfincs-worker stdout] {label}\n{stdout.strip()}", flush=True)
                    stderr_to_print = _suppress_benign_worker_stderr(stderr, args)
                    if stderr_to_print.strip():
                        print(f"[sfincs-worker stderr] {label}\n{stderr_to_print.strip()}", flush=True)
                completed += 1
                rho_value = None
                if payload_path is not None:
                    try:
                        payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
                        rho_value = float(payload.get("rho"))
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        rho_value = None
                rho_note = f" rho={rho_value:.4f}" if rho_value is not None else ""
                gpu_note = f" gpu={gpu_id}" if gpu_id is not None else ""
                print(f"[sfincs-scan] completed {completed}/{total}:{rho_note}{gpu_note}", flush=True)


def cmd_main(args: argparse.Namespace) -> int:
    config_path = Path(args.neopax_config).resolve()
    cfg = _load_toml(config_path)
    species = _parse_species_from_config(cfg)

    transport_solution = (
        Path(args.neopax_result).resolve()
        if args.neopax_result is not None
        else _default_transport_solution_path(config_path, cfg)
    )
    if not transport_solution.exists():
        raise FileNotFoundError(
            f"Could not find NEOPAX transport result at {transport_solution}. "
            "If needed, enable [transport_output].transport_write_hdf5 = true and rerun NEOPAX."
        )

    snapshot = _load_transport_snapshot(transport_solution, time_index=int(args.time_index))
    explicit_indices = _parse_index_list(args.rho_indices)
    radius_indices = _choose_radius_indices(
        snapshot.rho,
        explicit=explicit_indices,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        num_radii=args.num_radii,
    )

    template_path = Path(args.sfincs_template).resolve()
    template_text = template_path.read_text(encoding="utf-8")

    output_dir = Path(args.output_dir).resolve()
    run_dir = output_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    wout_path = _infer_wout_path(config_path, cfg, args.wout_path)
    resolution_overrides = {
        "Ntheta": args.ntheta,
        "Nzeta": args.nzeta,
        "Nxi": args.nxi,
        "NL": args.nl,
        "Nx": args.nx,
    }

    task_payloads: list[Path] = []
    for radius_index in radius_indices:
        rho_value = float(snapshot.rho[radius_index])
        run_name = f"rho_{radius_index:03d}_r{rho_value:.4f}".replace(".", "p")
        surface_dir = run_dir / run_name
        surface_dir.mkdir(parents=True, exist_ok=True)

        input_text = _prepare_input_text(
            template_text=template_text,
            species=species,
            snapshot=snapshot,
            radius_index=radius_index,
            include_phi1=args.include_phi1,
            resolution_overrides=resolution_overrides,
            solver_tolerance=args.solver_tolerance,
        )
        input_path = surface_dir / "input.namelist"
        input_path.write_text(input_text, encoding="utf-8")

        payload = {
            "radius_index": int(radius_index),
            "rho": rho_value,
            "input_path": str(input_path),
            "output_path": str(surface_dir / "sfincsOutput.h5"),
            "result_json": str(surface_dir / "result.json"),
            "wout_path": None if wout_path is None else str(wout_path),
            "n_species": len(species),
            "verbose": bool(args.verbose_workers),
        }
        payload_path = surface_dir / "payload.json"
        with payload_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        task_payloads.append(payload_path)

    backend = str(args.backend).lower()
    gpu_ids = [token.strip() for token in str(args.gpu_ids).split(",") if token.strip()]
    if backend == "gpu" and not gpu_ids:
        gpu_ids = ["0"]

    rho_min_val = float(np.min(snapshot.rho[radius_indices]))
    rho_max_val = float(np.max(snapshot.rho[radius_indices]))
    backend_note = f"backend={backend}"
    parallel_note = f"max_parallel={int(args.max_parallel)}"
    if backend == "gpu":
        placement_note = f"gpu_ids={','.join(gpu_ids)}"
    else:
        placement_note = f"cores_per_run={int(args.cores_per_run)}"
    print(
        "[sfincs-scan] "
        f"selected {len(radius_indices)} radii over rho in [{rho_min_val:.4f}, {rho_max_val:.4f}] "
        f"({backend_note}, {parallel_note}, {placement_note})"
    , flush=True)

    _run_tasks_in_parallel(
        task_payloads=task_payloads,
        args=args,
        gpu_ids=gpu_ids,
    )

    rho_out = []
    rhat_out = []
    gamma_out = []
    q_out = []
    upar_out = []
    raw_meta = {"Gamma_key": None, "Q_key": None, "Upar_key": None}
    for payload_path in sorted(task_payloads, key=lambda p: json.loads(p.read_text(encoding="utf-8"))["rho"]):
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        result_json = Path(payload["result_json"])
        summary = json.loads(result_json.read_text(encoding="utf-8"))
        rho_out.append(float(summary["rho"]))
        rhat_value = summary.get("rHat")
        if rhat_value is None:
            raise KeyError(f"Worker summary {result_json} is missing rHat.")
        rhat_out.append(float(rhat_value))
        gamma_out.append(np.asarray(summary["Gamma"], dtype=np.float64))
        q_out.append(np.asarray(summary["Q"], dtype=np.float64))
        upar_out.append(np.asarray(summary["Upar"], dtype=np.float64))
        raw_meta = dict(summary.get("meta", raw_meta))

    rho_arr = np.asarray(rho_out, dtype=np.float64)
    rhat_arr = np.asarray(rhat_out, dtype=np.float64)
    gamma_arr = np.stack(gamma_out, axis=1)
    q_arr = np.stack(q_out, axis=1)
    upar_arr = np.stack(upar_out, axis=1)

    out_h5 = output_dir / "sfincs_jax_flux_profiles.h5"
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("r", data=rhat_arr)
        f.create_dataset("rHat", data=rhat_arr)
        f.create_dataset("rho", data=rho_arr)
        f.create_dataset("Gamma", data=gamma_arr)
        f.create_dataset("Q", data=q_arr)
        f.create_dataset("Upar", data=upar_arr)
        f.create_dataset("species_names", data=np.asarray([sp.name.encode("utf-8") for sp in species]))
        f.attrs["source_transport_solution"] = str(transport_solution)
        f.attrs["source_sfincs_template"] = str(template_path)
        f.attrs["time_index"] = int(args.time_index)
        f.attrs["time_value"] = np.nan if snapshot.time_value is None else float(snapshot.time_value)
        f.attrs["backend"] = backend
        f.attrs["max_parallel"] = int(args.max_parallel)
        f.attrs["include_phi1"] = bool(args.include_phi1) if args.include_phi1 is not None else -1
        for key, value in raw_meta.items():
            if value is not None:
                f.attrs[f"raw_{key}"] = str(value)
        f.attrs["Upar_note"] = (
            "Upar is derived from sfincs_jax FSABFlow using the NTX fixed-field "
            "parallel-flow bridge factor 2*B0OverBBar/sqrt(pi)."
        )
        f.attrs["normalization_note"] = (
            "Gamma is converted from sfincs_jax particleFlux_vm_* using nbar*vbar/Rbar "
            "with nbar=1e20 m^-3, Tbar=1 keV, mbar=mp, Rbar=1 m. "
            "Q is converted from heatFlux_vm_* using nbar*mbar*vbar^3/Rbar with the same references. "
            "Upar uses the NTX archive-backed observable bridge rather than a raw FSABFlow alias."
        )
        f.attrs["radius_note"] = (
            "The saved coordinate r/rHat uses sfincs_jax's rHat output. "
            "rho is also saved separately from the source NEOPAX transport file."
        )

    if bool(args.plot):
        _write_summary_plots(
            output_dir=output_dir,
            rho=rho_arr,
            gamma=gamma_arr,
            q=q_arr,
            upar=upar_arr,
            species=species,
        )

    print(f"wrote {out_h5}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            __doc__
            + "\n\n"
            + "Unless overridden on the command line, this script forces the following "
            + "sfincs_jax resolution settings: "
            + "Ntheta=25, Nzeta=51, Nxi=100, NL=3, Nx=5, solverTolerance=1e-6."
        )
    )
    p.add_argument("--neopax-config", required=False, default=None, help="Path to the NEOPAX transport TOML.")
    p.add_argument(
        "--neopax-result",
        default=None,
        help="Optional explicit path to transport_solution.h5. If omitted, infer from the NEOPAX config.",
    )
    p.add_argument("--sfincs-template", required=False, default=None, help="Template sfincs_jax input.namelist.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for runs and collected fluxes.")
    p.add_argument("--time-index", type=int, default=-1, help="Time index in transport_solution.h5. Default: final.")
    p.add_argument("--rho-indices", default=None, help="Comma-separated explicit rho indices.")
    p.add_argument("--rho-min", type=float, default=None, help="Minimum rho to include.")
    p.add_argument("--rho-max", type=float, default=None, help="Maximum rho to include.")
    p.add_argument("--num-radii", type=int, default=None, help="Number of radii to sample inside the rho filter.")
    p.add_argument("--wout-path", default=None, help="Optional VMEC equilibrium override for sfincs_jax.")
    p.add_argument("--include-phi1", dest="include_phi1", action="store_true", help="Force includePhi1 = true.")
    p.add_argument("--no-include-phi1", dest="include_phi1", action="store_false", help="Force includePhi1 = false.")
    p.set_defaults(include_phi1=None)
    p.add_argument("--ntheta", type=int, default=25, help="Override Ntheta. Default: 25.")
    p.add_argument("--nzeta", type=int, default=51, help="Override Nzeta. Default: 51.")
    p.add_argument("--nxi", type=int, default=100, help="Override Nxi. Default: 100.")
    p.add_argument("--nl", type=int, default=3, help="Override NL. Default: 3.")
    p.add_argument("--nx", type=int, default=5, help="Override Nx. Default: 5.")
    p.add_argument("--solver-tolerance", type=float, default=1.0e-6, help="Override solverTolerance. Default: 1e-6.")
    p.add_argument("--backend", choices=("cpu", "gpu"), default="cpu", help="Parallel execution backend.")
    p.add_argument("--gpu-ids", default="0", help="Comma-separated GPU ids for backend=gpu.")
    p.add_argument("--max-parallel", type=int, default=1, help="Maximum concurrent sfincs_jax runs.")
    p.add_argument("--cores-per-run", type=int, default=1, help="CPU cores per run for backend=cpu.")
    p.add_argument("--plot", action="store_true", help="Write PNG plots of Gamma, Q, and Upar versus rho.")
    p.add_argument("--verbose-workers", action="store_true", help="Allow verbose sfincs_jax worker logging.")
    p.add_argument("--worker-payload", default=None, help=argparse.SUPPRESS)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.worker_payload:
        return _run_single_worker_from_payload(Path(args.worker_payload).resolve())
    if args.neopax_config is None:
        raise ValueError("--neopax-config is required unless using the hidden worker mode.")
    if args.sfincs_template is None:
        raise ValueError("--sfincs-template is required unless using the hidden worker mode.")
    return cmd_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
