"""Diagnose a saved profile-FD Newton failure without replaying its history.

Run on the GPU machine. Runtime geometry/database setup is still required;
the adaptive baseline, prior accepted steps, objectives and reverse sweep
are not executed. Snapshot arrays are loaded with allow_pickle=False.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def restore_carry(archive, name, template, metadata):
    """Require the saved carry to match the rebuilt model's numerical layout."""

    path_leaves, structure = jax.tree_util.tree_flatten_with_path(template)
    paths = [jax.tree_util.keystr(path) for path, _ in path_leaves]
    if paths != metadata["carry_contracts"][name]:
        raise ValueError(f"Snapshot {name} tree layout differs from the rebuilt model.")
    values = []
    for index, (_, template_value) in enumerate(path_leaves):
        value = np.asarray(archive[f"{name}_{index:04d}"])
        template_array = np.asarray(template_value)
        if value.shape != template_array.shape or value.dtype != template_array.dtype:
            raise ValueError(f"Snapshot {name} leaf {index} shape/dtype differs from the rebuilt model.")
        values.append(jnp.asarray(value))
    return jax.tree_util.tree_unflatten(structure, values)


def cold_carry(carry):
    return dataclasses.replace(
        carry, cache_valid=jnp.asarray(False),
        prev_stages=jnp.zeros_like(carry.prev_stages),
        prev_dt=jnp.zeros_like(carry.prev_dt),
        prev_theta_final=jnp.zeros_like(carry.prev_theta_final),
        prev_newton_iter_count=jnp.zeros_like(carry.prev_newton_iter_count),
    )


def array_summary(value):
    array = np.asarray(jax.device_get(value))
    finite = np.isfinite(array)
    good = array[finite]
    return {
        "shape": list(array.shape), "all_finite": bool(np.all(finite)),
        "min_finite": float(np.min(good)) if good.size else None,
        "max_finite": float(np.max(good)) if good.size else None,
        "max_abs_finite": float(np.max(np.abs(good))) if good.size else None,
        "max_abs_flat_index": int(np.argmax(np.where(finite, np.abs(array), -1))) if array.size else None,
    }


def state_summary(state):
    result = {}
    for name in ("density", "pressure", "temperature", "Er"):
        value = getattr(state, name, None)
        if value is not None:
            result[name] = array_summary(value)
            if name != "Er" and np.ndim(value) == 2:
                result[name]["species"] = [array_summary(row) for row in np.asarray(jax.device_get(value))]
    return result


def diagnose_saved_step(metadata, baseline, endpoint, baseline_prepared, endpoint_prepared, dt, force_rebuild):
    """Run only the saved stage equations, reusing each context's compiled step."""

    from NEOPAX._transport_solvers import (
        _RadauAcceptedStepAttemptContext,
        _execute_radau_accepted_step_attempt,
        _radau_carry_with_forward_only_jvp_fields,
    )

    def compiled_step_for(prepared):
        def step(carry):
            return _execute_radau_accepted_step_attempt(
                prepared.kernel_context, prepared.physics_context,
                _radau_carry_with_forward_only_jvp_fields(carry),
                _RadauAcceptedStepAttemptContext(
                    t_final=carry.t + dt,
                    use_transport_lagged_response=jnp.asarray(
                        prepared.kernel_context.use_transport_lagged_response
                    ),
                ),
            )
        return jax.jit(step)

    baseline_step = compiled_step_for(baseline_prepared)
    endpoint_step = compiled_step_for(endpoint_prepared)
    baseline = dataclasses.replace(
        baseline, dt=dt, cache_valid=baseline.cache_valid & ~force_rebuild,
    )
    endpoint = dataclasses.replace(
        baseline, t=endpoint.t, y=endpoint.y, dt=dt,
        lagged_response_cache=endpoint.lagged_response_cache,
        lagged_reference_y=endpoint.lagged_reference_y,
    )
    result = {
        "failure": metadata["failure"], "endpoint_label": metadata["endpoint_label"],
        "incoming_flat_state_difference": array_summary(endpoint.y - baseline.y),
        "trials": {},
    }
    for label, carry, prepared, compiled_step in (
        ("baseline_recorded_setup", baseline, baseline_prepared, baseline_step),
        ("endpoint_recorded_setup", endpoint, endpoint_prepared, endpoint_step),
        ("endpoint_cold_setup", cold_carry(endpoint), endpoint_prepared, endpoint_step),
    ):
        attempt = jax.block_until_ready(compiled_step(carry))
        rhs_value = jax.jit(prepared.physics_context.flat_rhs)(carry.t, carry.y)
        summary = {
            "t_start": float(carry.t), "dt": float(dt),
            "incoming_state": state_summary(prepared.physics_context.unpack_flat(carry.y)),
            "initial_rhs": array_summary(rhs_value),
            "trial_flat_state": array_summary(attempt.trial_y),
            "stage_history": array_summary(attempt.stage_history),
            "jacobian": array_summary(attempt.jacobian_out),
        }
        jacobian = np.asarray(jax.device_get(attempt.jacobian_out))
        real_matrix = (
            np.eye(jacobian.shape[0], dtype=jacobian.dtype)
            - float(prepared.kernel_context.radau_real_eig) * float(dt) * jacobian
        )
        try:
            condition = float(np.linalg.cond(real_matrix)) if np.all(np.isfinite(real_matrix)) else float("inf")
        except np.linalg.LinAlgError:
            condition = float("inf")
        summary["newton_real_matrix_condition"] = condition if np.isfinite(condition) else None
        summary["newton_real_matrix_condition_finite"] = bool(np.isfinite(condition))
        for name in (
            "converged", "newton_iter_count", "final_residual_norm", "final_delta_norm",
            "theta_final", "diverged_final", "slow_contraction_final", "residual_blowup_final",
            "newton_nonfinite_final", "finite_f0", "finite_z0", "finite_initial_residual",
            "nonfinite_stage_state", "nonfinite_stage_residual", "jacobian_reused",
        ):
            value = np.asarray(jax.device_get(getattr(attempt, name))).item()
            summary[name] = None if isinstance(value, float) and not np.isfinite(value) else value
        result["trials"][label] = summary
        print(f"[fd-saved-step] {label}: " + json.dumps(summary), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args(argv)
    with np.load(args.snapshot, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("schema_version") != 1:
            parser.error("Unsupported FD snapshot schema version.")
        parameter = metadata["endpoint_label"].rsplit(":", 1)[0]
        if parameter not in {
            "n0", "T0", "density_shape_power", "temperature_shape_power",
            "density_shape_alpha", "temperature_shape_alpha",
        }:
            parser.error("This saved-step diagnostic currently supports profile FD snapshots.")

        # The user runs plasma physics on a separate GPU machine. Refuse an
        # accidental local CPU reconstruction before building any runtime.
        try:
            device = jax.local_devices(backend="gpu")[0]
        except (RuntimeError, IndexError):
            parser.error("Run this diagnostic on the GPU machine; no runtime was built.")
        with jax.default_device(device):
            from benchmark_transport_realtime_geometry_forward_fd import _prepare_rollout
            from NEOPAX._orchestrator import build_runtime_context

            print("[fd-saved-step] rebuilding runtime only; no adaptive rollout or objectives", flush=True)
            runtime, state = build_runtime_context(metadata["config"])
            _, template, _ = _prepare_rollout(metadata["config"], runtime, state)
            baseline_initial_state = template.physics_context.unpack_flat(jnp.asarray(archive["baseline_initial_y"]))
            endpoint_initial_state = template.physics_context.unpack_flat(jnp.asarray(archive["endpoint_initial_y"]))
            _, baseline_prepared, _ = _prepare_rollout(metadata["config"], runtime, baseline_initial_state)
            _, endpoint_prepared, _ = _prepare_rollout(metadata["config"], runtime, endpoint_initial_state)
            baseline = restore_carry(archive, "baseline_carry", baseline_prepared.initial_carry, metadata)
            endpoint = restore_carry(archive, "endpoint_carry", endpoint_prepared.initial_carry, metadata)
            report = diagnose_saved_step(
                metadata, baseline, endpoint, baseline_prepared, endpoint_prepared,
                jnp.asarray(archive["dt"]), jnp.asarray(archive["force_jacobian_rebuild"]),
            )
    outpath = args.snapshot.with_suffix(".diagnosis.json")
    outpath.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"Wrote {outpath}", flush=True)


if __name__ == "__main__":
    main()
