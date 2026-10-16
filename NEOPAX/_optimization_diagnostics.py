"""Stream small host diagnostics from existing optimization/AD computations.

No physics replay, extra VJP, device-array history, or solver callback API is
needed. SciPy requests a Jacobian at its initial and accepted TRF points; the
adapter uses that boundary to distinguish accepted points from trial values.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import tempfile
import traceback

import numpy as np


_ACTIVE = ContextVar("neopax_optimization_diagnostics", default=None)


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return _json_value(np.asarray(value).tolist())
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def reverse_diagnostics_active():
    return _ACTIVE.get() is not None


def reverse_objective_values(values, labels):
    """Keep terminal transport values even if a subsequent reverse step fails."""
    active = _ACTIVE.get()
    if active is None:
        return
    import jax
    sink, trial = active
    scalars = np.asarray(jax.device_get(values), dtype=float)
    reported = {label: float(value) for label, value in zip(labels, scalars, strict=True)}
    sink.last_objectives = (trial, reported)
    sink.write("transport_objectives", trial=trial, objective_values=reported)


def reverse_checkpoint(stage, tree, *, objective_labels=(), summarize=False):
    """Inspect existing host-boundary values, transferring compact metadata.

    Reports the first bad index per leaf and affected objective rows, without
    raising, sanitizing cotangents, or retaining the tree. Disabled calls return
    before importing JAX or touching any array. These are boundary checks, not
    an instruction-level diagnosis of a compiled reverse step.
    """
    active = _ACTIVE.get()
    if active is None:
        return
    import jax
    import jax.numpy as jnp

    sink, trial = active
    entries, summaries = [], []
    labels = tuple(objective_labels)
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if leaf is None or not hasattr(leaf, "dtype"):
            continue
        array = jnp.asarray(leaf)
        if summarize and array.dtype == jnp.bool_:
            mask = np.asarray(jax.device_get(array), dtype=bool)
            summaries.append({"path": jax.tree_util.keystr(path), "shape": array.shape,
                              "true_count": int(np.count_nonzero(mask)),
                              "false_indices": np.argwhere(~mask)})
            continue
        if array.dtype == jax.dtypes.float0 or not jnp.issubdtype(array.dtype, jnp.inexact):
            continue
        if not array.size:
            continue
        finite = jnp.isfinite(array)
        # Reduce on device; do not copy database-sized masks to the host.
        if labels and array.ndim and array.shape[0] == len(labels):
            axes = tuple(range(1, array.ndim))
            good = jnp.all(finite, axis=axes) if axes else finite
        else:
            good = jnp.all(finite)
        good = np.asarray(jax.device_get(good), dtype=bool)
        name = jax.tree_util.keystr(path)
        if summarize:
            minimum, maximum = jax.device_get((
                jnp.min(jnp.where(finite, jnp.real(array), jnp.inf)),
                jnp.max(jnp.where(finite, jnp.real(array), -jnp.inf)),
            ))
            summary = {"path": name, "shape": array.shape,
                       "min_finite": float(minimum), "max_finite": float(maximum)}
            if array.ndim == 2:
                # State fields are species x radius; retain species-specific
                # minima, including an impurity sitting at its density floor.
                low, high, nonpositive = jax.device_get((
                    jnp.min(jnp.where(finite, jnp.real(array), jnp.inf), axis=1),
                    jnp.max(jnp.where(finite, jnp.real(array), -jnp.inf), axis=1),
                    jnp.sum(jnp.real(array) <= 0, axis=1),
                ))
                summary.update(min_by_leading_index=low, max_by_leading_index=high,
                               nonpositive_count_by_leading_index=nonpositive)
            summaries.append(summary)
        if bool(np.all(good)):
            continue
        flat_index, bad_count = jax.device_get((
            jnp.argmax((~finite).reshape(-1)), jnp.sum(~finite),
        ))
        entry = {"path": name, "shape": array.shape, "dtype": str(array.dtype),
                 "first_nonfinite_index": np.unravel_index(int(flat_index), array.shape),
                 "nonfinite_count": int(bad_count)}
        if good.ndim:
            entry["objectives"] = [labels[i] for i in np.flatnonzero(~good)]
        entries.append(entry)
    first_bad = bool(entries) and (sink.first_bad is None or sink.first_bad[0] != trial)
    if first_bad:
        sink.first_bad = (trial, stage)
    sink.write("reverse_checkpoint", trial=trial, stage=stage, first_bad_for_trial=first_bad,
               all_finite=not entries, nonfinite=entries, summaries=summaries)
    if first_bad:
        print(f"[optimization diagnostic] trial={trial} first bad reverse boundary "
              f"stage={stage} entries={json.dumps(_json_value(entries))}", flush=True)


class LeastSquaresDiagnostics:
    def __init__(self, directory, problem, *, bounds=(-np.inf, np.inf), trust_scale=1.0):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="run_", dir=root))
        n = int(problem.parameter_count)
        self.labels = tuple(getattr(problem, "parameter_labels", (f"parameter_{i}" for i in range(n))))
        self.physical_scale = np.broadcast_to(np.asarray(getattr(problem, "x_scale", 1.0), dtype=float), (n,)).copy()
        self.trust_scale = (None if isinstance(trust_scale, str) else
                            np.broadcast_to(np.asarray(trust_scale, dtype=float), (n,)).copy())
        self.profile = np.asarray([not label.startswith(("RBC:", "ZBS:", "vmec:")) for label in self.labels])
        if hasattr(bounds, "lb"):
            bounds = (bounds.lb, bounds.ub)
        self.lower = np.broadcast_to(np.asarray(bounds[0], dtype=float), (n,)).copy()
        self.upper = np.broadcast_to(np.asarray(bounds[1], dtype=float), (n,)).copy()
        self.anchor = None
        self.trial_count = 0
        self.accepted_trials = []
        self.last = None
        self.first_bad = None
        self.last_objectives = None
        self.write("setup", parameter_labels=self.labels, physical_scales=self.physical_scale,
                   trust_scales=self.trust_scale, lower_bounds=self.lower, upper_bounds=self.upper,
                   profile_coordinate_mode=getattr(problem, "profile_coordinate_mode", None),
                   baseline_profile_values=getattr(problem, "baseline_profile_values", None),
                   model_kind="linear weighted residual model; excludes bounded-TRF augmentation/regularization",
                   row_metric="fixed SciPy x_scale" if self.trust_scale is not None else "unit; adaptive x_scale not reconstructed",
                   acceptance_boundary="SciPy jacobian request; initial point is labeled separately",
                   precomputed_initial_reverse="not replayed; initial residual/Jacobian still recorded")
        print(f"[optimization diagnostic] writing {self.directory}", flush=True)

    def write(self, event, **values):
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(_json_value({"event": event, **values}), allow_nan=False) + "\n")

    @contextmanager
    def evaluation_scope(self):
        token = _ACTIVE.set((self, self.trial_count + 1))
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def _step_model(self, x):
        if self.anchor is None:
            return {}
        old_x, residual, jacobian, labels = self.anchor
        step = np.asarray(x, dtype=float) - old_x
        linear = jacobian @ step
        predicted_cost = 0.5 * (residual + linear) ** 2
        old_cost = 0.5 * residual ** 2
        return {"from_accepted_trial": self.accepted_trials[-1], "step": step,
                "physical_step": step * self.physical_scale,
                "geometry_step_norm": np.linalg.norm(step[~self.profile]),
                "profile_step_norm": np.linalg.norm(step[self.profile]),
                "trust_step_norm": None if self.trust_scale is None else np.linalg.norm(step / self.trust_scale),
                "predicted_cost": predicted_cost, "old_cost": old_cost,
                "predicted_reduction": float(np.sum(old_cost - predicted_cost)),
                "rows": [{"label": label, "linear_residual_change": linear[i],
                          "geometry_linear_change": jacobian[i, ~self.profile] @ step[~self.profile],
                          "profile_linear_change": jacobian[i, self.profile] @ step[self.profile],
                          "first_order_cost_change": residual[i] * linear[i]}
                         for i, label in enumerate(labels)]}

    def evaluation(self, x, evaluation):
        # Only small host residual/Jacobian arrays cross this boundary.
        residual = np.asarray(evaluation.residuals, dtype=float)
        jacobian = np.asarray(evaluation.jacobian, dtype=float)
        result = getattr(evaluation, "result", None)
        labels = tuple(getattr(result, "residual_labels", (f"residual_{i}" for i in range(residual.size))))
        self.trial_count += 1
        trial = self.trial_count
        snapshot = f"trial_{trial:04d}.npz"
        np.savez(self.directory / snapshot, x=np.asarray(x), residuals=residual, jacobian=jacobian,
                 residual_labels=np.asarray(labels), parameter_labels=np.asarray(self.labels),
                 physical_scales=self.physical_scale)
        finite = bool(np.all(np.isfinite(residual)) and np.all(np.isfinite(jacobian)))
        bad_residual = np.argwhere(~np.isfinite(residual))
        bad_jacobian = np.argwhere(~np.isfinite(jacobian))
        # Match the adapter's existing replacement policy in the model while
        # retaining the unmodified arrays and masks in the diagnostic files.
        residual = np.where(np.isfinite(residual), residual, 1e6)
        jacobian = np.where(np.isfinite(jacobian), jacobian, 0.0)
        model = self._step_model(x)
        costs = 0.5 * residual ** 2
        if model:
            actual = float(np.sum(model["old_cost"]) - np.sum(costs))
            predicted = model["predicted_reduction"]
            model.update(actual_reduction=actual,
                         actual_over_predicted=None if predicted == 0 else actual / predicted,
                         actual_cost_change=costs - model["old_cost"])
        rows = []
        if finite:
            gradient = residual[:, None] * jacobian
            total_gradient = gradient.sum(axis=0)
            metric = np.ones(jacobian.shape[1]) if self.trust_scale is None else self.trust_scale
            row_norms = np.linalg.norm(jacobian * metric, axis=1)
            gram = (jacobian * metric) @ (jacobian * metric).T
            denom = row_norms[:, None] * row_norms[None, :]
            cosine = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
            for i, label in enumerate(labels):
                rows.append({"label": label, "residual": residual[i], "cost": costs[i],
                             "jacobian_norm_geometry": np.linalg.norm(jacobian[i, ~self.profile]),
                             "jacobian_norm_profile": np.linalg.norm(jacobian[i, self.profile]),
                             "gradient_geometry": gradient[i, ~self.profile],
                             "gradient_profile": gradient[i, self.profile],
                             "gradient_norm_geometry": np.linalg.norm(gradient[i, ~self.profile]),
                             "gradient_norm_profile": np.linalg.norm(gradient[i, self.profile]),
                             "gradient_dot_total": gradient[i] @ total_gradient})
            geometry_s = np.linalg.svd(jacobian[:, ~self.profile], compute_uv=False)
            profile_s = np.linalg.svd(jacobian[:, self.profile], compute_uv=False)
            spectrum = np.linalg.svd(jacobian * metric, compute_uv=False)
        else:
            total_gradient, cosine, spectrum, geometry_s, profile_s = None, None, None, None, None
        self.write("trial", trial=trial, snapshot=snapshot, x=np.asarray(x), finite=finite,
                   nonfinite_residual_indices=bad_residual,
                   nonfinite_jacobian_indices=bad_jacobian,
                   cost=float(np.sum(costs)), objective_values=getattr(result, "objective_values", {}),
                   rows=rows, gradient=total_gradient, row_sensitivity_cosines=cosine,
                   singular_values=spectrum, geometry_singular_values=geometry_s,
                   profile_singular_values=profile_s, model=model)
        self.last = (np.asarray(x, dtype=float).copy(), residual.copy(), jacobian.copy(), labels, trial)

    def accepted(self, x):
        if self.last is None or not np.array_equal(x, self.last[0]):
            return
        if self.anchor is not None and np.array_equal(x, self.anchor[0]):
            return
        old = self.anchor
        point, residual, jacobian, labels, trial = self.last
        self.anchor = (point.copy(), residual.copy(), jacobian.copy(), labels)
        self.accepted_trials.append(trial)
        tolerance = 1e-10 * np.maximum(1.0, np.abs(point))
        self.write("initial" if old is None else "accepted", trial=trial, x=point,
                   lower_bound_near=np.flatnonzero(point - self.lower <= tolerance),
                   upper_bound_near=np.flatnonzero(self.upper - point <= tolerance))
        print(f"[optimization diagnostic] {'initial' if old is None else 'accepted'} "
              f"trial={trial} cost={0.5 * float(residual @ residual):.8e}", flush=True)

    def failure(self, x, error, *, phase, reason=None):
        self.trial_count += 1
        trial = self.trial_count
        details = ("".join(traceback.format_exception(type(error), error, error.__traceback__))
                   if error is not None else str(reason))
        filename = f"failed_trial_{trial:04d}.txt"
        (self.directory / filename).write_text(details, encoding="utf-8")
        self.write("failed_trial", trial=trial, x=np.asarray(x),
                   coordinate_times_physical_scale=np.asarray(x) * self.physical_scale,
                   phase=phase, exception_type=None if error is None else type(error).__name__,
                   reason=str(error) if error is not None else reason, traceback=filename,
                   first_nonfinite_boundary=(self.first_bad[1] if self.first_bad is not None
                                             and self.first_bad[0] == trial else None),
                   transport_objective_values=(self.last_objectives[1] if self.last_objectives is not None
                                               and self.last_objectives[0] == trial else None),
                   model=self._step_model(x))

    def finish(self, result):
        self.write("finished", x=getattr(result, "x", None), cost=getattr(result, "cost", None),
                   status=getattr(result, "status", None), message=str(getattr(result, "message", "")),
                   optimality=getattr(result, "optimality", None), active_mask=getattr(result, "active_mask", None),
                   nfev=getattr(result, "nfev", None), njev=getattr(result, "njev", None),
                   accepted_trials=self.accepted_trials)
