"""Audit callback/chain-rule tests. These do not run the plasma model."""

from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from NEOPAX import _reverse_ad_transport as reverse
from examples.benchmarks import _er_root_loss_audit as audit


LABELS = audit.AUDIT_OBJECTIVES


class PhysicalRootLossAuditTests(unittest.TestCase):
    def test_joint_parameter_chain_includes_er_and_direct_geometry_grid_derivatives(self):
        settings = audit.RootLossAuditSettings()
        callback, _, _ = audit.make_audit_callbacks(LABELS, settings)
        mesh = (np.arange(51) + 0.5) / 51

        def objectives(p):
            rho = jnp.asarray(mesh) * (1 + 0.02 * p[3])
            er = (1 + 0.1 * p[1]) * 500 * (0.55 + 0.001 * p[0] + 0.003 * p[2] - jnp.asarray(mesh))
            state, runtime = SimpleNamespace(Er=er), SimpleNamespace(geometry=SimpleNamespace(rho_grid=rho))
            return jnp.stack([callback(state, runtime, LABELS.index(name)) for name in audit.AUDIT_OBJECTIVES])

        def independent(p):
            rho = mesh * (1 + 0.02 * p[3])
            er = (1 + 0.1 * p[1]) * 500 * (0.55 + 0.001 * p[0] + 0.003 * p[2] - mesh)
            interpolated = np.interp(settings.target, rho, er)
            left = np.searchsorted(rho, settings.target) - 1
            right = left + 1
            positive = lambda x: 0.5 * (x + np.hypot(x, 1e-6))
            local = 0.5 * np.mean(positive(0.9 - np.array([er[left], -er[right]]) / 2)**2)
            signed = (settings.target - rho) / np.hypot(settings.target - rho, np.mean(np.diff(rho)) / 2)
            mask = (rho >= settings.inner_radius) & (rho <= settings.outer_radius)
            regional = 0.5 * np.mean(positive(0.9 * signed[mask]**2 - signed[mask] * er[mask] / 2)**2)
            zero = 0.5 * (interpolated / 2)**2
            true_chord = (0.55 + 0.001 * p[0] + 0.003 * p[2]) * (1 + 0.02 * p[3])
            return np.array([interpolated, zero + local, zero + regional, true_chord])

        point = np.zeros(4)
        ad = np.asarray(jax.jacrev(objectives)(jnp.asarray(point)))
        fd = np.column_stack([(independent(point + np.eye(4)[i] * 1e-5)
                               - independent(point - np.eye(4)[i] * 1e-5)) / 2e-5 for i in range(4)])
        np.testing.assert_allclose(objectives(jnp.asarray(point)), independent(point), rtol=1e-12)
        np.testing.assert_allclose(ad, fd, rtol=2e-6, atol=2e-7)
        self.assertGreater(abs(ad[2, 0]), 0.01)
        self.assertGreater(abs(ad[2, 2]), 0.01)
        self.assertAlmostEqual(ad[3, 3], 0.011, places=12)

    def test_observer_keeps_one_copied_host_snapshot_and_no_ad_graph(self):
        callback, observe, snapshot = audit.make_audit_callbacks(LABELS, audit.RootLossAuditSettings())
        rho = np.linspace(0.01, 0.99, 51)
        er = 100 * (0.55 - rho)
        runtime = SimpleNamespace(geometry=SimpleNamespace(rho_grid=rho))
        observe(SimpleNamespace(Er=er), runtime, 20.0)
        er[:] = -99
        self.assertFalse(np.all(snapshot["Er"] == -99))
        for value in (0.1, 0.2):
            state = SimpleNamespace(Er=jnp.asarray(100 * (0.55 - rho) + value))
            observe(state, runtime, 20.0)
            jax.grad(lambda field: callback(SimpleNamespace(Er=field), runtime,
                                            LABELS.index("audit_root_annular_cost")))(state.Er)
        self.assertEqual(set(snapshot), {"Er", "rho", "final_time"})
        self.assertIsInstance(snapshot["Er"], np.ndarray)
        self.assertIsInstance(snapshot["rho"], np.ndarray)
        self.assertEqual(snapshot["final_time"], 20.0)

    def test_missing_crossing_has_finite_diagnostic_and_explicitly_unavailable_directions(self):
        settings = audit.RootLossAuditSettings()
        er, rho = jnp.ones(51) * 10, (jnp.arange(51) + 0.5) / 51
        self.assertEqual(float(audit.nearest_downcrossing(er, rho, settings)), settings.target)
        np.testing.assert_array_equal(jax.grad(audit.nearest_downcrossing)(er, rho, settings), 0)
        report = self._summary(er=np.asarray(er), candidate_sign=1.0)
        self.assertFalse(report["nearest_downcrossing_available"])
        self.assertIsNone(report["nearest_downcrossing"])
        for item in report["comparisons"]:
            self.assertIsNone(item["groups"]["profiles"]["predicts_position_improvement"])

    @staticmethod
    def _summary(*, er=None, candidate_sign=1.0):
        settings = audit.RootLossAuditSettings()
        rho = (np.arange(51) + 0.5) / 51
        er = 500 * (0.55 - rho) if er is None else er
        callback, observe, snapshot = audit.make_audit_callbacks(LABELS, settings)
        state, runtime = SimpleNamespace(Er=jnp.asarray(er)), SimpleNamespace(geometry=SimpleNamespace(rho_grid=jnp.asarray(rho)))
        observe(state, runtime, 20.0)
        values = np.zeros(len(LABELS))
        for name in LABELS:
            values[LABELS.index(name)] = float(callback(state, runtime, LABELS.index(name)))
        jac = np.ones((len(LABELS), 2))
        jac[LABELS.index("audit_root_annular_cost")] = candidate_sign
        return audit.summarize_audit(
            residual_labels=["transport:" + name for name in LABELS], parameter_labels=["n0", "vmec:RBC:1:0"],
            values=values, jacobian=jac, profile_cfg={"n0": 4.21}, settings=settings, snapshot=snapshot,
        )

    def test_direction_check_distinguishes_root_progress_from_large_gradient_magnitude(self):
        for sign in (1.0, -1.0):
            report = self._summary(candidate_sign=sign)
            candidate = next(c for c in report["comparisons"] if c["name"] == "audit_root_annular_cost")
            for group in ("profiles", "geometry"):
                self.assertEqual(candidate["groups"][group]["predicts_position_improvement"], sign > 0)
                self.assertGreater(candidate["groups"][group]["scaled_gradient_l2"], 0)

    def test_internal_builder_preserves_custom_callbacks_without_running_a_solver(self):
        settings = audit.RootLossAuditSettings()
        callback, observer, _ = audit.make_audit_callbacks(LABELS, settings)
        context = reverse.RealtimeGeometryTransportReverseTableContext(
            config={}, baseline_values=jnp.ones(6), baseline_runtime=None, baseline_state=None,
            profile_cfg={}, neoclassical_cfg={},
        )
        builder = reverse.internal_realtime_geometry_transport_reverse_table_result_builder(
            table_context=context, geometry_context=object(), objective_labels=LABELS,
            objective_scalar_evaluator=callback, terminal_state_observer=observer,
        )
        closure = dict(zip(builder.__code__.co_freevars, [cell.cell_contents for cell in builder.__closure__], strict=True))
        dependencies = closure["configured_objective_dependencies"]
        self.assertIs(dependencies.terminal_state_observer, observer)
        rho = (jnp.arange(51) + 0.5) / 51
        state, runtime = SimpleNamespace(Er=500 * (0.55 - rho)), SimpleNamespace(geometry=SimpleNamespace(rho_grid=rho))
        np.testing.assert_allclose(
            dependencies.objective_scalar_by_index(state, runtime, LABELS.index("audit_root_annular_cost")),
            callback(state, runtime, LABELS.index("audit_root_annular_cost")),
        )

    def test_scales_match_nominal_profiles_and_shared_ess_with_explicit_trust_multiplier(self):
        from NEOPAX._reverse_ad_parameters import parse_vmec_boundary_parameter_spec, vmex_boundary_parameter_scales
        settings = audit.RootLossAuditSettings(profile_trust_multiplier=3.7, geometry_ess_alpha=1.2)
        labels = ("n0", "T0", "vmec:RBC:1:0", "vmec:ZBS:2:1")
        actual = audit.parameter_scales(labels, {"n0": 4.21, "T0": [17.8]}, settings)
        expected = [4.21 * 3.7, 17.8 * 3.7, *vmex_boundary_parameter_scales(
            tuple(parse_vmec_boundary_parameter_spec(p) for p in labels[2:]), ess_alpha=1.2)]
        np.testing.assert_allclose(actual, expected)

    def test_metrics_and_callback_reject_unresolved_band_and_existing_costs(self):
        with self.assertRaisesRegex(ValueError, "only new"):
            audit.make_audit_callbacks(("softmax_Er",), audit.RootLossAuditSettings())
        with self.assertRaisesRegex(ValueError, "resolve both sides"):
            audit.terminal_metrics(np.ones(3), np.array([0.1, 0.5, 0.9]),
                                   audit.RootLossAuditSettings(inner_radius=0.513, outer_radius=0.515))

    def test_descent_projection_reports_zero_root_motion_for_amplitude_only_control(self):
        settings = audit.RootLossAuditSettings()
        rho = (jnp.arange(51) + 0.5) / 51
        callback, observe, snapshot = audit.make_audit_callbacks(LABELS, settings)
        runtime = SimpleNamespace(geometry=SimpleNamespace(rho_grid=rho))

        def objective(amplitude):
            state = SimpleNamespace(Er=amplitude * 10 * jnp.tanh((0.65 - rho) / 0.02))
            return jnp.stack([callback(state, runtime, i) for i in range(len(LABELS))])

        er = 10 * jnp.tanh((0.65 - rho) / 0.02)
        observe(SimpleNamespace(Er=er), runtime, 20.0)
        summary = audit.summarize_audit(
            residual_labels=["transport:" + name for name in LABELS], parameter_labels=["n0"],
            values=objective(1.0), jacobian=np.asarray(jax.jacfwd(objective)(1.0))[:, None],
            profile_cfg={"n0": 1.0}, settings=settings, snapshot=snapshot,
        )
        for candidate in summary["comparisons"]:
            profiles = candidate["groups"]["profiles"]
            self.assertGreater(profiles["scaled_gradient_l2"], 0)
            self.assertAlmostEqual(profiles["predicted_root_change_per_unit_descent"], 0, places=12)

    def test_concrete_experiment_inputs_can_roundtrip_paths_and_numpy(self):
        from pathlib import Path
        result = audit.experiment_inputs({"geometry": {"vmec_input_file": Path("seed")}}, np.arange(8), "jax_selected_root")
        self.assertEqual(result["resolved_config"]["geometry"]["vmec_input_file"], "seed")
        self.assertEqual(result["parameter_values"], list(range(8)))


if __name__ == "__main__":
    unittest.main()
