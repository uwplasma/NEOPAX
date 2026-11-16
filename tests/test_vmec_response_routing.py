"""Exercise production routing with tiny arrays; no JAX or physics imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "NEOPAX/_geometry_autodiff.py"
AD = ROOT / "examples/benchmarks/benchmark_transport_reverse_ad_only.py"
FD = ROOT / "examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py"


def load_function(path, name, namespace):
    node = next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def numpy_jax():
    return SimpleNamespace(core=SimpleNamespace(Tracer=type("Tracer", (), {})),
        device_get=lambda value: value,
        tree_util=SimpleNamespace(tree_map=lambda fn, value: fn(value)),
        lax=SimpleNamespace(map=lambda fn, values: np.stack([fn(v) for v in values])))


class MainResponseRoutingTests(unittest.TestCase):
    def setUp(self):
        self.ns = dict(jnp=np, jax=numpy_jax())
        for name in ("normalize_vmec_response_mode", "_main_vmec_state_tangent_batch",
                     "_main_vmec_state_pullback_batch"):
            load_function(GEOMETRY, name, self.ns)
        self.params, self.cfg, self.state, self.mask = (object() for _ in range(4))

    def tangent_backend(self, converged=True):
        public = Mock(side_effect=lambda *args, **kwargs:
                      (2*args[4], SimpleNamespace(converged=np.array([converged]))))
        raw = Mock(side_effect=AssertionError("main must not call the raw API"))
        return SimpleNamespace(implicit_state_tangent_multi_rhs=public,
                               implicit_state_tangent_raw_block=raw)

    def test_main_factors_once_for_all_directions_and_preserves_primal(self):
        backend = self.tangent_backend()
        directions = np.arange(6.).reshape(2, 3)
        result = self.ns["_main_vmec_state_tangent_batch"](
            backend, self.params, self.cfg, self.state, self.mask, directions)
        np.testing.assert_array_equal(result, 2*directions)
        backend.implicit_state_tangent_multi_rhs.assert_called_once_with(
            self.params, self.cfg, self.state, self.mask, directions,
            probe_chunk_size=1, response_chunk_size=1)
        backend.implicit_state_tangent_raw_block.assert_not_called()

    def test_failed_main_certificate_cannot_be_used(self):
        with self.assertRaisesRegex(FloatingPointError, "convergence certificate"):
            self.ns["_main_vmec_state_tangent_batch"](
                self.tangent_backend(False), self.params, self.cfg,
                self.state, self.mask, np.ones((2, 3)))
        self.ns["jax"].core.Tracer = np.bool_
        result = self.ns["_main_vmec_state_tangent_batch"](
            self.tangent_backend(False), self.params, self.cfg,
            self.state, self.mask, np.ones((2, 3)))
        self.assertTrue(np.isnan(result).all())

    def test_pure_geometry_transpose_uses_public_block_api(self):
        public = Mock(return_value=np.array([[3., 4.], [5., 6.]]))
        backend = SimpleNamespace(implicit_state_pullback_multi_rhs=public)
        raw = SimpleNamespace(implicit=backend, implicit_params=self.params,
            implicit_cfg=self.cfg, state=self.state, dof_mask=self.mask,
            param_entries=("RBC", "ZBS"), vmec_response_mode="main")
        self.ns["_param_vector_gradient_from_implicit_param_grads"] = lambda bars, entries: bars
        function = load_function(GEOMETRY, "geometry_raw_block_transpose_from_state_bars", self.ns)
        bars = np.ones((2, 3))
        np.testing.assert_array_equal(function(raw, bars), public.return_value)
        public.assert_called_once_with(self.params, self.cfg, self.state, self.mask,
            bars, solver="block", probe_chunk_size=1, response_chunk_size=1)

    def test_transport_payload_contracts_every_main_direction(self):
        backend = self.tangent_backend()
        directions = np.array([[1., 2., 3.], [4., 5., 6.]])
        cotangents = np.array([[1., -1., 2.], [3., 0., -2.]])
        self.ns.update(combined_payload=True, return_branch_gradients=False,
            return_state_bars=False, extra_state_bars=None, extra_state_bars_factory=None,
            implicit=backend, response_mode="main", payload_kind="ntx_scan_runtime",
            _probe=lambda *args: None, support_branch_name="scan", support_from_state=None,
            payload_bars=({}, {}), native_vmec_face_coefficient_bars=None,
            param_entries=("RBC", "ZBS"), _implicit_param_tangent_batch=lambda: directions,
            _payload_branch_pullback_setup=lambda *args: dict(active=False,
                tangent_contraction=lambda tangent: cotangents @ tangent),
            os=SimpleNamespace(environ={}), progress_label=None,
            implicit_params=self.params, implicit_cfg=self.cfg, state=self.state, dof_mask=self.mask)
        function = load_function(GEOMETRY,
            "_try_compact_payload_tangent_contract_to_param_matrix", self.ns)
        np.testing.assert_array_equal(function(), cotangents @ (2*directions).T)
        backend.implicit_state_tangent_multi_rhs.assert_called_once()
        backend.implicit_state_tangent_raw_block.assert_not_called()
        self.ns["response_mode"] = "raw_block"
        backend.implicit_state_tangent_raw_block.side_effect = lambda *args, **kw: 2*args[4]
        np.testing.assert_array_equal(function(), cotangents @ (2*directions).T)
        self.assertEqual(backend.implicit_state_tangent_raw_block.call_count, 2)
        self.assertEqual(backend.implicit_state_tangent_multi_rhs.call_count, 1)

    def test_fd_uses_the_same_main_tangent_and_retains_raw_default(self):
        backend = self.tangent_backend()
        self.ns["im"] = backend
        function = load_function(FD, "_manual_implicit_forward_state_tangent", self.ns)
        direction = np.array([1., 2., 3.])
        args = dict(params=self.params, param_tangent=direction, cfg=self.cfg,
                    x_star=self.state, dof_mask=self.mask)
        np.testing.assert_array_equal(function(**args, vmec_response_mode="main"), 2*direction)
        backend.implicit_state_tangent_raw_block.assert_not_called()
        backend.implicit_state_tangent_raw_block.side_effect = lambda *args, **kw: 2*args[4]
        np.testing.assert_array_equal(function(**args), 2*direction)
        backend.implicit_state_tangent_raw_block.assert_called_once()

    def test_mode_validation(self):
        self.assertEqual(self.ns["normalize_vmec_response_mode"](" MAIN "), "main")
        with self.assertRaises(ValueError):
            self.ns["normalize_vmec_response_mode"]("auto")

    def test_ad_fd_cli_and_context_share_the_choice(self):
        for path in (AD, FD):
            tree = ast.parse(path.read_text())
            option = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
                and any(isinstance(a, ast.Constant) and a.value == "--vmec-response-mode"
                        for a in n.args))
            keywords = {k.arg: ast.literal_eval(k.value) for k in option.keywords}
            self.assertEqual(keywords["choices"], ("raw_block", "main"))
            self.assertIsNone(keywords["default"])
            self.assertIn('config.setdefault("geometry", {})["vmec_response_mode"] = args.vmec_response_mode',
                          path.read_text())
        for name in ("examples/benchmarks/benchmark_transport_reverse_ad_only.py",
                     "NEOPAX/_orchestrator.py", "NEOPAX/optimization.py"):
            tree = ast.parse((ROOT/name).read_text())
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "build_geometry_autodiff_context"]
            for call in calls:
                if name.endswith("optimization.py") and any(
                        k.arg == "vmec_input" for k in call.keywords):
                    continue
                # The geometry-only builder has no transport config; keep its default.
                if name.endswith("optimization.py") and isinstance(call.args[0], ast.Name) \
                        and call.args[0].id == "vmec_input":
                    continue
                self.assertIn("vmec_response_mode", [k.arg for k in call.keywords])

    def test_staged_and_saved_solve_reconstruction_preserve_the_response(self):
        for name in ("NEOPAX/_optimization_initial_root_stage.py",
                     "examples/benchmarks/check_saved_compact_geometry_gradient.py"):
            tree = ast.parse((ROOT/name).read_text())
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and ((isinstance(n.func, ast.Name) and n.func.id == "GeometryRawBlockSolve")
                          or (isinstance(n.func, ast.Attribute)
                              and n.func.attr == "GeometryRawBlockSolve"))]
            self.assertTrue(calls)
            for call in calls:
                self.assertIn("vmec_response_mode", [k.arg for k in call.keywords])
        replay = (ROOT / "examples/benchmarks/check_saved_compact_geometry_gradient.py").read_text()
        self.assertIn('captured.get("vmec_response_mode", "raw_block")', replay)


if __name__ == "__main__":
    unittest.main()
