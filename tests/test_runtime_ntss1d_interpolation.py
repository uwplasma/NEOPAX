"""Routing checks without JAX; opt-in small GPU kernel/derivative checks.

NEOPAX_TEST_RUNTIME_NTSS1D=1 enables GPU checks, without any NTX, VMEC or
transport solves. The full transport comparison remains a separate benchmark.
"""
import ast
import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace
import tomllib
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODE = "preprocessed_3d_radial_ntss1d"
BASE = "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_"


def definition(path, name, namespace):
    node = next(n for n in ast.walk(ast.parse((ROOT/path).read_text()))
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def numpy_namespace():
    ns = dict(jnp=np, dataclasses=dataclasses,
        jax=SimpleNamespace(tree_util=SimpleNamespace(register_dataclass=lambda cls: cls)),
        normalize_interpolation_mode=lambda value: str(value or "generic").strip().lower(),
        MONOENERGETIC_KIND_GENERIC="generic",
        MONOENERGETIC_KIND_PREPROCESSED_3D_RADIAL_NTSS1D=MODE, D11_POSITIVE_FLOOR=1e-20)
    for name in ("Monoenergetic", "MonoenergeticRuntimeNTSS1D"):
        definition("NEOPAX/_database.py", name, ns)
    definition("NEOPAX/_transport_flux_models.py", "_ntx_runtime_scan_to_neopax_monoenergetic", ns)
    return ns


def scan_arrays(array_module=np):
    rho = array_module.asarray([.12, .3, .5, .7, .9])
    nu = array_module.asarray([1e-5, 1e-4, 1e-3])
    er = array_module.tile(array_module.asarray([1e-5, 5e-5, 2e-4, 1e-3, 5e-3]), (5, 1))
    i = array_module.arange(75.).reshape(5, 3, 5)
    return SimpleNamespace(rho=rho, nu_v=nu, Er=er, drds=1+rho,
        D11=1e-3*(1+i/80), D13=.01*array_module.sin(i/8), D33=2+i/100)


class RuntimeNTSS1DRoutingTests(unittest.TestCase):
    def test_live_views_match_the_existing_ntss1d_kernel_with_numpy(self):
        ns = numpy_namespace()
        ns["jax"].jit = lambda function: function
        ns["jax"].lax = SimpleNamespace(cond=lambda flag, yes, no: (yes if flag else no)())
        ns["jax"].vmap = lambda function: lambda values: np.stack([function(v) for v in values])
        ns["contract_scalar_queries"] = lambda **kwargs: lambda function: function
        ns["NTSS1D_WIDTH"] = 16
        definition("NEOPAX/_interpolators_ntss_preprocessed.py", "_inpold_fixed", ns)
        for name in ("_clamped_interval_index", "_fraction", "_lagrange3", "_surface_ntss1d",
                     "get_Dij_preprocessed_3d_ntss_radius_ntss1d"):
            definition("NEOPAX/_interpolators_preprocessed.py", name, ns)
        for name in ("PreprocessedMonoenergetic3DNTSSRadius", "PreprocessedMonoenergetic3DNTSSRadiusNTSS1D",
                     "_add_ntss1d_padding"):
            definition("NEOPAX/_database_preprocessed.py", name, ns)
        db = ns["_ntx_runtime_scan_to_neopax_monoenergetic"](scan_arrays(), a_b=1.7, interpolation_mode=MODE)
        data = {k: getattr(db, k) for k in ("a_b", "rho", "r_grid", "nu_log", "Er_grid", "D11_log",
                "D13", "D33", "Er_lower_limit", "low_limit_r", "del_r")}
        reference = ns["PreprocessedMonoenergetic3DNTSSRadiusNTSS1D"](**ns["_add_ntss1d_padding"](data))
        kernel = ns["get_Dij_preprocessed_3d_ntss_radius_ntss1d"]
        with np.errstate(all="ignore"):
            for r in (.05, .51, .79, 1.6):
                for nu in (1e-6, 4e-5, 2e-3):
                    for er in (0., 1e-4, .1):
                        value = kernel(r, nu, er, db)
                        self.assertTrue(np.isfinite(value).all())
                        np.testing.assert_allclose(value, kernel(r, nu, er, reference), rtol=0., atol=0.)

    def test_only_interpolation_changes_in_the_comparison_toml(self):
        folder = ROOT/"examples/benchmarks"
        original = tomllib.loads((folder/(BASE+"black_box.toml")).read_text())
        selected = tomllib.loads((folder/(BASE+"ntss1d.toml")).read_text())
        self.assertEqual(selected["neoclassical"].pop("interpolation_mode"), MODE)
        self.assertEqual(original, selected)

    def test_conversion_retains_scan_leaves_and_exposes_ntss_coordinates(self):
        ns, scan = numpy_namespace(), scan_arrays()
        convert = ns["_ntx_runtime_scan_to_neopax_monoenergetic"]
        raw = convert(scan, a_b=1.7)
        db = convert(scan, a_b=1.7, interpolation_mode=MODE)
        self.assertIs(type(raw), ns["Monoenergetic"])
        self.assertIs(type(db), ns["MonoenergeticRuntimeNTSS1D"])
        self.assertEqual(len(dataclasses.fields(db)), 19)
        for field in dataclasses.fields(raw):
            np.testing.assert_array_equal(getattr(raw, field.name), getattr(db, field.name))
        np.testing.assert_array_equal(db.r_grid, 1.7*scan.rho)
        np.testing.assert_array_equal(db.Er_grid_ntss1d[:, :5], raw.Er_list)
        self.assertTrue(np.isinf(db.Er_grid_ntss1d[:, 5:]).all())
        self.assertEqual(db.er_count_ntss1d, 5)
        self.assertEqual(db.gmix_er_ntss1d, .1)
        self.assertEqual(db.del_r, .001)
        with self.assertRaises(ValueError):
            convert(scan, a_b=1.7, interpolation_mode="unsupported")
        changed = dataclasses.replace(db, a_b=2., Er_list=db.Er_list+.2)
        self.assertIs(type(changed), type(db))
        np.testing.assert_array_equal(changed.r_grid, 2*scan.rho)
        np.testing.assert_array_equal(changed.Er_grid_ntss1d[:, :5], db.Er_list+.2)

    def test_recorded_scan_transposes_preserve_the_interpolation_choice(self):
        tree = ast.parse((ROOT/"NEOPAX/_transport_flux_models.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "NTXRuntimeScanTransportModel")
        calls = [n for n in ast.walk(cls) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id == "_ntx_runtime_scan_to_neopax_monoenergetic"]
        self.assertEqual(len(calls), 3)
        for call in calls:
            value = next(k.value for k in call.keywords if k.arg == "interpolation_mode")
            self.assertEqual(ast.unparse(value), "self.interpolation_mode")

    def test_all_three_table_transposes_use_ntss1d_rule(self):
        for name in ("NEOPAX/_neoclassical.py", "NEOPAX/_transport_flux_models.py"):
            tree = ast.parse((ROOT/name).read_text())
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "ntss1d_interpolation_table_bar"]
            self.assertEqual(len(calls), 2 if name.endswith("_neoclassical.py") else 1)
        ns = numpy_namespace()
        function = definition("NEOPAX/_transport_flux_models.py", "_legacy_sparse_support_bar", ns)
        db = ns["_ntx_runtime_scan_to_neopax_monoenergetic"](scan_arrays(), a_b=1.7, interpolation_mode=MODE)
        with self.assertRaisesRegex(ValueError, "established"):
            function(db, {}, None)


@unittest.skipUnless(os.environ.get("NEOPAX_TEST_RUNTIME_NTSS1D") == "1", "opt-in GPU checks")
class RuntimeNTSS1DGPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        if jax.default_backend() != "gpu":
            raise RuntimeError("These opt-in checks require the GPU; no CPU fallback.")
        from NEOPAX._transport_flux_models import _ntx_runtime_scan_to_neopax_monoenergetic
        from NEOPAX._monoenergetic_interpolators import monoenergetic_interpolation_kernel, ntss1d_interpolation_table_bar
        from NEOPAX._monoenergetic import monoenergetic_database_kind
        cls.jax, cls.jnp = jax, jnp
        cls.convert = staticmethod(_ntx_runtime_scan_to_neopax_monoenergetic)
        cls.kernel = staticmethod(monoenergetic_interpolation_kernel)
        cls.table_bar = staticmethod(ntss1d_interpolation_table_bar)
        cls.kind = staticmethod(monoenergetic_database_kind)
        cls.scan = scan_arrays(jnp)
        cls.db = cls.convert(cls.scan, a_b=1.7, interpolation_mode=MODE)

    def test_primal_matches_existing_preprocessed_kernel(self):
        from NEOPAX._database_preprocessed import PreprocessedMonoenergetic3DNTSSRadiusNTSS1D, _add_ntss1d_padding
        data = {k: getattr(self.db, k) for k in ("a_b", "rho", "r_grid", "nu_log", "Er_grid",
            "D11_log", "D13", "D33", "Er_lower_limit", "low_limit_r", "del_r")}
        reference = PreprocessedMonoenergetic3DNTSSRadiusNTSS1D(**_add_ntss1d_padding(data))
        self.assertEqual(self.kind(self.db), MODE)
        for r in (.05, .51, .85, 1.6):
            for er in (0., 1e-4, .1):
                actual = self.kernel(self.db)(r, 4e-5, er, self.db)
                self.assertTrue(np.isfinite(np.asarray(actual)).all())
                np.testing.assert_allclose(actual,
                    self.kernel(reference)(r, 4e-5, er, reference), rtol=1e-12, atol=1e-12)
        self.assertEqual(len(self.jax.tree_util.tree_leaves(self.db)), 19)

    def test_coefficient_table_pullbacks_agree_with_fd(self):
        db, jax, jnp = self.db, self.jax, self.jnp
        for i, field in enumerate(("D11_log", "D13", "D33")):
            table = getattr(db, field)
            direction = jnp.cos(jnp.arange(table.size)).reshape(table.shape) * jnp.maximum(jnp.max(jnp.abs(table)), 1e-8)
            for r, nu, er in ((.79, 4e-5, 1e-4), (.05, 1e-6, 0.), (1.6, 2e-3, .1)):
                bar = self.table_bar(r, nu, er, .7, i, db)
                def objective(epsilon):
                    value = dataclasses.replace(db, **{field: table+epsilon*direction})
                    return .7*self.kernel(value)(r, nu, er, value)[i]
                h = 1e-6
                fd = (objective(h)-objective(-h))/(2*h)
                self.assertTrue(np.isfinite(np.asarray(bar)).all() and np.isfinite(float(fd)))
                np.testing.assert_allclose(jnp.sum(bar*direction), fd, rtol=2e-5, atol=2e-7)

    def test_recorded_scan_conversion_and_coordinate_chain_agree_with_fd(self):
        jax, jnp, scan = self.jax, self.jnp, self.scan
        def objective(epsilon):
            changed = SimpleNamespace(rho=scan.rho, nu_v=scan.nu_v*jnp.exp(.2*epsilon),
                Er=scan.Er*(1+.03*epsilon), drds=scan.drds*(1+.07*epsilon),
                D11=scan.D11*(1+.1*epsilon), D13=scan.D13+.01*epsilon,
                D33=scan.D33*(1+.05*epsilon))
            db = self.convert(changed, a_b=1.7*(1+.02*epsilon), interpolation_mode=MODE)
            return jnp.dot(jnp.array([.7, -.4, 1.2]), self.kernel(db)(.79, 4e-5, 1e-4, db))
        h = 1e-6
        ad = jax.grad(objective)(jnp.array(0.))
        fd = (objective(h)-objective(-h))/(2*h)
        self.assertTrue(np.isfinite(float(ad)) and np.isfinite(float(fd)))
        np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
