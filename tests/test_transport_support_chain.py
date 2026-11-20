"""Saved-array chain-rule diagnostics only; no JAX or physics imports."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/'examples/benchmarks/compare_transport_gradient_audit.py'
spec = importlib.util.spec_from_file_location('support_chain_check', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class Archive(dict):
    @property
    def files(self):
        return list(self)


class SupportChainTests(unittest.TestCase):
    def case(self):
        # S(p)=(p,p^2), y(S)=S0*S1, L(y,S)=y^2+3*S0+2*S1.
        # The analytic total derivative is 6*p^5+3+4*p.
        p, h = .7, 1e-5
        def endpoint(x):
            y = x**3
            return Archive(initial_y=np.zeros(1), final_y=np.asarray([y]),
                final_time=np.asarray(.1), values=np.asarray([y*y+3*x+2*x*x]),
                **{'labels[0]':np.asarray('objective'),
                   "fixed_inputs['species'].charge_qp":np.asarray([-1., 1.]),
                   "fixed_inputs['energy_grid'].x":np.asarray([1., 2.]),
                   "support['geometry'].scale":np.asarray(x),
                   "support['database'].D13":np.asarray(x*x)})
        base, minus, plus = endpoint(p), endpoint(p-h), endpoint(p+h)
        ad = Archive(base, final_y_bars=np.asarray([[2*p**3]]), **{
            "explicit_support_bars[0]['geometry'].scale":np.asarray(3.),
            "explicit_support_bars[0]['database'].D13":np.asarray(2.)})
        fold = Archive(**{'labels[0]':np.asarray('objective'),
            "support_bars[0]['geometry'].scale":np.asarray(2*p**5+3),
            "support_bars[0]['database'].D13":np.asarray(2*p**4+2)})
        report = dict(objective_labels=['objective'], objective_values=base['values'].tolist(),
            fd_step=h, gradient_fd=((plus['values']-minus['values'])/(2*h)).tolist(),
            baseline_geometry_final_state_fd=((plus['final_y']**2-minus['final_y']**2)/(2*h)).tolist(),
            parameter_kind='realtime_geometry')
        return ad, fold, base, minus, plus, report

    def test_nonlinear_total_derivative_matches_both_saved_chains(self):
        case = self.case()
        row, = audit.compare_support_chain(*case)
        expected = 6*.7**5+3+4*.7
        self.assertAlmostEqual(row['support_chain'], expected, places=9)
        self.assertLess(row['terminal_full_chain_relative_difference'], 1e-8)
        self.assertLess(row['support_chain_relative_difference'], 1e-8)
        self.assertNotIn('jax', sys.modules)

    def test_distinguishes_terminal_cost_and_transport_support_errors(self):
        ad, fold, base, minus, plus, report = self.case()
        wrong_fold = Archive(fold)
        wrong_fold["support_bars[0]['database'].D13"] = wrong_fold["support_bars[0]['database'].D13"] + .1
        row, = audit.compare_support_chain(ad, wrong_fold, base, minus, plus, report)
        self.assertLess(row['terminal_full_chain_relative_difference'], 1e-8)
        self.assertGreater(row['support_chain_relative_difference'], .01)
        wrong_ad = Archive(ad)
        wrong_ad["explicit_support_bars[0]['geometry'].scale"] = wrong_ad["explicit_support_bars[0]['geometry'].scale"] + .1
        row, = audit.compare_support_chain(wrong_ad, fold, base, minus, plus, report)
        self.assertGreater(row['terminal_full_chain_relative_difference'], .01)
        self.assertLess(row['support_chain_relative_difference'], 1e-8)

    def test_rejects_missing_data_different_points_and_unsupported_contracts(self):
        for failure in ('missing', 'point', 'shape', 'bar', 'labels', 'profile', 'step', 'native', 'fixed'):
            with self.subTest(failure=failure):
                ad, fold, base, minus, plus, report = self.case()
                key = "support['database'].D13"
                if failure == 'missing': del plus[key]
                if failure == 'point': base[key] = np.nextafter(base[key], np.inf)
                if failure == 'shape': plus[key] = np.asarray([1., 2.])
                if failure == 'bar': fold["support_bars[0]['database'].D13"] = np.asarray(np.nan)
                if failure == 'labels': fold['labels[0]'] = np.asarray('other')
                if failure == 'profile': report['parameter_kind'] = 'profile'
                if failure == 'step': report['fd_step'] = 0.
                if failure == 'native': fold['native_vmec_face_coefficient_bars[0]'] = np.ones(1)
                if failure == 'fixed': plus["fixed_inputs['energy_grid'].x"] = np.asarray([1., 3.])
                with self.assertRaises(ValueError):
                    audit.compare_support_chain(ad, fold, base, minus, plus, report)

    def test_cli_writes_extended_comparison_without_importing_jax(self):
        ad, fold, base, minus, plus, report = self.case()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'ad').mkdir()
            (root/'fd').mkdir()
            np.savez(root/'ad/ad_terminal_one.npz', **ad)
            np.savez(root/'fold.npz', **fold)
            for name, data in (('baseline', base), ('x_minus', minus), ('x_plus', plus)):
                np.savez(root/f'fd/fd_{name}_one.npz', **data)
            (root/'report.json').write_text(json.dumps(report))
            command = [sys.executable, str(SCRIPT), '--ad-dir', str(root/'ad'),
                '--fd-dir', str(root/'fd'), '--fd-report', str(root/'report.json'),
                '--out', str(root/'result.csv')]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            with (root/'result.csv').open() as stream:
                row, = csv.DictReader(stream)
            self.assertNotIn('support_chain', row)
            result = subprocess.run(command+['--ad-before-scan-fold', str(root/'fold.npz')],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            with (root/'result.csv').open() as stream:
                row, = csv.DictReader(stream)
            self.assertLess(float(row['support_chain_relative_difference']), 1e-8)
            self.assertIn('terminal_relative=', result.stdout)


if __name__ == '__main__':
    unittest.main()
