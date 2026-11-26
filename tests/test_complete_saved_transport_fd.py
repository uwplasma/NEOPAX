"""Follow-up planning and orchestration only; no JAX or physics imports."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import test_transport_ntss1d_ad_fd_runner as fixtures

FILE=fixtures.SCRIPT.with_name('complete_saved_transport_fd.py')
spec=importlib.util.spec_from_file_location('saved_fd_followup',FILE)
followup=importlib.util.module_from_spec(spec)
spec.loader.exec_module(followup)


class SavedFDFollowupTests(unittest.TestCase):
    def test_only_missing_columns_run_and_original_run_stays_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'original';out=root/'followup'
            arrays=fixtures.NTSS1DComparisonRunnerTests().saved_case(source)
            (source/'config.toml').write_text('# comparison: steps=16 rel=3e-07 abs=1e-10 mean=fixed capture_forward_ir=True source=abc123\n')
            (source/'ad/forward').mkdir()
            (source/'ad/forward/metadata.json').write_text(json.dumps({'versions':{'jax':'test'}}))
            missing=('fd_density_shape_alpha','fd_temperature_shape_alpha','fd_RBC_1_0','fd_ZBS_1_0')
            for name in missing:
                (source/name/'summary.json').unlink()
            before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
            argv=['followup','--run-dir',str(source),'--out-dir',str(out)]
            output=io.StringIO()
            with patch.object(sys,'argv',[*argv,'--plan-only']),redirect_stdout(output):
                followup.main()
            self.assertEqual(len(output.getvalue().splitlines()),4)
            self.assertNotIn('--reverse-parameter-mode',output.getvalue())
            self.assertEqual(output.getvalue().count('--ntx-scan-audit-continue'),4)
            self.assertFalse(out.exists())
            calls=[]
            report=root/'new_report.json'
            original_commands=followup.runner.commands
            def commands(args):
                return [(name,command,report) for name,command,_ in original_commands(args)]
            def worker(command,folder,env):
                calls.append(folder.name)
                self.assertEqual(env['NEOPAX_SOLVER_MEAN_REDUCTION'],'fixed')
                self.assertEqual(env['NEOPAX_CAPTURE_FORWARD_IR'],'1')
                self.assertIn('--ad-baseline-reference',command)
                self.assertIn('--ntx-scan-audit-compiler-ir',command)
                folder.mkdir();(folder/'snapshots').mkdir()
                np.savez(folder/'snapshots/fd_baseline_new.npz',**arrays)
                (folder/'solver_mean_reduction.json').write_text(json.dumps({'fixed_mean_sizes':[408]}))
                parameter=command[command.index('--parameter')+1]
                i=followup.runner.PARAMETERS.index(parameter)
                report.write_text(json.dumps(dict(parameter_name=parameter,accepted_step_limit=16,
                    database_interpolation_mode=followup.runner.MODE,vmec_response_mode='raw_block',
                    objective_labels=['softmax_Er','bootstrap_current_softmax_abs_scaled'],
                    objective_values=[2.,3.],gradient_fd=[float(i+1),float(-2*(i+1))])))
            with patch.object(sys,'argv',argv),patch.object(followup,'version',return_value='test'), \
                 patch.object(followup.runner,'commands',commands), \
                 patch.object(followup.runner,'worker',side_effect=worker), \
                 patch.object(followup.runner,'require_forward_capture') as verify,redirect_stdout(io.StringIO()):
                followup.main()
            self.assertEqual(tuple(calls),missing)
            self.assertEqual(verify.call_count,4)
            self.assertEqual(before,{str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()})
            provenance=json.loads((out/'reference_provenance.json').read_text())
            self.assertEqual(provenance['new_workers'],list(missing))
            self.assertEqual(len((out/'ad_vs_fd.csv').read_text().splitlines()),17)
        self.assertNotIn('jax',sys.modules)


if __name__=='__main__':
    unittest.main()
