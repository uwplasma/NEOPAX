"""Complete missing FD columns against saved AD in a fresh, separate follow-up.

The original run stays untouched. Uses its config, increments and solver-mean
scope, and the existing benchmarks, guards and comparison writer. No parent JAX.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re
import shlex
import shutil
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parent))
import benchmark_transport_ntss1d_ad_fd as runner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    source,out=args.run_dir.resolve(),args.out_dir.resolve()
    if out.is_relative_to(source):
        parser.error('The follow-up directory must be outside the original run')
    config=(source/'config.toml').read_text()
    settings=re.search(r'^# comparison: steps=(\d+) rel=(\S+) abs=(\S+) mean=(native|fixed) capture_forward_ir=(True|False) source=(\w+)\s*$',config,re.M)
    if settings is None:
        parser.error('Require the original comparison settings/source record')
    steps,rel,absolute,mean,capture_ir,source_hash=settings.groups()
    plan=SimpleNamespace(config=source/'config.toml',out_dir=out,
        accepted_step_limit=int(steps),fd_rel_step=float(rel),fd_abs_step=float(absolute))
    completed=[name for name,_,_ in runner.commands(plan)
               if (source/name/'summary.json').is_file()]
    if 'ad' not in completed:
        parser.error('Require completed saved AD')
    ad=json.loads((source/'ad/summary.json').read_text())
    objectives={label.removeprefix('transport:') for label in ad['residual_labels']
                if label.startswith('transport:')}
    tasks=[(name,command,report) for name,command,report in runner.commands(plan) if name not in completed]
    if out.exists() and any(out.iterdir()):
        parser.error('Choose a fresh empty output directory')
    for name,command,_ in tasks:
        command += ['--ntx-scan-audit-dir',str(out/name/'baseline_inputs'),
                    '--ntx-scan-audit-compiler-ir','--ntx-scan-audit-continue']
        print('[saved-ad-followup] '+name+': '+shlex.join(command),flush=True)
    if args.plan_only:
        return
    versions=json.loads((source/'ad/forward/metadata.json').read_text())['versions']
    if any(version(name)!=expected for name,expected in versions.items()):
        raise ValueError('Installed package versions differ from saved AD')
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.toml').write_text(config)
    (out/'reference_provenance.json').write_text(json.dumps(dict(
        comparison='saved AD versus follow-up FD; not a fresh all-worker run',
        reference_run=str(source),reference_source=source_hash,
        reused_workers=completed,new_workers=[name for name,_,_ in tasks],versions=versions,
        fd_source=runner.hashlib.sha256((runner.ROOT/runner.FD_SCRIPT).read_bytes()).hexdigest()),indent=2))
    for name in completed:
        folder=out/name
        folder.mkdir()
        shutil.copy2(source/name/'summary.json',folder/'summary.json')
        shutil.copytree(source/name/'snapshots',folder/'snapshots')
        for relative in ('forward/metadata.json','solver_mean_reduction.json'):
            if (source/name/relative).is_file():
                (folder/relative).parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source/name/relative,folder/relative)
    runner.prepare_ad_reference(out)
    env=runner.worker_environment()
    env.update(NEOPAX_SOLVER_MEAN_REDUCTION=mean,NEOPAX_CAPTURE_FORWARD_IR='1' if capture_ir=='True' else '0')
    def compare():
        rows=runner.compare(out,solver_mean_reduction=mean)
        if not rows or not all(row['comparison_baseline_matches'] for row in rows):
            raise ValueError('Baseline comparison failed; saved evidence retained')
        return rows
    compare()
    for name,command,report in tasks:
        folder=out/name
        runner.worker(command,folder,env)
        data=json.loads(report.read_text())
        if (data['parameter_name'].replace(':','_')!=name[3:]
            or data['accepted_step_limit']!=plan.accepted_step_limit
            or data.get('database_interpolation_mode')!=ad.get('database_interpolation_mode',runner.MODE)
            or data.get('vmec_response_mode')!='raw_block'
            or set(data['objective_labels'])!=objectives
            or not len(data['objective_labels'])==len(data['objective_values'])==len(data['gradient_fd'])
            or not all(runner.math.isfinite(v) for v in (*data['objective_values'],*data['gradient_fd']))):
            raise ValueError('FD report parameter, step count or finite values are invalid')
        if capture_ir=='True':
            runner.require_forward_capture(folder,ad=False)
        if mean=='fixed' and not json.loads((folder/'solver_mean_reduction.json').read_text()).get('fixed_mean_sizes'):
            raise ValueError('Fixed solver means were not executed')
        shutil.copy2(report,folder/'summary.json')
        compare()
    rows=compare()
    for row in rows:
        print(f'[saved-ad-followup] {row["parameter"]} {row["objective"]}: '
              f'AD={row["gradient_ad"]:.12e} FD={row["gradient_fd"]:.12e} '
              f'relative_to_fd={row["relative_difference_to_fd"]:.6e}',flush=True)
    print(f'[saved-ad-followup] saved {len(rows)} pairs: {out/"ad_vs_fd.csv"}',flush=True)


if __name__=='__main__':
    main()
