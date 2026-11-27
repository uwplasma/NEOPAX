"""Compare original pre-NTX initialization with inherited and one-thread settings.

Uses the original FD profile initializer and configured VMEX solve. Stops before
NTX, roots, transport or FD endpoints. The parent imports no JAX.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parent))
import benchmark_transport_ntss1d_ad_fd as runner
from check_saved_transport_baseline_repeatability import differences, read_one


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    source,out=args.run_dir.resolve(),args.out_dir.resolve()
    if out.is_relative_to(source) or (out.exists() and any(out.iterdir())):
        parser.error('Choose a fresh empty directory outside the saved run')
    config=(source/'config.toml').read_text()
    settings=re.search(r'^# comparison: steps=(\d+) rel=(\S+) abs=(\S+) mean=(native|fixed) ',config,re.M)
    if settings is None:
        parser.error('Require the saved comparison settings')
    steps,rel,absolute,mean=settings.groups()
    plan=SimpleNamespace(config=source/'config.toml',out_dir=source,
        accepted_step_limit=int(steps),fd_rel_step=float(rel),fd_abs_step=float(absolute))
    _,original,_=next(task for task in runner.commands(plan) if task[0]=='fd_density_shape_alpha')
    i=original.index('--ad-baseline-reference')
    command=original[:i]+original[i+2:]
    env=runner.worker_environment()
    env['NEOPAX_SOLVER_MEAN_REDUCTION']=mean
    jobs=[]
    for label in ('inherited','single_thread'):
        local=dict(env)
        if label=='single_thread':
            local.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        cmd=[*command,'--geometry-input-only-dir',str(out/label/'inputs'),'--geometry-input-compiler-ir']
        jobs.append((label,cmd,local))
        print(f'[vmec-thread] {label}: OPENBLAS_NUM_THREADS={local.get("OPENBLAS_NUM_THREADS")} '
              f'OMP_NUM_THREADS={local.get("OMP_NUM_THREADS")} '+shlex.join(cmd),flush=True)
    if args.plan_only:
        return
    if all(env.get(name)=='1' for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')):
        raise ValueError('Inherited settings already force one thread; this would not contrast environments')
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.toml').write_text(config)
    captures={};environments={};sources={};internal={}
    for label,cmd,local in jobs:
        runner.worker(cmd,out/label,local)
        folder=out/label/'inputs/geometry'
        manifest=json.loads((folder/'boundaries.json').read_text())
        calls=[row for row in manifest if row['stage']=='vmec_solve']
        if len(calls)!=1:
            raise ValueError(f'Expected one original VMEX call in {label}; captured {len(calls)}')
        captures[label]=read_one(folder,calls[0]['file'])
        internal[label]={phase:[read_one(folder,row['file']) for row in manifest if row['stage']==phase]
                         for phase in ('vmec_host','vmec_refinement')}
        for phase,records in internal[label].items():
            print('[vmec-thread]',label,phase,'default devices:',
                  [str(z.get('capture.default_device','not recorded')) for z in records],flush=True)
        environments[label]=json.loads((folder/'runtime_environment.json').read_text())
        sources[label]=json.loads((folder/'loaded_sources.json').read_text())
    a,b=captures['inherited'],captures['single_thread']
    arguments=lambda z:{k:v for k,v in z.items() if not k.startswith('capture.outputs.')}
    outputs=lambda z:{k:v for k,v in z.items() if k.startswith('capture.outputs.')}
    ad=read_one(source/'ad/snapshots','ad_compact_geometry_*.npz')
    expected={'capture.outputs.'+k[6:]:v for k,v in ad.items() if k.startswith('state.')}
    report=dict(arguments=differences(arguments(a),arguments(b)),outputs=differences(outputs(a),outputs(b)),
        vs_saved_ad={label:differences(expected,{k:v for k,v in outputs(z).items()
                        if not k.endswith('.__weak_type__')}) for label,z in captures.items()},
        source_hashes_match=sources['inherited']==sources['single_thread'],environments=environments)
    report['internal_phases']={}
    for phase in ('vmec_host','vmec_refinement'):
        left,right=internal['inherited'][phase],internal['single_thread'][phase]
        report['internal_phases'][phase]=dict(call_counts=[len(left),len(right)],
            differences=[differences(x,y) for x,y in zip(left,right)])
    (out/'comparison.json').write_text(json.dumps(report,indent=2))
    print('[vmec-thread] source_hashes_match:',report['source_hashes_match'],flush=True)
    for name in ('arguments','outputs'):
        print(f'[vmec-thread] {name} changed_fields={len(report[name])}',flush=True)
        for row in report[name]: print('[vmec-thread]',row,flush=True)
    for phase,data in report['internal_phases'].items():
        print('[vmec-thread]',phase,'call_counts:',data['call_counts'],flush=True)
        for i,rows in enumerate(data['differences']):
            print('[vmec-thread]',phase,'call:',i,'changed_fields:',len(rows),flush=True)
            for row in rows: print('[vmec-thread]',row,flush=True)
    for label,rows in report['vs_saved_ad'].items():
        print(f'[vmec-thread] {label} vs saved AD changed_fields={len(rows)}',flush=True)
        for row in rows: print('[vmec-thread]',row,flush=True)
        print('[vmec-thread]',label,'threadpools:',environments[label]['threadpools'],flush=True)
    if report['arguments'] or not report['source_hashes_match']:
        raise ValueError('Inputs or sources also changed; do not attribute the result to threads')
    print('[vmec-thread] saved',out,flush=True)


if __name__=='__main__':
    main()
