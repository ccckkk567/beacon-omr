"""Submit only six isolated Demo10 jobs; never update, hold or cancel old jobs."""
import argparse
from contextlib import contextmanager
import getpass
import json
from pathlib import Path
import re
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.omr.demo_runtime import TAG, BENCHMARKS, METHODS, run_id, binding, verify_integration
from experiments.omr.latest_checkpoint import atomic_json, LatestCheckpoint
from experiments.omr.resume_chain import command, clean_revision
from experiments.omr.submit_benchmark_training import submit_once


@contextmanager
def locked(directory):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def job_command(root, revision, benchmark, method, resume=False):
    return ['sbatch','--parsable','--account=prjsnnn0000269','--partition=testqueue',
        '--nodes=1','--ntasks=1','--gres=gpu:h200:2','--cpus-per-task=32','--mem=256G','--time=18:00:00',
        '--chdir='+str(root),'--job-name='+run_id(benchmark,method),
        '--output='+str(root/'logs/beacon-omr'/(run_id(benchmark,method)+'-%j.out')),
        str(REPO/'experiments/omr/demo10.sbatch'),str(REPO),str(root),revision,benchmark,method,
        'resume' if resume else 'fresh']


def submit(root, revision, *, apply=False, invoke=command):
    integration = verify_integration(root, revision)
    bindings = {b:binding(root,b,revision) for b in BENCHMARKS}
    directory=root/'analysis/beacon-omr/training-submissions'/TAG
    ledger=directory/'jobs.json'
    request=dict(revision=revision, root=str(root), demo=bindings,
                 model=integration['model'], standalone=True, automatic_resume=False)
    state=json.loads(ledger.read_text()) if ledger.exists() else dict(binding=request,jobs={})
    if state['binding'] != request:
        raise ValueError('Demo10 submission binding changed; preserve existing experiments')
    queue=invoke(['squeue','-h','-r','--user='+getpass.getuser(),'--format=%i|%j'])
    known={r['job_id'] for r in state['jobs'].values() if r.get('status')=='submitted'}
    names={run_id(b,m) for b in BENCHMARKS for m in METHODS}
    for line in queue.splitlines():
        job,name=map(str.strip,line.split('|',1))
        if name in names and job not in known:
            raise ValueError('Unrecorded Demo10 job exists; preserve and inspect '+job)
    commands=[job_command(root,revision,b,m) for b in BENCHMARKS for m in METHODS]
    for b in BENCHMARKS:
        for m in METHODS:
            run=root/'analysis/beacon-omr'/run_id(b,m)
            if run.is_symlink() or not run.resolve().is_relative_to(root.resolve()):
                raise ValueError('Demo output path escapes owned root')
            key=b+'/'+m
            expected={'revision':revision,'execution_profile_id':'perturn-legacy-v1','writer_profile_id':'sync-v1'}
            if run.exists():
                p=run/'execution-profile.json'
                if not p.is_file() or json.loads(p.read_text()) != expected:
                    raise ValueError('Demo output is not owned by this frozen request')
                if key not in state['jobs'] and any(p.name != 'execution-profile.json' for p in run.iterdir()):
                    raise ValueError('Unrecorded existing Demo training state; preserve it')
    preview=dict(commands=commands, protected_queue=queue, binding=request,
                 free_bytes=shutil.disk_usage(root).free,
                 minimum_free_bytes=6*48*1024**3, no_old_job_mutations=True)
    if not apply:
        return preview
    clean_revision(REPO,revision)
    if preview['free_bytes'] < preview['minimum_free_bytes']:
        raise ValueError('Insufficient shared free space for six latest+staging model stores; preserve old files')
    (root/'logs/beacon-omr').mkdir(parents=True,exist_ok=True)
    with locked(directory):
        # Reload under the lock so concurrent requests cannot use stale receipts.
        if ledger.exists():
            state=json.loads(ledger.read_text())
            if state['binding'] != request:
                raise ValueError('Submission binding changed under lock')
        atomic_json(directory/'dry-run.json',preview)
        for b in BENCHMARKS:
            for m in METHODS:
                run=root/'analysis/beacon-omr'/run_id(b,m)
                run.mkdir(parents=True,exist_ok=True)
                profile=run/'execution-profile.json'
                if not profile.exists():
                    atomic_json(profile,dict(revision=revision,execution_profile_id='perturn-legacy-v1',writer_profile_id='sync-v1'))
                job=submit_once(state,b+'/'+m,job_command(root,revision,b,m),ledger,invoke,id_field='job_id')
                print(f'{b}/{m}: {job} (Demo10; testqueue18h; standalone; no child)',flush=True)
    return state


def queue_preparation(root, revision, invoke=command):
    clean_revision(REPO,revision)
    directory=root/'analysis/beacon-omr/training-submissions'/TAG
    (root/'logs/beacon-omr').mkdir(parents=True,exist_ok=True)
    with locked(directory):
        ledger=directory/'prepare-jobs.json'
        request=dict(revision=revision,root=str(root))
        state=json.loads(ledger.read_text()) if ledger.exists() else dict(binding=request,jobs={})
        if state['binding'] != request:
            raise ValueError('Preparation source changed')
        job=submit_once(state,'prepare',['sbatch','--parsable','--account=prjsnnn0000269',
            '--partition=ioqueue','--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=8G','--time=00:30:00',
            '--job-name=omr-demo10-prepare','--chdir='+str(root),
            '--output='+str(root/'logs/beacon-omr/demo10-prepare-%j.out'),
            str(REPO/'experiments/omr/prepare_demo10.sbatch'),str(REPO),str(root),revision],
            ledger,invoke,id_field='job_id')
        print('Demo10 preparation: '+job+' (CPU; freezes manifests then submits six formal jobs)')


def resume_one(root, revision, condition, invoke=command):
    """Only explicitly requested continuations; never create automatic children."""
    b,m=condition.split('/')
    name=run_id(b,m)
    verify_integration(root,revision); binding(root,b,revision); clean_revision(REPO,revision)
    queue=invoke(['squeue','-h','-r','--user='+getpass.getuser(),'--format=%i|%j'])
    if any(line.split('|',1)[-1].strip()==name for line in queue.splitlines()):
        raise ValueError('This Demo10 condition still has an active job; preserve it')
    run=root/'analysis/beacon-omr'/name
    checkpoint, complete=LatestCheckpoint(run/'checkpoints',name).latest()
    if complete['update']>=150:
        raise ValueError('Training has already reached the requested budget')
    directory=root/'analysis/beacon-omr/training-submissions'/TAG
    with locked(directory):
        ledger=directory/('resume-'+b+'-'+m+'-'+str(complete['update'])+'.json')
        state=json.loads(ledger.read_text()) if ledger.exists() else dict(jobs={},checkpoint=str(checkpoint))
        job=submit_once(state,condition,job_command(root,revision,b,m,resume=True),ledger,invoke,id_field='job_id')
        print(f'{condition}: {job} (explicit resume from {complete["update"]}; no child)')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['queue-preparation','submit','preview','resume'])
    parser.add_argument('--root',required=True); parser.add_argument('--revision',required=True)
    parser.add_argument('--condition')
    args=parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}',args.revision):
        parser.error('Use a full frozen revision')
    root=Path(args.root).absolute()
    if args.action=='queue-preparation': queue_preparation(root,args.revision)
    elif args.action=='resume':
        if not args.condition: parser.error('Explicit benchmark/method required')
        resume_one(root,args.revision,args.condition)
    else: print(json.dumps(submit(root,args.revision,apply=args.action=='submit'),indent=2))


if __name__=='__main__':
    main()
