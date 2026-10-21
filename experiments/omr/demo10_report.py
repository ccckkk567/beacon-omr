"""Read only Demo10 outputs; render actual receipts without inventing acceptance."""
import argparse
import gzip
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from experiments.omr.demo_runtime import BENCHMARKS, METHODS, TAG, run_id


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def report(root):
    root=Path(root)
    journal=read(root/'analysis/beacon-omr/training-submissions'/TAG/'jobs.json') or {'jobs':{}}
    result=[]
    for b in BENCHMARKS:
        for m in METHODS:
            run=root/'analysis/beacon-omr'/run_id(b,m)
            row=dict(benchmark=b,method=m,submission=journal['jobs'].get(b+'/'+m),
                     progress=read(run/'progress.json'),latest_checkpoint=read(run/'checkpoints/latest.json'),
                     first_eight_eight=None,monitoring=[])
            for attempt in sorted((run/'attempts').glob('*')):
                transport=read(attempt/'swanlab/omr-transport.json')
                if transport: row['monitoring'].append(dict(attempt=attempt.name,**transport))
                if m=='active_full' and row['first_eight_eight'] is None:
                    for path in sorted((attempt/'rollouts').glob('train-*-starts.jsonl.gz')):
                        with gzip.open(path,'rt',encoding='utf-8') as stream:
                            starts=[json.loads(line) for line in stream]
                        if len(starts)==16 and sum(r['actual_checkpoint_id'] is not None for r in starts)==8:
                            # Restore success alone is not a completed optimizer update.
                            stem=path.name.removesuffix('-starts.jsonl.gz')
                            commit=read(path.with_name(stem+'-commit.json'))
                            if commit:
                                row['first_eight_eight']=dict(path=str(path),committed_update=commit['committed_update'],
                                    distinct_restart_tasks=len({r['task']['instance_id'] for r in starts if r['actual_checkpoint_id']}),
                                    roots=8,restarts=8,group_size=8)
                                break
            result.append(row)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    args=parser.parse_args()
    print(json.dumps(report(args.root),ensure_ascii=False,indent=2))
