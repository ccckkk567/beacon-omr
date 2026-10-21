"""Bounded read-only sampling of existing native flag records; no model loading."""
import gzip
import json
from pathlib import Path
from .latest_checkpoint import file_sha256


def audit_existing(root, *, byte_budget=8 * 1024 * 1024, max_rows=4096):
    root = Path(root)
    report = dict(schema=1, evidence='existing_saved_native_flags', migration_performed=False,
        counts_reused=False, samples=[], runs=[], gaps=[], decoded_bytes=0, rows=0)
    for method in ('baseline', 'full'):
        run = root / 'analysis/beacon-omr' / f'omr-v2-scienceworld-{method}-s0-fullbatch-20261019'
        row = dict(method=method, run=str(run))
        report['runs'].append(row)
        pointer = run / 'checkpoints/latest.json'
        if not pointer.is_file():
            report['gaps'].append(method + ': old latest unavailable')
            continue
        row['latest'] = json.loads(pointer.read_text())
        complete = run / 'checkpoints' / row['latest']['generation'] / 'COMPLETE.json'
        if not complete.is_file() or file_sha256(complete) != row['latest']['complete_sha256']:
            report['gaps'].append(method + ': COMPLETE metadata checksum not verified')
            continue
        row['complete_metadata'] = json.loads(complete.read_text())['metadata']
        row['verification'] = 'COMPLETE checksum only; model/optimizer files not read'
        if method != 'full':
            continue
        commits = sorted(run.glob('attempts/*/rollouts/*-commit.json'), reverse=True)
        checked = 0
        for commit in commits:
            data = json.loads(commit.read_text())
            update = data.get('committed_update')
            if not isinstance(update, int) or update > row['latest']['update']:
                continue  # Never claim failed/uncheckpointed tails as retained trials.
            steps = commit.with_name(commit.name.removesuffix('-commit.json') + '-steps.jsonl.gz')
            launch = commit.parents[1] / 'launch.json'
            if not steps.is_file() or not launch.is_file():
                continue
            source = json.loads(launch.read_text()).get('source_commit')
            sampled = dict(source_commit=source, source_steps=str(steps), committed_update=update,
                           rows=0, raw_flags_true=0, terminal_flags_true=0, contract_errors=[])
            row.setdefault('collections', []).append(sampled)
            with gzip.open(steps, 'rt', encoding='utf-8') as stream:
                while report['decoded_bytes'] < byte_budget and report['rows'] < max_rows:
                    line = stream.readline(min(1024 * 1024, byte_budget - report['decoded_bytes']) + 1)
                    if not line:
                        break
                    report['decoded_bytes'] += len(line.encode('utf-8'))
                    if not line.endswith('\n'):
                        report['gaps'].append('Sampling stopped at a partial/oversized row')
                        break
                    record = json.loads(line)
                    if record.get('split') != 'train':
                        continue
                    report['rows'] += 1
                    sampled['rows'] += 1
                    native = record.get('info', {})
                    flag = native.get('subgoal_completed')
                    if type(flag) is not bool:
                        sampled['contract_errors'].append(dict(step_id=record.get('step_id'), reason='native_flag_missing_or_wrong_type'))
                        continue
                    sampled['raw_flags_true'] += flag
                    sampled['terminal_flags_true'] += bool(flag and record['done'])
                    if flag and len(report['samples']) < 8:
                        report['samples'].append(dict(source=str(steps), step_id=record['step_id'],
                            task_id=record['task_id'], local_step=record['local_step'], raw_flag=flag,
                            native_done=record['done'], strict_goal=record.get('strict_success'),
                            score_pre=native.get('score_pre'), score_post=native.get('score_post'),
                            new_B_for_this_observed_action=True,
                            nonterminal_candidate_signal=not record['done']))
            checked += 1
            if checked >= 2 or report['decoded_bytes'] >= byte_budget or report['rows'] >= max_rows:
                break
        if not checked:
            report['gaps'].append('Full: no saved committed flag collection was available')
    report['gaps'].append('Bounded sample only; no old statistics migration or complete lineage/native replay claim')
    report['gaps'].append('Old records do not contain all legal commands; boundary keys are not invented retrospectively')
    return report
