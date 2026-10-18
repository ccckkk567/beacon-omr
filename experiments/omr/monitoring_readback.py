"""Bounded online SwanLab acceptance using the installed public query API."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shlex
import time

from experiments.omr.latest_checkpoint import atomic_json
from experiments.omr.monitoring import SwanLabMonitor


def swanlab_key_from_file(path):
    # Read only this literal assignment, never source/execute the credentials file.
    values = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        match = re.match(r'^\s*(?:export\s+)?SWANLAB_API_KEY=(.*)$', line)
        if match:
            tokens = shlex.split(match[1], comments=True)
            if len(tokens) != 1 or not re.fullmatch(r'[A-Za-z0-9._-]+', tokens[0]):
                raise ValueError('SWANLAB_API_KEY must be a literal nonempty token')
            values.append(tokens[0])
    if len(values) != 1:
        raise ValueError('Expected one literal SWANLAB_API_KEY assignment')
    return values[0]


def verify_points(response, expected):
    """Accept only exact metric keys and explicit matching steps and values."""
    series = {row['key']: row.get('metrics', []) for row in response.get('list', [])}
    for key, points in expected.items():
        received = series.get(key, [])
        for step, value in points:
            if not any(isinstance(item, dict) and item.get('step', item.get('index')) == step
                       and isinstance(item.get('value'), (int, float))
                       and math.isclose(item['value'], value, rel_tol=1e-7, abs_tol=1e-7)
                       for item in received):
                return False
    return True


def online_acceptance(payload, sdk=None, sleep=time.sleep):
    if sdk is None:
        import swanlab as sdk
    if not os.environ.get('SWANLAB_API_KEY'):
        raise RuntimeError('Online acceptance requires its child-only runtime credential')
    expected = {'fixture/value': [(0, 1.25), (1, 2.5)], 'axis/policy_update': [(0, 0), (1, 1)]}
    receipts = []
    for exit_code, label, expected_state in ((0, 'success', 'FINISHED'), (1, 'failure', 'CRASHED')):
        directory = Path(payload['output_directory']) / ('swanlab-online-' + label)
        monitor = SwanLabMonitor('beacon-omr-readiness', 'transport-' + payload['job_id'] + '-' + label,
            {'test_only': True, 'source_commit': payload['source_commit'], 'optimizer_updates': 0},
            {'mode': 'online', 'resume': 'never', 'logdir': str(directory)}, sdk=sdk)
        try:
            parts = monitor.run.path.strip('/').split('/')
            parts[0] = parts[0].lstrip('@')
            run_path = '/'.join(parts)
            for step, value in expected['fixture/value']:
                monitor.log({'fixture/value': value}, step)
        except Exception:
            monitor.finish(1, 'Online acceptance failed before explicit shutdown')
            raise
        monitor.finish(exit_code, 'Test-only intentional failure' if exit_code else None)
        api = sdk.Api(api_key=os.environ['SWANLAB_API_KEY'])
        verified, attempts, response, remote_state = False, 0, {}, None
        # Fresh query objects avoid caching an earlier RUNNING/empty response.
        # Job-level process timeout also bounds SDK network retries.
        for delay in (0, 3, 6, 10, 15, 20):
            if delay:
                sleep(delay)
            attempts += 1
            remote = api.run(run_path)
            remote_state = remote.state
            response = remote.metrics(keys=list(expected), sample=10)
            if remote_state == expected_state and verify_points(response, expected):
                verified = True
                break
        evidence = {'run_id': monitor.status['run_id'], 'url': monitor.status['url'],
                    'run_path': run_path, 'remote_state': remote_state,
                    'expected_state': expected_state, 'query_attempts': attempts,
                    'remote_readback_verified': verified, 'metrics': response}
        atomic_json(directory / 'remote-readback.json', evidence)
        if not verified:
            raise RuntimeError('Online metrics or final state were not confirmed by SwanLab readback')
        monitor.status.update(status='online_verified', remote_readback_verified=True)
        atomic_json(monitor.status_path, monitor.status)
        receipts.append(evidence)
    return {'online_verified': True, 'test_only': True, 'attempts': receipts}
