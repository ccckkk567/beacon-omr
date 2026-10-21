"""One SwanLab writer per trainer; local metrics are durable before upload.

The persisted status is transport evidence, never a claim that the server has
been queried to confirm metrics. A remote read-back is part of HPC readiness.
"""
from __future__ import annotations

import inspect
import json
import numbers
import os
from pathlib import Path
import re
import time

from .latest_checkpoint import atomic_json


def sanitized(value):
    if isinstance(value, dict):
        return {key: ('[REDACTED]' if re.search(r'(?i)(api[_-]?key|password|secret|access[_-]?token|hf_token|credential)', str(key)) else sanitized(item)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitized(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return value


def resume_policy(checkpoint_update, last_logged_update, previous_id, *, offline=False, allow_new_run=False):
    if checkpoint_update is None:
        return {"resume": "never", "id": None, "reason": "new_run"}
    if last_logged_update > checkpoint_update or offline:
        return {"resume": "never", "id": None, "reason": "rollback" if not offline else "offline_attempt",
                "resume_parent_id": previous_id, "resume_from_update": checkpoint_update}
    if not previous_id:
        if allow_new_run:
            return {'resume': 'never', 'id': None, 'reason': 'missing_previous_run_id',
                    'resume_from_update': checkpoint_update}
        raise ValueError("A normal monitoring resume requires its persisted run ID")
    return {"resume": "allow" if allow_new_run else "must", "id": previous_id, "reason": "continuous"}


class SwanLabMonitor:
    def __init__(self, project, name, config, options=None, sdk=None):
        if sdk is None:
            import swanlab as sdk
        self.sdk = sdk
        options = dict(options or {})
        self.directory = Path(options.pop('logdir', os.environ.get('SWANLAB_LOG_DIR', 'swanlog')))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.status_path = self.directory / 'omr-transport.json'
        self.metrics_path = self.directory / 'omr-metrics.jsonl'
        self.closed = False
        self.sequence = options.pop('initial_log_sequence', -1)
        self.step_mode = options.pop('step_mode', 'log_sequence')
        self.initial_policy_update = options.pop('initial_policy_update', 0)
        self.last_policy_update = None
        if self.step_mode not in ('log_sequence', 'policy_update'):
            raise ValueError('Unknown monitoring step axis')
        self.mode = options.pop('mode', os.environ.get('SWANLAB_MODE', 'online'))
        self.status = {"status": "initializing", "remote_readback_verified": False, "mode": self.mode}
        self.status.update(step_mode=self.step_mode, initial_policy_update=self.initial_policy_update)
        atomic_json(self.status_path, self.status)
        parameters = inspect.signature(sdk.init).parameters
        name_key = 'name' if 'name' in parameters else 'experiment_name'
        directory_key = 'log_dir' if 'log_dir' in parameters else 'logdir'
        kwargs = dict(project=project, config=sanitized(config or {}), mode=self.mode)
        kwargs.update({name_key: name, directory_key: str(self.directory)})
        allowed = {'workspace', 'id', 'resume', 'group', 'job_type', 'tags', 'description'}
        unexpected = set(options) - allowed
        if unexpected:
            raise ValueError('Unknown SwanLab options: ' + ', '.join(sorted(unexpected)))
        kwargs.update({key: value for key, value in options.items() if value is not None})
        if kwargs.get('resume') == 'never' and 'id' in kwargs:
            raise ValueError('SwanLab resume=never must not receive an id')
        # The SDK reads SWANLAB_API_KEY from the process. Do not persist login or
        # overwrite the user's shared login state with swanlab.login().
        self.run = sdk.init(**kwargs)
        public = getattr(self.run, 'public', None)
        # In SwanLab 0.10.2, url is a property that raises ValueError offline.
        # getattr(..., None) still invokes that property. Do not touch either
        # cloud URL path unless the run was explicitly initialized online.
        url = None
        if self.mode == 'online':
            url = (getattr(self.run, 'url', None)
                   or getattr(getattr(public, 'cloud', None), 'experiment_url', None))
        self.status.update(status='offline_pending_sync' if self.mode in ('offline', 'local') else 'online_unverified',
                           run_id=getattr(self.run, 'id', None) or getattr(public, 'run_id', None),
                           url=url)
        atomic_json(self.status_path, self.status)

    def log(self, data, step):
        if self.closed:
            raise RuntimeError('Cannot log to a closed attempt')
        if self.step_mode == 'policy_update':
            if (not isinstance(step, numbers.Integral) or isinstance(step, bool)
                    or step < self.initial_policy_update
                    or (self.last_policy_update is not None and step < self.last_policy_update)):
                raise ValueError('SwanLab policy step must continue from the restored update without reversal')
        self.sequence += 1
        record = {'sequence': self.sequence, 'policy_update': step, 'time': time.time(), 'metrics': sanitized(data)}
        # Fail loudly on lost local scientific evidence or non-serializable metrics.
        with self.metrics_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        sdk_step = int(step) if self.step_mode == 'policy_update' else self.sequence
        self.run.log({**data, 'axis/policy_update': step}, step=sdk_step)
        self.last_policy_update = step
        self.status.update(last_log_sequence=self.sequence, last_logged_update=step)
        atomic_json(self.status_path, self.status)

    def finish(self, exit_code=0, error=None):
        if self.closed:
            return
        self.status.update(exit_code=exit_code, training_status='completed' if exit_code == 0 else 'failed')
        atomic_json(self.status_path, self.status)
        parameters = inspect.signature(self.run.finish).parameters
        if 'state' in parameters:
            self.run.finish(state='success' if exit_code == 0 else 'crashed',
                            error=None if exit_code == 0 else (error or 'Training failed'))
        elif 'error' in parameters:
            self.run.finish(error=None if exit_code == 0 else (error or 'Training failed'))
        elif exit_code == 0:
            self.run.finish()
        else:
            raise RuntimeError('Installed SwanLab cannot explicitly mark failure; runtime compatibility required')
        self.closed = True
        self.status['sdk_finish_returned'] = True
        atomic_json(self.status_path, self.status)
