"""Process-owned startup evidence; no attachment to other processes or signals."""
from __future__ import annotations

import faulthandler
from functools import wraps
import json
import os
from pathlib import Path
import socket
import sys
import time

from .latest_checkpoint import atomic_json

_active = None


class StartupTrace:
    """Record startup stages and dump this process's stacks after a slow stage.

    Diagnostics never change a training exception or terminate training. The
    timer is cancelled before its file is closed, including exceptional exits.
    No locals, environment variables, configuration or exception text are saved.
    """

    def __init__(self, attempt, role='task-runner', timeout=180):
        self.directory = Path(attempt) / 'startup'
        self.stem = f'{role}-{os.getpid()}'
        self.identity = {'schema': 1, 'role': role, 'pid': os.getpid(),
                         'host': socket.gethostname()}
        self.timeout = timeout
        self.started = time.monotonic()
        self.stage = 'entered'
        self.stack = None
        self.closed = False

    def _warn(self, error):
        try:
            print('OMR_STARTUP diagnostic unavailable: ' + type(error).__name__,
                  file=sys.stderr, flush=True)
        except OSError:
            pass

    def _record(self, status, error_type=None):
        row = dict(self.identity, stage=self.stage, status=status,
                   unix_time=time.time(), elapsed_seconds=time.monotonic() - self.started)
        if error_type is not None:
            row['error_type'] = error_type
        try:
            print('OMR_STARTUP ' + json.dumps(row, sort_keys=True), flush=True)
            atomic_json(self.directory / (self.stem + '.json'), row)
            with (self.directory / (self.stem + '.jsonl')).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
        except OSError as error:
            self._warn(error)

    def __enter__(self):
        global _active
        if _active is not None:
            raise RuntimeError('A startup trace is already active in this process')
        _active = self
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.stack = (self.directory / (self.stem + '-stacks.log')).open('a', encoding='utf-8')
        except OSError as error:
            self._warn(error)
        self.mark('entered')
        return self

    def mark(self, stage):
        if self.closed:
            return
        self.stage = stage
        # Arm before metadata writes; the watchdog samples this process even
        # while the Python thread is waiting. Its output still needs writable
        # storage, so this is not a guarantee against a filesystem outage.
        if self.stack is not None:
            try:
                faulthandler.dump_traceback_later(self.timeout, repeat=True,
                                                file=self.stack, exit=False)
            except (OSError, RuntimeError, ValueError) as error:
                self._warn(error)
        self._record('starting')

    def finish(self, error_type=None):
        global _active
        if self.closed:
            return
        self.closed = True
        if self.stack is not None:
            faulthandler.cancel_dump_traceback_later()
            try:
                self.stack.close()
            except OSError as error:
                self._warn(error)
        _active = None
        self._record('failed' if error_type else 'initialized', error_type)

    def __exit__(self, kind, error, traceback):
        self.finish(kind.__name__ if kind else None)
        return False


def startup_stage(stage):
    if _active is not None:
        _active.mark(stage)


def startup_complete():
    if _active is not None:
        _active.mark('workers.initialized')
        _active.finish()


def trace_task_startup(method):
    @wraps(method)
    def wrapped(self, config):
        omr = config.get('omr', {})
        if not omr.get('protocol', False):
            return method(self, config)
        with StartupTrace(omr['attempt_dir']):
            return method(self, config)
    return wrapped
