"""Optional byte-bounded single writer; complete gzip chunks publish atomically."""
from collections import deque
import gzip
import os
from pathlib import Path
import threading
import time
import uuid

from .latest_checkpoint import atomic_json, file_sha256, fsync_directory


class BoundedWriter:
    def __init__(self, directory, prefix, max_bytes=64 * 1024**2,
                 flush_bytes=8 * 1024**2, flush_seconds=30.):
        self.directory, self.prefix = Path(directory), prefix
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.flush_bytes = min(flush_bytes, max_bytes)
        self.flush_seconds = flush_seconds
        self.condition = threading.Condition()
        self.queue, self.chunks = deque(), []
        self.buffered_bytes = self.enqueued = self.durable = 0
        self.error, self.stopped, self.closed = None, False, False
        self.thread = threading.Thread(target=self._run, name='omr-durable-writer', daemon=True)
        self.thread.start()

    def _check(self):
        if self.error is not None:
            raise RuntimeError('Trajectory writer failed: ' + repr(self.error)) from self.error

    def enqueue(self, data):
        if not isinstance(data, bytes):
            raise TypeError('Writer accepts immutable bytes only')
        if len(data) > self.max_bytes:
            # Explicit failure, never drop or bypass the byte budget.
            raise ValueError('Single log record exceeds bounded writer capacity')
        blocked = 0.
        with self.condition:
            self._check()
            if self.closed:
                raise RuntimeError('Writer is closed')
            while self.buffered_bytes + len(data) > self.max_bytes:
                self.queue.append(('flush', None, None))
                self.condition.notify_all()
                started = time.monotonic()
                self.condition.wait()
                blocked += time.monotonic() - started
                self._check()
            self.enqueued += 1
            self.buffered_bytes += len(data)
            self.queue.append(('data', self.enqueued, data))
            self.condition.notify_all()
            return self.enqueued, blocked

    def drain(self):
        started = time.monotonic()
        with self.condition:
            self._check()
            target = self.enqueued
            self.queue.append(('flush', None, None))
            self.condition.notify_all()
            while self.durable < target:
                self.condition.wait()
                self._check()
        return time.monotonic() - started

    def close(self):
        if self.closed:
            self._check()
            return
        try:
            self.drain()
        finally:
            with self.condition:
                self.closed, self.stopped = True, True
                self.condition.notify_all()
            self.thread.join()
        self._check()

    def _publish(self, pending):
        data = b''.join(item[1] for item in pending)
        path = self.directory / f'{self.prefix}-c{len(self.chunks):06d}-steps.jsonl.gz'
        temporary = path.with_suffix('.tmp-' + uuid.uuid4().hex)
        if path.exists():
            raise FileExistsError(path)
        with temporary.open('xb') as stream:
            stream.write(gzip.compress(data, mtime=0))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(self.directory)
        record = {'path': str(path), 'sha256': file_sha256(path),
                  'first_sequence': pending[0][0], 'last_sequence': pending[-1][0]}
        with self.condition:
            self.chunks.append(record)
            self.durable = pending[-1][0]
            self.buffered_bytes -= len(data)
            self.condition.notify_all()

    def _run(self):
        pending, size, last = [], 0, time.monotonic()
        try:
            while True:
                with self.condition:
                    if not self.queue and not self.stopped:
                        self.condition.wait(max(.01, self.flush_seconds - (time.monotonic() - last)))
                    item = self.queue.popleft() if self.queue else None
                    stop = self.stopped and not self.queue
                if item is not None and item[0] == 'data':
                    pending.append((item[1], item[2]))
                    size += len(item[2])
                if pending and (stop or (item and item[0] == 'flush') or
                                size >= self.flush_bytes or time.monotonic() - last >= self.flush_seconds):
                    self._publish(pending)
                    pending, size, last = [], 0, time.monotonic()
                elif not pending:
                    last = time.monotonic()
                if stop:
                    return
        except BaseException as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
            try:
                atomic_json(self.directory / (self.prefix + '-writer-error.json'), {'error': repr(error)})
            except Exception:
                pass  # Main-thread _check still propagates the original error.
