"""Shared-wrapper startup regression; no model or GPU environment is created."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.omr.launch_training import reused_runtime


class RuntimeTests(unittest.TestCase):
    def test_simultaneous_start_waits_for_complete_shared_wrapper(self):
        entered = threading.Event()
        second_lock = threading.Event()
        release = threading.Event()
        mutex = threading.Lock()
        attempts = []

        # Windows has no fcntl. Model its lock for the bootstrap interleaving;
        # on POSIX exercise the real flock implementation too.
        try:
            import fcntl
        except ImportError:
            fcntl = SimpleNamespace(LOCK_EX=1, LOCK_UN=2)
            fcntl.flock = lambda fd, op: (mutex.acquire() if op == 1 else mutex.release())
        native_flock = fcntl.flock

        def flock(fd, op):
            if op == fcntl.LOCK_EX:
                attempts.append(fd)
                if len(attempts) == 2:
                    second_lock.set()
            native_flock(fd, op)

        def create_wrapper(command, check):
            Path(command[-1]).mkdir(parents=True)
            entered.set()  # The prefix exists but is not ready for another job.
            if not release.wait(5):
                raise TimeoutError('Fixture was not released')

        locks = SimpleNamespace(LOCK_EX=fcntl.LOCK_EX, LOCK_UN=fcntl.LOCK_UN, flock=flock)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {'fcntl': locks}), \
                patch('experiments.omr.launch_training.subprocess.run', side_effect=create_wrapper) as create:
            root = Path(directory)
            base = root / 'base'
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(reused_runtime, root, 'shared', base)
                try:
                    self.assertTrue(entered.wait(5))
                    second = pool.submit(reused_runtime, root, 'shared', base)
                    self.assertTrue(second_lock.wait(5))
                    self.assertFalse(second.done())
                finally:
                    release.set()
                self.assertEqual(first.result(timeout=5), second.result(timeout=5))
            create.assert_called_once()
            marker = root / 'envs/shared/OMR-reuse.json'
            self.assertEqual(json.loads(marker.read_text())['base'], str(base))
            with self.assertRaisesRegex(RuntimeError, 'ownership/binding differs'):
                reused_runtime(root, 'shared', root / 'other-base')
            self.assertEqual(json.loads(marker.read_text())['base'], str(base))


if __name__ == '__main__':
    unittest.main()
