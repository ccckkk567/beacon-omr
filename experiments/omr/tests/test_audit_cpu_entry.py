import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ENTRY = Path(__file__).resolve().parents[1] / 'audit_cpu_entry.py'


class AuditCPUEntryTests(unittest.TestCase):
    def test_isolated_cpu_entry_skips_pth_and_runs_read_only_report_pipeline(self):
        if sys.version_info[:2] != (3, 12):
            self.skipTest('Existing audit NumPy ABI is Python3.12')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = root / 'envs/gigpo-scienceworld-py312-v1/lib/python3.12/site-packages'
            (site / 'numpy').mkdir(parents=True)
            # Startup-isolation fixture only. No manifest/schedule is present;
            # this does not stand in for a real NumPy schedule validation.
            (site / 'numpy/__init__.py').write_text("__version__='startup-fixture'\nclass Random:\n    class RandomState:\n        def __init__(self, seed): pass\n        def choice(self, n, k, replace): return list(range(k))\nrandom=Random()\n")
            poison = root / 'startup-hook-executed'
            (site / 'poison.pth').write_text(f'import pathlib; pathlib.Path({str(poison)!r}).write_text("unexpected")\n')
            output = root / 'analysis/beacon-omr/audits/startup-test'
            reply = subprocess.check_output([sys.executable, '-I', '-S', '-B', str(ENTRY), '--root', str(root),
                                             '--output', str(output)], text=True)
            self.assertFalse(poison.exists())
            for stage in ('audit-python-entry', 'audit-numpy-import', 'audit-randomstate', 'audit-metadata-and-records'):
                self.assertIn('STAGE ' + stage, reply)
            self.assertTrue((output / 'restart_opportunity_audit.md').is_file())
            self.assertEqual(json.loads((output / 'audit.json').read_text())['opportunities'], {})

    def test_site_enabled_entry_is_rejected_before_importing_training_packages(self):
        reply = subprocess.run([sys.executable, '-I', '-B', str(ENTRY), '--root', '.', '--output', '.'],
                               text=True, capture_output=True)
        self.assertNotEqual(reply.returncode, 0)
        self.assertIn('requires -I -S', reply.stderr)
        self.assertNotIn('STAGE audit-numpy-import', reply.stdout)


if __name__ == '__main__':
    unittest.main()
