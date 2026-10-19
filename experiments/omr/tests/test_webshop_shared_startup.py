"""Native text environment and reward equivalence; substitute JVM/NLP/Ray only."""
from pathlib import Path
import subprocess
import sys
import unittest


REPO = Path(__file__).resolve().parents[3]


class SharedStartupTests(unittest.TestCase):
    def test_native_catalog_rewards_sessions_and_ray_data_flow(self):
        script = Path(__file__).with_name('webshop_startup_fixture.py')
        result = subprocess.run([sys.executable, '-B', str(script)], cwd=REPO,
                                text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + '\n' + result.stderr)
        self.assertIn('8 native startup checks passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
