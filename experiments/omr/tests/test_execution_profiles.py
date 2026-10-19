import json
from pathlib import Path
import tempfile
import unittest
from experiments.omr.execution_profiles import resolve_profiles
from experiments.omr.train_entry import compose_config

REV = 'a' * 40


class ExecutionProfileTests(unittest.TestCase):
    def test_historical_defaults_and_independent_writer_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(resolve_profiles(root, REV), {
                'execution_profile_id': 'perturn-legacy-v1', 'writer_profile_id': 'sync-v1'})
            for execution in ('perturn-active-v1', 'perturn-legacy-v1'):
                for writer in ('bounded-v1', 'sync-v1'):
                    profile = dict(revision=REV, execution_profile_id=execution, writer_profile_id=writer)
                    (root / 'execution-profile.json').write_text(json.dumps(profile))
                    (root / 'repair-lineage.json').write_text(json.dumps(dict(profile, parent_attempt='old')))
                    resolved = resolve_profiles(root, REV)
                    self.assertEqual(resolved['parent_attempt'], 'old')
                    for bench in ('alfworld', 'webshop', 'scienceworld'):
                        for method in ('baseline', 'full'):
                            config = compose_config(dict(root=tmp, benchmark=bench, method=method,
                                run_id='fixture', run_dir=tmp, attempt_dir=tmp, source_commit=REV,
                                model=dict(path='/fixture/model', revision='model'), simulator_python=None,
                                webshop={'paths': {}}, **resolved))
                            self.assertEqual(config.omr.execution_profile_id, execution)
                            self.assertEqual(config.omr.writer_profile_id, writer)

    def test_conflicts_in_either_file_are_rejected_without_overwriting_history(self):
        for field, value in [('revision', 'b' * 40), ('execution_profile_id', 'perturn-active-v1'),
                             ('writer_profile_id', 'bounded-v1')]:
            for filename in ('execution-profile.json', 'repair-lineage.json'):
                with self.subTest(field=field, filename=filename), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    profile = dict(revision=REV, execution_profile_id='perturn-legacy-v1', writer_profile_id='sync-v1')
                    for name in ('execution-profile.json', 'repair-lineage.json'):
                        (root / name).write_text(json.dumps(dict(profile, **({field: value} if name == filename else {}))))
                    before = {p: p.read_bytes() for p in root.iterdir()}
                    with self.assertRaises(ValueError):
                        resolve_profiles(root, REV)
                    self.assertEqual(before, {p: p.read_bytes() for p in root.iterdir()})


if __name__ == '__main__':
    unittest.main()
