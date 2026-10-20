from copy import deepcopy
import ast
import json
import os
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from omegaconf import OmegaConf

from experiments.omr.webshop_migration import SOURCES, read_request, check_invariants, check_archive_evidence, authorize
from experiments.omr.latest_checkpoint import file_sha256
from experiments.omr.task_manifest import digest
from experiments.omr.train_entry import compose_config

TARGET = 'b' * 40


class MigrationTests(unittest.TestCase):
    def payload(self, root, method, source, attempt):
        run = root / 'analysis/beacon-omr' / f'omr-v2-webshop-{method}-s0-{SOURCES[method]["tag"]}'
        return dict(root=str(root), benchmark='webshop', method=method, source_commit=source,
            model={'path': '/model', 'revision': 'pinned'}, webshop={'paths': {}}, simulator_python=None,
            prepared_source='prepared', run_id=run.name, run_dir=str(run), attempt_dir=str(run / 'attempts' / attempt))

    def test_only_declared_input_source_and_monitoring_differences_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for method in SOURCES:
                old = compose_config(dict(self.payload(root, method, SOURCES[method]['source'], 'old'), input_profile_id='webshop-input-v1'))
                new = compose_config(self.payload(root, method, TARGET, 'new'))
                self.assertTrue(check_invariants(old, new))
                for path, value in [('env.max_steps', 16), ('data.max_prompt_length', 9000),
                                    ('actor_rollout_ref.actor.optim.lr', 0.01), ('omr.method', 'wrong'),
                                    ('actor_rollout_ref.model.path', '/wrong-model'), ('env.rollout.n', 4),
                                    ('algorithm.migpo.gamma', .5), ('omr.execution_profile_id', 'perturn-active-v1'),
                                    ('omr.input_profile_id', 'webshop-input-v1')]:
                    changed = deepcopy(new)
                    OmegaConf.update(changed, path, value)
                    with self.subTest(method=method, path=path), self.assertRaises(ValueError):
                        check_invariants(old, changed)

    def test_actual_frozen_source_composers_match_target_scientific_settings(self):
        repo = Path(__file__).resolve().parents[3]
        for method, spec in SOURCES.items():
            source = subprocess.check_output(['git', '-C', str(repo), 'show',
                spec['source'] + ':experiments/omr/train_entry.py'], text=True, encoding='utf-8')
            tree = ast.parse(source)
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compose_config')
            namespace = dict(Path=Path, os=os, digest=digest, __file__=str(repo / 'experiments/omr/train_entry.py'))
            exec(compile(ast.Module([function], type_ignores=[]), '<frozen-source-composer>', 'exec'), namespace)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                payload = self.payload(root, method, spec['source'], 'old')
                old = namespace['compose_config'](payload)
                new = compose_config(self.payload(root, method, TARGET, 'new'))
                self.assertTrue(check_invariants(old, new))

    def test_request_rejects_other_checkpoint_source_target_arm_and_input_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'request.json'
            valid = dict(schema=1, root=str(root), method='baseline', target_commit=TARGET,
                         input_profile_id='webshop-input-v2', **SOURCES['baseline'])
            path.write_text(json.dumps(valid))
            self.assertEqual(read_request(path, root, 'baseline', TARGET), valid)
            for field, value in [('source', 'c' * 40), ('update', 39), ('method', 'full'),
                                 ('complete_sha256', 'd' * 64), ('target_commit', 'e' * 40),
                                 ('input_profile_id', 'unknown')]:
                path.write_text(json.dumps(dict(valid, **{field: value})))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    read_request(path, root, 'baseline', TARGET)

    def test_untried_archive_preserves_state_but_nonzero_conditional_evidence_is_rejected(self):
        entry = dict(checkpoint_id='state', first_trial_update=None, total_selected_groups=0)
        controller = dict(committed_update=10, archive={'entries': [entry]}, statistics={'events': {}, 'discovery': {}})
        state = dict(archive_enabled=True, update=10, attempt='/fixture/run/attempts/old',
                     omr=dict(controller=controller, payload_refs=['ref'], candidate_rng='kept'))
        with patch('agent_system.milestone_reuse.persistence.PayloadStore') as store:
            store.return_value.dependency_closure.return_value = {'ref'}
            before = deepcopy(state)
            self.assertEqual(check_archive_evidence(state)['entries_preserved'], 1)
            self.assertEqual(state, before)
            state['omr']['controller']['statistics']['events']['state'] = {'Goal': {'trials': 1, 'successes': 0}}
            with self.assertRaisesRegex(ValueError, 'conditional evidence'):
                check_archive_evidence(state)

    def test_production_authorization_binds_complete_launch_signature_assets_and_protected_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_payload = self.payload(root, 'baseline', SOURCES['baseline']['source'], 'old')
            new_payload = self.payload(root, 'baseline', TARGET, 'new')
            old_config = compose_config(dict(old_payload, input_profile_id='webshop-input-v1'))
            config = compose_config(new_payload)
            run, attempt = Path(new_payload['run_dir']), Path(new_payload['attempt_dir'])
            for payload in (old_payload, new_payload):
                path = Path(payload['attempt_dir'])
                path.mkdir(parents=True)
                (path / 'launch.json').write_text(json.dumps(payload))
            request = dict(schema=1, root=str(root), method='baseline', target_commit=TARGET,
                           input_profile_id='webshop-input-v2', **SOURCES['baseline'])
            request_path = root / 'request.json'
            request_path.write_text(json.dumps(request))
            config.omr.migration_request = str(request_path)
            manifest, schedule = {'sha256': 'manifest'}, [['task']]
            signature = digest([old_config.omr.protocol_signature, manifest['sha256'], schedule])
            state = dict(update=35, task_cursor=35, archive_enabled=False,
                source_commit=SOURCES['baseline']['source'], signature=signature,
                attempt=old_payload['attempt_dir'], manifest=manifest, schedule_sha256=digest(schedule))
            generation = run / 'checkpoints/global_step_000035'
            generation.mkdir(parents=True)
            for name in ('protocol.pt', 'data.pt', 'COMPLETE.json'):
                (generation / name).write_bytes(b'fixture')
            # A synthetic COMPLETE hash is explicitly substituted in this CPU
            # fixture; production requires the real user-reported hash.
            complete = dict(update=35, files={'actor/model': {}}, metadata=dict(signature=signature,
                source_commit=state['source_commit'], manifest_sha256=manifest['sha256']))
            trainer = SimpleNamespace(config=config, manifest=manifest, schedule=schedule,
                run_dir=run, attempt=attempt, signature=digest([config.omr.protocol_signature, manifest['sha256'], schedule]))
            module = SimpleNamespace(compose_config=lambda payload: old_config)
            loader = SimpleNamespace(exec_module=lambda obj: None)
            spec = SimpleNamespace(loader=loader)
            real_hash = file_sha256
            def hashes(path):
                return request['complete_sha256'] if Path(path).name == 'COMPLETE.json' else real_hash(path)
            with patch('experiments.omr.resume_chain.clean_revision'), \
                 patch('experiments.omr.webshop_migration.subprocess.check_output', return_value='same-blob'), \
                 patch('experiments.omr.webshop_migration.importlib.util.spec_from_file_location', return_value=spec), \
                 patch('experiments.omr.webshop_migration.importlib.util.module_from_spec', return_value=module), \
                 patch('experiments.omr.webshop_migration.file_sha256', side_effect=hashes):
                permit = authorize(trainer, state, complete, generation)
                self.assertEqual(permit['restored_update'], 35)
                self.assertEqual(permit['source_signature'], signature)
                self.assertTrue((attempt / 'migration-permit.json').is_file())
                broken = deepcopy(state)
                broken['schedule_sha256'] = 'wrong'
                with self.assertRaisesRegex(ValueError, 'binding differs'):
                    authorize(trainer, broken, complete, generation)
                new_payload['model'] = {'path': '/wrong'}
                (attempt / 'launch.json').write_text(json.dumps(new_payload))
                with self.assertRaisesRegex(ValueError, 'asset identity'):
                    authorize(trainer, state, complete, generation)


if __name__ == '__main__':
    unittest.main()
