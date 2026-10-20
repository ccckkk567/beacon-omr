"""Production replay/input acceptance paths with controlled native/Ray fixtures."""
from copy import deepcopy
from dataclasses import asdict
from enum import Enum
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent_system.milestone_reuse.native_restart import RestartWorkerMixin, render_starts
from agent_system.milestone_reuse.persistence import PayloadStore
from agent_system.milestone_reuse.types import TaskSpec
from experiments.omr.tests.test_restart import Worker as WorldFixture, restart
from experiments.omr.tests.test_full_protocol import native_manager, config_for
from experiments.omr.tests.test_webshop_input import CharacterTokenizer
from experiments.omr.webshop_acceptance import check_full, check_baseline
from experiments.omr.root_protocol import dump_records


class Phase(Enum):
    SEARCH = 1
    DETAIL = 2


class Worker(RestartWorkerMixin, WorldFixture):
    def __init__(self):
        super().__init__()
        self.padding = 0
        self._adapter = None
    def get_restart_adapter(self):
        if self._adapter is None:
            self._adapter = restart.WebshopRestartAdapter(self, 'profile-1')
        return self._adapter
    def get_task_manifest(self):
        return {key: [asdict(task) for task in tasks] for key, tasks in self.get_restart_adapter().manifest.items()}
    def reset_task(self, *args):
        return self.get_restart_adapter().reset_task(*args)
    def restore_start(self, *args):
        return self.get_restart_adapter().restore(*args)
    def capture_start(self, manager):
        return self.get_restart_adapter().capture_state(manager)
    def reset(self, index):
        _, info = super().reset(index)
        text = f'WebShop [SEP] Instruction: [SEP] {self.env.instruction_text} [SEP] Search'
        self.prev_obs, self.env.prev_obs = text, [text]
        self.milestone_detector.phase = Phase.SEARCH
        self.milestone_detector.goal_option_values = set()
        info['available_actions'] = dict(has_search_bar=True, clickables=[])
        info['won'] = False
        return text, info
    def _step_native(self, action):
        _, reward, done, info = super()._step_native(action)
        text = 'Product B000 [SEP] blue [SEP] Buy Now ' + 'x' * self.padding
        self.prev_obs = text
        self.env.prev_obs[-1] = text
        self.env.browser.page_source = text
        self.milestone_detector.phase = Phase.DETAIL
        info.update(available_actions=dict(has_search_bar=False, clickables=['blue', 'buy now']),
                    executed_action=action, won=done)
        return text, reward, done, info
    def step(self, action):
        return self.get_restart_adapter().step(action)


class Remote:
    def __init__(self, worker):
        self.worker = worker
    def __getattr__(self, name):
        return SimpleNamespace(remote=getattr(self.worker, name))


class RecoveryCaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source_attempt = self.root / 'old'
        self.source_attempt.mkdir()
        self.config = config_for(self.root, 'webshop', 'full')
        self.config.omr.input_profile_id = 'webshop-input-v2'
        self.worker = Worker()
        self.remote = Remote(self.worker)
        self.manager = native_manager(self.config, [self.remote])
        self.manager.omr_tokenizer = CharacterTokenizer()
        task = self.worker.get_restart_adapter().manifest['train'][0]
        self.task = dict(id=task.instance_id, target=task.initialization['goal_index'])
        from experiments.omr.task_manifest import seed_for
        self.task['root_seed'] = seed_for('root', self.task['id'])
        self.trainer = SimpleNamespace(envs=self.manager, config=self.config, tokenizer=self.manager.omr_tokenizer,
            manifest={'splits': {'train': [self.task]}}, attempt=self.root / 'new')
        self.trainer.attempt.mkdir()
        self.transport = patch.dict(sys.modules, {'ray': SimpleNamespace(get=lambda refs: refs)})
        self.transport.start()
        self.addCleanup(self.transport.stop)

    def old_prefix(self):
        from experiments.omr.webshop_acceptance import local_manager
        packet = self.worker.omr_reset('webshop', self.task, None, 15, 'source')
        old = local_manager(self.trainer, 'webshop-input-v1')
        initial, _ = render_starts(old, 'webshop', [packet])
        raw, reward, done, info = self.worker.omr_step('search[blue]')
        old.memory.store({'text_obs': old.pre_text_obs, 'action': ['search[blue]']})
        old.pre_text_obs = old.format_obs([raw])
        text = old.build_text_obs(old.pre_text_obs, [info])[0]
        return old, initial, text, reward, done

    def test_actual_adapter_product_page_and_old_profile_history_pass_full_acceptance_without_new_root(self):
        old, initial, text, _, _ = self.old_prefix()
        state = dict(task_text=old.tasks[0], pre_text_obs=old.pre_text_obs[0], memory=deepcopy(old.memory[0]),
                     history_length=2, rendered_observation=text)
        payload = self.worker.omr_capture(state)
        self.assertNotIn('Instruction:', payload['expected']['observation'])
        with self.assertRaises(AssertionError):
            old.extract_task([payload['expected']['observation']])
        store = PayloadStore(self.root / 'payloads')
        ref = store.put(payload)
        entry = SimpleNamespace(restore_payload_ref=ref, task=TaskSpec('webshop', 'train', 'f', self.task['id'],
                    dict(target=self.task['target'], root_seed=self.task['root_seed'])))
        self.trainer.train_observer = SimpleNamespace(store=store,
            controller=SimpleNamespace(archive=SimpleNamespace(entries={'original-failed-cp': entry})))
        (self.source_attempt / 'train-error.json').write_text(json.dumps({'actual_start_ids': ['original-failed-cp']}))
        result = check_full(self.trainer, {'attempt': str(self.source_attempt)})
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['replayed_actions'], 1)
        self.assertEqual(result['history_actions'], 1)
        self.assertTrue(result['preview_equals_installed'])
        self.assertEqual(store.get(ref)['payload'], payload)
        self.assertEqual(len(self.worker.restart_adapter.trace), 2)

    def test_actual_prefix_reconstruction_requires_the_exact_old_token_count_and_fixed_prompt_budget(self):
        from agent_system.environments.webshop_input import chat_text
        old, initial, text, reward, done = self.old_prefix()
        self.worker.padding = 9910 - len(chat_text(self.trainer.tokenizer, text))
        self.assertGreater(self.worker.padding, 0)
        old, initial, text, reward, done = self.old_prefix()
        self.assertEqual(len(chat_text(self.trainer.tokenizer, text)), 9910)
        rows = [dict(slot=0, task_id=self.task['id'], local_step=0, input=initial['text'][0],
            next_observation=old.pre_text_obs[0], executed_action='search[blue]', reward=reward, done=done)]
        dump_records(self.source_attempt / 'rollouts/train-u000039-s0-b00001-steps.jsonl.gz', rows)
        result = check_baseline(self.trainer, {'attempt': str(self.source_attempt)})
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['original_tokens'], 9910)
        self.assertLessEqual(result['actual_tokens'], 7000)
        self.assertEqual(result['replayed_actions'], 1)


if __name__ == '__main__':
    unittest.main()
