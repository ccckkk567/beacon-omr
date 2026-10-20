"""Exercise production OMR collection with deterministic native-interface fixtures.

These fixtures verify integration invariants, not HPC/native simulator readiness.
"""
import ast
from copy import deepcopy
import json
import os
import random
from pathlib import Path
import runpy
import sys
import tempfile
import time
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from agent_system.memory import SimpleMemory
from agent_system.milestone_reuse.events import EventTrace, score_event
from agent_system.milestone_reuse.native_restart import NativeReplay, RestartWorkerMixin
from agent_system.milestone_reuse.types import content_id
from experiments.omr.full_protocol import FullCollectorObserver
from experiments.omr.root_protocol import step_workers
from experiments.omr.root_protocol import accumulated_cost
from experiments.omr.latest_checkpoint import LatestCheckpoint, atomic_json, file_sha256
from experiments.omr.task_manifest import digest
from experiments.omr.train_entry import compose_config

REPO = Path(__file__).resolve().parents[3]
TASK = {'id': 'science-fixture', 'family': '0', 'target': [0, 1], 'root_seed': 42}


class NativeScience(RestartWorkerMixin):
    def __init__(self):
        self.diverge = False
        self.root_calls = 0

    def info(self, action='', before=0):
        return {'task_num': 0, 'task_variation': 1, 'task_name': 'fixture', 'task_description': 'fixture task',
                'score': self.score, 'score_pre': before, 'score_post': self.score, 'task_score': self.score,
                'subgoal_completed': self.score > before, 'won': self.score == 100,
                'strict_success': self.score == 100, 'raw_completed': self.score == 100,
                'possible_actions': ['progress', 'advance', 'goal', 'wait'], 'available_actions': 'progress/advance/goal/wait',
                'executed_action': action}

    def reset_explicit(self, task):
        self.root_calls += 1
        self.score = self.prev_score = self.moves = 0
        return 'root', self.info()

    def step(self, action):
        if self.score == 100:
            raise RuntimeError('Native terminal mutated')
        self.moves += 1
        before = self.score
        self.score = {'progress': 25 + int(self.diverge), 'advance': 50, 'goal': 100}.get(action, self.score)
        self.prev_score = self.score
        return f'native observation {self.moves} score {self.score}', float(self.score > before), self.score == 100, self.info(action, before)


class Actor:
    def __init__(self, worker):
        self.worker = worker

    def __getattr__(self, name):
        return SimpleNamespace(remote=getattr(self.worker, name))


def config_for(root, benchmark='scienceworld', method='full'):
    config = compose_config({'benchmark': benchmark, 'method': method, 'source_commit': 'a' * 40,
        'model': {'path': '/existing/model'}, 'run_id': benchmark + method, 'run_dir': str(root),
        'attempt_dir': str(root / 'attempt'), 'root': str(root), 'simulator_python': None,
        'webshop': {'paths': {}}, 'input_profile_id': 'webshop-input-v1'})
    config.env.rollout.n = 2
    config.env.max_steps = 2
    config.omr.restart_fraction = 1.
    return config


def native_manager(config, actors):
    # Use the actual production manager renderer/step/history, with only native
    # worlds and Ray transport replaced by deterministic fixtures.
    tree = ast.parse((REPO / 'agent_system/environments/env_manager.py').read_text(encoding='utf-8'))
    web = config.omr.benchmark == 'webshop'
    name = 'WebshopEnvironmentManager' if web else 'SciWorldEnvironmentManager'
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    prompts = runpy.run_path(str(REPO / 'agent_system/environments/prompts' / ('webshop.py' if web else 'sciworld.py')))
    class Base:
        def __init__(self, envs, projection_f, config):
            self.envs, self.projection_f, self.config = envs, projection_f, config
    class Vector:
        def __init__(self):
            self._workers = actors
        def step(self, actions):
            return [list(values) for values in zip(*step_workers(self, self._workers, actions))]
    namespace = dict(EnvironmentManagerBase=Base, SimpleMemory=SimpleMemory, to_numpy=np.asarray, **prompts)
    module = ast.Module([ast.ImportFrom('__future__', [ast.alias('annotations')], 0), node], [])
    exec(compile(ast.fix_missing_locations(module), '<native-manager>', 'exec'), namespace)
    return namespace[name](Vector(), lambda actions: (actions, [True] * len(actions)), config)


class FullIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'attempt').mkdir()
        self.config = config_for(self.root)
        self.workers = [NativeScience(), NativeScience()]
        self.manager = native_manager(self.config, [Actor(w) for w in self.workers])
        tokenizer = SimpleNamespace(apply_chat_template=lambda chat, **kwargs: chat[0]['content'])
        class Tokenizer:
            apply_chat_template = staticmethod(tokenizer.apply_chat_template)
            def __call__(self, text, **kwargs):
                return {'input_ids': list(text.encode())}
        self.observer = FullCollectorObserver(self.config, self.manager, {'sha256': 'fixture'}, self.root / 'rollouts', Tokenizer())
        ray = ModuleType('ray')
        ray.get = lambda refs: refs
        self.patched = patch.dict(sys.modules, {'ray': ray})
        self.patched.start()
        self.addCleanup(self.patched.stop)

    def collect(self, update, actions):
        observer = self.observer
        observer.plan([TASK, TASK], update=update, split='train')
        obs, infos = observer.initialize()
        observer.reset(obs, infos)
        self.initial_memory = deepcopy(self.manager.memory._data)
        active = np.ones(2, dtype=bool)
        batches = []
        for step, action in enumerate(actions):
            next_obs, rewards, dones, infos = self.manager.step([action] * 2)
            batch = SimpleNamespace(batch={'responses': torch.ones(2, 2), 'attention_mask': torch.ones(2, 5)},
                non_tensor_batch={'uid': np.array(['group', 'group']), 'traj_uid': np.array([f'{update}-a', f'{update}-b']),
                    'analysis_step_id': np.array([f'{update}-a:{step}', f'{update}-b:{step}']),
                    'omr_synced_policy_version': np.array([update, update]),
                    'omr_weight_sync_sequence': np.array([step + 1, step + 1]),
                    'raw_response': np.array([action, action])})
            observer.record(batch, obs, next_obs, infos, rewards, dones, active, step)
            observer.flush()
            batches.append(batch)
            obs, active = next_obs, active & ~dones
        observer.finish()
        return batches

    def test_pending_next_update_suffix_only_statistics_and_resume(self):
        self.collect(0, ['progress', 'advance'])
        self.assertFalse(self.observer.controller.archive.entries)
        self.observer.commit(1)
        entry = min(self.observer.controller.archive.entries.values(), key=lambda e: e.initial_score)
        self.assertNotIn(entry.checkpoint_id, self.observer.controller.statistics.events)
        self.assertIn(score_event(50), entry.event_ids)
        self.assertNotIn(score_event(25), entry.event_ids)
        snapshot = json.loads(json.dumps(self.observer.state_dict()))
        self.observer.checkpoint_committed()
        batches = self.collect(1, ['wait', 'goal'])
        entry = self.observer.controller.archive.entries[self.observer.actual_ids[0]]
        prefix = entry.prefix_action_count
        self.assertEqual(len(self.initial_memory[0]), prefix)
        self.assertEqual(len(self.observer.steps), 4)  # Only two new actions x two workers.
        self.assertEqual(self.observer.compute['replayed_action_steps'], 2 * prefix)
        self.assertEqual(self.workers[0].moves, prefix + 2)
        self.assertTrue(all(row['start_mode'] == 'restart' for row in self.observer.summary))
        self.assertEqual(batches[0].non_tensor_batch['start_id'].tolist(), [entry.checkpoint_id] * 2)
        self.observer.commit(2)
        view = self.observer.controller.statistics.view(entry.checkpoint_id, entry.event_ids, 2)
        self.assertAlmostEqual(view['events']['Goal']['N'], 2 * 2 ** (-1/20))
        self.assertTrue(self.observer.controller.archive.entries[entry.checkpoint_id].first_trial_update == 1)
        # Rollback restores sampler/archive/thresholds, while the old complete
        # checkpoint's external payloads remain protected from GC.
        self.observer.load_state_dict(snapshot)
        self.assertEqual(self.observer.controller.committed_update, 1)
        self.assertNotIn(entry.checkpoint_id, self.observer.controller.statistics.events)
        self.assertEqual(set(snapshot['payload_refs']), self.observer.latest_payload_refs)

    def test_one_bad_restore_falls_back_whole_group_once_and_quarantines(self):
        self.collect(0, ['progress'])
        self.observer.commit(1)
        key = next(iter(self.observer.controller.archive.entries))
        self.workers[1].diverge = True
        self.collect(1, ['wait'])
        self.assertEqual(self.observer.actual_ids, [None, None])
        self.assertEqual(self.observer.compute['restore_failed_groups'], 1)
        self.assertEqual(self.observer.controller.archive.entries[key].validity_status, 'restore_failed')
        self.assertEqual([w.root_calls for w in self.workers], [3, 3])
        self.observer.commit(2)
        self.assertNotIn(key, self.observer.controller.statistics.events)

    def test_corrupt_active_payload_prevents_resume(self):
        self.collect(0, ['progress'])
        self.observer.commit(1)
        state = self.observer.state_dict()
        self.observer.store._path(state['payload_refs'][0]).write_bytes(b'corrupt')
        with self.assertRaises(Exception):
            self.observer.load_state_dict(state)

    def test_complete_model_checkpoint_carries_matching_archive_and_sampler(self):
        from omegaconf import open_dict
        with open_dict(self.config.omr):
            self.config.omr.writer_profile_id = 'bounded-v1'
        self.collect(0, ['progress'])
        self.observer.commit(1)
        tree = ast.parse((REPO / 'experiments/omr/training.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ProtocolTrainer')
        namespace = dict(globals(), RayPPOTrainer=object, __package__='experiments.omr', __spec__=None)
        exec(compile(ast.Module([node], []), 'training.py', 'exec'), namespace)
        trainer = namespace['ProtocolTrainer'].__new__(namespace['ProtocolTrainer'])
        class Actors:
            world_size = 2
            def save_checkpoint(self, path, remote, update, max_ckpt_to_keep):
                Path(path).mkdir()
                for rank in range(2):
                    for kind in ('model', 'optim', 'extra_state'):
                        torch.save({'update': update}, Path(path) / f'{kind}_world_size_2_rank_{rank}.pt')
            def load_checkpoint(self, path, del_local_after_load):
                self.loaded_update = torch.load(Path(path) / 'model_world_size_2_rank_0.pt')['update']
        class Loader:
            def state_dict(self):
                return {'cursor': 16}
            def load_state_dict(self, state):
                self.cursor = state['cursor']
        trainer.run_dir, trainer.attempt = self.root, self.root / 'attempt'
        trainer.config = self.config
        trainer.checkpoints = LatestCheckpoint(self.root / 'checkpoints', 'full-fixture')
        trainer.actor_rollout_wg, trainer.train_dataloader = Actors(), Loader()
        trainer.signature, trainer.current_epoch = 'full-fixture', 0
        trainer.global_steps, trainer.last_saved, trainer.final_test_done = 1, -1, False
        trainer.cost, trainer.manifest, trainer.schedule = {}, {'sha256': 'fixture'}, []
        trainer.train_observer = self.observer
        trainer.logger = SimpleNamespace(logger={'swanlab': SimpleNamespace(status={
            'mode': 'online', 'run_id': 'fixture', 'last_logged_update': 0, 'last_log_sequence': 0})})
        expected = self.observer.state_dict()
        with patch('experiments.omr.latest_checkpoint.shutil.disk_usage', return_value=SimpleNamespace(free=10**12)):
            trainer._save_checkpoint()
        directory, _ = trainer.checkpoints.latest()
        state = torch.load(directory / 'protocol.pt', weights_only=False)
        self.assertTrue(state['archive_enabled'])
        self.assertEqual(state['omr'], expected)
        watermark = state['log_watermarks']['train_observer']
        self.assertEqual(watermark['enqueued'], watermark['durable'])
        self.assertTrue(watermark['chunks'])
        for chunk in watermark['chunks']:
            self.assertEqual(file_sha256(chunk['path']), chunk['sha256'])
        self.collect(1, ['wait'])
        self.observer.commit(2)
        trainer._load_checkpoint()
        self.assertEqual(trainer.actor_rollout_wg.loaded_update, 1)
        self.assertEqual(self.observer.controller.committed_update, 1)
        self.assertEqual(self.observer.state_dict(), expected)
        self.assertEqual(trainer.train_dataloader.cursor, 16)
        # New failed-attempt chunks never mutate the checkpoint's references.
        self.assertTrue(all(file_sha256(c['path']) == c['sha256'] for c in watermark['chunks']))
        Path(watermark['chunks'][0]['path']).write_bytes(b'corrupted scientific record')
        with self.assertRaisesRegex(ValueError, 'Immutable scientific chunk'):
            trainer._load_checkpoint()


class EventAndReplayTests(unittest.TestCase):
    def test_both_arms_keep_configured_webshop_history_above_old_character_cutoff(self):
        for method in ('baseline', 'full'):
            manager = native_manager(config_for(Path('/fixture'), 'webshop', method), [])
            manager.tasks = ['buy blue']
            manager.memory.reset(1)
            manager.memory.store({'text_obs': ['actual old observation'], 'action': ['search[blue]']})
            rendered = manager.build_text_obs(['detail ' * 3000], [{'available_actions':
                {'has_search_bar': False, 'clickables': ['blue']}}])[0]
            self.assertGreater(len(rendered), 13000)
            self.assertIn('search[blue]', rendered)

    def test_alfworld_worker_batch_shape_and_independent_actual_prefix(self):
        class Alf(RestartWorkerMixin):
            def reset_explicit(self, task):
                self.moves = 0
                self.task = task
                return ['Your task is to: put apple'], self.info()
            def info(self):
                return {'won': [False], 'extra.gamefile': [self.task['target']],
                        'admissible_commands': [['take apple', 'wait']]}
            def step(self, action):
                self.moves += 1
                return [f'You pick up apple at move {self.moves}'], [0.], [False], self.info()
        task = {'id': 'family_trial_T1', 'target': '/data/family/trial_T1/game.tw-pddl', 'root_seed': 0}
        source, target = Alf(), Alf()
        packet = source.omr_reset('alfworld', task, None, 30, 'source')
        self.assertTrue(packet['ok'])
        result = source.omr_step('take apple')
        self.assertEqual(result[-1]['omr_action_admissible'], [True])
        payload = source.omr_capture({'memory': [], 'pre_text_obs': result[0][0]})
        restored = target.omr_reset('alfworld', task, payload, 30, 'target')
        self.assertTrue(restored['ok'])
        self.assertEqual(restored['replay_actions'], 1)
        self.assertEqual(len(restored['manager']['memory']), 1)
        for _ in range(30):
            target.omr_step('wait')
        self.assertEqual((source.moves, target.moves), (1, 31))

    def test_webshop_training_mixin_uses_verified_browser_detector_adapter(self):
        from dataclasses import asdict
        from enum import Enum
        from experiments.omr.tests.test_restart import Worker, restart
        class Phase(Enum):
            SEARCH = 1
            DETAIL = 2
        class Web(Worker, RestartWorkerMixin):
            def get_restart_adapter(self):
                return restart.WebshopRestartAdapter(self, 'fixture-profile')
            def get_task_manifest(self):
                return {key: [asdict(task) for task in tasks]
                        for key, tasks in self.get_restart_adapter().manifest.items()}
            def reset_task(self, task, token):
                return self.get_restart_adapter().reset_task(task, token)
            def restore_start(self, state, token, horizon):
                return self.get_restart_adapter().restore(state, token, horizon)
            def capture_start(self, manager):
                return self.restart_adapter.capture_state(manager)
            def reset(self, index):
                result = super().reset(index)
                self.milestone_detector.phase = Phase.SEARCH
                self.milestone_detector.goal_option_values = set()
                return result
            def _step_native(self, action):
                result = super()._step_native(action)
                self.milestone_detector.phase = Phase.DETAIL
                return result
            def step(self, action):
                return self.restart_adapter.step(action)
        source, target = Web(), Web()
        task = {'id': content_id(source.goals[500]), 'target': 500, 'root_seed': 0}
        self.assertTrue(source.omr_reset('webshop', task, None, 15, 'source')['ok'])
        result = source.omr_step('search')
        self.assertEqual((result[-1]['omr_phase_before'], result[-1]['omr_phase_after']), (1, 2))
        state = source.omr_capture({'memory': [{'action': 'search'}]})
        restored = target.omr_reset('webshop', task, state, 15, 'target')
        self.assertTrue(restored['ok'])
        self.assertEqual(restored['replay_actions'], 1)
        self.assertEqual(restored['manager']['memory'], [{'action': 'search'}])
        self.assertFalse(target.omr_event_context()['web_options_required'])

    def test_score_events_activate_without_retroactive_failures_and_unknown_goal(self):
        trace = EventTrace('scienceworld', TASK, [25., 50.], initial_score=25.)
        info = dict(score_post=40., subgoal_completed=True, strict_success=None)
        self.assertEqual(trace.observe('progress', 'obs', info, False), score_event(40))
        self.assertNotIn(score_event(25), trace.outcomes())
        self.assertNotIn(score_event(40), trace.outcomes())
        self.assertIsNone(trace.outcomes()['Goal'])
        self.assertFalse(trace.outcomes()[score_event(50)])

    def test_webshop_optionless_detail_skip_and_goal_are_distinct(self):
        trace = EventTrace('webshop', {'id': 'goal'}, web_options_required=False)
        self.assertNotIn('options', trace.events)
        info = {'won': False, 'milestone_achieved': True, 'omr_phase_before': 2, 'omr_phase_after': 4}
        self.assertEqual(trace.observe('click[product]', 'detail', info, False), 'detail')
        self.assertTrue(trace.outcomes()['detail'])
        self.assertFalse(trace.outcomes()['Goal'])

    def test_alfworld_matched_string_without_native_progress_is_unknown(self):
        from migpo.milestone_loader import get_milestone_loader
        with patch.object(get_milestone_loader(), 'get_milestones', return_value=['take apple', 'put apple']):
            trace = EventTrace('alfworld', {'id': 'fixture'})
            self.assertIsNone(trace.observe('take apple', 'Nothing happens.',
                                          {'won': False, 'omr_action_admissible': False}, False))
            self.assertEqual(trace.offset, 1)
            self.assertIsNone(trace.outcomes()['milestone_0'])
            self.assertFalse(trace.categories)

    def test_native_replay_recursive_prefix_horizon_and_terminal_rejection(self):
        worker = NativeScience()
        replay = NativeReplay(worker, 'scienceworld', TASK, 30)
        replay.reset()
        replay.step('progress')
        payload = replay.capture({'memory': [], 'pre_text_obs': replay.observation})
        target = NativeReplay(NativeScience(), 'scienceworld', TASK, 30)
        manager = target.restore(json.loads(json.dumps(payload)))
        self.assertEqual(len(manager['memory']), 1)
        for _ in range(30):
            target.step('wait')
        self.assertEqual(len(target.trace), 31)
        recursive = target.capture(manager)
        replay.restore(recursive)
        self.assertEqual(len(replay.trace), 31)
        replay.step('goal')
        with self.assertRaises(ValueError):
            replay.capture(manager)

    def test_baseline_and_full_share_profiles_except_explicit_method_switch(self):
        for bench in ('alfworld', 'webshop', 'scienceworld'):
            base = config_for(Path('/fixture'), bench, 'baseline')
            full = config_for(Path('/fixture'), bench, 'full')
            for key in ('algorithm', 'actor_rollout_ref', 'env', 'data'):
                self.assertEqual(OmegaConf.to_container(base[key]), OmegaConf.to_container(full[key]))
            self.assertFalse(base.omr.enabled)
            self.assertTrue(full.omr.enabled)


if __name__ == '__main__':
    unittest.main()
