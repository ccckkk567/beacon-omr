import ast
from collections import defaultdict
from copy import deepcopy
import itertools
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from agent_system.milestone_reuse.identity import webshop_goal_catalog, webshop_goal_id
from experiments.omr.task_manifest import build_catalog, training_schedule
from experiments.omr.tests.test_restart import restart, Worker


REPO = Path(__file__).resolve().parents[3]


def duplicated_goals():
    goals = [{'instruction_text': f'goal {i}', 'asin': f'product-{i}', 'category': str(i % 3)}
             for i in range(600)]
    for index in (10, 500, 501):
        goals[index] = deepcopy(goals[0])
    goals[503] = deepcopy(goals[502])
    goals[504]['instruction_text'] = goals[502]['instruction_text']
    return goals


def native_catalog(goals, other=None):
    ray = ModuleType('ray')
    ray.get = lambda result: result
    def vector(rows):
        worker = SimpleNamespace(get_goals=SimpleNamespace(remote=lambda: deepcopy(rows)))
        return SimpleNamespace(_workers=[worker])
    with patch.dict(sys.modules, {'ray': ray}):
        return build_catalog('webshop', vector(goals), vector(goals if other is None else other))


class WebshopGoalCatalogTests(unittest.TestCase):
    def test_full_content_duplicates_have_one_identity_with_test_precedence_and_audit(self):
        goals = duplicated_goals()
        before = deepcopy(goals)
        pools, audit = webshop_goal_catalog(goals)
        self.assertEqual(goals, before)
        self.assertEqual(audit['native_counts'], {'train': 100, 'test': 500})
        self.assertEqual(audit['canonical_counts'], {'train': 97, 'test': 499})
        self.assertEqual(audit['dropped_occurrences'], {'train': 3, 'test': 1})
        self.assertEqual(audit['within_split_duplicate_occurrences'], {'train': 2, 'test': 1})
        self.assertEqual(audit['cross_split_goal_count'], 1)
        self.assertEqual(audit['removed_train_occurrences_overlapping_test'], 2)
        self.assertEqual(audit['duplicate_groups'][0]['native_indices'], [0, 10, 500, 501])
        ids = {split: {row['id'] for row in rows} for split, rows in pools.items()}
        self.assertFalse(ids['train'] & ids['test'])
        self.assertEqual(ids['test'], {webshop_goal_id(goal) for goal in goals[:500]})
        self.assertEqual(ids['train'] | ids['test'], {webshop_goal_id(goal) for goal in goals})
        # Same instruction with a different product stays a separate instance.
        self.assertNotEqual(webshop_goal_id(goals[502]), webshop_goal_id(goals[504]))
        self.assertTrue({502, 504} <= {row['target'] for row in pools['train']})
        for rows in pools.values():
            for row in rows:
                self.assertEqual(row['id'], webshop_goal_id(goals[row['target']]))
        self.assertEqual((pools, audit), webshop_goal_catalog(deepcopy(goals)))

    def test_native_synthetic_generator_can_emit_identical_goal_rows(self):
        path = REPO / 'agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'get_synthetic_goals')
        namespace = dict(defaultdict=defaultdict, itertools=itertools, random=random)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        product = {'asin': 'A', 'instruction_text': 'buy a shirt', 'instruction_attributes': ['cotton'],
                   'options': {'color': ['blue', 'blue']}, 'category': 'clothes', 'query': 'shirt',
                   'name': 'shirt', 'Title': 'shirt', 'product_category': 'clothes'}
        generated = namespace['get_synthetic_goals']([product], None)
        self.assertEqual(len(generated), 2)
        self.assertEqual(generated[0], generated[1])
        goals = duplicated_goals() + generated
        manifest = native_catalog(goals)
        self.assertEqual(manifest['provenance']['goal_canonicalization']['dropped_occurrences']['train'], 4)

    def test_baseline_and_full_adapter_share_canonical_targets_and_holdout_is_excluded(self):
        goals = duplicated_goals()
        manifest = native_catalog(goals)
        adapter_manifest = restart.task_manifest(goals, 'profile')
        canonical = {split: {task.instance_id: task.initialization['goal_index'] for task in tasks}
                     for split, tasks in adapter_manifest.items()}
        training_pool = manifest['splits']['train'] + manifest['splits']['validation']
        self.assertEqual({row['id']: row['target'] for row in training_pool}, canonical['train'])
        for row in manifest['splits']['test']:
            self.assertEqual(row['target'], canonical['test'][row['id']])
        ids = [row['id'] for rows in manifest['splits'].values() for row in rows]
        self.assertEqual(len(ids), len(set(ids)))
        held_out = {row['id'] for split in ('test', 'validation') for row in manifest['splits'][split]}
        schedule = training_schedule(manifest, 'webshop')
        self.assertFalse(held_out & {row['id'] for batch in schedule for row in batch})
        self.assertEqual(manifest, native_catalog(deepcopy(goals)))

    def test_worker_permutation_disagreement_is_still_rejected(self):
        goals = duplicated_goals()
        other = deepcopy(goals)
        other[20], other[21] = other[21], other[20]
        with self.assertRaisesRegex(ValueError, 'permutation'):
            native_catalog(goals, other)

    def test_restart_rejects_duplicate_alias_index_and_test_goal_as_train(self):
        worker = Worker()
        worker.goals = duplicated_goals()
        adapter = restart.WebshopRestartAdapter(worker, 'profile')
        task = next(task for task in adapter.manifest['train'] if task.initialization['goal_index'] == 502)
        obs, info = adapter.reset_task(task, 'assignment')
        self.assertEqual(info['goal_id'], task.instance_id)
        from dataclasses import replace
        alias = replace(task, initialization=dict(task.initialization, goal_index=503))
        with self.assertRaisesRegex(restart.RestoreMismatch, 'outside'):
            adapter.reset_task(alias, 'alias')
        wrong_split = replace(adapter.manifest['test'][0], split='train')
        with self.assertRaisesRegex(restart.RestoreMismatch, 'outside'):
            adapter.reset_task(wrong_split, 'leak')


if __name__ == '__main__':
    unittest.main()
