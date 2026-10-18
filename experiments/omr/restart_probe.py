"""Real WebShop replay/continuation acceptance; fixture actions never train a policy."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace


def webshop(payload):
    from agent_system.environments.env_package.webshop.envs import WebshopWorker
    from agent_system.environments.env_package.webshop.projection import webshop_projection
    from agent_system.environments.env_package.webshop.restart import (
        capture_manager_slot, restore_manager_slots, RestoreMismatch)
    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.milestone_reuse.types import content_id

    class SingleWorker:
        def __init__(self, worker):
            self.worker = worker

        def step(self, actions):
            obs, reward, done, info = self.worker.step(actions[0])
            return [obs], [reward], [done], [info]

    def make_manager(worker):
        return WebshopEnvironmentManager(SingleWorker(worker), webshop_projection,
                                         SimpleNamespace(env=SimpleNamespace(history_length=2)))

    def act(manager, command):
        return manager.step([f'<think>Test-only fixture.</think><action>{command}</action>'])

    workers = []
    binding = payload['binding']
    try:
        for seed in (0, 1000):
            workers.append(WebshopWorker(seed, dict(binding['paths'], num_products=binding['num_products'],
                observation_mode='text', human_goals=False, goal_manifest_seed=0)))
        source, target = workers
        manifest = source.get_task_manifest()
        if manifest != target.get_task_manifest():
            raise RuntimeError('Independent workers disagree on task manifests')
        task = manifest['train'][0]
        obs, info = source.reset_task(task, 'fixture-source')
        manager = make_manager(source)
        restore_manager_slots(manager, [obs], [info])
        native = source.env.unwrapped
        query = native.server.user_sessions[native.session]['goal']['query']
        rendered, *_ = act(manager, f'search[{query}]')
        search_payload = source.capture_start(capture_manager_slot(manager, 0, rendered['text'][0]))
        product = next((name for name in source.restart_adapter.info['available_actions']['clickables']
                        if name.upper() in native.server.product_item_dict), None)
        if product is None:
            raise RuntimeError('Fixture search returned no product')
        rendered, *_ = act(manager, f'click[{product}]')
        # Exercise a native option when present, including the counter/session state.
        options = native.server.product_item_dict[product.upper()].get('options', {})
        available = source.restart_adapter.info['available_actions']['clickables']
        option = next((str(value).lower() for values in options.values() for value in values
                       if str(value).lower() in available), None)
        if option is not None:
            rendered, *_ = act(manager, f'click[{option}]')
        detail_payload = source.capture_start(capture_manager_slot(manager, 0, rendered['text'][0]))
        receipts = []
        for label, state in (('search', search_payload), ('detail', detail_payload)):
            obs, restored_info = target.restore_start(state, 'fixture-' + label, 15)
            target_manager = make_manager(target)
            restored = restore_manager_slots(target_manager, [obs], [restored_info])
            assert restored['text'][0] == state['manager']['rendered_observation']
            assert not restored_info['milestone_achieved']
            assert restored_info['omr_start']['replay_actions'] == len(state['trace'])
            receipts.append(restored_info['omr_start'])
        # Restore source too, then compare actual continuation with an independent
        # target. Replay never calls manager.step, so history is not duplicated.
        obs, info = source.restore_start(detail_payload, 'fixture-source-continuation', 15)
        manager = make_manager(source)
        restore_manager_slots(manager, [obs], [info])
        before = deepcopy(detail_payload)
        left = act(manager, 'click[Description]')
        right = act(target_manager, 'click[Description]')
        if left[0] != right[0] or left[3] != right[3]:
            raise RuntimeError('Restored instances diverged on the same new action')
        assert detail_payload == before
        # A recursive prefix must include the old prefix exactly once.
        recursive = target.capture_start(capture_manager_slot(target_manager, 0, right[0]['text'][0]))
        assert len(recursive['trace']) == len(detail_payload['trace']) + 1
        source.restore_start(recursive, 'fixture-recursive', 15)
        # Full 15 new actions after a nonempty prefix; no inherited step cutoff.
        for _ in range(15):
            _, _, done, _ = source.step('invalid-fixture-command')
            if done:
                raise RuntimeError('Nonterminal replay consumed the new-action horizon')
        # Altering one independent instance must not alter the other's state.
        assert len(target.restart_adapter.trace) == len(recursive['trace'])
        corrupt = deepcopy(detail_payload)
        corrupt['trace'][0]['action'] = 'search[wrong fixture]'
        try:
            target.restore_start(corrupt, 'fixture-corrupt', 15)
        except RestoreMismatch:
            corruption_rejected = True
        else:
            raise RuntimeError('Corrupt restore payload was accepted')
        # Purchase is terminal, even when its reference reward is zero.
        obs, info = target.restore_start(detail_payload, 'fixture-terminal', 15)
        target_manager = make_manager(target)
        restore_manager_slots(target_manager, [obs], [info])
        result = act(target_manager, 'click[Buy Now]')
        if not bool(result[2][0]):
            raise RuntimeError('Purchase fixture did not terminate')
        try:
            target.capture_start(capture_manager_slot(target_manager, 0, result[0]['text'][0]))
        except ValueError:
            terminal_rejected = True
        else:
            raise RuntimeError('Terminal purchase was admitted as a continuing state')
        return {'restore_verified': True, 'test_only': True, 'training_ready': False,
                'manifest_digest': content_id(manifest),
                'split_counts': {key: len(value) for key, value in manifest.items()},
                'same_prompt_history': True, 'same_continuation': True,
                'independent_states': True, 'recursive_replay': True,
                'new_actions_after_restore': 15, 'option_exercised': option is not None,
                'corruption_rejected': corruption_rejected, 'terminal_rejected': terminal_rejected,
                'starts': receipts}
    finally:
        for worker in workers:
            searcher = worker.env.unwrapped.server.search_engine
            if callable(getattr(searcher, 'close', None)):
                searcher.close()
            worker.close()


def main():
    from experiments.omr.preparation import source_receipt
    from experiments.omr.latest_checkpoint import atomic_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['webshop', 'swanlab-online'])
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    source_receipt(payload['repo'], payload['prefix'])
    if args.action == 'webshop':
        result = webshop(payload)
    else:
        from experiments.omr.monitoring_readback import online_acceptance
        result = online_acceptance(payload)
    atomic_json(args.output, {'passed': True, 'result': result})


if __name__ == '__main__':
    main()
