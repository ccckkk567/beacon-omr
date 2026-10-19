"""One native interface acceptance in the formal allocation, never training data."""
from copy import deepcopy
import importlib.metadata
import inspect
from pathlib import Path


def native_identity_check(task):
    from agent_system.environments.env_package.alfworld.envs import AlfworldWorker, load_config_file
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import AlfredInfos
    from agent_system.milestone_reuse.native_restart import NativeReplay
    import textworld
    path = Path(inspect.getfile(AlfworldWorker)).parent / 'configs/config_tw.yaml'
    config = load_config_file(path)
    for method in ('rl', 'dagger'):
        config[method]['training']['max_nb_steps_per_episode'] = 2 ** 31 - 1
    specification = {'train_eval': 'train', 'game_files': [task['target']]}
    workers = []
    result = {'schema': 1, 'evidence_level': 'native_simulator',
        'textworld_version': importlib.metadata.version('textworld'),
        'textworld_module': textworld.__file__, 'wrapper_source': inspect.getfile(AlfredInfos),
        'worker_source': inspect.getfile(AlfworldWorker), 'requested_task': task, 'checks': []}
    try:
        source = AlfworldWorker(config, task['root_seed'], deepcopy(specification))
        workers.append(source)
        replay = NativeReplay(source, 'alfworld', task, 30)
        obs, info = replay.reset()
        native = getattr(source.env, 'unwrapped', source.env)
        result.update(batch_size=native.batch_size, wrapper_type=type(source.env).__name__,
                      auto_reset=getattr(native, 'auto_reset', 'unavailable'))
        result['checks'].append({'phase': 'reset', 'info': deepcopy(info), 'observation': obs})
        command = 'look' if 'look' in info['admissible_commands'] else info['admissible_commands'][0]
        replay.step(command)
        result['checks'].append({'phase': 'step', 'info': deepcopy(replay.info)})
        payload = replay.capture({'memory': [], 'pre_text_obs': replay.observation})
        replay.step(command)
        target = AlfworldWorker(config, task['root_seed'], deepcopy(specification))
        workers.append(target)
        restored = NativeReplay(target, 'alfworld', task, 30)
        restored.restore(payload)
        for _ in range(2):
            restored.step(command)
        result.update(status='passed', replayed_actions=restored.replayed,
                      continuation_steps=2, raw_gamefile=restored.info['extra.gamefile'],
                      raw_type=type(restored.info['extra.gamefile']).__name__)
        return result
    finally:
        for worker in workers:
            worker.env.close()


def check_in_allocation(config, task):
    import ray
    from .latest_checkpoint import atomic_json
    from .task_manifest import seed_for
    options = {'num_cpus': 1}
    executable = config.env.alfworld.get('worker_python')
    if executable:
        options['runtime_env'] = {'py_executable': executable}
    task = dict(task, root_seed=seed_for('root', task['id']))
    result = ray.get(ray.remote(**options)(native_identity_check).remote(task))
    atomic_json(Path(config.omr.attempt_dir) / 'alfworld-native-identity.json', result)
    return result
