# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ray
from agent_system.milestone_reuse.native_restart import RestartWorkerMixin


def get_environment(env_type):
    # Importing the vendored package also imports installed alfworld.info.
    # With worker_python configured, only the simulator runtime owns ALFWorld
    # and TextWorld; the learner must be able to import/serialize this module.
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment as resolve
    return resolve(env_type)

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    import torch
    import torchvision.transforms as T

    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker(RestartWorkerMixin):
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        if isinstance(base_env, dict):
            # The simulator can reuse its own Python packages while the GPU
            # learner uses the pinned Torch/vLLM training stack.
            specification = base_env
            cls = get_environment(config['env']['type'])
            base_env = cls.__new__(cls)
            base_env.config = config
            base_env.train_eval = specification['train_eval']
            base_env.game_files = specification['game_files']
            base_env.num_games = len(base_env.game_files)
            base_env.use_expert = False
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        self._omr_local_step = getattr(self, '_omr_local_step', 0) + 1
        infos['omr_local_step'] = [self._omr_local_step]
        infos['omr_reset_epoch'] = [getattr(self, '_omr_reset_epoch', 0)]
        infos['executed_action'] = [action]
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        self._omr_reset_epoch = getattr(self, '_omr_reset_epoch', 0) + 1
        self._omr_local_step = 0
        infos['omr_reset_epoch'] = [self._omr_reset_epoch]
        infos['omr_local_step'] = [0]
        infos['observation_text'] = obs
        return obs, infos

    def reset_explicit(self, task):
        from experiments.omr.root_protocol import seed_root
        seed_root(task['root_seed'])
        self.env.seed(task['root_seed'])
        # TextWorld 1.7 returns TextworldBatchGymEnv directly (no Gym base or
        # .unwrapped). Older Gym-wrapped instances expose the same iterator on
        # their unwrapped environment. Always assign on its actual owner.
        native = getattr(self.env, 'unwrapped', self.env)
        if native.batch_size != 1:
            raise ValueError('Explicit ALFWorld tasks require one game per worker')
        previous = native._gamefiles_iterator
        native._gamefiles_iterator = iter([task['target']])
        try:
            obs, infos = self.reset()
            from agent_system.milestone_reuse.episode_identity import verify_gamefile
            verify_gamefile(infos.get('extra.gamefile'), task['target'], task_id=task['id'],
                            identity_source='native_reset', reset_epoch=self._omr_reset_epoch)
            return obs, infos
        finally:
            native._gamefiles_iterator = previous
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        if env_kwargs.get('omr_protocol'):
            # The collector owns H. Keep real world time/termination, removing
            # only TextWorld's separate artificial collection cutoff.
            for method in ('rl', 'dagger'):
                config[method]['training']['max_nb_steps_per_episode'] = 2 ** 31 - 1
        env_type = config['env']['type']
        split = 'train' if is_train else eval_dataset
        worker_python = env_kwargs.get('worker_python')
        actor_options = dict(resources_per_worker)
        if worker_python:
            actor_options['runtime_env'] = {'py_executable': worker_python}
            catalog = ray.remote(num_cpus=1, runtime_env={'py_executable': worker_python})(alfworld_catalog)
            self.game_files = ray.get(catalog.remote(config, split))
            base_env = {'train_eval': split, 'game_files': self.game_files}
        else:
            base_env = get_environment(env_type)(config, train_eval=split)
            self.game_files = list(base_env.game_files)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**actor_options)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        from experiments.omr.root_protocol import step_workers

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = step_workers(self, self.workers, actions)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info['task_score'] = float(scores[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        tasks = getattr(self, 'omr_tasks', None)
        for index, worker in enumerate(self.workers):
            future = worker.reset.remote() if tasks is None else worker.reset_explicit.remote(tasks[index])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)


def alfworld_catalog(config, split):
    return get_environment(config['env']['type'])(config, train_eval=split).game_files
