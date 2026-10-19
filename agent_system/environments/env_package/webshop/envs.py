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

import ray
import gym
import numpy as np
from copy import deepcopy
from agent_system.milestone_reuse.identity import webshop_goal_id

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

from agent_system.milestone_reuse.native_restart import RestartWorkerMixin


class WebshopWorker(RestartWorkerMixin):
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs, static_catalog=None):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        import random
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)

        env_kwargs = deepcopy(env_kwargs)
        if static_catalog is not None:
            from .shared import settings
            if static_catalog['binding'] != settings(env_kwargs):
                raise ValueError('Shared WebShop preparation settings differ')
            env_kwargs['prepared_catalog'] = static_catalog['catalog']
            env_kwargs['title_tokens'] = static_catalog['title_tokens']
            self._restart_profile = static_catalog['restart_profile']
            self._restart_manifest_cache = (self._restart_profile, static_catalog['restart_manifest'],
                                            static_catalog['restart_tasks'])
        # The goal permutation defines train/test membership. It must not vary
        # with the seed of a worker, group, or validation actor.
        env_kwargs['seed'] = env_kwargs.pop('goal_manifest_seed', 0)
        previous_rng = random.getstate()
        try:
            # Products/prices/goals themselves consume randomness before the
            # simulator shuffles goals. Seed the complete manifest construction.
            random.seed(env_kwargs['seed'])
            # Gym 0.24's generic checker can execute extra reset/step calls;
            # WebShop has a text-action API and is validated by native fixtures.
            self.env = gym.make('WebAgentTextEnv-v0', disable_env_checker=True, **env_kwargs)
        finally:
            random.setstate(previous_rng)

        # MIGPO: 里程碑检测器相关
        self.milestone_detector = None
        self.prev_obs = None
        self.restart_adapter = None

    def get_restart_adapter(self):
        from .restart import WebshopRestartAdapter
        if not hasattr(self, '_restart_profile'):
            from .shared import restart_profile
            env = self.env.unwrapped
            self._restart_profile = restart_profile(env.server.product_item_dict, env.server.product_prices,
                dict(env.kwargs, observation_mode=env.observation_mode, show_attrs=env.server.show_attrs,
                     num_prev_obs=env.num_prev_obs, num_prev_actions=env.num_prev_actions))
        return WebshopRestartAdapter(self, self._restart_profile)

    def get_task_manifest(self):
        from dataclasses import asdict
        return {split: [asdict(task) for task in tasks]
                for split, tasks in self.get_restart_adapter().manifest.items()}

    def reset_task(self, task, assignment_token):
        return self.get_restart_adapter().reset_task(task, assignment_token)

    def restore_start(self, payload, assignment_token, new_rollout_horizon):
        return self.get_restart_adapter().restore(payload, assignment_token, new_rollout_horizon)

    def capture_start(self, manager_state):
        if self.restart_adapter is None:
            raise RuntimeError('Explicit task initialization is required before capture')
        return self.restart_adapter.capture_state(manager_state)
    
    def step(self, action):
        if self.restart_adapter is not None:
            return self.restart_adapter.step(action)
        return self._step_native(action)

    def _step_native(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward
        info['executed_action'] = action
        info['goal_id'] = self.goal_id

        # MIGPO: 里程碑检测
        if self.milestone_detector is not None:
            verbose_info = None
            if done:
                session_id = self.env.session
                verbose_info = self.env.server.user_sessions.get(session_id, {}).get('verbose_info')

            result = self.milestone_detector.process(
                action=action,
                state=self.prev_obs,
                next_state=obs,
                info={'verbose': verbose_info} if verbose_info else None
            )

            info['milestone_phase'] = result.phase.value
            info['milestone_achieved'] = result.achieved

        self.prev_obs = obs

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        self.restart_adapter = None
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False

        # MIGPO: 初始化里程碑检测器
        from migpo.webshop_milestone_detector import MilestoneDetector
        session_id = self.env.session
        goal_data = self.env.server.user_sessions[session_id]['goal']
        self.goal_id = webshop_goal_id(goal_data)
        info['goal_id'] = self.goal_id
        info['goal_index'] = idx
        self.milestone_detector = MilestoneDetector(
            goal_data,
            self.env.server.product_item_dict,
            self.env.server.product_prices
        )
        self.prev_obs = obs

        # MIGPO: 初始里程碑信息
        info['milestone_phase'] = self.milestone_detector.phase.value
        info['milestone_achieved'] = False

        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered

    def reset_explicit(self, task):
        from experiments.omr.root_protocol import seed_root
        seed_root(task['root_seed'])
        obs, info = self.reset(task['target'])
        if info['goal_id'] != task['id']:
            raise ValueError('Native WebShop reset loaded a different goal identity')
        return obs, info
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = dict(env_kwargs) if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}
        self._static_catalog_ref = self._env_kwargs.pop('_static_catalog_ref', None)

        # -------------------------- Ray actors setup --------------------------
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        for i in range(self.num_processes):
            # A top-level ref is resolved by Ray before actor construction.
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs, self._static_catalog_ref)
            self._workers.append(worker)

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))
            
        print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        from experiments.omr.root_protocol import step_workers

        # Collect results
        results = step_workers(self, self._workers, actions)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        tasks = getattr(self, 'omr_tasks', None)
        idx = (np.repeat(self._rng.choice(self.goal_idxs, size=self.env_num, replace=False),
                         self.group_n).tolist() if tasks is None else [None] * self.num_processes)

        # Send reset commands to all workers
        futures = []
        for slot, (worker, i) in enumerate(zip(self._workers, idx)):
            future = worker.reset.remote(i) if tasks is None else worker.reset_explicit.remote(tasks[slot])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
