from .subproc_vec_env import SubprocVecEnv, _flatten_obs
import numpy as np
import random


class PrioSubprocVecEnv(SubprocVecEnv):
    def __init__(self, n_active_envs, env_fns):
        super().__init__(env_fns, n_active_envs=n_active_envs)
        self.nenvs = len(env_fns)
        self.n_active_envs = n_active_envs
        self.active_envs = None
        self.set_active_envs(list(range(self.n_active_envs)))

    def set_active_envs(self, active_idx):
        self.active_envs = active_idx

    def step_async(self, actions):
        self._assert_not_closed()
        for action, env_i in zip(actions, self.active_envs):
            remote = self.remotes[env_i]
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = []
        for i in self.active_envs:
            results.append(self.remotes[i].recv())
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos
