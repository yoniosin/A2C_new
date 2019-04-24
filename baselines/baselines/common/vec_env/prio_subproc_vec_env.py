from .subproc_vec_env import SubprocVecEnv, _flatten_obs
import numpy as np

LIMIT_TIME = 1000000000

class PrioSubprocVecEnv(SubprocVecEnv):
    def __init__(self, n_active_envs, env_fns, time_limit):
        super().__init__(env_fns, n_active_envs=n_active_envs)
        self.nenvs = len(env_fns)
        self.n_active_envs = n_active_envs
        self.active_envs = None
        self.set_active_envs(list(range(self.n_active_envs)))
        self.time_from_last_activation = np.zeros(self.nenvs)
        self.exploration_list = []
        self.time_limit = time_limit

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
            self.time_from_last_activation[i] = 0
        for env in set(list(range(self.nenvs))).difference(set(self.active_envs)):
            self.time_from_last_activation[env] += 1
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def step_async_exploration(self):
        self.exploration_list = [e for e in range(self.nenvs) if self.time_from_last_activation[e] > self.time_limit]
        map(lambda env: self.remotes[env].send(('random_step', None)), self.exploration_list)

        # if self.exploration_list:
        #     print('activate envs              ' + str(self.active_envs))
        #     print('exploration envs           ' + str(self.exploration_list))
        #     print('time from last activation  ' + str(self.time_from_last_activation))

    def step_wait_exploration(self):
        results = []
        for i in self.exploration_list:
            results.append(self.remotes[i].recv())
        if not self.exploration_list:
            return None
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos, self.exploration_list

    def zero_time_from_last_activation(self):
        self.time_from_last_activation[self.exploration_list] = 0


