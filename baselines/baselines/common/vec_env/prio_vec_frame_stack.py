from baselines.common.vec_env.vec_frame_stack import VecFrameStack
import numpy as np


class PrioVecFrameStack(VecFrameStack):
    def __init__(self, venv, nstack):
        super().__init__(venv, nstack)
        self.n_active_envs = self.venv.n_active_envs

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        active_idx = self.venv.active_envs
        for i, (env, new) in enumerate(zip(active_idx, news)):
            self.stackedobs[env] = np.roll(self.stackedobs[env], shift=-1, axis=-1)
            if new:
                self.stackedobs[env] = 0
            self.stackedobs[env,..., -obs.shape[-1]:] = obs[i]
        return self.stackedobs[active_idx], rews, news, infos

    def step_wait_exploration(self):
        res = self.venv.step_wait_exploration()
        if not res:
            return None
        obs, rews, news, infos, exploration_list = res
        for i, (env, new) in enumerate(zip(exploration_list, news)):
            self.stackedobs[env] = np.roll(self.stackedobs[env], shift=-1, axis=-1)
            if new:
                self.stackedobs[env] = 0
            self.stackedobs[env, ..., -obs.shape[-1]:] = obs[i]
        return self.stackedobs[exploration_list], rews, news, infos, exploration_list
