from baselines.common.vec_env.vec_normalize import VecNormalize
import numpy as np


class PrioVecNormalize(VecNormalize):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """
    def step_wait(self):
        active_idx = self.venv.active_envs
        obs, rews, news, infos = self.venv.step_wait()
        for i, r in zip(active_idx, rews):
            self.ret[i] *= self.gamma
            self.ret[i] += r
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret[active_idx])
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        for i, new in zip(active_idx, news):
            if new:
                self.ret[i] = 0.
        return obs, rews, news, infos

    def step_wait_exploration(self):
        res = self.venv.step_wait_exploration()
        if not res:
            return None
        obs, rews, news, infos, exploration_list = res
        for i, r in zip(exploration_list, rews):
            self.ret[i] *= self.gamma
            self.ret[i] += r
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret[exploration_list])
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        for i, new in zip(exploration_list, news):
            if new:
                self.ret[i] = 0.
        return obs, rews, news, infos, exploration_list

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)

    @property
    def n_active_envs(self):
        return self.venv.n_active_envs
