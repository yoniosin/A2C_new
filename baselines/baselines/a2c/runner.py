import time

import numpy as np
from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner
from baselines.a2c.prioritizer import PrioritizerFactory
from copy import deepcopy


class Runner(AbstractEnvRunner):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """

    def __init__(self, env, model, nsteps, gamma):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype
        self.exploration_steps = 0
        self.counter = np.zeros(env.num_envs)

    def run(self, **kwargs):
        # We initialize the lists that will contain the mb (memory buffer) of experiences
        # self.choose_envs_to_activate(**kwargs)
        mb = MemoryBuffer()
        mb.states = self.states
        epinfos = []
        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, states, _ = self.model.step(self.obs, S=self.states, M=self.dones)

            # Append the experiences
            mb.append(obs=self.obs, a=actions, v=values, dones=self.dones)

            # Take actions in env and look the results
            # time.sleep(.002)
            obs, rewards, dones, infos, exploration_res = self.env.step(actions, self.exploration_steps)
            self.update_all_envs_exploration(exploration_res)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0
            self.obs = obs
            mb.append(r=rewards)
        mb.append(dones=self.dones)

        # Batch of steps to batch of rollouts
        mb.batch_to_rollouts(self.ob_dtype, self.batch_ob_shape,
                             self.model.train_model.action.dtype.name)

        # discount reward
        last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()
        mb.discount_reward(self.gamma, last_values)
        mb.flatten(self.batch_action_shape)

        self.update_all_envs()
        return mb.results() + (epinfos,)

    # def reorder_obs(self):
    #     self.obs = list(map(lambda env_id: self.obs[env_id], self.env.venv.active_envs))

    # def choose_envs_to_activate(self, _):
    #     self.counter += 1

    def update_all_envs(self):
        pass

    def update_all_envs_exploration(self, _):
        pass


class PrioRunner(Runner):
    def __init__(self, env, model, prio_args, nsteps, gamma):
        super().__init__(env, model, nsteps, gamma)
        self.prioritizer = PrioritizerFactory(prio_args)
        self.all_envs = AllEnvData(obs=self.obs, states=self.states, dones=self.dones)
        self.batch_ob_shape = (prio_args.n_active_envs * nsteps,) + env.observation_space.shape

    def run(self, **kwargs):
        self.choose_envs_to_activate(**kwargs)
        return super().run()

    def choose_envs_to_activate(self, **kwargs):
        active_idx = self.prioritizer.pick_active_envs(kwargs['prio_score'])
        self.counter[active_idx] += 1
        if active_idx:
            self.active_envs = list(active_idx)

            self.env.set_active_envs(active_idx)
            self.obs = [self.all_envs.obs[i] for i in self.active_envs]
            self.states = self.all_envs.states[self.active_envs] if self.all_envs.states else self.all_envs.states
            self.dones = [self.all_envs.dones[i] for i in self.active_envs]

    def update_all_envs(self):
        self.all_envs.update(obs=self.obs, dones=self.dones, states=self.states, envs_idx=self.active_envs)


class ExpPrioRunner(PrioRunner):
    def __init__(self, env, model, prio_args, nsteps, gamma):
        super().__init__(env, model, prio_args, nsteps, gamma)
        self.exploration_steps = prio_args.exploration_steps

    def update_all_envs_exploration(self, exploration_res):
        self.all_envs.update_exploration(exploration_res)


class AllEnvData:
    def __init__(self, obs, states, dones):
        self.obs = deepcopy(obs)
        self.states = deepcopy(states)
        self.dones = deepcopy(dones)
        # self.prio_score = np.full((len(dones)), np.inf)

    def update(self, obs, dones, states, envs_idx):
        for i, env in enumerate(envs_idx):
            self.obs[env] = deepcopy(obs[i])
            self.dones[env] = deepcopy(dones[i])
            if states:
                self.states[env] = deepcopy(states[i])

    def update_exploration(self, exploration_res):
        if exploration_res:
            (obs, rewards, dones, infos, exploration_list) = exploration_res[-1]
            self.update(obs, dones, None, exploration_list)


class MemoryBuffer:
    def __init__(self):
        self.obs = []
        self.r = []
        self.a = []
        self.v = []
        self.dones = []
        self.states = None
        self.masks = None

    def append(self, obs=None, a=None, v=None, dones=None, r=None):
        if obs is not None:
            self.obs.append(np.copy(obs))
        if a is not None:
            self.a.append(a)
        if v is not None:
            self.v.append(v)
        if dones is not None:
            self.dones.append(dones)
        if r is not None:
            self.r.append(r)

    def batch_to_rollouts(self, ob_dtype, batch_ob_shape, a_dtype):
        self.obs = np.asarray(self.obs, dtype=ob_dtype).swapaxes(1, 0).reshape(batch_ob_shape)
        self.r = np.asarray(self.r, dtype=np.float32).swapaxes(1, 0)
        self.a = np.asarray(self.a, dtype=a_dtype).swapaxes(1, 0)
        self.v = np.asarray(self.v, dtype=np.float32).swapaxes(1, 0)
        self.dones = np.asarray(self.dones, dtype=np.bool).swapaxes(1, 0)
        self.dones = self.dones[:, 1:]
        self.masks = self.dones[:, :-1]

    def discount_reward(self, gamma, last_values):
        if gamma > 0.0:
            # Discount/bootstrap off value fn
            for n, (rewards, dones, value) in enumerate(zip(self.r, self.dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards + [value], dones + [0], gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, gamma)

                self.r[n] = rewards

    def flatten(self, batch_a_shape):
        self.a = self.a.reshape(batch_a_shape)
        self.r = self.r.flatten()
        self.v = self.v.flatten()
        self.masks = self.masks.flatten()

    def results(self):
        return self.obs, self.states, self.r, self.masks, self.a, self.v


def RunnerBuilder(env, model, nsteps=5, gamma=0.99, prio_args=None):
    if not prio_args:
        return Runner(env, model, nsteps=nsteps, gamma=gamma)
    else:
        if prio_args.time_limit < np.inf:
            return ExpPrioRunner(env, model, nsteps=nsteps, gamma=gamma, prio_args=prio_args)
        else:
            return PrioRunner(env, model, nsteps=nsteps, gamma=gamma, prio_args=prio_args)