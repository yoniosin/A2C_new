import random
from heapq import nlargest


class Prioritizer:
    def __init__(self, _, __):
        pass

    def pick_active_envs(self, _=None):
        pass


class RandomPrioritizer(Prioritizer):
    def __init__(self, envs_num, active_envs_num, epsilon=0.15):
        super().__init__(envs_num, active_envs_num)
        self.epsilon = epsilon
        self.active_envs_num = active_envs_num
        self.envs_num = envs_num
        self.all_envs = frozenset(range(self.envs_num))

    def pick_active_envs(self, _=None):
        # if self.active_envs_num == self.envs_num:
        #     return list(range(self.active_envs_num))
        return list(random.sample(list(range(self.envs_num)), self.active_envs_num))


class GreedyValuePrioritizer(RandomPrioritizer):
    def pick_active_envs(self, prio_val=None):
        if prio_val is None:
            return super().pick_active_envs(prio_val)

        best_envs = nlargest(self.active_envs_num, list(range(self.envs_num)), key=lambda x: prio_val[x])
        chosen_envs = set()

        def epsilon_greedy():
            if random.random() > self.epsilon:
                return best_envs.pop(0)
            return random.sample(self.all_envs.difference(chosen_envs), 1)[0]

        # _envs_to_add = [epsilon_greedy() for _ in range(self.active_envs_num)]
        # chosen_envs.add([env for env in _envs_to_add])

        return best_envs

def PrioritizerFactory(prio_args):
    PrioritizerClass = Prioritizer

    if prio_args.prio_type == 'random':
        PrioritizerClass = RandomPrioritizer
    elif prio_args.prio_type == 'greedy':
        PrioritizerClass = GreedyValuePrioritizer

    return PrioritizerClass(prio_args.num_env, prio_args.n_active_envs)
