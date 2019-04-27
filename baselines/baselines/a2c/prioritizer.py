import random
from heapq import nlargest


class RandomPrioritizer:
    def __init__(self, envs_num, active_envs_num, epsilon=0):
        super().__init__(envs_num, active_envs_num)
        self.epsilon = epsilon
        self.active_envs_num = active_envs_num
        self.envs_num = envs_num
        self.all_envs = frozenset(range(self.envs_num))

    def pick_active_envs(self, _=None):
        return list(random.sample(range(self.envs_num), self.active_envs_num))


class GreedyValuePrioritizer(RandomPrioritizer):
    def pick_active_envs(self, prio_val=None):
        if prio_val is None:  # TODO - why do we need this?
            return super().pick_active_envs(prio_val)

        def epsilon_greedy():
            def random_choice(): return random.sample(self.all_envs.difference(chosen_envs), 1)[0]
            chosen_envs.add(best_envs.pop(0) if random.random() >= self.epsilon else random_choice())

        best_envs = nlargest(self.active_envs_num, range(self.envs_num), key=lambda x: prio_val[x])

        chosen_envs = set()
        for _ in range(self.active_envs_num):
            epsilon_greedy()

        return list(chosen_envs)


def prioritizer_factory(prio_args):
    if prio_args is None:
        return None

    if prio_args.prio_type == 'random':
        prioritizer_class = RandomPrioritizer
    elif prio_args.prio_type == 'greedy':
        prioritizer_class = GreedyValuePrioritizer
    else:
        raise IOError('prioritization parameter mismatch')

    return prioritizer_class(prio_args.num_env, prio_args.n_active_envs)
