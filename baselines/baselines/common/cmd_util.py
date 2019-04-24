"""
Helpers for scripts like run_atari.py.
"""

import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym
from gym.wrappers import FlattenDictWrapper
from baselines import logger
from baselines.bench.monitor import MonitorFactory
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.prio_subproc_vec_env import PrioSubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common import retro_wrappers
import numpy as np

def make_vec_env(env_id, env_type, num_env, seed, silent_monitor,
                 wrapper_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None,
                 prio_args=None):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    def make_thunk(rank):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            subrank = rank,
            seed=seed,
            reward_scale=reward_scale,
            gamestate=gamestate,
            flatten_dict_observations=flatten_dict_observations,
            wrapper_kwargs=wrapper_kwargs,
            silent_monitor=silent_monitor
        )

    set_global_seeds(seed)
    if prio_args is not None:
        return PrioSubprocVecEnv(prio_args.n_active_envs, [make_thunk(i + start_index) for i in range(num_env)], prio_args.time_limit)
    if num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(start_index)])


def make_env(env_id, env_type, subrank=0, seed=None, reward_scale=1.0, gamestate=None, flatten_dict_observations=True, wrapper_kwargs=None, silent_monitor=False):
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    wrapper_kwargs = wrapper_kwargs or {}
    if env_type == 'atari':
        env = make_atari(env_id)
    elif env_type == 'retro':
        import retro
        gamestate = gamestate or retro.State.DEFAULT
        env = retro_wrappers.make_retro(game=env_id, max_episode_steps=10000, use_restricted_actions=retro.Actions.DISCRETE, state=gamestate)
    else:
        env = gym.make(env_id)

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        keys = env.observation_space.spaces.keys()
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

    env.seed(seed + subrank if seed is not None else None)

    monitor = MonitorFactory(silent_monitor)
    env = monitor(env,
                  logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)

    if env_type == 'atari':
        env = wrap_deepmind(env, **wrapper_kwargs)
    elif env_type == 'retro':
        env = retro_wrappers.wrap_deepmind_retro(env, **wrapper_kwargs)

    if reward_scale != 1:
        env = retro_wrappers.RewardScaler(env, reward_scale)

    return env


def make_mujoco_env(env_id, seed, reward_scale=1.0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    myseed = seed  + 1000 * rank if seed is not None else None
    set_global_seeds(myseed)
    env = gym.make(env_id)
    logger_path = None if logger.get_dir() is None else os.path.join(logger.get_dir(), str(rank))
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(seed)
    if reward_scale != 1.0:
        from baselines.common.retro_wrappers import RewardScaler
        env = RewardScaler(env, reward_scale)
    return env

def make_robotics_env(env_id, seed, rank=0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        info_keywords=('is_success',))
    env.seed(seed)
    return env

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def mujoco_arg_parser():
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--base_folder', default='logs', type=str)
    parser.add_argument('--output', default=None, nargs='*')

    # TODO
    parser.add_argument('--n_active_envs', help='number of active envs within total envs (num_env)', type=int,
                        default=8)
    parser.add_argument('--prio', dest='prio', action='store_true')
    parser.add_argument('--no-prio', dest='prio', action='store_false')
    # parser.add_argument('--prio', help='prioritize agents', type=bool, default=False)
    parser.add_argument('--prio_type', help='prioritization by {greedy, gittins}', type=str, default=None),
    parser.add_argument('--prio_param', help='prioritization by {reward, TD error}', type=str, default=None)
    parser.add_argument('--time_limit', help='time from last activation for exploration', type=int, default=np.inf)
    parser.add_argument('--exploration_steps', help='number of steps after env cross the time limit from last activation', type=int, default=0)

    return parser

def prio_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--n_active_envs', help='number of active envs within total envs (num_env)', type=int, default=8)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--prio', dest='prio', action='store_true')
    parser.add_argument('--no-prio', dest='prio', action='store_false')
    # parser.add_argument('--prio', help='prioritize agents', type=bool, default=False)
    parser.add_argument('--prio_type', help='prioritization by {greedy, gittins}', type=str, default=None),
    parser.add_argument('--prio_param', help='prioritization by {reward, TD error}', type=str, default=None)
    parser.set_defaults(prio=False)
    parser.add_argument('--time_limit', help='time from last activation for exploration', type=int, default=np.inf)
    parser.add_argument('--exploration_steps', help='number of steps after env cross the time limit from last activation', type=int, default=0)

    return parser

def robotics_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser


def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dicitonary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval
