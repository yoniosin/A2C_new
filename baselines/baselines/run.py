import glob
import os
import sys
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.prio_vec_frame_stack import PrioVecFrameStack
from baselines.common.cmd_util import common_arg_parser, prio_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.prio_vec_normalize import PrioVecNormalize

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args, prio_args, dir):
    env_type, env_id = get_env_type(args.env)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args, silent_monitor=True, prio_args=prio_args)
    eval_env = build_env(args, silent_monitor=False)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.Logger.CURRENT.dir, "videos"),
                               record_video_trigger=lambda x: x % args.save_video_interval == 0,
                               video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    # Prioritization
    if prio_args is not None:
        print('-------------------------------')
        print('Prioritization parameters:')
        print('number of running envs: {} / {}'.format(prio_args.n_active_envs, args.num_env))
        print('prioritization method: {} {}'.format(prio_args.prio_type, prio_args.prio_param))
        print('exploration after {} steps'.format(prio_args.time_limit))
        print('-------------------------------')
    else:
        print('------------------------------')
        print('running without prioritization')
        print('------------------------------')

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        eval_env=eval_env,
        seed=seed,
        total_timesteps=total_timesteps,
        prio_args=prio_args,
        dir=dir,
        **alg_kwargs
    )

    return model, env


def build_env(args, silent_monitor, prio_args=None):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args.env)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale,
                               prio_args=prio_args, silent_monitor=silent_monitor)
            if prio_args is None:
                env = VecFrameStack(env, frame_stack_size)
            else:
                env = PrioVecFrameStack(env, frame_stack_size)

            # TODO prio vec frame stack

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        num_env = args.n_active_envs if prio_args is None else args.num_env
        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, num_env or 1, seed, reward_scale=args.reward_scale,
                           flatten_dict_observations=flatten_dict_observations,
                           prio_args=prio_args, silent_monitor=silent_monitor)

        if env_type == 'mujoco':
            if prio_args is None:
                env = VecNormalize(env)
            else:
                env = PrioVecNormalize(env)

    return env


def get_env_type(env_id):
    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''

    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def build_name(args):
    if not args.prio:
        return 'no_prio_' + str(args.n_active_envs)

    return '_'.join([args.prio_type if args.prio_type else 'None',
                     str(args.prio_param) if args.prio_param and args.prio_type != 'random' else 'None',
                     str(args.n_active_envs),
                     str(args.num_env)] +
                    (['no_exp'] if args.time_limit is np.inf else ['exp_freq',
                                                                   str(args.time_limit),
                                                                   'exp_steps',
                                                                   str(args.exploration_steps)]))


def check_prio_args(prio_args):
    if prio_args.prio:
        if not prio_args.prio_type:
            raise IOError("prio type not specified")
        elif prio_args.prio_type == 'random':
            prio_args.prio_param = None  # random requires no parameter
        elif not prio_args.prio_param:
            raise IOError("prio param not specified")

        if prio_args.time_limit < np.inf and prio_args.exploratio_steps <= 0:
            raise IOError("exploration run, but exploration step is {}".format(prio_args.exploration_steps))

        if prio_args.exploration_steps > 0 and prio_args.time_limit == np.inf:
            raise IOError("exploration run, but time limit is {}".format(prio_args.time_limit))


    else:
        # if prio is not required, all envs must activated
        if prio_args.num_env != prio_args.n_active_envs: raise IOError("number of active envs differ from num_envs")


def main(main_args):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    print(main_args)
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(main_args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    prio_parser = prio_arg_parser()
    prio_args, unknown_args_prio = prio_parser.parse_known_args(main_args)
    check_prio_args(prio_args)
    if not prio_args.prio:
        prio_args = None
    print(args)
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        name = build_name(args)
        dir = args.base_folder + '/' + args.env + '/' + name
        listing = glob.glob(dir + '*')
        dir += '-' + str(len(listing))
        dir = logger.configure(dir=dir, format_strs=args.output)
        with open(dir + "/args.txt", "w") as f:
            f.write('-----------args------------\n')
            for key, value in vars(args).items():
                f.write("%s:   %s\n" % (key, value))
            f.write('\n')
            if prio_args is not None:
                f.write('---------prio args---------\n')
                for key, value in vars(prio_args).items():
                    f.write("%s:   %s\n" % (key, value))
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = train(args, extra_args, prio_args, dir=dir)
    env.close()

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        env = build_env(args)
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, _, done, _ = env.step(actions)
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done

            if done:
                obs = env.reset()

        env.close()
    print('return one process: ' + dir)
    return 0
    # return model


if __name__ == '__main__':
    main(sys.argv)
