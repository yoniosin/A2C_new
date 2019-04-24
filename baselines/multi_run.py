import multiprocessing as mp
import random
import sys

from baselines.run import main
from itertools import product

# Define an output queue
output = mp.Queue()

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# define a example function
def run_process(args, output):
    res = main(args)
    output.put(res)


def multi_run_arg_parse():
    parser = arg_parser()

    parser.add_argument('--runs_num', type=int, default=1)
    parser.add_argument('--env_list',  type=str, default='Hopper-v2', nargs='*')
    parser.add_argument('--num_envs_list',  type=int, default=8, nargs='*')
    parser.add_argument('--active_envs_list',  type=int, default=4, nargs='*')
    parser.add_argument('--prio_type_list',  type=str, default=None, nargs='*')
    parser.add_argument('--time_limit_list',  type=int, default=4, nargs='*')
    parser.add_argument('--exploration_steps_list',  type=int, default=4, nargs='*')

    return parser

if __name__ == '__main__':
    # Setup a list of processes that we want to run

    arg_parser = multi_run_arg_parse()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)

    if len(unknown_args) > 1:
        raise IOError(str(unknown_args))

    # runs_num = args.runs_num
    runs_num = 4
    const_args_list = ['--alg=a2c',
                       '--num_timesteps=5e6',
                       '--prio_param=error',
                       '--output', 'log', 'tensorboard', 'csv']


    # env_list = args.env_list
    env_list = ['Humanoid-v2']
    # env_list = ['Hopper-v2', 'HalfCheetah-v2', 'Ant-v2', 'Humanoid-v2', 'PongNoFrameskip-v4']
    # num_envs_list = args.num_envs_list
    num_envs_list = [8]
    # active_envs_list = args.active_envs_list
    active_envs_list = [3]
    # prio_type_list = args.prio_type_list
    prio_type_list = ['greedy', 'random', None]
    # time_limit_list = args.time_limit_list
    time_limit_list = [None]
    # exploration_steps_list = args.exploration_steps_list
    exploration_steps_list = [1]

    no_prio = False
    if None in prio_type_list:
        no_prio = True
        prio_type_list.remove(None)

    for env, num_env, active_env in product(env_list, num_envs_list, active_envs_list):
        processes = []
        if no_prio:
            args_list = const_args_list + ['--env=' + str(env),
                                           '--num_env=' + str(active_env),
                                           '--n_active_envs=' + str(active_env),
                                           '--no-prio']

            [processes.append(mp.Process(target=run_process, args=(args_list, output))) for _ in range(runs_num)]

        for prio_type, time_limit, exploration_steps in product(prio_type_list, time_limit_list, exploration_steps_list):
            if active_env > num_env: continue

            args_list = const_args_list + ['--env=' + str(env),
                                           '--num_env=' + str(num_env),
                                           '--n_active_envs=' + str(active_env),
                                           '--prio',
                                           '--prio_type=' + str(prio_type)]
            if time_limit is not None:
                args_list.append('--time_limit=' + str(time_limit))
                args_list.append('--exploration_steps=' + str(exploration_steps))
            [processes.append(mp.Process(target=run_process, args=(args_list, output))) for _ in range(runs_num)]

        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

    # Get process results from the output queue

    print('all done')
