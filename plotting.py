import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from baselines.a2c.a2c import plot_activations
import re
from baselines.common import plot_util as pu

import mujoco_py.builder
# /home/deep3/gittins/lib/python3.6/site-packages/mujoco_py/builder.py
def plot_monitors():
    # If you want to average results for multiple seeds, LOG_DIRS must contain subfolders in the
    # following format: <name_exp0>-0, <name_exp0>-1, <name_exp1>-0, <name_exp1>-1.
    # Where names correspond to experiments you want to compare separated with random seeds by dash.

    LOG_DIRS = '/home/deep3/logs/Humanoid-v2/'
    # LOG_DIRS = '/home/deep3/logs/Hopper-v2/'
    # Uncomment below to see the effect of the timit limits flag
    # LOG_DIRS = 'time_limit_logs/reacher'

    results = pu.load_results(LOG_DIRS, running_agents=3)

    fig, ax = pu.plot_results(results, average_group=True, split_fn=lambda _: '', shaded_std=False)
    # ax[0,0].set_ylim([300,750])

    # plt.show()


def plot_activations_list(dir, pattern):
    path = Path(dir)
    activations_path = list(path.rglob('*' + pattern + '*/' + '*envs_activation.pickle'))
    for path in activations_path:
        active_envs = int(re.search("\d", path.parts[-2])[0])
        nsteps = 5 ## TODO !!!! nsteps
        nbatch = active_envs * nsteps
        with open(str(path), 'rb') as f:
            activations = pickle.load(f)
        name = path.parts[-3] + path.parts[-2]
        if not activations: continue
        plot_activations(activations, name, nbatch, ylim=650000)


if __name__ == '__main__':
    plot_monitors()
    # dir = '/home/deep3/PycharmProjects/A2C_new/logs/Hopper-v2'
    # plot_activations_list('/home/deep3/logs/Hopper-v2', '')
    # plot_activations_list('/home/deep3/PycharmProjects/A2C_new/baselines/logs/Humanoid-v2/', '')
    plt.show()

    print('done')