import pickle
import time
import functools
import tensorflow as tf

from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.common.policies import build_policy

from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.runner import RunnerBuilder
from baselines.ppo2.ppo2 import safemean
from collections import deque

from tensorflow import losses

from baselines.a2c.MyNN import MyNN
import matplotlib.pyplot as plt
import numpy as np


def GetValuesForPrio(prio_type, prio_param, advs, rewards):
    if prio_type == 'random':
        return None
    if prio_type == 'greedy':
        if prio_param == 'reward':
            return rewards
        if prio_param == 'error':
            return abs(advs)
        if prio_param == 'minus_error':
            return -abs(advs)
    raise NotImplementedError(prio_type + ' ' + prio_param)


class Model(object):
    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model
    """

    @staticmethod
    def get_active_envs(env):
        try:
            return env.venv.n_active_envs
        except AttributeError:
            return env.venv.venv.n_active_envs

    def __init__(self, policy, env, nsteps,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear', prio_args=None,
                 network='cnn'):
        self.prio_args = prio_args
        # self.prio_score = None
        prio = False if prio_args is None else prio_args.prio
        sess = tf_util.get_session()

        nenvs = env.num_envs if not prio else env.n_active_envs
        nbatch = nenvs * nsteps

        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        """prio model"""
        with tf.variable_scope('a2c_model_prio', reuse=tf.AUTO_REUSE):
            prio_model = MyNN(env, nbatch, nsteps, env.n_active_envs, network)

        P_R = tf.placeholder(tf.float32, [nbatch])
        PRIO = tf.placeholder(tf.float32, [nbatch])
        P_LR = tf.placeholder(tf.float32, [])

        # prio_model_loss = losses.mean_squared_error(tf.squeeze(prio_model.out), P_R) # Reward
        prio_model_loss = losses.mean_squared_error(tf.squeeze(prio_model.out), PRIO)  # TD Error

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")
        params_prio = find_trainable_variables("a2c_model_prio")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        prio_grads = tf.gradients(prio_model_loss, params_prio)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            prio_grads, prio_grad_norm = tf.clip_by_global_norm(prio_grads, max_grad_norm)
        grads = list(zip(grads, params))
        prio_grads = list(zip(prio_grads, params_prio))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        prio_trainer = tf.train.RMSPropOptimizer(learning_rate=P_LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)
        _prio_train = prio_trainer.apply_gradients(prio_grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X: obs, A: actions, ADV: advs, R: rewards, LR: cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )

            prio_loss = 0
            if prio:
                prio_values = GetValuesForPrio(self.prio_args.prio_type, self.prio_args.prio_param, advs, rewards)
                if prio_values is not None:
                    prio_td_map = {prio_model.X: obs, P_R: rewards, P_LR: cur_lr, PRIO: prio_values}

                    prio_loss, _, p_td = sess.run(
                        [prio_model_loss, _prio_train, PRIO],
                        prio_td_map
                    )
                    # mb aranged as 1D-vector = [[env_1: n1, ..., n_nstep],...,[env_n_active]]
                    # need to take last value of each env's buffer
                    # self.prio_score = prio_values[
                    #     list(filter(lambda x: x % nsteps == (nsteps - 1), range(len(prio_values))))]
            return policy_loss, value_loss, policy_entropy, prio_loss

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.prio_model = prio_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)


def learn(
        network,
        env,
        dir,
        eval_env,
        seed=None,
        nsteps=5,
        total_timesteps=int(80e6),
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        lr=7e-4,
        lrschedule='linear',
        epsilon=1e-5,
        alpha=0.99,
        gamma=0.99,
        log_interval=100,
        load_path=None,
        prio_args=None,
        **network_kwargs):
    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)


    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)

    total_timesteps:    int, total number of timesteps to train on (default: 80M)

    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    alpha:              float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''

    global eval_runner
    set_global_seeds(seed)
    prio = False if not prio_args else prio_args.prio

    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                  lrschedule=lrschedule,
                  network=network, prio_args=prio_args)
    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    runner = RunnerBuilder(env, model, nsteps=nsteps, gamma=gamma, prio_args=prio_args)
    epinfobuf = deque(maxlen=100)

    # eval_env
    eval_runner = RunnerBuilder(eval_env, model, nsteps=nsteps, gamma=gamma)
    eval_epinfobuf = deque(maxlen=100)

    # Calculate the batch_size
    active_envs = env.n_active_envs if prio_args else nenvs
    nbatch = active_envs * nsteps

    # Start total timer
    tstart = time.time()
    envs_activation = []
    updates_num = total_timesteps // nbatch + 1
    prio_score = np.full(nenvs, np.inf) if prio else None
    for update in range(1, updates_num):
        # Get mini batch of experiences
        # TODO fix evaluate to all envs
        obs, states, rewards, masks, actions, values, epinfos = runner.run(prio_score=prio_score)
        prio_score = model.prio_model.evaluate_chunks(runner.all_envs.obs) if prio else None
        epinfobuf.extend(epinfos)

        eval_obs, eval_states, eval_rewards, eval_masks, eval_actions, eval_values, eval_epinfos = eval_runner.run()
        eval_epinfobuf.extend(eval_epinfos)

        if update % log_interval == 0 or update == 1:
            envs_activation.append(np.copy(runner.counter))

        policy_loss, value_loss, policy_entropy, prio_loss = model.train(obs, states, rewards, masks, actions, values)
        # if prio and model.prio_score is not None:
        #     for i, env in enumerate(runner.active_envs):
        #         runner.all_envs.prio_score[env] = model.prio_score[i]
        nseconds = time.time() - tstart

        # Calculate the fps (frame per second)
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("prio_loss", float(prio_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.record_tabular("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.record_tabular("eval_eprewmean", safemean([epinfo['r'] for epinfo in eval_epinfobuf]))
            logger.record_tabular("eval_eplenmean", safemean([epinfo['l'] for epinfo in eval_epinfobuf]))
            logger.dump_tabular()

        if update % 100000 == 0:
            print(dir + 'update number ' + str(update))

    with open(dir + '/envs_activation.pickle', 'wb') as handle:
        pickle.dump(envs_activation, handle)
    plot_activations(envs_activation, dir, nbatch)
    plt.show()

    return model


def plot_activations(envs_activation, dir, nbatch, ylim=None):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(envs_activation)) * nbatch, np.asarray(envs_activation))
    ax.set_title(dir)
    ax.set_xlabel('total simulation steps')
    ax.set_ylabel('envs activation')
    ax.legend(list(map(str, np.arange(envs_activation[0].shape[0]))))
    if ylim is not None:
        ax.set_ylim([0, ylim])
    # plt.show()
