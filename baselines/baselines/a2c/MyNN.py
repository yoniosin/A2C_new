from baselines.common.models import get_network_builder
from baselines.a2c.utils import fc
from baselines.common.input import observation_placeholder, encode_observation
import tensorflow as tf
from baselines.common.tf_util import get_session, adjust_shape
from functools import reduce
from itertools import zip_longest


def prio_network_builder(env, nbatch, nsteps, nenvs, network_type, **policy_kwargs):
    network = get_network_builder(network_type)(**policy_kwargs)

    def prio_net_fn(nbatch=None, nsteps=None, sess=None):
        ob_space = env.observation_space

        X = observation_placeholder(ob_space, batch_size=nbatch)
        return MyNN

    return prio_net_fn()


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


class MyNN:
    def __init__(self, env, nbatch, nsteps, nenvs, network, **policy_kwargs):
        self.nbatch = nbatch
        self.nsteps = nsteps
        self.nenvs = nenvs
        self.ob_space = env.observation_space
        self.OUT = tf.placeholder(tf.float32, [nenvs])
        self.X = observation_placeholder(self.ob_space, batch_size=nbatch)
        encoded_x = encode_observation(self.ob_space, self.X)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            self.net = get_network_builder(network)(**policy_kwargs)
            self.h1 = self.net(encoded_x)
        self.h2 = fc(self.h1, 'vf', 1)
        self.out = self.h2[:, 0]

    def evaluate(self, obs):
        eval_X = observation_placeholder(self.ob_space, batch_size=self.nenvs)
        sess = get_session()
        feed_dict = {eval_X: adjust_shape(eval_X, obs)}
        return sess.run(self.out, feed_dict)

    def evaluate_chunks(self, obs):
        return reduce(lambda a, b: a + b, map(self.evaluate, grouper(obs, self.nbatch)))[self.nsteps - 1::self.nsteps]
