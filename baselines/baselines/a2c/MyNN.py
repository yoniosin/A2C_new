from baselines.common.models import get_network_builder
from baselines.a2c.utils import fc
from baselines.common.input import observation_placeholder, encode_observation
import tensorflow as tf
from baselines.common.tf_util import get_session, adjust_shape


class MyNN:
    def __init__(self, env, nbatch, network, **policy_kwargs):
        ob_space = env.observation_space
        self.X = observation_placeholder(ob_space, batch_size=nbatch)
        encoded_x = encode_observation(ob_space, self.X)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            self.net = get_network_builder(network)(**policy_kwargs)
            self.h1 = self.net(encoded_x)
        self.h2 = fc(self.h1, 'vf', 1)
        self.out = self.h2[:, 0]

    def evaluate(self, vars, input):
        sess = get_session()
        feed_dict = {self.X: adjust_shape(self.X, input)}

        return sess.run(vars, feed_dict)

