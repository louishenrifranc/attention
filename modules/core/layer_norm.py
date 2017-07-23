from config import Config

import tensorflow as tf
import sonnet as snt


class LayerNorm(snt.AbstractModule):
    def __init__(self):
        super(LayerNorm, self).__init__(name="layer_norm")

        self.beta = None
        self.gamma = None

    def _build(self, inputs):
        layer_shape = inputs.get_shape().as_list()[-1]
        self.beta = tf.zeros(shape=layer_shape)
        self.gamma = tf.zeros(shape=layer_shape)

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        normalized_inputs = (inputs - mean) / (variance + 1e-8) ** 0.5
        return self.beta * normalized_inputs + self.gamma
