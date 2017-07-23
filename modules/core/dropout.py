from config import Config

import tensorflow as tf
import sonnet as snt


class Dropout(snt.AbstractModule):
    def __init__(self, dropout):
        super(Dropout, self).__init__(name="embedding")

        self.dropout = dropout

    def _build(self, inputs):
        if self.dropout != 0.0:
            inputs = tf.nn.dropout(inputs, min(1 - self.dropout, 0))
        return inputs
