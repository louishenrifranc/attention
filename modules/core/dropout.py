import tensorflow as tf
import sonnet as snt


class Dropout(snt.AbstractModule):
    def __init__(self, dropout):
        super(Dropout, self).__init__(name="embedding")

        self.dropout = dropout

    def _build(self, inputs, is_training):
        if self.dropout != 0.0:
            inputs = tf.layers.dropout(inputs,
                                       rate=self.dropout,
                                       training=is_training)
        return inputs
