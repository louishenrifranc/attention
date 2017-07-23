import sonnet as snt
import tensorflow as tf
from dropout import Dropout


class PointWiseFeedForward(snt.AbstractModule):
    def __init__(self,
                 hidden_size,
                 output_size,
                 dropout=0.0):
        super(PointWiseFeedForward, self).__init__(name="pointwise_feedforward")
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

    def _build(self, inputs):
        """

        :param inputs:
        :return:
        """
        hidden_size, output_size = self.hidden_size, self.output_size
        dropout = self.dropout

        def pointwise(x):
            hidden = snt.Conv1D(output_channels=output_size, kernel_shape=(1, 1))(x)
            hidden = Dropout(dropout)(hidden)
            hidden = tf.nn.relu(hidden)

            outputs = snt.Conv1D(output_channels=output_size, kernel_shape=(1, 1))(hidden)
            outputs = Dropout(dropout)(outputs)
            return tf.nn.relu(outputs)

        return pointwise(inputs)
