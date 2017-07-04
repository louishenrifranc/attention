import tensorflow as tf
import sonnet as snt
from dropout import Dropout
from residual import Residual


class PointWiseFeedForward(snt.AbstractModule):
    def __init__(self,
                 hidden_size,
                 output_size,
                 dropout,
                 residual_fn=Residual,
                 normalizer_fn=snt.LayerNorm):
        super(PointWiseFeedForward, self).__init__(name="pointwise_feedforward")
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        self.residual_fn = residual_fn
        self.normalizer_fn = normalizer_fn
        pass

    def _build(self, inputs):
        """

        :param inputs:
        :return:
        """
        hidden_size, output_size = self.hidden_size, self.output_size
        dropout = self.dropout

        normalizer_fn, residual_fn = self.normalizer_fn, self.normalizer_fn

        def pointwise(x):
            hidden = snt.Conv1D(output_channels=output_size, kernel_shape=(1, 1))(x)
            hidden = Dropout(dropout)(hidden)
            hidden = tf.nn.relu(hidden)

            outputs = snt.Conv1D(output_channels=output_size, kernel_shape=(1, 1))(hidden)
            outputs = Dropout(dropout)(outputs)
            return tf.nn.relu(outputs)

        if residual_fn:
            out = residual_fn(pointwise)(inputs)
        else:
            out = pointwise(inputs)

        if normalizer_fn:
            out = normalizer_fn(out)
        return out
