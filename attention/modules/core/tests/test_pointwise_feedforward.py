import tensorflow as tf
import numpy as np
from attention.modules import PointWiseFeedForward

class TestPointWiseFeedForward(tf.test.TestCase):
    def setUp(self):
        super(TestPointWiseFeedForward, self).setUp()

        self.hidden_size =  128
        self.output_size = 256
        self.dropout_rate = 0.0
        self.module = PointWiseFeedForward(hidden_size=self.hidden_size,
                                           output_size=self.output_size,
                                           dropout_rate=self.dropout_rate)

    def test_pointwise_feed_forward(self):
        batch_size = 32
        seq_len = 50
        embed_dim = 64
        inputs = tf.random_uniform((batch_size, seq_len, embed_dim))

        out = self.module(inputs)

        self.assertEqual(out.get_shape().as_list(), [batch_size, seq_len, self.output_size])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            out_tensor = out.eval()
            self.assertShapeEqual(out_tensor, out)
