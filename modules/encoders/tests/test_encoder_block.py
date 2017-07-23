import tensorflow as tf
from attention.modules.encoders import EncoderBlock

class TestEncoderBlock(tf.test.TestCase):
    def setUp(self):
        super(TestEncoderBlock, self).setUp()

        self.num_heads = 4
        self.hidden_size = 10
        self.dropout_rate = 0.0
        self.module = EncoderBlock(num_heads=self.num_heads,
                                    hidden_size=self.hidden_size,
                                   dropout_rate=self.dropout_rate)

    def test_shape(self):
        batch_size = 4
        seq_len = 10
        embed_dim = 32
        inputs = tf.random_uniform((batch_size, seq_len, embed_dim))

        out = self.module(inputs, is_training=True)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            res = out.eval()

            self.assertShapeEqual(res, inputs)
