from attention.modules.decoders import DecoderBlock
import tensorflow as tf


class TestDecoderBlock(tf.test.TestCase):
    def setUp(self):
        super(TestDecoderBlock, self).setUp()

        self.num_heads = 4
        self.hidden_size = 10
        self.dropout_rate = 0.0
        self.module = DecoderBlock(num_heads=self.num_heads,
                                   hidden_size=self.hidden_size,
                                   dropout_rate=self.dropout_rate)

    def test_shape(self):
        batch_size = 4
        seq_decoder_len = 10
        seq_encoder_len = 12
        embed_dim = 32
        inputs = tf.random_uniform((batch_size, seq_decoder_len, embed_dim))
        encoder_outputs = tf.random_uniform(
            (batch_size, seq_encoder_len, embed_dim))
        out = self.module(inputs=inputs,
                          encoder_output=encoder_outputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            res = out.eval()

            self.assertShapeEqual(res, inputs)
