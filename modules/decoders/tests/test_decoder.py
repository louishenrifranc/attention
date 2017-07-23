import tensorflow as tf
from attention.modules.decoders import Decoder


class TestDecoder(tf.test.TestCase):
    def setUp(self):
        super(TestDecoder, self).setUp()

        self.params = {
            "dropout_rate": 0.0,
            "num_blocks": 8,
            "vocab_size": 30
        }

        self.block_params = {
            "num_heads": 4,
            "hidden_size": 64,
            "dropout_rate": 0.0
        }

        self.embed_params = {
            "vocab_size": 30,
            "embed_dim": 40
        }

        self.module = Decoder(params=self.params,
                              block_params=self.block_params,
                              embed_params=self.embed_params)

    def test_shape(self):
        batch_size = 4
        seq_len_decoder = 10
        seq_len_encoder = 13
        embed_dim = self.embed_params["embed_dim"]

        inputs = tf.random_uniform(
            (batch_size, seq_len_decoder, 1),
            0, self.embed_params["vocab_size"],
            dtype=tf.int64)
        labels = tf.random_uniform(
            (batch_size, seq_len_decoder),
            0, self.params["vocab_size"],
            dtype=tf.int64)
        labels = tf.one_hot(labels, depth=self.params["vocab_size"], axis=-1)

        encoder_output = tf.random_uniform(
    (batch_size, seq_len_encoder, embed_dim))

        out = self.module(inputs=inputs,
                    labels=labels,
                    encoder_output=encoder_output,
                    is_training=True)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            loss, softmax = sess.run(out)

            self.assertEqual(loss.shape, ())
            self.assertEqual(softmax.shape, (batch_size, seq_len_decoder, self.params["vocab_size"]))
