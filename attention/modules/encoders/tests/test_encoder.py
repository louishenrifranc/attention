import tensorflow as tf
from attention.modules.encoders import Encoder
from attention.utils.config import AttrDict

class TestEncoder(tf.test.TestCase):
    def setUp(self):
        super(TestEncoder, self).setUp()

        self.params = AttrDict.from_nested_dict({
            "dropout_rate": 0.0,
            "num_blocks": 8,
        })

        self.block_params = AttrDict.from_nested_dict({
            "num_heads": 4,
            "hidden_size": 64,
            "dropout_rate": 0.0
        })

        self.embed_params = AttrDict.from_nested_dict({
            "vocab_size": 30,
            "embed_dim": 40
        })

        self.module = Encoder(params=self.params,
                              block_params=self.block_params,
                              embed_params=self.embed_params)

    def test_shape(self):
        batch_size = 4
        seq_len_encoder = 13

        inputs = tf.random_uniform(
            (batch_size, seq_len_encoder, 1),
            0, self.embed_params["vocab_size"],
            dtype=tf.int64)
        sequences_length = tf.convert_to_tensor([1, seq_len_encoder, 5, 10])
        out, _ = self.module(inputs=inputs, sequences_length=sequences_length)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            res = sess.run(out)

            self.assertEqual(res.shape, (batch_size,
                                         seq_len_encoder,
                                         self.embed_params["embed_dim"]))
