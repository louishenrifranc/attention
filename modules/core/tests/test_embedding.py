import tensorflow as tf
from attention.modules import PositionnalEmbedding

class TestPositionnalEmbedding(tf.test.TestCase):
    def setUp(self):
        super(TestPositionnalEmbedding, self).setUp()
        self.vocab_size = 100
        self.embed_dim = 123
        self.module = PositionnalEmbedding(vocab_size=self.vocab_size,
                                           embed_dim=self.embed_dim)

    def test_build(self):
        batch_size = 32
        sequence_length = 200
        ids = tf.random_uniform((batch_size, sequence_length), 0, self.vocab_size, tf.int64)


        out = self.module(ids)
        variables = self.module.get_variables()
        self.assertEqual(len(variables), 2)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            ids = out.eval()
            self.assertEqual(ids.shape, (batch_size, sequence_length, self.embed_dim))

