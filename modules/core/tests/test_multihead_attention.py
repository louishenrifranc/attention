import tensorflow as tf
from attention.modules import MultiHeadAttention
import numpy as np


class TestMultiHeadAttention(tf.test.TestCase):
    def setUp(self):
        super(TestMultiHeadAttention, self).setUp()
        self.num_heads = 2
        self.module = MultiHeadAttention(
            num_heads=self.num_heads,
            dropout_rate=0.0,
            mask_leftward_decoder=False
        )
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

    def test_create_mask_for_keys(self):
        keys = np.array([
            [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
            [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ], dtype=np.float32)
        out = self.module.create_mask_for_keys(keys)
        with self.test_session():
            res = out.eval()
            self.assertEqual(res.shape, (keys.shape[0],
                                         self.num_heads,
                                         1,
                                         keys.shape[1]
                                         ))

            self.assertAllClose(res[2, :, :, :], np.full(
                (self.num_heads, 1, keys.shape[1]),  -2 ** 30))
            self.assertAllClose(res[0, :, :, 0], res[1, :, :, 0])
            self.assertTrue(all(res[0, :, :, 0] != res[1, :, :, 1]))

    def test_create_mask_for_queries(self):
        queries = np.array([
            [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
            [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ], dtype=np.float32)

        out = self.module.create_mask_for_queries(queries)
        with self.test_session():
            res = out.eval()

            self.assertEqual(res.shape, (queries.shape[0],
                                         self.num_heads,
                                         queries.shape[1],
                                         1))

            self.assertAllEqual(res[2, :, :, :], np.zeros(
                (self.num_heads, queries.shape[1], 1)))
            self.assertAllClose(res[0, :, 0, :], res[1, :, 0, :])
            self.assertTrue(all(res[0, :, 0, :] != res[1, :, 1, :]))

    def test_create_mask_for_decoding(self):
        batch_size = 3
        seq_len_keys = 4
        seq_len_queries = 4  # queries = key when mask_for_decoding is used
        tensor = tf.random_uniform(
            (batch_size, self.num_heads, seq_len_queries, seq_len_keys))

        out = self.module.create_mask_for_decoding(tensor)
        with self.test_session():
            res = out.eval()

            self.assertEqual(res.shape, (batch_size,
                                         self.num_heads,
                                         seq_len_queries,
                                         seq_len_keys))
            for bs in range(batch_size):
                for n_h in range(self.num_heads):
                    for i in range(seq_len_queries):
                        self.assertTrue(all(res[bs, n_h, i, :i+1] == 0))
                        self.assertTrue(all(res[bs, n_h, i, i+1:] != 0))

    def test_build_queries_equal_keys(self):
        queries = np.array([
            [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
            [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
        ], dtype=np.float32)

        queries = tf.convert_to_tensor(queries)
        keys = queries
        out = self.module(queries=queries,
                          keys=keys)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            res = out.eval()

            self.assertEqual(res.shape, queries.shape)


    def test_build_queries_different_keys(self):
        queries = np.array([
            [[1, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0]],
            [[1, 1, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]
        ], dtype=np.float32)

        keys = np.array([
            [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1]],
            [[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]],
        ], dtype=np.float32)

        queries = tf.convert_to_tensor(queries)
        keys = tf.convert_to_tensor(keys)

        module = MultiHeadAttention(
            num_heads=2
        )

        out = module(queries, keys)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            res = out.eval()

            p = 1
