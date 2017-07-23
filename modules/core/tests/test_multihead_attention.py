import tensorflow as tf
from attention.modules import MultiHeadAttention
import numpy as np


class TestMultiHeadAttention(tf.test.TestCase):
    def setUp(self):
        super(TestMultiHeadAttention, self).setUp()
        self.num_heads = 4
        self.module = MultiHeadAttention(
            num_heads=4,
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
        with self.test_session() as sess:
            res = out.eval()
            self.assertEqual(res.shape, (keys.shape[0],
                                         self.num_heads,
                                         keys.shape[1],
                                         1))

            self.assertAllClose(res[2, :, :, :], np.full(
                (self.num_heads, keys.shape[1], 1),  -2 ** 30))
            self.assertAllClose(res[0, :, 0, :], res[1, :, 0, :])
            self.assertTrue(all(res[0, :, 0, :] != res[1, :, 1, :]))


