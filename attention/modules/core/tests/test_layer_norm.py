import numpy as np
import tensorflow as tf
from attention.modules import LayerNorm

class TestLayerNorm(tf.test.TestCase):
    def setUp(self):
        super(TestLayerNorm, self).setUp()
        self.vocab_size = 100
        self.embed_dim = 123
        self.module = LayerNorm()

    def test_shape_tensor(self):
        inputs = tf.random_uniform((32, 3, 10), 0, 1)

        outputs = self.module(inputs)
        self.assertEqual(outputs.get_shape().as_list(), [32, 3, 10])

    def test_layer_norm(self):
        inputs = np.array([
            [[1, 1, 1], [2, 1, 1]],
            [[2, 2, 2], [3, 2, 1]]
        ], dtype=np.float32)

        input_tensor = tf.convert_to_tensor(inputs)
        self.module = LayerNorm()

        output = self.module(input_tensor)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = output.eval()
            self.assertTrue(all(out[0,0,:] ==  0))
            self.assertTrue(all(out[1, 0, :] == 0))
            self.assertTrue(out[1, 1, 1] ==  0)
            val_0_1_1 = (inputs[0, 1, 1] - np.mean(inputs[0, 1, :])) / np.std(inputs[0, 1, :])
            self.assertEqual(out[0, 1, 1], val_0_1_1)
