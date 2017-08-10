import os
import tensorflow as tf
import numpy as np

from attention.utils.mock import mock_dialogue_gen
from attention.algorithms.transformer.inputs_fn import create_sample, filter_and_modify_dialogue, get_input_fn, create_textline_file


class TestInputFunction(tf.test.TestCase):
    def setUp(self):
        super(TestInputFunction, self).setUp()

    def test_create_sample(self):
        sample_gen = create_sample(dialogue_gen=mock_dialogue_gen())
        for i, sample in enumerate(sample_gen):
            self.assertIsInstance(sample, dict)
            self.assertIsInstance(sample["context"], list)
            self.assertIsInstance(sample["answer"], list)


    def test_get_input_fn(self):
        context_filename = "context.txt"
        answer_filename = "answer.txt"

        batch_size, num_epochs = 10, 3
        sample_gen = mock_dialogue_gen()
        create_textline_file(sample_gen, context_filename, answer_filename)

        input_fn = get_input_fn(batch_size, num_epochs, context_filename, answer_filename, max_sequence_len=20)
        inputs, _ = input_fn()

        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            for _ in range(num_epochs):
                res = sess.run(inputs)

                self.assertEqual(len(res[0][0].shape), 2)
                self.assertEqual(res[0][0].shape[0], batch_size)

                len_context = res[0][1]
                self.assertAllEqual(len_context, np.sum(np.where(res[0][0] > 0, 1, 0), axis=1))

                self.assertEqual(len(res[1][0].shape), 2)
                self.assertEqual(res[1][0].shape[0], batch_size)

                len_context = res[1][1]
                self.assertAllEqual(len_context, np.sum(np.where(res[1][0] > 0, 1, 0), axis=1))

        os.remove(context_filename)
        os.remove(answer_filename)

