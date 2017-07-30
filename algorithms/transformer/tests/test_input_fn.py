import os
import tensorflow as tf
import numpy as np

from utils.mock import mock_dialogue_gen
from algorithms.transformer.inputs_fn import make_example, create_sample, create_tf_records, filter_and_modify_dialogue, \
    get_input_fn, create_textline_file


class TestInputFunction(tf.test.TestCase):
    def setUp(self):
        super(TestInputFunction, self).setUp()

    def test_make_example(self):
        context = [0, 1, 2, 4, 5]
        answer = [1, 1, 4, 5, 2, 2]
        example = make_example(context, answer)

        self.assertIsNotNone(example)

    def test_create_sample(self):
        sample_gen = create_sample(dialogue_gen=mock_dialogue_gen())
        for i, sample in enumerate(sample_gen):
            self.assertIsInstance(sample, dict)
            self.assertIsInstance(sample["context"], list)
            self.assertIsInstance(sample["answer"], list)

    def test_create_tf_records(self):
        filename = "file.tfrecords"
        sample_gen = mock_dialogue_gen()
        create_tf_records(sample_gen, filename)

        self.assertTrue(os.path.isfile(filename))
        os.remove(filename)

    def test_get_input_fn(self):
        context_filename = "context.txt"
        answer_filename = "answer.txt"

        batch_size, num_epochs = 10, 3
        sample_gen = mock_dialogue_gen()
        create_textline_file(sample_gen, context_filename, answer_filename)

        input_fn = get_input_fn(batch_size, num_epochs, context_filename, answer_filename)
        inputs = input_fn()

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
