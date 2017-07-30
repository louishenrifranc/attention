import tensorflow as tf
import os
import re
import numpy as np

from attention.algorithms import TransformerAlgorithm
from attention.default_config import train_params, model_params, estimator_params
from attention.utils.mock import mock_dialogue_gen
from attention.algorithms.transformer.inputs_fn import create_textline_file
from attention.utils.config import AttrDict


class TestHREDAlgorithm(tf.test.TestCase):
    def setUp(self):
        super(TestHREDAlgorithm, self).setUp()

        test_folder = os.makedirs("test", exist_ok=True)
        self.params = AttrDict.from_nested_dict(model_params)
        self.train_params = train_params
        self.train_params["context_filename"] = "context.txt"
        self.train_params["answer_filename"] = "answer.txt"

        create_textline_file(mock_dialogue_gen(num_samples=10000),
                             self.train_params["context_filename"],
                             self.train_params["answer_filename"])

        estimator_params["model_dir"] = test_folder
        estimator_run_config = tf.estimator.RunConfig()
        estimator_run_config.replace(**estimator_params)
        self.algorithm = TransformerAlgorithm(estimator_run_config, params=self.params)

    def test_train(self):
        with self.assertLogs() as cm:
            self.algorithm.train(self.train_params)

        logs = cm.output
        self.assertTrue(logs[1].startswith("INFO:tensorflow:Saving checkpoints for 1 into"))
        steps = re.findall("loss = \d*.\d*", ' '.join(logs))
        final_step = logs[-1][:-1]
        losses = np.asarray([float(step.split()[-1]) for step in steps] + [float(final_step.split()[-1])])
        self.assertGreater(losses[0], losses[-1])
