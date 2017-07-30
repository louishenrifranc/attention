import tensorflow as tf
from algorithms import TransformerAlgorithm
from default_config import train_params, model_params


class TestHREDAlgorithm(tf.test.TestCase):
    def setUp(self):
        super(TestHREDAlgorithm, self).setUp()

        self.params = model_params
        self.train_params = train_params
        self.algorithm = TransformerAlgorithm(params=self.params)

    def test_train(self):
        self.algorithm.train(self.train_params)
