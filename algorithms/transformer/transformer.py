import tensorflow  as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.estimator.model_fn import EstimatorSpec, ModeKeys
from modules import TransformerModule


class TransformerAlgorithm:
    def __init__(self, params: dict = None):
        self.model_params = params
        self.estimator = tf.estimator.Estimator(self.get_model_fn(),
                                                config=None,
                                                params=self.model_params)
        self.training_params = {}

        self.stop_hook = None

    def get_model_fn(self):
        def model_fn(features, labels, mode, params=None, config=None):
            train_op = None
            loss = None
            eval_metrics = None
            predictions = None
            if mode == ModeKeys.TRAIN:
                transformer_model = TransformerModule(params=params)
                step = slim.get_or_create_global_step()
                features["is_training"] = True
                loss = transformer_model(features)
                train_op = slim.optimize_loss(loss=loss,
                                              global_step=step,
                                              learning_rate=self.training_params["learning_rate"],
                                              clip_gradients=self.training_params["clip_gradients"],
                                              optimizer=self.model_params["optimizer"],
                                              summaries=slim.OPTIMIZER_SUMMARIES
                                              )
            elif mode == ModeKeys.PREDICT:
                raise NotImplementedError
            elif mode == ModeKeys.EVAL:
                features["is_training"] = False
                transformer_model = TransformerModule(params=params)
                loss = transformer_model(features)
            return EstimatorSpec(train_op=train_op, loss=loss, eval_metric_ops=eval_metrics,
                                 predictions=predictions,
                                 mode=mode)

        return model_fn

    def train(self, train_params):
        self.training_params = train_params
