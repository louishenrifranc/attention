import tensorflow  as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.estimator.model_fn import EstimatorSpec, ModeKeys
from attention.modules import TransformerModule

from attention.algorithms.transformer.inputs_fn import get_input_fn


class TransformerAlgorithm:
    def __init__(self, estimator_run_config, params = None):
        self.model_params = params
        self.estimator = tf.estimator.Estimator(self.get_model_fn(),
                                                params=self.model_params,
                                                config=estimator_run_config,
                                                model_dir=estimator_run_config.model_dir)
        self.experiment = None
        self.training_params = {}

    def get_model_fn(self):
        def model_fn(features, labels, mode, params=None, config=None):
            train_op = None
            loss = None
            eval_metrics = None
            predictions = None
            if mode == ModeKeys.TRAIN:
                transformer_model = TransformerModule(params=self.model_params)
                step = slim.get_or_create_global_step()
                loss = transformer_model(features)
                train_op = slim.optimize_loss(loss=loss,
                                              global_step=step,
                                              learning_rate=self.training_params["learning_rate"],
                                              clip_gradients=self.training_params["clip_gradients"],
                                              optimizer=params["optimizer"],
                                              summaries=slim.OPTIMIZER_SUMMARIES
                                              )
            elif mode == ModeKeys.PREDICT:
                raise NotImplementedError
            elif mode == ModeKeys.EVAL:
                transformer_model = TransformerModule(params=self.model_params)
                loss = transformer_model(features)

            return EstimatorSpec(train_op=train_op, loss=loss, eval_metric_ops=eval_metrics, predictions=predictions,
                                 mode=mode)

        return model_fn

    def train(self, train_params, train_context_filename, train_answer_filename, extra_hooks=None):
        self.training_params = train_params

        input_fn = get_input_fn(batch_size=train_params["batch_size"], num_epochs=train_params["num_epochs"],
                                context_filename=train_context_filename,
                                answer_filename=train_answer_filename)

        # if validation_params is not None:
        hooks = extra_hooks
        self.estimator.train(input_fn=input_fn, steps=train_params.get("steps", None),
                             max_steps=train_params.get("max_steps", None), hooks=hooks)

    def train_and_evaluate(self, train_params, train_context_filename, train_answer_filename, validation_params,
                           validation_context_filename, validation_answer_filename, extra_hooks=None):
        self.training_params = train_params

        input_fn = get_input_fn(batch_size=train_params["batch_size"],
                                num_epochs=train_params["num_epochs"],
                                context_filename=train_context_filename,
                                answer_filename=train_answer_filename)

        validation_input_fn = get_input_fn(batch_size=validation_params["batch_size"],
                                           num_epochs=validation_params["num_epochs"],
                                           context_filename=validation_context_filename,
                                           answer_filename=validation_answer_filename)

        self.experiment = tf.contrib.learn.Experiment(estimator=self.estimator,
                                                      train_input_fn=input_fn,
                                                      eval_input_fn=validation_input_fn,
                                                      train_steps=train_params.get("steps", None),
                                                      eval_steps=1,
                                                      train_monitors=extra_hooks,
                                                      train_steps_per_iteration=100)

        self.experiment.train_and_evaluate()
