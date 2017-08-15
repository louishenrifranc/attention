from attention.modules import TransformerModule
from attention.algorithms.transformer.inputs_fn import get_train_input_fn, get_predict_input_fn
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.estimator.model_fn import EstimatorSpec, ModeKeys

tf.logging.set_verbosity(tf.logging.INFO)


class TransformerAlgorithm:
    def __init__(self, estimator_run_config, params=None):
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
            transformer_model = TransformerModule(params=self.model_params, mode=mode)
            if mode == ModeKeys.TRAIN:
                step = slim.get_or_create_global_step()
                loss, _, _ = transformer_model(features)
                train_op = slim.optimize_loss(loss=loss,
                                              global_step=step,
                                              learning_rate=self.training_params["learning_rate"],
                                              clip_gradients=self.training_params["clip_gradients"],
                                              optimizer=params["optimizer"],
                                              summaries=slim.OPTIMIZER_SUMMARIES
                                              )
            elif mode == ModeKeys.PREDICT:
                transformer_output = transformer_model(features)
                predictions = {key: val for key, val in transformer_output._asdict().iteritems()}
            elif mode == ModeKeys.EVAL:
                loss, _, _ = transformer_model(features)

            return EstimatorSpec(train_op=train_op, loss=loss, eval_metric_ops=eval_metrics, predictions=predictions,
                                 mode=mode)

        return model_fn

    def train(self, train_params, train_context_filename, train_answer_filename, extra_hooks=None):
        self.training_params = train_params

        input_fn = get_train_input_fn(batch_size=train_params["batch_size"], num_epochs=train_params["num_epochs"],
                                      context_filename=train_context_filename,
                                      answer_filename=train_answer_filename,
                                      max_sequence_len=train_params["max_sequence_len"])

        hooks = extra_hooks
        self.estimator.train(input_fn=input_fn, steps=train_params.get("steps", None),
                             max_steps=train_params.get("max_steps", None), hooks=hooks)

    def train_and_evaluate(self, train_params, train_context_filename, train_answer_filename, validation_params,
                           validation_context_filename, validation_answer_filename, extra_hooks=None):
        self.training_params = train_params

        input_fn = get_train_input_fn(batch_size=train_params["batch_size"],
                                      num_epochs=train_params["num_epochs"],
                                      context_filename=train_context_filename,
                                      answer_filename=train_answer_filename,
                                      max_sequence_len=train_params["max_sequence_len"])

        validation_input_fn = get_train_input_fn(batch_size=validation_params["batch_size"],
                                                 num_epochs=validation_params["num_epochs"],
                                                 context_filename=validation_context_filename,
                                                 answer_filename=validation_answer_filename,
                                                 max_sequence_len=validation_params["max_sequence_len"])

        logging_tensor_hook = tf.train.LoggingTensorHook(tensors=["transformer/decoder/loss/accuracy:0"],
                                                         every_n_iter=1)
        if extra_hooks is None:
            extra_hooks = [logging_tensor_hook]
        else:
            extra_hooks.append(logging_tensor_hook)

        self.experiment = tf.contrib.learn.Experiment(estimator=self.estimator,
                                                      train_input_fn=input_fn,
                                                      eval_input_fn=validation_input_fn,
                                                      train_steps=train_params.get("steps", None),
                                                      eval_steps=validation_params["steps"],
                                                      train_monitors=extra_hooks,
                                                      min_eval_frequency=validation_params.get("min_eval_frequency",
                                                                                               None))

        self.experiment.train_and_evaluate()

    def predict(self, inputs, predict_params):
        input_fn = get_predict_input_fn(inputs, bos_token=self.model_params.bos_token)
        return self.estimator.predict(input_fn, predict_keys=["prediction_ids"])
