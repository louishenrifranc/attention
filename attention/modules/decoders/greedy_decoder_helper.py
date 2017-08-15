from ..decoders import Decoder

import sonnet as snt
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys


class GreedyDecoderHelper(snt.AbstractModule):
    def __init__(
            self,
            params,
            block_params,
            embed_params):
        super(GreedyDecoderHelper, self).__init__(name="greedy_decoder_helper")
        self.params = params
        self.block_params = block_params
        self.embed_params = embed_params

        self._input_shape = None

    @property
    def input_shape(self):
        return self._input_shape

    def initializer(self, inputs):
        self._output = None
        return inputs, tf.ones(self._input_shape[0])

    def sample(self, logprobs):
        predicted_ids = tf.argmax(logprobs, axis=-1)
        return predicted_ids

    def prepare_output(self, logprobs):
        if self._output is None:
            self._output = logprobs
        else:
            self._output = tf.concat([self._output, logprobs], axis=1)

    def next_inputs(self, time, inputs, sample_ids):
        inputs[:, time] = sample_ids
        return inputs, tf.ones(self._input_shape[0]) * (time + 1)

    def _build(self, *, inputs, encoder_output, encoder_sequence_length, embedding_lookup=None, **kwargs):
        self._input_shape = tf.shape(inputs)[:2]

        decoder = Decoder(params=self.params, block_params=self.block_params, embed_params=self.embed_params,
                          mode=ModeKeys.PREDICT)
        inputs, sequence_length = self.initializer(inputs)
        for time in range(1, max_len):
            _, logprobs = decoder(inputs, sequence_length, None, encoder_output, encoder_sequence_length,
                                     embedding_lookup)
            self.prepare_output(logprobs)
            sample_ids = self.sample(logprobs)
            inputs, sequence_length = self.next_inputs(time, inputs, sample_ids)
        return None, self._output
