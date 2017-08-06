from ..decoders import DecoderBlock
from ...modules import PositionnalEmbedding
import sonnet as snt
import tensorflow as tf


class Decoder(snt.AbstractModule):
    def __init__(
            self,
            params,
            block_params,
            embed_params):
        super(Decoder, self).__init__(name="decoder")
        self.params = params
        self.block_params = block_params
        self.embed_params = embed_params

    def _build(self, inputs, sequence_length, labels, encoder_output, encoder_sequence_length, embedding_lookup=None):
        if embedding_lookup is None:
            output = PositionnalEmbedding(**self.embed_params)(inputs)
        else:
            output = embedding_lookup(inputs)
        output = tf.layers.dropout(
            output, self.params.dropout_rate)

        for _ in range(self.params.num_blocks):
            output = DecoderBlock(**self.block_params)(output, sequence_length,
                                                       encoder_output, encoder_sequence_length)

        logits = tf.contrib.layers.fully_connected(
            output, self.params.vocab_size)

        labels = tf.one_hot(labels, self.params.vocab_size, axis=-1)
        with tf.name_scope("loss"):
            mask_loss = tf.to_float(tf.not_equal(tf.reduce_sum(labels, -1), 0))

            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                           labels=labels)

            loss *= mask_loss

            loss = tf.reduce_sum(loss, 1) / tf.reduce_sum(mask_loss, 1)
            mean_loss = tf.reduce_sum(loss)

        return mean_loss, tf.nn.log_softmax(logits)
