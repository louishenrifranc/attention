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

        max_sequence_length = tf.shape(inputs)[1]
        one_hot_labels = tf.one_hot(labels, self.params.vocab_size, axis=-1)
        with tf.name_scope("loss"):
            mask_loss = tf.sequence_mask(sequence_length, maxlen=max_sequence_length, dtype=tf.float32)
            logits = tf.reshape(logits, [-1, self.params.vocab_size])
            one_hot_labels = tf.reshape(one_hot_labels, [-1, self.params.vocab_size])
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                           labels=one_hot_labels)
            loss = tf.reshape(loss, [-1, max_sequence_length])
            loss *= mask_loss
            loss = tf.reduce_sum(loss, 1) / tf.reduce_sum(mask_loss, 1)
            mean_loss = tf.reduce_sum(loss)

            logits = tf.reshape(logits, [-1, max_sequence_length, self.params.vocab_size])
            pred = tf.argmax(logits, axis=-1)
            acc = tf.equal(pred, labels)
            acc = tf.reduce_sum(tf.to_float(acc) * mask_loss, 1) / tf.reduce_sum(mask_loss, 1)
            acc = tf.reduce_mean(acc, name="accuracy")
        return mean_loss, tf.nn.log_softmax(logits)
