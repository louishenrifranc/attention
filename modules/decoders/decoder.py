from ..decoders import DecoderBlock
from ...modules import PositionnalEmbedding
import sonnet as snt
import tensorflow as tf


class Decoder(snt.AbstractModule):
    def __init__(
            self,
            params,
            num_blocks,
            vocab_size,
            block_params,
            embed_params):
        super(Decoder, self).__init__(name="decoder")
        self.params = params
        self.num_blocks = num_blocks
        self.block_params = block_params
        self.embed_params = embed_params
        self.vocab_size = vocab_size

    def _build(self, inputs, labels, encoder_output, is_training):
        # TODO: reuse encoder embeddings
        output = PositionnalEmbedding(**self.embed_params)(inputs)
        output = tf.layers.dropout(
            output, self.params["dropout_rate"],
            training=is_training)

        for _ in range(self.params["num_blocks"]):
            output = DecoderBlock(**self.block_params)(output,
                                                       encoder_output, is_training)

        logits = tf.contrib.layers.fully_connected(
            output, self.params["vocab_size"])

        with tf.name_scope("loss"):
            mask_loss = tf.to_float(tf.not_equal(labels, 0))

            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                           labels=labels)
            loss *= mask_loss
            loss /= tf.reduce_sum(mask_loss, axis=1)

        return loss, tf.nn.log_softmax(logits)
