import sonnet as snt
from ...modules import PositionnalEmbedding
from ..encoders import EncoderBlock
import tensorflow as tf

class Encoder(snt.AbstractModule):
    def __init__(self, params, block_params, embed_params):
        super(Encoder, self).__init__(name="encoder")
        self.params = params
        self.block_params = block_params
        self.embed_params = embed_params

    def _build(self, inputs, is_training):
        output = PositionnalEmbedding(**self.embed_params)(inputs)
        output = tf.squeeze(output)
        output = tf.layers.dropout(output, self.params["dropout_rate"], is_training)

        for _ in range(self.params["num_blocks"]):
            encoder_block = EncoderBlock(**self.block_params)
            output = encoder_block(output,
                                   is_training=is_training)
        return output
