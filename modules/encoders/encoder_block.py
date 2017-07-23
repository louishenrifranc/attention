import sonnet as snt
from modules import MultiHeadAttention, PointWiseFeedForward, LayerNorm


class EncoderBlock(snt.AbstractModule):
    def __init__(self, num_heads, hidden_size, dropout_rate):
        super(EncoderBlock, self).__init__(name="encoder_block")

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def _build(self, inputs, is_training):
        keys = queries = inputs
        output = MultiHeadAttention(num_heads=self.num_heads)(queries=queries,
                                                              keys=keys,
                                                              dropout=self.dropout_rate,
                                                              is_training=is_training)
        output += queries
        output = LayerNorm()(output)

        output = PointWiseFeedForward(hidden_size=self.hidden_size,
                                      output_size=output.get_shape().as_list()[-1])
        output = LayerNorm()(output)
        return output
