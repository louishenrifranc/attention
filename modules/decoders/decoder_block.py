import sonnet as snt
from ...modules import MultiHeadAttention, PointWiseFeedForward, LayerNorm


class DecoderBlock(snt.AbstractModule):
    def __init__(self, num_heads, hidden_size, dropout_rate):
        super(DecoderBlock, self).__init__(name="encoder_block")

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def _build(self, inputs, encoder_output, is_training):
        keys = queries = inputs
        multi_head_attention = MultiHeadAttention(
            num_heads=self.num_heads,
            mask_leftward_decoder=True,
            dropout_rate=self.dropout_rate)
        output = multi_head_attention(queries=queries,
                                      keys=keys,
                                      is_training=is_training)
        output += queries
        output = LayerNorm()(output)

        output = MultiHeadAttention(
            num_heads=self.num_heads)(
            queries=output,
            keys=encoder_output,
            is_training=is_training)
        output += queries
        output = LayerNorm()(output)

        pointwise_module = PointWiseFeedForward(
            hidden_size=self.hidden_size,
            output_size=output.get_shape().as_list()[-1],
            dropout_rate=self.dropout_rate)
        output = pointwise_module(output, is_training)
        output = LayerNorm()(output)
        return output
