import sonnet as snt
from modules import MultiHeadAttention, PointWiseFeedForward, LayerNorm


class DecoderBlock(snt.AbstractModule):
    def __init__(self, num_heads, hidden_size):
        super(DecoderBlock, self).__init__(name="encoder_block")

        self.num_heads = num_heads
        self.hidden_size = hidden_size

    def _build(self, inputs, encoder_output):
        keys = queries = inputs
        output = MultiHeadAttention(num_heads=self.num_heads, mask_leftward_decoder=True)(queries=queries,
                                                                                          keys=keys)
        output += queries
        output = LayerNorm()(output)

        output = MultiHeadAttention(num_heads=self.num_heads)(queries=output,
                                                              keys=encoder_output)
        output += queries
        output = LayerNorm()(output)

        output = PointWiseFeedForward(hidden_size=self.hidden_size,
                                      output_size=output.get_shape().as_list()[-1])
        output = LayerNorm()(output)
        return output
