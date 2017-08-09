import sonnet as snt
from ...modules import MultiHeadAttention, PointWiseFeedForward, LayerNorm


class EncoderBlock(snt.AbstractModule):
    def __init__(self, num_heads, hidden_size, dropout_rate):
        super(EncoderBlock, self).__init__(name="encoder_block")

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def _build(self, inputs, sequence_length):
        keys = queries = inputs
        keys_len =  queries_len = sequence_length
        output = MultiHeadAttention(num_heads=self.num_heads, dropout_rate=self.dropout_rate)(queries=queries,
                                                                                              keys=keys,
                                                                                              queries_len=queries_len,
                                                                                              keys_len=keys_len)
        output += queries
        output = LayerNorm()(output)
        pointwise_module = PointWiseFeedForward(
            hidden_size=self.hidden_size,
            output_size=output.get_shape().as_list()[-1],
            dropout_rate=self.dropout_rate)
        output = pointwise_module(output)
        output = LayerNorm()(output)
        return output
