import sonnet as snt
from multihead_attention import MultiHeadAttention
from pointwise_feedforward import PointWiseFeedForward
from residual import Residual


class Layer(snt.AbstractModule):
    def __init__(self,
                 attention_key_channels,
                 attention_value_channels,
                 hidden_size,
                 num_heads,
                 dropout,
                 filter_size):
        super(Layer, self).__init__(name="layer")

        self.attention_key_channel = attention_key_channels or hidden_size
        self.attention_value_channels = attention_value_channels or hidden_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.filter_size = filter_size

    def _build(self, inputs, memory_antecedent, self_attention_bias):
        args = {"keys": None, "bias": self_attention_bias, "total_key_depth": self.attention_key_channel,
                "total_value_depth": self.attention_value_channels, "output_depth": self.hidden_size}
        outputs = Residual(MultiHeadAttention(num_heads=self.num_heads, dropout=self.dropout))(inputs, args)

        if memory_antecedent:
            args["keys"] = memory_antecedent
            outputs = Residual(MultiHeadAttention(num_heads=self.num_heads, dropout=self.dropout))(outputs, args)

        outputs = PointWiseFeedForward(hidden_size=self.filter_size, output_size=self.hidden_size,
                                       dropout=self.dropout)(outputs)
        return outputs
