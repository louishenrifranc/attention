import sonnet as snt
from modules import PositionnalEmbedding
from ..encoders import EncoderBlock


class Encoder(snt.AbstractModule):
    def __init__(self, num_blocks, block_params, embed_params):
        super(Encoder, self).__init__(name="encoder_block")
        self.num_blocks = num_blocks
        self.block_params = block_params
        self.embed_params = embed_params

    def _build(self, inputs):
        output = PositionnalEmbedding(embed_dim=self.embed_params["embed_dim"],
                                      vocab_size=self.embed_params["vocab_size"])(inputs)
        for _ in range(self.num_blocks):
            output = EncoderBlock(num_heads=self.block_params["num_heads"],
                                  hidden_size=self.block_params["hidden_size"])(output)
        return output
