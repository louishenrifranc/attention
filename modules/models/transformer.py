import sonnet as snt
from ..encoders import Encoder
from ..decoders import Decoder


class TransformerModule(snt.AbstractModule):
    def __init__(self, params):
        super(TransformerModule, self).__init(name="transformer")
        self.params = params

    def build(self, features):
        encoder_inputs = features["encoder_inputs"]
        decoder_inputs = features["decoder_inputs"]

        is_training = features["is_training"]

        encoder = Encoder(
            params=self.params.encoder_params.params,
            block_params=self.params.encoder_params.encoder_block_params,
            embed_params=self.params.encoder_params.embed_params
        )

        encoder_output = encoder(inputs=encoder_inputs, is_training=is_training)

        decoder = Decoder(
            params=self.params.encoder_params.params,
            block_params=self.params.encoder_params.encoder_block_params,
            embed_params=self.params.encoder_params.embed_params
        )

        # TODO: incorrect
        labels = decoder_inputs[:, 1:]
        loss, _ = decoder(inputs=decoder_inputs, labels=labels, encoder_output=encoder_output, is_training=is_training)
        return loss
