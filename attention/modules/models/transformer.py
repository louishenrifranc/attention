import sonnet as snt
import tensorflow as tf
from ..encoders import Encoder
from ..decoders import Decoder, GreedyDecoderHelper
from tensorflow.python.estimator.model_fn import ModeKeys
from collections import namedtuple

TransformerOuput = namedtuple("TransformerOuput", "loss logits predicted_ids")


class TransformerModule(snt.AbstractModule):
    def __init__(self, params, mode):
        super(TransformerModule, self).__init__(name="transformer")
        self.params = params
        self.mode = mode

    def _build(self, features):
        encoder_inputs, encoder_length = features[0]
        decoder_inputs, decoder_length = features[1]

        encoder = Encoder(
            params=self.params.encoder_params.params,
            block_params=self.params.encoder_params.encoder_block_params,
            embed_params=self.params.encoder_params.embed_params
        )

        encoder_output, positional_embedding = encoder(inputs=encoder_inputs, sequences_length=encoder_length)
        eos = bos = tf.expand_dims(tf.ones_like(decoder_inputs[:, 0]), -1)
        eos *= self.params.eos_token
        bos *= self.params.bos_token
        decoder_inputs = tf.concat([bos, decoder_inputs], axis=-1)
        labels = tf.concat([decoder_inputs, eos], axis=-1)

        decoder_class = GreedyDecoderHelper if self.mode == ModeKeys.PREDICT else Decoder
        compute_loss = self.mode == ModeKeys.TRAIN
        decoder = decoder_class(
            params=self.params.decoder_params.params,
            block_params=self.params.decoder_params.decoder_block_params,
            embed_params=self.params.decoder_params.embed_params
        )

        loss, logprobs = decoder(inputs=decoder_inputs, sequence_length=decoder_length, labels=labels,
                                 encoder_output=encoder_output, encoder_sequence_length=encoder_length,
                                 embedding_lookup=positional_embedding, compute_loss=compute_loss)
        return TransformerOuput(loss=loss, logits=logprobs, predicted_ids=tf.argmax(logits, axis=-1))
