from config import Config

import tensorflow as tf
import sonnet as snt


class Embedding(snt.AbstractModule):
    def __init__(self, vocab_size, config: Config):
        super(Embedding, self).__init(name="embedding")
        
        self.vocab_size = vocab_size + 1 if self.zero_pad else 0

        self.scale = getattr(config, "scale", True)
        self.emb_size = getattr(config, "emb_size", 200)
        self.zero_pad = getattr(config, "zero_pad", False)
        self.initializer = getattr(config, "initializer", tf.contrib.layers.xavier_initializer)

    def _build(self, inputs):
        """

        :param inputs:
        :return:
        """
        embedding = tf.get_variable('lookup_embedding',
                                    dtype=tf.float32,
                                    shape=[self.vocab_size, self.emb_size],
                                    initializer=self.initializer())

        if self.zero_pad:
            embedding[0, :] = 0

        outputs = tf.nn.embedding_lookup(embedding, inputs)
        if self.scale:
            outputs *= (self.emb_size ** 0.5)
        return outputs
