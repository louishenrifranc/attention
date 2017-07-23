import tensorflow as tf
import sonnet as snt


class PositionnalEmbedding(snt.Embed):
    def __init__(self, vocab_size,
                 embed_dim):
        super(PositionnalEmbedding, self).__init__(vocab_size=vocab_size,
                                                   embed_dim=embed_dim,
                                                   name="embedding")

    def _build(self, ids):
        emb_lookup = super(PositionnalEmbedding, self)(ids)
        positionnal_embedding = tf.get_variable('positional_embedding',
                                                dtype=tf.float32,
                                                shape=emb_lookup[0].get_shape(),
                                                initializer=tf.contrib.layers.xavier_initializer())
        return emb_lookup + positionnal_embedding
