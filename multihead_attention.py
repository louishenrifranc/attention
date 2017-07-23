from config import Config
import tensorflow as tf
import sonnet as snt


class MultiHeadAttention(snt.AbstractModule):
    def __init__(self, num_heads, dropout):
        super(MultiHeadAttention, self).__init__(name="multihead_attention")

        self.num_heads = num_heads
        self.dropout = dropout

    def create_mask_tensor(self, tensor):
        return tf.equal(tf.reduce_sum(tensor, axis=-1), 0.0)

    def _build(self, queries, keys, values=None):
        if values is None:
            values = keys

        num_outputs = keys.get_shape().as_list()[-1]
        queries = snt.BatchLinear(num_outputs, num_dims=3)(queries)  # batch_size x query_l x d_model
        keys = snt.BatchLinear(num_outputs, num_dims=3)(keys)  # batch_size x keys_l x d_model
        values = snt.BatchLinear(num_outputs, num_dims=3)(values)  # batch_size x values_l x d_model

        queries = tf.transpose(tf.split(queries, num=self.num_heads, axis=2), [0, 1, 2])
        keys = tf.transpose(tf.split(keys, num=self.num_heads, axis=2), [0, 1, 2])
        values = tf.split(values, num=self.num_heads, axis=2)

        def dot_product(query, key):
            head_i = tf.matmul(query, tf.transpose(key, [0, 2, 1])) / tf.sqrt(keys.get_shape().as_list()[-1])
            return head_i

        dot_prod_op = snt.BatchApply(dot_product)
        logits = dot_prod_op(queries, keys)

        mask_keys = tf.cast(self.create_mask_tensor(keys), tf.float32)
        logits =
