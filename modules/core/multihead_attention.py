import tensorflow as tf
import sonnet as snt


class MultiHeadAttention(snt.AbstractModule):
    def __init__(self, num_heads, dropout_rate=0.0, mask_leftward_decoder=False):
        super(MultiHeadAttention, self).__init__(name="multihead_attention")

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.mask_leftward_decoder = mask_leftward_decoder

    def create_mask_for_keys(self, tensor):
        mask = tf.to_float(tf.equal(tf.reduce_sum(tensor, axis=-1), 0.0))
        mask *= -2 ** 30
        mask = tf.expand_dims(tf.expand_dims(mask, 1), -1)
        mask = tf.tile(mask, [1, self.num_heads, 1, 1])
        mask = tf.transpose(mask, [0, 1, 3, 2])
        return mask

    def create_mask_for_queries(self, tensor):
        mask = tf.to_float(tf.not_equal(tf.reduce_sum(tensor, axis=-1), 0.0))
        mask = tf.expand_dims(tf.expand_dims(mask, 1), -1)
        mask = tf.tile(mask, [1, self.num_heads, 1, 1])
        return mask

    def create_mask_for_decoding(self, tensor):
        masking_leftward = 1 - tf.contrib.linalg.LinearOperatorTriL(tf.ones_like(tensor[0, 0, :, :])).to_dense()
        masking_leftward = tf.expand_dims(tf.expand_dims(masking_leftward, 0), 0)
        masking_leftward = tf.tile(masking_leftward,
                                   [tensor.get_shape().as_list()[0], self.num_heads, 1, 1])
        masking_leftward *= - 2 ** 30
        return masking_leftward

    def _build(self, queries, keys, values=None, is_training=False):
        if values is None:
            values = keys

        import pytest; pytest.set_trace()
        num_outputs = queries.get_shape().as_list()[-1]
        q_w = tf.contrib.layers.fully_connected(queries, num_outputs)  # batch_size x query_l x d_model
        k_w = tf.contrib.layers.fully_connected(keys, num_outputs)  # batch_size x keys_l x d_model
        v_w = tf.contrib.layers.fully_connected(values, num_outputs)  # batch_size x values_l x d_model

        q_wi = tf.transpose(tf.split(q_w, self.num_heads, axis=2), [1, 0, 2, 3])
        k_wi = tf.transpose(tf.split(k_w, self.num_heads, axis=2), [1, 0, 2, 3])
        v_wi = tf.transpose(tf.split(v_w, self.num_heads, axis=2), [1, 0, 2, 3])

        def dot_product(query, key):
            head_i = tf.matmul(query, tf.transpose(key, [0, 2, 1])) / key.get_shape().as_list()[-1] ** 0.5
            return head_i

        dot_prod_op = snt.BatchApply(dot_product)
        logits_q_wi_k_wi = dot_prod_op(q_wi, k_wi)

        mask_keys = self.create_mask_for_keys(keys)
        logits_q_wi_k_wi += mask_keys

        if self.mask_leftward_decoder:
            logits_q_wi_k_wi += self.create_mask_for_decoding(logits_q_wi_k_wi)

        softmax_q_wi_k_wi = tf.nn.softmax(logits_q_wi_k_wi)

        mask_queries = self.create_mask_for_queries(queries)
        softmax_q_wi_k_wi *= mask_queries
        softmax_q_wi_k_wi = tf.layers.dropout(softmax_q_wi_k_wi, self.dropout_rate, is_training)

        attention_qwi_kwi = tf.matmul(softmax_q_wi_k_wi, v_wi)
        attention_qwi_kwi = tf.transpose(attention_qwi_kwi, [0, 2, 3, 1])

        concat_attention = tf.reshape(attention_qwi_kwi, [-1, queries.get_shape().as_list()[1], num_outputs])

        multi_attention = tf.contrib.layers.fully_connected(concat_attention, num_outputs)
        return multi_attention


