from config import Config

import tensorflow as tf
import sonnet as snt


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
      x: a Tensor with shape [..., m]
      n: an integer.
    Returns:
      a Tensor with shape [..., n, m/n]
    """
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return ret


def split_heads(x, num_heads):
    """Split channels (dimension 3) into multiple heads (becomes dimension 1).
    Args:
      x: a Tensor with shape [batch, length, channels]
      num_heads: an integer
    Returns:
      a Tensor with shape [batch, num_heads, length, channels / num_heads]
    """

    return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0):
    """dot-product attention.
    Args:
      q: a Tensor with shape [batch, heads, length_q, depth_k]
      k: a Tensor with shape [batch, heads, length_kv, depth_k]
      v: a Tensor with shape [batch, heads, length_kv, depth_v]
      bias: bias Tensor (see attention_bias())
      dropout_rate: a floating point number
      summaries: a boolean
      image_shapes: optional quadruple of integer scalars for image summary.
        If the query positions and memory positions represent the
        pixels of a flattened image, then pass in their dimensions:
          (query_rows, query_cols, memory_rows, memory_cols).
      name: an optional string
    Returns:
      A Tensor.
    """
    logits = tf.matmul(q, k, transpose_b=True)
    if bias is not None:
        logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    # dropping out the attention links for each of the heads
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    return tf.matmul(weights, v)


def combine_heads(x):
    """Inverse of split_heads.
    Args:
      x: a Tensor with shape [batch, num_heads, length, channels / num_heads]
    Returns:
      a Tensor with shape [batch, length, channels]
    """
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    Args:
      x: a Tensor with shape [..., a, b]
    Returns:
      a Tensor with shape [..., ab]
    """
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret


class MultiHeadAttention(snt.AbstractModule):
    def __init__(self, num_heads, dropout):
        super(MultiHeadAttention, self).__init__(name="multiheadattention")

        self.num_heads = num_heads
        self.dropout = dropout

    def _build(self, queries, keys, bias, total_key_depth, total_value_depth, output_depth):
        num_heads, dropout = self.num_heads, self.dropout

        if keys is None:
            # self attention
            combined = snt.Conv1D(
                output_channels=total_key_depth * 2 + total_value_depth,
                kernel_shape=1)(queries)
            q, k, v = tf.split(
                combined, [total_key_depth, total_key_depth, total_value_depth],
                axis=2)
        else:
            q = snt.Conv1D(output_channels=total_key_depth,
                           kernel_shape=1)(queries)
            combined = snt.Conv1D(output_channels=total_key_depth + total_value_depth,
                                  kernel_shape=1)(keys)
            k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)
        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head ** -0.5
        x = dot_product_attention(
            q, k, v, bias, dropout)
        x = combine_heads(x)
        x = snt.Conv1D(output_channels=output_depth,
                       kernel_shape=1)(x)

        return x
