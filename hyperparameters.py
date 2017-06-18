import tensorflow as tf

# import sonnet as snt


FLAGS = tf.flags

FLAGS.DEFINE_integer('batch_size', 32, 'Size of a batch')
FLAGS.DEFINE_integer('hidden_unit', 512, 'Number of hidden units')
# FLAGS.DEFINE_string()
# FLAGS.DEFINE_string()
