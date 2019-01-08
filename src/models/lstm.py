from tensor2tensor.models.lstm import lstm, lstm_seq2seq
import tensorflow as tf
from tensorflow.contrib.seq2seq import BahdanauAttention


FLAGS = tf.app.flags.FLAGS


def _dropout_lstm_cell(train):
  return tf.contrib.rnn.DropoutWrapper(
      tf.contrib.rnn.LSTMCell(FLAGS.hidden_size),
      input_keep_prob=1.0 - FLAGS.keep_prob * tf.to_float(train))


class LSTMLayer(tf.layers.Layer):

    def __init__(self,layer_name, **kwargs):
        self.layer_name = layer_name
        self.hidden_size = FLAGS.hidden_size
        self.num_layers = FLAGS.num_layers
        self.is_regularize = FLAGS.is_regularize

        if FLAGS.use_attention:
            with tf.name_scope('attention'):
                self.q = tf.get_variable('q_attention',
                                         shape=[FLAGS.batch_size, FLAGS.hidden_size],
                                         dtype=tf.float32,
                                         trainable=True
                                         )

        self.layers = [tf.contrib.rnn.LSTMCell(FLAGS.hidden_size)
                       for _ in range(FLAGS.num_layers)]

        super(LSTMLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        inputs_length = kwargs.pop('inputs_length')
        output, _ = tf.nn.dynamic_rnn(
            tf.contrib.rnn.MultiRNNCell(self.layers),
            inputs,
            inputs_length,
            initial_state = None,
            dtype=tf.float32,
            time_major=False
        )

        if FLAGS.use_attention:
            with tf.name_scope('attention'):
                attention = BahdanauAttention(FLAGS.hidden_size, output, inputs_length)
                alignment, _ = attention(self.q, output)
                alignment = tf.expand_dims(alignment, 1)
                context =  tf.matmul(alignment, attention.values)
                return tf.expand_dims(tf.reduce_mean(context, axis=[1,2]), -1)

        return tf.expand_dims(tf.reduce_mean(output, axis=[1,2]), -1)
