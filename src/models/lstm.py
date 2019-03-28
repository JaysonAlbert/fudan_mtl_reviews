import tensorflow as tf
from tensorflow.contrib.seq2seq import BahdanauAttention


FLAGS = tf.app.flags.FLAGS


def _dropout_lstm_cell(train):
  return tf.contrib.rnn.DropoutWrapper(
      tf.contrib.rnn.LSTMCell(FLAGS.hidden_size),
      input_keep_prob=1.0 - FLAGS.keep_prob * tf.to_float(train))


class LSTMLayer(tf.keras.layers.Layer):

    def __init__(self,cell_type, **kwargs):
        self.cell_type = cell_type
        self.layer_name = cell_type
        self.hidden_size = FLAGS.hidden_size
        self.num_layers = FLAGS.num_layers
        self.is_regularize = FLAGS.is_regularize
        self.alignment = None

        super(LSTMLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        if FLAGS.use_attention:
            with tf.name_scope('attention'):
                self.q = tf.get_variable('q_attention',
                                         shape=[1, self.hidden_size],
                                         dtype=tf.float32,
                                         trainable=True
                                         )

        if self.cell_type == "lstm":
            cell = tf.contrib.rnn.LSTMCell
        elif self.cell_type == "gru":
            cell = tf.contrib.rnn.GRUCell
        else:
            raise Exception("{} cell not supported, please use lstm or gru".format(self.cell_type))

        self.layers = [cell(FLAGS.hidden_size)
                       for _ in range(FLAGS.num_layers)]

        super(LSTMLayer, self).build(input_shape)


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
                self.alignment = alignment
                alignment = tf.expand_dims(alignment, 1)
                context = tf.matmul(alignment, attention.values)
                return tf.squeeze(context, axis=1)

        return tf.squeeze(tf.reduce_mean(output, axis=[1]), axis=1)
