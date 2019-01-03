from tensor2tensor.models.lstm import lstm, lstm_seq2seq
import tensorflow as tf


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

        super(LSTMLayer, self).__init__(**kwargs)
        self.layers = [tf.contrib.rnn.LSTMCell(FLAGS.hidden_size)
                        for _ in range(FLAGS.num_layers)]

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
        return tf.expand_dims(tf.reduce_mean(output, axis=[1,2]), -1)
