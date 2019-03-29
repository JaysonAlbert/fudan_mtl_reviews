import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import nest


def _maybe_mask_score(score, memory_sequence_length, score_mask_value):
    if memory_sequence_length is None:
        return score
    message = ("All values in memory_sequence_length must greater than zero.")
    with tf.control_dependencies(
            [tf.assert_positive(memory_sequence_length, message=message)]):
        score_mask = tf.sequence_mask(
            memory_sequence_length, maxlen=tf.shape(score)[1])
        score_mask_values = score_mask_value * tf.ones_like(score)
        return tf.where(score_mask, score, score_mask_values)


def _prepare_memory(memory, memory_sequence_length, check_inner_dims_defined):
    """Convert to tensor and possibly mask `memory`.

    Args:
      memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
      memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.
      check_inner_dims_defined: Python boolean.  If `True`, the `memory`
        argument's shape is checked to ensure all but the two outermost
        dimensions are fully defined.

    Returns:
      A (possibly masked), checked, new `memory`.

    Raises:
      ValueError: If `check_inner_dims_defined` is `True` and not
        `memory.shape[2:].is_fully_defined()`.
    """
    memory = nest.map_structure(
        lambda m: tf.convert_to_tensor(m, name="memory"), memory)
    if memory_sequence_length is not None:
        memory_sequence_length = tf.convert_to_tensor(
            memory_sequence_length, name="memory_sequence_length")
    if check_inner_dims_defined:
        def _check_dims(m):
            if not m.get_shape()[2:].is_fully_defined():
                raise ValueError("Expected memory %s to have fully defined inner dims, "
                                 "but saw shape: %s" % (m.name, m.get_shape()))

        nest.map_structure(_check_dims, memory)
    if memory_sequence_length is None:
        seq_len_mask = None
    else:
        seq_len_mask = tf.sequence_mask(
            memory_sequence_length,
            maxlen=tf.shape(nest.flatten(memory)[0])[1],
            dtype=nest.flatten(memory)[0].dtype)
        seq_len_batch_size = (
                tf.dimension_value(memory_sequence_length.shape[0])
                or tf.shape(memory_sequence_length)[0])

    def _maybe_mask(m, seq_len_mask):
        rank = m.get_shape().ndims
        rank = rank if rank is not None else tf.rank(m)
        extra_ones = tf.ones(rank - 2, dtype=tf.int32)
        m_batch_size = tf.dimension_value(
            m.shape[0]) or tf.shape(m)[0]
        if memory_sequence_length is not None:
            message = ("memory_sequence_length and memory tensor batch sizes do not "
                       "match.")
            with tf.control_dependencies([
                tf.assert_equal(
                    seq_len_batch_size, m_batch_size, message=message)]):
                seq_len_mask = tf.reshape(
                    seq_len_mask,
                    tf.concat((tf.shape(seq_len_mask), extra_ones), 0))
                return m * seq_len_mask
        else:
            return m

    return nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask), memory)


class Attention(tf.keras.layers.Layer):

    def __init__(self, num_layer, **kwargs):
        self.num_layer = num_layer
        self.alignments = None

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        last_dim = input_shape[-1]
        self.layers = [tf.keras.layers.Dense(last_dim, activation='relu')]
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        with tf.name_scope('attention'):
            inputs_length = kwargs.pop('inputs_length')
            memory = _prepare_memory(inputs, inputs_length, check_inner_dims_defined=True)

            for layer in self.layers:
                memory = layer(memory)

            logits = self.output_layer(inputs)
            logits = tf.squeeze(logits, axis=2)

            score_mask_value = tf.as_dtype(
                memory.dtype).as_numpy_dtype(-np.inf)
            alphas = tf.nn.softmax(_maybe_mask_score(logits, inputs_length, score_mask_value), dim=1)

            self.alignments = alphas

            alphas = tf.expand_dims(alphas, axis=-1)

            return tf.reduce_sum(inputs * alphas, 1)
