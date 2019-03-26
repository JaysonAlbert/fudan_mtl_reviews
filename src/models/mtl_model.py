from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder

from models.base_model import *
from models.lstm import LSTMLayer

FLAGS = tf.app.flags.FLAGS
from inputs import fudan
from inputs.util import get_vocab_file

TASK_NUM=14


def _get_model():
    if FLAGS.model == "cnn":
      return ConvLayer('conv', FILTER_SIZES)
    elif FLAGS.model in ["lstm", "gru"]:
      return LSTMLayer(FLAGS.model)
    else:
      raise "model type '{}'not support, only cnn and lstm are supported".format(FLAGS.model)


def length_from_sentence(sentence):
    return tf.math.count_nonzero(sentence, axis=1)


def _summary_mean(tensor, pos, name):
  mean = tf.stack([t[pos] for t in tensor], axis=0)
  mean = tf.reduce_mean(mean, axis=0)
  return tf.summary.scalar(name, mean)


class MTLModel(BaseModel):

  def __init__(self, word_embed, all_data, adv, is_train):
    # input data
    # self.all_data = all_data
    self.is_train = is_train
    self.adv = adv

    # embedding initialization
    if word_embed is not None:
      self.word_dim = word_embed.shape[1]
      self.vocab_size = word_embed.shape[0]
      w_trainable = True if self.word_dim==50 else False
      shape = None
    else:
      encoder = SubwordTextEncoder(get_vocab_file())
      self.word_dim = FLAGS.hidden_size
      self.vocab_size = encoder.vocab_size
      word_embed = tf.random_normal_initializer(0.0, self.word_dim**-0.5)
      w_trainable = True
      shape = [self.vocab_size, self.word_dim]
    
    self.word_embed = tf.get_variable('word_embed', 
                                      initializer=word_embed,
                                      shape=shape,
                                      dtype=tf.float32,
                                      trainable=w_trainable)

    self.shared_conv = _get_model()
    self.shared_linear = LinearLayer('linear_shared', TASK_NUM, True)

    self.tensors = []
    self.pred = {}
    self.separate_acc = {}
    self.metric_tensors = []
    self.data = {}
    self.alignments = {}

    for task_name, data in all_data:
      with tf.variable_scope(task_name):
        self.build_task_graph(data, task_name)

  def adversarial_loss(self, feature, task_label):
    '''make the task classifier cannot reliably predict the task based on 
    the shared feature
    '''
    # input = tf.stop_gradient(input)
    feature = flip_gradient(feature)
    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to TASK_NUM classes
    logits, loss_l2 = self.shared_linear(feature)

    label = tf.one_hot(task_label, TASK_NUM)
    loss_adv = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

    return loss_adv, loss_l2
  
  def diff_loss(self, shared_feat, task_feat):
    '''Orthogonality Constraints from https://github.com/tensorflow/models,
    in directory research/domain_adaptation
    '''
    task_feat -= tf.reduce_mean(task_feat, 0)
    shared_feat -= tf.reduce_mean(shared_feat, 0)

    task_feat = tf.nn.l2_normalize(task_feat, 1)
    shared_feat = tf.nn.l2_normalize(shared_feat, 1)

    correlation_matrix = tf.matmul(
        task_feat, shared_feat, transpose_a=True)

    cost = tf.reduce_mean(tf.square(correlation_matrix))
    cost = tf.where(cost > 0, cost, 0, name='value')

    assert_op = tf.Assert(tf.is_finite(cost), [cost])
    with tf.control_dependencies([assert_op]):
      loss_diff = tf.identity(cost)

    return loss_diff

  def attention_diff_loss(self, shared_feat, task_feat):
      correlation_matrix = tf.matmul(
          task_feat, shared_feat, transpose_a=True)

      cost = tf.reduce_mean(tf.square(correlation_matrix))
      cost = tf.where(cost > 0, cost, 0, name='value')

      assert_op = tf.Assert(tf.is_finite(cost), [cost])
      with tf.control_dependencies([assert_op]):
          loss_diff = tf.identity(cost)

      return loss_diff

  def build_task_graph(self, data, task_name):
    task_label, labels, sentence = data

    inputs_length = length_from_sentence(sentence)
    sentence = tf.nn.embedding_lookup(self.word_embed, sentence)

    if self.is_train:
      sentence = tf.nn.dropout(sentence, 1 - FLAGS.symbol_dropout)
    
    conv_layer = _get_model()
    conv_out = conv_layer(sentence, inputs_length=inputs_length)

    shared_out = self.shared_conv(sentence, inputs_length=inputs_length)

    if self.adv:
      feature = tf.concat([conv_out, shared_out], axis=1)
    else:
      feature = conv_out

    if self.is_train:
      feature = tf.nn.dropout(feature, FLAGS.keep_prob)

    # Map the features to 2 classes
    linear = LinearLayer('linear', 2, True)
    logits, loss_l2 = linear(feature)
    
    xentropy = tf.nn.softmax_cross_entropy_with_logits(
                          labels=tf.one_hot(labels, 2), 
                          logits=logits)
    loss_ce = tf.reduce_mean(xentropy)

    loss_adv, loss_adv_l2 = self.adversarial_loss(shared_out, task_label)

    if FLAGS.model in ["lstm", "gru"] and FLAGS.attention_diff:
      loss_diff = self.attention_diff_loss(self.shared_conv.alignment, conv_layer.alignment)
    else:
      loss_diff = self.diff_loss(shared_out, conv_out)

    loss_adv = FLAGS.adv_weight * loss_adv
    loss_diff = FLAGS.diff_weight * loss_diff
    loss_l2 = FLAGS.l2_coef*(loss_l2+loss_adv_l2)

    def separate_accuracy(linear_layer, private_out, shared_out):
        w1, w2 = tf.split(linear_layer.weights[0], 2)
        # b1, b2 = tf.split(linear_layer.weights[1], 2)
        logits1 = tf.nn.xw_plus_b(private_out, w1, linear_layer.weights[1])
        logits2 = tf.nn.xw_plus_b(shared_out, w2, linear_layer.weights[1])

        def calcute_acc(logits, labels):
            pred = tf.argmax(logits, axis=1)
            acc = tf.cast(tf.equal(pred, labels), tf.float32)
            return tf.reduce_mean(acc)

        return calcute_acc(logits1, labels), calcute_acc(logits2, labels), logits1, logits2


    if self.adv:
      loss = loss_ce + loss_adv + loss_l2 + loss_diff
    else:
      loss = loss_ce  + FLAGS.l2_coef*loss_l2
    
    pred = tf.argmax(logits, axis=1)
    acc = tf.cast(tf.equal(pred, labels), tf.float32)
    acc = tf.reduce_mean(acc)

    # fetches
    self.data[task_name] = data
    self.pred[task_name] = pred
    self.separate_acc[task_name] = separate_accuracy(linear, conv_out, shared_out)
    if FLAGS.model in ["lstm", "gru"]:
        self.alignments[task_name] = (conv_layer.alignment, self.shared_conv.alignment)
    self.metric_tensors.append((
      loss_ce, loss_adv, loss_diff, loss_l2, acc ,loss
    ))
    self.tensors.append((acc, loss))

  def merged_summary(self, name_scope):
    summarys = []
    metric_names = ['loss-ce', 'loss-adv', 'loss-diff', 'loss-l2', 'acc', 'loss']
    for i, data in enumerate(self.metric_tensors):
      with tf.name_scope(fudan.get_task_name(i)):
        with tf.name_scope(name_scope):
          summarys.extend([tf.summary.scalar(metric_names[index], tensor) for index, tensor in enumerate(data)])

    with tf.name_scope('mean'):
      with tf.name_scope(name_scope):
        for i, name in enumerate(metric_names):
          summarys.append(_summary_mean(self.metric_tensors, i, name))

    return tf.summary.merge(summarys)
    
  def build_train_op(self):
    if self.is_train:
      self.train_ops = []
      for _, loss in self.tensors:
        train_op = optimize(loss)
        self.train_ops.append(train_op)

def build_train_valid_model(model_name, word_embed, all_train, all_test, adv, test):
  with tf.name_scope("Train"):
    with tf.variable_scope(model_name, reuse=None):
      m_train = MTLModel(word_embed, all_train, adv, is_train=True)
      m_train.build_train_op()
      m_train.set_saver(model_name)
      # if not test:
      #   m_train.build_train_op()
  with tf.name_scope('Valid'):
    with tf.variable_scope(model_name, reuse=True):
      m_valid = MTLModel(word_embed, all_test, adv, is_train=False)
      m_valid.build_train_op()
      m_valid.set_saver(model_name)
  
  return m_train, m_valid