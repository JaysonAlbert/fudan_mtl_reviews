import os
import time
import sys
import tensorflow as tf
import numpy as np

from inputs import util
from inputs import fudan
from models import mtl_model
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tensor2tensor.data_generators.generator_utils import to_example
# tf.set_random_seed(0)
# np.random.seed(0)


PAD = "<pad>"
RESERVED_TOKENS = [PAD]

FLAGS = tf.app.flags.FLAGS

def build_data():
  '''load raw data, build vocab, build TFRecord data, trim embeddings
  '''
  def _build_vocab(all_data):
    print('build vocab')
    data = []
    for task_data in all_data:
      train_data, test_data = task_data
      data.extend(train_data + test_data)
    vocab = fudan.build_vocab(data)
    util.write_vocab(vocab)

    util.stat_length(data)

  def _build_subtoken_vocab(all_data):
    print('build subtoken vocab')

    def data_generator():
      for task_data in all_data:
        train_data, test_data = task_data
        for d in train_data + test_data:
          yield ' '.join(d.sentence)

    def summary(vocab):
      lens = [len(vocab.encode(sentence)) for sentence in data_generator()]
      length = sorted(lens)
      length = np.asarray(length)

      max_len = np.max(length)
      avg_len = np.mean(length)
      med_len = np.median(length)
      print('max_len: {}, avg_len: {}, med_len: {}'.format(max_len, avg_len, med_len))

    encoder = SubwordTextEncoder()
    vocab = encoder.build_from_generator(data_generator(), 2**13, 200,reserved_tokens=RESERVED_TOKENS)

    vocab_file = FLAGS.vocab_file
    base = os.path.dirname(vocab_file)
    tf.gfile.MakeDirs(base)
    vocab.store_to_file(vocab_file)

    summary(vocab)
    return vocab

  def _build_subword_data(all_data, vocab):
    print('build subword data')

    def write(data, writer):
      for d in data:
        d = d._asdict()
        d['sentence'] = vocab.encode(' '.join(d['sentence']))
        util._pad_or_truncate(d, fudan.MAX_LEN, 0)
        example = fudan._build_sequence_example(d)
        writer.write(example.SerializeToString())

    for task_id, task_data in enumerate(all_data):
      train_data, test_data = task_data
      train_record_file = os.path.join(fudan.OUT_DIR, fudan.DATASETS[task_id] + '.train.tfrecord')
      test_record_file = os.path.join(fudan.OUT_DIR, fudan.DATASETS[task_id] + '.test.tfrecord')
      train_writer = tf.python_io.TFRecordWriter(train_record_file)
      test_writer = tf.python_io.TFRecordWriter(test_record_file)

      write(train_data, train_writer)
      write(test_data, test_writer)
    
  def _build_data(all_data):
    print('build data')
    vocab2id = util.load_vocab2id()

    for task_id, task_data in enumerate(all_data):
      train_data, test_data = task_data
      fudan.write_as_tfrecord(train_data, test_data, task_id, vocab2id)

  def _trim_embed():
    print('trimming pretrained embeddings')
    util.trim_embeddings(FLAGS.word_dim)

  print('load raw data')
  all_data = []
  for task_data in fudan.load_raw_data():
    all_data.append(task_data)

  if FLAGS.subword:
    vacob = _build_subtoken_vocab(all_data)
    _build_subword_data(all_data, vacob)
  else:
    _build_vocab(all_data)

    _build_data(all_data)
    _trim_embed()


  
def train(sess, m_train, m_valid):
  best_acc, best_step= 0., 0
  start_time = time.time()
  orig_begin_time = start_time

  summary_prefix = os.path.join(FLAGS.logdir, model_name())

  train_writer = tf.summary.FileWriter(summary_prefix + '/train', sess.graph)
  valid_writer = tf.summary.FileWriter(summary_prefix + '/valid', sess.graph)

  n_task = len(m_train.tensors)

  batches = int(16 * 82/FLAGS.batch_size)

  merged_train = m_train.merged_summary('train')
  merged_valid = m_valid.merged_summary('valid')

  global_step = tf.train.get_or_create_global_step()

  num_step = 0
  for epoch in range(FLAGS.num_epochs):
    all_loss, all_acc = 0., 0.
    for batch in range(batches):
      train_fetch = [m_train.tensors, m_train.train_ops, merged_train, global_step]

      res, _, summary, gs = sess.run(train_fetch)    # res = [[acc], [loss]]
      res = np.array(res)

      train_writer.add_summary(summary, gs)

      all_loss += sum(res[:, 1].astype(np.float))
      all_acc += sum(res[:, 0].astype(np.float))
      num_step = num_step + 1

    all_loss /= (batches*n_task)
    all_acc /= (batches*n_task)

    # epoch duration
    now = time.time()
    duration = now - start_time
    start_time = now

    # valid accuracy
    valid_acc = 0.

    res, summary, gs = sess.run([m_valid.tensors, merged_valid, global_step])
    for  acc, _ in res:
      valid_acc += acc
    valid_writer.add_summary(summary, gs)

    valid_acc /= n_task

    # if best_acc < valid_acc:
    #   best_acc = valid_acc
    #   best_step = epoch

    m_train.save(sess, global_step)
      
    print("Epoch %d loss %.2f acc %.2f %.4f time %.2f" % 
             (epoch, all_loss, all_acc, valid_acc, duration))
    sys.stdout.flush()
  
  duration = time.time() - orig_begin_time
  duration /= 3600
  print('Done training, best_epoch: %d, best_acc: %.4f' % (best_step, best_acc))
  print('duration: %.2f hours' % duration)
  sys.stdout.flush()


def inspect(data, align):
  encoder = SubwordTextEncoder(FLAGS.vocab_file)

  def topNarg(arr, N=5):
    return np.sort(arr.argsort()[-N:][::-1])

  def decode(s, array=False):
    if array:
      return encoder.decode_list(s)
    return encoder.decode(s)

  def plot(index):

    # plot all attention weights
    cur_len = length[index]

    selected_private = private_ali[index]
    plt.plot(selected_private[:cur_len])
    selected_shared = shared_ali[index]
    plt.plot(selected_shared[:cur_len])

    plt.show()

    top_n = 10
    x = range(top_n)
    # plot top N attention weights
    private_index = topNarg(selected_private, top_n)
    sent = decode(sentence[index][private_index], array=True)
    plt.plot(x, selected_private[private_index])
    plt.xticks(x, sent)
    plt.title("private attention")
    print(decode(sentence[index][private_index]))
    plt.show()

    shared_index = topNarg(selected_shared, top_n)
    sent = decode(sentence[index][shared_index], array=True)
    plt.plot(x, selected_private[shared_index])
    plt.xticks(x, sent)
    plt.title("shared attention")
    print(decode(sentence[index][shared_index]))
    plt.show()


  # for every category, plot a attention weights that
  for key in data:
    task_label, label, sentence = data[key]
    private_ali, shared_ali = align[key]
    length = np.count_nonzero(sentence, axis=1)
    align_similarity = np.diag(cosine_similarity(private_ali, shared_ali))
    min_sim = np.argmin(align_similarity)
    plot(min_sim)

def test(sess, m_valid):
  m_valid.restore(sess)

  n_task = len(m_valid.tensors)
  errors = []

  print('dataset\terror rate')
  res, data, align = sess.run([m_valid.tensors, m_valid.data, m_valid.alignments])   # res = [[acc], [loss]]
  inspect(data, align)
  for i, ((acc, _), d, a) in enumerate(zip(res,data.values(), align.values())):
    err = 1-acc
    print('%s\t%.4f' % (fudan.get_task_name(i), err))
    errors.append(err)
  errors = np.asarray(errors)
  print('mean\t%.4f' % np.mean(errors))


def model_name():
    model_name = 'fudan-mtl'
    if FLAGS.model in ['lstm', 'gru']:
      model_name += '-' + FLAGS.model
    if FLAGS.adv:
      model_name += '-adv'
    if FLAGS.subword:
      model_name += '-subword'
    return model_name


def main(_):
  if FLAGS.build_data:
    build_data()
    return

  if FLAGS.subword:
    word_embed = None
  else:
    word_embed = util.load_embedding(word_dim=FLAGS.word_dim)

  with tf.Graph().as_default():
    all_train = []
    all_test = []
    data_iter = fudan.read_tfrecord(FLAGS.num_epochs, FLAGS.batch_size)
    for task_id, (train_data, test_data) in enumerate(data_iter):
      task_name = fudan.get_task_name(task_id)
      all_train.append((task_name, train_data))
      all_test.append((task_name, test_data))

    m_train, m_valid = mtl_model.build_train_valid_model(
            model_name(), word_embed, all_train, all_test, FLAGS.adv, FLAGS.test)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:

      try:
        m_train.restore(sess)
      except Exception as e:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())  # for file queue
        sess.run(init_op)
        tf.logging.warning("restore failed: {}".format(str(e)))

      print('='*80)

      if FLAGS.test:
        test(sess, m_valid)
      else:
        train(sess, m_train, m_valid)
      

if __name__ == '__main__':
  tf.app.run()
