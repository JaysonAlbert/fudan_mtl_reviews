import collections
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder
from termcolor import cprint

from inputs import fudan
from inputs import util
from inputs.util import data_dir, get_vocab_file, get_logdir
from models import mtl_model

# tf.set_random_seed(0)
# np.random.seed(0)


PAD = "<pad>"
RESERVED_TOKENS = [PAD]

FLAGS = tf.app.flags.FLAGS


NUM_VALID_SAMPLES = 400


id = 0

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

      for d in fudan.load_unlabeled_data():
        yield d

    def summary(vocab):
      lens = [len(vocab.encode(sentence)) for sentence in data_generator()]
      length = sorted(lens)
      length = np.asarray(length)

      max_len = np.max(length)
      avg_len = np.mean(length)
      med_len = np.median(length)
      print('max_len: {}, avg_len: {}, med_len: {}'.format(max_len, avg_len, med_len))

    encoder = SubwordTextEncoder()
    vocab_size = 2**10 * FLAGS.vocab_size
    vocab = encoder.build_from_generator(data_generator(), vocab_size, 200,reserved_tokens=RESERVED_TOKENS)

    vocab_file = get_vocab_file()
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
      train_record_file = os.path.join(data_dir(), fudan.DATASETS[task_id] + '.train.tfrecord')
      test_record_file = os.path.join(data_dir(), fudan.DATASETS[task_id] + '.test.tfrecord')
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
    util.trim_embeddings(FLAGS.hidden_size)

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

  summary_prefix = os.path.join(get_logdir(), model_name())

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

    if best_acc < valid_acc:
      best_acc = valid_acc
      best_step = epoch

    m_train.save(sess, global_step)
      
    print("Epoch %d loss %.2f acc %.2f %.4f time %.2f" % 
             (epoch, all_loss, all_acc, valid_acc, duration))
    sys.stdout.flush()
  
  duration = time.time() - orig_begin_time
  duration /= 3600
  print('Done training, best_epoch: %d, best_acc: %.4f' % (best_step, best_acc))
  print('duration: %.2f hours' % duration)
  sys.stdout.flush()


def topNarg(arr, N=5):
    return np.sort(arr.argsort()[-N:][::-1])


def decode(s, array=False):
    encoder = SubwordTextEncoder(get_vocab_file())
    if array:
        return encoder.decode_list(s)
    return encoder.decode(s)


def color_print(orig, color_list):
    sent = decode(orig, True)

    sentence_length = 0
    for i, v in enumerate(orig):
        word = sent[i].replace("_", " ")
        if i in color_list:
            cprint(word, "red", end="")
        else:
            print(word, end="")

        sentence_length = sentence_length + len(word)

        if sentence_length > 100:
            sentence_length = 0
            print()

    print()


def plot(sentence, private_attention, shared_attention):
    # plot all attention weights
    ax1 = plt.subplot(212)
    ax2 = plt.subplot(221)
    ax3 = plt.subplot(222)

    ax1.plot(private_attention, label="private")
    ax1.plot(shared_attention, label="shared")

    ax1.legend()
    plt.ylabel("weight")
    plt.xlabel("index")

    top_n = min(len(private_attention), 10)
    x = range(top_n)

    def plot_attention(ax, attention, type):
        top_index = topNarg(attention, top_n)
        sent = decode(sentence[top_index], array=True)
        ax.plot(x, attention[top_index])

        plt.sca(ax)
        plt.xticks(x, sent, rotation='vertical')

        plt.title("{} attention".format(type))

        print("Attention type: {}:".format(type))
        color_print(sentence, top_index)

        plt.ylabel("weight")
        plt.xlabel("word")
        plt.tight_layout()

    plot_attention(ax2, private_attention, "private")

    print('-' * 100)

    plot_attention(ax3, shared_attention, "shared")

    tf.gfile.MakeDirs("fig")

    plt.savefig("fig/{}.png".format(id))

    plt.clf()


def inspect(data, align, pred):
  # for every category, plot a attention weights that

  for key in data:
    if FLAGS.vader:
      task_labels, labels, sentences, vaders = data[key]
    else:
      task_labels, labels, sentences = data[key]
    task_preds = pred[key]
    private_alis, shared_alis = align[key]
    lengths = np.count_nonzero(sentences, axis=1)
    align_similarity = np.diag(cosine_similarity(private_alis, shared_alis))
    min_sim = np.argmin(align_similarity)
    for i in range(len(sentences)):
      task_pred = task_preds[i]
      length = lengths[i]
      label = labels[i]
      private_attention = private_alis[i][:length]
      shared_attention = shared_alis[i][:length]
      sentence = sentences[i][:length]

      # ----------------------------------------------------------------------------------------------------
      global id
      print("=" * 100)
      print("Id: {}".format(id))
      print("Task: {}".format(key))
      print("Label: {}".format(label))
      if label != task_pred:
        print("Result: Failed")
      else:
        print("Reuslt: Pass")

      plot(sentence, private_attention, shared_attention)
      id = id + 1
      # ----------------------------------------------------------------------------------------------------
      print("=" * 100)
      print("\n\n")


def check_separate_acc(data, align, pred, separate_acc):
    task_name = 'sports_outdoors'
    if FLAGS.vader:
        task_labels, labels, sentences, vaders = data[task_name]
        private_acc, shared_acc, vader_acc, logits1, logits2, logits3 = separate_acc[task_name]
    else:
        task_labels, labels, sentences = data[task_name]
        private_acc, shared_acc, logits1, logits2 = separate_acc[task_name]
    private_align, shared_align = align[task_name]
    pred = pred[task_name]




def test(sess, m_valid):
  m_valid.restore(sess)

  errors = collections.defaultdict(list)

  for _ in range(int(NUM_VALID_SAMPLES / FLAGS.batch_size)):
      res, data, align, pred, separate_acc = sess.run([m_valid.tensors, m_valid.data, m_valid.alignments, m_valid.pred,
                                                       m_valid.separate_acc])  # res = [[acc], [loss]]
      # inspect(data, align, pred)
      # check_separate_acc(data, align, pred, separate_acc)
      if FLAGS.vader:
          for i, ((acc, _), (private_acc, shared_acc, vader_acc, logits1, logits2, logits3)) in enumerate(
                  zip(res, separate_acc.values())):
              errors[fudan.get_task_name(i)].append(
                  [float(acc), float(private_acc), float(shared_acc), float(vader_acc)])
      else:
          for i, ((acc, _), (private_acc, shared_acc, logits1, logits2)) in enumerate(zip(res, separate_acc.values())):
            errors[fudan.get_task_name(i)].append([float(acc), float(private_acc), float(shared_acc)])

      f = open("result.json", 'w')
      json.dump(errors,f)
      f.close()

  columns = ['err', 'private_err', 'shared_err', 'vader_err'] if FLAGS.vader else ['err', 'private_err', 'shared_err']

  df = 1 - pd.DataFrame(
      data=np.array(list(errors.values())).mean(axis=1),
      index=list(errors.keys()),
      columns=columns
  )
  print(df)
  print(df.mean())
  df.mean().to_csv('result_{}.csv'.format(FLAGS.restore_ckpt))


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
    word_embed = util.load_embedding(word_dim=FLAGS.hidden_size)

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
