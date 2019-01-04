import os
import time
import sys
import tensorflow as tf
import numpy as np

from inputs import util
from inputs import fudan
from models import mtl_model
# tf.set_random_seed(0)
# np.random.seed(0)
tf.enable_eager_execution()

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
    
  def _build_data(all_data):
    print('build data')
    vocab2id = util.load_vocab2id()

    for task_id, task_data in enumerate(all_data):
      train_data, test_data = task_data
      fudan.write_as_tfrecord(train_data, test_data, task_id, vocab2id)

  def _trim_embed():
    print('trimming pretrained embeddings')
    # util.trim_embeddings(50)
    util.trim_embeddings(FLAGS.word_dim)

  print('load raw data')
  all_data = []
  for task_data in fudan.load_raw_data():
    all_data.append(task_data)
  
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

  num_step = 0
  for epoch in range(FLAGS.num_epochs):
    all_loss, all_acc = 0., 0.
    for batch in range(batches):
      train_fetch = [m_train.tensors, m_train.train_ops, merged_train]

      res, _, summary = sess.run(train_fetch)    # res = [[acc], [loss]]
      res = np.array(res)

      train_writer.add_summary(summary, num_step)

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

    res, summary = sess.run([m_valid.tensors, merged_valid])
    for  acc, _ in res:
      valid_acc += acc
    valid_writer.add_summary(summary, num_step)

    valid_acc /= n_task

    if best_acc < valid_acc:
      best_acc = valid_acc
      best_step = epoch
      m_train.save(sess, epoch)
      
    print("Epoch %d loss %.2f acc %.2f %.4f time %.2f" % 
             (epoch, all_loss, all_acc, valid_acc, duration))
    sys.stdout.flush()
  
  duration = time.time() - orig_begin_time
  duration /= 3600
  print('Done training, best_epoch: %d, best_acc: %.4f' % (best_step, best_acc))
  print('duration: %.2f hours' % duration)
  sys.stdout.flush()

def test(sess, m_valid):
  m_valid.restore(sess)
  n_task = len(m_valid.tensors)
  errors = []

  print('dataset\terror rate')
  res = sess.run(m_valid.tensors)   # res = [[summary], [acc], [loss]]
  for i, acc, _ in enumerate(res):
    err = 1-acc
    print('%s\t%.4f' % (fudan.get_task_name(i), err))
    errors.append(err)
  errors = np.asarray(errors)
  print('mean\t%.4f' % np.mean(errors))


def model_name():
    model_name = 'fudan-mtl'
    if FLAGS.model == 'lstm':
      model_name += '-lstm'
    if FLAGS.adv:
      model_name += '-adv'
    return model_name


def main(_):
  if FLAGS.build_data:
    build_data()
    exit()

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
      
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())# for file queue
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
      sess.run(init_op)
      print('='*80)

      if FLAGS.test:
        test(sess, m_valid)
      else:
        train(sess, m_train, m_valid)
      

if __name__ == '__main__':
  tf.app.run()
