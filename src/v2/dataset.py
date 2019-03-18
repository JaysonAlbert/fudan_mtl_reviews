import json
import os
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf

from inputs.util import VOCAB_FILE, data_dir
from v2 import upgraded_utils as generator_utils

EOS = "<EOS>"
RESERVED_TOKENS = [EOS]

FLAGS = tf.compat.v1.flags.FLAGS

DATASETS = ['apparel', 'baby', 'books', 'camera_photo',  'electronics',
      'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines',
      'music', 'software', 'sports_outdoors', 'toys_games', 'video']
# 'dvd','MR',
SUFFIX = ['.task.train', '.task.test', '.task.unlabel']


SHUFFLE_BUFFER_SIZE = [
    1400, 400, 2000  # train, test, unlabeled
]

MAX_LEN = 512


def tfrecord_filenames(category, task_id):
    return [os.path.join(data_dir(), "{}.{}.tfrecord".format(DATASETS[task_id], SUFFIX[category].split('.')[-1]))]


def generate_samples(category, task_id):
    '''

    :param category: 0: train, 1: test, 2: unlabeld
    :return:
    '''

    task = DATASETS[task_id]
    filename = os.path.join(FLAGS.raw_data, task + SUFFIX[category])
    with open(filename) as f:
        for line in f:
            if category == 2:
                yield {
                    'inputs': line,
                    'task': task_id
                }
            else:
                segments = line.strip().split('\t')
                if len(segments) > 1:
                    yield {
                        'inputs': segments[1],
                        'task': task_id,
                        'label': int(segments[0])
                    }


def generate_sample_for_vocab():
    for i in range(3):
        for task_id in range(len(DATASETS)):
            for d in generate_samples(i, task_id):
                yield d['inputs']


def get_encoder():
    vocab_size = 2 ** 10 * FLAGS.vocab_size
    return generator_utils.get_or_generate_vocab_inner(data_dir(),
                                                VOCAB_FILE,
                                                vocab_size, generate_sample_for_vocab(),
                                                200,
                                                reserved_tokens=RESERVED_TOKENS)


def generated_data(category, task_id):
    '''

    :param category: 0: train, 1: test, 2: unlabeld
    :return:
    '''

    def summary(vocab):

      statics_file = 'statics.json'
      filename = os.path.join(data_dir(), statics_file)
      if tf.io.gfile.exists(filename):
          with open(filename) as f:
              statics = json.load(f)
      else:
          lens = [len(vocab.encode(sentence)) for sentence in generate_sample_for_vocab()]
          length = sorted(lens)
          length = np.asarray(length)

          statics = {
              'max_len': int(np.max(length)),
              'avg_len': int(np.mean(length)),
              'med_len': int(np.median(length))
          }

          with open(filename,'w') as f:
              json.dump(statics, f)

      print('max_len: {}, avg_len: {}, med_len: {}'.format(statics['max_len'], statics['avg_len'], statics['med_len']))

    encoder = get_encoder()

    summary(encoder)

    def generate_encoded():
        for sample in generate_samples(category, task_id):
            sample["inputs"] = encoder.encode(sample["inputs"])
            sample["inputs"].append(get_encoder().encode(EOS)[0])
            sample["task"] = [sample["task"]]
            if "label" in sample:
                sample["label"] = [sample["label"]]
            yield sample

    filenames = tfrecord_filenames(category, task_id)
    generator_utils.generate_files(generate_encoded(), filenames)
    generator_utils.shuffle_dataset(filenames)


def load_data(category, task_id):
    '''

    :param category: 0: train, 1: test, 2: unlabeld
    :return:
    '''

    if category == 2:
        reading_spec = {
            'inputs': tf.io.VarLenFeature(tf.int64),
            'task': tf.io.FixedLenFeature([1], tf.int64)
        }
    else:
        reading_spec = {
            'inputs': tf.io.VarLenFeature(tf.int64),
            'task': tf.io.FixedLenFeature([1], tf.int64),
            'label': tf.io.FixedLenFeature([1], tf.int64)
        }

    def decode_samples(example_proto):
        parsed_feature = tf.io.parse_single_example(example_proto, reading_spec)
        parsed_feature["inputs"] = tf.sparse.to_dense(parsed_feature["inputs"])
        return parsed_feature

    filenames = tfrecord_filenames(category, task_id)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(decode_samples, cpu_count())

    batch_size = FLAGS.batch_size
    num_epochs = FLAGS.num_epochs

    if category == 2:
        padded_shapes = {
            'inputs': [None],
            'task': [None]
        }
    else:
        padded_shapes = {
            'inputs': [None],
            'task': [None],
            'label': [None]
        }

    return dataset.padded_batch(batch_size, padded_shapes).repeat(num_epochs).shuffle(SHUFFLE_BUFFER_SIZE[category])
