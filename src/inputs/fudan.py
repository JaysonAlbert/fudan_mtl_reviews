import os
from collections import namedtuple

import tensorflow as tf

from inputs import util
from inputs.util import data_dir

FLAGS = tf.app.flags.FLAGS

DATASETS = ['apparel', 'baby', 'books', 'camera_photo', 'electronics',
            'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines',
            'music', 'software', 'sports_outdoors', 'toys_games', 'video']
# 'dvd','MR',
SUFFIX = ['.task.train', '.task.test', '.task.unlabel']
Raw_Example = namedtuple('Raw_Example', 'label task sentence')
V_Raw_Example = namedtuple('V_Raw_Example', 'label task sentence vader')

MAX_LEN = 500


def get_task_name(task_id):
    return DATASETS[task_id]


def _load_raw_data_from_file(filename, task_id):
    data = []
    with open(filename, encoding='utf-8') as f:

        if FLAGS.vader:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
        # try:
        for line in f:
            segments = line.strip().split('\t')
            if len(segments) == 2:
                label = int(segments[0])
                tokens = segments[1].split(' ')
                if FLAGS.vader:
                    vader_score = analyzer.polarity_scores(segments[1])['compound']
                    example = V_Raw_Example(label, task_id, tokens, vader_score)
                else:
                    example = Raw_Example(label, task_id, tokens)
                data.append(example)
        # except UnicodeDecodeError:
        #   print(filename)
        #   exit()
    return data


def load_unlabeled_data():
    for task in DATASETS:
        file = "{}/{}{}".format(FLAGS.raw_data, task, SUFFIX[2])
        with open(file, encoding='utf-8') as f:
            for line in f:
                yield line


def _load_raw_data(dataset_name, task_id):
    train_file = os.path.join(FLAGS.raw_data, dataset_name + '.task.train')
    train_data = _load_raw_data_from_file(train_file, task_id)
    test_file = os.path.join(FLAGS.raw_data, dataset_name + '.task.test')
    test_data = _load_raw_data_from_file(test_file, task_id)
    return train_data, test_data


def load_raw_data():
    for task_id, dataset in enumerate(DATASETS):
        yield _load_raw_data(dataset, task_id)


def build_vocab(raw_data):
    vocab = set()
    for example in raw_data:
        for w in example.sentence:
            vocab.add(w)

    return vocab


def _build_sequence_example(raw_example):
    '''build tf.train.SequenceExample from Raw_Example
    context features : label, task
    sequence features: sentence

    Args:
      raw_example : type Raw_Example._asdict()

    Returns:
      tf.trian.SequenceExample
    '''
    ex = tf.train.SequenceExample()

    label = raw_example['label']
    ex.context.feature['label'].int64_list.value.append(label)

    task = raw_example['task']
    ex.context.feature['task'].int64_list.value.append(task)

    if FLAGS.vader:
        vader_score = raw_example['vader']
        ex.context.feature['vader'].float_list.value.append(vader_score)

    for word_id in raw_example['sentence']:
        word = ex.feature_lists.feature_list['sentence'].feature.add()
        word.int64_list.value.append(word_id)

    return ex


def write_as_tfrecord(train_data, test_data, task_id, vocab2id):
    '''convert the raw data to TFRecord format and write to disk
    '''
    dataset = DATASETS[task_id]
    train_record_file = os.path.join(data_dir(), dataset + '.train.tfrecord')
    test_record_file = os.path.join(data_dir(), dataset + '.test.tfrecord')

    util.write_as_tfrecord(train_data,
                           vocab2id,
                           train_record_file,
                           MAX_LEN,
                           _build_sequence_example)
    util.write_as_tfrecord(test_data,
                           vocab2id,
                           test_record_file,
                           MAX_LEN,
                           _build_sequence_example)

    util._shuf_and_write(train_record_file)


def _parse_tfexample(serialized_example):
    '''parse serialized tf.train.SequenceExample to tensors
    context features : label, task
    sequence features: sentence
    '''
    context_features = {'label': tf.FixedLenFeature([], tf.int64),
                        'task': tf.FixedLenFeature([], tf.int64)}
    if FLAGS.vader:
        context_features['vader'] = tf.FixedLenFeature([], tf.float32)
    sequence_features = {'sentence': tf.FixedLenSequenceFeature([], tf.int64)}
    context_dict, sequence_dict = tf.parse_single_sequence_example(
        serialized_example,
        context_features=context_features,
        sequence_features=sequence_features)

    sentence = sequence_dict['sentence']
    label = context_dict['label']
    task = context_dict['task']

    if FLAGS.vader:
        vader = context_dict['vader']
        return task, label, sentence, vader
    else:
        return task, label, sentence
    

def load_dataset(epoch, batch_size, train):

    for dataset in DATASETS:
        suffix =  '.train.tfrecord' if train else '.test.tfrecord'
        record_file = os.path.join(data_dir(), dataset + suffix)

        yield dataset, util.read_tfrecord(
            record_file,
            epoch,
            batch_size,
            _parse_tfexample,
            shuffle=train
        )
