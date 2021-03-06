import os
import random
import re

import numpy as np
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string("logdir", "saved_models/", "where to save the model")

flags.DEFINE_string("raw_data", "data/mtl-dataset/", "raw data path")

flags.DEFINE_boolean('plot', False, 'whether plot attention in test')

flags.DEFINE_string("data_dir", "data/generated", "tensorflow record dir")

flags.DEFINE_boolean('subword', False, 'use subword text encoder')

flags.DEFINE_boolean('use_attention', False, 'whether to use attention')

flags.DEFINE_boolean('vader', False, 'use vader')

flags.DEFINE_integer('restore_ckpt', -1, 'index of checkpoint to restore')

flags.DEFINE_boolean('freeze_shared', False, '')

flags.DEFINE_integer("num_filters", 100, "cnn number of output unit")

flags.DEFINE_float("lrn_rate", 0.001, "learning rate")

flags.DEFINE_float("l2_coef", 0.01, "l2 loss coefficient")

flags.DEFINE_float("keep_prob", 0.7, "dropout keep probability")

flags.DEFINE_float("symbol_dropout", 0.2, "dropout keep probability")

flags.DEFINE_integer("batch_size", 512, "batch size")

flags.DEFINE_integer("num_epochs", 100, "number of epochs")

flags.DEFINE_boolean('adv', False, 'set True to adv training')

flags.DEFINE_boolean('build_data', False, 'set True to generate data')

flags.DEFINE_boolean('test', False, 'set True to test')

flags.DEFINE_integer("num_layers", 2, "number of layers")

flags.DEFINE_integer("hidden_size", 32, "hidden size")

flags.DEFINE_boolean('is_regularize', True, "regularize")

flags.DEFINE_string("model", "cnn", "model type: cnn | lstm | gru")

flags.DEFINE_float("adv_weight", 0.25, "hyper parameter: λ")

flags.DEFINE_float("diff_weight", 0.01, "hyper parameter: γ")

flags.DEFINE_string("vocab_file", "data/generated/vocab.mtl.txt",
                    "vocab of train and test data")

flags.DEFINE_string("google_embed300_file",
                    "data/pretrain/embed300.google.npy",
                    "google news word embeddding")
flags.DEFINE_string("google_words_file",
                    "data/pretrain/google_words.lst",
                    "google words list")
flags.DEFINE_string("trimmed_embed300_file",
                    "data/generated/embed300.trim.npy",
                    "trimmed google embedding")

flags.DEFINE_string("senna_embed50_file",
                    "data/pretrain/embed50.senna.npy",
                    "senna words embeddding")
flags.DEFINE_string("senna_words_file",
                    "data/pretrain/senna_words.lst",
                    "senna words list")
flags.DEFINE_string("trimmed_embed50_file",
                    "data/generated/embed50.trim.npy",
                    "trimmed senna embedding")

flags.DEFINE_boolean('attention_diff', False, "use attention diff loss")

flags.DEFINE_integer("vocab_size", 8, "vocab size, unit of k")

FLAGS = tf.app.flags.FLAGS  # load FLAGS.hidden_size

PAD_WORD = "<pad>"
VOCAB_FILE = "vocab.mtl.txt"

# similar to nltk.tokenize.regexp.WordPunctTokenizer
# decimal, inter, 'm, 's, 'll, 've, 're, 'd, n't, words, punctuations
regexp = re.compile(r"\d*\.\d+|\d+|'m|'s|'ll|'ve|'re|'d|n't|\w+|[^\w\s]+")


def wordpunct_tokenizer(line):
    '''tokenizer sentence by decimal, inter,
    'm, 's, 'll, 've, 're, 'd, n't, words, punctuations
    '''
    # replace html tags, <br /> in imdb text
    line = re.sub(r'<[^>]*>', ' ', line)
    line = re.sub(r"n't", " n't", line)
    return regexp.findall(line)


def get_vocab_file():
    return "{}/{}".format(data_dir(), VOCAB_FILE)


def data_dir():
    if FLAGS.vader:
        return "{}-vader-{}k".format(FLAGS.data_dir, FLAGS.vocab_size)
    else:
        return "{}-{}k".format(FLAGS.data_dir, FLAGS.vocab_size)


def get_logdir():
    if FLAGS.vader:
        return "{}-vader-{}-{}-{}-{}/".format(FLAGS.logdir,
                                              FLAGS.hidden_size,
                                              FLAGS.vocab_size,
                                              FLAGS.keep_prob,
                                              FLAGS.lrn_rate)
    else:
        return "{}-{}-{}-{}-{}/".format(FLAGS.logdir,
                                        FLAGS.hidden_size,
                                        FLAGS.vocab_size,
                                        FLAGS.keep_prob,
                                        FLAGS.lrn_rate)


def write_vocab(vocab, vocab_file=get_vocab_file()):
    '''write vocab to the file

    Args:
      vocab: a set of tokens
      vocab_file: filename of the file
    '''
    base = os.path.dirname(vocab_file)
    tf.gfile.MakeDirs(base)
    with open(vocab_file, 'w') as f:
        f.write('%s\n' % PAD_WORD)  # make sure the pad id is 0
        for w in sorted(list(vocab)):
            f.write('%s\n' % w)


def _load_vocab(vocab_file):
    # load vocab from file
    vocab = []
    with open(vocab_file) as f:
        for line in f:
            w = line.strip()
            vocab.append(w)

    return vocab


def load_embedding(embed_file=None, word_dim=None):
    '''Load embeddings from file
    '''
    if embed_file is None:
        if word_dim == 50:
            embed_file = FLAGS.trimmed_embed50_file
        elif word_dim == 300:
            embed_file = FLAGS.trimmed_embed300_file

    embed = np.load(embed_file)
    return embed


def load_vocab2id(vocab_file=None):
    if vocab_file is None:
        vocab_file = get_vocab_file()

    vocab2id = {}
    vocab = _load_vocab(vocab_file)
    for id, token in enumerate(vocab):
        vocab2id[token] = id

    return vocab2id


def trim_embeddings(word_dim):
    '''trim unnecessary words from original pre-trained word embedding'''
    print('word_dim %d' % word_dim)
    if word_dim == 50:
        pretrain_embed_file = FLAGS.senna_embed50_file
        pretrain_words_file = FLAGS.senna_words_file
        trimed_embed_file = FLAGS.trimmed_embed50_file
    elif word_dim == 300:
        pretrain_embed_file = FLAGS.google_embed300_file
        pretrain_words_file = FLAGS.google_words_file
        trimed_embed_file = FLAGS.trimmed_embed300_file

    pretrain_embed = load_embedding(pretrain_embed_file, word_dim)
    pretrain_words2id = load_vocab2id(pretrain_words_file)

    word_embed = []
    vocab = _load_vocab(get_vocab_file())
    for w in vocab:
        if w in pretrain_words2id:
            id = pretrain_words2id[w]
            word_embed.append(pretrain_embed[id])
        else:
            vec = np.random.normal(0, 0.1, [word_dim])
            word_embed.append(vec)
    pad_id = -1
    word_embed[pad_id] = np.zeros([word_dim])

    word_embed = np.asarray(word_embed)
    np.save(trimed_embed_file, word_embed.astype(np.float32))


def stat_length(raw_data):
    '''get max_len and avg_len from data
    '''
    length = [len(example.sentence) for example in raw_data]
    length = sorted(length)
    length = np.asarray(length)

    max_len = np.max(length)
    avg_len = np.mean(length)
    med_len = np.median(length)
    print('max_len: %d, avg_len: %d, med_len: %d' % (max_len, avg_len, med_len))


def _map_tokens_to_ids(raw_example, vocab2id):
    '''inplace convert sentence from a list of tokens to a list of ids
    Args:
      raw_example: an instance of Raw_Example._asdict()
      vocab2id: dict<token, id> {token0: id0, ...}
    '''
    sent_id = []
    for token in raw_example['sentence']:
        if token in vocab2id:
            tok_id = vocab2id[token]
            sent_id.append(tok_id)
    raw_example['sentence'] = sent_id


def _pad_or_truncate(raw_example, max_len, pad_id):
    '''inplace pad or truncate a sentence to max_len
    Args:
      raw_example: an instance of Raw_Example._asdict()
      max_len: int
      pad_id: token id of PAD_WORD
    '''
    # truncate if len(sentence) > max_len
    # else nothing happens
    raw_example['sentence'] = raw_example['sentence'][:max_len]

    # pad if len(sentence) < max_len
    # else nothing happens
    pad_n = max_len - len(raw_example['sentence'])
    raw_example['sentence'].extend(pad_n * [pad_id])


def _write_text_for_debug(text_writer, raw_example, vocab2id):
    '''write raw_example['sentence'] to the disk, for debug

    Args:
      text_writer: text_writer = open(file, 'w')
      raw_example: an instance of Raw_Example._asdict()
      vocab2id: dict<token, id> {token0: id0, ...}
    '''
    tokens = []
    for token in raw_example['sentence']:
        if token in vocab2id:
            tokens.append(token)
    text_writer.write(' '.join(tokens) + '\n')


def write_as_tfrecord(raw_data, vocab2id, filename, max_len, build_func):
    '''convert the raw data to TFRecord format and write to disk

    Args:
      raw_data: a list of Raw_Example
      vocab2id: dict<token, id>
      filename: file to write in
      max_len: int, pad or truncate sentence to max_len
      build_func: function to convert Raw_Example to tf.train.SequenceExample
    '''
    writer = tf.python_io.TFRecordWriter(filename)
    # text_writer = open(filename+'.debug.txt', 'w')
    pad_id = vocab2id[PAD_WORD]

    for raw_example in raw_data:
        raw_example = raw_example._asdict()

        # _write_text_for_debug(text_writer, raw_example, vocab2id)
        _map_tokens_to_ids(raw_example, vocab2id)
        _pad_or_truncate(raw_example, max_len, pad_id)

        example = build_func(raw_example)
        writer.write(example.SerializeToString())
    writer.close()
    # text_writer.close()
    del raw_data


def read_tfrecord(filename, epoch, batch_size, parse_func, shuffle=True):
    '''read TFRecord file to get batch tensors for tensorflow models

    Returns:
      a tuple of batched tensors
    '''
    with tf.device('/cpu:0'):
        dataset = tf.data.TFRecordDataset([filename])
        # Parse the record into tensors
        dataset = dataset.map(parse_func)
        dataset = dataset.repeat(epoch)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        batch = iterator.get_next()
        return batch


def _shuf_and_write(filename):
    reader = tf.python_io.tf_record_iterator(filename)
    records = []
    for record in reader:
        # record is of <class 'bytes'>
        records.append(record)
    reader.close()

    random.shuffle(records)

    writer = tf.python_io.TFRecordWriter(filename)
    for record in records:
        writer.write(record)
    writer.close()
