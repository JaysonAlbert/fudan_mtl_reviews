import collections
import os

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
        vocab_size = 2 ** 10 * FLAGS.vocab_size
        vocab = encoder.build_from_generator(data_generator(), vocab_size, 200, reserved_tokens=RESERVED_TOKENS)

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


def model_name():
    model_name = 'fudan-mtl'
    if FLAGS.model in ['lstm', 'gru']:
        model_name += '-' + FLAGS.model
    if FLAGS.adv:
        model_name += '-adv'
    if FLAGS.subword:
        model_name += '-subword'
    return model_name


def input_fn(is_train):
    return fudan.load_dataset(FLAGS.num_epochs, FLAGS.batch_size, is_train)


def run_model(num_examples, is_train, inspect_data=False, plot=False):
    if FLAGS.subword:
        word_embed = None
    else:
        word_embed = util.load_embedding(word_dim=FLAGS.hidden_size)

    name = "train" if is_train else "eval"

    with tf.Graph().as_default():

        global_step = tf.train.get_or_create_global_step()
        model = mtl_model.MTLModel(word_embed, input_fn(is_train=is_train), FLAGS.adv, is_train=is_train)
        model.build_train_op()
        model.set_saver(model_name())

        with tf.Session() as sess:
            try:
                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())  # for file queue
                sess.run(init_op)
                model.restore(sess)
            except Exception as e:
                tf.logging.warning("restore failed: {}".format(str(e)))

            summary_prefix = os.path.join(get_logdir(), model_name())
            writer = tf.summary.FileWriter(summary_prefix + '/{}'.format(name), sess.graph)

            n_task = len(model.tensors)
            batches = num_examples / FLAGS.batch_size

            merged = model.merged_summary(name)

            all_loss, all_acc = 0., 0.

            if inspect_data:
                eval_errors = collections.defaultdict(list)

            for batch in range(int(batches)):
                if inspect_data:
                    eval_fetch = [model.tensors, model.data, model.alignments, model.pred, model.separate_acc]
                    res, data, align, pred, separate_acc = sess.run(eval_fetch)  # res = [[acc], [loss]]

                    if plot:
                        inspect(data, align, pred)

                    if FLAGS.vader:
                        for i, ((acc, _), (private_acc, shared_acc, vader_acc)) in enumerate(zip(res, separate_acc.values())):
                            eval_errors[fudan.get_task_name(i)].append([float(acc), float(private_acc), float(shared_acc), float(vader_acc)])
                    else:
                        for i, ((acc, _), (private_acc, shared_acc)) in enumerate(zip(res, separate_acc.values())):
                            eval_errors[fudan.get_task_name(i)].append(
                                [float(acc), float(private_acc), float(shared_acc)])

                else:
                    if is_train:
                        train_fetch = [model.tensors, model.train_ops, merged, global_step]
                        res, _, summary, gs = sess.run(train_fetch)  # res = [[acc], [loss]]
                        writer.add_summary(summary, gs)
                    else:
                        eval_fetch = [model.tensors, merged, global_step]
                        res, summary, gs = sess.run(eval_fetch)  # res = [[acc], [loss]]
                        global eval_step
                        writer.add_summary(summary, eval_step)
                        eval_step = eval_step + 1



                res = np.array(res)

                all_loss += sum(res[:, 1].astype(np.float))
                all_acc += sum(res[:, 0].astype(np.float))


            all_loss /= (batches * n_task)
            all_acc /= (batches * n_task)

            if is_train:
                model.save(sess, global_step)

            if inspect_data:
                columns = ['err', 'private_err', 'shared_err', 'vader_err'] if FLAGS.vader else ['err', 'private_err', 'shared_err', 'vader_err']
                df = 1- pd.DataFrame(
                    data=np.array(list(eval_errors.values())).mean(axis=1),
                    index=list(eval_errors.keys()),
                    columns=columns
                )

                print(df)
                print(df.mean())

            return all_loss, all_acc

def train_model():
    return run_model(1600, is_train=True)


def eval_model():
    return run_model(400, is_train=False)


def get_or_create_eval_step():
    graph = tf.get_default_graph()

    eval_steps = graph.get_collection(tf.GraphKeys.EVAL_STEP)
    if len(eval_steps) == 1:
        return eval_steps[0]
    elif len(eval_steps) > 1:
        raise ValueError('Multiple tensors added to tf.GraphKeys.EVAL_STEP')
    else:
        counter = tf.get_variable(
            'eval_step',
            shape=[],
            dtype=tf.int64,
            initializer=tf.zeros_initializer(),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.EVAL_STEP])
        return counter


eval_step = 0


def main(_):
    if FLAGS.build_data:
        build_data()
        return

    if FLAGS.test:
        run_model(400, is_train=False, inspect_data=True, plot=FLAGS.plot)
    else:
        res = []

        for i in range(FLAGS.num_epochs):
            train_loss, train_acc = train_model()
            eval_loss, eval_acc = eval_model()
            res.append({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'eval_loss': eval_loss,
                'eval_acc': eval_acc
            })

            print("Epoch: {}, {}".format(i, res[-1]))

if __name__ == '__main__':
    tf.app.run()
