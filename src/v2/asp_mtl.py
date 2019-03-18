import tensorflow as tf

FLAGS = tf.compat.v1.flags.FLAGS

from v2.dataset import *


def make_input_fn(category):
    problem_step = tf.Variable(tf.constant(0, dtype=tf.int64),
                               trainable=False,
                               dtype=tf.int64, name="problem_step")

    datasets = [load_data(category, task_id) for task_id in range(len(DATASETS))]

    dataset_iterators = [d.__iter__() for d in datasets]

    def mix_data(example):
        del example

        def get_next_from_dataset(dataset_iter):
            problem_step.assign_add(1)
            return dataset_iter.get_next()

        @tf.function
        def sample_task(curr_task, num_tasks_left, randnum):
            if num_tasks_left == 0:
                return get_next_from_dataset(dataset_iterators[curr_task])

            thresholds = [1.0 / len(DATASETS) for i in range(len(DATASETS))]
            thresholds = [sum(thresholds[:i + 1]) for i in range(len(thresholds))]

            prob_sum = thresholds[curr_task]
            if randnum < prob_sum:
                return get_next_from_dataset(dataset_iterators[curr_task])
            else:
                return sample_task(curr_task + 1, num_tasks_left - 1, randnum)

        return tf.data.Dataset.from_tensors(
            sample_task(0, len(DATASETS) - 1, tf.random.uniform([])))

    single_mtl_dataset = tf.data.Dataset.from_tensors(tf.zeros([1])).repeat()
    single_mtl_dataset = single_mtl_dataset.flat_map(mix_data)
    return single_mtl_dataset


def train_input_fn():
    return make_input_fn(0)


def eval_input_fn():
    return make_input_fn(1)
