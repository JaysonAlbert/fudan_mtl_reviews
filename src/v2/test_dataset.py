from v2.asp_mtl import *


def test_generate_data():
    # generate data
    for category in range(len(SUFFIX)):
        for task_id in range(len(DATASETS)):
            generated_data(category, task_id)


# load data
def test_load_data():
    dataset = load_data(0, 3)
    for data in dataset.take(5):
        print(data)


if __name__ == "__main__":

    test_generate_data()

    for data in train_input_fn():
        print(data)
