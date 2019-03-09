import tensorflow as tf
import numpy as np
import common
from model import Model

params = {
    'weight_init': tf.contrib.layers.xavier_initializer(),
    'bias_init': tf.zeros_initializer(),
    'reg': None,
    'lr': 1e-2,
    'epochs': 50
}


def main():
    n = 50000
    minibatch_size = 128
    data, labels = common.load_train_data()
    xdev, ydev = common.load_dev_data()

    def data_gnr():
        index = list(range(n))
        np.random.shuffle(index)
        start = 0
        while start < n - 1:
            end = start + minibatch_size
            yield data[index[start: end]], labels[index[start: end]]
            start = end

    model = Model(params)
    model.train(data_gnr, minibatch_size, xdev, ydev)


if __name__ == '__main__':
    main()