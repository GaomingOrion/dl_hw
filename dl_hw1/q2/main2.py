import tensorflow as tf
import numpy as np

import sys
sys.path.append('..')

import common
from model import BaseModel

# 模型超参数
params = {
    'weight_init': tf.contrib.layers.xavier_initializer(),
    'bias_init': tf.zeros_initializer(),
    'reg': None,
    'optimizer': tf.train.GradientDescentOptimizer(5e-5),
    'epochs': 150
}

# load data
data, labels = common.load_train_data('../data/cifar-10-batches-py/')
xdev, ydev = common.load_dev_data('../data/cifar-10-batches-py/')

# batch_gd
n = 50000
minibatch_size = 1
def data_gnr():
    sample_idx = np.random.choice(list(range(n)), n, replace=True)
    for i in sample_idx:
        yield (data[[i]], labels[[i]])

def main():
    model = BaseModel(params)
    model.train(data_gnr, minibatch_size, logpath='./logs/2', dev=(xdev, ydev))


if __name__ == '__main__':
    main()