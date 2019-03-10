import tensorflow as tf

import sys
sys.path.append('..')

import common
from model import BaseModel

# 模型超参数
params = {
    'weight_init': tf.contrib.layers.xavier_initializer(),
    'bias_init': tf.zeros_initializer(),
    'reg': tf.contrib.layers.l2_regularizer(scale=1e-4),  # add l2 regularization
    'optimizer': tf.train.GradientDescentOptimizer(2e-3),
    'epochs': 150
}

# load data
data, labels = common.load_train_data('../data/cifar-10-batches-py/')
xdev, ydev = common.load_dev_data('../data/cifar-10-batches-py/')

# mini_batch_sgd
n = 50000
minibatch_size = 64
data_gnr = lambda: common.mini_batch_gnr(n, minibatch_size, data, labels)

def main():
    model = BaseModel(params)
    model.train(data_gnr, minibatch_size, logpath='./logs/1', dev=(xdev, ydev))


if __name__ == '__main__':
    main()