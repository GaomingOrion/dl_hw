import tensorflow as tf

import sys
sys.path.append('..')

import common
from model import BaseModel

# 模型超参数
params = {
    'weight_init': tf.contrib.layers.xavier_initializer(),
    'bias_init': tf.zeros_initializer(),
    'reg': None,
    'optimizer': tf.train.GradientDescentOptimizer(0.5),
    'epochs': 500
}

# load data
data, labels = common.load_train_data('../data/cifar-10-batches-py/')
xdev, ydev = common.load_dev_data('../data/cifar-10-batches-py/')

# batch_gd
minibatch_size = 50000
def data_gnr():
    yield (data, labels)

def main():
    model = BaseModel(params)
    model.train(data_gnr, minibatch_size, logpath='./logs/1', dev=(xdev, ydev))


if __name__ == '__main__':
    main()