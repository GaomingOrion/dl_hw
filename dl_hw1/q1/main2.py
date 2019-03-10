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
    'optimizer': tf.train.GradientDescentOptimizer(2e-3),
    'epochs': 150
}


# 模型减少一个隐藏层
class Model(BaseModel):
    def _graph_kernel(self, inp):
        x = self._fc_layer(name='fc1', inp=inp, units=2000)
        x = tf.nn.relu(x, name='ac1')
        logits = self._fc_layer(name='fc_final', inp=x, units=10)
        return logits

# load data
data, labels = common.load_train_data('../data/cifar-10-batches-py/')
xdev, ydev = common.load_dev_data('../data/cifar-10-batches-py/')

# mini_batch_sgd
n = 50000
minibatch_size = 64
data_gnr = lambda: common.mini_batch_gnr(n, minibatch_size, data, labels)

def main():
    model = BaseModel(params)
    model.train(data_gnr, minibatch_size, logpath='./logs/2', dev=(xdev, ydev))


if __name__ == '__main__':
    main()