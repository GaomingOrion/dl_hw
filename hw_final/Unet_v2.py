import tensorflow as tf
from tensorflow.contrib import slim
from common import config

class Model:
    def __init__(self):
        self._is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        self._image_bw = tf.placeholder(tf.float32, shape=(None,)+ config.image_shape+ (1,) , name='bw')
        self._image_ab = tf.placeholder(tf.float32, shape=(None,)+ config.image_shape+ (2,) , name='ab')
        self._label = tf.placeholder(tf.int32, shape=(None, ), name='label')

        self.place_holders = {'image_bw': self._image_bw,
                              'image_ab': self._image_ab,
                              'label': self._label,
                              'is_training': self._is_training
                              }

    def _gragh1(self):
        with tf.variable_scope('conv_job1'):
            with slim.arg_scope([slim.conv2d], normalizer_fn=None,
                                normalizer_params={'is_training': self._is_training}):
                conv1 = slim.conv2d(self._image_bw, 32, [3, 3], scope='conv1_1')
                conv1 = slim.conv2d(conv1, 32, [3, 3], scope='conv1_2')
                pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
                conv2 = slim.conv2d(pool1, 64, [3, 3], scope='conv2_1')
                conv2 = slim.conv2d(conv2, 64, [3, 3], scope='conv2_2')
                pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
                conv3 = slim.conv2d(pool2, 128, [3, 3], scope='conv3_1')
                conv3 = slim.conv2d(conv3, 128, [3, 3], scope='conv3_2')
                pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')
                conv4 = slim.conv2d(pool3, 256, [3, 3], scope='conv4_1')
        return conv4

    def build_infer_graph(self):
        conv_class = self._gragh1()
        fc_in = tf.layers.flatten(conv_class)
        fc1 = tf.layers.dense(fc_in, 1000, activation=tf.sigmoid)
        logits = tf.layers.dense(fc1, 10)
        with tf.variable_scope('conv'):
            with slim.arg_scope([slim.conv2d], normalizer_fn=None,
                                normalizer_params={'is_training': self._is_training}):
                conv1 = slim.conv2d(self._image_bw, 32, [3, 3], scope='conv1_1')
                conv1 = slim.conv2d(conv1, 32, [3, 3], scope='conv1_2')
                pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
                conv2 = slim.conv2d(pool1, 64, [3, 3], scope='conv2_1')
                conv2 = slim.conv2d(conv2, 64, [3, 3], scope='conv2_2')
                pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
                conv3 = slim.conv2d(pool2, 128, [3, 3], scope='conv3_1')
                conv3 = slim.conv2d(conv3, 128, [3, 3], scope='conv3_2')
                pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')
                conv4 = slim.conv2d(pool3, 256, [3, 3], scope='conv4_1')
                conv4 = tf.concat([conv4, conv_class], axis=3)
                conv4 = slim.conv2d(conv4, 256*2, [3, 3], scope='conv4_2')


        with tf.variable_scope('conv_transpose'):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], normalizer_fn=None,
                                normalizer_params={'is_training': self._is_training}):
                up5 = slim.conv2d_transpose(conv4, 128*2, [2, 2], scope='up5', stride=2)
                up5 = tf.concat([up5, conv3], axis=3)
                conv5 = slim.conv2d(up5, 128*2, [3, 3], scope='conv5_1')
                conv5 = slim.conv2d(conv5, 128*2, [3, 3], scope='conv5_2')
                up6 = slim.conv2d_transpose(conv5, 64*2, [2, 2], scope='up6', stride=2)
                up6 = tf.concat([up6, conv2], axis=3)
                conv6 = slim.conv2d(up6, 64*2, [3, 3], scope='conv6_1')
                conv6 = slim.conv2d(conv6, 64*2, [3, 3], scope='conv6_2')
                up7 = slim.conv2d_transpose(conv6, 32*2, [2, 2], scope='up7', stride=2)
                up7 = tf.concat([up7, conv1], axis=3)
                conv7 = slim.conv2d(up7, 32*2, [3, 3], scope='conv7_1')
                conv7 = slim.conv2d(conv7, 32*2, [3, 3], scope='conv7_2')
                net_out = slim.conv2d(conv7, 2, [3, 3], scope='out')
        return net_out, logits

    def compute_loss(self, net_out, logits):
        label_onehot = tf.one_hot(self._label, 10)
        loss1 = tf.reduce_mean((net_out-self._image_ab)**2)
        loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_onehot, logits=logits))
        return loss1 + 0.001*loss2

if __name__ == '__main__':
    m = Model()
    net_out, logits = m.build_infer_graph()
    loss = m.compute_loss(net_out, logits)