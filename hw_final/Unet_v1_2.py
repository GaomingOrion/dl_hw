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

    def build_infer_graph(self):
        net_out = []
        for i in range(2):
            with tf.variable_scope('conv_%i'%i):
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
                    conv4 = slim.conv2d(conv4, 256, [3, 3], scope='conv4_2')


            with tf.variable_scope('conv_transpose_%i'%i):
                with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], normalizer_fn=None,
                                    normalizer_params={'is_training': self._is_training}):
                    up5 = slim.conv2d_transpose(conv4, 128, [2, 2], scope='up5', stride=2)
                    up5 = tf.concat([up5, conv3], axis=3)
                    conv5 = slim.conv2d(up5, 128, [3, 3], scope='conv5_1')
                    conv5 = slim.conv2d(conv5, 128, [3, 3], scope='conv5_2')
                    up6 = slim.conv2d_transpose(conv5, 64, [2, 2], scope='up6', stride=2)
                    up6 = tf.concat([up6, conv2], axis=3)
                    conv6 = slim.conv2d(up6, 64, [3, 3], scope='conv6_1')
                    conv6 = slim.conv2d(conv6, 64, [3, 3], scope='conv6_2')
                    up7 = slim.conv2d_transpose(conv6, 128, [2, 2], scope='up7', stride=2)
                    up7 = tf.concat([up7, conv1], axis=3)
                    conv7 = slim.conv2d(up7, 32, [3, 3], scope='conv7_1')
                    conv7 = slim.conv2d(conv7, 32, [3, 3], scope='conv7_2')
                    net_out.append(slim.conv2d(conv7, 1, [3, 3], scope='out'))
        return tf.concat(net_out, axis=3)

    def compute_loss(self, net_out):
        return tf.reduce_mean((net_out-self._image_ab)**2)

if __name__ == '__main__':
    m = Model()
    net_out = m.build_infer_graph()
    loss = m.compute_loss(net_out)