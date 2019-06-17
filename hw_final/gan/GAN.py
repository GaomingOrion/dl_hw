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
        self.smothing = 0.9

    def gen(self, img_bw, scope='gen'):
        with tf.variable_scope(scope):
            with tf.variable_scope('conv'):
                conv1 = slim.conv2d(img_bw, 32, [3, 3], scope='conv1_1')
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


            with tf.variable_scope('conv_transpose'):
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
                net_out = slim.conv2d(conv7, 2, [3, 3], scope='out')
        return net_out

    def dis(self, img_bw, img_ab, scope='dis'):
        img = tf.concat([img_bw, img_ab], axis=3)
        with tf.variable_scope(scope):
            with tf.variable_scope('conv'):
                conv1 = slim.conv2d(img, 32, [3, 3], scope='conv1_1')
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
            fc_in = tf.layers.flatten(conv4)
            fc1 = tf.layers.dense(fc_in, 1000)
            logits = tf.layers.dense(fc1, 1)
        return logits

    def build_infer_graph(self):
        return self.gen(self._image_bw)

    def build_train_graph(self):
        gen_out = self.gen(self._image_bw)
        logits_real = self.dis(self._image_bw, self._image_ab, 'dis_real')
        logits_fake = self.dis(self._image_bw, gen_out, 'dis_fake')

        gen_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake))
        dis_real_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real)*self.smothing)
        dis_fake_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake))

        dis_loss = tf.reduce_mean(tf.reduce_mean(dis_real_ce)+tf.reduce_mean(dis_fake_ce))
        gen_loss_gan = tf.reduce_mean(gen_ce)
        gen_loss_l1 = tf.reduce_mean(tf.abs(self._image_ab-gen_out))*100
        gen_loss = gen_loss_gan + gen_loss_l1
        return gen_out, dis_loss, gen_loss


if __name__ == '__main__':
    m = Model()
    gen_out, dis_loss, gen_loss = m.build_train_graph()