import tensorflow as tf
from tensorflow.contrib import slim
from common import config

class Model:
    def __init__(self):
        self._is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        self._image_bw = tf.placeholder(tf.float32, shape=(None,)+ config.image_shape+ (1,) , name='bw')
        self._image_ab = tf.placeholder(tf.float32, shape=(None,)+ config.image_shape+ (2,) , name='ab')
        self._label = tf.placeholder(tf.int32, shape=(None, ), name='label')
        self._alpha = tf.placeholder(tf.float32, shape=(), name='alpha')
        self._label_onehot = tf.one_hot(self._label, 10)

        self.place_holders = {'image_bw': self._image_bw,
                              'image_ab': self._image_ab,
                              'label': self._label,
                              'is_training': self._is_training,
                              'alpha': self._alpha
                              }

    def gen(self, img_bw, label, scope='gen'):
        channel_lst = [64, 128, 256, 512]
        to_tile = tf.reshape(label, [-1, 1, 1, 10])
        to_concat = tf.tile(to_tile, [1, 4, 4, 1])
        with tf.variable_scope(scope):
            with tf.variable_scope('conv'):
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.leaky_relu):
                    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                        normalizer_params={'is_training': self._is_training}):
                        conv1 = slim.conv2d(img_bw, channel_lst[0], [3, 3], scope='conv1_1')
                        conv1 = slim.conv2d(conv1, channel_lst[0], [3, 3], scope='conv1_2')
                        pool1 = slim.avg_pool2d(conv1, [2, 2], scope='pool1')
                        conv2 = slim.conv2d(pool1, channel_lst[1], [3, 3], scope='conv2_1')
                        conv2 = slim.conv2d(conv2, channel_lst[1], [3, 3], scope='conv2_2')
                        pool2 = slim.avg_pool2d(conv2, [2, 2], scope='pool2')
                        conv3 = slim.conv2d(pool2, channel_lst[2], [3, 3], scope='conv3_1')
                        conv3 = slim.conv2d(conv3, channel_lst[2], [3, 3], scope='conv3_2')
                        pool3 = slim.avg_pool2d(conv3, [2, 2], scope='pool3')
                        conv4 = slim.conv2d(pool3, channel_lst[3], [3, 3], scope='conv4_1')
                        conv4 = tf.concat([conv4, to_concat], axis=3)
                        conv4 = slim.conv2d(conv4, channel_lst[3], [3, 3], scope='conv4_2')

            with tf.variable_scope('conv_transpose'):
                with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': self._is_training}):
                    with slim.arg_scope([slim.conv2d_transpose], activation_fn=tf.nn.leaky_relu):
                        up5 = slim.conv2d_transpose(conv4, channel_lst[2], [2, 2], scope='up5', stride=2)
                        up5 = tf.concat([up5, conv3], axis=3)
                        conv5 = slim.conv2d(up5, channel_lst[2], [3, 3], scope='conv5_1')
                        conv5 = slim.conv2d(conv5, channel_lst[2], [3, 3], scope='conv5_2')
                        up6 = slim.conv2d_transpose(conv5, channel_lst[1], [2, 2], scope='up6', stride=2)
                        up6 = tf.concat([up6, conv2], axis=3)
                        conv6 = slim.conv2d(up6, channel_lst[1], [3, 3], scope='conv6_1')
                        conv6 = slim.conv2d(conv6, channel_lst[1], [3, 3], scope='conv6_2')
                        up7 = slim.conv2d_transpose(conv6, channel_lst[0], [2, 2], scope='up7', stride=2)
                        up7 = tf.concat([up7, conv1], axis=3)
                        conv7 = slim.conv2d(up7, channel_lst[0], [3, 3], scope='conv7_1')
                        conv7 = slim.conv2d(conv7, channel_lst[0], [3, 3], scope='conv7_2')
                    net_out = slim.conv2d(conv7, 2, [1, 1], scope='out', activation_fn=tf.nn.tanh)
        return net_out

    def dis(self, img_bw, img_ab, scope='dis', reuse=None):
        channel_lst1 = [32, 64, 128, 256]
        channel_lst2 = [32, 64, 128, 256]
        img = tf.concat([img_bw, img_ab], axis=3)
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope('conv1'):
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.leaky_relu):
                    conv1 = slim.conv2d(img, channel_lst1[0], [3, 3], scope='conv1_1')
                    conv1 = slim.conv2d(conv1, channel_lst1[0], [3, 3], scope='conv1_2')
                    pool1 = slim.avg_pool2d(conv1, [2, 2], scope='pool1')
                    conv2 = slim.conv2d(pool1, channel_lst1[1], [3, 3], scope='conv2_1')
                    conv2 = slim.conv2d(conv2, channel_lst1[1], [3, 3], scope='conv2_2')
                    pool2 = slim.avg_pool2d(conv2, [2, 2], scope='pool2')
                    conv3 = slim.conv2d(pool2, channel_lst1[2], [3, 3], scope='conv3_1')
                    conv3 = slim.conv2d(conv3, channel_lst1[2], [3, 3], scope='conv3_2')
                    pool3 = slim.avg_pool2d(conv3, [2, 2], scope='pool3')
                    conv4 = slim.conv2d(pool3, channel_lst1[3], [3, 3], scope='conv4_1')
                    conv4 = slim.conv2d(conv4, channel_lst1[3], [3, 3], scope='conv4_2')
            fc_in = tf.layers.flatten(conv4)
            fc1 = tf.layers.dense(fc_in, 1000, activation=tf.tanh)
            logits1 = tf.layers.dense(fc1, 1)

            with tf.variable_scope('conv2'):
                # with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                #                     normalizer_params={'is_training': self._is_training}):
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.leaky_relu):
                    conv1 = slim.conv2d(img, channel_lst2[0], [3, 3], scope='conv1_1')
                    conv1 = slim.conv2d(conv1, channel_lst2[0], [3, 3], scope='conv1_2')
                    pool1 = slim.avg_pool2d(conv1, [2, 2], scope='pool1')
                    conv2 = slim.conv2d(pool1, channel_lst2[1], [3, 3], scope='conv2_1')
                    conv2 = slim.conv2d(conv2, channel_lst2[1], [3, 3], scope='conv2_2')
                    pool2 = slim.avg_pool2d(conv2, [2, 2], scope='pool2')
                    conv3 = slim.conv2d(pool2, channel_lst2[2], [3, 3], scope='conv3_1')
                    conv3 = slim.conv2d(conv3, channel_lst2[2], [3, 3], scope='conv3_2')
                    pool3 = slim.avg_pool2d(conv3, [2, 2], scope='pool3')
                    conv4 = slim.conv2d(pool3, channel_lst2[3], [3, 3], scope='conv4_1')
                    conv4 = slim.conv2d(conv4, channel_lst2[3], [3, 3], scope='conv4_2')
            fc_in = tf.layers.flatten(conv4)
            fc2 = tf.layers.dense(fc_in, 1000, activation=tf.tanh)
            logits2 = tf.layers.dense(fc2, 10)
        return logits1, logits2

    def build_infer_graph(self):
        return self.gen(2*self._image_bw-1, self._label_onehot)

    def build_train_graph(self):
        gen_out = self.gen(2*self._image_bw-1, self._label_onehot)
        logits_real, logits_real_class = self.dis(2*self._image_bw-1, 2*self._image_ab-1,  'dis', reuse=False)
        logits_fake, logits_fake_class = self.dis(2*self._image_bw-1, gen_out, 'dis', reuse=True)

        logits_lst = [logits_real, logits_real_class, logits_fake, logits_fake_class]

        # gen_loss_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake)*1.0))
        # dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real)*self._alpha))
        # dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake)*(1-self._alpha)))

        gen_loss_gan = tf.reduce_mean((logits_fake)**2)
        dis_loss_real = tf.reduce_mean((logits_real-self._alpha)**2)
        dis_loss_fake = tf.reduce_mean((logits_fake+self._alpha)**2)


        dis_loss_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real_class, labels=self._label_onehot)) \
                        + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake_class, labels=self._label_onehot))
        dis_loss = dis_loss_real + dis_loss_fake + dis_loss_class
        gen_loss_reg = tf.reduce_mean(tf.abs(self._image_ab-gen_out))*100.0

        gen_loss_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake_class, labels=self._label_onehot))

        gen_loss = gen_loss_gan + gen_loss_reg + gen_loss_class
        loss = [gen_loss, gen_loss_gan, gen_loss_reg, gen_loss_class,
                dis_loss, dis_loss_real, dis_loss_fake, dis_loss_class]
        return gen_out, logits_lst, loss


if __name__ == '__main__':
    m = Model()
    m.build_train_graph()
