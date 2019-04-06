#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:47:32 2019

@author: zhanghuangzhao
"""

import tensorflow as tf
import numpy


class CNN(object):

    def __init__(self, dtype="fp32", scope_name="", is_inference=False,
                 lr=[1e-5, 2e-2], lr_decay=2000, grad_clip=5):

        # Data type
        if dtype == "fp16":
            self.dtype = numpy.float16
        elif dtype == "fp32":
            self.dtype = numpy.float32
        elif dtype == "fp64":
            self.dtype = numpy.float64
        else:
            assert False, "Invalid data type (%s). Use \"fp16\", \"fp32\" or \"fp64\" only" % dtype

        # Hyper-parameters
        self.K = 24  # Conv-1 depth
        self.stride1 = 2  # Conv-1 stride
        self.L = 48  # Conv-2 depth
        self.stride2 = 2  # Conv-2 stride
        self.N = 200  # FC width
        self.min_lr = lr[0]  # Minimum learning rate
        self.max_lr = lr[1]  # Maximum learning rate
        self.decay_step = lr_decay  # Learning rate exponentional decay
        self.grad_clipping = grad_clip  # Gradient clipping by absolute value

        # Add placeholders
        self.add_placeholders()
        # Add variables
        self.add_variables()

        # Build graph
        (self.Ylogits, \
         self.Y, \
         self.update_ema) = self.build_graph(self.X)
        self.trainables = [x for x in tf.global_variables() \
                             if x.name.startswith(scope_name)]

        if not is_inference:
            (self.loss, \
             self.accuracy, \
             self.train_step) = self.build_training_graph()
        else:
            (self.loss, \
             self.accuracy, \
             self.train_step) = self.build_inference_graph()

        self.saver = tf.train.Saver(var_list=self.trainables,
                                      max_to_keep=1)

        # Gradient on input image
        self.grad = tf.gradients(self.loss, self.X)

    def add_placeholders(self):

        self.X = tf.placeholder(self.dtype, [None, 28, 28, 1])
        self.Y_ = tf.placeholder(self.dtype, [None, 10])
        self.tst = tf.placeholder(tf.bool)  # test flag for batch norm
        self.iter = tf.placeholder(tf.int32)
        self.pkeep = tf.placeholder(self.dtype)  # dropout probability
        self.pkeep_conv = tf.placeholder(self.dtype)

    def add_variables(self):

        # Conv-1 weights
        self.W1 = tf.Variable(tf.truncated_normal([6, 6, 1, self.K],
                                                    stddev=0.1, dtype=self.dtype))
        self.B1 = tf.Variable(tf.constant(0.1, self.dtype, [self.K]))
        # Conv-2 weights
        self.W2 = tf.Variable(tf.truncated_normal([5, 5, self.K, self.L],
                                                    stddev=0.1, dtype=self.dtype))
        self.B2 = tf.Variable(tf.constant(0.1, self.dtype, [self.L]))
        # FC weights
        self.W4 = tf.Variable(tf.truncated_normal([7 * 7 * self.L, self.N],
                                                    stddev=0.1, dtype=self.dtype))
        self.B4 = tf.Variable(tf.constant(0.1, self.dtype, [self.N]))
        # Softmax weights
        self.W5 = tf.Variable(tf.truncated_normal([self.N, 10],
                                                    stddev=0.1, dtype=self.dtype))
        self.B5 = tf.Variable(tf.constant(0.1, self.dtype, [10]))

    def build_graph(self, X):

        # output shape is 14x14x24
        Y1l = tf.nn.conv2d(X, self.W1,
                           strides=[1, self.stride1, self.stride1, 1],
                           padding='SAME')
        Y1bn, update_ema1 = self.batchnorm(Y1l, self.tst, self.iter,
                                             self.B1, convolutional=True)
        Y1r = tf.nn.relu(Y1bn)
        Y1 = tf.nn.dropout(Y1r, self.pkeep_conv,
                           self.compatible_convolutional_noise_shape(Y1r))
        # output shape is 7x7x48
        Y2l = tf.nn.conv2d(Y1, self.W2,
                           strides=[1, self.stride2, self.stride2, 1],
                           padding='SAME')
        Y2bn, update_ema2 = self.batchnorm(Y2l, self.tst, self.iter,
                                             self.B2, convolutional=True)
        Y2r = tf.nn.relu(Y2bn)
        Y2 = tf.nn.dropout(Y2r, self.pkeep_conv,
                           self.compatible_convolutional_noise_shape(Y2r))
        YY = tf.reshape(Y2, shape=[-1, 7 * 7 * self.L])
        Y4l = tf.matmul(YY, self.W4)
        Y4bn, update_ema4 = self.batchnorm(Y4l, self.tst,
                                             self.iter, self.B4)
        Y4r = tf.nn.relu(Y4bn)
        Y4 = tf.nn.dropout(Y4r, self.pkeep)
        Ylogits = tf.matmul(Y4, self.W5) + self.B5
        Y = tf.nn.softmax(Ylogits)
        update_ema = tf.group(update_ema1, update_ema2,
                               update_ema4)

        return Ylogits, Y, update_ema

    def build_training_graph(self):

        loss_ = tf.nn.softmax_cross_entropy_with_logits(logits=self.Ylogits,
                                                        labels=self.Y_)
        loss = tf.reduce_mean(loss_)

        correct_prediction = tf.equal(tf.argmax(self.Y, 1),
                                      tf.argmax(self.Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, self.dtype))
        lr = self.min_lr + tf.train.exponential_decay(self.max_lr,
                                                        self.iter,
                                                        self.decay_step,
                                                        1 / numpy.e)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        grads_and_vars = opt.compute_gradients(loss=loss,
                                               var_list=self.trainables)
        grads_and_vars_ = []
        for g, v in grads_and_vars:
            grads_and_vars_.append((tf.clip_by_value(g,
                                                     -self.grad_clipping,
                                                     self.grad_clipping), v))
        train_step = opt.apply_gradients(grads_and_vars_)
        return loss, accuracy, train_step

    def build_inference_graph(self):

        loss_ = tf.nn.softmax_cross_entropy_with_logits(logits=self.Ylogits,
                                                        labels=self.Y_)
        loss = tf.reduce_mean(loss_)

        correct_prediction = tf.equal(tf.argmax(self.Y, 1),
                                      tf.argmax(self.Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, self.dtype))
        return loss, accuracy, None

    def batchnorm(self, Ylogits, is_test, iteration, offset, convolutional=False):

        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_averages

    def no_batchnorm(self, Ylogits, is_test, iteration, offset, convolutional=False):

        return Ylogits, tf.no_op()

    def compatible_convolutional_noise_shape(self, Y):

        noiseshape = tf.shape(Y)
        noiseshape = noiseshape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])
        return noiseshape

    def train_op(self, sess, x, y, iter_, pkeep=1.0, pkeep_conv=1.0):

        (_, acc, loss) = sess.run([self.train_step, self.accuracy, self.loss],
                                  feed_dict={self.X: x,
                                             self.Y_: y,
                                             self.iter: iter_,
                                             self.tst: False,
                                             self.pkeep: pkeep,
                                             self.pkeep_conv: pkeep_conv})
        (_) = sess.run([self.update_ema], feed_dict={self.X: x,
                                                       self.Y_: y,
                                                       self.iter: iter_,
                                                       self.tst: False,
                                                       self.pkeep: 1.0,
                                                       self.pkeep_conv: 1.0})
        return acc, loss

    def eval_op(self, sess, x, y):

        (acc, loss) = sess.run([self.accuracy, self.loss],
                               feed_dict={self.X: x,
                                          self.Y_: y,
                                          self.iter: 0,
                                          self.tst: True,
                                          self.pkeep: 1.0,
                                          self.pkeep_conv: 1.0})
        return acc, loss

    def infer_op(self, sess, x):

        (y) = sess.run([self.Y],
                       feed_dict={self.X: x,
                                  self.iter: 0,
                                  self.tst: True,
                                  self.pkeep: 1.0,
                                  self.pkeep_conv: 1.0})
        return y

    def grad_op(self, sess, x, y):

        (grad) = sess.run([self.grad],
                          feed_dict={self.X: x,
                                     self.Y_: y,
                                     self.iter: 0,
                                     self.tst: True,
                                     self.pkeep: 1.0,
                                     self.pkeep_conv: 1.0})
        return grad

    def save(self, sess, path):

        self.saver.save(sess, path)

    def restore(self, sess, path):

        self.saver.restore(sess, path)

    def get_trainables(self):

        return self.trainables


if __name__ == "__main__":
    from fmnist_dataset import Fashion_MNIST

    with tf.variable_scope("fmnist_cnn") as vs:
        m = CNN(scope_name="fmnist_cnn", dtype="fp64")
    d = Fashion_MNIST()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    x, y = d.train.next_batch(1)
    print(m.eval_op(sess, x, y))
    print(m.infer_op(sess, x))
    print(m.train_op(sess, x, y, 0, 0.9, 1.0))
    print(m.eval_op(sess, x, y))
    print(m.infer_op(sess, x))
