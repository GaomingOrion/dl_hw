import tensorflow as tf
import numpy as np


class BaseModel:
    def __init__(self, params):
        self.weight_init = params['weight_init']
        self.bias_init = params['bias_init']
        self.reg = params['reg']
        self.optimizer = params['optimizer']
        self.epochs = params['epochs']

    def _fc_layer(self, name, inp, units, dropout=None):
        with tf.variable_scope(name):
            shape = inp.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(inp, [-1, dim])
            if dropout is not None:
                x = tf.nn.dropout(x, keep_prob=dropout, name='dropout')
            x = tf.layers.dense(x, units, kernel_initializer=self.weight_init,
                                bias_initializer=self.bias_init, kernel_regularizer=self.reg)
        return x

    def _graph_kernel(self, inp):
        x = self._fc_layer(name='fc1', inp=inp, units=2000)
        x = tf.nn.relu(x, name='ac1')
        x = self._fc_layer(name='fc2', inp=x, units=1000)
        x = tf.nn.relu(x, name='ac2')
        logits = self._fc_layer(name='fc_final', inp=x, units=10)
        return logits

    def build_graph(self):
        data = tf.placeholder(tf.float32, shape=(None, 3072))
        label = tf.placeholder(tf.int32, shape=(None,))
        label_onehot = tf.one_hot(label, 10, dtype=tf.int32)

        logits = self._graph_kernel(inp=data)

        preds = tf.nn.softmax(logits)

        reg_loss_list = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg_loss_list) != 0:
            loss_reg = tf.add_n(reg_loss_list)
        else:
            loss_reg = tf.constant(0.0)
        loss = tf.losses.softmax_cross_entropy(label_onehot, logits) + loss_reg

        opt = self.optimizer
        train_op = opt.minimize(loss)

        placeholders = {'data': data, 'label': label}
        return placeholders, preds, loss, train_op


    def train(self, data_gnr, batch_size, logpath, dev=None):
        placeholders, preds, loss, train_op = self.build_graph()

        if dev is not None:
            xdev, ydev = dev

        # tensorborad
        tf.summary.scalar('loss', loss)
        writer = tf.summary.FileWriter(logpath, tf.get_default_graph())
        merged = tf.summary.merge_all()

        global_cnt = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(1, self.epochs+1):
                for xtrain, ytrain in data_gnr():
                    global_cnt += 1
                    feed_dict = {placeholders['data']: xtrain,
                                 placeholders['label']: ytrain}
                    loss_value, _, summary = sess.run([loss, train_op, merged], feed_dict=feed_dict)
                    writer.add_summary(summary, global_cnt*batch_size)

                total_true = 0
                if dev:
                    for i in range(100):
                        feed_dict = {placeholders['data']: xdev[100*i:100*(i+1)]}
                        preds_value = sess.run(preds, feed_dict=feed_dict)
                        #print(preds_value, ydev[100*i:100*(i+1)])
                        total_true += np.sum(np.argmax(preds_value, axis=1) == ydev[100*i:100*(i+1)])
                acc = total_true/10000
                print(
                    "e:{},".format(epoch),
                    'loss: {:.3f}'.format(loss_value),
                    'val_acc: {:.4f}'.format(acc)
                )






