import tensorflow as tf
from tensorflow.contrib import slim
from common import config

class CRNN:
    def __init__(self):
        self._is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        self._inputdata = tf.placeholder(tf.float32, shape=(None, ) + config.input_shape, name='input')
        self._label = tf.sparse_placeholder(tf.int32, shape=(None, config.seq_length), name='label')
        self.place_holders = {'inputdata': self._inputdata,
                              'label': self._label,
                              'is_training': self._is_training
                              }

    def _feature_sequence_extraction(self, inputs, scope):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': self._is_training}):
                net = slim.conv2d(inputs, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.conv2d(net, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.conv2d(net, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.conv2d(net, 512, [3, 3], scope='conv4')
                net = slim.conv2d(net, 512, [8, 1], padding='VALID', scope='conv5')
        return net

    def _map_to_sequence(self, inputdata, name):
        """ Implements the map to sequence part of the network.
        This is used to convert the CNN feature map to the sequence used in the stacked LSTM layers later on.
        Note that this determines the length of the sequences that the LSTM expects
        """
        with tf.variable_scope(name_or_scope=name):
            shape = inputdata.get_shape().as_list()
            assert shape[1] == 1  # H of the feature map must equal to 1
            ret = tf.squeeze(input=inputdata, axis=1, name='squeeze')
        return ret

    def _sequence_label(self, inputdata, name):
        """ Implements the sequence label part of the network

        :param inputdata:
        :param name:
        :return: net_out: time_major logits (time_steps, batch, num_clasees)
                 raw_pred: softmax of logits (batch, time_steps, num_classes)
        """
        with tf.variable_scope(name_or_scope=name):
            # construct stack lstm rcnn layer
            # forward lstm cell
            fw_cell_list = [tf.nn.rnn_cell.LSTMCell(nh, forget_bias=1.0, initializer=tf.orthogonal_initializer()) for
                            nh in [config.rnn_hidden_units] * config.rnn_layer_num]
            # Backward direction cells
            bw_cell_list = [tf.nn.rnn_cell.LSTMCell(nh, forget_bias=1.0, initializer=tf.orthogonal_initializer()) for
                            nh in [config.rnn_hidden_units] * config.rnn_layer_num]

            stack_lstm_layer, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                fw_cell_list, bw_cell_list, inputdata,
                dtype=tf.float32
            )
            stack_lstm_layer = slim.dropout(stack_lstm_layer,
                                            keep_prob=config.rnn_keep_prob,
                                            is_training=self._is_training,
                                            scope='sequence_drop_out')

            shape = tf.shape(stack_lstm_layer)
            rnn_reshaped = tf.reshape(stack_lstm_layer, [shape[0] * shape[1], shape[2]])

            w = tf.get_variable(
                name='w',
                shape=[config.rnn_hidden_units*2, config.class_num],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
                trainable=True
            )

            # Doing the affine projection
            logits = tf.matmul(rnn_reshaped, w, name='logits')

            logits = tf.reshape(logits, [shape[0], shape[1], config.class_num], name='logits_reshape')

            raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')

            # Swap batch and batch axis
            rnn_out = tf.transpose(logits, [1, 0, 2], name='transpose_time_major')  # [width, batch, n_classes]

        return rnn_out, raw_pred

    def build_infer_graph(self):
        # first apply the cnn feature extraction stage
        cnn_out = self._feature_sequence_extraction(
            self._inputdata, 'feature_extraction_module'
        )

        # second apply the map to sequence stage
        sequence = self._map_to_sequence(
            inputdata=cnn_out, name='map_to_sequence_module'
        )

        # third apply the sequence label stage
        net_out, raw_pred = self._sequence_label(
            inputdata=sequence, name='sequence_rnn_module'
        )
        return net_out, raw_pred

    def compute_loss(self, net_out):
        loss = tf.reduce_mean(
            tf.nn.ctc_loss(
                labels=self._label, inputs=net_out,
                sequence_length=tf.to_int32(tf.fill(tf.shape(self._inputdata)[:1], config.seq_length))
            ),
            name='ctc_loss'
        )
        return loss

if __name__ == '__main__':
    a = CRNN()
    net_out, _ = a.build_infer_graph()
    loss = a.compute_loss(net_out)