import tensorflow as tf
from modules import embedding, multihead_attention, feedforward, label_smoothing
from common import config

class Model():
    def __init__(self):
        tf.reset_default_graph()
        self.hidden_units = config.hidden_units
        self.input_vocab_size = config.input_vocab_size
        self.label_vocab_size = config.label_vocab_size
        self.max_length = config.max_length
        self.num_heads = config.num_heads
        self.num_blocks = config.num_blocks
        self.dropout_rate = config.dropout_rate

        # input placeholder
        self.x = tf.placeholder(tf.int32, shape=(None, None))
        self.y = tf.placeholder(tf.int32, shape=(None, None))
        self.de_inp = tf.placeholder(tf.int32, shape=(None, None))
        self.is_training = tf.placeholder(tf.bool, shape=())


        # Encoder
        with tf.variable_scope("encoder"):
            # embedding
            self.en_emb = embedding(self.x, vocab_size=self.input_vocab_size, num_units=self.hidden_units, scale=True,
                                    scope="enc_embed")
            self.enc = self.en_emb + embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                vocab_size=self.max_length, num_units=self.hidden_units, zero_pad=False, scale=False, scope="enc_pe")
            ## Dropout
            self.enc = tf.layers.dropout(self.enc,
                                         rate=self.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))

            ## Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(key_emb=self.en_emb,
                                                   que_emb=self.en_emb,
                                                   queries=self.enc,
                                                   keys=self.enc,
                                                   is_training=self.is_training,
                                                   num_units=self.hidden_units,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout_rate,
                                                   causality=False)

            ### Feed Forward
            self.enc = feedforward(self.enc, num_units=[4 * self.hidden_units, self.hidden_units])

        # Decoder
        with tf.variable_scope("decoder"):
            # embedding
            self.de_emb = embedding(self.de_inp, vocab_size=self.label_vocab_size, num_units=self.hidden_units,
                                    scale=True, scope="dec_embed")
            self.dec = self.de_emb + embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.de_inp)[1]), 0), [tf.shape(self.de_inp)[0], 1]),
                vocab_size=self.max_length, num_units=self.hidden_units, zero_pad=False, scale=False, scope="dec_pe")
            ## Dropout
            self.dec = tf.layers.dropout(self.dec,
                                         rate=self.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))

            ## Multihead Attention ( self-attention)
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.dec = multihead_attention(key_emb=self.de_emb,
                                                   que_emb=self.de_emb,
                                                   queries=self.dec,
                                                   keys=self.dec,
                                                   is_training=self.is_training,
                                                   num_units=self.hidden_units,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout_rate,
                                                   causality=True,
                                                   scope='self_attention')

            ## Multihead Attention (vanilla attention)
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.dec = multihead_attention(key_emb=self.en_emb,
                                                   que_emb=self.de_emb,
                                                   queries=self.dec,
                                                   keys=self.enc,
                                                   is_training=self.is_training,
                                                   num_units=self.hidden_units,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout_rate,
                                                   causality=True,
                                                   scope='vanilla_attention')

                    ### Feed Forward
            self.outputs = feedforward(self.dec, num_units=[4 * self.hidden_units, self.hidden_units])

        # Final linear projection
        self.logits = tf.layers.dense(self.outputs, self.label_vocab_size)
        self.preds = tf.to_int32(tf.argmax(tf.nn.softmax(self.logits), axis=-1))

        # Loss
        self.istarget = tf.to_float(tf.not_equal(self.y, 0))
        self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=self.label_vocab_size))
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
        self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))


if __name__ == '__main__':
    model = Model()