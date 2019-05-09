import tensorflow as tf
import numpy as np
import os
from model import Model
from dataset import Dataset
from common import config

def train(prev_model_path=None):
    # prepare dataset
    dataset = Dataset()
    config.input_vocab_size = dataset.encoder_vocab_size
    config.label_vocab_size = dataset.decoder_vocab_size
    config.train_size = dataset.train_size
    config.test_size = dataset.test_size

    # define computing graph
    model = Model()
    net_out, preds, loss = model.logits, model.preds, model.mean_loss
    # set optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        #optimizer = tf.train.AdamOptimizer()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
        train_op = optimizer.minimize(
            loss=loss, global_step=global_step)

    # # decoder
    # decoded, _ = tf.nn.ctc_beam_search_decoder(net_out,
    #                     sequence_length=tf.to_int32(tf.fill(tf.shape(model._inputdata)[:1], config.seq_length)),
    #                     beam_width=5,
    #                     merge_repeated=False, top_paths=1)
    # decoded = decoded[0]
    # decoded_paths = tf.sparse_tensor_to_dense(decoded, default_value=config.class_num-1)


    # evaluate on test set
    def evaluate(sess):
        loss_lst = []
        for en_input_batch, de_input_batch, de_target_batch in dataset.one_epoch_generator_for_test():
            loss_val = sess.run(loss, feed_dict={
                model.x: en_input_batch,
                model.y: de_target_batch,
                model.de_inp: de_input_batch,
                model.is_training: False
                })
            loss_lst.append(loss_val)
        return np.mean(loss_lst)

    # Set tf summary
    tboard_save_dir = config.tboard_save_dir
    os.makedirs(tboard_save_dir, exist_ok=True)
    tf.summary.scalar(name='train_loss', tensor=loss)
    merged = tf.summary.merge_all()

    # Set saver configuration
    saver = tf.train.Saver(max_to_keep=10)
    model_save_dir = config.model_save_dir
    os.makedirs(model_save_dir, exist_ok=True)

    # Set sess configuration
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(tboard_save_dir)
    summary_writer.add_graph(sess.graph)

    # training
    global_cnt = 0
    with sess.as_default():
        if prev_model_path is None:
            sess.run(tf.global_variables_initializer())
            print('Initialiation finished!')
            epoch = 0
        else:
            print('Restore model from {:s}'.format(prev_model_path))
            saver.restore(sess=sess, save_path=prev_model_path)
            epoch = 0
        while epoch < config.epochs:
            epoch += 1
            for batch_idx, (en_input_batch, de_input_batch, de_target_batch) in enumerate(dataset.one_epoch_generator_for_train()):
                global_cnt += 1
                loss_val, _, summary = sess.run([loss, train_op, merged], feed_dict={
                    model.x: en_input_batch,
                    model.y: de_target_batch,
                    model.de_inp: de_input_batch,
                    model.is_training: True
                })
                summary_writer.add_summary(summary, global_cnt)
                if (batch_idx+1)%config.evaluate_batch_interval == 0:
                    test_loss_val = evaluate(sess)
                    print("----Epoch-{:n}, progress:{:.2%}, evaluation results:".format(epoch,
                            (batch_idx+1)*config.train_batch_size/config.train_size))
                    print("--Train_loss: {:.4f}".format(loss_val))
                    print("--Test_loss: {:.4f}".format(test_loss_val))
                    summary_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss_val)]),
                        global_cnt)


            if epoch % config.save_epoch_interval == 0:
                test_loss_val = evaluate(sess)
                print("----Epoch-{:n} finished, evaluation results:".format(epoch))
                print("--Test_loss: {:.4f}".format(test_loss_val))
                model_name = 'model-adam-e%i-loss_%.4f.ckpt'%(epoch, test_loss_val)
                model_save_path = os.path.join(model_save_dir, model_name)
                print('Saving model...')
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
                print('Saved!')

if __name__ == '__main__':
    train('.\\tf_ckpt\\model-adam-e5-loss_4.1589.ckpt-5')
    #train()
