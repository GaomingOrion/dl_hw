import tensorflow as tf
import numpy as np
import os, time
from crnn_SVNH import CRNN
from dataset import Dataset
from common import config


def train(prev_model_path=None):
    # prepare dataset
    dataset_train = Dataset('train')
    dataset_test = Dataset('test')

    # define computing graph
    model = CRNN()
    net_out, raw_pred = model.build_infer_graph()
    loss = model.compute_loss(net_out)
    # set optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        #optimizer = tf.train.AdamOptimizer()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)
        train_op = optimizer.minimize(
            loss=loss, global_step=global_step)
    # decoder
    decoded, _ = tf.nn.ctc_beam_search_decoder(net_out,
                        sequence_length=tf.to_int32(tf.fill(tf.shape(model._inputdata)[:1], config.seq_length)),
                        beam_width=5,
                        merge_repeated=False, top_paths=1)
    decoded = decoded[0]
    decoded_paths = tf.sparse_tensor_to_dense(decoded, default_value=config.class_num-1)


    # evaluate on test set
    def evaluate(sess, dataset):
        loss_lst = []
        label_pred = []
        label_true = []
        for inputdata, sparse_label, raw_label in dataset.one_epoch_generator():
            decoded_paths_val, loss_val = sess.run([decoded_paths, loss], feed_dict={
                                    model.place_holders['inputdata']: inputdata,
                                    model.place_holders['label']: sparse_label,
                                    model.place_holders['is_training']: False
                })
            for x in decoded_paths_val:
                label_pred.append([idx for idx in x if idx != config.class_num-1])
            for x in raw_label:
                label_true.append(x)
            loss_lst.append(loss_val)
        acc = cal_acc(label_pred, label_true)
        return np.mean(loss_lst), acc

    # Set tf summary
    tboard_save_dir = config.tboard_save_dir
    os.makedirs(tboard_save_dir, exist_ok=True)
    tf.summary.scalar(name='train_loss', tensor=loss)
    merged = tf.summary.merge_all()

    # Set saver configuration
    saver = tf.train.Saver()
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
            for batch_idx, (inputdata, sparse_label, raw_label) in enumerate(dataset_train.one_epoch_generator()):
                global_cnt += 1
                loss_val, _, summary = sess.run([loss, train_op, merged], feed_dict={
                                    model.place_holders['inputdata']: inputdata,
                                    model.place_holders['label']: sparse_label,
                                    model.place_holders['is_training']: True
                })
                summary_writer.add_summary(summary, global_cnt)
                if (batch_idx+1)%config.evaluate_batch_interval == 0:
                    test_loss_val, test_acc = evaluate(sess, dataset_test)
                    print("----Epoch-{:n}, progress:{:.2%}, evaluation results:".format(epoch,
                            (batch_idx+1)*config.train_batch_size/config.train_size))
                    print("--Train_loss: {:.4f}".format(loss_val))
                    print("--Test_loss: {:.4f}".format(test_loss_val))
                    print("--Test_accuarcy: {:.4f}\n".format(test_acc))
                    summary_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss_val)]),
                        global_cnt)
                    summary_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='test_acc', simple_value=test_acc)]),
                        global_cnt)


            if epoch % config.save_epoch_interval == 0:
                test_loss_val, test_acc = evaluate(sess, dataset_test)
                train_loss_val, train_acc = evaluate(sess, dataset_train)
                print("----Epoch-{:n} finished, evaluation results:".format(epoch))
                print("--Train_loss: {:.4f}".format(train_loss_val))
                print("--Train_accuarcy: {:.4f}".format(train_acc))
                print("--Test_loss: {:.4f}".format(test_loss_val))
                print("--Test_accuarcy: {:.4f}\n".format(test_acc))
                model_name = 'CRNN-e{:n}-acc{:.1f}.ckpt'.format(epoch, 100*test_acc)
                model_save_path = os.path.join(model_save_dir, model_name)
                print('Saving model...')
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
                print('Saved!')



def cal_acc(label_pred, label_true):
    assert len(label_pred) == len(label_true)
    cnt = 0
    for i in range(len(label_pred)):
        if label_pred[i] == label_true[i]:
            cnt += 1
    return cnt/len(label_pred)

if __name__ == '__main__':
    train('.\\tf_ckpt\\CRNN-e7-acc59.7.ckpt-7')
    #train()
