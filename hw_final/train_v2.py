import tensorflow as tf
import numpy as np
import os, time
from Unet_v2_4 import Model
from dataset import Dataset
from common import config


def train(prev_model_path=None):
    # prepare dataset
    dataset_train = Dataset('train')
    dataset_test = Dataset('test')

    # define computing graph
    model = Model()
    net_out, logits = model.build_infer_graph()
    probs = tf.nn.softmax(logits)
    loss = model.compute_loss(net_out, logits)
    # set optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer()
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss, global_step=global_step)


    # evaluate on test set
    def evaluate(sess, dataset):
        loss_lst = []
        mse_lst = []
        acc_lst = []
        for image_bw, image_ab, label in dataset.one_epoch_generator():
            image_ab_pred,probs_val, loss_val = sess.run([net_out, probs, loss], feed_dict={
                                    model.place_holders['image_bw']: image_bw,
                                    model.place_holders['image_ab']: image_ab,
                                    model.place_holders['label']: label,
                                    model.place_holders['is_training']: False
                })
            loss_lst.append(loss_val)
            mse_lst.append(np.mean((image_ab-image_ab_pred)**2))
            # print(np.argmax(probs_val, axis=1))
            # print(label)
            acc_lst.append(np.sum(np.argmax(probs_val, axis=1)==label)/len(label))
        return np.mean(loss_lst), np.mean(mse_lst), np.mean(acc_lst)

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
            for batch_idx, (image_bw, image_ab, label) in enumerate(dataset_train.one_epoch_generator()):
                global_cnt += 1
                _, loss_val, summary = sess.run([train_op, loss, merged], feed_dict={
                    model.place_holders['image_bw']: image_bw,
                    model.place_holders['image_ab']: image_ab,
                    model.place_holders['label']: label,
                    model.place_holders['is_training']: True
                })
                summary_writer.add_summary(summary, global_cnt)
                if (batch_idx+1)%config.evaluate_batch_interval == 0:
                    test_loss_val, test_mse, test_acc = evaluate(sess, dataset_test)
                    print("----Epoch-{:n}, progress:{:.2%}, evaluation results:".format(epoch,
                            (batch_idx+1)*config.train_batch_size/config.train_size))
                    print("--Train_loss: {:.4f}".format(loss_val))
                    print("--Test_loss: {:.4f}".format(test_loss_val))
                    print("--Test_mse: {:.4f}".format(test_mse))
                    print("--Test_acc: {:.4f}\n".format(test_acc))
                    summary_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss_val)]),
                        global_cnt)
                    summary_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='test_mse', simple_value=test_mse)]),
                        global_cnt)


            if epoch % config.save_epoch_interval == 0:
                test_loss_val, test_mse, test_acc = evaluate(sess, dataset_test)
                train_loss_val, train_mse, train_acc = evaluate(sess, dataset_train)
                print("----Epoch-{:n} finished, evaluation results:".format(epoch))
                print("--Train_loss: {:.4f}".format(train_loss_val))
                print("--Train_mse: {:.4f}".format(train_mse))
                print("--Train_acc: {:.4f}\n".format(train_acc))
                print("--Test_loss: {:.4f}".format(test_loss_val))
                print("--Test_mse: {:.4f}".format(test_mse))
                print("--Test_acc: {:.4f}\n".format(test_acc))
                model_name = 'UNet_v2-e{:n}-mse{:.6f}.ckpt'.format(epoch, test_mse)
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
    train()
    #train()
