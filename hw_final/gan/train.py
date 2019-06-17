import tensorflow as tf
import numpy as np
import os, time
from GAN import Model
from dataset import Dataset
from common import config


def train(prev_model_path=None):
    # prepare dataset
    dataset_train = Dataset('train')
    dataset_test = Dataset('test')

    # define computing graph
    model = Model()
    gen_out, dis_loss, gen_loss = model.build_train_graph()
    # set optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        learning_rate = tf.maximum(1e-6, tf.train.exponential_decay(
            learning_rate=0.001,
            global_step=global_step,
            decay_steps=200,
            decay_rate=0.95))
        gen_optimizer = tf.train.AdamOptimizer(learning_rate)
        dis_optimizer = tf.train.AdamOptimizer(learning_rate/10)
        # gen_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        # dis_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        gen_op = gen_optimizer.minimize(
            loss=gen_loss, global_step=global_step)
        dis_op = dis_optimizer.minimize(
            loss=dis_loss, global_step=global_step)


    # evaluate on test set
    def evaluate(sess, dataset):
        mse_lst = []
        for image_bw, image_ab, label in dataset.one_epoch_generator():
            image_ab_pred = sess.run(gen_out, feed_dict={
                                    model.place_holders['image_bw']: image_bw,
                                    model.place_holders['image_ab']: image_ab,
                                    model.place_holders['label']: label,
                                    model.place_holders['is_training']: False
                })
            mse_lst.append(np.mean((image_ab-image_ab_pred)**2))
        return np.mean(mse_lst)

    # Set tf summary
    tboard_save_dir = config.tboard_save_dir
    os.makedirs(tboard_save_dir, exist_ok=True)

    # Set saver configuration
    saver = tf.train.Saver(max_to_keep=100)
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

        for batch_idx, (image_bw, image_ab, label) in enumerate(dataset_train.one_epoch_generator()):
            global_cnt += 1
            _, dis_loss_val = sess.run([dis_op, dis_loss], feed_dict={
                model.place_holders['image_bw']: image_bw,
                model.place_holders['image_ab']: image_ab,
                model.place_holders['label']: label,
                model.place_holders['is_training']: True
            })
            summary_writer.add_summary(
                tf.Summary(value=[tf.Summary.Value(tag='train_dis_loss', simple_value=dis_loss_val)]),
                global_cnt)
            if batch_idx > 50:
                print('warm up finished!')
                break

        while epoch < config.epochs:
            epoch += 1
            for batch_idx, (image_bw, image_ab, label) in enumerate(dataset_train.one_epoch_generator()):
                global_cnt += 1
                _, gen_loss_val= sess.run([gen_op, gen_loss], feed_dict={
                    model.place_holders['image_bw']: image_bw,
                    model.place_holders['image_ab']: image_ab,
                    model.place_holders['label']: label,
                    model.place_holders['is_training']: True
                })
                _, dis_loss_val = sess.run([dis_op, dis_loss], feed_dict={
                    model.place_holders['image_bw']: image_bw,
                    model.place_holders['image_ab']: image_ab,
                    model.place_holders['label']: label,
                    model.place_holders['is_training']: True
                })
                summary_writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag='train_gen_loss', simple_value=gen_loss_val)]),
                    global_cnt)
                summary_writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag='train_dis_loss', simple_value=dis_loss_val)]),
                    global_cnt)
                if (batch_idx+1)%config.evaluate_batch_interval == 0:
                    test_mse = evaluate(sess, dataset_test)
                    print("----Epoch-{:n}, progress:{:.2%}, evaluation results:".format(epoch,
                            (batch_idx+1)*config.train_batch_size/config.train_size))
                    print("--Train_gen_loss: {:.4f}".format(gen_loss_val))
                    print("--Train_dis_loss: {:.4f}".format(dis_loss_val))
                    print("--Test_mse: {:.4f}\n".format(test_mse))
                    summary_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='test_mse', simple_value=test_mse)]),
                        global_cnt)
                    model_name = 'GAN_v1_2-e{:n}-mse{:.6f}.ckpt'.format(epoch, test_mse)
                    model_save_path = os.path.join(model_save_dir, model_name)
                    print('Saving model...')
                    saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
                    print('Saved!')


            if epoch % config.save_epoch_interval == 0:
                test_mse = evaluate(sess, dataset_test)
                print("----Epoch-{:n} finished, evaluation results:".format(epoch))
                print("--Test_mse: {:.4f}\n".format(test_mse))
                model_name = 'GAN_v1_2-e{:n}-mse{:.6f}.ckpt'.format(epoch, test_mse)
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
    #train('./ckpt/UNet_v1_2-e3-mse0.0084.ckpt-3')
    train()
