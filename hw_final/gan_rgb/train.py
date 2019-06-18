import tensorflow as tf
import numpy as np
import os, time
import cv2
from GAN_test import Model
from dataset import Dataset
from common import config


def train(prev_model_path=None):
    # prepare dataset
    dataset_train = Dataset('train')
    dataset_test = Dataset('test')

    # define computing graph
    model = Model()
    gen_out, loss = model.build_train_graph()
    # set optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # get dis, gen variables
    vars = tf.trainable_variables()
    gen_vars = [x for x in vars if x.name.startswith('gen/')]
    dis_vars = [x for x in vars if x.name.startswith('dis/')]

    with tf.control_dependencies(update_ops):
        learning_rate = tf.maximum(1e-6, tf.train.exponential_decay(
            learning_rate=0.01,
            global_step=global_step,
            decay_steps=200,
            decay_rate=0.9))
        gen_optimizer = tf.train.AdamOptimizer(learning_rate)
        dis_optimizer = tf.train.AdamOptimizer(learning_rate/10)
        # gen_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        # dis_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        gen_ops = [gen_optimizer.minimize(
            loss=loss[0], global_step=global_step, var_list=gen_vars)]
        dis_ops = [dis_optimizer.minimize(
            loss=loss[3], global_step=global_step, var_list=dis_vars)]


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

    def show_eval(sess, dataset, path):
        flabel_dict = {0: 'rog', 1: 'uck', 2: 'eer', 3: 'ile', 4: 'ird', 5: 'rse', 6: 'hip', 7: 'cat', 8: 'dog',
                       9: 'ane'}
        for batch_idx, (image_bw, image_ab, label) in enumerate(dataset.one_epoch_generator()):
            image_ab_pred = sess.run(gen_out, feed_dict={
                model.place_holders['image_bw']: image_bw,
                model.place_holders['image_ab']: image_ab,
                model.place_holders['label']: label,
                model.place_holders['is_training']: False
            })

            img_pred = np.uint8(image_ab_pred[0, :, :, :] * 255)
            bwimg = np.uint8(image_bw[0] * 255)

            label1 = flabel_dict[label[0]]
            cv2.imwrite(path + str(batch_idx) + '_' + label1 + '_bw.png', bwimg)
            cv2.imwrite(path + str(batch_idx) + '_' + label1 + '.png', np.uint8(image_ab[0, :, :, :] * 255))
            cv2.imwrite(path + str(batch_idx) + '_' + label1 + '_pred.png', img_pred)

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



        while epoch < config.epochs:
            epoch += 1
            for batch_idx, (image_bw, image_ab, label) in enumerate(dataset_train.one_epoch_generator()):
                #alpha = 1.0 if np.random.random()>0.2 else 0.0
                alpha = 1.0
                global_cnt += 1
                gen_loss_val= sess.run(loss[:3]+gen_ops, feed_dict={
                    model.place_holders['image_bw']: image_bw,
                    model.place_holders['image_ab']: image_ab,
                    model.place_holders['label']: label,
                    model.place_holders['is_training']: True,
                    model.place_holders['alpha']: alpha
                })
                summary_writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag='gen_loss', simple_value=gen_loss_val[0])]),
                    global_cnt)
                summary_writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag='gen_loss_gan', simple_value=gen_loss_val[1])]),
                    global_cnt)
                summary_writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag='gen_loss_reg', simple_value=gen_loss_val[2])]),
                    global_cnt)
                if np.random.random()>0.8:
                    dis_loss_val = sess.run(loss[3:]+dis_ops, feed_dict={
                        model.place_holders['image_bw']: image_bw,
                        model.place_holders['image_ab']: image_ab,
                        model.place_holders['label']: label,
                        model.place_holders['is_training']: True,
                        model.place_holders['alpha']: alpha
                    })
                    summary_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='dis_loss', simple_value=dis_loss_val[0])]),
                        global_cnt)
                    summary_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='dis_loss_real', simple_value=dis_loss_val[1])]),
                        global_cnt)
                    summary_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='dis_loss_fake', simple_value=dis_loss_val[2])]),
                        global_cnt)
                else:
                    dis_loss_val = [0.0]*3
                if (batch_idx+1)%config.evaluate_batch_interval == 0:
                    test_mse = evaluate(sess, dataset_test)
                    print("----Epoch-{:n}, progress:{:.2%}, evaluation results:".format(epoch,
                            (batch_idx+1)*config.train_batch_size/config.train_size))
                    print("--Train_gen_loss: {:.4f}".format(gen_loss_val[0]))
                    print("--Train_dis_loss: {:.4f}".format(dis_loss_val[0]))
                    print("--Test_mse: {:.4f}\n".format(test_mse))
                    summary_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='test_mse', simple_value=test_mse)]),
                        global_cnt)
                    # model_name = 'GAN_v1_2-e{:n}-mse{:.6f}.ckpt'.format(epoch, test_mse)
                    # model_save_path = os.path.join(model_save_dir, model_name)
                    # print('Saving model...')
                    # saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
                    # print('Saved!')


            if epoch % config.save_epoch_interval == 0:
                test_mse = evaluate(sess, dataset_test)
                print("----Epoch-{:n} finished, evaluation results:".format(epoch))
                print("--Test_mse: {:.4f}\n".format(test_mse))
                model_name = 'GAN_v1_2-e{:n}-mse{:.6f}.ckpt'.format(epoch, test_mse)
                model_save_path = os.path.join(model_save_dir, model_name)
                print('Saving model...')
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
                print('Saved!')

                show_path = './output-e%i/'%epoch
                if not os.path.exists(show_path):
                    os.mkdir(show_path)
                show_eval(sess, dataset_test, show_path)



if __name__ == '__main__':
    train('./tf_ckpt/GAN_v1_2-e4-mse0.005613.ckpt-4')
    #train()
