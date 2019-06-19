import tensorflow as tf
import numpy as np
import os, time
import cv2
from GAN import Model
from dataset import Dataset
from common import config

def train_gen(epochs=10, prev_model_path=None):
    # prepare dataset
    dataset_train = Dataset('train')
    dataset_test = Dataset('test')

    # define computing graph
    model = Model()
    gen_out = model.build_infer_graph()
    loss = tf.reduce_mean((gen_out-model.place_holders['image_ab'])**2)

    # set optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # get dis, gen variables
    vars = tf.trainable_variables()
    gen_vars = [x for x in vars if x.name.startswith('gen/')]

    all_vars = tf.global_variables()
    all_dis_vars = [x for x in all_vars if x.name.startswith('dis/')]
    all_gen_vars = [x for x in all_vars if x.name.startswith('gen/')]

    with tf.control_dependencies(update_ops):
        learning_rate = tf.maximum(1e-6, tf.train.exponential_decay(
            learning_rate=0.01,
            global_step=global_step,
            decay_steps=200,
            decay_rate=0.9))
        gen_optimizer = tf.train.AdamOptimizer(learning_rate)
        # gen_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        gen_op = gen_optimizer.minimize(
            loss=loss, global_step=global_step, var_list=gen_vars)

    # Set tf summary
    tboard_save_dir = config.tboard_save_dir
    os.makedirs(tboard_save_dir, exist_ok=True)

    # Set saver configuration
    saver = tf.train.Saver(max_to_keep=100)
    model_save_dir = config.model_save_dir
    os.makedirs(model_save_dir, exist_ok=True)

    # Set sess configuration
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(tboard_save_dir+'/gen')

    # evaluate on test set
    def evaluate_gen(sess, dataset):
        mse_lst = []
        for image_bw, image_ab, label in dataset.one_epoch_generator():
            image_ab_pred = sess.run(gen_out, feed_dict={
                model.place_holders['image_bw']: image_bw,
                model.place_holders['image_ab']: image_ab,
                model.place_holders['label']: label,
                model.place_holders['is_training']: False
            })
            mse_lst.append(np.mean((image_ab - image_ab_pred) ** 2))
        return np.mean(mse_lst)

    # training
    global_cnt = 0
    with sess.as_default():
        if prev_model_path is None:
            sess.run(tf.global_variables_initializer())
            print('Initialiation finished!')
            epoch = 0
        else:
            print('Restore model from {:s}'.format(prev_model_path))
            try:
                saver.restore(sess=sess, save_path=prev_model_path)
            except:
                tf.train.Saver(all_dis_vars+[global_step]).restore(sess=sess, save_path=prev_model_path)
                sess.run(tf.variables_initializer(all_gen_vars))
            epoch = 0

        while epoch < epochs:
            epoch += 1
            for batch_idx, (image_bw, image_ab, label) in enumerate(dataset_train.one_epoch_generator()):
                global_cnt += 1
                gen_loss_val, _ = sess.run([loss, gen_op], feed_dict={
                    model.place_holders['image_bw']: image_bw,
                    model.place_holders['image_ab']: image_ab,
                    model.place_holders['label']: label,
                    model.place_holders['is_training']: True,
                })
                summary_writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag='gen_loss', simple_value=gen_loss_val)]),
                    global_cnt)

                if (batch_idx + 1) % config.evaluate_batch_interval == 0:
                    test_mse = evaluate_gen(sess, dataset_test)
                    print("----Epoch-{:n}, progress:{:.2%}, evaluation results:".format(epoch,
                                (batch_idx + 1) * config.train_batch_size / config.train_size))
                    print("--Train_gen_loss: {:.4f}".format(gen_loss_val))
                    print("--Test_mse: {:.4f}\n".format(test_mse))
                    summary_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='test_mse', simple_value=test_mse)]),
                        global_cnt)


            if epoch % config.save_epoch_interval == 0:
                test_mse = evaluate_gen(sess, dataset_test)
                print("----Epoch-{:n} finished, evaluation results:".format(epoch))
                print("--Train_gen_loss: {:.4f}".format(gen_loss_val))
                print("--Test_mse: {:.4f}\n".format(test_mse))
                model_name = 'GAN-e{:n}-mse{:.6f}.ckpt'.format(epoch, test_mse)
                model_save_path = os.path.join(model_save_dir, model_name)
                print('Saving model...')
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
                print('Saved!')


def train_dis(epochs=10, prev_model_path=None):
    # prepare dataset
    dataset_train = Dataset('train')
    dataset_test = Dataset('test')

    # define computing graph
    model = Model()
    _, logits_lst, loss_lst = model.build_train_graph()
    loss = loss_lst[4]

    # set optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # get dis, gen variables
    vars = tf.trainable_variables()
    dis_vars = [x for x in vars if x.name.startswith('dis/')]

    all_vars = tf.all_variables()
    all_dis_vars = [x for x in all_vars if x.name.startswith('dis/')]
    all_gen_vars = [x for x in all_vars if x.name.startswith('gen/')]
    with tf.control_dependencies(update_ops):
        learning_rate = tf.maximum(1e-6, tf.train.exponential_decay(
            learning_rate=0.01,
            global_step=global_step,
            decay_steps=200,
            decay_rate=0.9))
        gen_optimizer = tf.train.AdamOptimizer(learning_rate)
        # gen_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # dis_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        gen_op = gen_optimizer.minimize(
            loss=loss, global_step=global_step, var_list=dis_vars)


    # Set tf summary
    tboard_save_dir = config.tboard_save_dir
    os.makedirs(tboard_save_dir, exist_ok=True)

    # Set saver configuration
    saver = tf.train.Saver(max_to_keep=100)
    model_save_dir = config.model_save_dir
    os.makedirs(model_save_dir, exist_ok=True)

    # Set sess configuration
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(tboard_save_dir + '/dis')
    summary_writer.add_graph(sess.graph)

    # evaluate on test set
    def evaluate_dis(sess, dataset):
        dis_acc_real_lst, dis_acc_fake_lst, class_acc_real_lst, class_acc_fake_lst = [], [], [], []
        for image_bw, image_ab, label in dataset.one_epoch_generator():
            logits_val_lst = sess.run(logits_lst, feed_dict={
                model.place_holders['image_bw']: image_bw,
                model.place_holders['image_ab']: image_ab,
                model.place_holders['label']: label,
                model.place_holders['is_training']: False,
                model.place_holders['alpha']: 1.0
            })
            dis_acc_real_lst.append(np.mean(logits_val_lst[0] > 0))
            class_acc_real_lst.append(np.mean(np.argmax(logits_val_lst[1], axis=1)==label))
            dis_acc_fake_lst.append(np.mean(logits_val_lst[2] < 0))
            class_acc_fake_lst.append(np.mean(np.argmax(logits_val_lst[3], axis=1)==label))
        return np.mean(dis_acc_real_lst), np.mean(dis_acc_fake_lst), np.mean(class_acc_real_lst), np.mean(class_acc_fake_lst)

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

        while epoch < epochs:
            epoch += 1
            for batch_idx, (image_bw, image_ab, label) in enumerate(dataset_train.one_epoch_generator()):
                global_cnt += 1
                dis_loss_val, _ = sess.run([loss, gen_op], feed_dict={
                    model.place_holders['image_bw']: image_bw,
                    model.place_holders['image_ab']: image_ab,
                    model.place_holders['label']: label,
                    model.place_holders['is_training']: True,
                    model.place_holders['alpha']: 1.0
                })
                summary_writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag='gen_loss', simple_value=dis_loss_val)]),
                    global_cnt)

                if (batch_idx + 1) % config.evaluate_batch_interval == 0:
                    dis_acc_real, dis_acc_fake, class_acc_real, class_acc_fake = evaluate_dis(sess, dataset_test)
                    print("----Epoch-{:n}, progress:{:.2%}, evaluation results:".format(epoch,
                            (batch_idx+1)*config.train_batch_size/config.train_size))
                    print("--Train_dis_loss: {:.4f}".format(dis_loss_val))
                    print("--Test_dis_accuracy_of_real_image: {:.4f}".format(dis_acc_real))
                    print("--Test_dis_accuracy_of_fake_image: {:.4f}".format(dis_acc_fake))
                    print("--Test_classify_accuracy_of_real_image: {:.4f}".format(class_acc_real))
                    print("--Test_classify_accuracy_of_fake_image: {:.4f}\n".format(class_acc_fake))

            if epoch % config.save_epoch_interval == 0:
                dis_acc_real, dis_acc_fake, class_acc_real, class_acc_fake = evaluate_dis(sess, dataset_test)
                print("----Epoch-{:n} finished, evaluation results:".format(epoch))
                print("--Train_dis_loss: {:.4f}".format(dis_loss_val))
                print("--Test_dis_accuracy_of_real_image: {:.4f}".format(dis_acc_real))
                print("--Test_dis_accuracy_of_fake_image: {:.4f}".format(dis_acc_fake))
                print("--Test_classify_accuracy_of_real_image: {:.4f}".format(class_acc_real))
                print("--Test_classify_accuracy_of_fake_image: {:.4f}\n".format(class_acc_fake))
                model_name = 'GAN-dis-e{:n}.ckpt'.format(epoch)
                model_save_path = os.path.join(model_save_dir, model_name)
                print('Saving model...')
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
                print('Saved!')


def train(epochs=20, cycle=5, prev_model_path=None, lr=0.001, prob=0.05):
    # prepare dataset
    dataset_train = Dataset('train')
    dataset_test = Dataset('test')

    # define computing graph
    model = Model()
    gen_out, logits_lst, loss = model.build_train_graph()
    # set optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # get dis, gen variables
    vars = tf.trainable_variables()
    gen_vars = [x for x in vars if x.name.startswith('gen/')]
    dis_vars = [x for x in vars if x.name.startswith('dis/')]


    with tf.control_dependencies(update_ops):
        learning_rate = tf.maximum(1e-6, tf.train.exponential_decay(
            learning_rate=lr,
            global_step=global_step,
            decay_steps=200,
            decay_rate=0.99))
        gen_optimizer = tf.train.AdamOptimizer(learning_rate)
        dis_optimizer = tf.train.AdamOptimizer(learning_rate/10)
        # gen_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        # dis_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        gen_ops = [gen_optimizer.minimize(
            loss=loss[0], global_step=global_step, var_list=gen_vars)]
        dis_ops = [dis_optimizer.minimize(
            loss=loss[4], global_step=global_step, var_list=dis_vars)]

    # evaluate on test set
    def evaluate_gen(sess, dataset):
        mse_lst = []
        for image_bw, image_ab, label in dataset.one_epoch_generator():
            image_ab_pred = sess.run(gen_out, feed_dict={
                model.place_holders['image_bw']: image_bw,
                model.place_holders['image_ab']: image_ab,
                model.place_holders['label']: label,
                model.place_holders['is_training']: False
            })
            mse_lst.append(np.mean((image_ab - image_ab_pred) ** 2))
        return np.mean(mse_lst)

    # evaluate on test set
    def evaluate_dis(sess, dataset):
        dis_acc_real_lst, dis_acc_fake_lst, class_acc_real_lst, class_acc_fake_lst = [], [], [], []
        #loss_lst = []
        for image_bw, image_ab, label in dataset.one_epoch_generator():
            logits_val_lst = sess.run(logits_lst, feed_dict={
                model.place_holders['image_bw']: image_bw,
                model.place_holders['image_ab']: image_ab,
                model.place_holders['label']: label,
                model.place_holders['is_training']: False,
                model.place_holders['alpha']: 1.0
            })
            dis_acc_real_lst.append(np.mean(logits_val_lst[0] > 0))
            class_acc_real_lst.append(np.mean(np.argmax(logits_val_lst[1], axis=1)==label))
            dis_acc_fake_lst.append(np.mean(logits_val_lst[2] < 0))
            class_acc_fake_lst.append(np.mean(np.argmax(logits_val_lst[3], axis=1)==label))
           # loss_lst.append(logits_val_lst[-1])
        return np.mean(dis_acc_real_lst), np.mean(dis_acc_fake_lst), np.mean(class_acc_real_lst), np.mean(class_acc_fake_lst)


    def show_eval(sess, dataset, output_path):
        flabel_dict = {0: 'rog', 1: 'uck', 2: 'eer', 3: 'ile', 4: 'ird', 5: 'rse', 6: 'hip', 7: 'cat', 8: 'dog',
                       9: 'ane'}
        for batch_idx, (image_bw, image_ab, label) in enumerate(dataset.one_epoch_generator()):
            image_ab_pred = sess.run(gen_out, feed_dict={
                model.place_holders['image_bw']: image_bw,
                model.place_holders['image_ab']: image_ab,
                model.place_holders['label']: label,
                model.place_holders['is_training']: False
            })
            imgs = np.concatenate([image_bw, image_ab], axis=3)
            imgs_pred = np.concatenate([image_bw, image_ab_pred], axis=3)
            img = np.uint8(imgs[0, :, :, :] * 255)
            img_pred = np.uint8(imgs_pred[0, :, :, :] * 255)
            bgrimg = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
            bgrimg_pred = cv2.cvtColor(img_pred, cv2.COLOR_LAB2BGR)
            bwimg = np.uint8(image_bw[0] * 255)

            label1 = flabel_dict[label[0]]
            cv2.imwrite(output_path + str(batch_idx) + '_' + label1 + '_bw.png', bwimg)
            cv2.imwrite(output_path + str(batch_idx) + '_' + label1 + '.png', bgrimg)
            cv2.imwrite(output_path + str(batch_idx) + '_' + label1 + '_pred.png', bgrimg_pred)

    # Set tf summary
    tboard_save_dir = config.tboard_save_dir
    os.makedirs(tboard_save_dir, exist_ok=True)

    # Set saver configuration
    saver = tf.train.Saver(max_to_keep=100)
    model_save_dir = config.model_save_dir
    os.makedirs(model_save_dir, exist_ok=True)

    # Set sess configuration
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(tboard_save_dir+'/adv')
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

        gen_loss_val, dis_loss_val = [0] * 4, [0] * 4
        while epoch < epochs:
            epoch += 1
            for batch_idx, (image_bw, image_ab, label) in enumerate(dataset_train.one_epoch_generator()):
                alpha = 0.8 if np.random.random()>prob else 0.2
                #alpha = 1.0
                # print(np.mean(sess.run(logits_lst[2], feed_dict={
                #         model.place_holders['image_bw']: image_bw,
                #         model.place_holders['image_ab']: image_ab,
                #         model.place_holders['label']: label,
                #         model.place_holders['is_training']: True,
                #         model.place_holders['alpha']: alpha
                #     })))
                global_cnt += 1
                if epoch % (2*cycle) <=cycle-1:
                    gen_loss_val= sess.run(loss[:4]+gen_ops, feed_dict={
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
                    summary_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='gen_loss_class', simple_value=gen_loss_val[3])]),
                        global_cnt)
                else:
                    dis_loss_val = sess.run(loss[4:]+dis_ops, feed_dict={
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
                    summary_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='dis_loss_class', simple_value=dis_loss_val[3])]),
                        global_cnt)

                if (batch_idx+1)%config.evaluate_batch_interval == 0:
                    test_mse = evaluate_gen(sess, dataset_test)
                    print("----Epoch-{:n}, progress:{:.2%}, evaluation results:".format(epoch,
                            (batch_idx+1)*config.train_batch_size/config.train_size))
                    print("--Train_gen_loss: {:.4f}".format(gen_loss_val[0]))
                    print("--Test_mse: {:.4f}".format(test_mse))

                    if epoch % (2*cycle) >cycle-1:
                        dis_acc_real, dis_acc_fake, class_acc_real, class_acc_fake = evaluate_dis(sess, dataset_test)
                        print("--Train_dis_loss: {:.4f}".format(dis_loss_val[0]))
                        print("--Test_dis_accuracy_of_real_image: {:.4f}".format(dis_acc_real))
                        print("--Test_dis_accuracy_of_fake_image: {:.4f}".format(dis_acc_fake))
                        print("--Test_classify_accuracy_of_real_image: {:.4f}".format(class_acc_real))
                        print("--Test_classify_accuracy_of_fake_image: {:.4f}\n".format(class_acc_fake))

                        # dis_acc_real, dis_acc_fake, class_acc_real, class_acc_fake = evaluate_dis(sess, dataset_train)
                        # print("--Train_dis_accuracy_of_real_image: {:.4f}".format(dis_acc_real))
                        # print("--Train_dis_accuracy_of_fake_image: {:.4f}".format(dis_acc_fake))
                        # print("--Train_classify_accuracy_of_real_image: {:.4f}".format(class_acc_real))
                        # print("--Train_classify_accuracy_of_fake_image: {:.4f}\n".format(class_acc_fake))


            if epoch % config.save_epoch_interval == 0:
                test_mse = evaluate_gen(sess, dataset_test)
                print("----Epoch-{:n} finished, evaluation results:".format(epoch))
                print("--Train_gen_loss: {:.4f}".format(gen_loss_val[0]))
                print("--Test_mse: {:.4f}".format(test_mse))
                if epoch % (2*cycle) >cycle-1:
                    dis_acc_real, dis_acc_fake, class_acc_real, class_acc_fake = evaluate_dis(sess, dataset_test)
                    print("--Train_dis_loss: {:.4f}".format(dis_loss_val[0]))
                    print("--Test_dis_accuracy_of_real_image: {:.4f}".format(dis_acc_real))
                    print("--Test_dis_accuracy_of_fake_image: {:.4f}".format(dis_acc_fake))
                    print("--Test_classify_accuracy_of_real_image: {:.4f}".format(class_acc_real))
                    print("--Test_classify_accuracy_of_fake_image: {:.4f}\n".format(class_acc_fake))
                else:
                    show_path = './output/output-e%i/' % epoch
                    if not os.path.exists(show_path):
                        os.mkdir(show_path)
                    show_eval(sess, dataset_test, show_path)


                model_name = 'GAN-e{:n}-mse{:.6f}.ckpt'.format(epoch, test_mse)
                model_save_path = os.path.join(model_save_dir, model_name)
                print('Saving model...')
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
                print('Saved!')




if __name__ == '__main__':
    #train('./tf_ckpt/GAN_v1_2-e1-mse0.019040.ckpt-1')
    #train_gen(10)
    #train_dis(10, './tf_ckpt/GAN-e1-mse0.002987.ckpt-1')
    #train_dis(10, './tf_ckpt/GAN-e2-mse0.005236.ckpt-2')
    #train(1, './tf_ckpt/GAN-e7-mse0.008097.ckpt-7')
    train(100, 1, './tf_ckpt/GAN-e7-mse0.002895.ckpt-7', 0.005, 0.05)
    exit()
