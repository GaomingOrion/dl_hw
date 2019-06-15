#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from GAN import Model
from dataset import Dataset
import cv2

def eval(prev_model_path, output_path='./output/'):
    dataset_test = Dataset('test')

    # define computing graph
    model = Model()
    net_out = model.build_infer_graph()
    if type(net_out) == tuple:
        net_out = net_out[0]

    # Set saver configuration
    saver = tf.train.Saver()

    # Set sess configuration
    sess = tf.Session()

    saver.restore(sess=sess, save_path=prev_model_path)
    flabel_dict = {0:'rog', 1:'uck', 2:'eer', 3:'ile', 4:'ird', 5:'rse', 6:'hip', 7:'cat', 8:'dog', 9:'ane'}
    for batch_idx, (image_bw, image_ab, label) in enumerate(dataset_test.one_epoch_generator()):
        image_ab_pred = sess.run(net_out, feed_dict={
            model.place_holders['image_bw']: image_bw,
            model.place_holders['image_ab']: image_ab,
            model.place_holders['label']: label,
            model.place_holders['is_training']: False
        })

        imgs = np.concatenate([image_bw, image_ab], axis=3)
        imgs_pred = np.concatenate([image_bw, image_ab_pred], axis=3)
        img = np.uint8(imgs[0,:,:,:]*255)
        img_pred = np.uint8(imgs_pred[0,:,:,:]*255)
        bgrimg = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        bgrimg_pred = cv2.cvtColor(img_pred, cv2.COLOR_LAB2BGR)
        bwimg = np.uint8(image_bw[0]*255)

        label1 = flabel_dict[label[0]]
        cv2.imwrite(output_path + str(batch_idx) + '_' + label1 + '_bw.png', bwimg)
        cv2.imwrite(output_path+str(batch_idx)+'_'+label1+'.png', bgrimg)
        cv2.imwrite(output_path + str(batch_idx) + '_' + label1 + '_pred.png', bgrimg_pred)

if __name__ == '__main__':
    import os
    path = './output/'
    if not os.path.exists(path):
        os.mkdir(path)
    eval('./tf_ckpt/UNet_v1_2-e2-mse0.0071.ckpt-2', path)