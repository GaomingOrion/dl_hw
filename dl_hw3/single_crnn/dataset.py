import cv2
import numpy as np
import tensorflow as tf
import h5py
import pickle
import os
from common import config
class Dataset:
    def __init__(self, dataset_name='train', read_from_pkl=True):
        assert dataset_name in ['train', 'test']
        self.ds_name = dataset_name
        self.data_path = os.path.join(config.data_dir, self.ds_name)
        self.size = config.train_size if self.ds_name == 'train' else config.test_size
        self.batch_size = config.train_batch_size if self.ds_name == 'train' else config.test_batch_size
        if not read_from_pkl:
            imgs = self.load_imgs()
            infos = self.read_digitStruct_mat(os.path.join(self.data_path, 'digitStruct.mat'))
            self.X, self.labels, self.box_pos = self.raw_process(imgs, infos)
            del imgs, infos
            with open(os.path.join(self.data_path, 'X_raw.pkl'), 'wb') as f:
                pickle.dump(self.X, f)
            with open(os.path.join(self.data_path, 'labels.pkl'), 'wb') as f:
                pickle.dump(self.labels, f)
            with open(os.path.join(self.data_path, 'box_pos.pkl'), 'wb') as f:
                pickle.dump(self.box_pos, f)
        else:
            with open(os.path.join(self.data_path, 'X_raw.pkl'), 'rb') as f:
                self.X = pickle.load(f)
            with open(os.path.join(self.data_path, 'labels.pkl'), 'rb') as f:
                self.labels = pickle.load(f)


    def load_imgs(self):
        imgs = []
        for idx in range(1, self.size+1):
            png_path = os.path.join(self.data_path, '%i.png'%idx)
            img = cv2.imread(png_path)
            imgs.append(img)
        return imgs

    @ staticmethod
    def read_digitStruct_mat(file_path):
        f = h5py.File(file_path, 'r')
        data_dict = {'label': [], 'left': [], 'top': [], 'height': [], 'width': []}
        def get_attrs(name, obj):
            vals = []
            if obj.shape[0] == 1:
                vals.append(int(obj[0][0]))
            else:
                for k in range(obj.shape[0]):
                    vals.append(int(f[obj[k][0]][0][0]))
            data_dict[name].append(vals)
        for item in f['/digitStruct/bbox']:
            f[item[0]].visititems(get_attrs)
        f.close()
        return data_dict

    @staticmethod
    def raw_label_to_sparse(labels, max_length):
        indices, values = [], []
        dense_shape = [len(labels), max_length]
        for i in range(len(labels)):
            assert len(labels[i]) <= max_length
            for j in range(len(labels[i])):
                indices.append([i, j])
                values.append(labels[i][j])
        return tf.SparseTensorValue(indices, values, dense_shape)


    def process(self, imgs):
        res = []
        for i in range(len(imgs)):
            # if self.ds_name == 'test':
            #     n = imgs[i].shape[1]
            #     tmp = imgs[i][:, int(0.1*n):int(0.9*n),:]
            #     res.append(cv2.resize(tmp, config.image_shape[::-1]))
            # else:
                res.append(cv2.resize(imgs[i], config.image_shape[::-1]))
        res = np.float32(res)[:, :, :, ::-1]
        return res

    @staticmethod
    def raw_process(imgs, infos):
        labels, box_pos = [], []
        for i in range(len(imgs)):
            h, w, _ = imgs[i].shape
            labels.append([(lambda x:x if x!=10 else 0)(idx) for idx in infos['label'][i]])
            num = len(infos['label'][i])
            left_pos = min(infos['left'][i][j] for j in range(num))/w
            top_pos = min(infos['top'][i][j] for j in range(num))/h
            right_pos = max(infos['left'][i][j]+infos['width'][i][j] for j in range(num))/w
            bottom_pos = max(infos['top'][i][j]+infos['height'][i][j] for j in range(num))/h
            box_pos.append([(top_pos+bottom_pos)/2,
                            (left_pos+right_pos)/2,
                            min(bottom_pos-top_pos, 1.0),
                            min(right_pos-left_pos, 1.0)])
        return np.array(imgs), np.array(labels), np.float32(box_pos)


    def one_epoch_generator(self):
        idx = list(range(self.size))
        if self.ds_name == 'train':
            np.random.shuffle(idx)
        start = 0
        while start < self.size:
            end = start + self.batch_size
            yield self.process(self.X[idx[start:end]]), \
                        self.raw_label_to_sparse(self.labels[idx[start:end]], config.seq_length), \
                        self.labels[idx[start:end]]
            start = end


if __name__ == '__main__':
    d = Dataset('test')
    d = Dataset('train')



