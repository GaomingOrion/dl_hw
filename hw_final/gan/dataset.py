import cv2
import numpy as np
import pickle
import os
import glob
from common import config

class Dataset:
    def __init__(self, dataset_name='train', read_from_pkl=True):
        assert dataset_name in ['train', 'test']
        self.ds_name = dataset_name
        self.data_path = os.path.join(config.data_dir, 'images_'+self.ds_name)
        self.size = config.train_size if self.ds_name == 'train' else config.test_size
        self.batch_size = config.train_batch_size if self.ds_name == 'train' else config.test_batch_size

        label_dict = {'rog':0, 'uck':1, 'eer':2, 'ile':3, 'ird':4, 'rse':5, 'hip':6, 'cat':7, 'dog':8, 'ane':9}
        if not read_from_pkl:
            xlist = glob.glob(self.data_path+'/*.png')
            imgs = []
            imgs_black = []
            labels = []
            for i in xlist:
                img = cv2.imread(i)
                labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                imgs_black.append(labimg[:,:,[0]])
                imgs.append(labimg[:,:,[1,2]])
                labels.append(label_dict[i[-7:-4]])
            self.X_black, self.X, self.labels = np.array(imgs_black), np.array(imgs), np.array(labels)
            with open(os.path.join(self.data_path, 'X.pkl'), 'wb') as f:
                pickle.dump(self.X, f)
            with open(os.path.join(self.data_path, 'X_black.pkl'), 'wb') as f:
                pickle.dump(self.X_black, f)
            with open(os.path.join(self.data_path, 'labels.pkl'), 'wb') as f:
                pickle.dump(self.labels, f)
        else:
            with open(os.path.join(self.data_path, 'X.pkl'), 'rb') as f:
                self.X = pickle.load(f)
            with open(os.path.join(self.data_path, 'X_black.pkl'), 'rb') as f:
                self.X_black = pickle.load(f)
            with open(os.path.join(self.data_path, 'labels.pkl'), 'rb') as f:
                self.labels = pickle.load(f)


    def one_epoch_generator(self):
        idx = list(range(self.size))
        if self.ds_name == 'train':
            np.random.shuffle(idx)
        start = 0
        while start < self.size:
            end = start + self.batch_size
            yield self.X_black[idx[start:end]]/255, \
                  self.X[idx[start:end]]/255, self.labels[idx[start:end]]
            start = end

if __name__ == '__main__':
    d = Dataset('test', True)
    #d = Dataset('test', False)
