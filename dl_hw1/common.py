import pickle
import numpy as np


def load_train_data():
    data, labels = None, None
    for i in range(1, 6):
        file = './data/cifar-10-batches-py/data_batch_%i'%i
        with open(file, 'rb') as fo:
            tmp = pickle.load(fo, encoding='bytes')
        data = tmp[b'data'] if i == 1 else np.concatenate([data, tmp[b'data']])
        labels = tmp[b'labels'] if i == 1 else np.concatenate([labels, tmp[b'labels']])
    data = data.astype(np.float32) / 255 * 2 - 1.0
    return data, labels

def load_dev_data():
    file = './data/cifar-10-batches-py/test_batch'
    with open(file, 'rb') as fo:
        tmp = pickle.load(fo, encoding='bytes')
        data = tmp[b'data']
        labels = tmp[b'labels']
    data = data / 255 * 2 - 1.0
    return data, labels
