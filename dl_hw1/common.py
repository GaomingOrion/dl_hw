import pickle
import numpy as np


def load_train_data(data_dir):
    data, labels = None, None
    for i in range(1, 6):
        file = data_dir + 'data_batch_%i'%i
        with open(file, 'rb') as fo:
            tmp = pickle.load(fo, encoding='bytes')
        data = tmp[b'data'] if i == 1 else np.concatenate([data, tmp[b'data']])
        labels = tmp[b'labels'] if i == 1 else np.concatenate([labels, tmp[b'labels']])
    data = data.astype(np.float32) / 255 * 2 - 1.0
    return data, labels

def load_dev_data(data_dir):
    file = data_dir + 'test_batch'
    with open(file, 'rb') as fo:
        tmp = pickle.load(fo, encoding='bytes')
        data = tmp[b'data']
        labels = tmp[b'labels']
    data = data / 255 * 2 - 1.0
    return data, labels

# mini_batch generator
def mini_batch_gnr(n, minibatch_size, data, labels):
    index = list(range(n))
    np.random.shuffle(index)
    start = 0
    while start < n - 1:
        end = start + minibatch_size
        yield data[index[start: end]], labels[index[start: end]]
        start = end