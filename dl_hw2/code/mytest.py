import tensorflow as tf
import numpy as np
from model import CNN
from fmnist_dataset import Fashion_MNIST


fmnist = Fashion_MNIST()
targets = np.array(list(range(1, 10)) + [0])
model_path = '../model/fmnist_cnn.ckpt'


class WhiteAttack:
    def __init__(self, max_iter, learning_rate):
        with tf.variable_scope("fmnist_cnn") as vs:
            self.white_model = CNN(scope_name="fmnist_cnn", is_inference=False)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        # load checkpoint
        self.sess = tf.Session()
        self.white_model.restore(self.sess, '../model/fmnist_cnn.ckpt')



    def attack(self, target_img, target_label):
        target_label = np.eye(10)[target_label]
        cnt = 0
        while cnt < self.max_iter:
            grad_val = self.white_model.grad_op(self.sess, target_img, target_label)[0][0]
            target_img -= self.learning_rate * grad_val
            label_val = self.white_model.infer_op(self.sess, target_img)
            if np.argmax(label_val[0]) == np.argmax(target_label[0]):
                return cnt
            cnt += 1
        return -1

if __name__ == '__main__':
    a = WhiteAttack(100, 0.01)
    target_img = np.float32(fmnist.valid.images[[0]])
    target_label = targets[fmnist.valid.labels[[0]]]
    res = a.attack(target_img, target_label)



