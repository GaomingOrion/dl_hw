import tensorflow as tf
import numpy as np
from model import CNN
from fmnist_dataset import Fashion_MNIST
import matplotlib.pyplot as plt
import pickle
import os

fmnist = Fashion_MNIST()
targets = np.array(list(range(1, 10)) + [0])
attack_data_path = '../attack_data/correct_1k.pkl'
model_path = '../model/fmnist_cnn.ckpt'
result_save_path = '../taskC_result/'
if not os.path.exists(result_save_path):
    os.mkdir(result_save_path)

class BlackAttack:
    def __init__(self, max_iter, noise_sigma):
        with tf.variable_scope("fmnist_cnn") as vs:
            self.white_model = CNN(scope_name="fmnist_cnn", is_inference=True)
        self.max_iter = max_iter
        self.noise_sigma = noise_sigma
        # load checkpoint
        self.sess = tf.Session()
        self.white_model.restore(self.sess, model_path)

    def __get_attack_data(self):
        with open(attack_data_path, 'rb') as f:
            self.attack_img, self.attack_labels = pickle.load(f)
        self.attack_img = self.attack_img.reshape(-1, 28, 28, 1)
        self.attack_labels = np.argmax(self.attack_labels, axis=-1).reshape(-1,)
        print("已选出1000个攻击样本")

    def __black_attack_one(self, target_img, target_label):
        # target_img: (1, 28, 28, 1); target_label: (1,)
        attack_img = np.copy(target_img)
        cnt = 0
        while cnt < self.max_iter:
            cnt += 1
            mc_sample = attack_img + self.noise_sigma*(np.random.random([1, 28, 28, 1]))
            label_val = self.white_model.infer_op(self.sess, attack_img)[0]
            if np.argmax(label_val[0]) == target_label[0]:
                return cnt, attack_img
            if np.random.random() > label_val[0][target_label[0]]:
                attack_img = mc_sample
        return None

    def attack(self):
        success_attack = {'idx':[], 'cnt':[], 'attack_img':[]}
        for i in range(1000):
            target_label = targets[self.attack_labels[[i]]]
            res = self.__black_attack_one(self.attack_img[[i]], target_label)
            if res:
                success_attack['idx'].append(i)
                success_attack['cnt'].append(res[0])
                success_attack['attack_img'].append(res[1])
        attack_sucess_rate = len(success_attack['cnt'])/1000
        attack_cnt_mean = np.mean(success_attack['cnt']) if attack_sucess_rate > 0 else -1
        print("黑盒攻击成功率为%.3f，平均需要的攻击次数为%i"%(attack_sucess_rate, attack_cnt_mean))
        for i in range(10):
            plt.imsave(result_save_path+'%i_origin_label%i.jpg'%(i, self.attack_labels[i]),
                       self.attack_img[success_attack['idx'][i]].reshape(28, 28))
            plt.imsave(result_save_path+'%i_attack_label%i.jpg'%(i, targets[self.attack_labels[i]]),
                       success_attack['attack_img'][i].reshape(28, 28))

    def main(self):
        self.__get_attack_data()
        self.attack()
        self.sess.close()

if __name__ == '__main__':
    b = BlackAttack(200, 2.0)
    b.main()
    pass