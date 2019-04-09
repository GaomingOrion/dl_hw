import tensorflow as tf
import numpy as np
from white_model import CNN
from fmnist_dataset import Fashion_MNIST
import matplotlib.pyplot as plt
import os

targets = np.array(list(range(1, 10)) + [0])

class TaskBAttack:
    def __init__(self, model_path, result_save_paths, data_dir, max_iter, learning_rate, noise_sigma):
        self.model_path = model_path
        self.result_save_paths = result_save_paths
        for result_save_path in self.result_save_paths:
            if not os.path.exists(result_save_path):
                os.mkdir(result_save_path)
        with tf.variable_scope("fmnist_cnn") as vs:
            self.white_model = CNN(scope_name="fmnist_cnn", is_inference=True)
        self.fmnist = Fashion_MNIST(data_dir=data_dir)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.noise_sigma = noise_sigma
        # load checkpoint
        self.sess = tf.Session()
        self.white_model.restore(self.sess, self.model_path)

    def __get_attack_data(self):
        y_pred = self.white_model.infer_op(self.sess, self.fmnist.test.images)[0]
        labels_pred = np.argmax(y_pred, axis=1)
        test_acc = np.sum(self.fmnist.test.labels == labels_pred)/self.fmnist.test.size
        print("测试集准确率为%.4f"%test_acc)
        right_idx = np.where(self.fmnist.test.labels == labels_pred)[0]
        np.random.shuffle(right_idx)
        chosen_idx = right_idx[:1000]
        self.attack_img = np.float32(self.fmnist.test.images[chosen_idx]) #(1000, 28, 28, 1)
        self.attack_labels = self.fmnist.test.labels[chosen_idx]        # (1000, 1)
        print("已选出1000个攻击样本")

    # 核心代码， 对一张图片进行白盒攻击
    def __white_attack_one(self, target_img, target_label):
        # target_img: (1, 28, 28, 1); target_label: (1, )
        attack_img = np.copy(target_img)
        cnt = 0
        while cnt < self.max_iter:
            cnt += 1
            grad_val = self.white_model.grad_op(self.sess, attack_img, np.eye(10)[target_label])[0][0]
            attack_img -= self.learning_rate * grad_val
            label_val = self.white_model.infer_op(self.sess, attack_img)[0]
            if np.argmax(label_val[0]) == target_label[0]:
                return cnt, attack_img
        return None

    # 核心代码， 对一张图片进行黑盒攻击
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

    def attack(self, method='white'):
        if method == 'white':
            attack_method = self.__white_attack_one
            result_save_path = self.result_save_paths[0]
        elif method == 'black':
            attack_method = self.__black_attack_one
            result_save_path = self.result_save_paths[1]
        else:
            print('method error!')
            return
        success_attack = {'idx':[], 'cnt':[], 'attack_img':[]}
        for i in range(1000):
            target_label = targets[self.attack_labels[[i]]]
            res = attack_method(self.attack_img[[i]], target_label)
            if res:
                success_attack['idx'].append(i)
                success_attack['cnt'].append(res[0])
                success_attack['attack_img'].append(res[1])
        attack_sucess_rate = len(success_attack['cnt'])/1000
        attack_cnt_mean = np.mean(success_attack['cnt']) if attack_sucess_rate > 0 else -1
        print(">攻击成功率为%.3f，平均需要的攻击次数为%i"%(attack_sucess_rate, attack_cnt_mean))
        # 挑选10组记录结果
        for i in range(min(10, len(success_attack['cnt']))):
            plt.imsave(result_save_path+'%i_origin_label%i.jpg'%(i, self.attack_labels[i]),
                       self.attack_img[success_attack['idx'][i]].reshape(28, 28))
            plt.imsave(result_save_path+'%i_attack_label%i.jpg'%(i, targets[self.attack_labels[i]]),
                       success_attack['attack_img'][i].reshape(28, 28))
        return success_attack

    def main(self):
        self.__get_attack_data()
        print(">>进行白盒攻击：")
        self.attack('white')
        print(">>进行黑盒攻击：")
        self.attack('black')
        self.sess.close()


if __name__ == '__main__':
    model_path = '../white_model/before_adv_train/fmnist_cnn.ckpt'
    result_save_paths = ['../taskB_result/white_attack_result/', '../taskB_result/black_attack_result/']
    a = TaskBAttack(model_path, result_save_paths, '../data', 200, 50.0, 2.0)
    res = a.main()
    pass


