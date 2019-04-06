import tensorflow as tf
import numpy as np
from white_model import CNN
from fmnist_dataset import Fashion_MNIST
import pickle
import os

targets = np.array(list(range(1, 10)) + [0])
model_path = '../../white_model/before_adv_train/fmnist_cnn.ckpt'
adv_data_save_path = '../../taskD_result/'
if not os.path.exists(adv_data_save_path):
    os.mkdir(adv_data_save_path)

class TaskDCreateAdvData:
    def __init__(self, max_iter, learning_rate):
        with tf.variable_scope("fmnist_cnn") as vs:
            self.white_model = CNN(scope_name="fmnist_cnn", is_inference=True)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.fmnist = Fashion_MNIST(data_dir='../../data')
        # load checkpoint
        self.sess = tf.Session()
        self.white_model.restore(self.sess, model_path)

    def __get_attack_data(self):
        # 分批计算
        start = 0
        y_pred = []
        while start < self.fmnist.train.size-1:
            y_pred.append(self.white_model.infer_op(self.sess, self.fmnist.train.images[start:start+10000])[0])
            start += 10000
        y_pred = np.concatenate(y_pred)
        labels_pred = np.argmax(y_pred, axis=1)
        test_acc = np.sum(self.fmnist.train.labels == labels_pred)/self.fmnist.train.size
        print("训练集准确率为%.4f"%test_acc)
        right_idx = np.where(self.fmnist.train.labels == labels_pred)[0]
        np.random.shuffle(right_idx)
        chosen_idx = right_idx[:10000]
        self.attack_img = np.float32(self.fmnist.train.images[chosen_idx]) #(1000, 28, 28, 1)
        self.attack_labels = self.fmnist.train.labels[chosen_idx]        # (1000, 1)
        print("已选出10000个攻击样本")

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

    def attack(self):
        success_attack = {'idx':[], 'cnt':[], 'attack_img':[]}
        i = 0
        while len(success_attack['idx']) < 1000:
            target_label = targets[self.attack_labels[[i]]]
            res = self.__white_attack_one(self.attack_img[[i]], target_label)
            if res:
                success_attack['idx'].append(i)
                success_attack['cnt'].append(res[0])
                success_attack['attack_img'].append(res[1])
            i += 1
        attack_cnt_mean = np.mean(success_attack['cnt'])
        print("攻击成功率为%.4f"%(1000/i))
        print(">平均需要的攻击次数为%i"%(attack_cnt_mean))
        # 保存对抗样本
        adv_img = np.concatenate(success_attack['attack_img'])
        print(adv_img.shape)
        adv_labels = self.attack_labels[success_attack['idx']]
        with open(adv_data_save_path + 'adv_data.pkl', 'wb') as f:
            pickle.dump([adv_img, adv_labels], f)
        print('已保存1000个对抗样本')

    def main(self):
        self.__get_attack_data()
        self.attack()
        self.sess.close()


if __name__ == '__main__':
    a = TaskDCreateAdvData(200, 50)
    res = a.main()
    pass


