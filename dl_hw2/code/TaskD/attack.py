import os
from TaskB import TaskBAttack

model_path = '../../white_model/adv_train/fmnist_cnn.ckpt'
result_save_paths = ['../../taskD_result/white_attack_result/', '../../taskD_result/black_attack_result/']
for result_save_path in result_save_paths:
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)

d = TaskBAttack(model_path, result_save_paths, '../../data', 200, 50, 2)
d.main()