import os
dataset = ['14lap', '14res', '15res', '16res']
layers = [0, 1, 2, 3, 4, 5]

for d in dataset:
    for l in layers:
        save_dir = "random1_layers{}_{}".format(str(l), d)
        print('python train.py --dataset {} --gcn_layers {} --save_dir {}'.format(str(d), str(l), save_dir))
        os.system('python train.py --dataset {} --gcn_layers {} --save_dir {}'.format(str(d), str(l), save_dir))
