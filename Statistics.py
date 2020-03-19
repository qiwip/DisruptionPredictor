import os
import numpy as np
import matplotlib.pyplot as plt


threshold = 0.5

path_npy = os.path.join('model', 'result')
path_pic = os.path.join('model', 'pic')

shots = dict()
with open(os.path.join('log', 'ShotsInDataset.txt'), 'r') as f:
    for i in f.readlines():
        shots[(i.split(' ')[0])] = (i.split(' ')[1], i.split(' ')[2].replace('\n', ''))

if not os.path.exists(path_pic):
    os.makedirs(os.path.join(path_pic, 'd'))
    os.makedirs(os.path.join(path_pic, 'u'))

und_total = 0
und_true = 0
dis_total = 0
dis_true = 0
for file in os.listdir(path_npy):
    shot = file.replace('.npy', '').replace('y_y_', '')
    if int(shot) < 1064579 or int(shot) > 1065136:
        continue
    data = np.load(os.path.join(path_npy, file))
    y = data[0]
    y_ = data[1]
    y_ = np.where(y_ >= 0, y_, 0)
    if shots[shot][1] == 'u' and shots[shot][0] == '0':
        und_total += 1
        if max(y_) < threshold:
            und_true += 1
    elif shots[shot][1] == 'd' and shots[shot][0] == '0':
        dis_total += 1
        if max(y_) > threshold:
            dis_true += 1

print(und_true, und_total, und_true/und_total)
print(dis_true, dis_total, dis_true/dis_total)