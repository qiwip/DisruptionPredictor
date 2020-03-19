import os
import numpy as np
import matplotlib.pyplot as plt


path_npy = os.path.join('model', 'result')
path_pic = os.path.join('model', 'pic')

shots = dict()
with open(os.path.join('log', 'ShotsInDataset.txt'), 'r') as f:
    for i in f.readlines():
        shots[(i.split(' ')[0])] = (i.split(' ')[1], i.split(' ')[2].replace('\n', ''))

if not os.path.exists(path_pic):
    os.makedirs(os.path.join(path_pic, 'd'))
    os.makedirs(os.path.join(path_pic, 'u'))

for file in os.listdir(path_npy):
    if shots[file.replace('.npy', '').replace('y_y_', '')][1] == 'u':
        continue
    data = np.load(os.path.join(path_npy, file))
    y = data[0]
    y_ = data[1]
    y_ = np.where(y_ >= 0, y_, 0)
    plt.figure()
    plt.plot(y, label='y')
    plt.plot(y_, label='y_predict')
    plt.legend()
    plt.savefig(os.path.join(path_pic, shots[file.replace('.npy', '').replace('y_y_', '')][1],
                             file.replace('npy', 'png')))
    plt.close()
