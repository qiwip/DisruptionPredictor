import os
import numpy as np
import matplotlib.pyplot as plt


path_npy = os.path.join('model', 'result')
path_pic = os.path.join('model', 'pic')

if not os.path.exists(path_pic):
    os.makedirs(path_pic)
for file in os.listdir(path_npy):
    data = np.load(os.path.join(path_npy, file))

    plt.figure()
    plt.plot(data[0], label='y')
    plt.plot(data[1], label='y_predict')
    plt.legend()
    plt.savefig(os.path.join(path_pic, file.replace('npy', 'png')))
