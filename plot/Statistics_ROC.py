import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12)
thresholds = np.linspace(0, 1, 100)

if os.path.exists('./fpr.npy'):
    fpr = np.load('./fpr.npy')
    tpr = np.load('./tpr.npy')

else:
    path_npy = os.path.join('..', 'model', 'result')

    shots = dict()
    with open(os.path.join('..', 'log', 'ShotsInDataset.txt'), 'r') as f:
        for i in f.readlines():
            shots[(i.split(' ')[0])] = (i.split(' ')[1], i.split(' ')[2].replace('\n', ''))

    tpr = list()
    fpr = list()
    for threshold in thresholds:
        und_total = 0
        und_true = 0
        dis_total = 0
        dis_true = 0
        pre_time = list()
        for file in os.listdir(path_npy):
            shot = file.replace('.npy', '').replace('y_y_', '')

            data = np.load(os.path.join(path_npy, file))
            y = data[0]
            y_ = data[1]
            y_ = np.where(y_ >= 0, y_, 0)
            if shots[shot][1] == 'u':
                und_total += 1
                if max(y_) > threshold:
                    und_true += 1
            elif shots[shot][1] == 'd' and shots[shot][0] == '0':
                dis_total += 1
                if max(y_) > threshold:
                    dis_true += 1
                    pre_time.append(y_.shape[0] - np.where(y_ > threshold)[0][0])

        fpr.append(und_true/und_total)
        tpr.append(dis_true/dis_total)
    fpr = np.array(fpr)
    tpr = np.array(tpr)

    np.save('./fpr.npy', fpr)
    np.save('./tpr.npy', tpr)
# plot
plt.figure(figsize=(19.20, 10.80))
# plt.plot(tpr, tpr)
plt.plot(thresholds, tpr, color='#6CA6CD', linewidth=2.0, label='Successful alarm Rate')
plt.plot(thresholds, fpr, color='r', linewidth=2.0, label='False alarm Rate')
plt.legend(loc='lower left')
plt.legend(loc='best', fontsize=12)
plt.show()
