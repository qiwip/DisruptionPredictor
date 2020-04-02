import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12)


threshold = 0.5

path_npy = os.path.join('model', 'result')

shots = dict()
with open(os.path.join('log', 'ShotsInDataset.txt'), 'r') as f:
    for i in f.readlines():
        shots[(i.split(' ')[0])] = (i.split(' ')[1], i.split(' ')[2].replace('\n', ''))

und_total = 0
und_true = 0
dis_total = 0
dis_true = 0
pre_time = list()
for file in os.listdir(path_npy):
    shot = file.replace('.npy', '').replace('y_y_', '')
    # if int(shot) < 1065136 or int(shot) > 1065492:
    # if int(shot) < 1064579 or int(shot) > 1065136:
    #     continue
    data = np.load(os.path.join(path_npy, file))
    y = data[0]
    y_ = data[1]
    y_ = np.where(y_ >= 0, y_, 0)
    if shots[shot][1] == 'u' and shots[shot][0] == '0':
        und_total += 1
        if max(y_) > threshold:
            und_true += 1
    elif shots[shot][1] == 'd' and shots[shot][0] == '0':
        dis_total += 1
        if max(y_) > threshold:
            dis_true += 1
            pre_time.append(y_.shape[0] - np.where(y_ > threshold)[0][0])


print(und_true, und_total)
print(dis_true, dis_total)
print(np.average(pre_time))

# plot
plt.figure(figsize=(19.20, 10.80))
group = [i for i in range(0, 60, 5)]
plt.hist(pre_time, group, histtype='bar', rwidth=0.8)
plt.xticks([i for i in range(0, 60, 5)])
plt.xlabel('提前时间', FontProperties=font)
plt.show()
